"""OpenAI-compatible HTTP surface for the routing product (beta).

`POST /v1/chat/completions` with the standard request body, so an existing OpenAI client works by
changing `base_url` alone. Clients ask for the single virtual model `thirtyspokes`; the `model`
field is otherwise IGNORED, because choosing the model is the entire product.

The response is branded as `thirtyspokes` and still discloses everything about the delegation —
which pool model really answered, what it cost, and whether the head or the baseline chose it — in
the `X-ThirtySpokes-*` headers, the `x_thirtyspokes` body block, and the request log. Branding the
`model` field must never become a way of hiding the routing decision; that disclosure IS the
product.

TWO THINGS THIS IS BUILT AROUND.

RELIABILITY BEFORE ROUTING. Four of the seven pinned pool models returned an empty response on
39-74% of measured code problems, and five of seven exceeded a 130s budget at a raised token cap
(`docs/ROUTER_V2.md (removed with v2, 2026-09-07)` §2a-ter). In a benchmark an empty answer grades as wrong; in a product it is a
failed request. So every routed call is bounded by a deadline and falls back to the baseline on
empty, error, or timeout. The fallback RATE is a first-class metric rather than something hidden by
a default allowlist: if the head keeps picking rungs that cannot answer, that is the pool telling
you to re-screen it, and it should be visible in `/stats` on day one.

THE CLAIM IS A COMPARISON, SO IT IS INSTRUMENTED. Run the same traffic in `baseline` mode (no head)
and in head mode, and compare the recorded cost and fallback rate. Nothing here asserts that routing
wins; it makes the question answerable on real traffic, which is the input every measurement in this
project has lacked.

PRIVACY. Served traffic is user data. The request log records a SHA-256 of the prompt by default,
never the text; `log_prompts=True` is opt-in and must not be enabled without consent from whoever
sent the traffic. Prompts that end up in a public holdout corpus are revealed after scoring
(`koth/holdout_feed.py`), so an unconsented prompt logged here can become a published prompt later.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import Lock

from ..gateway import signing
from .policy import RoutingPolicy, price_of

# The single virtual model clients ask for. There is exactly one because picking the model is the
# product: `thirtyspokes` means "you choose", and offering a second name would mean offering a way
# to opt out of the thing being sold.
MODEL_NAME = "thirtyspokes"

DEFAULT_TIMEOUT_S = 120.0
# Not a round number picked for comfort. Several pool rungs are reasoning models that spend the
# token budget before emitting any visible content, so a SMALL max_tokens turns them into guaranteed
# empty responses and therefore guaranteed fallbacks — at double the latency and double the cost.
# Measured live on `qwen/qwen3.7-flash` with "what is 17*23": at max_tokens=64 the content came back
# empty and the baseline had to answer; at 2048 the same rung answered directly in 244 tokens. A
# client that sends its own small max_tokens will silently push its traffic onto the baseline, which
# is safe but not free — watch `fallback_rate` in `/stats` before blaming the head.
DEFAULT_MAX_TOKENS = 4096


class Stats:
    """Counters the beta is actually judged on. Locked because Starlette runs sync endpoints in a
    threadpool, so requests genuinely overlap."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests = 0
        self.routed = 0            # the policy's first choice was served
        self.fallbacks = 0         # the first choice failed; a later ladder entry answered
        self.errors = 0            # nothing answered
        self.cost_usd = 0.0
        self.by_model: dict[str, int] = {}
        self.shadow_calls = 0
        self.shadow_cost_usd = 0.0

    def record(self, model: str, cost: float, *, routed: bool, fell_back: bool) -> None:
        with self._lock:
            self.requests += 1
            self.routed += routed
            self.fallbacks += fell_back
            self.cost_usd += cost
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def record_error(self) -> None:
        with self._lock:
            self.requests += 1
            self.errors += 1

    def spend_shadow(self, budget: float, projected: float) -> bool:
        """Reserve room for one shadow call. False = the budget is spent; the flywheel stops
        SILENTLY IN THE RESPONSE PATH but visibly in `/stats` — sampling must never break serving."""
        with self._lock:
            if self.shadow_cost_usd + projected > budget:
                return False
            self.shadow_calls += 1
            return True

    def add_shadow_cost(self, cost: float) -> None:
        with self._lock:
            self.shadow_cost_usd += cost

    def snapshot(self) -> dict:
        with self._lock:
            n = max(1, self.requests)
            return {"requests": self.requests, "routed": self.routed,
                    "fallbacks": self.fallbacks, "errors": self.errors,
                    "fallback_rate": round(self.fallbacks / n, 4),
                    "cost_usd": round(self.cost_usd, 6),
                    "cost_per_request_usd": round(self.cost_usd / n, 8),
                    "by_model": dict(self.by_model),
                    "shadow_calls": self.shadow_calls,
                    "shadow_cost_usd": round(self.shadow_cost_usd, 6)}


def _prompt_of(messages: list[dict]) -> str:
    """What the router sees. The LAST user turn, because that is the ask; earlier turns are context
    the head was never trained on and concatenating them would drift the embedding away from the
    single-ask distribution every measurement in this project was taken on."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def create_app(policy, backend, *, stats: Stats | None = None,
               log_path: str | None = None, log_prompts: bool = False,
               log_features: bool = False,
               shadow_rate: float = 0.0, shadow_budget_usd: float = 0.0,
               shadow_path: str | None = None,
               request_timeout: float = DEFAULT_TIMEOUT_S,
               max_tokens: int = DEFAULT_MAX_TOKENS):
    """`policy` is anything with `.decide(prompt) -> Decision` — RoutingPolicy or TieredPolicy.

    `log_features` writes the prompt's EMBEDDING (float16, base64) into each log record. It is what
    the training loop consumes (`thirtyspokes-serve-train`), and it is opt-in for the same reason
    `log_prompts` is: embeddings carry no text but they are semantic and partially invertible, so a
    features log must be kept private. Without it the service still runs — it just cannot learn.

    `shadow_rate`/`shadow_budget_usd` sample a fraction of requests across the REST of the pool, in
    the background, after the response is sent. This is the data flywheel: it is how the served
    traffic becomes an ask x model outcome matrix — the input for pool re-screening, for the
    escalate trainer, and eventually for the subnet's holdout corpus. It spends real money (a full
    7-model shadow measured ~$0.25/ask on code traffic), so it is off by default and hard-capped by
    the budget: when the budget is spent the flywheel stops and serving continues, never the other
    way around.
    """
    import random as _random
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from fastapi import FastAPI  # noqa: PLC0415 — optional dep, imported only when serving
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    from ..koth import harness  # noqa: PLC0415
    from ..koth.harness import pool_models  # noqa: PLC0415
    from . import escalate as E  # noqa: PLC0415

    app = FastAPI(title="ThirtySpokes router", version="beta")
    app.state.stats = stats if stats is not None else Stats()
    app.state.shadow_pool = ThreadPoolExecutor(max_workers=2) if shadow_rate > 0 else None
    log_file = Path(log_path) if log_path else None
    shadow_file = Path(shadow_path) if shadow_path else None
    log_lock = Lock()

    def _append(path: Path | None, record: dict) -> None:
        if path is None:
            return
        with log_lock:                       # one line per request, never interleaved
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _log(record: dict) -> None:
        _append(log_file, record)

    def _call(model: str, messages: list, params: dict) -> tuple[str, int, int, float, float]:
        started = time.monotonic()
        text, tin, tout, cost = backend.complete(model, messages, params)
        return text, tin, tout, cost, time.monotonic() - started

    def _shadow(prompt: str, messages: list, params: dict, skip: set[str]) -> None:
        """Run the rest of the pool over an already-served ask, budget-capped. Every row records
        whether the model produced a non-empty answer — the same free label serving collects for
        the tiers, now for the whole pool, which is what a pool re-screen needs."""
        sha = signing.sha256_hex(prompt)
        for model in pool_models():
            if model in skip:
                continue
            if not app.state.stats.spend_shadow(shadow_budget_usd, 0.0):
                return
            row = {"ts": time.time(), "prompt_sha256": sha, "model": model}
            try:
                text, tin, tout, cost, latency = _call(model, messages, dict(params))
                row.update({"well_formed": bool(text.strip()), "tokens_in": tin,
                            "tokens_out": tout, "cost_usd": cost, "latency_s": round(latency, 3)})
                app.state.stats.add_shadow_cost(cost)
            except Exception as exc:                  # noqa: BLE001 — a dead rung is itself the datum
                row.update({"well_formed": False, "error": f"{type(exc).__name__}: {exc}"[:160]})
            if log_prompts:
                row["prompt"] = prompt
            _append(shadow_file, row)

    @app.get("/healthz")
    def healthz() -> dict:
        kind = type(policy).__name__
        if kind == "TieredPolicy":
            mode = "tiered+predictor" if policy._predict is not None else "tiered"
        else:
            mode = "head" if getattr(policy, "_head", None) is not None else "baseline"
        return {"status": "ok", "mode": mode, "policy": kind}

    @app.get("/stats")
    def stats_endpoint() -> dict:
        return app.state.stats.snapshot()

    @app.get("/v1/models")
    def models() -> dict:
        """OpenAI clients probe this. One virtual model: the point is that you do not pick."""
        return {"object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "thirtyspokes"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        messages = body.get("messages") or []
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": {"message": "messages must not be empty"}})

        prompt = _prompt_of(messages)
        decision = policy.decide(prompt)
        ladder = decision.ladder or (decision.model,)
        params = {"max_tokens": int(body.get("max_tokens") or max_tokens),
                  "temperature": body.get("temperature", 0.0),
                  "_timeout": request_timeout}

        # Walk the ladder: serve the first entry that produces a non-empty answer. An empty answer
        # is a SUCCESSFUL HTTP call — the measured failure mode of reasoning rungs at a small token
        # budget — so it must be checked, not caught. `wasted` prices the failed-but-billed attempts
        # so /stats reports what routing actually cost, not just what the served call cost.
        served, text, tin, tout, cost, latency = None, "", 0, 0, 0.0, 0.0
        failures: list[str] = []
        wasted = 0.0
        for model in ladder:
            try:
                text, tin, tout, cost, latency = _call(model, messages, params)
                if not text.strip():
                    wasted += cost
                    raise ValueError("empty response")
                served = model
                break
            except Exception as exc:                   # noqa: BLE001 — any failure falls through
                failures.append(f"{model}: {type(exc).__name__}: {exc}"[:200])

        if served is None:
            app.state.stats.record_error()
            _log({"ts": time.time(), "prompt_sha256": signing.sha256_hex(prompt),
                  "chosen_model": decision.model, "source": decision.source,
                  "error": " | ".join(failures), **({"prompt": prompt} if log_prompts else {})})
            return JSONResponse(status_code=502,
                                content={"error": {"message": " | ".join(failures)}})

        fell_back = served != decision.model
        app.state.stats.record(served, cost + wasted, routed=decision.routed and not fell_back,
                               fell_back=fell_back)
        record = {"ts": time.time(), "prompt_sha256": signing.sha256_hex(prompt),
                  "chosen_model": decision.model, "served_model": served,
                  "source": decision.source, "fell_back": fell_back,
                  "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
                  "latency_s": round(latency, 3), "ladder": list(ladder)}
        if wasted:
            record["wasted_cost_usd"] = wasted
        if failures:
            record["first_error"] = failures[0]
        if decision.p_escalate is not None:
            record["p_escalate"] = round(decision.p_escalate, 4)
        if log_features:
            record["features_b64"] = E.encode_features(harness.encode([prompt])[0])
        if log_prompts:
            record["prompt"] = prompt
        _log(record)

        if (app.state.shadow_pool is not None and shadow_budget_usd > 0
                and _random.random() < shadow_rate):
            attempted = set(ladder[: len(failures) + 1])
            app.state.shadow_pool.submit(_shadow, prompt, messages, params, attempted)

        pin, pout = price_of(served)
        body_out = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            # The product's name, not the pool model's. A client asked for `thirtyspokes` and that is
            # what answered; which rung it delegated to is an implementation detail of this service.
            # It is NOT hidden, though — WHICH model really answered is the core disclosure of a
            # routing product, so it appears three ways that survive a strict client: the response
            # headers below, `x_thirtyspokes` here, and the request log.
            "model": MODEL_NAME,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout,
                      "total_tokens": tin + tout},
            # Namespaced so it cannot collide with a field a real OpenAI client expects.
            "x_thirtyspokes": {"chosen_model": decision.model, "served_model": served,
                               "rung": decision.rung, "source": decision.source,
                               "fell_back": fell_back, "cost_usd": cost,
                               "latency_s": round(latency, 3),
                               "advertised_price_per_m": {"input": pin, "output": pout}},
        }
        # Headers, because a strict OpenAI client deserialises into a fixed schema and DROPS unknown
        # JSON fields — which would make the served model invisible to exactly the clients most
        # likely to be pointed at this. Headers survive that, and `curl -i` shows them for free.
        return JSONResponse(content=body_out, headers={
            "X-ThirtySpokes-Model": served,
            "X-ThirtySpokes-Chosen-Model": decision.model,
            "X-ThirtySpokes-Fell-Back": str(fell_back).lower(),
            "X-ThirtySpokes-Cost-Usd": f"{cost:.8f}",
        })

    return app
