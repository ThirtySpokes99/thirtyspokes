"""The real OpenRouter client — where §2.1's rule becomes bytes on the wire (§4, §5.1b, §8b.3).

`worker.py` holds the *seam*: `complete(model_id, task_text, params) -> (text, cost)`, three
arguments wide so a Conductor-produced string has nowhere to sit. This module is the one
implementation of it that spends money, plus the free catalog fetch `pool.snapshot` consumes. It
lives apart from the seam because a seam is a contract and this is a pile of HTTP failure modes;
mixing them would make the contract hard to read and the failure modes hard to test.

WHY httpx AND NOT THE OPENAI SDK, given the rest of this repo reaches for the SDK.

* **The body has to be exactly what we say it is.** The pinned provider block and the §2.1
  single-message body are the whole point; an SDK that reshapes, renames or drops unknown fields
  between versions would make "two miners send the same object" a claim about a dependency's
  changelog. Here the body is a dict this module builds and a test reads back off the wire.
* **Retries have to share one clock.** MEASURED in `gateway/gateway.py`: the SDK's `timeout` bounds
  one attempt, not the call, so `max_retries=3` turned a 300s-looking call into a 1200s one and a
  live epoch died at its operator deadline. `_send` gives all attempts a single budget.
* **`usage.cost` is not an SDK field.** It arrives in `model_extra`, which is the SDK working
  against us for a number `final_b` prices the arm on.

WHAT IS PINNED IN THE REQUEST, AND WHAT PINNING CANNOT REACH (§11-1). `PROVIDER_ROUTING` fixes the
things the request body can fix: no silent fallback to a second provider, no provider that quietly
drops `temperature`, and a deterministic pick among the endpoints that remain. It does **not** close
§11-1, and the docs are explicit about why: request-level provider lists *narrow within* the
account's, `zdr` ORs with the account's, and BYOK endpoints are prioritised by account
configuration. A miner's key can therefore still route `openai/gpt-5.4` somewhere else. That is the
hole the owner-run gateway (the build plan M5) closes by holding the credentials, and this module is
shaped to sit behind it: `chat_body` and `read_completion` are module-level pure functions the
gateway can reuse verbatim, and pointing `base_url` at the gateway changes nothing else. What this
module contributes to that hole meanwhile is *evidence* — `Completion.provider` and
`Completion.served_model` record what actually served each call, so drift is visible in the trace
(D15) rather than merely suspected.

THREE OUTCOMES, TWO CHANNELS (§8b.3). A worker that errors is data about that rung, not a crash, so
the two failure kinds are told apart by *how they return*, not by an error code the caller must
interpret:

* **answered** — a `Completion` whose `text` is non-empty (`finish_reason == "stop"`). Cost charged.
* **answered badly** — a `Completion` too: empty content, a refusal sentence, `finish_reason` of
  `"length"` or `"content_filter"`. **Cost is still charged**, because it was still spent, and the
  benchmark's grader is what judges the content. This is the case a provider-failure branch must not
  swallow: a reasoning model that burns its budget thinking and returns nothing is a fact about that
  rung's quality-per-dollar, which is exactly what the arm is scored on.
* **transport failed** — `WorkerError` is raised and there is no cost to charge. Connect/read
  timeouts, transient statuses after the last attempt, permanent 4xx, a 200 whose body will not
  parse, a 200 carrying only an `error` object (OpenRouter's documented non-streaming provider
  failure), `finish_reason == "error"`, and a response with no readable `usage.cost`.

`scaffold._call_worker` catches both channels identically today — a raise becomes a recorded failed
step — so the distinction is not for control flow. It is for the trace: an answered-badly rung has a
cost and a `finish_reason` in the record, and a transport-failed one has neither, and a reader of a
published window must be able to tell "this model is weak here" from "this provider was down".
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .pool import snapshot
from .types import Catalog
from .worker import WorkerError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The two endpoints this subnet uses. `/models` is the only free one — everything else spends the
# miner's allowance (§4) — which is why the catalog fetch is safe to run on every window open and
# why the M2a cost probe is the first thing that costs anything.
MODELS_PATH = "/models"
CHAT_PATH = "/chat/completions"
# What a key can still spend (D18): the key's own remaining limit, and the account's remaining
# credits. Both are read before an arm runs on a miner's key, so an empty account is deferred
# rather than discovered one refused call at a time.
KEY_PATH = "/auth/key"
CREDITS_PATH = "/credits"

# Owner-pinned provider routing: the request-body half of "`openai/gpt-5.4` means the same object
# for every miner" (§11-1). Each entry buys one specific thing:
#
# * `allow_fallbacks: False` — no silent substitution. A fallback would answer the task on a
#   different endpoint than the one the price and the pick were reasoned about, and the arm would be
#   scored for a routing decision nobody made. The cost is availability: when the chosen endpoint is
#   down the call fails (HTTP 503, "no available provider meets your routing requirements") instead
#   of quietly succeeding elsewhere. That is the right trade here — a failed rung is data (§8b.3),
#   while a substituted one is a corrupted measurement.
# * `require_parameters: True` — a provider that does not honour `temperature` is refused rather
#   than served. §1 pins greedy decoding so a paired duel compares routing and not luck; a provider
#   that drops the parameter puts the sampling noise back in, silently.
# * `sort: "price"` — a deterministic pick among the survivors, and the cheap end is where a
#   cost-scored objective (§5.1b) should look first anyway. Without it the endpoint is chosen by
#   load balancing, i.e. by the clock, and the king's arm and the challenger's minutes later would
#   face different machines.
#
# A read-only mapping for the same reason `worker.WORKER_PARAMS` is one: this is a pin, and a pin
# something can mutate at runtime is a convention.
PROVIDER_ROUTING: Mapping[str, object] = MappingProxyType({
    "allow_fallbacks": False,
    "require_parameters": True,
    "sort": "price",
})

# NOT PINNED, AND RECORDED HERE SO IT IS NOT MISTAKEN FOR AN OVERSIGHT: `quantizations`. Providers
# serve the same slug at int4 through bf16, and two quantisations are not the same model — the same
# argument `config.DTYPE` makes about the Conductor. Pinning it here would drop the cheapest
# endpoints of most models and, for some, every endpoint, turning a determinism pin into a window
# that cannot run. The residual is that §6.2's frozen catalog freezes the *slug and its price*, not
# the machine behind it; `Completion.provider` is what makes the drift auditable.

# Statuses worth another attempt: a timeout, a rate limit, and any 5xx — the documented family
# (502 "model is down", 503 "no provider meets your routing requirements") plus the 52x a CDN in
# front of the API can inject. Everything else (400 malformed, 401 bad key, 402 out of credits,
# 403 moderation, 404 unknown model) is a fact that will be just as true in two seconds, and
# retrying it spends the episode's clock to learn nothing.
# The per-`recv` read timeout handed to httpx inside one attempt. Short on purpose: it bounds how
# long a SILENT provider can hold a call past the budget, while `_bounded`'s per-chunk deadline
# check bounds a TRICKLING one. Neither is the call's budget — `OpenRouterClient(timeout=...)` is.
READ_SLICE_SECONDS = 30.0

RETRY_STATUSES = frozenset({408, 429})


def transient(status: int) -> bool:
    """Is this status worth another attempt inside the same call's budget?"""
    return status in RETRY_STATUSES or status >= 500


@dataclass(frozen=True)
class Completion:
    """One worker response, read rather than estimated.

    `cost_usd` comes from the provider's own `usage.cost` (§5.1b). `tokens_in`/`tokens_out` are
    carried for the gateway's receipts, not for arithmetic — deriving cost from them and a local
    price table is the thing this field exists to avoid.

    `provider` and `served_model` are the §11-1 audit evidence: the request can ask for an endpoint
    but only the response can say which one answered, and a window's published trace (D15) is where
    a substitution would show up.
    """

    text: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    finish_reason: str
    served_model: str
    provider: str


def chat_body(model_id: str, task_text: str, params: Mapping[str, object]) -> dict:
    """The exact JSON a worker call sends. Pure, so the owner gateway can build the same one.

    ONE USER MESSAGE, THE TASK TEXT, NOTHING ELSE (§2.1). No system prompt, no wrapper, no
    formatting instructions: this function is where the invariant stops being a rule about the
    scaffold and becomes bytes, so anything added here would have to be argued against §2.1 first.

    THE PINS ARE SPREAD LAST, and that ordering is the load-bearing part. `params` is
    `worker.WORKER_PARAMS` in production, but this function is also the gateway's, and a mapping
    that could set `model`, `messages` or `provider` by colliding with a pinned key would be a
    channel — the model ID is the one Conductor-made value that reaches a provider, and it must
    reach it through the validated argument or not at all.
    """
    return {**params,
            "model": model_id,
            "messages": [{"role": "user", "content": task_text}],
            "provider": dict(PROVIDER_ROUTING),
            # Deprecated as of the API docs read 2026-09-01 — usage is now returned unconditionally
            # — and sent anyway. It is a no-op field on the current endpoint, both arms send it
            # identically, and the failure it guards against (a response with no `cost`, which
            # `read_completion` must refuse) costs a paid call and its answer.
            "usage": {"include": True},
            "stream": False}


def read_completion(payload: Mapping[str, Any]) -> Completion:
    """A `/chat/completions` body -> a `Completion`, or `WorkerError` if it cannot be scored. Pure.

    THE 200 THAT IS A FAILURE. OpenRouter's HTTP status is final once sent, so a provider failure on
    a non-streaming request arrives as `200 OK` whose body holds an `error` object and no `choices`
    — and a mid-generation failure arrives as a choice with `finish_reason: "error"`. Both are
    transport failures wearing a success status, and reading either as an answer would score a rung
    the provider never ran.

    A MISSING `usage.cost` IS REFUSED, NOT ESTIMATED (§5.1b). The alternative is computing spend
    from the catalog's prices and the token counts — the local price table `worker.py` rules out:
    it puts the owner's arithmetic inside the score and misprices every arm that used a repriced
    model. Refusing forfeits an answer that was paid for, which is the safe direction: the arm loses
    the quality along with the unpriced spend, so unpriced spend can never buy quality.
    """
    error = payload.get("error")
    if error is not None:
        raise WorkerError(f"provider error in a 200 body: {_error_text(error)}")
    choices = payload.get("choices") or ()
    if not choices or not isinstance(choices[0], Mapping):
        raise WorkerError("response carried no readable choices")
    choice = choices[0]
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason == "error":
        raise WorkerError(f"generation failed mid-response: {_error_text(choice.get('error'))}")
    usage = payload.get("usage")
    cost = _number(usage.get("cost")) if isinstance(usage, Mapping) else None
    if cost is None:
        raise WorkerError("response carried no readable usage.cost, so the call cannot be priced")
    message = choice.get("message")
    text = message.get("content") if isinstance(message, Mapping) else None
    return Completion(
        text=text or "",
        cost_usd=cost,
        tokens_in=int(_number(usage.get("prompt_tokens")) or 0),
        tokens_out=int(_number(usage.get("completion_tokens")) or 0),
        finish_reason=finish_reason,
        served_model=str(payload.get("model") or ""),
        provider=str(payload.get("provider") or ""))


@dataclass(frozen=True)
class Funding:
    """What a key reports it can still spend (D18). `None` means the account set no such bound.

    `limit_remaining` is the key's own credit limit less its usage (OpenRouter lets a miner cap a
    key when they create it — the cheapest bound on their exposure); `credits_remaining` is the
    account's purchased credits less its usage. A window's allowance on the key is the miner's own
    `cap_usd` bounded by whichever of these exist.
    """

    limit_remaining: float | None
    credits_remaining: float | None

    def bound(self, cap_usd: float) -> float:
        known = [value for value in (self.limit_remaining, self.credits_remaining)
                 if value is not None]
        return min([float(cap_usd), *known])


class OpenRouterClient:
    """The live worker (§4). Structurally a `worker.Worker`, so it drops in where `MockWorker` sits.

    `base_url` IS THE GATEWAY SEAM. The owner gateway (the build plan M5) speaks the same
    OpenAI-compatible shape, so putting it in front of a duel is a different `base_url` and a
    per-duel token in `api_key` — no wrapper, no second client, and no import of a module that does
    not exist yet.

    `transport` exists so that this class is *tested* rather than marked `# pragma: no cover` like
    the seam it replaces. The client that runs against OpenRouter is the one the tests drive; only
    the bytes' destination differs. Given what this object does — spends a miner's money, decides
    what counts as an outcome — "covered by inspection" was not good enough.
    """

    def __init__(self, api_key: str, base_url: str = OPENROUTER_BASE_URL,
                 timeout: float = 300.0, attempts: int = 3, backoff: float = 1.5,
                 transport: object | None = None):
        import httpx  # noqa: PLC0415

        self._timeout = float(timeout)
        self._attempts = max(1, int(attempts))
        self._backoff = float(backoff)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport)  # type: ignore[arg-type]

    def close(self) -> None:
        self._client.close()

    def models(self) -> dict:
        """`GET /api/v1/models` -> the raw payload, unchanged (§6.2).

        Returned raw because the owner archives it beside the window record: which rows `snapshot`
        dropped stays derivable afterwards rather than a decision this process made and forgot.

        It goes through the retry loop where `pool.fetch` does not, and that is the whole reason
        this method exists next to it: the catalog fetch happens once per window, and a transient
        502 on it does not degrade a window, it prevents one.
        """
        return dict(self._send("GET", MODELS_PATH))

    def catalog(self) -> Catalog:
        """The window's frozen action space, straight from the live list (§6.2)."""
        return snapshot(self.models())

    def funding(self) -> Funding:
        """What this key can still spend (D18): `GET /auth/key`, then `GET /credits`.

        The first is the probe: a key OpenRouter refuses (401) raises `WorkerError` with that
        status here, before an arm is opened on it. The second is best-effort — an account that
        will not report its credits to this key is read as "no known bound", not as broke, because
        the call that would tell us is the 402 the gateway already classifies.
        """
        key = self._send("GET", KEY_PATH).get("data")
        key = key if isinstance(key, Mapping) else {}
        try:
            credits = self._send("GET", CREDITS_PATH).get("data")
        except WorkerError:
            credits = None
        remaining = None
        if isinstance(credits, Mapping):
            bought, used = _number(credits.get("total_credits")), _number(credits.get("total_usage"))
            if bought is not None and used is not None:
                remaining = bought - used
        return Funding(limit_remaining=_number(key.get("limit_remaining")),
                       credits_remaining=remaining)

    def chat(self, model_id: str, task_text: str, params: Mapping[str, object]) -> Completion:
        """One worker call. Returns an outcome; raises `WorkerError` only for transport failure."""
        return read_completion(self._send("POST", CHAT_PATH,
                                          chat_body(model_id, task_text, params)))

    def complete(self, model_id: str, task_text: str,
                 params: Mapping[str, object]) -> tuple[str, float]:
        """The `worker.Worker` contract: `(text, dollars)`.

        The narrowing is deliberate — the scaffold is given exactly what §2.1 lets it have back, and
        anything richer (which provider served, why generation stopped) belongs in the trace, which
        is what `chat` is for.
        """
        completion = self.chat(model_id, task_text, params)
        return completion.text, completion.cost_usd

    def _send(self, method: str, path: str, body: dict | None = None) -> Mapping[str, Any]:
        """Attempts sharing ONE wall-clock budget, then `WorkerError`.

        `self._timeout` bounds the whole call, not each attempt, and every attempt is handed the
        remaining budget. MEASURED (`gateway/gateway.py`): with a per-attempt timeout and SDK
        retries, a call that looked 300s-bounded could take 1200s, and a live epoch spent 49 minutes
        of work before dying on its operator deadline. Under a per-episode wall clock (§8b.2) that
        is not a slow call, it is an abandoned episode.

        An unparseable 200 is retried like a 502, which is also measured: OpenRouter has returned
        HTML error pages and truncated bodies on a 200, and one of them killed a whole epoch's work
        by raising out of `.json()`.
        """
        import httpx  # noqa: PLC0415

        deadline = time.monotonic() + self._timeout
        detail = "no attempt fitted inside the budget"
        made = 0
        for attempt in range(self._attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                # Counted and named rather than folded into the attempt count: §5.8 wants a verdict
                # decided by something other than routing legible in the reveal, and "the provider
                # refused three times" and "the clock ran out after one" are different facts.
                detail = f"{detail}; then the {self._timeout:.1f}s budget was spent"
                break
            made += 1
            try:
                status, content = self._bounded(method, path, body, deadline)
            except httpx.HTTPError as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:160]}"
            else:
                text = content.decode("utf-8", errors="replace")
                if status == 200:
                    try:
                        return json.loads(content)
                    except ValueError:
                        detail = f"200 with an unparseable body: {text[:160]!r}"
                elif transient(status):
                    detail = f"HTTP {status}: {text[:160]}"
                else:
                    # Permanent by the time we could ask again: refuse now rather than spend two
                    # more attempts of the episode's clock confirming it.
                    raise WorkerError(f"{method} {path} refused with HTTP {status}: {text[:160]}",
                                      status=status)
            if attempt + 1 < self._attempts and deadline - time.monotonic() > self._backoff:
                time.sleep(self._backoff)
        raise WorkerError(f"{method} {path} failed after {made} attempt(s): {detail}")

    def _bounded(self, method: str, path: str, body: dict | None,
                 deadline: float) -> tuple[int, bytes]:
        """One attempt whose WHOLE wall time — headers and body — ends at `deadline`.

        MEASURED 2026-09-07, on the first live M9 window: a worker call sat in `ssl.recv` for 55
        minutes with the arm wedged behind it. `httpx` applies a scalar timeout to each socket
        `recv`, not to the body, so a response that trickles bytes — OpenRouter keeps a connection
        alive with periodic bytes while a provider hangs — completes every `recv` inside the budget
        and never ends. Every clock above this seam (§8b.2's episode clock, the duel clock) is
        checked between steps and cannot interrupt a blocking read, so the budget this docstring's
        caller promises has to be enforced HERE, per chunk, or it is not a budget.

        The read slice is short so a provider that goes silent fails within `READ_SLICE_SECONDS` of
        the deadline rather than a full `remaining` after it; the body is iterated and the deadline
        checked on every chunk, so a trickle fails at the deadline too. `ReadTimeout` is what
        `_send` already treats as a transport failure, so the outcome is the one §8b.3 assigns.
        """
        import httpx  # noqa: PLC0415

        remaining = deadline - time.monotonic()
        timeout = httpx.Timeout(connect=remaining, write=remaining, pool=remaining,
                                read=min(remaining, READ_SLICE_SECONDS))
        chunks: list[bytes] = []
        with self._client.stream(method, path, json=body, timeout=timeout) as response:
            for chunk in response.iter_bytes():
                if time.monotonic() >= deadline:
                    raise httpx.ReadTimeout(
                        f"body still arriving after the {self._timeout:.1f}s budget "
                        f"({sum(map(len, chunks))} bytes so far)")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)


def _error_text(error: object) -> str:
    """An OpenRouter `error` object rendered for the trace: `{code, message, metadata}`."""
    if isinstance(error, Mapping):
        return f"code={error.get('code')} {str(error.get('message') or '')[:160]}"
    return str(error or "unspecified")[:160]


def _number(raw: object) -> float | None:
    """A JSON number, or None. A bool is not a number here: `cost: true` must not price a call."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        return float(raw)
    except ValueError:
        return None
