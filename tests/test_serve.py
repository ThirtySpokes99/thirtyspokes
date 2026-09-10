"""The beta routing API (`serve/`).

The property that matters most here is not "does it route" — it is "does routing ever leave a
customer worse off than calling the cheap reliable model directly". Four of seven pinned pool models
returned an empty answer on 39-74% of measured code problems, so a router that forwards those
failures is a reliability REGRESSION dressed up as a product. Most of these tests are about the
fallback path and about the fallback being COUNTED, because an invisible fallback rate is how a
broken pool looks like a working router.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thirtyspokes.koth import harness
from thirtyspokes.koth.harness import EMBED_DIM, pool_models, save_head
from thirtyspokes.router import RouterHead
from thirtyspokes.serve.app import MODEL_NAME, Stats, create_app
from thirtyspokes.serve.policy import DEFAULT_BASELINE, RoutingPolicy

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

POOL = pool_models()
HIDDEN = 8


@pytest.fixture(autouse=True)
def _offline_encoder(monkeypatch):
    """The pinned encoder needs a model on disk; the routing decision under test does not depend on
    which vector it returns, only on the head applied to it."""
    monkeypatch.setattr(harness, "encode",
                        lambda prompts: np.full((len(prompts), EMBED_DIM), 1.0 / np.sqrt(EMBED_DIM)))


def head_that_picks(rung: int) -> bytes:
    """A legal head whose output bias alone decides the rung, so the choice is exact rather than
    something an optimiser happened to converge on."""
    head = RouterHead(EMBED_DIM, len(POOL), HIDDEN)
    theta = np.zeros(head.n_params)
    theta[-len(POOL) + rung] = 1.0            # b2[rung]; W1/b1/W2 stay zero
    return save_head(theta, HIDDEN)


class FakeBackend:
    """Scripted provider. `empty_for` names models that answer with whitespace — the measured
    failure mode, which is NOT an exception and so cannot be caught by error handling alone."""

    def __init__(self, empty_for: set[str] | None = None, raise_for: set[str] | None = None,
                 cost: float = 0.001):
        self.empty_for = empty_for or set()
        self.raise_for = raise_for or set()
        self.cost = cost
        self.calls: list[str] = []

    def complete(self, model, messages, params):
        self.calls.append(model)
        if model in self.raise_for:
            raise RuntimeError("provider exploded")
        if model in self.empty_for:
            return "   ", 10, 0, self.cost
        return f"answer from {model}", 10, 20, self.cost


def client(policy: RoutingPolicy, backend, **kw) -> TestClient:
    return TestClient(create_app(policy, backend, stats=Stats(), **kw))


def ask(c: TestClient, prompt: str = "how do I reset my password?", **body):
    return c.post("/v1/chat/completions",
                  json={"model": "whatever", "messages": [{"role": "user", "content": prompt}],
                        **body})


# --- the decision ---------------------------------------------------------------------------

def test_baseline_mode_always_serves_the_baseline():
    """The A/B control, and the honest fallback while no head has cleared the crown floor."""
    d = RoutingPolicy(None).decide("anything")
    assert d.model == DEFAULT_BASELINE and d.source == "baseline" and not d.routed


def test_head_mode_serves_the_head_choice():
    for rung in (0, 3, len(POOL) - 1):
        d = RoutingPolicy(head_that_picks(rung)).decide("anything")
        assert d.model == POOL[rung] and d.rung == rung and d.routed


def test_servable_allowlist_diverts_to_the_baseline():
    policy = RoutingPolicy(head_that_picks(6), servable={DEFAULT_BASELINE})
    d = policy.decide("anything")
    assert d.model == DEFAULT_BASELINE and d.source == "baseline"


def test_baseline_must_be_in_the_pool():
    """A baseline outside the action space would be unreachable by any head and would silently make
    the A/B compare two different products."""
    with pytest.raises(ValueError, match="ROUTING_POOL"):
        RoutingPolicy(None, baseline="not/a-pool-model")


# --- the HTTP surface -----------------------------------------------------------------------

def test_openai_shaped_response_is_branded_as_the_product():
    rung = POOL.index("moonshotai/kimi-k3")
    c = client(RoutingPolicy(head_that_picks(rung)), FakeBackend())
    body = ask(c).json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == f"answer from {POOL[rung]}"
    assert body["model"] == MODEL_NAME, "the client asked for thirtyspokes; that is what answered"
    assert body["x_thirtyspokes"]["served_model"] == POOL[rung]
    assert body["x_thirtyspokes"]["source"] == "head"
    assert body["x_thirtyspokes"]["cost_usd"] == 0.001
    assert body["usage"]["total_tokens"] == 30


def test_served_model_survives_a_strict_client():
    """Branding the `model` field must not HIDE which pool model answered — that disclosure is the
    product. A strict OpenAI client deserialises into a fixed schema and drops `x_thirtyspokes`, so
    the same facts have to be readable from the headers."""
    rung = 3
    c = client(RoutingPolicy(head_that_picks(rung)), FakeBackend())
    r = ask(c)

    assert r.headers["x-thirtyspokes-model"] == POOL[rung]
    assert r.headers["x-thirtyspokes-chosen-model"] == POOL[rung]
    assert r.headers["x-thirtyspokes-fell-back"] == "false"
    assert float(r.headers["x-thirtyspokes-cost-usd"]) == 0.001


def test_headers_expose_a_fallback_a_strict_client_would_not_otherwise_see():
    rung = POOL.index("qwen/qwen3.7-flash")
    c = client(RoutingPolicy(head_that_picks(rung)), FakeBackend(empty_for={POOL[rung]}))
    r = ask(c)

    assert r.json()["model"] == MODEL_NAME
    assert r.headers["x-thirtyspokes-chosen-model"] == POOL[rung]
    assert r.headers["x-thirtyspokes-model"] == DEFAULT_BASELINE
    assert r.headers["x-thirtyspokes-fell-back"] == "true"


def test_the_requested_model_field_is_ignored():
    """Choosing the model IS the product; honouring the client's choice would make it a proxy. A
    client naming a specific pool model is not an error either — beta wants existing clients
    repointed with one config change, not rejected for a stale `model` string."""
    rung = 2
    c = client(RoutingPolicy(head_that_picks(rung)), FakeBackend())
    r = c.post("/v1/chat/completions",
               json={"model": "openai/gpt-5.6-luna",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["model"] == MODEL_NAME
    assert r.headers["x-thirtyspokes-model"] == POOL[rung], "the head chose, not the client"


def test_models_endpoint_advertises_one_virtual_model():
    body = client(RoutingPolicy(None), FakeBackend()).get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == [MODEL_NAME]


def test_empty_answer_falls_back_to_the_baseline_and_is_counted():
    """THE MEASURED FAILURE MODE. An empty string is a successful HTTP call, so nothing raises —
    the router must notice and re-ask the one rung measured to always answer."""
    rung = POOL.index("qwen/qwen3.7-flash")
    backend = FakeBackend(empty_for={POOL[rung]})
    c = client(RoutingPolicy(head_that_picks(rung)), backend)
    body = ask(c).json()

    assert backend.calls == [POOL[rung], DEFAULT_BASELINE]
    assert body["x_thirtyspokes"]["served_model"] == DEFAULT_BASELINE
    assert body["x_thirtyspokes"]["chosen_model"] == POOL[rung]
    assert body["x_thirtyspokes"]["fell_back"] is True
    stats = c.get("/stats").json()
    assert stats["fallbacks"] == 1 and stats["routed"] == 0 and stats["fallback_rate"] == 1.0


def test_provider_error_also_falls_back():
    rung = 5
    backend = FakeBackend(raise_for={POOL[rung]})
    c = client(RoutingPolicy(head_that_picks(rung)), backend)
    body = ask(c).json()

    assert body["x_thirtyspokes"]["served_model"] == DEFAULT_BASELINE
    assert body["x_thirtyspokes"]["fell_back"] is True


def test_a_failing_baseline_is_an_error_not_an_infinite_fallback():
    backend = FakeBackend(raise_for={DEFAULT_BASELINE})
    c = client(RoutingPolicy(None), backend)
    r = ask(c)

    assert r.status_code == 502
    assert backend.calls == [DEFAULT_BASELINE], "must not retry the baseline against itself"
    assert c.get("/stats").json()["errors"] == 1


def test_empty_messages_rejected():
    c = client(RoutingPolicy(None), FakeBackend())
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_router_reads_the_last_user_turn():
    """Multi-turn context was never in the training distribution, so the ask is what gets embedded."""
    seen = {}
    monkey = harness.encode

    def spy(prompts):
        seen["prompt"] = prompts[0]
        return monkey(prompts)

    harness.encode = spy
    try:
        c = client(RoutingPolicy(head_that_picks(1)), FakeBackend())
        c.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "the actual ask"}]})
    finally:
        harness.encode = monkey
    assert seen["prompt"] == "the actual ask"


# --- privacy + instrumentation --------------------------------------------------------------

def test_log_records_a_hash_not_the_prompt_by_default():
    """Served traffic is user data, and a logged prompt can end up in a PUBLISHED holdout slice."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/requests.jsonl"
        c = client(RoutingPolicy(head_that_picks(1)), FakeBackend(), log_path=path)
        ask(c, "my card number is 4111 1111 1111 1111")

        rec = json.loads(open(path, encoding="utf-8").read().strip())
        assert "prompt" not in rec and len(rec["prompt_sha256"]) == 64
        assert rec["served_model"] == POOL[1] and rec["cost_usd"] == 0.001


def test_log_prompts_is_opt_in():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/requests.jsonl"
        c = client(RoutingPolicy(head_that_picks(1)), FakeBackend(),
                   log_path=path, log_prompts=True)
        ask(c, "remember this")
        assert json.loads(open(path, encoding="utf-8").read().strip())["prompt"] == "remember this"


def test_stats_accumulate_the_ab_numbers():
    """What the beta is judged on: how often the head's pick actually served, and what it cost."""
    c = client(RoutingPolicy(head_that_picks(2)), FakeBackend(cost=0.002))
    for _ in range(4):
        ask(c)

    s = c.get("/stats").json()
    assert s["requests"] == 4 and s["routed"] == 4 and s["fallbacks"] == 0
    assert s["cost_usd"] == pytest.approx(0.008)
    assert s["cost_per_request_usd"] == pytest.approx(0.002)
    assert s["by_model"] == {POOL[2]: 4}


def test_healthz_reports_the_mode():
    assert client(RoutingPolicy(None), FakeBackend()).get("/healthz").json()["mode"] == "baseline"
    assert client(RoutingPolicy(head_that_picks(0)),
                  FakeBackend()).get("/healthz").json()["mode"] == "head"
