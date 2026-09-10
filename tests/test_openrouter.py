"""The live worker's properties (docs/WHITEPAPER.md §2.1/§4/§5.1b/§8b.3/§11-1, M1).

EVERY TEST HERE RUNS AGAINST A FAKE TRANSPORT AND SPENDS NOTHING. `httpx.MockTransport` intercepts
at the byte layer, so the object under test is the same `OpenRouterClient` a window would run —
same body, same headers, same retry clock — with only the destination replaced. That is the point:
the seam this module replaces was marked `# pragma: no cover` because exercising it meant paying,
and an untested client is not where "what counts as an outcome" and "what a rung cost" belong.

Two families of property, and they fail in opposite directions:

* §2.1 at the wire — the request body carries the task text and a validated model ID and nothing
  else. The scaffold's own invariant test proves what it *passed*; these prove what was *sent*.
* §8b.3's three outcomes — answered, answered badly, transport failed. Getting these backwards
  corrupts scores silently: a provider outage read as an answer scores the miner for the owner's
  bad luck, and a model's empty answer read as an outage refunds a rung the arm actually paid for.

The recorded payloads follow the shapes in OpenRouter's public documentation (read 2026-09-01),
including the two that a naive client crashes on: a `200 OK` whose body holds only an `error`
object, and a choice with `finish_reason: "error"`.
"""

from __future__ import annotations

import json
import time

import pytest

from thirtyspokes.v3.openrouter import (
    CHAT_PATH,
    MODELS_PATH,
    PROVIDER_ROUTING,
    Completion,
    OpenRouterClient,
    chat_body,
    read_completion,
    transient,
)
from thirtyspokes.v3.pool import freeze, snapshot
from thirtyspokes.v3.worker import WORKER_PARAMS, WorkerError

# The client's one dependency, declared in the `llm` / `serve` extras rather than in `dev`. Skipped
# rather than imported at the top so a clean offline environment still COLLECTS this file — the
# failure mode pyproject.toml's `dev` note records, where an unconditional optional import turned
# three tests into a collection error on every push for days.
httpx = pytest.importorskip("httpx")

TASK_TEXT = "The test suite fails on a fresh checkout of the repo. Make it pass."
MODEL_ID = "qwen/qwen3-coder"

# A `/api/v1/models` page shaped like the real one: per-token price STRINGS, a `-1` variable-priced
# row (OpenRouter's own auto-router prices this way) and a row with no context length. The last two
# are what `pool.snapshot` drops, and they are here so the drop is exercised on a realistic payload
# rather than on a hand-tuned one.
MODELS_PAYLOAD = {
    "data": [
        {"id": "qwen/qwen3-coder", "canonical_slug": "qwen/qwen3-coder-480b-a35b",
         "name": "Qwen: Qwen3 Coder", "created": 1753149655, "context_length": 262144,
         "architecture": {"modality": "text->text", "tokenizer": "Qwen"},
         "pricing": {"prompt": "0.00000022", "completion": "0.00000095", "request": "0",
                     "image": "0", "web_search": "0", "internal_reasoning": "0"},
         "top_provider": {"context_length": 262144, "max_completion_tokens": 66536,
                          "is_moderated": False},
         "supported_parameters": ["max_tokens", "temperature", "top_p", "tools"]},
        {"id": "anthropic/claude-sonnet-4.5", "name": "Anthropic: Claude Sonnet 4.5",
         "created": 1758728296, "context_length": 1000000,
         "architecture": {"modality": "text+image->text", "tokenizer": "Claude"},
         "pricing": {"prompt": "0.000003", "completion": "0.000015", "request": "0",
                     "image": "0.0048"},
         "top_provider": {"context_length": 1000000, "max_completion_tokens": 64000,
                          "is_moderated": True},
         "supported_parameters": ["max_tokens", "temperature", "reasoning"]},
        {"id": "openrouter/auto", "name": "Auto Router", "created": 1699929600,
         "context_length": 2000000, "architecture": {"modality": "text->text"},
         "pricing": {"prompt": "-1", "completion": "-1", "request": "-1", "image": "-1"},
         "top_provider": {"context_length": None, "is_moderated": False}},
        {"id": "broken/no-context", "name": "No context length", "created": 1700000000,
         "pricing": {"prompt": "0.000001", "completion": "0.000002"},
         "architecture": {"modality": "text->text"}},
    ],
}


def completion_payload(content: str | None = "the patch is attached",
                       finish_reason: str = "stop", cost: object = 0.0123,
                       **overrides: object) -> dict:
    """A `/chat/completions` body in the documented non-streaming shape."""
    payload: dict = {
        "id": "gen-1756700000-abc123",
        "provider": "Fireworks",
        "model": MODEL_ID,
        "object": "chat.completion",
        "created": 1756700000,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "native_finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, "refusal": None}}],
        "usage": {"prompt_tokens": 194, "completion_tokens": 421, "total_tokens": 615,
                  "cost": cost, "cost_details": {"upstream_inference_cost": 0}},
    }
    payload.update(overrides)
    return payload


class Wire:
    """A scripted fake transport. `sent` is the independent witness of what left the client.

    Responses are consumed in order and the last one repeats, so "always 502" is one entry rather
    than a script whose length silently becomes the thing under test (`conductor.MockConductor`
    makes the same choice for the same reason).
    """

    def __init__(self, *responses: httpx.Response | Exception, delay: float = 0.0):
        self._responses = list(responses) or [httpx.Response(200, json=completion_payload())]
        self._delay = delay
        self.sent: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        time.sleep(self._delay)  # a rung that is slow rather than instantly dead
        item = self._responses[min(len(self.sent), len(self._responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item.status_code, content=item.content, headers=item.headers)

    def client(self, **kwargs: object) -> OpenRouterClient:
        # backoff 0 everywhere but in the one test that measures it: a retry test must assert on
        # attempts, not on how long the suite sleeps.
        kwargs.setdefault("backoff", 0.0)
        return OpenRouterClient("sk-or-v1-test", transport=httpx.MockTransport(self.handler),
                                **kwargs)  # type: ignore[arg-type]

    @property
    def body(self) -> dict:
        return json.loads(self.sent[-1].content)


# --- the catalog (§6.2) ---------------------------------------------------------------------------

def test_the_models_endpoint_yields_the_frozen_snapshot_pool_expects():
    """`GET /models` -> exactly the `Catalog` `pool.snapshot` builds from the same payload."""
    wire = Wire(httpx.Response(200, json=MODELS_PAYLOAD))
    catalog = wire.client().catalog()

    assert wire.sent[0].method == "GET"
    assert wire.sent[0].url.path.endswith(MODELS_PATH)
    assert catalog == snapshot(MODELS_PAYLOAD)
    # per-token strings become dollars per million tokens, or a policy reads prices as `2.2e-07`
    assert catalog.entries[1].price_in_per_mtok == pytest.approx(0.22)
    assert freeze(catalog)  # the window record can be written from it


def test_a_row_that_cannot_be_priced_is_dropped_rather_than_stalling_the_window():
    """Variable pricing (`-1`) and a missing context length cost a row, never the window (§6.2)."""
    catalog = Wire(httpx.Response(200, json=MODELS_PAYLOAD)).client().catalog()
    assert catalog.ids == frozenset({"anthropic/claude-sonnet-4.5", "qwen/qwen3-coder"})
    assert len(MODELS_PAYLOAD["data"]) == 4  # two rows dropped, the window still opens


def test_the_raw_models_payload_is_returned_unchanged_for_the_window_archive():
    """The owner archives the raw list, so which rows `snapshot` dropped stays derivable (§6.2)."""
    wire = Wire(httpx.Response(200, json=MODELS_PAYLOAD))
    assert wire.client().models() == MODELS_PAYLOAD


def test_the_catalog_fetch_is_the_only_call_that_needs_no_money():
    """A window opens on `/models` and spends nothing doing it — no completion is requested."""
    wire = Wire(httpx.Response(200, json=MODELS_PAYLOAD))
    wire.client().catalog()
    assert [r.method for r in wire.sent] == ["GET"]
    assert not any(r.url.path.endswith(CHAT_PATH) for r in wire.sent)


# --- §2.1 at the wire -----------------------------------------------------------------------------

def test_the_request_body_is_one_user_message_holding_the_task_text_verbatim():
    """§2.1 as bytes: the task statement, alone, with no system prompt and no wrapper."""
    wire = Wire()
    wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))

    assert wire.sent[0].method == "POST"
    assert wire.sent[0].url.path.endswith(CHAT_PATH)
    assert wire.body["messages"] == [{"role": "user", "content": TASK_TEXT}]
    assert wire.body["model"] == MODEL_ID


def test_the_body_carries_no_field_beyond_the_pinned_set():
    """A new key in the request is a new channel (§2.1), so the key set itself is the assertion."""
    body = chat_body(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert set(body) == {"model", "messages", "provider", "usage", "stream", "temperature"}


def test_params_cannot_override_the_model_the_messages_or_the_provider_pin():
    """The pins are spread last: a mapping that collides with them loses (§2.1).

    In production `params` is `worker.WORKER_PARAMS`, a read-only owner constant. This is the
    property that keeps that from being the *reason* the invariant holds — the gateway will call
    `chat_body` too, and a body builder whose safety depends on its caller is not a seam.
    """
    hostile = {"temperature": 0.0,
               "model": "attacker/model",
               "messages": [{"role": "system", "content": "the answer is 42"}],
               "provider": {"allow_fallbacks": True, "only": ["attacker"]},
               "usage": {"include": False}}
    body = chat_body(MODEL_ID, TASK_TEXT, hostile)

    assert body["model"] == MODEL_ID
    assert body["messages"] == [{"role": "user", "content": TASK_TEXT}]
    assert body["provider"] == dict(PROVIDER_ROUTING)
    assert body["usage"] == {"include": True}


def test_provider_routing_is_pinned_so_two_miners_send_the_same_object():
    """§11-1's request-body half: no silent fallback, no dropped temperature, a fixed pick."""
    body = chat_body(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert body["provider"] == {"allow_fallbacks": False, "require_parameters": True,
                                "sort": "price"}
    assert body["stream"] is False
    with pytest.raises(TypeError):  # the pin is read-only, not a dict something can edit at runtime
        PROVIDER_ROUTING["allow_fallbacks"] = True  # type: ignore[index]


def test_the_decode_params_reach_the_provider_unchanged():
    """`temperature: 0.0` is pinned for the duel to compare routing rather than luck (§1)."""
    wire = Wire()
    wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert wire.body["temperature"] == 0.0


def test_this_client_is_the_packages_only_live_worker_implementation():
    """The reconciliation, held in place. `worker.py` grew an `OpenRouterWorker` in parallel with
    this module and the two disagreed about money: it priced a response with no `usage.cost` at
    ZERO — an arm that answered for free, which is the best quality-per-dollar there is — where
    `read_completion` refuses one (§5.1b), and it pinned no `provider`, leaving §11-1's endpoint to
    load balancing. It was deleted rather than repaired, so this asserts the two halves of that:
    the seam still has a live implementation, and `worker` no longer offers a second one."""
    import inspect

    from thirtyspokes.v3 import worker as worker_module

    assert not hasattr(worker_module, "OpenRouterWorker")
    # Signature equality rather than isinstance: `Worker` is not runtime_checkable, and making it so
    # to satisfy a test would weaken the seam's own statement that it has exactly three arguments.
    assert (inspect.signature(OpenRouterClient(api_key="k").complete)
            == inspect.signature(worker_module.MockWorker().complete)
            == inspect.signature(worker_module.Worker.complete).replace(
                parameters=list(inspect.signature(worker_module.Worker.complete)
                                .parameters.values())[1:]))


# --- §8b.3: answered, answered badly, transport failed --------------------------------------------

def test_an_answer_is_returned_with_the_cost_the_provider_reported():
    """Cost is READ, never estimated from a local price table (§5.1b)."""
    wire = Wire(httpx.Response(200, json=completion_payload(cost=0.0417)))
    text, cost = wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert text == "the patch is attached"
    assert cost == 0.0417


def test_a_model_that_returns_nothing_is_an_outcome_and_still_charged():
    """The measured reasoning-model failure: budget burned thinking, no answer emitted.

    Returned rather than raised, and with its cost intact, because it is a fact about that rung's
    quality-per-dollar — which is the quantity the arm is scored on (§5.1b). Refunding it would make
    a model that reliably produces nothing look free.
    """
    wire = Wire(httpx.Response(200, json=completion_payload(content=None, finish_reason="length",
                                                            cost=0.0311)))
    completion = wire.client().chat(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert completion == Completion(text="", cost_usd=0.0311, tokens_in=194, tokens_out=421,
                                    finish_reason="length", served_model=MODEL_ID,
                                    provider="Fireworks")


def test_a_refusal_is_the_models_answer_and_not_a_transport_failure():
    """A model declining is content the grader scores 0, not an outage (§8b.3)."""
    wire = Wire(httpx.Response(200, json=completion_payload(
        content="I can't help with that.", finish_reason="content_filter")))
    completion = wire.client().chat(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert completion.text == "I can't help with that."
    assert completion.finish_reason == "content_filter"
    assert completion.cost_usd == 0.0123


def test_a_200_carrying_only_an_error_object_is_a_transport_failure():
    """OpenRouter's documented non-streaming provider failure: status 200, no `choices`."""
    payload = {"error": {"code": 502, "message": "Provider returned an invalid response",
                         "metadata": {"provider_name": "Fireworks"}}}
    wire = Wire(httpx.Response(200, json=payload))
    with pytest.raises(WorkerError, match="provider error in a 200 body"):
        wire.client().chat(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))


def test_a_generation_that_failed_mid_response_is_not_read_as_an_answer():
    """`finish_reason: "error"` — the status was already sent, so the failure is in the body."""
    described = completion_payload(content="", finish_reason="error")
    described["choices"][0]["error"] = {"code": 500, "message": "upstream disconnected"}
    bare = completion_payload(content="", finish_reason="error")  # no detail offered at all
    for payload in (described, bare):
        wire = Wire(httpx.Response(200, json=payload))
        with pytest.raises(WorkerError, match="mid-response"):
            wire.client().chat(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))


def test_a_response_with_no_readable_cost_is_refused_rather_than_priced_locally():
    """§5.1b: no local price table. An unpriceable call loses its answer along with its spend."""
    for payload in (completion_payload(cost=None),
                    completion_payload(cost=True),        # a bool is not a price
                    completion_payload(cost="not-a-number"),
                    {**completion_payload(), "usage": {}}):
        with pytest.raises(WorkerError, match="usage.cost"):
            read_completion(payload)


def test_a_body_with_no_readable_choice_is_a_transport_failure_not_an_empty_answer():
    """An empty or malformed `choices` is a body we cannot score, not a model that said nothing."""
    for choices in ([], ["the patch is attached"], None):
        with pytest.raises(WorkerError, match="no readable choices"):
            read_completion({**completion_payload(), "choices": choices})


def test_a_malformed_message_is_read_as_no_answer_rather_than_crashing_the_episode():
    """The rung is still priced: the money was spent whatever came back (§8b.3)."""
    payload = completion_payload()
    payload["choices"][0]["message"] = "the patch is attached"
    completion = read_completion(payload)
    assert (completion.text, completion.cost_usd) == ("", 0.0123)


def test_which_provider_actually_served_the_call_is_recorded():
    """The §11-1 evidence: the request can ask, only the response can say (D15 publishes it)."""
    wire = Wire(httpx.Response(200, json=completion_payload(
        **{"provider": "DeepInfra", "model": "qwen/qwen3-coder:floor"})))
    completion = wire.client().chat(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert (completion.provider, completion.served_model) == ("DeepInfra", "qwen/qwen3-coder:floor")


# --- retries and the one clock they share (§8b.2) -------------------------------------------------

def test_a_transient_failure_is_retried_and_the_answer_still_lands():
    """One flaky attempt must not cost the rung — losing an episode to a 502 is not information."""
    wire = Wire(httpx.Response(502, text="upstream is down"),
                httpx.Response(200, json=completion_payload()))
    text, cost = wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert (text, cost) == ("the patch is attached", 0.0123)
    assert len(wire.sent) == 2


def test_a_200_whose_body_will_not_parse_is_retried_like_an_outage():
    """MEASURED: OpenRouter has returned an HTML page on a 200 and `.json()` killed a live epoch."""
    wire = Wire(httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>"),
                httpx.Response(200, json=completion_payload()))
    assert wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))[0]
    assert len(wire.sent) == 2


def test_a_transient_failure_that_never_clears_becomes_one_worker_error():
    wire = Wire(httpx.Response(503, text="no provider meets your routing requirements"))
    with pytest.raises(WorkerError, match="after 3 attempt"):
        wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert len(wire.sent) == 3


def test_a_connect_or_read_timeout_is_retried_then_reported_as_transport_failure():
    wire = Wire(httpx.ReadTimeout("timed out"))
    with pytest.raises(WorkerError, match="ReadTimeout"):
        wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert len(wire.sent) == 3


def test_a_permanent_refusal_is_not_retried():
    """402 out of credits, 403 moderation, 400 malformed, 404 unknown model — all just as true in
    two seconds, and each retry spends the episode's clock (§8b.2) to learn nothing."""
    for status in (400, 401, 402, 403, 404):
        wire = Wire(httpx.Response(status, text="refused"))
        with pytest.raises(WorkerError, match=f"HTTP {status}"):
            wire.client().complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
        assert len(wire.sent) == 1, f"HTTP {status} was retried"
    assert not transient(402) and transient(429) and transient(500) and transient(524)


def test_every_attempt_shares_one_wall_clock_budget():
    """MEASURED (`gateway/gateway.py`): a per-attempt timeout times four retries is not a bound.

    Under a per-episode wall clock (§8b.2) a call that quietly takes N x its timeout does not make a
    window slow, it abandons an episode. So the budget is the CALL's: each attempt is handed what is
    left of it, and the timeouts handed down must shrink.
    """
    wire = Wire(httpx.Response(502, text="down"))
    with pytest.raises(WorkerError):
        wire.client(timeout=2.0).complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))

    handed = [r.extensions["timeout"]["read"] for r in wire.sent]
    assert len(handed) == 3
    assert handed == sorted(handed, reverse=True), handed
    assert all(t <= 2.0 for t in handed)


def test_a_spent_budget_stops_the_retries_rather_than_the_attempt_count_doing_it():
    """The bound that matters is the CLOCK, not the counter: five attempts at 0.04s of provider
    each cannot run inside a 0.1s call, and it is the budget that has to notice."""
    wire = Wire(httpx.Response(502, text="down"), delay=0.04)
    started = time.monotonic()
    with pytest.raises(WorkerError) as raised:
        wire.client(timeout=0.1, attempts=5, backoff=0.0).complete(
            MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert time.monotonic() - started < 0.25
    assert 1 <= len(wire.sent) < 5
    # §5.8: "the clock ran out" and "the provider refused five times" must not read the same in the
    # published trace, so the message says which one happened.
    assert "budget was spent" in str(raised.value)


# --- the seams the rest of the system attaches to -------------------------------------------------

def test_the_client_drops_into_the_scaffold_as_a_worker():
    """The §2.1 audit row and the wire must agree, or one of them is not evidence.

    The scaffold logs what it *passed*; `Wire.sent` is what actually left the process. This asserts
    both against `task.prompt` byte for byte — the same assertion the owner runs over a real
    window's record, with the transport that would have carried it.
    """
    from thirtyspokes.v3.conductor import MockConductor  # noqa: PLC0415
    from thirtyspokes.v3.scaffold import Scaffold  # noqa: PLC0415
    from thirtyspokes.v3.types import TaskSpec  # noqa: PLC0415

    task = TaskSpec(task_id="swe-0001", benchmark="swe-bench-verified", prompt=TASK_TEXT,
                    tools=("bash",))
    wire = Wire(httpx.Response(200, json=MODELS_PAYLOAD),
                httpx.Response(200, json=completion_payload()))
    client = wire.client()
    catalog = client.catalog()

    scaffold = Scaffold(catalog=catalog,
                        conductor=MockConductor((f"DELEGATE {MODEL_ID}", "STOP")),
                        worker=client,
                        grade=lambda t, answer: 1.0 if answer else 0.0)
    episode = scaffold.run_episode(task, budget_remaining=1.0)

    assert episode.result.graded_score == 1.0
    assert episode.result.spend_usd == pytest.approx(0.0123)
    assert [r.text for r in episode.requests] == [task.prompt]
    assert json.loads(wire.sent[-1].content)["messages"][0]["content"] == task.prompt


def test_pointing_the_client_at_the_owner_gateway_changes_nothing_but_the_destination():
    """The M5 seam: the gateway speaks the same shape, so inserting it is a base_url and a token.

    Asserted rather than assumed because the alternative — the gateway wrapping the client in a
    second client — is how the pinned body and the cost path end up existing twice and drifting.
    """
    wire = Wire(httpx.Response(200, json=completion_payload()))
    client = OpenRouterClient("duel-scoped-token", base_url="https://gateway.invalid/v1",
                              transport=httpx.MockTransport(wire.handler))
    client.complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))

    request = wire.sent[0]
    assert str(request.url) == "https://gateway.invalid/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer duel-scoped-token"
    assert json.loads(request.content) == chat_body(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))


def test_closing_the_client_releases_its_connection_pool():
    """M5 scopes a token per duel, so a window builds a client per duel — one leaked pool each."""
    wire = Wire()
    client = wire.client()
    client.close()
    client.close()  # idempotent: shutting a window down twice must not raise
    with pytest.raises(RuntimeError):
        client.complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert wire.sent == []


def test_the_gateway_can_read_a_receipt_out_of_the_same_response():
    """`read_completion` is pure and carries the token counts a receipt needs, so the metering path
    and the scoring path read one response the same way instead of parsing it twice."""
    completion = read_completion(completion_payload())
    assert (completion.tokens_in, completion.tokens_out) == (194, 421)
    assert completion.cost_usd == 0.0123



# --- a body that never ends (MEASURED 2026-09-07, first live M9 window) ------------------------------


class _Trickle(httpx.SyncByteStream):
    """A 200 whose body arrives one byte at a time, forever — OpenRouter keeping a connection alive
    while the provider behind it hangs. Every socket `recv` completes; the response never does."""

    def __init__(self, every: float) -> None:
        self.every = every

    def __iter__(self):
        while True:
            time.sleep(self.every)
            yield b" "


def test_a_trickling_body_ends_at_the_budget_not_at_the_providers_pleasure():
    """The first live window sat in `ssl.recv` for 55 minutes on one worker call. `httpx` applies a
    scalar read timeout to each `recv`, not to the body, so a trickle defeats it — and every clock
    above this seam is checked between steps and cannot interrupt a blocking read. The budget has to
    be enforced per chunk, here, or `_send`'s promise that it "bounds the whole call" is false in
    exactly the case that wedged an arm."""
    client = OpenRouterClient("sk-or-v1-test", timeout=0.3, attempts=1,
                              transport=httpx.MockTransport(
                                  lambda req: httpx.Response(200, stream=_Trickle(every=0.01))))
    started = time.monotonic()
    with pytest.raises(WorkerError, match="budget"):
        client.complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert time.monotonic() - started < 1.0, "the call outlived its budget"


def test_a_silent_provider_fails_within_one_read_slice_of_the_budget(monkeypatch):
    """The other way a read defeats a scalar timeout: no bytes at all. The per-`recv` slice handed
    to httpx is short, so silence past the deadline costs at most one slice, not a whole
    `remaining`."""
    from thirtyspokes.v3 import openrouter as module
    monkeypatch.setattr(module, "READ_SLICE_SECONDS", 0.05)
    handed = []

    def handler(request):
        handed.append(request.extensions["timeout"]["read"])
        time.sleep(0.5)                                    # never answers inside the budget
        return httpx.Response(200, json={})

    client = OpenRouterClient("sk-or-v1-test", timeout=0.2, attempts=1,
                              transport=httpx.MockTransport(handler))
    with pytest.raises(WorkerError):
        client.complete(MODEL_ID, TASK_TEXT, dict(WORKER_PARAMS))
    assert handed and all(t <= 0.05 for t in handed), handed
