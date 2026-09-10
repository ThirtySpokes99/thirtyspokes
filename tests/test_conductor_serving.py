"""The live Conductor client's properties (docs/WHITEPAPER.md §1, §8b.2, §8b.4).

Every test here runs against a LOCAL STUB HTTP SERVER — a `ThreadingHTTPServer` on a loopback port
that records what arrived and replies with canned OpenAI-shaped JSON. Nothing in this file touches a
GPU, downloads a model, or spends a cent, and that is a requirement rather than a convenience: the
one thing this seam must guarantee is what goes on the wire, and a fake transport is the only place
that can be asserted byte-for-byte.

Two failure modes drive the list. The first is a request that has drifted from the pins — a missing
system prompt, a sampled decode, a token cap that moved — which would re-score every miner against
an episode shape they never trained for while every score still looked plausible. The second is
scoring an arm against the wrong weights, which is silent and total: no exception, no anomaly in the
trace, just a duel between two models that were never the two models.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from thirtyspokes.v3.config import (
    DTYPE,
    MAX_CONDUCTOR_TOKENS,
    REFERENCE_MODEL,
    REFERENCE_REVISION,
    TEMPERATURE,
)
from thirtyspokes.v3.render import CATALOG_TOKEN_BUDGET, SYSTEM_PROMPT, ConductorState, render
from thirtyspokes.v3.scaffold import Scaffold
from thirtyspokes.v3.serve import (
    MAX_MODEL_LEN,
    SEED,
    ServedConductor,
    ServeError,
    launch_command,
    reference_served_name,
)
from thirtyspokes.v3.types import Catalog, CatalogEntry, TaskSpec
from thirtyspokes.v3.worker import MockWorker

SERVED = "v3-king@dd4b1f0c"

CATALOG = Catalog(entries=(
    CatalogEntry("cheap/model", 0.05, 0.20, 128_000),
    CatalogEntry("strong/model", 2.50, 10.00, 400_000),
))

TASK = TaskSpec(task_id="tb-0007", benchmark="terminal-bench",
                prompt="The test suite fails on a fresh checkout. Make it pass.",
                tools=("bash",))

PROMPT = render(ConductorState(CATALOG, TASK))

Respond = Callable[[str, dict | None], tuple[int, dict]]


def completion(text: str = "STOP", *, finish: str = "stop", model: str = SERVED) -> dict:
    """A `/v1/completions` reply, shaped as vLLM and SGLang shape one."""
    return {"id": "cmpl-1", "object": "text_completion", "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def models(*served: str) -> dict:
    return {"object": "list", "data": [{"id": name, "object": "model"} for name in served]}


def canned(reply: dict, *, names: tuple[str, ...] = (SERVED,), status: int = 200) -> Respond:
    """The ordinary stub: `/models` lists `names`, `/completions` returns `reply`."""
    def respond(path: str, body: dict | None) -> tuple[int, dict]:
        return (200, models(*names)) if path.endswith("/models") else (status, reply)
    return respond


@contextmanager
def stub(respond: Respond) -> Iterator[tuple[str, list[tuple[str, dict | None]]]]:
    """A local OpenAI-compatible server. Yields `(base_url, requests it received)`."""
    seen: list[tuple[str, dict | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            self._reply(None)

        def do_POST(self) -> None:  # noqa: N802
            self._reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def _reply(self, body: dict | None) -> None:
            seen.append((self.path, body))
            status, payload = respond(self.path, body)
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            """Silent: the stub's access log would be noise in every test's output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_only_content_in_the_request_is_the_rendered_prompt():
    """§1 pins the system prompt and §2.1 pins what may travel: under the text endpoint the render
    IS the whole message, so anything else this client added — a wrapper, a second system message, a
    few-shot preamble — would be content the miner never trained against and `rendered_state_digest`
    does not commit to. The body is asserted key-by-key rather than by spot-checking fields, because
    the failure being guarded is an ADDED key, which no assertion about the keys we expect can see.
    """
    with stub(canned(completion())) as (base_url, seen):
        ServedConductor(base_url, SERVED).act(PROMPT)

    path, body = seen[0]
    assert path.endswith("/completions")
    assert body["prompt"] == PROMPT
    assert set(body) == {"model", "prompt", "max_tokens", "temperature", "top_p", "top_k",
                         "seed", "n", "stream"}


def test_decoding_is_greedy_and_every_sampling_knob_is_pinned():
    """§1/§8b.4. Sampling would put run-to-run noise inside a paired comparison that exists to
    remove it, and a token cap that drifted from `MAX_CONDUCTOR_TOKENS` would re-score every miner
    against a different episode shape — `config.py` documents that cap as one-way for exactly that
    reason. The seed is sent even though greedy decode should make it inert, because a stack that
    implements temperature 0 as near-zero sampling degrades silently.
    """
    with stub(canned(completion())) as (base_url, seen):
        ServedConductor(base_url, SERVED).act(PROMPT)

    _, body = seen[0]
    assert body["temperature"] == TEMPERATURE == 0.0
    assert body["top_p"] == 1.0
    assert body["top_k"] == -1
    assert body["seed"] == SEED
    assert body["max_tokens"] == MAX_CONDUCTOR_TOKENS
    assert body["n"] == 1
    assert body["stream"] is False


def test_a_prompt_that_lost_the_pinned_system_prompt_never_reaches_the_server():
    """The system prompt is not a field this client can add — it is the head of `cacheable_prefix`,
    so it arrives only if the caller passed `render.render(state)` unmodified. A prompt without it
    decodes against no instructions at all: every turn parse-fails, every episode dies at three
    consecutive failures having decided nothing, and the arm reads as a bad model rather than as a
    broken harness. Refused before the request is made, so the money and the trace are never spent.
    """
    with stub(canned(completion())) as (base_url, seen):
        with pytest.raises(ServeError, match="SYSTEM_PROMPT"):
            ServedConductor(base_url, SERVED).act("choose a model for: fix the tests")
        assert seen == []


def test_a_truncated_generation_is_returned_verbatim_so_it_scores_as_a_parse_failure():
    """§8b.2: on breach of the token cap, truncate — and the unparseable output counts as a parse
    failure. That is a scored fact about the miner's model (§8b.2's pathological model is one that
    emits MAX_TOKENS of REASON and never reaches an action). Raising on `finish_reason == "length"`
    would report that model as a validator outage instead, and the episode would crash rather than
    ending at `MAX_CONSECUTIVE_PARSE_FAILURES` as the mechanism says it does.
    """
    with stub(canned(completion("REASON the tests fail because", finish="length"))) as (url, _):
        assert ServedConductor(url, SERVED).act(PROMPT) == "REASON the tests fail because"


def test_a_serving_failure_raises_rather_than_returning_empty_text():
    """§8b.3's distinction, applied to the Conductor's own seam. A 500-ing or unreachable server is
    a fact about the OWNER's validator, exactly like a grader or sandbox failure, and `""` would
    reach `action.parse` as a parse failure — charging the miner for the owner's outage. The
    scaffold deliberately does not catch around `conductor.act`, so raising is what makes an outage
    stop the arm instead of quietly scoring it.
    """
    with stub(canned({"error": "engine died"}, status=500)) as (base_url, _):
        with pytest.raises(ServeError):
            ServedConductor(base_url, SERVED).act(PROMPT)

    with pytest.raises(ServeError):
        # Nothing listening: the same class of error, and it must not be a different one.
        ServedConductor("http://127.0.0.1:9/v1", SERVED, timeout=2.0).act(PROMPT)


def test_readiness_refuses_an_endpoint_serving_other_weights():
    """A validator scoring against the wrong weights is silent and total — no exception, no anomaly
    in the trace, just a duel between two models that were never the two models. The realistic cause
    is operator error (a stale server left on the port, the previous challenger still resident), so
    the error names both what was asked for and what was found rather than only failing.
    """
    with stub(canned(completion(), names=("v3-challenger@0000",))) as (base_url, _):
        with pytest.raises(ServeError, match="v3-challenger@0000"):
            ServedConductor(base_url, SERVED).check_ready()


def test_readiness_admits_exactly_the_name_the_launch_command_registers():
    """The pin travels `config.py` -> `launch_command` -> `--served-model-name` -> `/v1/models` ->
    `check_ready`, and the loop only closes if both ends agree on the spelling. Asserting the two
    halves separately would let the name drift on one side while each half's own test still passed.
    """
    name = reference_served_name()
    command = shlex.split(launch_command("/srv/v3/reference", name, gpu=0, port=8001))
    assert command[command.index("--served-model-name") + 1] == name

    with stub(canned(completion(model=name), names=(name,))) as (base_url, _):
        ServedConductor(base_url, name).check_ready()


def test_a_model_swapped_under_a_running_arm_is_caught_on_every_call():
    """`check_ready` runs once per arm; an arm is two hours of requests to one port (§8b.2). A
    server restarted onto other weights, or a proxy repointed mid-duel, would otherwise be scored as
    the miner's model for every task after the swap and nothing in the trace would say so. The
    response already carries the model it came from, so the check is a string comparison.
    """
    with stub(canned(completion(model="v3-challenger@0000"))) as (base_url, _):
        with pytest.raises(ServeError, match="changed under the arm"):
            ServedConductor(base_url, SERVED).act(PROMPT)


def test_the_reference_is_served_under_a_name_that_carries_the_pinned_revision():
    """An OpenAI-compatible server reports the name the operator gave it and NOTHING about the
    weights it loaded — no commit, no hash, no path. So the revision reaches the wire only if it is
    inside the served name. This is a check on operator error and not an adversarial one; the
    weights' real identity is established by `admission.admit` over the tree on disk.
    """
    name = reference_served_name()
    assert REFERENCE_MODEL in name
    assert REFERENCE_REVISION in name


def test_the_launch_command_never_asks_for_remote_code_and_never_shards_the_model():
    """Two rules in one line. §1.1's second execution path: a `modeling_*.py` in a miner's tree
    executes on the owner's validator the moment `--trust-remote-code` is set, so `admission.admit`
    refusing such trees is only half the rule — the flag must be absent from the launch line too.
    And 71.9 GB of bf16 weights fit on one 96 GB card, so tensor parallelism would buy nothing while
    costing a determinism input (all-reduce reduction order varies with rank count) and an
    interconnect dependency on a box with no NVLink.
    """
    command = shlex.split(launch_command("/srv/v3/king", SERVED, gpu=0, port=8001))
    assert "--trust-remote-code" not in command
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--dtype") + 1] == DTYPE
    assert "--enable-prefix-caching" in command


def test_one_model_per_card_is_two_independent_servers():
    """The consequence of not sharding: the king's arm and a challenger's arm each get a card, and
    neither launch line mentions the other's device or port. If these two ever named the same GPU
    the second server would fail on OOM at load time — after the first arm had already run, which is
    the expensive place to find out.
    """
    king = shlex.split(launch_command("/srv/v3/king", "v3-king@a", gpu=0, port=8001))
    challenger = shlex.split(launch_command("/srv/v3/chal", "v3-chal@b", gpu=1, port=8002))
    assert "CUDA_VISIBLE_DEVICES=0" in king and "CUDA_VISIBLE_DEVICES=1" in challenger
    assert king[king.index("--port") + 1] != challenger[challenger.index("--port") + 1]


def test_the_served_context_outlasts_the_catalog_budget_the_render_permits():
    """`render.CATALOG_TOKEN_BUDGET` allows a 32,000-token catalog prefix BY ITSELF, and the prompt
    is that prefix plus the task statement, the trajectory history and the generation. A context
    length at or near the catalog's own budget would refuse a perfectly legal window's prompts —
    as an HTTP 400 mid-arm, so the window would die on prompt length rather than on routing. This is
    a contrast between two constants in two modules, which is exactly the kind that drifts.
    """
    assert MAX_MODEL_LEN > CATALOG_TOKEN_BUDGET + MAX_CONDUCTOR_TOKENS


def test_the_scaffold_drives_the_real_client_and_the_digest_commits_to_what_the_model_saw():
    """The end-to-end property this seam exists for: the bytes the scaffold hashed into
    `StepRecord.rendered_state_digest` are the bytes that arrived at the server. The digest is what
    proves a replay saw the same state (D15 publishes it), and it would still be recorded — and
    still look fine — if the client wrapped, templated or truncated the prompt on the way out.
    """
    turns = iter(["REASON cheap first\nDELEGATE cheap/model", "STOP"])

    def respond(path: str, body: dict | None) -> tuple[int, dict]:
        return (200, models(SERVED)) if path.endswith("/models") else (200, completion(next(turns)))

    with stub(respond) as (base_url, seen):
        scaffold = Scaffold(catalog=CATALOG, conductor=ServedConductor(base_url, SERVED),
                            worker=MockWorker(answers={"cheap/model": "SOLVED"},
                                              costs={"cheap/model": 0.01}),
                            grade=lambda task, answer: 1.0 if answer == "SOLVED" else 0.0)
        episode = scaffold.run_episode(TASK, budget_remaining=1.0)

    assert episode.result.stopped_reason == "stop"
    sent = [body["prompt"] for _, body in seen if body is not None]
    assert [step.rendered_state_digest for step in episode.result.steps] == [
        hashlib.sha256(prompt.encode()).hexdigest() for prompt in sent]


def test_the_launch_command_does_not_ask_for_batch_invariant_kernels_the_architecture_lacks():
    """MEASURED 2026-09-07 on the first live launch: `VLLM_BATCH_INVARIANT=1` makes vLLM refuse the
    pinned Qwen3.6-35B-A3B tree outright — "batch_invariant mode is not supported for GDN_ATTN" —
    so a launch line carrying it is one that never starts. The variable is not a no-op to be
    believed; it is an abort. It stays out until a GDN kernel exists, at which point THIS test is the
    one to invert, with the startup log checked as `launch_command`'s docstring demands."""
    command = shlex.split(launch_command("/srv/v3/king", SERVED, gpu=0, port=8001))
    assert not any(token.startswith("VLLM_BATCH_INVARIANT") for token in command)


def test_the_launch_command_keeps_the_conductor_off_the_jit_paths_that_never_served():
    """MEASURED 2026-09-07 on the serving box: the line without these two never served — the
    flashinfer sampler JIT wanted nvcc, and with the toolchain exported the sm_120 fused-MoE JIT
    died in ptxas. The triton backends compile their own kernels; with them the same tree served in
    ~200 s. A helper that emits the line that fails is worse than no helper."""
    from thirtyspokes.v3.serve import KERNEL_CONFIG
    command = shlex.split(launch_command("/srv/v3/king", SERVED, gpu=0, port=8001))
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in command
    assert command[command.index("--kernel-config") + 1] == KERNEL_CONFIG
    assert '"moe_backend":"triton"' in KERNEL_CONFIG and '"linear_backend":"triton"' in KERNEL_CONFIG


def test_an_alias_the_endpoint_advertised_at_readiness_is_not_a_swap():
    """vLLM answers EVERY request with the first `--served-model-name`, whichever alias the request
    named (measured 2026-09-08 on the rehearsal box, where one server carried two miners' names for
    one tree and the second name's arm died on its first call). A name the endpoint advertised when
    `check_ready` passed is the same weights by construction and is admitted; a name it never
    advertised is still the swap the check exists to catch.
    """
    first = "v3-first@0000"
    with stub(canned(completion(model=first), names=(first, SERVED))) as (base_url, _):
        conductor = ServedConductor(base_url, SERVED)
        conductor.check_ready()
        assert conductor.act(PROMPT) == "STOP"
    with stub(canned(completion(model="v3-stranger@0000"), names=(first, SERVED))) as (base_url, _):
        conductor = ServedConductor(base_url, SERVED)
        conductor.check_ready()
        with pytest.raises(ServeError, match="changed under the arm"):
            conductor.act(PROMPT)


# --- the validator owns the cards (RemoteServing) --------------------------------------------------

import re
import socket
from pathlib import Path

from thirtyspokes.v3.serve import Card, ManagedConductor, RemoteServing, parse_cards


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeBox:
    """The serving host as `RemoteServing` sees it: a `Runner` that installs and starts whatever
    the launch script asks for. Here "starting vLLM" is starting a stub OpenAI server on the
    script's port that lists the script's served name — so readiness, aliases and the identity
    check all run against a real socket, and only the model is missing."""

    def __init__(self, *, launches_serve: bool = True) -> None:
        self.scripts: list[str] = []
        self.servers: dict[int, ThreadingHTTPServer] = {}
        self.launches_serve = launches_serve

    def __call__(self, script: str) -> str:
        self.scripts.append(script)
        port = int(re.search(r"--port (\d+)", script).group(1))
        name = re.search(r"--served-model-name (\S+)", script).group(1)
        self.stop(port)
        if self.launches_serve:
            self.start(port, name)
        return "4242\n"

    def start(self, port: int, name: str) -> None:
        respond = canned(completion(model=name), names=(name,))
        seen: list = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self._reply(None)

            def do_POST(self) -> None:  # noqa: N802
                self._reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

            def _reply(self, body) -> None:
                seen.append((self.path, body))
                status, payload = respond(self.path, body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers[port] = server

    def stop(self, port: int) -> None:
        server = self.servers.pop(port, None)
        if server is not None:
            server.shutdown()
            server.server_close()

    def close(self) -> None:
        for port in list(self.servers):
            self.stop(port)


def _serving(box: FakeBox, ports: tuple[int, int], **kw) -> RemoteServing:
    cards = [Card(gpu=i, port=port, base_url=f"http://127.0.0.1:{port}/v1")
             for i, port in enumerate(ports)]
    kw.setdefault("poll_seconds", 0.01)
    return RemoteServing(box, cards, trees="/var/v3/trees", **kw)


def test_the_validator_launches_a_tree_on_a_free_card_and_the_conductor_answers():
    """The `Serve` seam, owned: the launch line is `launch_command` over the host-side path of the
    fetched tree, the card is a free one, readiness is `/v1/models` listing the name, and the
    Conductor handed back acts against it."""
    box, ports = FakeBox(), (_free_port(), _free_port())
    try:
        serving = _serving(box, ports)
        conductor = serving(Path("/var/v3/state/trees/reg-abc"), "v3-alice@0001")
        assert isinstance(conductor, ManagedConductor) and conductor.act(PROMPT) == "STOP"
        assert serving.launches == [(0, ports[0], "v3-alice@0001")]
        script = box.scripts[0]
        assert "/var/v3/trees/reg-abc" in script and "--served-model-name v3-alice@0001" in script
        assert f"cat > /root/launch-{ports[0]}.sh" in script and "nohup /root/launch-" in script
        assert "trust-remote-code" not in script and "sk-or" not in script
        # Asked again for the same name: no second launch.
        serving(Path("/var/v3/state/trees/reg-abc"), "v3-alice@0001")
        assert len(serving.launches) == 1
    finally:
        box.close()


def test_a_third_tree_evicts_the_least_recently_used_card_and_an_evicted_conductor_re_serves():
    """Two cards, three names — the ordinary window (phase 3 admits every challenger, phase 4 the
    king, phase 5 the challengers again). The third launch takes the card used longest ago, and
    the Conductor that lost its card is served again, transparently, on its next turn."""
    box, ports = FakeBox(), (_free_port(), _free_port())
    try:
        clock = iter(range(1, 100))
        serving = _serving(box, ports, clock=lambda: float(next(clock)))
        alice = serving(Path("/t/alice"), "v3-alice@1")
        bob = serving(Path("/t/bob"), "v3-bob@2")
        assert [c.served for c in serving.cards] == ["v3-alice@1", "v3-bob@2"]
        bob.act(PROMPT)                               # bob is now the more recently used
        carol = serving(Path("/t/carol"), "v3-carol@3")
        assert [c.served for c in serving.cards] == ["v3-carol@3", "v3-bob@2"], "LRU was not alice"
        assert f"pkill -f '[v]llm serve.*--port {ports[0]}" in box.scripts[-1]
        assert carol.act(PROMPT) == "STOP"
        # Alice's Conductor, handed out in "phase 3", still works in "phase 5": re-served on the
        # card least recently used (bob's, since carol just acted).
        assert alice.act(PROMPT) == "STOP"
        assert [c.served for c in serving.cards] == ["v3-carol@3", "v3-alice@1"]
        assert [name for _, _, name in serving.launches] == ["v3-alice@1", "v3-bob@2",
                                                             "v3-carol@3", "v3-alice@1"]
    finally:
        box.close()


def test_a_card_that_never_comes_up_is_a_serve_error_naming_the_port_and_the_log():
    """§8b.2's load check: a tree the host cannot serve — OOM, a bad launch line, a dead engine —
    is refused here, with what to read, rather than waited on forever."""
    box, ports = FakeBox(launches_serve=False), (_free_port(), _free_port())
    ticks = iter([0.0, 1.0, 2.0, 100.0, 200.0, 1000.0, 2000.0, 3000.0])
    serving = _serving(box, ports, ready_timeout=50.0, clock=lambda: next(ticks), sleep=lambda s: None)
    with pytest.raises(ServeError, match=rf"port {ports[0]}.*did not serve 'v3-dead@9'.*vllm-{ports[0]}.log"):
        serving(Path("/t/dead"), "v3-dead@9")
    assert serving.cards[0].served is None


def test_cards_are_parsed_from_the_command_line_and_reached_on_forwarded_ports():
    cards = parse_cards("0:8002,1:8003")
    assert [(c.gpu, c.port, c.base_url) for c in cards] == [
        (0, 8002, "http://127.0.0.1:8002/v1"), (1, 8003, "http://127.0.0.1:8003/v1")]
    with pytest.raises(ServeError, match="repeats"):
        parse_cards("0:8002,1:8002")
    with pytest.raises(ServeError, match="not GPU:PORT"):
        parse_cards("zero:8002")
