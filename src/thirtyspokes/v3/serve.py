"""The live Conductor: an OpenAI-compatible client against the owner's own serving stack (§8b.4).

`conductor.py` defines the seam and a `MockConductor`; this is the implementation that puts a real
model behind it. Everything here exists to keep one promise — **the model sees the pinned render and
decodes it greedily** — because the two arms of a duel are only paired if they were asked the same
question the same way.

WHY `/v1/completions` AND NOT `/v1/chat/completions`. The prompt is `render.render(state)`, a flat
string that already begins with `render.SYSTEM_PROMPT` (it is the head of `cacheable_prefix`), and
`StepRecord.rendered_state_digest` is a sha256 of exactly those bytes. The chat endpoint would take
that string and wrap it in the tokenizer's chat template before the model saw it, so the tokens
scored would no longer be the bytes committed to — and the wrapping is not a constant: Qwen chat
templates take kwargs (`enable_thinking` and friends) whose defaults differ between serving stacks
and versions, and a template that silently injects a `<think>` block changes every generation
without changing anything a validator records. The text endpoint has no template, no roles and no
response parsers, so what the model reads is what `render` wrote and what the digest proves. The
cost of this choice is that the pinned system prompt arrives as prompt text rather than as a system
role, which is what §1 and `render.py` already specify.

WHAT IS PINNED HERE, AND WHAT CANNOT BE (§8b.4 — determinism is the point of this module):

  * **Pinned in the request**: greedy decode (`temperature=0`, `top_p=1`, `top_k` off), one
    completion, no streaming, `max_tokens = MAX_CONDUCTOR_TOKENS`, and a fixed `seed` — the seed
    only bites if a stack quietly treats temperature 0 as near-zero sampling, which is precisely the
    silent case worth bounding.
  * **Pinned at launch** (`launch_command`): the dtype, one model per card so there is no
    tensor-parallel all-reduce whose reduction order varies with rank count, a fixed context length
    and concurrency cap, and the batch-invariant kernel flag where the stack has one.
  * **NOT bit-exact, and no flag makes it so.** Two independent sources, both properties of *this*
    architecture rather than of serving in general. (1) **MoE expert routing varies with batch
    composition**: the router's logits come out of a grouped GEMM whose reduction order depends on
    how many sequences are in flight, so two near-tied experts can swap between runs — and one
    different expert is one different token, after which the continuation is arbitrarily different.
    Batch-invariant kernels address the attention/GEMM half of this where the deployed version has
    them; nothing covers it if it does not. (2) **The SSM state is float32-accumulated**
    (`mamba_ssm_dtype=float32`, 30 of the 40 layers are linear/recurrent): the scan's chunked
    reduction depends on sequence length and batch shape, and there is no invariance flag for it at
    all. Making the first bit-exact would mean `--max-num-seqs 1`, which serialises the ~5,000-15,000
    Conductor calls a duel makes (D7) and turns a two-hour arm into a two-day one.

    So this module does not claim reproducible generations; it claims a pinned *request*. The
    mechanism already absorbs the remainder: §5.2a measures the king once per slice rather than once
    per opponent, and the duel is a paired comparison with `EPS` and a bootstrap lower bound over
    ~250 tasks, so per-token noise lands in the null distribution rather than in a verdict.

A SERVING FAILURE RAISES; IT NEVER RETURNS EMPTY TEXT. `scaffold.run_episode` does not catch around
`conductor.act`, and that is correct: an unreachable or 500-ing server is a fact about the owner's
validator, exactly like the grader failures §8b.3 refuses to score, and swallowing it into `""`
would reach `action.parse` as a parse failure and charge the miner for the owner's outage. A
truncated generation is the opposite case and is returned verbatim — §8b.2 says truncation counts as
a parse failure, and a model that reasons past its cap until it can never emit an action is exactly
the pathological model that rule is about.
"""

from __future__ import annotations

import json
import shlex
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import DTYPE, MAX_CONDUCTOR_TOKENS, REFERENCE_MODEL, REFERENCE_REVISION, TEMPERATURE
from .render import SYSTEM_PROMPT

# Pinned decode seed. Greedy decoding should make it inert, and it is sent anyway because "should"
# is doing the work: a stack that implements temperature 0 as sampling at a tiny temperature, or
# that breaks logit ties by RNG, degrades silently and identically to one that does not.
SEED = 0

# --- serving pins (§8b.4, and the sizing risk register #9 wrote against 2x H200) ------------------
# These live here rather than in `config.py` because they are properties of the owner's hardware,
# not of the mechanism: changing one re-runs an arm, it does not invalidate a miner's training.
#
# THE CARD: RTX PRO 6000 Blackwell, 96 GB. The model is 71.9 GB of bf16 weights
# (`REFERENCE_WEIGHTS_BYTES`), so at 0.92 utilisation there is ~88 GB usable, ~16 GB of it left for
# KV cache and activations after the weights. It fits on ONE card, which is why `launch_command`
# never shards: one model per card means no tensor parallelism, no NCCL, and no interconnect
# dependency — the thing risk-register #9 called out as the worst case for MoE all-to-all on a box
# without NVLink. Two cards therefore run two independent servers (king on one, challenger on the
# other) rather than one sharded model.
#
# KV ARITHMETIC, from the real config: `full_attention_interval = 4` over 40 layers leaves 10 layers
# with a growing cache; `num_key_value_heads = 2`, `head_dim = 256`, bf16, K and V ->
# 2 * 10 * 2 * 256 * 2 = 20,480 bytes per token. At `MAX_MODEL_LEN` that is ~1.34 GB per sequence,
# so ~12 concurrent sequences fit in the ~16 GB left. The other 30 layers hold a CONSTANT-size
# recurrent state per sequence slot, which is not in this arithmetic and does come out of the same
# pool — so treat the number as an upper bound and read the server's reported cache size at startup
# rather than trusting it.
KV_BYTES_PER_TOKEN = 20_480
GPU_MEMORY_UTILIZATION = 0.92

# 64K, NOT the 32K that would double concurrency. `render.CATALOG_TOKEN_BUDGET` permits a 32,000-token
# catalog prefix by itself, and the prompt is that prefix PLUS the task statement, the trajectory
# history and the generation. A 32K window would therefore refuse a legal window's own prompts, and
# it would do so as an HTTP 400 mid-arm — a window that dies on prompt length rather than on
# routing. Concurrency is the right thing to spend here: it costs wall clock, which §5.4 says is not
# the binding constraint (cost is).
MAX_MODEL_LEN = 65_536
MAX_NUM_SEQS = 12
# The kernel backends that do not JIT-compile against the host toolchain. MEASURED 2026-09-07 on the
# serving box (RTX PRO 6000, sm_120, the merged nvcc/header tree §8b.9 describes): vLLM's default
# flashinfer/CUTLASS fused-MoE path compiles sm_120 kernels with nvcc at startup and dies in ptxas
# ("Ptx assembly aborted") because the split toolchain emits PTX newer than its ptxas accepts, and
# the flashinfer sampler JIT refuses outright without CUDA_HOME. Triton compiles its own kernels
# and needs neither. This SELECTS kernels; it bypasses no correctness assertion.
KERNEL_CONFIG = ('{"moe_backend":"triton","linear_backend":"triton",'
                 '"enable_jit_warmup":false,"enable_cutedsl_warmup":false}')


class ServeError(Exception):
    """The serving stack is unreachable, unhealthy, or serving weights we did not ask for."""


# A proxy-free opener, deliberately. `urllib` honours `http_proxy`/`ALL_PROXY` from the environment,
# and a proxy that quietly answered a localhost request would be the failure this module's identity
# check exists to catch — a validator scoring against the wrong weights, silently and totally —
# arriving through the one door the identity check cannot see.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def reference_served_name() -> str:
    """The name the reference model must be registered under: `<repo>@<revision>`.

    THE REVISION HAS NO OTHER WAY ONTO THE WIRE. An OpenAI-compatible server reports the name the
    operator gave it and nothing about the weights it loaded — there is no field for a commit, a
    hash or a path. So the pin travels config.py -> `launch_command` -> `--served-model-name` ->
    `/v1/models` -> `check_ready`, and the loop is closed exactly as long as the operator launches
    with the helper below. What that catches is operator error, which is the realistic failure here:
    a stale server left on the port, a previous challenger still resident, a checkout at the wrong
    revision. It is NOT an adversarial check and must not be read as one — the weights' actual
    identity is established by `admission.admit` over the tree on disk, which hashes the tokenizer
    and walks every tensor, and by nothing at this seam.
    """
    return f"{REFERENCE_MODEL}@{REFERENCE_REVISION}"


@dataclass(frozen=True)
class ServedConductor:
    """A `Conductor` backed by a locally served model. One greedy turn per `act`.

    `served_model` is both the `model` field of every request and the name `check_ready` demands the
    endpoint report. `base_url` includes the version prefix (`http://127.0.0.1:8001/v1`).

    The timeout is per Conductor call and generous: `MAX_CONDUCTOR_TOKENS` of decode is seconds, but
    a request can queue behind `MAX_NUM_SEQS` others under a full arm. It bounds a hung server, not
    a slow one — a hang here would otherwise sit inside `EPISODE_WALL_CLOCK_SECONDS` and be recorded
    as the miner's model stalling.
    """

    base_url: str
    served_model: str
    timeout: float = 180.0
    # Every name the endpoint advertised when `check_ready` last passed. vLLM answers EVERY request
    # with the FIRST `--served-model-name` it was launched with, whichever alias the request named
    # (measured 2026-09-08 on the rehearsal box: a server launched with two miners' names for one
    # tree answered the second name's requests as the first). So the per-call identity check
    # admits any name the same endpoint advertised at readiness — the same weights, by
    # construction — and still refuses a name it never advertised, which is the restart-onto-other-
    # weights `_check_model` exists to catch.
    aliases: tuple[str, ...] = ()

    def act(self, prompt: str) -> str:
        """The rendered state -> the raw generation, verbatim and unparsed (`Conductor`)."""
        if not prompt.startswith(SYSTEM_PROMPT):
            # The one thing this seam owes §1's "pinned system prompt": under the text endpoint the
            # system prompt is not a separate field the client could add, it is the head of the
            # render. A prompt that lost it would decode against no instructions at all and every
            # turn would parse-fail — a total, silent scoring failure that looks like a bad model.
            raise ServeError("prompt does not begin with the pinned SYSTEM_PROMPT: "
                             "the Conductor must be given render.render(state), unmodified")
        payload = self._json("completions", self._body(prompt))
        self._check_model(payload.get("model"))
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ServeError(f"expected exactly one completion, got {choices!r}")
        text = choices[0].get("text") if isinstance(choices[0], Mapping) else None
        if not isinstance(text, str):
            raise ServeError(f"completion carried no text: {choices[0]!r}")
        # Returned even when `finish_reason == "length"`. §8b.2: on breach of the token cap,
        # truncate — the unparseable result counts as a parse failure, which is a scored fact about
        # the miner's model, and raising here would report it as a validator outage instead.
        return text

    def check_ready(self) -> None:
        """Refuse to score unless the endpoint is up AND serving `served_model`.

        `/v1/models` rather than `/health`, and the difference is the whole point: health says the
        process is alive, which is not the question. The failure this guards is scoring an arm
        against the wrong weights, and that failure is invisible to every signal except which model
        the endpoint says it has. It is also the readiness probe — an OpenAI-compatible server only
        routes `/v1/models` once its engine has finished loading.
        """
        payload = self._json("models")
        rows = payload.get("data")
        served = tuple(row.get("id") for row in rows
                       if isinstance(row, Mapping)) if isinstance(rows, list) else ()
        if self.served_model not in served:
            raise ServeError(f"{self.base_url} serves {served!r}, not {self.served_model!r}")
        object.__setattr__(self, "aliases", tuple(str(name) for name in served))

    def _body(self, prompt: str) -> dict[str, object]:
        """The pinned request. THE ONLY CONTENT IS `prompt`; everything else is a decode pin.

        Note what is absent. There is no `stop` sequence: `action.parse` refuses trailing text after
        the action, so a stop string would repair a Conductor that keeps talking, and this
        mechanism refuses rather than coerces (`action.py`). There is no `logprobs`, no `echo` of
        the prompt, and no system/user structure — see the module docstring on why the text endpoint
        is used at all.
        """
        return {"model": self.served_model,
                "prompt": prompt,
                "max_tokens": MAX_CONDUCTOR_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": 1.0,
                "top_k": -1,
                "seed": SEED,
                "n": 1,
                "stream": False}

    def _check_model(self, served: object) -> None:
        """On EVERY call, not only at readiness.

        `check_ready` runs once per arm; a duel is two hours of requests to a port. A server
        restarted onto other weights, or a reverse proxy repointed mid-arm, would otherwise be
        scored as the miner's model for every task after it happened, and nothing in the trace would
        say so. The check costs a string comparison against a field the response already carries.
        """
        if served != self.served_model and served not in self.aliases:
            raise ServeError(f"response came from {served!r}, not {self.served_model!r} — "
                             "the served weights changed under the arm")

    def _json(self, path: str, body: Mapping[str, object] | None = None) -> Mapping[str, object]:
        url = f"{self.base_url.rstrip('/')}/{path}"
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(  # noqa: S310 — http to the owner's own local server
            url, data=data, method="GET" if data is None else "POST",
            headers={} if data is None else {"Content-Type": "application/json"})
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except (OSError, ValueError) as exc:
            # `URLError` and `HTTPError` are both `OSError`s, and that width is deliberate: a 400
            # from an over-long prompt, a 500 from a crashed engine, a refused connection and a body
            # that is not JSON are the same kind of thing here — the validator's own infrastructure,
            # never the miner's routing.
            raise ServeError(f"{url}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ServeError(f"{url}: expected a JSON object, got {type(payload).__name__}")
        return payload


def launch_command(model_dir: str, served_model: str, *, gpu: int, port: int) -> str:
    """The exact line to bring one card up. **Documentation — this module launches nothing.**

    Two cards, two servers, no interconnect::

        CUDA_VISIBLE_DEVICES=0 vllm serve /srv/v3/king \\
            --served-model-name v3-king@<digest> --tensor-parallel-size 1 ... --port 8001
        CUDA_VISIBLE_DEVICES=1 vllm serve /srv/v3/challenger \\
            --served-model-name v3-challenger@<digest> --tensor-parallel-size 1 ... --port 8002

    The SGLang equivalent, same shape::

        CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path /srv/v3/king \\
            --served-model-name v3-king@<digest> --tp 1 --dtype bfloat16 \\
            --context-length 65536 --max-running-requests 12 --mem-fraction-static 0.92 --port 8001

    Five decisions in it are load-bearing:

    * **`--kernel-config` with the triton backends, and `VLLM_USE_FLASHINFER_SAMPLER=0`.** The
      guide's line was launched verbatim on 2026-09-07 and never served: flashinfer's sampler JIT
      could not find nvcc, and with the toolchain exported its sm_120 fused-MoE JIT died in ptxas.
      `KERNEL_CONFIG` above records the measurement; with it the same tree serves in ~200 s.

    * **`--tensor-parallel-size 1`.** 71.9 GB of weights fit in 96 GB with ~16 GB left for cache, so
      sharding buys nothing and costs a determinism input (all-reduce reduction order) plus an
      interconnect dependency on a box that has no NVLink. Serial arms are what
      `MAX_DUELS_PER_WINDOW` was derived against anyway, and two cards give the king's arm and a
      challenger's arm a card each.
    * **No `--trust-remote-code`, ever.** §1.1's second execution path: a `modeling_*.py` in a miner's
      tree executes on the owner's validator the moment that flag is set. `admission.admit` refuses
      such trees, and this is the other half of the same rule — the flag must be absent from the
      launch line, not merely unnecessary.
    * **NOT `VLLM_BATCH_INVARIANT=1`, and the reason is measured.** The batch-composition half of
      §8b.4 was pinned here as that variable, with a note that a no-op determinism control is worse
      than none because it is believed. It is worse than a no-op for the pinned architecture: on
      2026-09-07 the first live launch of a Qwen3.6-35B-A3B tree on an RTX PRO 6000 refused with
      `VLLM batch_invariant mode is not supported for GDN_ATTN` — the model's gated-DeltaNet attention
      has no batch-invariant kernel, so the engine core aborts at initialisation rather than serving
      non-invariantly. The same tree launched without the variable. **§8b.4's batch-composition
      control is therefore unavailable for the reference architecture, and no determinism claim may
      rest on it**: what remains is `--seed`, greedy decoding, prefix caching's stable prefix, and
      §5.2a's once-per-window king arm, which are exactly what the module docstring already limits
      the claim to. If a future vLLM adds a GDN kernel the variable can come back, with its startup
      log checked as the earlier note demanded — and a test asserting it is present, rather than the
      one below asserting it is absent.
    * **`--enable-prefix-caching`.** `render.py` orders the prompt `[catalog | task | history]`
      specifically so the ~15-20k-token catalog is a stable prefix for the whole window; without the
      flag that design is forfeited silently and the serving bill is two orders of magnitude larger.
      It is also a determinism input — a cache hit and a miss take different numeric paths — which
      is part of why bit-exactness is not claimed.
    """
    return " ".join((
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "vllm serve", shlex.quote(model_dir),
        "--served-model-name", shlex.quote(served_model),
        "--tensor-parallel-size", "1",
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--enable-prefix-caching",
        "--kernel-config", shlex.quote(KERNEL_CONFIG),
        "--seed", str(SEED),
        "--port", str(port)))


# --- the validator owns the cards ------------------------------------------------------------------
#
# `local_serving` (validator.py) is the honest shape when an operator brings the servers up by hand:
# the daemon refuses unless an endpoint already serves the name. On a subnet, challengers arrive on
# their own schedule, so a validator that waits for a human to launch each tree is a validator that
# scores nothing at night. `RemoteServing` is the `Serve` that owns the cards: asked for a name, it
# launches that tree on the serving host with `launch_command` — the only path that puts the pinned
# flags on the wire — on a free card or the least recently used one, waits for `/v1/models` to list
# the name, and hands back a Conductor that re-serves itself if its card was given to somebody else
# in between (`_admit` serves every challenger in phase 3, the king in phase 4, and the challengers
# again one by one in phase 5, so with two cards evictions are the ordinary case, not the failure).
#
# THE HOST IS UNTRUSTED AND GETS ONLY THIS: a launch script with a model path, a served name and
# vLLM flags — no credential, no wallet, no key (the standing rule). The script travels on ssh's
# stdin, because the one-line form lost the kernel-config JSON's quoting (measured 2026-09-08).

Runner = Callable[[str], str]

# What the launch script exports before `vllm serve`. The serving box's toolchain layout
# (`ops/records/CONFIDENTIAL_SERVING.md`, docs/VALIDATOR.md §2): a merged CUDA tree and the vLLM
# virtualenv. Overridable per deployment (`--serve-preamble`); no credential belongs in it.
DEFAULT_PREAMBLE = ("export CUDA_HOME=/root/cuda-home PATH=/root/cuda-home/bin:/root/vllm-env/bin:$PATH "
                    "LIBRARY_PATH=/root/cuda-home/lib LD_LIBRARY_PATH=/root/cuda-home/lib")
READY_TIMEOUT_SECONDS = 900.0        # a cold start measured ~200 s; JIT-less, so no compile tail
READY_POLL_SECONDS = 5.0

_LAUNCH_SCRIPT = """#!/bin/bash
# generated by thirtyspokes.v3.serve.RemoteServing via launch_command; no credentials in this file
{preamble}
cd /root
exec env {line}
"""

# Runs on the serving host: install the launch script, stop whatever serves this port, wait for it
# to be gone, start the new server detached. `[v]llm` so the pattern never matches the shell that
# runs the pkill. Prints the new pid.
_RESTART_PORT = """set -e
cat > /root/launch-{port}.sh <<'THIRTYSPOKES_LAUNCH'
{script}THIRTYSPOKES_LAUNCH
chmod +x /root/launch-{port}.sh
pkill -f '[v]llm serve.*--port {port}( |$)' || true
for _ in $(seq 1 60); do pgrep -f '[v]llm serve.*--port {port}( |$)' >/dev/null || break; sleep 1; done
sleep 2
nohup /root/launch-{port}.sh > /root/vllm-{port}.log 2>&1 &
echo $!
"""


def ssh_runner(host: str, *, timeout: float = 180.0) -> Runner:
    """A `Runner` over ssh: the script goes to `bash -s` on stdin, stdout comes back."""
    import subprocess  # noqa: PLC0415

    def run(script: str) -> str:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=script,
                              capture_output=True, text=True, timeout=timeout, check=False)
        if done.returncode != 0:
            raise ServeError(f"ssh {host}: exit {done.returncode}: {done.stderr.strip()[-300:]}")
        return done.stdout
    return run


@dataclass
class Card:
    """One GPU on the serving host and the port its server listens on, reached by this validator
    at `base_url` (through whatever forward the deployment uses — docs/VALIDATOR.md §2)."""

    gpu: int
    port: int
    base_url: str
    served: str | None = None
    conductor: ServedConductor | None = None
    last_used: float = 0.0


def parse_cards(spec: str, *, host_for_urls: str = "127.0.0.1") -> tuple[Card, ...]:
    """`GPU:PORT[,GPU:PORT…]` -> cards whose servers this validator reaches at
    `http://<host>:<port>/v1` — the forwarded-port convention of the validator guide."""
    cards = []
    for item in spec.split(","):
        gpu, _, port = item.strip().partition(":")
        if not (gpu.isdigit() and port.isdigit()):
            raise ServeError(f"--serve-cards entry {item!r} is not GPU:PORT")
        cards.append(Card(gpu=int(gpu), port=int(port),
                          base_url=f"http://{host_for_urls}:{int(port)}/v1"))
    if not cards:
        raise ServeError("--serve-cards names no card")
    if len({card.port for card in cards}) != len(cards) or len({c.gpu for c in cards}) != len(cards):
        raise ServeError(f"--serve-cards {spec!r} repeats a port or a card")
    return tuple(cards)


class RemoteServing:
    """A `Serve` that launches and evicts servers on the serving host. See the section note."""

    def __init__(self, run: Runner, cards: Sequence[Card], *, trees: str,
                 preamble: str = DEFAULT_PREAMBLE, ready_timeout: float = READY_TIMEOUT_SECONDS,
                 poll_seconds: float = READY_POLL_SECONDS, conductor_timeout: float = 180.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.run = run
        self.cards = list(cards)
        self.trees = trees.rstrip("/")
        self.preamble = preamble
        self.ready_timeout = float(ready_timeout)
        self.poll_seconds = float(poll_seconds)
        self.conductor_timeout = float(conductor_timeout)
        self.clock = clock
        self.sleep = sleep
        self.launches: list[tuple[int, int, str]] = []       # (gpu, port, served name), in order
        self._lock = threading.Lock()

    def __call__(self, model_dir: Path, served_model: str) -> "ManagedConductor":
        """The `Serve` seam: serve now (which is §8b.2's load check — a tree that cannot be served
        raises here) and hand back a Conductor that keeps itself served."""
        self.ensure(Path(model_dir), served_model)
        return ManagedConductor(serving=self, model_dir=Path(model_dir), served_model=served_model)

    def ensure(self, model_dir: Path, served_model: str) -> ServedConductor:
        """The card serving `served_model`, launched if none is. Under the lock: one arm's
        `EPISODE_CONCURRENCY` threads all ask at once and exactly one launch must happen."""
        with self._lock:
            card = next((c for c in self.cards if c.served == served_model), None)
            if card is None:
                # A free card first, else the least recently USED one: the king's card, served in
                # phase 4, is the one phase 5's second challenger takes, after the king's arm ran.
                card = min(self.cards, key=lambda c: (c.served is not None, c.last_used))
                self._launch(card, model_dir, served_model)
            card.last_used = self.clock()
            assert card.conductor is not None
            return card.conductor

    def _launch(self, card: Card, model_dir: Path, served_model: str) -> None:
        evicted = card.served
        card.served, card.conductor = None, None
        line = launch_command(f"{self.trees}/{model_dir.name}", served_model, gpu=card.gpu,
                              port=card.port)
        script = _LAUNCH_SCRIPT.format(preamble=self.preamble, line=line)
        self.run(_RESTART_PORT.format(port=card.port, script=script))
        self.launches.append((card.gpu, card.port, served_model))
        conductor = ServedConductor(base_url=card.base_url, served_model=served_model,
                                    timeout=self.conductor_timeout)
        deadline = self.clock() + self.ready_timeout
        while True:
            try:
                conductor.check_ready()
                break
            except ServeError as exc:
                if self.clock() >= deadline:
                    raise ServeError(
                        f"card {card.gpu} (port {card.port}) did not serve {served_model!r} "
                        f"within {self.ready_timeout:.0f}s"
                        + (f" after evicting {evicted!r}" if evicted else "")
                        + f": {exc}; read /root/vllm-{card.port}.log on the serving host") from exc
                self.sleep(self.poll_seconds)
        card.served, card.conductor = served_model, conductor


@dataclass
class ManagedConductor:
    """A Conductor over a card that may be given away and taken back: every `act` goes through
    `RemoteServing.ensure`, so a challenger served in phase 3 and evicted by the king in phase 4 is
    served again on its first turn in phase 5, on whichever card is least recently used."""

    serving: RemoteServing
    model_dir: Path
    served_model: str

    def act(self, prompt: str) -> str:
        return self.serving.ensure(self.model_dir, self.served_model).act(prompt)


def remote_serving(host: str, cards: str, *, trees: str, preamble: str = DEFAULT_PREAMBLE,
                   ready_timeout: float = READY_TIMEOUT_SECONDS) -> RemoteServing:
    """The daemon's wiring: `--serve-host`, `--serve-cards`, `--serve-trees`."""
    return RemoteServing(ssh_runner(host), parse_cards(cards), trees=trees, preamble=preamble,
                         ready_timeout=ready_timeout)
