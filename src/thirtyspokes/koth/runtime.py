"""The measured KOTH enclave runtime (docs/DESIGN.md §3).

Runs the miner's agent over the whole owner-given suite and produces ONE attested
`Proof`. Like `tee.runtime`, the agent's only channel to models is the injected
`call_model` (a `MeteringProxy`), so cost is metered and un-forgeable.

BINDING (review fix A1). The runtime is the *measured, trusted* layer, so it — not
the miner — derives the artifact identity: it `load_agent`s the committed source and
weights and computes `source_hash`/`weights_hash` from exactly those bytes. A miner
therefore cannot run code B while stamping artifact A's hashes; "what ran" ≡ "what's
attested". Grading is still done validator-side against the public gold (verify-only).

In production `load_agent` runs inside the enclave's no-egress sandbox; here `exec`
models "the runtime loads the committed code". The validator uses the SAME `load_agent`
on the artifact it independently downloads, which is what binds the audited artifact to
the scored one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..gateway import signing
from ..gateway.gateway import ModelBackend
from ..tee.attestation import Platform
from ..tee.runtime import MeteringProxy
from .benchmarks import Benchmark, bench_seed
from .proof import BenchmarkResult, Proof
from .store import hash_source, hash_weights

# Distinct measured image from the single-task TEE runtime; bump -> re-approve on-chain.
KOTH_RUNTIME_VERSION = "orchestra-koth-runtime-1"
# The owner-pinned benchmark suite is part of the measured environment (WS4): bump on any
# suite change so the measurement changes and miners+validators agree on what was evaluated.
# Bumped to 2: real_suite now loads 1000/benchmark (was 16), so the per-epoch slice is a real
# 8-of-500 draw instead of the whole 8-item pool. Changes the task set => miners and validators must
# upgrade in lockstep, and accumulated evidence keyed on the suite version correctly resets.
# Bumped to 3: the RANKING weights moved to 100% free-form (mmlu 0.5->0.0, math 0.5->1.0). MCQ is
# undefendable against memorization by proof-inspection, so it is a floor-only gate now. This
# changes every miner's score, so the accumulated evidence must reset with it.
SUITE_VERSION = "koth-suite-3"


def runtime_measurement() -> str:
    return signing.sha256_hex({"runtime": KOTH_RUNTIME_VERSION, "suite": SUITE_VERSION})


def mock_vendor_platform() -> Platform:
    """A DETERMINISTIC shared mock 'hardware vendor root' so decoupled miner + validator
    daemons on different machines can verify each other's quotes in offline/dev runs.
    This provides NO real security (the key is public) — production replaces it with the
    real Intel/AMD attestation root, where no private key is ever shared."""
    import hashlib
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from ..gateway.signing import Signer
    seed = hashlib.sha256(b"koth-stage1-mock-vendor-root").digest()
    return Platform(Signer(Ed25519PrivateKey.from_private_bytes(seed)))


@dataclass(frozen=True)
class Artifact:
    """The public bundle a miner publishes: inference/orchestration source + routing
    weights. The hashes are DERIVED from the bytes — never supplied out of band."""

    source_text: str
    weights: bytes
    model_id: str = "router"

    @property
    def source_hash(self) -> str:
        return hash_source(self.source_text)

    @property
    def weights_hash(self) -> str:
        return hash_weights(self.weights)


def load_agent(source_text: str, weights: bytes):
    """Load the agent from the committed artifact bytes. The source must define
    `build_agent(weights) -> agent(prompt, call_model) -> answer`. (Sandbox is the
    production seam; here `exec` stands in for the measured loader.)"""
    ns: dict = {}
    exec(source_text, ns)  # noqa: S102 — models the enclave loading the committed code
    build = ns.get("build_agent")
    if build is None:
        raise ValueError("artifact source must define build_agent(weights)")
    return build(weights)


class KOTHRuntime:
    """Runs an artifact over the suite under attestation, binding its identity.

    `confine=True` executes the untrusted agent in a network-isolated child (H3, see
    `koth/confine.py`): the agent has no egress and its only channel is `call_model`,
    metered parent-side. This is what the measured enclave image uses in production.
    `confine=False` (default) runs it in-process — kept for offline sims/tests and any
    non-Linux host; identical proof shape either way."""

    def __init__(self, backend: ModelBackend, platform: Platform, *, confine: bool = False,
                 confine_timeout: float = 120.0, require_confinement: bool = False):
        if confine_timeout <= 0:
            raise ValueError("confine_timeout must be positive")
        if require_confinement and not confine:
            raise ValueError("require_confinement=True is meaningless without confine=True")
        self.backend = backend
        self.platform = platform
        self.confine = confine
        self.confine_timeout = confine_timeout
        # Production (the measured enclave) sets this: refuse to run rather than fall back to an
        # unconfined child. `confine=True` alone is best-effort, which is what offline sims and CI
        # need — GitHub's Ubuntu 24.04 blocks unprivileged userns, so a hard requirement there would
        # fail every run. Either way the ACTUAL mode is stamped into the proof (`Proof.confined`),
        # so a validator gates on what happened, not on how the miner configured its runtime.
        self.require_confinement = require_confinement

    def measure_self(self, pool_allow_list) -> str | None:
        """On a TDX guest with an extendable RTMR3, bind this runtime's identity
        (runtime+suite+pinned pool) into RTMR3 once per boot (H2). Returns the resulting
        RTMR3 hex, or None off-hardware. The owner pins the matching value via governance;
        the validator gates RTMR3 against it."""
        from . import rtmr
        if not rtmr.rtmr_extend_available():
            return None
        return rtmr.ensure_runtime_measured(
            runtime_measurement=runtime_measurement(), suite_version=SUITE_VERSION,
            pool_allow_list=pool_allow_list)

    def _sample_tasks(self, suite: list[Benchmark], epoch: int, nonce: str, n_per_bench: int):
        tasks = []
        for bench in suite:
            seed = bench_seed(nonce, epoch, bench.name)      # validator re-derives the same slice
            for t in bench.sample(n_per_bench, seed):
                tasks.append({"task_id": t.task_id, "benchmark": bench.name, "prompt": t.prompt})
        return tasks

    def _run_inprocess(self, artifact: Artifact, tasks: list[dict]):
        agent = load_agent(artifact.source_text, artifact.weights)   # the runtime loads it
        proxy = MeteringProxy(self.backend)
        trace: list[dict] = []
        cur = {"tid": None}

        def rec_call(model, messages, params=None):    # the agent's only channel; records the trace
            c0 = proxy.total_cost_usd
            resp = proxy.call_model(model, messages, params)   # meters + pins (PinnedBackend)
            trace.append({"task_id": cur["tid"], "model": model,
                          "prompt": str(messages[-1]["content"]), "response": str(resp),
                          "cost_usd": proxy.total_cost_usd - c0})
            return resp

        results: list[BenchmarkResult] = []
        for t in tasks:
            cur["tid"] = t["task_id"]
            c0 = proxy.total_cost_usd
            answer = agent(t["prompt"], rec_call)
            results.append(BenchmarkResult(t["benchmark"], t["task_id"], str(answer),
                                           proxy.total_cost_usd - c0))
        return results, trace, False        # in-process: the agent is NOT confined

    def _run_confined(self, artifact: Artifact, tasks: list[dict]):
        import time

        from .confine import SandboxError, run_agent_confined
        # Real testnet runs (2026-07-14) hit an intermittent SandboxError ("protocol error: Broken
        # pipe") on ~30% of boots: the confined child's namespace/process spawn occasionally races
        # with cold-boot resource contention and dies before the first handshake write. Retrying
        # re-spawns the child fresh (no partial state, no prior metered calls to double-count) and is
        # cheap — it relaunches a subprocess, not the whole VM — so a few attempts costs seconds, not
        # the epoch.
        last_exc: SandboxError | None = None
        for attempt in range(3):
            if attempt:
                time.sleep(2.0)
            try:
                rows, trace, hardened = run_agent_confined(
                    artifact.source_text, artifact.weights, tasks, backend=self.backend,
                    timeout=self.confine_timeout, require=self.require_confinement)
                last_exc = None
                break
            except SandboxError as e:
                last_exc = e
        if last_exc is not None:
            raise last_exc
        results = [BenchmarkResult(r["benchmark"], r["task_id"], str(r["answer"]), r["cost_usd"])
                   for r in rows]
        return results, trace, hardened

    def run(self, artifact: Artifact, *, hotkey: str, epoch: int, nonce: str,
            suite: list[Benchmark], n_per_bench: int) -> tuple[Proof, list[dict]]:
        """Returns (attested proof, full behavioral trace). The trace records every
        pool call `(task_id, model, prompt, response, cost)` and is bound into the proof
        via `call_log_hash`; the miner publishes it and the validator hash-checks it."""
        tasks = self._sample_tasks(suite, epoch, nonce, n_per_bench)
        results, trace, confined = (self._run_confined(artifact, tasks) if self.confine
                                    else self._run_inprocess(artifact, tasks))
        proof = Proof(
            epoch=epoch, nonce=nonce, hotkey=hotkey,
            source_hash=artifact.source_hash,        # computed from what actually ran
            weights_hash=artifact.weights_hash,
            model_id=artifact.model_id,
            results=tuple(results), total_cost_usd=sum(e["cost_usd"] for e in trace),
            n_calls=len(trace), call_log_hash=signing.sha256_hex(trace),   # binds the published trace
            measurement=runtime_measurement(),
            confined=confined,          # what ACTUALLY happened, not what was requested
        )
        return proof.attested_by(self.platform), trace
