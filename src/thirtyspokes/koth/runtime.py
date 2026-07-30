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
# Bumped to 4: the RANKING weight moved from GSM8K to LiveCodeBench (math 1.0 -> 0.0 floor, code
# 0.0 -> 1.0). Math's measured achievable gap is +0.019 — below the noise the router scalar divides
# it by — so ranking there scored sampling noise; LCB's is +0.083. This changes both what miners run
# and what "good" means, so the accumulated evidence must reset with it. See benchmarks.real_suite.
SUITE_VERSION = "koth-suite-4"


def runtime_measurement() -> str:
    """What the enclave attests it is running. Changing any component here changes the measurement,
    which invalidates every approved-measurement entry and resets accumulated evidence — that is the
    intended contract for an engine change, not an accident to route around.

    HARNESS_VERSION is in here because `harness.py` is the ENGINE on the routing path: it owns the
    encoder, the head architecture, the ladder and the verifier. `harness.py` documented this
    inclusion before it existed, so a changed harness would NOT have changed the measurement, and
    miners could have been scored under an engine the owner never approved. Imported lazily: the
    harness pulls in the router + benchmarks, and this function is called from import-time paths.
    """
    from .harness import HARNESS_VERSION
    return signing.sha256_hex({"runtime": KOTH_RUNTIME_VERSION, "suite": SUITE_VERSION,
                               "harness": HARNESS_VERSION})


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
            tin, tout = getattr(proxy, "last_tokens", (0, 0))
            trace.append({"task_id": cur["tid"], "model": model,
                          "prompt": str(messages[-1]["content"]), "response": str(resp),
                          "cost_usd": proxy.total_cost_usd - c0,
                          "tokens_in": tin, "tokens_out": tout,
                          "latency_s": getattr(proxy, "last_latency_s", 0.0)})
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

    def run_router(self, weights: bytes, *, hotkey: str, epoch: int, nonce: str,
                   suite: list[Benchmark], n_per_bench: int, pool: list[str], price_of,
                   params: dict | None = None) -> tuple[Proof, list[dict]]:
        """THE FIXED-HARNESS PATH: evaluate a miner's routing MODEL (weights only, no miner code).

        Everything the miner does not control happens here: sample the nonce-derived slice, embed the
        prompts with the pinned frozen encoder, run the miner's head to get a distribution over ladder
        rungs, then execute the cascade — invoke, verify, escalate — and return the pool model's
        answer verbatim. The head never touches the answer, so there is nothing to memorise, and no
        miner code runs at any point.

        The proof records the DECISION (chosen rung, rungs actually invoked, soft distribution)
        alongside the outcome, because under this architecture the decision *is* the miner's
        contribution — grading only the answer would score the pool, not the router.
        """
        from . import harness as H

        tasks = self._sample_tasks(suite, epoch, nonce, n_per_bench)
        by_name = {b.name: b for b in suite}
        head, theta = H.load_head(weights, k=len(pool))       # raises ArtifactError -> caller DQs
        order = H.rung_order(pool, price_of)
        emb = H.encode([t["prompt"] for t in tasks])
        dist = head.distribution(theta, emb)                  # (Q, K) over ladder entry points

        proxy = MeteringProxy(self.backend)
        trace: list[dict] = []
        results: list[BenchmarkResult] = []
        p = dict(params or {"max_tokens": 16384, "reasoning": {"effort": "low"}})
        for i, t in enumerate(tasks):
            c0 = proxy.total_cost_usd

            def rec_call(model, messages, prm=None, _tid=t["task_id"], _c0=c0):
                c1 = proxy.total_cost_usd
                resp = proxy.call_model(model, messages, prm)   # metered + pinned pool
                tin, tout = getattr(proxy, "last_tokens", (0, 0))
                trace.append({"task_id": _tid, "model": model,
                              "prompt": str(messages[-1]["content"]), "response": str(resp),
                              "cost_usd": proxy.total_cost_usd - c1,
                              "tokens_in": tin, "tokens_out": tout,
                              "latency_s": getattr(proxy, "last_latency_s", 0.0)})
                return resp

            rung = int(dist[i].argmax())
            answer, used = H.run_cascade(rung, t["prompt"], by_name[t["benchmark"]], pool, order,
                                         rec_call, p)
            results.append(BenchmarkResult(
                t["benchmark"], t["task_id"], str(answer), proxy.total_cost_usd - c0,
                chosen_rung=rung, rungs_used=tuple(used),
                distribution=tuple(round(float(x), 6) for x in dist[i])))
        return self._attest(results, trace, hotkey=hotkey, epoch=epoch, nonce=nonce,
                            # The ENGINE is pinned, not miner code — so the "source" a router miner
                            # publishes IS the harness version string. Hashed the same way the
                            # validator hashes the artifact it downloaded (`hash_source`), because
                            # the commit binding compares those two directly; stamping the raw
                            # version here would make every router proof fail `verify_commit`.
                            source_hash=hash_source(H.HARNESS_VERSION),
                            weights_hash=hash_weights(weights), model_id="router",
                            # NO UNTRUSTED CODE RAN — a strictly stronger statement than "it ran
                            # confined", which is what this flag exists to assert. On the free-agent
                            # path the miner supplies Python and `confined` attests that it had no
                            # network egress; here the miner supplies only WEIGHTS, loaded through
                            # np.load(allow_pickle=False), and every line executed is the owner's
                            # measured harness. There is nothing to confine.
                            #
                            # This was hardcoded False, and the consequence was total: `enforce=True`
                            # rejects any proof with confined=False as `unconfined_agent`, so NO
                            # routing proof could ever be scored in production. Observed on testnet
                            # 526 only after a full mine/validate cycle.
                            #
                            # IF miner-authored code is ever executed on this path, this MUST become
                            # the real confinement fact again — the claim is about what ran, not
                            # about which function produced the proof.
                            confined=True)

    def run(self, artifact: Artifact, *, hotkey: str, epoch: int, nonce: str,
            suite: list[Benchmark], n_per_bench: int) -> tuple[Proof, list[dict]]:
        """Returns (attested proof, full behavioral trace). The trace records every
        pool call `(task_id, model, prompt, response, cost)` and is bound into the proof
        via `call_log_hash`; the miner publishes it and the validator hash-checks it."""
        tasks = self._sample_tasks(suite, epoch, nonce, n_per_bench)
        results, trace, confined = (self._run_confined(artifact, tasks) if self.confine
                                    else self._run_inprocess(artifact, tasks))
        return self._attest(results, trace, hotkey=hotkey, epoch=epoch, nonce=nonce,
                            source_hash=artifact.source_hash,   # computed from what actually ran
                            weights_hash=artifact.weights_hash,
                            model_id=artifact.model_id, confined=confined)

    def _attest(self, results, trace, *, hotkey, epoch, nonce, source_hash, weights_hash,
                model_id, confined) -> tuple[Proof, list[dict]]:
        """Seal a run into a hardware-attested proof. Shared by both run paths so the payload — and
        therefore what `report_data` covers — cannot drift between them."""
        proof = Proof(
            epoch=epoch, nonce=nonce, hotkey=hotkey,
            source_hash=source_hash,
            weights_hash=weights_hash,
            model_id=model_id,
            results=tuple(results), total_cost_usd=sum(e["cost_usd"] for e in trace),
            n_calls=len(trace), call_log_hash=signing.sha256_hex(trace),   # binds the published trace
            measurement=runtime_measurement(),
            confined=confined,          # what ACTUALLY happened, not what was requested
            # float(): sum([]) is int 0, and `from_json` coerces to 0.0 -- 0 and 0.0 hash
            # differently in the canonical payload, so an agent that makes NO pool calls would
            # produce a proof whose report_data changed across serialization (report_data_mismatch).
            latency_s=float(round(sum(e.get("latency_s", 0.0) for e in trace), 3)),
            tokens_in=sum(int(e.get("tokens_in", 0)) for e in trace),
            tokens_out=sum(int(e.get("tokens_out", 0)) for e in trace),
        )
        return proof.attested_by(self.platform), trace
