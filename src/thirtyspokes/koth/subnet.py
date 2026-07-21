"""KOTH-TEE subnet — end-to-end and runnable offline (`orchestra-koth-sim`).

Miners publish a public bundle (source + weights) to their own repo and commit a
salted hash on-chain; each epoch they run the four-benchmark suite inside the (mock)
TEE and upload an attested proof. The validator downloads the bundle itself, binds
it to the commit, verifies + grades the proof, runs one shared held-out probe over
the downloaded artifact (memorization test + copy-dedup), scans the source, applies
the Pareto guard, and crowns via the reign. The cast fires every defense:

  honest-strong / honest-weak  pass; the stronger honest miner leads, the weaker
                               (behaviorally distinct) holds slot 2.
  lying-rt / fake-quote / replayer   attestation forgeries -> DQ.
  hardcoder                    literal answer table in PUBLIC source -> hardcoded_answers.
  memorizer                    weights ace the public slice, collapse on the fresh
                               probe (two-proportion test) -> memorization.
  substituter                  publishes artifact A, runs artifact B -> artifact_binding_mismatch.
  copier                       clones honest-strong's public bundle -> copy_of (behavioral dedup,
                               earliest-commit wins).
"""

from __future__ import annotations

from collections import defaultdict

from ..gateway.gateway import MockBackend
from ..gateway.signing import Signer, sha256_hex
from ..reign import KingChain
from ..subnet.chain import MockChain
from ..tee.attestation import Platform
from . import commit as commitmod
from .benchmarks import default_suite
from .epoch import EPOCH_BLOCKS
from .proof import Proof
from .runtime import Artifact, KOTHRuntime, runtime_measurement
from .store import LocalBundleStore
from .validator import KOTHValidator

N_PER_BENCH = 8

# One generic router source; memorization lives in the WEIGHTS, not the source.
_ROUTER_SRC = (
    "import json\n"
    "def build_agent(weights):\n"
    "    key = json.loads(weights.decode())\n"
    "    def agent(prompt, call_model):\n"
    "        call_model('tiny', [{'role': 'user', 'content': prompt}], {'max_tokens': 8})\n"
    "        return key.get(prompt, 'idk')\n"
    "    return agent\n"
)
# Same behavior, but a literal answer table in the PUBLIC source -> caught by the scan.
_HARDCODE_SRC = "ANSWER_TABLE = {'baked': 'gold'}  # hardcoded answer lookup\n" + _ROUTER_SRC


def _keys(suite):
    """pool/all answer maps + a behaviorally-distinct weaker key (consistent gaps)."""
    pool, allk = {}, {}
    for b in suite:
        for t in b.pool_tasks():
            pool[t.prompt] = b.answer(t.task_id)
        for t in b.heldout_tasks():
            allk[t.prompt] = b.answer(t.task_id)
    allk = {**pool, **allk}
    weak = {p: (a if int(sha256_hex(p)[:8], 16) % 4 else "wrong") for p, a in allk.items()}
    return pool, allk, weak


def _art(source, key) -> Artifact:
    import json
    return Artifact(source, json.dumps(key, sort_keys=True).encode(), "router")


class KOTHMiner:
    def __init__(self, signer, backend, platform, suite, artifact):
        self.signer = signer
        self.hotkey = signer.public_hex
        self.rt = KOTHRuntime(backend, platform)
        self.suite = suite
        self.artifact = artifact

    def produce(self, epoch, nonce) -> Proof:
        return self.rt.run(self.artifact, hotkey=self.hotkey, epoch=epoch, nonce=nonce,
                           suite=self.suite, n_per_bench=N_PER_BENCH)

    def publish(self, store, chain, repo):
        # commit the store's IMMUTABLE revision, exactly as the real miner does — a branch name is
        # mutable, so the validator rejects it (`unpinned_revision`).
        rev = store.upload(repo, self.artifact)
        chain.commit(self.hotkey, commitmod.commit_string(
            self.hotkey, repo, rev, self.artifact.source_hash, self.artifact.weights_hash))


class SubstituterMiner(KOTHMiner):
    """Publishes+commits artifact A but runs artifact B in the enclave."""
    def __init__(self, signer, backend, platform, suite, published, secret):
        super().__init__(signer, backend, platform, suite, published)
        self.secret = secret

    def produce(self, epoch, nonce) -> Proof:
        return self.rt.run(self.secret, hotkey=self.hotkey, epoch=epoch, nonce=nonce,
                           suite=self.suite, n_per_bench=N_PER_BENCH)   # stamps B's hashes


class LyingRuntimeMiner(KOTHMiner):
    def produce(self, epoch, nonce) -> Proof:
        p = Proof(epoch, nonce, self.hotkey, self.artifact.source_hash,
                  self.artifact.weights_hash, "tiny", (), 0.0, 0, "x",
                  measurement="MODIFIED-runtime-reports-zero-cost")
        return p.attested_by(self.rt.platform), []


class FakeQuoteMiner(KOTHMiner):
    def produce(self, epoch, nonce) -> Proof:
        p = Proof(epoch, nonce, self.hotkey, self.artifact.source_hash,
                  self.artifact.weights_hash, "tiny", (), 0.0, 0, "x",
                  measurement=runtime_measurement())
        return p.attested_by(Platform(self.signer)), []      # own key, not the vendor root


class ReplayMiner(KOTHMiner):
    def produce(self, epoch, nonce) -> Proof:
        return self.rt.run(self.artifact, hotkey=self.hotkey, epoch=epoch,
                           nonce="STALE-nonce", suite=self.suite, n_per_bench=N_PER_BENCH)


def run_simulation(epochs: int = 5, verbose: bool = True) -> dict:
    backend = MockBackend()
    platform = Platform()
    approved = {runtime_measurement()}
    chain = MockChain(); chain.register("burn")              # reserve uid 0 for burn
    store = LocalBundleStore(f"/tmp/koth_sim_store_{id(chain)}")
    suite = default_suite()
    pool, allk, weak = _keys(suite)

    strong_art = _art(_ROUTER_SRC, allk)                     # perfect generalizer
    # This red-team models routers as answer-from-weights (skill = which answers the key covers),
    # distinguished from a memorizer only by held-out generalization -> the re-execution PROBE path.
    # The default "grounding" mode (which requires the answer to come from the pool) is demonstrated
    # by the decoupled `run_local` demo instead. See docs/DESIGN.md.
    val = KOTHValidator(approved, platform.public_hex, chain, KingChain(), suite, store, backend,
                        n_per_bench=N_PER_BENCH, budget=0.5, f_min=0.1, audit_mode="probe")

    k = {n: Signer() for n in ["honest-strong", "honest-weak", "memorizer", "hardcoder",
                               "substituter", "lying-rt", "fake-quote", "replayer", "copier"]}
    mk = lambda sig, art: KOTHMiner(sig, backend, platform, suite, art)  # noqa: E731
    benign = _art(_ROUTER_SRC, {})
    miners = {
        "honest-strong": mk(k["honest-strong"], strong_art),
        "honest-weak": mk(k["honest-weak"], _art(_ROUTER_SRC, weak)),
        "memorizer": mk(k["memorizer"], _art(_ROUTER_SRC, pool)),          # held-out unseen
        "hardcoder": mk(k["hardcoder"], _art(_HARDCODE_SRC, pool)),
        "substituter": SubstituterMiner(k["substituter"], backend, platform, suite,
                                        published=_art(_ROUTER_SRC, weak), secret=_art(_ROUTER_SRC, pool)),
        "lying-rt": LyingRuntimeMiner(k["lying-rt"], backend, platform, suite, benign),
        "fake-quote": FakeQuoteMiner(k["fake-quote"], backend, platform, suite, benign),
        "replayer": ReplayMiner(k["replayer"], backend, platform, suite, benign),
        "copier": mk(k["copier"], strong_art),                            # identical to honest-strong
    }
    name = {m.hotkey: n for n, m in miners.items()}
    for m in miners.values():
        val.register(m.hotkey)

    # honest + cheaters commit early (block 0); the copier commits later, so the
    # original wins the earliest-commit dedup tiebreak.
    for n, m in miners.items():
        if n != "copier":
            m.publish(store, chain, f"{n}/repo")
    chain.advance(10)
    miners["copier"].publish(store, chain, "copier/repo")

    by_hk = {m.hotkey: m for m in miners.values()}
    def get_proof(hk, epoch, nonce, repo):        # sim adapter: run the miner directly
        m = by_hk.get(hk)
        return m.produce(epoch, nonce) if m else None

    emissions = defaultdict(float)
    dq_final: dict[str, str] = {}
    for _e in range(epochs):
        chain.advance(EPOCH_BLOCKS)                # advance to the next scoring epoch
        rep = val.run_epoch(get_proof)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name.get(uid_hk.get(uid, ""), "burn")] += w
        dq_final.update({name.get(hk, hk): r for hk, r in rep.dq.items()})

    if verbose:
        total = sum(v for kk, v in emissions.items() if kk != "burn") or 1.0
        print("cumulative emissions:")
        for who, v in sorted(emissions.items(), key=lambda kv: -kv[1]):
            tag = f"  [DQ: {dq_final[who]}]" if who in dq_final else ("  [burn]" if who == "burn" else "")
            share = "  -- " if who == "burn" else f"{v/total*100:5.1f}%"
            print(f"  {who:14s} {v:6.3f}  {share}{tag}")
        print("\ndisqualifications:", dq_final)
    return {"emissions": dict(emissions), "disqualifications": dq_final}


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
