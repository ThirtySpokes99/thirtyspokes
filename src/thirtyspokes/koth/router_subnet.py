"""ROUTER-ARCHITECTURE subnet, end-to-end and offline (`orchestra-koth-router-sim`).

The companion to `koth/subnet.py`, which simulates the free-agent architecture miners are moving off.
Here a miner ships ONLY a routing head: `weights.npz`, no source, no code. The owner's harness inside
the measured image samples the slice, embeds the prompts with the pinned encoder, runs the head to
pick a ladder entry point, executes the cascade, and attests both the outcome and the DECISION. The
validator downloads the weights, binds them to the on-chain commit, verifies the quote, and scores
the decision against the owner's reference — running no inference and no miner code.

WHAT THIS SIM IS FOR. `subnet.py`'s cast tests defences against miners who author answers. None of
those adversaries can exist here (a head cannot emit an answer), so that sim proves nothing about
this architecture. The three adversaries below are the ones this design actually has to survive:

  honest-a / honest-b  independently trained on a DISJOINT split, so they must generalise to the
                       scored asks. The baseline both cheats are measured against.
  memoriser            trains on the scored bank itself — no generalisation, pure enumeration of
                       `task -> best rung`. The param cap was supposed to stop this. It does not:
                       `scripts/memoriser_capacity.py` measures a 6.4K-param head fitting a RANDOM
                       table for 1,000 tasks outright, and this sim shows what that buys in
                       emissions. Sweep `--bank` to watch the edge decay (edge <15% needs ~20k asks).
  copier               clones the leader's head and jitters the weights, so the artifact hash differs
                       and hash-dedup misses. Must be caught on the SOFT distribution — argmax dedup
                       cannot do it, since honest routers already agree 0.954 of the time.
  forger               reports a flawless slice it never ran. Must be rejected by the quote; this is
                       the adversary that justifies keeping the TEE after miner code went away.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from ..gateway.gateway import MockBackend
from ..gateway.signing import Signer
from ..reign import KingChain
from ..subnet.chain import MockChain
from ..tee.attestation import Platform
from . import commit as commitmod
from . import harness as H
from .benchmarks import BenchTask, RealBenchmark, bench_seed, grade_choice
from .epoch import EPOCH_BLOCKS
from .proof import Proof
from .runtime import SUITE_VERSION, Artifact, KOTHRuntime, runtime_measurement
from .store import LocalBundleStore, hash_source
from .validator import KOTHValidator

POOL = ["tiny", "small", "mid", "large"]      # MockBackend prices, cheap -> expensive
N_PER_BENCH = 8
HIDDEN = 8
LETTERS = "ABCD"


class RouterWorld:
    """A task bank where routing is genuinely possible but only partly predictable.

    Each task gets a TIER (how strong a model it needs) that is visible in the embedding, plus
    per-(task, model) LUCK that is not. That split is the measured shape of real routing — routers
    capture only 1-12% of the achievable band — and it is what makes the sim discriminating: with a
    fully predictable world an honest router would be an oracle and memorisation would buy nothing,
    which would quietly turn every adversary test into a pass.
    """

    def __init__(self, bank: int, seed: int = 0, luck: float = 0.25):
        rng = np.random.default_rng(seed)
        self.n, self.rng = bank, rng
        self.tier = rng.integers(0, len(POOL), bank)          # learnable from the prompt
        self.gold = [LETTERS[i] for i in rng.integers(0, 4, bank)]
        strength = np.arange(len(POOL))[None, :]
        self.correct = (strength >= self.tier[:, None]).astype(float)
        flip = rng.random((bank, len(POOL))) < luck            # NOT visible in the embedding
        self.correct = np.where(flip, 1.0 - self.correct, self.correct)
        # a weak model on a hard ask often emits something the verifier cannot even parse — the
        # signal the cascade escalates on, and the reason the start-rung choice is non-trivial
        self.parses = rng.random((bank, len(POOL))) > np.clip(
            0.55 * (self.tier[:, None] - strength) / len(POOL), 0, 0.6)
        self.parses[:, -1] = True                              # top rung always returns something
        self.tid = [f"t{i:06d}" for i in range(bank)]
        self.by_tid = {t: i for i, t in enumerate(self.tid)}

    def prompt(self, i: int) -> str:
        # the tier is stated in the prompt: the ONLY routable signal, exactly as a real router must
        # read difficulty out of the text rather than out of an oracle
        return f"[{self.tid[i]}] tier-{self.tier[i]} question. Choose A, B, C or D."

    def benchmark(self, held_from: int) -> RealBenchmark:
        tasks = [BenchTask(self.tid[i], self.prompt(i), self.gold[i]) for i in range(held_from)]
        held = [BenchTask(self.tid[i], self.prompt(i), self.gold[i])
                for i in range(held_from, self.n)]
        return RealBenchmark("routing-mc", 1.0, tasks, held or tasks[:1], grade_choice,
                             answer_kind="choice")

    def embed(self, prompts: list[str]) -> np.ndarray:
        """Stand-in for the pinned encoder: the tier is recoverable, the luck is not."""
        out = np.zeros((len(prompts), H.EMBED_DIM))
        for j, p in enumerate(prompts):
            i = self.by_tid[p.split("]")[0][1:]]
            r = np.random.default_rng(10_000 + i)
            v = r.normal(0, 1, H.EMBED_DIM)
            v[: H.EMBED_DIM // 4] += 3.0 * self.tier[i]        # the learnable direction
            out[j] = v / np.linalg.norm(v)
        return out

    def reference(self, tids: list[str]) -> dict:
        """The owner's reference for these asks — every (ask, pool-model) cell. Real subnets pay
        inference for this; here it is the world matrix, which is exactly what a reference measures."""
        idx = [self.by_tid[t] for t in tids]
        price = MockBackend()._price
        cost = np.array([[price(m) * 0.02 for m in POOL] for _ in idx])
        return {"suite_version": SUITE_VERSION, "models": list(POOL), "task_ids": list(tids),
                "scores": self.correct[idx].tolist(), "costs": cost.tolist(),
                "verifier_ok": self.parses[idx].tolist()}


class WorldBackend(MockBackend):
    """Answers according to the world: right, wrong, or unparseable."""

    def __init__(self, world: RouterWorld):
        self.world = world

    def complete(self, model, messages, params):
        w = self.world
        prompt = str(messages[-1]["content"])
        i = w.by_tid[prompt.split("]")[0][1:]]
        m = POOL.index(model)
        if not w.parses[i, m]:
            text = "..."                                        # verifier rejects -> escalate
        elif w.correct[i, m]:
            text = f"Answer: {w.gold[i]}"
        else:
            text = f"Answer: {LETTERS[(LETTERS.index(w.gold[i]) + 1) % 4]}"
        tin, tout = 32, 8
        return text, tin, tout, (tin + tout) / 1000.0 * self._price(model)


def _oracle_rungs(world: RouterWorld, idx: list[int], lam: float = 0.5) -> np.ndarray:
    """The best ladder entry point per ask — the label both an honest router and a memoriser want.
    Derived through the same cascade semantics the harness executes, so the sim trains against the
    thing it is scored on rather than a proxy."""
    from .verify import cascade_outcomes
    ref = world.reference([world.tid[i] for i in idx])
    correct, cost = cascade_outcomes(ref, list(range(len(POOL))))
    return (correct - lam * cost / (cost.max() or 1.0)).argmax(axis=1)


def _fit(X: np.ndarray, y: np.ndarray, seed: int, iters: int = 400) -> np.ndarray:
    """Fit the harness's OWN head architecture and pack it into the flat vector `load_head` parses.

    `RouterHead` is [W1, b1, W2, b2] with tanh — the exact shape of a one-hidden-layer MLP — so a
    gradient fit can be transplanted straight into the miner's artifact. Using a real optimizer
    matters: an adversary hand-built by fiat would prove nothing about what is actually reachable.
    """
    from sklearn.neural_network import MLPClassifier
    import warnings
    clf = MLPClassifier(hidden_layer_sizes=(HIDDEN,), activation="tanh", solver="adam",
                        max_iter=iters, random_state=seed, tol=1e-9, n_iter_no_change=iters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    w1, w2 = clf.coefs_
    b1, b2 = clf.intercepts_
    # sklearn drops absent classes; re-expand to the full action space or the head is the wrong shape
    W2 = np.zeros((HIDDEN, len(POOL)))
    B2 = np.full(len(POOL), -10.0)
    for j, c in enumerate(clf.classes_):
        W2[:, c], B2[c] = w2[:, j], b2[j]
    return np.concatenate([w1.ravel(), b1.ravel(), W2.ravel(), B2.ravel()])


class RouterMiner:
    def __init__(self, signer, backend, platform, suite, theta):
        self.signer, self.hotkey = signer, signer.public_hex
        self.rt = KOTHRuntime(backend, platform)
        self.suite = suite
        self.weights = H.save_head(theta, HIDDEN)
        self.artifact = Artifact(H.HARNESS_VERSION, self.weights, "router")

    def produce(self, epoch, nonce):
        return self.rt.run_router(self.weights, hotkey=self.hotkey, epoch=epoch, nonce=nonce,
                                  suite=self.suite, n_per_bench=N_PER_BENCH, pool=POOL,
                                  price_of=MockBackend()._price, params={"max_tokens": 8})

    def publish(self, store, chain, repo):
        rev = store.upload(repo, self.artifact)
        chain.commit(self.hotkey, commitmod.commit_string(
            self.hotkey, repo, rev, hash_source(H.HARNESS_VERSION),
            self.artifact.weights_hash))


class ForgerMiner(RouterMiner):
    """Never runs the harness; reports a flawless slice signed with its own key."""

    def produce(self, epoch, nonce):
        p = Proof(epoch, nonce, self.hotkey, hash_source(H.HARNESS_VERSION),
                  self.artifact.weights_hash, "router", (), 0.0, 0, "x",
                  measurement=runtime_measurement())
        return p.attested_by(Platform(self.signer)), []


def run_simulation(epochs: int = 4, bank: int = 400, dedup_l1: float = 0.05,
                   verbose: bool = True) -> dict:
    world = RouterWorld(bank)
    scored_n = int(bank * 0.6)                     # the public, scoreable bank
    suite = [world.benchmark(scored_n)]
    backend = WorldBackend(world)
    platform = Platform()
    chain = MockChain(); chain.register("burn")
    store = LocalBundleStore(f"/tmp/koth_router_sim_{id(chain)}")

    # The pinned encoder is replaced by the world's deterministic embedding: the sim must run
    # offline in CI, and MiniLM would add a model download without changing what is being tested.
    # RESTORED in the finally below — this used to be a bare assignment, which left every later
    # caller in the process holding a stand-in encoder that KeyErrors on any prompt the sim did not
    # generate. Harmless-looking, and it silently broke an unrelated test.
    _real_encode = H.encode
    H.encode = world.embed
    try:
        return _run(world, suite, backend, platform, chain, store, scored_n, bank,
                    epochs, dedup_l1, verbose)
    finally:
        H.encode = _real_encode


def _run(world, suite, backend, platform, chain, store, scored_n, bank,
         epochs, dedup_l1, verbose) -> dict:

    scored_idx = list(range(scored_n))
    train_idx = list(range(scored_n, bank))        # disjoint: honest routers must generalise
    Xs, Xt = world.embed([world.prompt(i) for i in scored_idx]), \
        world.embed([world.prompt(i) for i in train_idx])
    ys, yt = _oracle_rungs(world, scored_idx), _oracle_rungs(world, train_idx)

    honest_a = _fit(Xt, yt, seed=1)
    honest_b = _fit(Xt, yt, seed=7)
    memoriser = _fit(Xs, ys, seed=3)               # fitted to the asks it will be SCORED on
    copier = honest_a + np.random.default_rng(0).normal(0, 0.01, honest_a.size)

    names = ["honest-a", "honest-b", "memoriser", "copier", "forger"]
    thetas = dict(zip(names, [honest_a, honest_b, memoriser, copier, honest_a]))
    signers = {n: Signer() for n in names}
    miners = {}
    for n in names:
        cls = ForgerMiner if n == "forger" else RouterMiner
        miners[n] = cls(signers[n], backend, platform, suite, thetas[n])

    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(), suite,
                        store, backend, n_per_bench=N_PER_BENCH, budget=0.5, f_min=0.0,
                        audit_mode="grounding", scoring_mode="accumulate",
                        # explicit: the validator DEFAULTS this off, and the sweep below
                        # shows why — at a large bank it disqualifies honest routers
                        dedup_max_l1=dedup_l1,
                        # the sim's own mock ladder, not the production pool: the validator must
                        # score decisions against the action space the heads actually emitted into
                        routing_pool=list(POOL),
                        # the owner's reference for exactly this epoch's slice — the same asks the
                        # harness will draw, so the decision is scored against every alternative
                        pool_reference=lambda e, nc: world.reference(
                            [t.task_id for b in suite
                             for t in b.sample(N_PER_BENCH, bench_seed(nc, e, b.name))]))
    for m in miners.values():
        val.register(m.hotkey)

    for n, m in miners.items():                    # the copier commits later -> loses the tiebreak
        if n != "copier":
            m.publish(store, chain, f"{n}/repo")
    chain.advance(10)
    miners["copier"].publish(store, chain, "copier/repo")

    by_hk = {m.hotkey: m for m in miners.values()}
    name = {m.hotkey: n for n, m in miners.items()}

    def get_proof(hk, epoch, nonce, repo):
        m = by_hk.get(hk)
        return m.produce(epoch, nonce) if m else None

    emissions: dict[str, float] = defaultdict(float)
    dq_final: dict[str, str] = {}
    decisions: dict[str, list] = defaultdict(list)
    for _e in range(epochs):
        chain.advance(EPOCH_BLOCKS)
        rep = val.run_epoch(get_proof)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name.get(uid_hk.get(uid, ""), "burn")] += w
        dq_final.update({name.get(hk, hk): r for hk, r in rep.dq.items()})
        for hk, d in (rep.audit.get("decision") or {}).items():
            decisions[name.get(hk, hk)].append(d.get("decision_quality", 0.0))

    if verbose:
        total = sum(v for k, v in emissions.items() if k != "burn") or 1.0
        print(f"router sim — bank={bank} scored={scored_n} pool={len(POOL)} epochs={epochs}\n")
        print(f"{'miner':12s} {'emissions':>10s} {'share':>7s} {'decision-q':>11s}  status")
        for who in names + ["burn"]:
            v = emissions.get(who, 0.0)
            dq = decisions.get(who) or []
            share = "   -- " if who == "burn" else f"{v/total*100:5.1f}%"
            tag = f"DQ: {dq_final[who]}" if who in dq_final else ""
            print(f"{who:12s} {v:10.3f} {share} "
                  f"{(np.mean(dq) if dq else float('nan')):11.3f}  {tag}")
        mem = emissions.get("memoriser", 0.0)
        hon = max(emissions.get("honest-a", 0.0), emissions.get("honest-b", 0.0))
        print()
        fp = [n for n in ("honest-a", "honest-b") if n in dq_final]
        if fp:
            print(f"FALSE POSITIVE: {', '.join(fp)} disqualified as {dq_final[fp[0]]} — an honest "
                  f"router convergent with another honest router. Copy-dedup is not safe at this "
                  f"bank size; the earliest-commit tiebreak is what should handle copying.")
        if "memoriser" in dq_final:
            print("memoriser was CAUGHT")
        elif mem > hon:
            print(f"MEMORISER WINS ({mem/max(hon,1e-9):.2f}x the honest leader) — no defence fired. "
                  f"The param cap does not bind at bank={bank}; see scripts/memoriser_capacity.py.")
        else:
            print(f"memoriser did not lead at bank={bank} (generalisation competitive)")
    return {"emissions": dict(emissions), "disqualifications": dq_final,
            "decision_quality": {k: float(np.mean(v)) for k, v in decisions.items() if v}}


def main() -> None:
    ap = argparse.ArgumentParser(description="router-architecture subnet simulation")
    ap.add_argument("--bank", type=int, default=400, help="task bank size (sweep to see the "
                                                          "memoriser's edge decay)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dedup-l1", type=float, default=0.05,
                    help="router copy-dedup threshold (validator default is 0 = off)")
    args = ap.parse_args()
    run_simulation(epochs=args.epochs, bank=args.bank, dedup_l1=args.dedup_l1)


if __name__ == "__main__":
    main()
