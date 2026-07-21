"""End-to-end local subnet simulation (no chain, no API).

Runs the ENTIRE loop through the real plumbing — MockChain + LocalArtifactStore +
the real Validator (ingest -> verify salted hash -> validate -> sanity -> dedup ->
execute engine -> score -> reign -> set weights) — against a cast of miners
including adversaries, proving the subnet works before touching a chain.

  honest_early   strong policy, submitted early           -> earns
  improver       re-commits improving policies over epochs -> climbs
  camper         mediocre, never improves                  -> mid/low
  degenerate     constant policy                           -> sanity-gate DQ
  copier         republishes a strong policy's weights     -> fingerprint DQ
"""

from __future__ import annotations

import tempfile
from collections import defaultdict

import numpy as np

from ..synth_tinyrouter import TinyConfig
from . import miner
from .artifact import PolicyArtifact
from .chain import MockChain
from .policy import train_baseline
from .pool import SanctionedPool
from .protocol import CompetitionParams
from .store import LocalArtifactStore


def _degenerate_artifact(pool) -> PolicyArtifact:
    """All-zero policy -> uniform logits -> constant argmax -> sanity-gate reject."""
    hidden = 8
    art = PolicyArtifact(pool.embed_dim, pool.k_models, hidden,
                         np.zeros(_nparams(pool, hidden)), "degenerate")
    return art


def _nparams(pool, hidden):
    from ..orchestrator import make_policy_head
    return make_policy_head(pool.embed_dim, pool.k_models, hidden).n_params


def run_simulation(epochs: int = 8, verbose: bool = True) -> dict:
    params = CompetitionParams(pool=TinyConfig(q=1500, seed=7), eval_questions=1500)
    pool = SanctionedPool(params.pool)
    chain = MockChain()
    tmp = tempfile.mkdtemp(prefix="orchestra_")
    store = LocalArtifactStore(tmp)

    chain.register("burn")              # reserve uid 0 as the burn account
    from .validator import Validator
    val = Validator(pool, params, chain, store)

    published: dict[str, PolicyArtifact] = {}

    def reg_publish(hk, repo, hidden, iters, seed):
        chain.register(hk)
        art = train_baseline(pool, params, hidden=hidden, iters=iters, seed=seed, miner_id=hk)
        miner.publish(art, chain, store, hk, repo)
        published[hk] = art
        return art

    if verbose:
        print(f"[sim] pool={pool.names}  (tmp store: {tmp})")

    emissions: dict[str, float] = defaultdict(float)
    dq_final: dict[str, str] = {}

    for epoch in range(epochs):
        # --- miner actions this epoch ---
        if epoch == 0:
            reg_publish("honest_early", "he/v1", hidden=32, iters=120, seed=20)
            reg_publish("camper", "cm/v1", hidden=16, iters=40, seed=21)
        if epoch == 1:
            reg_publish("improver", "im/v1", hidden=8, iters=25, seed=30)
        if epoch == 3:
            reg_publish("improver", "im/v2", hidden=24, iters=80, seed=31)   # re-commit (new artifact/block)
        if epoch == 5:
            reg_publish("improver", "im/v3", hidden=32, iters=150, seed=32)
        if epoch == 2:                          # degenerate -> sanity DQ
            chain.register("degenerate")
            miner.publish(_degenerate_artifact(pool), chain, store, "degenerate", "dg/v1")
        if epoch == 4:                          # copier steals honest_early's weights
            chain.register("copier")
            src = published["honest_early"]
            copy = PolicyArtifact(src.embed_dim, src.k_models, src.hidden,
                                  src.theta + np.random.default_rng(1).normal(0, 1e-4, src.theta.shape),
                                  "copier")
            miner.publish(copy, chain, store, "copier", "cp/v1")

        chain.advance(100)
        rep = val.run_epoch(epoch_seed=2000 + epoch)

        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[uid_hk.get(uid, f"uid{uid}")] += w
        dq_final.update(rep.disqualifications)
        if verbose:
            print(f"[epoch {epoch}] scored {rep.n_scored}/{rep.n_commits}  "
                  f"champion={rep.champion or '-':13s} slots={rep.slots}")

    total = sum(v for k, v in emissions.items() if k != "burn") or 1.0
    if verbose:
        print("\ncumulative emissions:")
        for hk, v in sorted(emissions.items(), key=lambda kv: -kv[1]):
            tag = f"  [DQ: {dq_final[hk]}]" if hk in dq_final else ("  [burn]" if hk == "burn" else "")
            share = "  -- " if hk == "burn" else f"{v/total*100:5.1f}%"
            print(f"  {hk:14s} {v:6.3f}  {share}{tag}")
        print("\ndisqualifications:", dq_final)

    return {"emissions": dict(emissions), "disqualifications": dq_final}


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
