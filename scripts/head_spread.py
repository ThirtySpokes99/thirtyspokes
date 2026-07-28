#!/usr/bin/env python
"""Phase-0 gate: does a CAPPED router head leave anything to compete over?

The re-architecture makes miners submit only weights for a fixed harness. That deletes a whole class
of attack -- but it also shrinks the competitive surface to "train a small head on frozen
embeddings", and `scripts/encoder_ceiling_experiment.py` already recorded that a tiny CMA-ES head
lands near the strongest supervised predictor this data allows. If every competent miner therefore
converges to the same held-out score, the reign has no gradient: nobody can ever clear the incumbent's
epsilon margin, and emissions freeze on the earliest committer forever.

So this measures the two numbers the new design actually rests on, both from independently-seeded
honest routers trained on identical data:

  SPREAD   held-out score range across seeds, compared against the reign's eps margin (0.02 -> 0.002).
           spread << eps  =>  OSSIFICATION. No miner can dethrone another; the architecture needs a
           richer action space before launch.

  AGREEMENT how often two HONEST routers pick the same action, and how close their soft
           distributions are. This calibrates the copy-dedup threshold: today's default flags >=0.95
           agreement as a copy, and over a small action space independently-trained routers may
           legitimately agree that often. Calibrating from measured honest baselines is the only way
           to avoid DQ'ing convergent honest miners.

Offline, free, no API calls. Needs data/routerbench_0shot.pkl + data/emb_minilm.npy.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router

EPS0, EPS_FLOOR = 0.02, 0.002       # reign.KingChain incumbency margin (see reign.py)


def main() -> None:
    p = argparse.ArgumentParser(description="measure competitive spread of capped router heads")
    p.add_argument("--seeds", type=int, default=8, help="independently-trained honest routers")
    p.add_argument("--hidden", type=int, default=16, help="head width (16 -> ~6.3K params at d=384)")
    p.add_argument("--lam", type=float, default=0.5, help="cost weight in the training objective")
    p.add_argument("--iters", type=int, default=220)
    p.add_argument("--pool", choices=["curated", "all"], default="curated")
    p.add_argument("--eval-frac", type=float, default=0.3)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    models = rd.CURATED_POOL if args.pool == "curated" else rd.ALL_MODELS
    emb = np.load("data/emb_minilm.npy")
    if len(emb) != len(df):
        raise SystemExit(f"embedding rows {len(emb)} != df rows {len(df)}")
    cache = rd.to_cache(df, models, emb)
    train, test = cache.split(args.eval_frac, seed=0)
    print(f"pool={args.pool} K={cache.K}  Q={cache.Q}  D={cache.D}  "
          f"train={train.Q} test={test.Q}  hidden={args.hidden}  lam={args.lam}", flush=True)

    # --- reference points on the SAME held-out slice ------------------------------------------
    best_single, best_k = None, None
    for k in range(test.K):
        s = score_choice(test, constant_choice(test, k), args.lam).score
        if best_single is None or s > best_single:
            best_single, best_k = s, k
    orc = score_choice(test, per_sample_oracle_choice(test, args.lam), args.lam).score
    print(f"best-single ({test.models[best_k].name}) = {best_single:+.4f}   oracle = {orc:+.4f}   "
          f"band = {orc - best_single:.4f}", flush=True)

    # --- N honest routers, identical data, different seeds ------------------------------------
    scores, choices, dists = [], [], []
    for seed in range(args.seeds):
        tr = train_router(train, lam=args.lam, hidden=args.hidden, iters=args.iters, seed=seed)
        held = tr.evaluate(test)
        scores.append(held.score)
        choices.append(tr.choose(test.embeddings))
        dists.append(tr.distribution(test.embeddings))
        print(f"  seed {seed}: held-out {held.score:+.4f}  acc={held.accuracy:.4f} "
              f"ncost={held.norm_cost:.4f}  params={tr.head.n_params}", flush=True)

    scores = np.array(scores)
    spread = scores.max() - scores.min()
    print()
    print("=" * 78)
    print(f"SPREAD  held-out score range = {spread:.5f}   (min {scores.min():+.4f} / "
          f"max {scores.max():+.4f} / std {scores.std():.5f})")
    print(f"        reign eps margin     = {EPS_FLOOR:.4f} .. {EPS0:.4f}")
    verdict = ("OSSIFIES — spread below the eps floor; no miner can dethrone another"
               if spread < EPS_FLOOR else
               "MARGINAL — spread within the eps band; only early-reign challenges succeed"
               if spread < EPS0 else
               "COMPETITIVE — spread exceeds eps; better routers can take the crown")
    print(f"        VERDICT: {verdict}")
    if orc - best_single > 0:
        print(f"        spread as a share of the routable band = "
              f"{100 * spread / (orc - best_single):.1f}%")

    # --- honest-vs-honest agreement (dedup calibration) ---------------------------------------
    print("=" * 78)
    agree, l1 = [], []
    for i, j in itertools.combinations(range(args.seeds), 2):
        agree.append(float((choices[i] == choices[j]).mean()))
        l1.append(float(np.abs(dists[i] - dists[j]).sum(axis=1).mean()))
    agree, l1 = np.array(agree), np.array(l1)
    print(f"AGREEMENT between HONEST routers: action-match mean={agree.mean():.3f} "
          f"max={agree.max():.3f}   soft-dist L1 mean={l1.mean():.3f}")
    print(f"        current dedup threshold = 0.95 action agreement")
    if agree.max() >= 0.95:
        print(f"        WARNING: honest routers already reach {agree.max():.3f} — the current "
              f"threshold would DQ them as copies. Dedup must use soft distributions and/or a "
              f"threshold above the honest baseline.")
    else:
        print(f"        headroom to threshold = {0.95 - agree.max():.3f} — dedup is safe here")


if __name__ == "__main__":
    main()
