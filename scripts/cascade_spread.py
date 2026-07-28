#!/usr/bin/env python
"""Phase-0 gate, part 2: does the CASCADE action space have competition in it?

`scripts/head_spread.py` showed the pick-one engine has none on this traffic: 8 independently-trained
capped heads span 0.005 in held-out score, all within noise of just-call-gpt-4, capturing 1.1% of a
0.121 routable band. The reason is structural -- a pick-one router must PREDICT which model succeeds
from the prompt alone, and five separate measurements in this project now say that is close to
impossible on broad traffic.

The cascade action space asks a different question, using a different kind of information. The router
picks WHERE TO ENTER a cheap->expensive ladder; the harness then invokes, runs a pinned verifier on
the ANSWER, and escalates on reject. That is post-hoc evidence, not prediction -- so it is not bound
by the prompt-predictability ceiling. `cascade.to_cascade_cache` already reduces the whole thing to a
derived (Q,K) cache, so oracle/scorer/trainer/reign work on it unchanged.

The open variable is VERIFIER QUALITY, which nothing has measured. So sweep it: a verifier that
accepts a correct answer with prob `tpr` and wrongly accepts an incorrect one with prob `fpr`.
  perfect (1.0, 0.0)  -> cheap-first is free; the ladder is pure upside
  useless (1.0, 1.0)  -> always accepts rung r == pick-one, recovering the earlier result
Between them lies whatever the real design can hope for.

Reports, per verifier quality: the cascade band (oracle - best-single over ladder entry points), what
a trained capped head captures, and the SPREAD across seeds -- the ossification gate.

Offline, free. Needs data/routerbench_0shot.pkl + data/emb_minilm.npy.
"""
from __future__ import annotations

import argparse

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.cascade import to_cascade_cache
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router

EPS0, EPS_FLOOR = 0.02, 0.002


def with_verifier(cache, tpr: float, fpr: float, seed: int):
    """Attach a synthetic pinned verifier of known quality.

    verifier_ok[q,k] = accept model k's answer on q. Correct answers are accepted with prob `tpr`;
    incorrect ones are wrongly accepted with prob `fpr` (that is the expensive error -- it banks a
    wrong answer and stops the ladder early)."""
    rng = np.random.default_rng(seed)
    p = np.where(cache.correct, tpr, fpr)
    vok = rng.random(cache.correct.shape) < p
    return type(cache)(
        models=cache.models, embeddings=cache.embeddings, correct=cache.correct,
        cost=cache.cost, verifier_ok=vok, domain=cache.domain,
        question_ids=cache.question_ids, cost_norm=cache.cost_max,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="cascade action-space competitive spread")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=180)
    p.add_argument("--eval-frac", type=float, default=0.3)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    base = rd.to_cache(df, rd.CURATED_POOL, emb)
    print(f"pool K={base.K}  Q={base.Q}  lam={args.lam}  hidden={args.hidden}\n", flush=True)

    print(f"{'verifier (tpr/fpr)':22s} {'best-entry':>11s} {'oracle':>9s} {'band':>8s} "
          f"{'router':>9s} {'captured':>9s} {'spread':>8s}  verdict", flush=True)

    for tpr, fpr in [(1.00, 0.00), (0.95, 0.05), (0.90, 0.10), (0.80, 0.20),
                     (0.70, 0.30), (1.00, 1.00)]:
        casc = to_cascade_cache(with_verifier(base, tpr, fpr, seed=7))
        tr_c, te_c = casc.split(args.eval_frac, seed=0)

        best_single = max(score_choice(te_c, constant_choice(te_c, k), args.lam).score
                          for k in range(te_c.K))
        orc = score_choice(te_c, per_sample_oracle_choice(te_c, args.lam), args.lam).score
        band = orc - best_single

        scores = []
        for s in range(args.seeds):
            tr = train_router(tr_c, lam=args.lam, hidden=args.hidden, iters=args.iters, seed=s)
            scores.append(tr.evaluate(te_c).score)
        scores = np.array(scores)
        spread = scores.max() - scores.min()
        captured = (scores.max() - best_single) / band if band > 1e-9 else 0.0

        verdict = ("OSSIFIES" if spread < EPS_FLOOR else
                   "MARGINAL" if spread < EPS0 else "COMPETITIVE")
        print(f"{tpr:.2f}/{fpr:.2f}{'':12s} {best_single:+11.4f} {orc:+9.4f} {band:8.4f} "
              f"{scores.max():+9.4f} {100*captured:8.1f}% {spread:8.5f}  {verdict}", flush=True)

    print("\nband     = oracle over ladder entry points minus the best FIXED entry point")
    print("captured = share of that band the best trained head actually reaches")
    print("spread   = held-out score range across seeds; below 0.002 the reign cannot be dethroned")


if __name__ == "__main__":
    main()
