#!/usr/bin/env python
"""Where, if anywhere, is routing LEARNABLE? The hole in the negative result.

Seven measurements in this investigation say a trained router captures 1-9% of a real routable band.
But every one of them ran on traffic whose routable signal is DIFFICULTY or LUCK -- whether a given
model happens to get a given question right. That is close to unpredictable from a prompt, which is
exactly what the encoder-ceiling work found.

A different kind of signal was never tested. Language, domain and format are near-perfectly readable
from an embedding: "is this question in Chinese?" is trivial where "will gpt-4 get this right?" is
not. If a pool contains genuine SPECIALISTS, routing on that signal is a fundamentally easier problem
and the negative result would not apply.

So this measures, per benchmark, the number that actually matters -- not the oracle gap (measured
plenty, always large) but the CAPTURED FRACTION: what a trained capped head reaches on held-out data,
as a share of the band between the best single model and the per-ask oracle.

  captured ~ 0    routing is not learnable here (the signal is luck)
  captured high   routing IS learnable here -- and that benchmark's property is what the suite needs

This is the experiment that decides whether the subnet thesis is dead or merely aimed at the wrong
traffic. Offline, free.
"""
from __future__ import annotations

import argparse

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router


def main() -> None:
    p = argparse.ArgumentParser(description="per-benchmark routing learnability")
    p.add_argument("--min-rows", type=int, default=380)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--pool", choices=["curated", "all"], default="all")
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    models = rd.ALL_MODELS if args.pool == "all" else rd.CURATED_POOL
    print(f"pool={args.pool} K={len(models)} lam={args.lam} hidden={args.hidden}\n", flush=True)
    print(f"{'benchmark':32s} {'n':>6s} {'best1':>8s} {'oracle':>8s} {'band':>7s} "
          f"{'router':>8s} {'captured':>9s} {'spread':>8s}", flush=True)

    counts = df["eval_name"].value_counts()
    names = [n for n in counts.index if counts[n] >= args.min_rows]
    rows = []
    for name in names:
        sel = (df["eval_name"] == name).to_numpy()
        sub, e = df[sel].reset_index(drop=True), emb[sel]
        cache = rd.to_cache(sub, models, e)
        tr, te = cache.split(0.3, seed=0)
        if te.Q < 60 or tr.Q < 150:
            continue
        best1 = max(score_choice(te, constant_choice(te, k), args.lam).score for k in range(te.K))
        orc = score_choice(te, per_sample_oracle_choice(te, args.lam), args.lam).score
        band = orc - best1
        got = [train_router(tr, lam=args.lam, hidden=args.hidden, iters=args.iters,
                            seed=s).evaluate(te).score for s in range(args.seeds)]
        got = np.array(got)
        cap = (got.max() - best1) / band if band > 1e-9 else 0.0
        rows.append((name, te.Q, cap, got.max() - got.min()))
        print(f"{name[:32]:32s} {cache.Q:6d} {best1:+8.4f} {orc:+8.4f} {band:7.4f} "
              f"{got.max():+8.4f} {100*cap:8.1f}% {got.max()-got.min():8.5f}", flush=True)

    if not rows:
        print("\nno benchmark had enough rows")
        return
    rows.sort(key=lambda r: -r[2])
    print("\n" + "=" * 78)
    print("RANKED by captured fraction — where routing is actually learnable:")
    for name, n, cap, spread in rows[:8]:
        print(f"  {name[:34]:34s} captured {100*cap:6.1f}%   spread {spread:.5f}   n={n}")
    caps = np.array([r[2] for r in rows])
    print(f"\nmedian captured = {100*np.median(caps):.1f}%   max = {100*caps.max():.1f}%")
    print("If the max is still small, the signal is luck everywhere and the thesis is done.")
    print("If some benchmark captures a large share, ITS property is what the suite must be built on.")


if __name__ == "__main__":
    main()
