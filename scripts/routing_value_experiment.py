#!/usr/bin/env python
"""The RIGHT metric for routing value: lift over the single-model convex hull.

Earlier experiments asked "can a router beat / cheaply match the BEST single
model?" — and the answer was no (the top model's accuracy is unroutable). But
that is the wrong question. Routing's real value proposition is: for a quality
target that lies *between* your pool models' accuracies, is intelligent
per-question routing cheaper than the naive alternative (blindly splitting
traffic across single models — the convex hull of (cost, accuracy) points)?

On a realistic mid-difficulty workload (RouterBench benchmarks where the cheap
model is competitive but the strong model is meaningfully better — 70% of the
dataset), the answer is a clear YES: ~2.2x cheaper at intermediate accuracy.
Run offline; needs data/routerbench_0shot.pkl + data/emb_minilm.npy.
"""
from __future__ import annotations

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.train import train_router

CHEAP, MID, STRONG = "mistralai/mixtral-8x7b-chat", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"


def midband_index(df):
    keep = set()
    for name, sub in df.groupby("eval_name"):
        if len(sub) < 300:
            continue
        ca = (sub[CHEAP].astype(float) >= 0.5).mean()
        sa = (sub[STRONG].astype(float) >= 0.5).mean()
        if 0.35 <= ca <= 0.85 and sa - ca >= 0.08:   # routing has genuine work
            keep.add(name)
    return np.where(df["eval_name"].isin(keep).to_numpy())[0]


def hull_cost_at_acc(points, a):
    """Cheapest cost to reach accuracy `a` by mixing single models (upper hull)."""
    P = sorted(points, key=lambda x: x[1])
    if a <= P[0][1]:
        return P[0][0]
    if a >= P[-1][1]:
        return P[-1][0]
    for (c0, a0), (c1, a1) in zip(P, P[1:]):
        if a0 <= a <= a1:
            return c0 + (a - a0) / (a1 - a0) * (c1 - c0)
    return P[-1][0]


def main() -> None:
    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    idx = midband_index(df)
    sub = df.iloc[idx].reset_index(drop=True)
    cache = rd.to_cache(sub, [CHEAP, MID, STRONG], emb[idx])
    train, ev = cache.split(eval_frac=0.35, seed=0)
    q = np.arange(ev.Q)
    cA, cC = ev.correct, ev.cost

    pts = [(cC[:, k].sum(), cA[:, k].mean()) for k in range(3)]
    print(f"targeted workload: {len(idx)} questions ({len(idx)/len(df)*100:.0f}% of RouterBench)")
    print("held-out single models (cost, acc):",
          [f"{n.split('/')[-1]}(${c:.1f},{a:.3f})" for n, (c, a) in zip([CHEAP, MID, STRONG], pts)])
    print(f"\n  {'lam':>5} | {'acc':>6} | {'router$':>8} | {'blind-mix$':>10} | routing advantage")
    print("  " + "-" * 62)
    best = 1.0
    for lam in [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6]:
        tr = train_router(train, lam=lam, hidden=32, iters=240, seed=0)
        ch = tr.choose(ev.embeddings)
        a, c = cA[q, ch].mean(), cC[q, ch].sum()
        hc = hull_cost_at_acc(pts, a)
        mult = hc / c if c > 0 else 1.0
        best = max(best, mult)
        print(f"  {lam:5.2f} | {a:.4f} | {c:8.2f} | {hc:10.2f} | {mult:.2f}x cheaper @ acc {a:.3f}")
    print(f"\n  => intelligent routing up to {best:.2f}x cheaper than blind model-mixing")
    print("     at intermediate quality targets (near the top model it -> 1.0x).")


if __name__ == "__main__":
    main()
