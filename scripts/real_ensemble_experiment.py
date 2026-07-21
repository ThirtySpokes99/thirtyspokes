#!/usr/bin/env python
"""Partial Phase-1 gate on REAL data, no API needed — the ensemble primitive.

RouterBench gives authoritative per-model correctness on real MC benchmarks. For
majority voting, the correct models all chose the gold answer, so `n_correct`
(from labels) is exactly the size of the correct vote-cluster. Strict-majority
voting is therefore correct iff n_correct > k/2 — computable from labels alone,
no answer parsing, fully robust. (This is a lower bound on plurality voting, which
can also win with pluralities; if even the lower bound beats best-single, the
effect is real.)

Tests the pivot's core claim for the one primitive we can execute offline:
  does real majority-vote BEAT the best single real model — and on which pools?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALL_MODELS = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat",
]


def main() -> None:
    df = pd.read_pickle("data/routerbench_0shot.pkl")
    corr = np.stack([(df[m].astype(float).to_numpy() >= 0.5) for m in ALL_MODELS], axis=1)
    cost = np.stack([df[f"{m}|total_cost"].to_numpy() for m in ALL_MODELS], axis=1)
    N = len(df)
    solo = corr.mean(axis=0)
    order = np.argsort(-solo)

    print("solo accuracy (all RouterBench, real):")
    for k in order:
        print(f"  {ALL_MODELS[k].split('/')[-1]:26s} {solo[k]:.3f}  (${cost[:,k].mean():.5f}/q)")
    oracle = corr.any(axis=1).mean()
    print(f"  [per-sample oracle: {oracle:.3f}]")

    def vote(subset):
        s = list(subset)
        k = len(s)
        n_correct = corr[:, s].sum(axis=1)
        acc = float((2 * n_correct > k).mean())     # strict majority (lower bound)
        c = cost[:, s].sum()
        best = float(solo[s].max())
        return acc, best, c / cost[:, s[int(np.argmax(solo[s]))]].sum()

    print("\n[A] vote over TOP-k by accuracy (pool has a dominator: gpt-4):")
    print(f"  {'k':>2} | {'vote':>6} | {'best-in-set':>11} | {'gap':>7} | {'cost x best':>11}")
    for k in [1, 3, 5, 7, 9, 11]:
        acc, best, cmult = vote(order[:k])
        print(f"  {k:2d} | {acc:.4f} | {best:11.4f} | {(acc-best)*100:+6.2f}p | {cmult:10.2f}x")

    # [B] a COMPARABLE-strength, diverse cluster (no single dominator) — where
    # ensembling is supposed to help
    band = [i for i in range(len(ALL_MODELS)) if 0.55 <= solo[i] <= 0.72]
    band = sorted(band, key=lambda i: -solo[i])
    print(f"\n[B] vote over COMPARABLE cluster (solo acc 0.55-0.72, no dominator):")
    print("    members:", [f"{ALL_MODELS[i].split('/')[-1]}={solo[i]:.2f}" for i in band])
    for k in range(1, len(band) + 1, 2):
        acc, best, cmult = vote(band[:k] if k <= len(band) else band)
        print(f"    k={k}: vote={acc:.4f}  best-in-set={best:.4f}  gap={(acc-best)*100:+.2f}p  cost={cmult:.2f}x")

    print("\nreading: [A] with a dominant gpt-4, equal voting can't beat it (weak")
    print("  models outvote the strong one). [B] among comparable diverse models,")
    print("  majority vote SHOULD exceed the best member -- composition beating")
    print("  best-single on real data, gated on a non-dominated pool (the ceiling")
    print("  principle again). A learned orchestrator picks [B]-style votes when")
    print("  they help and routes to the strong model otherwise.")


if __name__ == "__main__":
    main()
