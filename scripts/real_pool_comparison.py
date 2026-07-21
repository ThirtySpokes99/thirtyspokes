#!/usr/bin/env python
"""Which REAL pool (if any) is worth a routing subnet? Tests the ceiling principle
on RouterBench: pool composition should decide routable headroom.

For each candidate pool: routable accuracy gap (does complementarity exist?) and
the learned-router iso-accuracy cost saving (the RouteLLM 'same quality, cheaper'
product metric), all on held-out real data.
"""
from __future__ import annotations

import numpy as np

from thirtyspokes import oracle, realdata as rd
from thirtyspokes.train import train_router

POOLS = {
    "curated-5 (mistral->gpt4)": rd.CURATED_POOL,
    "no-dominator-5": ["mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
                       "zero-one-ai/Yi-34B-Chat", "gpt-3.5-turbo-1106", "claude-instant-v1"],
    "diverse-strong-3": ["mistralai/mixtral-8x7b-chat", "claude-v2", "gpt-4-1106-preview"],
    "cheap+strong-2 (RouteLLM)": ["mistralai/mistral-7b-chat", "gpt-4-1106-preview"],
    "mid+strong-2": ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
}


def iso_accuracy_cost_saving(train, ev, best_k, tol=0.005):
    """Max cost saving vs always-best-single at accuracy >= best_single_acc - tol."""
    q = np.arange(ev.Q)
    ref_acc = ev.correct[:, best_k].mean()
    ref_cost = ev.cost[:, best_k].sum()
    best_saving, best_pt = 0.0, (ref_acc, 1.0)
    for lam in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
        tr = train_router(train, lam=lam, hidden=32, iters=200, seed=0)
        ch = tr.choose(ev.embeddings)
        acc = ev.correct[q, ch].mean()
        cost = ev.cost[q, ch].sum()
        if acc >= ref_acc - tol:
            saving = 1 - cost / ref_cost
            if saving > best_saving:
                best_saving, best_pt = saving, (acc, cost / ref_cost)
    return best_saving, best_pt, ref_acc


def main() -> None:
    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")

    print(f"{'pool':28s} | {'bestsingle':>10} | {'routable_gap':>12} | iso-acc cost saving")
    print("-" * 92)
    for name, mlist in POOLS.items():
        cache = rd.to_cache(df, mlist, emb)
        rep = oracle.compute(cache, lam=0.3, n_groups=96)
        best_k = int(cache.model_accuracy().argmax())
        best_name = cache.models[best_k].name.split("/")[-1]
        train, ev = cache.split(eval_frac=0.35, seed=0)
        # recompute best_k on the eval-split reference for the frontier
        ev_best_k = int(ev.model_accuracy().argmax())
        saving, (acc, costpct), ref_acc = iso_accuracy_cost_saving(train, ev, ev_best_k)
        print(f"{name:28s} | {rep.best_single_acc:.3f} {best_name:>4.4} | "
              f"{rep.routable_accuracy_gap_points:+6.2f} pts   | "
              f"{saving*100:5.1f}% cheaper @ acc {acc:.3f} (ref {ref_acc:.3f}, {costpct*100:.0f}% cost)")

    print("\nreading: 'routable_gap' >= 5 pts => accuracy moat; 'iso-acc cost saving'")
    print("is the honest 'same quality, X% cheaper' number a learned router delivers.")


if __name__ == "__main__":
    main()
