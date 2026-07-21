#!/usr/bin/env python
"""The decisive Phase-1 experiment: oracle-gap gate + learned-router cost-quality
frontier on a REAL pool.

Runs entirely offline from RouterBench (real questions, real per-model correctness,
real dollar cost) + real MiniLM embeddings. Answers, on real data:
  1. How much of the per-sample-oracle headroom is actually ROUTABLE (accuracy)?
  2. Can a learned <=1M-param router trace a cost-quality frontier below GPT-4 --
     i.e. GPT-4-class accuracy at a fraction of the cost?
"""
from __future__ import annotations

import numpy as np

from thirtyspokes import oracle, realdata as rd
from thirtyspokes.train import train_router


def main() -> None:
    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    domains, dom_names = rd._factorize(df["eval_name"].to_numpy())
    cache = rd.to_cache(df, rd.CURATED_POOL, emb)
    dom_cache = rd.to_cache(df, rd.CURATED_POOL,
                            rd.domain_onehot_embeddings(df, domains, len(dom_names)))
    models = [m.name for m in cache.models]
    gpt4 = models.index("gpt-4-1106-preview")
    Q = cache.Q

    print("=" * 78)
    print("REAL POOL:", models)
    print(f"questions={Q}  domains={len(dom_names)}  encoder=MiniLM({emb.shape[1]}d)")
    print("=" * 78)

    # 1. oracle-gap gate: does a finer (semantic) encoder unlock accuracy headroom
    #    the coarse domain encoder misses?
    print("\n[1] ORACLE-GAP GATE  (lambda=0.3)")
    for label, c in [("domain-only encoder", dom_cache), ("semantic MiniLM encoder", cache)]:
        rep = oracle.compute(c, lam=0.3, n_groups=96)
        print(f"  {label:26s}: per-sample={rep.ps_oracle_acc:.4f}  "
              f"routable={rep.routable_acc:.4f}  gap={rep.routable_accuracy_gap_points:+.2f}pts")

    # 2. cost-quality frontier: train a router at several lambdas on held-out eval.
    print("\n[2] LEARNED-ROUTER COST-QUALITY FRONTIER (held-out eval, vs always-GPT-4)")
    train, ev = cache.split(eval_frac=0.35, seed=0)
    q = np.arange(ev.Q)
    gpt4_acc = ev.correct[:, gpt4].mean()
    gpt4_cost = ev.cost[:, gpt4].sum()
    any_correct = ev.correct.any(axis=1).mean()
    print(f"  reference: always-GPT-4 acc={gpt4_acc:.4f} cost=${gpt4_cost:,.2f} | "
          f"per-sample-oracle acc={any_correct:.4f}")
    print(f"  {'lambda':>7} | {'acc':>6} | {'cost$':>8} | {'%GPT4$':>7} | vs GPT-4")
    print("  " + "-" * 60)
    for lam in [0.0, 0.05, 0.15, 0.4, 1.0, 2.5]:
        tr = train_router(train, lam=lam, hidden=32, iters=240, seed=0)
        ch = tr.choose(ev.embeddings)
        acc = ev.correct[q, ch].mean()
        cost = ev.cost[q, ch].sum()
        dacc = (acc - gpt4_acc) * 100
        dcost = (1 - cost / gpt4_cost) * 100
        print(f"  {lam:7.2f} | {acc:.4f} | {cost:8.2f} | {cost/gpt4_cost*100:6.1f}% | "
              f"{dacc:+.2f}pts acc, {dcost:+.1f}% cost")

    print("\n[3] READING: a point at/above GPT-4 accuracy with cost% < 100 is")
    print("    'GPT-4-class quality cheaper'; the router picks the lambda (its own")
    print("    operating point) per its training objective.")


if __name__ == "__main__":
    main()
