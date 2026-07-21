#!/usr/bin/env python
"""THE decisive Phase-0 test for the orchestration/MoE pivot (ARCHITECTURE.md v5 §2b).

Three questions, in order of importance:
  1. Composition gap: does composing the pool beat the BEST SINGLE MODEL on
     accuracy? (Routing's gap was ~0; composition should be strongly positive.)
  2. Saturation curve: as the orchestration budget grows, does achievable quality
     keep RISING or PLATEAU? (Plateau => saturates like routing.)
  3. Churn: when a new specialist model enters the pool, does headroom REOPEN?
     (This is the empirical signature of a sustainable subnet.)
"""
from __future__ import annotations

import numpy as np

from thirtyspokes import oracle
from thirtyspokes.synth_orchestra import OrchestraConfig, build_workflow_cache, generate_world
from thirtyspokes.train import train_router


def route_accuracy(cache, is_route):
    accs = cache.correct.mean(axis=0)
    return float(accs[is_route].max()), cache.models[int(np.flatnonzero(is_route)[accs[is_route].argmax()])].name


def comp_ceilings(cache, seed=0, n_groups=48):
    Q = cache.Q
    ps = oracle.per_sample_oracle_choice(cache, lam=0.0)
    ps_acc = float(cache.correct[np.arange(Q), ps].mean())
    r = oracle._routable_choice(cache, 0.0, n_groups, seed)
    routable_acc = float(cache.correct[np.arange(Q), r].mean())
    return ps_acc, routable_acc


def main() -> None:
    # harder world: shifted difficulty + more domains so composition leaves real
    # residual headroom (comp-oracle well below 1.0), the honest setting to test
    # whether new capability (churn) reopens the frontier.
    cfg = OrchestraConfig(q=4000, n_domains=8, difficulty_shift=1.1,
                          specialization=0.5, seed=1,
                          pool=(("small", "cheap", -0.4, 1.0),
                                ("medium", "mid", 0.1, 6.0),
                                ("large", "strong", 0.5, 25.0)))
    world = generate_world(cfg)
    print("=" * 74)
    print(f"pool: {world.model_names}  (composition over {len(world.model_names)} models)")
    print("=" * 74)

    print("\n[1+2] COMPOSITION GAP & SATURATION CURVE (accuracy, held-out routable)")
    print(f"  {'budget':>6} | {'#workflows':>10} | {'best-single':>11} | {'comp-routable':>13} | "
          f"{'gap(pts)':>8} | {'comp-oracle':>11}")
    print("  " + "-" * 78)
    for B in range(1, 7):
        cache, is_route = build_workflow_cache(world, budget=B)
        bs_acc, bs_name = route_accuracy(cache, is_route)
        ps_acc, r_acc = comp_ceilings(cache)
        gap = (r_acc - bs_acc) * 100
        print(f"  {B:6d} | {cache.K:10d} | {bs_acc:11.4f} | {r_acc:13.4f} | "
              f"{gap:+8.2f} | {ps_acc:11.4f}")

    print("\n[train an ACTUAL orchestrator at budget=5 (policy over workflows)]")
    cache, is_route = build_workflow_cache(world, budget=5)
    tr_c, ev_c = cache.split(eval_frac=0.35, seed=1)
    bs_acc, bs_name = route_accuracy(ev_c, is_route)
    orch = train_router(tr_c, lam=0.0, hidden=32, iters=220, seed=1)
    ev = orch.evaluate(ev_c)
    print(f"  best single model (held-out): {bs_acc:.4f}  [{bs_name}]")
    print(f"  trained orchestrator        : {ev.accuracy:.4f}  ({(ev.accuracy-bs_acc)*100:+.2f} pts over best single)")

    print("\n[3] POOL CHURN: do new domain-specialists keep lifting quality? (budget=4)")
    print(f"  {'pool size':>9} | {'best-single':>11} | {'comp-routable':>13} | {'comp-oracle':>11} | {'gap(pts)':>8}")
    print("  " + "-" * 66)
    for extra in range(0, 5):
        w = generate_world(cfg, extra_specialists=extra)
        cache, is_route = build_workflow_cache(w, budget=4)
        bs_acc, _ = route_accuracy(cache, is_route)
        ps_acc, r_acc = comp_ceilings(cache)
        print(f"  {len(w.model_names):9d} | {bs_acc:11.4f} | {r_acc:13.4f} | {ps_acc:11.4f} | {(r_acc-bs_acc)*100:+8.2f}")

    print("\nreading: gap>0 => composition beats best single (routing couldn't).")
    print("  rising comp-routable with budget => open frontier; churn lifting both")
    print("  best-single AND the gap => the sustained-competition signature.")


if __name__ == "__main__":
    main()
