#!/usr/bin/env python
"""TinyRouter, fully: the multi-turn Thinker/Worker/Verifier orchestrator as the
subnet's competition, over an extended model pool.

Compares, on held-out tasks:
  * best single model              (the frontier baseline)
  * 1-turn routing policy          (pick-one — our v1; max_turns=1)
  * full multi-turn orchestrator   (TinyRouter's actual mechanism; max_turns=5)
  * per-sample worker oracle        (noise-inflated 1-turn upper bound)

The point: routing ~ best single (no accuracy moat), but the multi-turn
orchestrator EXCEEDS the best single model by composing Thinker->Worker->Verifier.
"""
from __future__ import annotations

import numpy as np

from thirtyspokes.orchestrator import THINKER, VERIFIER, WORKER, run_episodes, train_orchestrator
from thirtyspokes.synth_tinyrouter import SyntheticBackend, TinyConfig


def main() -> None:
    cfg = TinyConfig(q=6000, seed=2)
    be = SyntheticBackend(cfg)
    rng = np.random.default_rng(0)
    perm = rng.permutation(be.Q)
    train_mask = np.zeros(be.Q, bool); train_mask[perm[: be.Q * 65 // 100]] = True
    eval_mask = ~train_mask
    ev = np.flatnonzero(eval_mask)

    print("=" * 74)
    print(f"extended pool ({be.K} models): {be.names}")
    print(f"  solo worker accuracy: "
          + ", ".join(f"{n.split('-')[0]}={a:.2f}" for n, a in zip(be.names, be.single_model_accuracy())))
    solo = be.single_model_accuracy()
    best_single = float(solo.max())
    best_name = be.names[int(solo.argmax())]
    cost_ref = float(cfg.max_turns * be.price.max())

    # per-sample 1-turn worker oracle (upper bound for pick-one)
    correct0 = np.stack([be.u_work[:, 0] < be.worker_prob(np.full(be.Q, m), np.zeros(be.Q))
                         for m in range(be.K)], axis=1)
    worker_oracle = correct0[ev].any(axis=1).mean()

    LAM = 0.15
    print(f"\nbest single worker: {best_name}  acc={best_single:.4f}  "
          f"(cost/q=${be.price[int(solo.argmax())]:.1f})")
    print(f"per-sample worker oracle (1-turn upper bound): {worker_oracle:.4f}")

    print("\ntraining policies (CMA-ES)...")
    routing = train_orchestrator(be, lam=LAM, max_turns=1, iters=100, train_mask=train_mask, seed=0)
    orch = train_orchestrator(be, lam=LAM, max_turns=cfg.max_turns, iters=160, train_mask=train_mask, seed=0)

    def report(name, tr):
        r = run_episodes(tr.head, tr.theta, be, tr.max_turns)
        acc = r.correct[ev].mean()
        cost = r.cost[ev].mean()
        turns = r.turns_used[ev].mean()
        tot = sum(r.role_counts.values()) or 1
        roles = {"T": r.role_counts[THINKER], "W": r.role_counts[WORKER], "V": r.role_counts[VERIFIER]}
        rolestr = " ".join(f"{k}={v/tot*100:.0f}%" for k, v in roles.items())
        print(f"  {name:26s} acc={acc:.4f}  cost/q=${cost:5.1f}  turns={turns:.2f}  roles[{rolestr}]")
        return acc

    print("\n[held-out comparison]")
    print(f"  {'best single model':26s} acc={best_single:.4f}")
    r_acc = report("1-turn routing (v1)", routing)
    o_acc = report("multi-turn orchestrator", orch)

    print(f"\n  routing vs best-single      : {(r_acc-best_single)*100:+.2f} pts  (pick-one has ~no moat)")
    print(f"  orchestrator vs best-single : {(o_acc-best_single)*100:+.2f} pts  (composition exceeds the frontier)")
    print(f"  orchestrator vs routing     : {(o_acc-r_acc)*100:+.2f} pts  (the multi-turn loop's value)")


if __name__ == "__main__":
    main()
