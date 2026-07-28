#!/usr/bin/env python
"""Does the param cap actually stop a routing-table memoriser? Measure it, don't assume it.

Under the fixed harness a miner cannot memorise ANSWERS -- the head only emits a distribution over
ladder rungs, and the harness returns the pool model's response verbatim. But it can still try to
memorise DECISIONS: precompute offline which rung is best for every task in the public pool, then fit
a head that reproduces that lookup. That router has no generalisation and would collapse on unseen
traffic, yet on a static public pool it scores like an oracle.

The only thing standing in its way is the parameter cap. The claim in `harness.py` is that a
~6.2K-param head "cannot memorise a task pool of thousands". That is a capacity argument, and capacity
arguments are exactly the kind that feel obvious and turn out wrong -- so this measures where it
actually breaks.

METHOD, deliberately generous to the attacker: give it random labels (the hardest thing to memorise,
so a pass here bounds every easier case), the exact capped architecture, and a STRONGER optimizer
than a miner would realistically use (gradient-based MLP fitting rather than CMA-ES). If the attacker
still cannot fit, the bound is real. If it can, the cap is not doing the work claimed and either the
cap must fall or the task pool must grow.

What the answer means operationally:
    fit ~= chance    the pool is large enough that the cap binds -> memorisation infeasible
    fit >> chance    a memoriser wins on this pool size -> REQUIRED POOL SIZE is larger than this
"""
from __future__ import annotations

import argparse

import numpy as np

from thirtyspokes.koth.harness import DEFAULT_HIDDEN, EMBED_DIM, PARAM_CAP
from thirtyspokes.router import RouterHead


def fit_accuracy(n_tasks: int, k: int, hidden: int, seed: int, iters: int) -> float:
    """Strongest practical attack: fit the capped architecture to random rung labels."""
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(seed)
    # unit-norm random vectors, matching what a frozen sentence encoder emits
    X = rng.normal(0, 1, (n_tasks, EMBED_DIM))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.integers(0, k, n_tasks)                       # the memoriser's lookup table
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
                        max_iter=iters, random_state=seed, tol=1e-9, n_iter_no_change=iters)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return float((clf.predict(X) == y).mean())           # TRAIN accuracy: pure memorisation


def main() -> None:
    p = argparse.ArgumentParser(description="routing-table memoriser capacity probe")
    p.add_argument("--k", type=int, default=12, help="ladder rungs (= pool size)")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--sizes", default="56,200,500,1000,2000,5000")
    args = p.parse_args()

    head = RouterHead(EMBED_DIM, args.k, args.hidden)
    chance = 1.0 / args.k
    print(f"capped head: d={EMBED_DIM} k={args.k} hidden={args.hidden} -> {head.n_params} params "
          f"(cap {PARAM_CAP})")
    print(f"chance accuracy = {chance:.3f}\n")
    print(f"{'bank size':>10s} {'fit':>7s} {'EDGE':>7s}  what the memoriser captures")

    target = None
    for n in [int(x) for x in args.sizes.split(",")]:
        acc = fit_accuracy(n, args.k, args.hidden, seed=0, iters=args.iters)
        # EDGE: how much of the way from guessing to a perfect oracle the table gets the attacker.
        # This -- not a binary "does the cap bind" -- is the quantity the mechanism cares about,
        # because the memoriser's whole advantage is the unpredictable component of the labels.
        edge = (acc - chance) / (1.0 - chance)
        if edge <= 0.15 and target is None:
            target = n
        print(f"{n:10d} {acc:7.3f} {edge:6.1%}  "
              f"{'the full oracle — mechanism defeated' if edge > 0.6 else
                 'most of the oracle' if edge > 0.3 else
                 'a real but bounded advantage' if edge > 0.15 else
                 'residual; generalisation now competitive'}")

    print(f"\nparams/task at the live LCB bank (112 problems): {head.n_params / 112:.0f}")
    if target:
        print(f"BANK SIZE NEEDED to hold the memoriser's edge under 15%: >= {target} tasks")
    else:
        print("No tested bank size held the edge under 15% — the bank must be larger still")
    print("\nThe edge shrinks with bank size but never reaches zero: capacity is a dial, not a wall.\n"
          "A bank that a miner can cheaply self-label is a bank the miner will self-label, so the\n"
          "suite must be chosen for SIZE as well as routability.")


if __name__ == "__main__":
    main()
