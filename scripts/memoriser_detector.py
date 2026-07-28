#!/usr/bin/env python
"""Can a validator tell a routing-table memoriser from an honest router, for free?

`memoriser_capacity.py` established that the param cap does NOT stop decision-memorisation: a
6.4K-param head fits a RANDOM rung table for 1,000 tasks perfectly. Worse, the mechanism hands the
attacker its training labels -- the owner's reference record publishes per-(ask, model) outcomes so
validators can score without inference, and that same record is exactly the `task -> best rung` table
a memoriser needs. On a static public pool, memorisation is cheap and well-supplied.

So the defence has to be behavioural. The hypothesis measured here:

    An honest router reads MEANING. Semantically similar prompts get similar decisions, because the
    thing it learned -- "asks like this are cheap-solvable" -- is a property of the region, not the
    point. Its decision surface is SMOOTH.

    A memoriser reads an INDEX. Its labels are uncorrelated with semantics, so it must carve a
    separate cell around every memorised point. Its decision surface is JAGGED.

That difference is visible without any labels, without any inference, and without the owner holding
secret tasks: perturb an embedding slightly and watch how far the output distribution moves. A
validator already has the weights and the pinned encoder, so this costs one matrix multiply.

WHY THE THIRD ARM MATTERS. The honest/random contrast is easy and proves nothing on its own -- real
routing signal is partly luck (measured: routers capture 1-12% of the band), so a real miner is
fitting genuinely noisy labels and is PART memoriser. If the detector cannot separate a memoriser
from an honest router trained on 50%-noise labels, it is a false-positive machine and must not gate
emissions. That arm is the actual test.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from thirtyspokes.koth.harness import DEFAULT_HIDDEN, EMBED_DIM


def make_world(n: int, k: int, d: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings with real structure, plus the labels an HONEST router would learn.

    Sentence embeddings are not isotropic noise -- they cluster by topic, and routability means the
    right rung is a property of the topic region. So: cluster centres, points around them, and a
    label rule that is a smooth function of position (a random linear rule). That rule is learnable
    and generalises, which is precisely what distinguishes it from a lookup table.
    """
    centres = rng.normal(0, 1, (k * 2, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    who = rng.integers(0, len(centres), n)
    X = centres[who] + rng.normal(0, 0.35, (n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    W = rng.normal(0, 1, (d, k))
    return X, np.argmax(X @ W, axis=1)


def fit(X: np.ndarray, y: np.ndarray, hidden: int, iters: int, seed: int):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
                        max_iter=iters, random_state=seed, tol=1e-9, n_iter_no_change=iters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def jaggedness(clf, X: np.ndarray, sigma: float, reps: int, rng) -> tuple[float, float]:
    """How far does the decision move under a semantically negligible nudge?

    Returns (mean L1 drift of the distribution, argmax flip rate). Perturbations are applied in
    embedding space and renormalised, so the probe point stays on the unit sphere the encoder emits
    onto -- an off-manifold probe would measure the head's behaviour somewhere it never has to be
    sane, and would flag everyone.
    """
    base = clf.predict_proba(X)
    base_arg = base.argmax(1)
    drift, flip = [], []
    for _ in range(reps):
        Xp = X + rng.normal(0, sigma, X.shape)
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        p = clf.predict_proba(Xp)
        drift.append(np.abs(p - base).sum(1).mean())
        flip.append((p.argmax(1) != base_arg).mean())
    return float(np.mean(drift)), float(np.mean(flip))


def main() -> None:
    ap = argparse.ArgumentParser(description="label-free memoriser detector")
    ap.add_argument("--n", type=int, default=500, help="task bank size")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--sigma", type=float, default=0.05, help="embedding nudge, ~1/7 of cluster width")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"bank={args.n} rungs={args.k} hidden={args.hidden} sigma={args.sigma}")
    print("honest = label is a smooth function of the embedding (generalises)")
    print("noisy  = honest rule, 50% of labels randomised (real routing signal is partly luck)")
    print("memo   = pure lookup table, labels uncorrelated with semantics\n")
    print(f"{'router':>8s} {'train':>7s} {'test':>7s} {'L1 drift':>10s} {'flip':>7s}")

    rows: dict[str, list] = {"honest": [], "noisy": [], "memo": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        X, y = make_world(args.n + 200, args.k, EMBED_DIM, rng)
        Xtr, ytr, Xte, yte = X[:args.n], y[:args.n], X[args.n:], y[args.n:]

        noisy = ytr.copy()
        mask = rng.random(args.n) < 0.5
        noisy[mask] = rng.integers(0, args.k, mask.sum())

        variants = {"honest": ytr, "noisy": noisy, "memo": rng.integers(0, args.k, args.n)}
        for name, labels in variants.items():
            clf = fit(Xtr, labels, args.hidden, args.iters, seed)
            tr = float((clf.predict(Xtr) == labels).mean())
            # test accuracy is measured against the TRUE rule always: it asks "did this router learn
            # anything transferable?", which is the property the detector claims to be a proxy for.
            te = float((clf.predict(Xte) == yte).mean())
            d, f = jaggedness(clf, Xtr, args.sigma, args.reps, np.random.default_rng(1000 + seed))
            rows[name].append((tr, te, d, f))

    for name, vals in rows.items():
        tr, te, d, f = np.mean(vals, axis=0)
        print(f"{name:>8s} {tr:7.3f} {te:7.3f} {d:10.3f} {f:7.3f}")

    hon = np.mean([v[2] for v in rows["honest"]])
    noi = np.mean([v[2] for v in rows["noisy"]])
    mem = np.mean([v[2] for v in rows["memo"]])
    print(f"\nmemoriser is {mem / max(hon, 1e-9):.1f}x jaggier than honest, "
          f"{mem / max(noi, 1e-9):.1f}x jaggier than noisy-honest")
    # The noisy arm is the one that decides deployability: a gate has to sit between noisy-honest and
    # memo, and a margin under ~1.5x is not a threshold anyone can set safely on live traffic.
    if mem > noi * 1.5:
        print("USABLE: a threshold between noisy-honest and memoriser exists")
    else:
        print("NOT USABLE: honest routers on noisy labels look like memorisers — "
              "this gate would DQ real miners; the anti-memorisation defence must come from "
              "task-bank size or held-out reference instead")


if __name__ == "__main__":
    main()
