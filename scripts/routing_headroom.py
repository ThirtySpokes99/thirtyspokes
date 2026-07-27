"""Does this pool have routing headroom? The go/no-go gate over a counterfactual pool matrix.

Reads data/pool_matrix_math.jsonl (built by pool_matrix_math.py) and answers the four questions that
decide whether a routing subnet has anything to compete over. No API calls, no cost, deterministic.

  1. ACCURACY HEADROOM   oracle (per-query best) vs best-single. If ~0 there is nothing to route
                         FOR ACCURACY -- the dominator-pool result, five times over.
  2. COST HEADROOM       cheapest-sufficient (cheapest model that gets it right) vs best-single at
                         matched quality. RouterBench's real finding: 13.6-14.6x. This is the axis
                         that survives when accuracy saturates.
  3. IS IT LEARNABLE     a cross-validated router over prompt embeddings -- the ACHIEVABLE policy,
                         not the oracle. The RouterBench case died here: the gap existed but was
                         per-query noise, not a learnable signal.
  4. DOES IT BEAT THE BASELINES  RouterBench's Zero router (cost-parameterised mixture over the
                         pool's own non-decreasing convex hull, NO per-query features) and
                         best-single. RouterBench: "a router is deemed significant only if it
                         demonstrates superior performance compared to Rzero."

All comparisons are PAIRED on identical prompts -- the variance reduction that makes any of this
resolvable at feasible sample sizes (n ~= 0.227/delta^2 unpaired).

  python scripts/routing_headroom.py --matrix data/pool_matrix_code.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_math.jsonl"



def load(cache=None):
    global CACHE
    if cache: CACHE = pathlib.Path(cache)
    cells = defaultdict(dict)
    meta = {}
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("score") is None:
            continue                                   # errored cell -> treat as missing
        cells[c["task_id"]][c["model"]] = c
        meta[c["task_id"]] = c["tier"]
    return cells, meta


def zero_router(points, target_cost):
    """RouterBench's Zero router: the best quality reachable at `target_cost` by RANDOMISING between
    fixed models, with no per-query features. Build the non-decreasing frontier over (cost, quality),
    take its upper convex hull, and interpolate -- every interpolated point is physically realisable
    as a probabilistic mixture of the two neighbouring models."""
    pts = sorted(points)                               # [(cost, quality)]
    ndec, best = [], -1.0
    for c, q in pts:                                   # non-decreasing extrapolation
        best = max(best, q)
        ndec.append((c, best))
    hull = []                                          # upper convex hull
    for c, q in ndec:
        while len(hull) >= 2:
            (c1, q1), (c2, q2) = hull[-2], hull[-1]
            if (q2 - q1) * (c - c1) <= (q - q1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append((c, q))
    if target_cost <= hull[0][0]:
        return hull[0][1]
    for (c1, q1), (c2, q2) in zip(hull, hull[1:]):
        if c1 <= target_cost <= c2:
            t = (target_cost - c1) / (c2 - c1) if c2 > c1 else 0.0
            return q1 + t * (q2 - q1)
    return hull[-1][1]


def paired(a, b):
    """Paired difference a-b with a t-statistic. Identical prompts, so this is the low-variance test."""
    d = np.asarray(a) - np.asarray(b)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), float("nan")
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--matrix", default=None, help="path to a matrix jsonl")
    args = ap.parse_args()

    cells, tier = load(args.matrix)
    models = sorted({m for row in cells.values() for m in row})
    tasks = sorted(t for t, row in cells.items() if len(row) == len(models))   # complete rows only
    if not tasks:
        print("no complete rows in the matrix yet — run scripts/pool_matrix_math.py first")
        return
    S = np.array([[cells[t][m]["score"] for m in models] for t in tasks])      # (T, M) scores
    C = np.array([[cells[t][m]["cost_usd"] for m in models] for t in tasks])   # (T, M) dollars
    T, M = S.shape
    is_hard = np.array([tier[t] == "hard" for t in tasks])

    print(f"matrix: {T} complete prompts x {M} models "
          f"({(~is_hard).sum()} easy / {is_hard.sum()} hard)\n")

    print(f"{'model':<32}{'acc':>7}{'easy':>7}{'hard':>7}{'$/query':>11}{'sole-correct':>14}")
    print("-" * 78)
    for j, m in enumerate(models):
        sole = ((S[:, j] == 1) & (S.sum(axis=1) == 1)).mean()
        print(f"{m:<32}{S[:,j].mean():>7.3f}{S[~is_hard,j].mean():>7.3f}{S[is_hard,j].mean():>7.3f}"
              f"{C[:,j].mean():>11.5f}{sole:>13.1%}")

    best_j = int(S.mean(axis=0).argmax())
    best_acc, best_cost = S[:, best_j].mean(), C[:, best_j].mean()
    oracle = S.max(axis=1)
    # cheapest model that is correct on each query (else the cheapest overall)
    order = np.argsort(C.mean(axis=0))
    cheap_sel = np.array([next((j for j in order if S[t, j] == 1), order[0]) for t in range(T)])
    cheap_acc = S[np.arange(T), cheap_sel].mean()
    cheap_cost = C[np.arange(T), cheap_sel].mean()

    print(f"\n{'BEST-SINGLE':<24}{models[best_j]:<30}acc {best_acc:.3f}  ${best_cost:.5f}/query")
    print(f"{'ORACLE (upper bound)':<24}{'':<30}acc {oracle.mean():.3f}  "
          f"gap +{oracle.mean()-best_acc:.3f}")
    print(f"{'CHEAPEST-SUFFICIENT':<24}{'':<30}acc {cheap_acc:.3f}  ${cheap_cost:.5f}/query  "
          f"({best_cost/max(cheap_cost,1e-12):.1f}x cheaper)")
    print(f"{'nobody correct':<24}{'':<30}{(oracle==0).mean():.1%} of prompts")

    # --- the ACHIEVABLE router: cross-validated P(correct | prompt embedding) per model ---
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold
    except ImportError:
        print("\n(sentence-transformers/sklearn missing — skipping the learned router)")
        return
    prompts = [cells[t][models[0]].get("prompt", t) for t in tasks]
    print("\nencoding prompts (MiniLM) ...", flush=True)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = enc.encode(prompts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    pred = np.zeros((T, M))
    for tr, te in KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X):
        for j in range(M):
            y = S[tr, j]
            if len(set(y)) < 2:                        # degenerate fold -> fall back to the prior
                pred[te, j] = y.mean()
                continue
            pred[te, j] = LogisticRegression(max_iter=2000).fit(X[tr], y).predict_proba(X[te])[:, 1]

    # route to the CHEAPEST model whose predicted success is within tol of the best prediction —
    # difficulty-conditioned cost arbitrage, which is the policy class that survives a dominator pool
    results = {}
    for tol in (0.0, 0.05, 0.10):
        sel = []
        for t in range(T):
            ok = pred[t] >= pred[t].max() - tol
            sel.append(min((j for j in range(M) if ok[j]), key=lambda j: C[:, j].mean()))
        sel = np.array(sel)
        acc, cost = S[np.arange(T), sel].mean(), C[np.arange(T), sel].mean()
        results[tol] = (acc, cost, sel)

    pool_pts = [(C[:, j].mean(), S[:, j].mean()) for j in range(M)]
    print(f"\n{'router (tol)':<16}{'acc':>7}{'$/query':>11}{'vs best-single':>17}"
          f"{'vs Zero@cost':>15}{'paired t':>10}")
    print("-" * 78)
    for tol, (acc, cost, sel) in results.items():
        z = zero_router(pool_pts, cost)
        d, tstat = paired(S[np.arange(T), sel], S[:, best_j])
        print(f"tol={tol:<11.2f}{acc:>7.3f}{cost:>11.5f}"
              f"{d:>+16.3f} {acc-z:>+14.3f}{tstat:>10.2f}")

    print(f"\nZero router at best-single's cost (${best_cost:.5f}): "
          f"{zero_router(pool_pts, best_cost):.3f}")
    print("\nGATE: accuracy gap needs |t| > ~2 to be real; if it is not, read the COST column —")
    print("      a large cheapest-sufficient ratio at flat accuracy is a COST-routing subnet.")


if __name__ == "__main__":
    main()
