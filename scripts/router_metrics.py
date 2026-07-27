"""The scoring metric for a router agent, stated as the product goal states it:
"for a given user ask, deliver the best answer at the lowest price."

Three properties the CURRENT scalar (S = Sum_b w_b*wilson_lcb - 0.02*cost_frac) does not have:

  1. PER-QUERY, not per-benchmark. The user cares about THEIR ask. A benchmark average cannot see
     whether the router made a good decision on any individual query.
  2. BASELINE-RELATIVE. If the pool is 95% accurate every router scores ~95%, so ~98% of the score
     measures the POOL and ~2% measures the miner. Regret cancels the pool out.
  3. A FRONTIER, not a point. A fixed lambda bakes in ONE quality/price exchange rate -- the owner's.
     That tradeoff belongs to the user: a throwaway ask wants the cheapest adequate answer, a
     business-critical ask wants the best available. Score the whole curve (AIQ), not one point.

Metrics computed here, all against a counterfactual matrix (every (query, model) cell known):

  REGRET      per query: quality_regret = max_m s(q,m) - s(q,chosen)
                         cost_regret    = cost(q,chosen) - min{cost(q,m): s(q,m)=1}
              Zero regret = you matched what was ACHIEVABLE on that specific ask. Queries nobody
              can solve contribute zero regret to everyone, instead of dragging all miners down.

  AIQ         area under the router's non-decreasing cost-quality hull, normalised by the cost
              range (RouterBench). Rewards being good across budgets, not at one operating point.

  vs ZERO     RouterBench's featureless baseline: the best quality reachable at a given cost by
              RANDOMISING over fixed models. "A router is deemed significant only if it
              demonstrates superior performance compared to Rzero." Without this gate, "always
              call the cheapest" scores like a real router.

  HEADROOM    fraction of the achievable oracle gap the router actually captured. This is the
              sentence you can put in front of a miner or a customer: "captures 40% of headroom".

  python scripts/router_score.py --matrix data/pool_matrix_code.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np


def load(path):
    cells, tier = defaultdict(dict), {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("score") is None:
            continue
        cells[c["task_id"]][c["model"]] = c
        tier[c["task_id"]] = c.get("tier", "?")
    models = sorted({m for r in cells.values() for m in r})
    tasks = sorted(t for t, r in cells.items() if len(r) == len(models))
    S = np.array([[cells[t][m]["score"] for m in models] for t in tasks])
    C = np.array([[cells[t][m]["cost_usd"] for m in models] for t in tasks])
    return models, tasks, S, C, [tier[t] for t in tasks]


def hull(points):
    """Non-decreasing upper convex hull of (cost, quality) -- the Zero-router frontier. Every
    interpolated point is realisable by randomising between the two neighbouring models."""
    pts, out, best = sorted(points), [], -1.0
    for c, q in pts:
        best = max(best, q)
        out.append((c, best))
    h = []
    for c, q in out:
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (c - h[-2][0]) <= (q - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append((c, q))
    return h


def on_hull(h, cost):
    if cost <= h[0][0]:
        return h[0][1]
    for (c1, q1), (c2, q2) in zip(h, h[1:]):
        if c1 <= cost <= c2:
            t = (cost - c1) / (c2 - c1) if c2 > c1 else 0.0
            return q1 + t * (q2 - q1)
    return h[-1][1]


def aiq(curve, c_lo, c_hi, n=200):
    """Area under a router's non-decreasing hull over a SHARED cost domain, normalised."""
    h = hull(curve)
    grid = np.linspace(c_lo, c_hi, n)
    return float(np.mean([on_hull(h, c) for c in grid]))


def oracle_frontier(S, C):
    """(cost, quality) points a BUDGET-CONSTRAINED per-query oracle can reach.

    Baseline: the cheapest model on each query. Then buy the upgrades that are cheapest per query
    solved, in order. For binary scores the only upgrade worth buying on a query is to the cheapest
    model that is CORRECT there, so the frontier is exact rather than greedy-approximate."""
    T, M = S.shape
    base = C.argmin(axis=1)
    cur_s, cur_c = S[np.arange(T), base].copy(), C[np.arange(T), base].copy()
    ups = []
    for t in range(T):
        ok = [j for j in range(M) if S[t, j] > cur_s[t]]
        if ok:
            j = min(ok, key=lambda j: C[t, j])          # cheapest model that improves this query
            ups.append((C[t, j] - cur_c[t], S[t, j] - cur_s[t], t, j))
    ups.sort()                                          # cheapest improvement first
    pts = [(cur_c.mean(), cur_s.mean())]
    for dc, ds, t, j in ups:
        cur_c[t], cur_s[t] = C[t, j], S[t, j]
        pts.append((cur_c.mean(), cur_s.mean()))
    return pts


def regret(S, C, sel):
    """Per-query quality and cost regret against what was ACHIEVABLE on that query."""
    T = S.shape[0]
    idx = np.arange(T)
    qual = S.max(axis=1) - S[idx, sel]
    cost_ok = np.array([                      # cheapest model that was CORRECT on this query
        min([C[t, j] for j in range(S.shape[1]) if S[t, j] == 1], default=C[t].min())
        for t in range(T)])
    return qual, C[idx, sel] - cost_ok


def policies(S, C, X=None, folds=5, seed=1):
    """The policies to compare, as a user would experience them."""
    T, M = S.shape
    out = {}
    for j in range(M):
        out[f"always {MODELS[j].split('/')[-1]}"] = np.full(T, j)
    cheapest = int(np.argmin(C.mean(axis=0)))
    out["always cheapest"] = np.full(T, cheapest)
    if X is None:
        return out
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    pred = np.zeros((T, M))
    for tr, te in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        for j in range(M):
            y = S[tr, j]
            if len(set(y)) < 2:
                pred[te, j] = y.mean()
            else:
                pred[te, j] = LogisticRegression(max_iter=2000).fit(X[tr], y).predict_proba(X[te])[:, 1]
    cost_rank = np.argsort(C.mean(axis=0))
    for tol in (0.0, 0.10, 0.25):
        sel = [min((j for j in range(M) if pred[t, j] >= pred[t].max() - tol),
                   key=lambda j: C[:, j].mean()) for t in range(T)]
        out[f"LEARNED router tol={tol:.2f}"] = np.array(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--no-learn", action="store_true")
    a = ap.parse_args()

    global MODELS
    MODELS, tasks, S, C, tier = load(a.matrix)
    T, M = S.shape
    print(f"matrix: {T} queries x {M} models   ({sum(1 for x in tier if x=='hard')} hard)")
    oracle_q = S.max(axis=1).mean()
    best_j = int(S.mean(axis=0).argmax())
    print(f"pool: best-single = {MODELS[best_j].split('/')[-1]} @ {S[:,best_j].mean():.3f}; "
          f"oracle {oracle_q:.3f}; achievable gap {oracle_q - S[:,best_j].mean():+.3f}; "
          f"unsolvable-by-anyone {(S.max(axis=1)==0).mean():.1%}\n")

    X = None
    if not a.no_learn:
        try:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            X = enc.encode([str(t) for t in tasks], normalize_embeddings=True,
                           show_progress_bar=False)
        except ImportError:
            pass

    pool_pts = [(C[:, j].mean(), S[:, j].mean()) for j in range(M)]
    zero_h = hull(pool_pts)
    orac_h = hull(oracle_frontier(S, C))       # the budget-constrained per-query upper bound
    c_lo, c_hi = min(p[0] for p in pool_pts), max(p[0] for p in pool_pts)
    zero_aiq = aiq(pool_pts, c_lo, c_hi)

    print(f"{'policy':<26}{'acc':>7}{'$/q':>10}{'q-regret':>10}{'$-regret':>10}"
          f"{'headroom':>10}{'vs Zero':>9}")
    print("-" * 82)
    rows = []
    for name, sel in policies(S, C, X).items():
        acc = S[np.arange(T), sel].mean()
        cost = C[np.arange(T), sel].mean()
        qr, cr = regret(S, C, sel)
        # FRONTIER-RELATIVE headroom: 0% = no better than randomising over fixed models AT YOUR
        # PRICE; 100% = matched the budget-constrained per-query oracle at that price. The old
        # quality-only version scored a policy that matched quality at 1/50th the cost as NEGATIVE,
        # which is backwards for "best answer at the lowest price".
        z_at, o_at = on_hull(zero_h, cost), on_hull(orac_h, cost)
        cap = (acc - z_at) / max(o_at - z_at, 1e-9)
        vz = acc - z_at
        rows.append((name, acc, cost, qr.mean(), cr.mean(), cap, vz))
    for n_, acc, cost, q, cst, cap, vz in sorted(rows, key=lambda r: -(r[6])):
        print(f"{n_:<26}{acc:>7.3f}{cost:>10.5f}{q:>10.3f}{cst:>+10.5f}{cap:>9.0%}{vz:>+9.3f}")

    print(f"\nZero-router AIQ over [{c_lo:.5f}, {c_hi:.5f}] = {zero_aiq:.4f}")
    print("A router EARNS only if 'vs Zero' > 0: it must beat randomising over fixed models\n"
          "at the same price. 'headroom' = share of the achievable per-query gap it captured.")


if __name__ == "__main__":
    main()
