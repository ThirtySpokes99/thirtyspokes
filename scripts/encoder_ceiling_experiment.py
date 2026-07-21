#!/usr/bin/env python
"""Upper-bound experiment: is the ~10% routing ceiling an ENCODER limitation or
FUNDAMENTAL to broad-traffic routing?

RouterBench gives ground-truth per-(question, model) correctness, so we can train
the STRONGEST supervised difficulty predictor the data allows -- a gradient-boosted
classifier per pool model predicting P(model succeeds) -- and trace its cost-quality
frontier on held-out data. That is a near-upper-bound on what ANY prompt-conditioned
router could achieve on this pool + distribution.

Reference points it sits between:
  * tiny CMA-ES head on MiniLM (what a miner submits)   -- ~10% cost saving (weak)
  * per-sample oracle (uses hidden luck)                -- ~84% cost saving (unreachable)

Where the supervised frontier lands is the answer:
  near the tiny head  -> ceiling is FUNDAMENTAL (routing has little juice here)
  near the oracle     -> the ENCODER/HEAD was the bottleneck (subnet has a moat)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from thirtyspokes import realdata as rd
from thirtyspokes.train import train_router


def frontier(pcorrect, correct, cost, ncost, gpt4_k, lambdas):
    """Trace (accuracy, cost%) by utility-max routing over predicted success."""
    q = np.arange(len(correct))
    gpt4_acc = correct[:, gpt4_k].mean()
    gpt4_cost = cost[:, gpt4_k].sum()
    pts = []
    for lam in lambdas:
        util = pcorrect - lam * ncost           # expected utility per model
        ch = util.argmax(1)
        acc = correct[q, ch].mean()
        c = cost[q, ch].sum()
        pts.append((lam, acc, c / gpt4_cost))
    return pts, gpt4_acc


def iso_saving(pts, gpt4_acc, tol=0.005):
    ok = [(1 - cp) for _, a, cp in pts if a >= gpt4_acc - tol]
    return max(ok) if ok else 0.0


def main() -> None:
    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    domains, dom_names = rd._factorize(df["eval_name"].to_numpy())
    cache = rd.to_cache(df, rd.CURATED_POOL, emb)
    models = [m.name for m in cache.models]
    gpt4 = models.index("gpt-4-1106-preview")
    correct, cost, ncost = cache.correct, cache.cost, cache.normalized_cost()
    Q, K = cache.Q, cache.K

    # prompt-derived features
    plen = np.array([len(p) for p in df["prompt"]], dtype=np.float32)
    plen = ((plen - plen.mean()) / (plen.std() + 1e-9)).reshape(-1, 1)
    dom_oh = np.zeros((Q, len(dom_names)), dtype=np.float32)
    dom_oh[np.arange(Q), domains] = 1.0

    feature_sets = {
        "MiniLM only": emb,
        "MiniLM + len": np.hstack([emb, plen]),
        "MiniLM + len + task-id": np.hstack([emb, plen, dom_oh]),  # generous: knows task type
    }

    rng = np.random.default_rng(0)
    perm = rng.permutation(Q)
    tr_idx, te_idx = perm[: Q // 2], perm[Q // 2:]
    lambdas = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 24)])

    gpt4_acc = correct[te_idx][:, gpt4].mean()
    any_correct = correct[te_idx].any(1).mean()
    print("=" * 76)
    print(f"REAL pool {models}")
    print(f"held-out: always-GPT-4 acc={gpt4_acc:.4f} | per-sample-oracle acc={any_correct:.4f}")
    print("=" * 76)

    # reference: tiny CMA-ES head on MiniLM (what a miner submits)
    train_c, ev_c = cache.split(eval_frac=0.5, seed=0)
    q_ev = np.arange(ev_c.Q)
    best_tiny = 0.0
    for lam in [0.0, 0.05, 0.15, 0.4, 1.0]:
        th = train_router(train_c, lam=lam, hidden=32, iters=200, seed=0)
        ch = th.choose(ev_c.embeddings)
        if ev_c.correct[q_ev, ch].mean() >= ev_c.correct[:, gpt4].mean() - 0.005:
            best_tiny = max(best_tiny, 1 - ev_c.cost[q_ev, ch].sum() / ev_c.cost[:, gpt4].sum())
    print(f"\n[reference] tiny CMA-ES head on MiniLM: iso-accuracy cost saving = {best_tiny*100:.1f}%")

    print("\n[supervised upper bound] HistGradientBoosting per-model success predictors:")
    for fname, X in feature_sets.items():
        pcorr_te = np.zeros((len(te_idx), K))
        for k in range(K):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                 max_leaf_nodes=31, random_state=0)
            clf.fit(X[tr_idx], correct[tr_idx, k].astype(int))
            pcorr_te[:, k] = clf.predict_proba(X[te_idx])[:, 1]
        pts, g_acc = frontier(pcorr_te, correct[te_idx], cost[te_idx], ncost[te_idx], gpt4, lambdas)
        save = iso_saving(pts, g_acc)
        # also the best accuracy achievable at <= GPT-4 cost
        acc_at_isocost = max((a for _, a, cp in pts if cp <= 1.0), default=g_acc)
        print(f"  {fname:26s}: iso-accuracy cost saving = {save*100:5.1f}%  |  "
              f"max acc @<=GPT-4 cost = {acc_at_isocost:.4f} (GPT-4={g_acc:.4f})")

    print("\nverdict: compare supervised saving to the tiny-head ~10% and the")
    print("oracle ~84%. Near the tiny head => fundamental; near the oracle => encoder-bound.")


if __name__ == "__main__":
    main()
