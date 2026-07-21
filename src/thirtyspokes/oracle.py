"""The oracle gap — the subnet's north-star metric and Phase-0 admission gate.

Two different "oracles", and the difference between them matters a lot:

  * per-sample oracle (UPPER BOUND) — per question, pick the action with the best
    *realized* cached outcome. This is what a validator can trivially compute, but
    it is noise-inflated: it "knows" which model happened to get lucky on each
    question, which no router conditioned only on the embedding can predict. Use
    it as a ceiling, never as the achievable target.

  * routable ceiling (ACHIEVABLE) — the best routing *function of the embedding*,
    estimated honestly by 2-fold cross-fit: cluster embeddings, pick the best
    model per cluster on one fold, score on the held-out fold. Noise that isn't
    predictable from the embedding does not survive the fold change, so this is a
    fair estimate of the headroom a real router can capture.

The admission gate keys on the routable ACCURACY gap (comparable to TinyRouter's
"+2.3 points"). We also report the routable COST-AWARE gap, because a pool with
little capability complementarity can still be worth routing purely for cost
(send easy queries to cheap models) — that shows up here even when the accuracy
gap is ~0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import OutcomeCache
from .scorer import ScoreBreakdown, constant_choice, per_question_objective, score_choice


# --- tiny k-means over embeddings (for the routable-ceiling cross-fit) -------
def _kmeans(x: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = _assign(x, centroids)
        new = np.stack([x[assign == j].mean(0) if np.any(assign == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids


def _assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # squared distances via ||x||^2 - 2 x.c + ||c||^2 (no (N,k,D) broadcast)
    d = (x ** 2).sum(1)[:, None] - 2 * x @ centroids.T + (centroids ** 2).sum(1)[None, :]
    return d.argmin(1)


def _routable_choice(cache: OutcomeCache, lam: float, n_groups: int, seed: int) -> np.ndarray:
    """2-fold cross-fit: best model per embedding-cluster, applied out-of-fold."""
    obj = per_question_objective(cache, lam)          # (Q, K)
    x = cache.embeddings
    Q = cache.Q
    perm = np.random.default_rng(seed).permutation(Q)
    folds = [perm[: Q // 2], perm[Q // 2:]]
    choice = np.zeros(Q, dtype=np.int64)
    for fit_idx, app_idx in [(folds[0], folds[1]), (folds[1], folds[0])]:
        cent = _kmeans(x[fit_idx], n_groups, seed)
        a_fit = _assign(x[fit_idx], cent)
        a_app = _assign(x[app_idx], cent)
        # empty-cluster fallback fitted on the FIT fold only (review L1: using
        # full-cache stats here leaked out-of-fold outcomes into the estimate)
        fallback = int(obj[fit_idx].mean(0).argmax())
        best = np.full(len(cent), fallback, dtype=np.int64)
        for c in range(len(cent)):
            members = fit_idx[a_fit == c]
            if len(members):
                best[c] = int(obj[members].mean(0).argmax())
        choice[app_idx] = best[a_app]
    return choice


@dataclass
class OracleReport:
    lam: float
    per_model_acc: dict[str, float]
    best_single_acc: float
    best_single_acc_model: str
    best_single_obj: ScoreBreakdown
    best_single_obj_model: str
    # per-sample oracle (upper bound, noise-inflated)
    ps_oracle_acc: float
    ps_oracle_obj: ScoreBreakdown
    # routable ceiling (achievable, cross-fit)
    routable_acc: float
    routable_obj: ScoreBreakdown
    routable_accuracy_gap_points: float      # (routable_acc - best_single_acc)*100
    routable_cost_aware_gap: float           # routable_obj.score - best_single_obj.score
    threshold_points: float
    passes: bool

    def summary(self) -> str:
        return "\n".join([
            "Oracle-gap report",
            "-----------------",
            f"  lambda                    : {self.lam}",
            "  per-model accuracy        : "
            + ", ".join(f"{m}={a:.3f}" for m, a in self.per_model_acc.items()),
            f"  best single (acc)         : {self.best_single_acc:.4f}  [{self.best_single_acc_model}]",
            f"  best single (obj)         : {self.best_single_obj.score:+.4f}  [{self.best_single_obj_model}]",
            f"  per-sample oracle (upper) : acc={self.ps_oracle_acc:.4f}  obj={self.ps_oracle_obj.score:+.4f}   (noise-inflated ceiling)",
            f"  routable ceiling (achiev.): acc={self.routable_acc:.4f}  obj={self.routable_obj.score:+.4f}   (cross-fit)",
            f"  ROUTABLE ACCURACY GAP     : {self.routable_accuracy_gap_points:+.2f} points",
            f"  ROUTABLE COST-AWARE GAP   : {self.routable_cost_aware_gap:+.4f}   (cost headroom even if acc gap ~0)",
            f"  admission (>= {self.threshold_points:.1f} pts)    : "
            + ("PASS  -- pool worth building on"
               if self.passes else "FAIL  -- accuracy headroom too small (cost-only value at best)"),
        ])


def per_sample_oracle_choice(cache: OutcomeCache, lam: float) -> np.ndarray:
    return per_question_objective(cache, lam).argmax(axis=1)


def compute(
    cache: OutcomeCache,
    lam: float,
    threshold_points: float = 5.0,
    n_groups: int = 16,
    seed: int = 0,
) -> OracleReport:
    names = [m.name for m in cache.models]
    per_model_acc = {n: float(a) for n, a in zip(names, cache.model_accuracy())}

    accs = cache.model_accuracy()
    bi_acc = int(accs.argmax())
    single = [score_choice(cache, constant_choice(cache, k), lam) for k in range(cache.K)]
    bi_obj = int(np.argmax([s.score for s in single]))

    # per-sample oracle (upper bound)
    ps_choice = per_sample_oracle_choice(cache, lam)
    ps_obj = score_choice(cache, ps_choice, lam)
    ps_acc = float(cache.correct[np.arange(cache.Q), ps_choice].mean())

    # routable ceiling (cross-fit): accuracy criterion (lam=0) and cost-aware criterion
    r_acc_choice = _routable_choice(cache, 0.0, n_groups, seed)
    r_obj_choice = _routable_choice(cache, lam, n_groups, seed)
    routable_acc = float(cache.correct[np.arange(cache.Q), r_acc_choice].mean())
    routable_obj = score_choice(cache, r_obj_choice, lam)

    routable_accuracy_gap_points = (routable_acc - float(accs[bi_acc])) * 100.0
    routable_cost_aware_gap = routable_obj.score - single[bi_obj].score

    return OracleReport(
        lam=lam,
        per_model_acc=per_model_acc,
        best_single_acc=float(accs[bi_acc]),
        best_single_acc_model=names[bi_acc],
        best_single_obj=single[bi_obj],
        best_single_obj_model=names[bi_obj],
        ps_oracle_acc=ps_acc,
        ps_oracle_obj=ps_obj,
        routable_acc=routable_acc,
        routable_obj=routable_obj,
        routable_accuracy_gap_points=routable_accuracy_gap_points,
        routable_cost_aware_gap=routable_cost_aware_gap,
        threshold_points=threshold_points,
        passes=routable_accuracy_gap_points >= threshold_points,
    )
