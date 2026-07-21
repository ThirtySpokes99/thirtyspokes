"""Cost-aware scoring — the objective every miner is judged by.

    score(policy) = mean_q[ correct ]  -  lambda * mean_q[ normalized_cost ]

Both terms live in [0, 1], so lambda is a direct "how many accuracy points is a
full unit of normalized cost worth" exchange rate. All policies (single-shot
here; cascades in cascade.py) reduce to choosing, per question, which model's
cached outcome to bank — so scoring is a pure table lookup over the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import OutcomeCache


@dataclass
class ScoreBreakdown:
    score: float        # cost-aware objective
    accuracy: float     # mean correctness
    norm_cost: float    # mean normalized cost
    lam: float

    def __str__(self) -> str:
        return (f"score={self.score:+.4f}  acc={self.accuracy:.4f}  "
                f"norm_cost={self.norm_cost:.4f}  (lambda={self.lam})")


def score_choice(cache: OutcomeCache, choice: np.ndarray, lam: float) -> ScoreBreakdown:
    """Score a single-shot policy: choice[q] = index of the model used for q."""
    q = np.arange(cache.Q)
    ncost = cache.normalized_cost()
    acc = float(cache.correct[q, choice].mean())
    nc = float(ncost[q, choice].mean())
    return ScoreBreakdown(score=acc - lam * nc, accuracy=acc, norm_cost=nc, lam=lam)


def per_question_objective(cache: OutcomeCache, lam: float) -> np.ndarray:
    """(Q, K) objective value of picking each model for each question.

    The building block for both the oracle (row-wise argmax) and for turning a
    router's soft distribution into an expected objective during training.
    """
    return cache.correct.astype(np.float32) - lam * cache.normalized_cost()


def constant_choice(cache: OutcomeCache, k: int) -> np.ndarray:
    return np.full(cache.Q, k, dtype=np.int64)


def per_batch_winrate(
    cache: OutcomeCache,
    choices: dict[str, np.ndarray],
    lam: float,
    n_batches: int = 200,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Rank policies by how often each wins a random eval batch (the SN9 pattern).

    More robust than a single global mean: absorbs per-question noise and matches
    how the on-chain reign compares miners. Returns win fraction per policy
    (ties split evenly), summing to ~1 across policies.
    """
    rng = np.random.default_rng(seed)
    names = list(choices)
    obj = per_question_objective(cache, lam)  # (Q, K)
    # per-policy per-question objective -> (P, Q)
    pol_obj = np.stack([obj[np.arange(cache.Q), choices[n]] for n in names], axis=0)
    wins = np.zeros(len(names))
    for _ in range(n_batches):
        idx = rng.integers(0, cache.Q, size=batch_size)
        batch_means = pol_obj[:, idx].mean(axis=1)      # (P,)
        top = np.flatnonzero(batch_means == batch_means.max())
        wins[top] += 1.0 / len(top)
    total = wins.sum() or 1.0
    return {n: float(w / total) for n, w in zip(names, wins)}
