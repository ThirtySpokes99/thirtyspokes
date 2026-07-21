"""Train a router head with separable CMA-ES to maximize the cost-aware score.

Fitness is the *expected* objective under the head's soft distribution
(sum_k dist[q,k] * objective[q,k], averaged over questions). This smooth
surrogate optimizes far better than the piecewise-constant hard-argmax score,
while the head we return is evaluated with the real hard argmax decision — the
same soft-train / hard-eval split Sakana-Fugu-base uses in its SFT stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import OutcomeCache
from .router import RouterHead
from .scorer import ScoreBreakdown, per_question_objective, score_choice
from .sepcmaes import SepCMAES


@dataclass
class TrainedRouter:
    head: RouterHead
    theta: np.ndarray
    lam: float
    train_score: float
    history: list[float]

    def distribution(self, embeddings: np.ndarray) -> np.ndarray:
        return self.head.distribution(self.theta, embeddings)

    def choose(self, embeddings: np.ndarray) -> np.ndarray:
        return self.head.choose(self.theta, embeddings)

    def evaluate(self, cache: OutcomeCache) -> ScoreBreakdown:
        return score_choice(cache, self.choose(cache.embeddings), self.lam)


def train_router(
    train_cache: OutcomeCache,
    lam: float,
    hidden: int = 16,
    iters: int = 220,
    sigma0: float = 0.3,
    seed: int = 0,
    max_train_q: int = 4000,
    verbose: bool = False,
) -> TrainedRouter:
    head = RouterHead(train_cache.D, train_cache.K, hidden)

    # optional subsample for speed on large corpora (deterministic, own stream)
    emb = train_cache.embeddings
    obj = per_question_objective(train_cache, lam)  # (Q, K)
    if train_cache.Q > max_train_q:
        idx = np.random.default_rng(seed + 2).choice(train_cache.Q, max_train_q, replace=False)
        emb, obj = emb[idx], obj[idx]

    def neg_fitness(theta: np.ndarray) -> float:
        dist = head.distribution(theta, emb)             # (q, K)
        return -float((dist * obj).sum(axis=1).mean())   # maximize expected objective

    # distinct streams for init vs sampling (review L5 hygiene)
    es = SepCMAES(head.init_theta(seed), sigma0=sigma0, seed=seed + 1)
    best_theta = es.mean.copy()
    best_neg = neg_fitness(best_theta)
    history: list[float] = []

    for it in range(iters):
        pop = es.ask()
        fits = np.array([neg_fitness(x) for x in pop])
        es.tell(pop, fits)
        # track the best already-evaluated individual (review L5: previously only
        # es.mean was considered, discarding better sampled candidates)
        pop_best = int(fits.argmin())
        if fits[pop_best] < best_neg:
            best_neg, best_theta = float(fits[pop_best]), pop[pop_best].copy()
        cand_neg = neg_fitness(es.mean)
        if cand_neg < best_neg:
            best_neg, best_theta = cand_neg, es.mean.copy()
        history.append(-best_neg)
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"  iter {it:4d}  expected_obj={-best_neg:+.4f}  sigma={es.sigma:.3f}")

    train_hard = score_choice(train_cache, head.choose(best_theta, train_cache.embeddings), lam)
    return TrainedRouter(
        head=head, theta=best_theta, lam=lam,
        train_score=train_hard.score, history=history,
    )
