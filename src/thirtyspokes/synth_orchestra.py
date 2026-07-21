"""Synthetic composition world -> workflow OutcomeCache, for the Phase-0 pivot test.

The decisive question for the orchestration/MoE subnet (ARCHITECTURE.md v5 §2b):
does composing a pool have *rising, non-saturating* headroom, or does it plateau
like routing? To answer it honestly, composition primitives must have PRINCIPLED
effects grounded in real phenomena, so the saturation curve EMERGES rather than
being hard-coded:

  * ensemble/vote  -- helps when model errors are INDEPENDENT (variance
    reduction); fails when they are CORRELATED (a shared 'trap' wrong answer
    everyone falls for). Modeled via a per-question trap_pull.
  * verify_cascade -- cost-efficient error correction, capped by verifier quality
    (imperfect tpr/fpr).
  * refine         -- iterative self-correction with DIMINISHING returns (each
    round fixes a decaying fraction of remaining errors; a residual is unfixable).

The elegant reduction: enumerate a bounded set of candidate workflows, precompute
each workflow's (correct, cost) per question, and the whole thing is an
OutcomeCache whose K "actions" are workflows. Every verified thirtyspokes component
(oracle gap, train_router, reign) then applies unchanged -- the orchestrator is
just a policy over workflow-actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .cache import OutcomeCache, PoolModel


@dataclass
class OrchestraConfig:
    q: int = 6000
    n_domains: int = 6
    embed_dim: int = 80
    embed_noise: float = 0.4
    n_choices: int = 5          # MC-style: answer 0 is correct, 1 is the "trap"
    sharpness: float = 2.5
    specialization: float = 0.6
    refine_rounds: int = 3      # max refinement depth
    refine_fix0: float = 0.45   # prob a refine round fixes a wrong answer (round 1)
    refine_decay: float = 0.55  # per-round decay of fix prob (diminishing returns)
    difficulty_shift: float = 0.0   # >0 => harder tasks (leaves residual headroom)
    specialist_skill: float = 1.0   # base skill of churn-added specialists
    verifier_tpr: float = 0.85
    verifier_fpr: float = 0.20
    # pool: (name, role, base_skill, price)
    pool: tuple[tuple[str, str, float, float], ...] = (
        ("small",  "cheap",  -0.2, 1.0),
        ("medium", "mid",     0.3, 6.0),
        ("large",  "strong",  0.7, 25.0),
    )
    seed: int = 0


@dataclass
class World:
    cfg: OrchestraConfig
    answer: np.ndarray        # (Q, M) answer id each model gives (0 = correct)
    verifier_ok: np.ndarray   # (Q, M) verifier accepts model m's answer
    refine_correct: np.ndarray  # (Q, M, R+1) correct after r refine rounds
    prices: np.ndarray        # (M,)
    embeddings: np.ndarray     # (Q, D)
    domain: np.ndarray
    model_names: list[str]
    model_roles: list[str]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_world(cfg: OrchestraConfig, extra_specialists: int = 0) -> World:
    rng = np.random.default_rng(cfg.seed)
    Q, Dm = cfg.q, cfg.n_domains
    pool = list(cfg.pool)
    # churn: optionally add STRONG domain-specialist models. Each covers a distinct
    # home domain (spread across the space), so it adds genuinely non-redundant
    # capability -- the MoE / decentralized-experts dynamic.
    n_base = len(pool)
    for i in range(extra_specialists):
        pool.append((f"specialist{i+1}", "specialist", cfg.specialist_skill, 8.0))
    M = len(pool)

    domain = rng.integers(0, Dm, size=Q)
    difficulty = rng.normal(cfg.difficulty_shift, 1, size=Q)
    trap_pull = rng.uniform(0.1, 0.9, size=Q)   # per-question error correlation

    base = np.array([p[2] for p in pool])
    prices = np.array([p[3] for p in pool])
    spec_bonus = np.zeros((M, Dm))
    home = rng.integers(0, Dm, size=M)
    for i in range(extra_specialists):        # spread specialists over distinct domains
        home[n_base + i] = i % Dm
    for m in range(M):
        spec_bonus[m, home[m]] = rng.uniform(0.8, 1.6)
    skill = base[:, None] + cfg.specialization * spec_bonus     # (M, Dm)

    p_c = _sigmoid(cfg.sharpness * (skill[:, domain].T - difficulty[:, None]))  # (Q, M)

    # answers: correct (0), else trap (1) w.p. trap_pull, else a diverse wrong one
    answer = np.zeros((Q, M), dtype=np.int64)
    for m in range(M):
        correct = rng.random(Q) < p_c[:, m]
        trap = rng.random(Q) < trap_pull
        wrong = np.where(trap, 1, rng.integers(2, cfg.n_choices, size=Q))
        answer[:, m] = np.where(correct, 0, wrong)

    # verifier verdicts
    is_corr = answer == 0
    accept_p = np.where(is_corr, cfg.verifier_tpr, cfg.verifier_fpr)
    verifier_ok = rng.random((Q, M)) < accept_p

    # refine outcomes: monotone, diminishing per round
    R = cfg.refine_rounds
    refine_correct = np.zeros((Q, M, R + 1), dtype=bool)
    refine_correct[:, :, 0] = is_corr
    for r in range(1, R + 1):
        fix_p = cfg.refine_fix0 * (cfg.refine_decay ** (r - 1))
        prev = refine_correct[:, :, r - 1]
        newly = (~prev) & (rng.random((Q, M)) < fix_p)
        refine_correct[:, :, r] = prev | newly

    # frozen encoder: noisy view of (domain, difficulty, trap_pull)
    sig = np.zeros((Q, Dm + 2))
    sig[np.arange(Q), domain] = 1.0
    sig[:, Dm] = difficulty
    sig[:, Dm + 1] = trap_pull
    proj = rng.normal(0, 1, size=(Dm + 2, cfg.embed_dim))
    clean = sig @ proj
    clean /= clean.std() + 1e-8
    emb = (clean + rng.normal(0, cfg.embed_noise, size=(Q, cfg.embed_dim))).astype(np.float32)

    return World(
        cfg=cfg, answer=answer, verifier_ok=verifier_ok, refine_correct=refine_correct,
        prices=prices, embeddings=emb, domain=domain,
        model_names=[p[0] for p in pool], model_roles=[p[1] for p in pool],
    )


# --- composition primitives: each yields (correct[Q], cost[Q], n_calls) -------
def _majority_correct(answers: np.ndarray, n_choices: int, strong_local: int) -> np.ndarray:
    """Row-wise majority vote over selected models' answers; correct if 0 wins.
    Ties broken toward the strongest model's answer (strong_local = its column)."""
    Q = answers.shape[0]
    onehot = answers[:, :, None] == np.arange(n_choices)[None, None, :]  # (Q, n, C)
    counts = onehot.sum(axis=1)                                          # (Q, C)
    maxc = counts.max(axis=1)
    strong_ans = answers[:, strong_local]
    strong_wins = counts[np.arange(Q), strong_ans] == maxc
    fallback = counts.argmax(axis=1)                                     # first max choice
    winner = np.where(strong_wins, strong_ans, fallback)
    return winner == 0


def enumerate_workflows(world: World, budget: int):
    """All workflows within a per-task call budget. Returns list of dicts with
    name, correct(Q), cost(Q), n_calls, kind."""
    A = world.answer
    V = world.verifier_ok
    RC = world.refine_correct
    price = world.prices
    Q, M = A.shape
    wfs = []

    def add(name, correct, cost, ncalls, kind):
        wfs.append(dict(name=name, correct=correct, cost=cost.astype(np.float32),
                        n_calls=ncalls, kind=kind))

    # route(m): 1 call
    for m in range(M):
        add(f"route[{world.model_names[m]}]", A[:, m] == 0,
            np.full(Q, price[m]), 1, "route")

    # ensemble/vote over subsets (size 2..budget)
    for size in range(2, budget + 1):
        for subset in combinations(range(M), size):
            strong_local = int(np.argmax(price[list(subset)]))
            corr = _majority_correct(A[:, subset], world.cfg.n_choices, strong_local)
            cost = np.full(Q, price[list(subset)].sum())
            add(f"vote{list(subset)}", corr, cost, size, "vote")

    # verify_cascade: try models in cost order, escalate on verifier reject
    order = np.argsort(price)  # cheap -> expensive
    for start in range(min(2, M)):          # a couple of entry points
        rungs = order[start:]
        if len(rungs) < 2 or len(rungs) > budget:
            continue
        banked = np.zeros(Q, dtype=bool)
        summed = np.zeros(Q)
        done = np.zeros(Q, dtype=bool)
        for pos, m in enumerate(rungs):
            active = ~done
            summed[active] += price[m]
            last = pos == len(rungs) - 1
            accept = V[:, m]
            newly = active & (accept | last)
            banked[newly] = A[newly, m] == 0
            done[newly] = True
        add(f"cascade@{world.model_names[order[start]]}", banked, summed, len(rungs), "cascade")

    # refine(m, r): r extra calls (cost ~ price*(1+r))
    for m in range(M):
        for r in range(1, world.cfg.refine_rounds + 1):
            if 1 + r > budget:
                continue
            add(f"refine[{world.model_names[m]}x{r}]", RC[:, m, r],
                np.full(Q, price[m] * (1 + r)), 1 + r, "refine")

    return wfs


def build_workflow_cache(world: World, budget: int) -> tuple[OutcomeCache, np.ndarray]:
    """OutcomeCache whose K actions are workflows. Returns (cache, is_route_mask)."""
    wfs = enumerate_workflows(world, budget)
    Q = world.answer.shape[0]
    correct = np.stack([w["correct"] for w in wfs], axis=1)
    cost = np.stack([w["cost"] for w in wfs], axis=1)
    models = [PoolModel(name=w["name"], price_per_token=float(w["cost"].mean()),
                        role=w["kind"]) for w in wfs]
    is_route = np.array([w["kind"] == "route" for w in wfs])
    cache = OutcomeCache(
        models=models, embeddings=world.embeddings, correct=correct, cost=cost,
        verifier_ok=None, domain=world.domain,
        question_ids=[f"q{i}" for i in range(Q)],
        cost_norm=float(cost.max()),
    )
    return cache, is_route
