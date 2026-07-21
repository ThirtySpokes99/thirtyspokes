"""Synthetic world -> OutcomeCache, so all of Phase 0 runs offline with no keys.

The synthetic world is deliberately faithful to the mechanism it stress-tests:

  * Each question has a latent *domain* and *difficulty*.
  * Each model has a per-domain *skill* and a *price*. Cheap models are weaker on
    average but can specialize; strong models are broadly good but expensive.
  * A model is correct with prob sigmoid(sharpness * (skill - difficulty)).
  * The frozen "encoder" emits an embedding that *partially* reveals domain and
    difficulty (controlled by embed_noise) — so a trained router can extract
    routing signal, but never perfectly. This is the whole point: the encoder
    is the ceiling on how well any head can route (the representation bound).

The single most important knob is `specialization`:
  * near 0 -> every model has the same skill profile -> best single model is
    almost as good as the oracle -> SMALL oracle gap -> pool not worth routing.
  * near 1 -> models specialize in different domains -> oracle >> best single ->
    LARGE oracle gap -> genuine headroom for miners.

This lets us demonstrate the oracle-gap admission gate doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import OutcomeCache, PoolModel


@dataclass
class SynthConfig:
    q: int = 6000                # questions
    n_domains: int = 8           # latent task domains (math, code, knowledge, ...)
    embed_dim: int = 96          # frozen-encoder dimensionality
    embed_noise: float = 0.45    # how much the encoder hides domain/difficulty (0=perfect,1=blind)
    sharpness: float = 3.0       # steepness of skill-vs-difficulty -> correctness
    specialization: float = 0.85 # 0=redundant pool, 1=strongly complementary pool
    # Pool: (name, role, base_skill, price_per_token, mean_tokens)
    # base_skill is on the same latent scale as difficulty (~N(0,1)). The ladder is
    # deliberately *compressed* (0.05..0.55) so no single model dominates every
    # domain -- combined with strong `specialization`, different models become the
    # outright accuracy winner in different domains (real accuracy headroom), while
    # the ~60x price span (cheap->frontier) creates cost headroom too.
    pool: tuple[tuple[str, str, float, float, float], ...] = (
        ("tiny-3b",    "cheap",  0.05, 1.0,  220.0),
        ("cheap-9b",   "cheap",  0.20, 2.0,  260.0),
        ("mid-70b",    "mid",    0.35, 10.0, 300.0),
        ("large-200b", "strong", 0.45, 25.0, 340.0),
        ("frontier",   "strong", 0.55, 60.0, 380.0),
    )
    # Pinned verifier quality (used to fill verifier_ok for the cascade action space)
    verifier_tpr: float = 0.90   # P(accept | answer correct)
    verifier_fpr: float = 0.15   # P(accept | answer wrong)
    seed: int = 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: SynthConfig) -> OutcomeCache:
    rng = np.random.default_rng(cfg.seed)
    Q, Dm, K = cfg.q, cfg.n_domains, len(cfg.pool)

    # --- questions: domain + difficulty ---
    domain = rng.integers(0, Dm, size=Q)
    difficulty = rng.normal(0.0, 1.0, size=Q)

    # --- models: per-domain skill = base + specialization * specialist_bonus ---
    base_skill = np.array([p[2] for p in cfg.pool])          # (K,)
    prices = np.array([p[3] for p in cfg.pool])              # (K,)
    mean_tokens = np.array([p[4] for p in cfg.pool])         # (K,)

    # Each model gets a distinct "home" domain it is unusually good at; the
    # specialization knob scales how pronounced that edge is. Cheap models with a
    # strong home-domain edge are what make cost-aware routing pay off.
    specialist_bonus = np.zeros((K, Dm))
    home = rng.permutation(np.arange(Dm))  # assign homes; if K>Dm they wrap
    for k in range(K):
        d_home = home[k % Dm]
        specialist_bonus[k, d_home] = rng.uniform(1.0, 1.8)
        # mild random edges elsewhere so it isn't a clean one-hot
        specialist_bonus[k] += rng.uniform(-0.2, 0.2, size=Dm)
    skill = base_skill[:, None] + cfg.specialization * specialist_bonus  # (K, Dm)

    # --- correctness ~ Bernoulli(sigmoid(sharpness*(skill - difficulty))) ---
    skill_on_q = skill[:, domain].T                          # (Q, K)
    p_correct = _sigmoid(cfg.sharpness * (skill_on_q - difficulty[:, None]))
    correct = rng.random((Q, K)) < p_correct

    # --- cost = generated_tokens * price ---
    tokens = np.clip(rng.normal(mean_tokens, mean_tokens * 0.12, size=(Q, K)), 32, None)
    cost = (tokens * prices).astype(np.float32)

    # --- pinned verifier verdicts (imperfect) ---
    accept_p = np.where(correct, cfg.verifier_tpr, cfg.verifier_fpr)
    verifier_ok = rng.random((Q, K)) < accept_p

    # --- frozen-encoder embedding: partial view of (domain, difficulty) ---
    # A clean signal block (one-hot domain + difficulty) is projected to embed_dim
    # then corrupted by noise. Higher embed_noise -> less routable signal.
    signal = np.zeros((Q, Dm + 1))
    signal[np.arange(Q), domain] = 1.0
    signal[:, Dm] = difficulty
    proj = rng.normal(0, 1, size=(Dm + 1, cfg.embed_dim))
    clean = signal @ proj
    clean /= (clean.std() + 1e-8)
    noise = rng.normal(0, cfg.embed_noise, size=(Q, cfg.embed_dim))
    embeddings = (clean + noise).astype(np.float32)

    models = [PoolModel(name=p[0], price_per_token=p[3], role=p[1]) for p in cfg.pool]
    return OutcomeCache(
        models=models,
        embeddings=embeddings,
        correct=correct,
        cost=cost,
        verifier_ok=verifier_ok,
        domain=domain,
        question_ids=[f"synq{i}" for i in range(Q)],
    )
