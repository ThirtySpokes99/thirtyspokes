"""Synthetic world + backend for the TinyRouter-faithful multi-turn orchestrator.

TinyRouter's real mechanism (not pick-one routing): per turn the conductor chooses
a (model, role) with role in {Thinker, Worker, Verifier}; turns accumulate in a
shared transcript for up to N turns; a Verifier turn can accept and stop early.
This models that loop's *effects* with principled, diminishing behavior so the
composition value (and its saturation) emerge rather than being hard-coded:

  * Thinker(model): adds reasoning to the transcript that raises the success
    probability of a subsequent Worker — a boost with diminishing returns
    (each extra Thinker adds less), scaled by the thinker model's skill.
  * Worker(model): produces an answer; P(correct) = sigmoid(skill - difficulty +
    accumulated thinker_context). This is where composition beats a single model:
    a strong Thinker can lift a cheap Worker over the line.
  * Verifier(model): judges the current answer; accepts (and stops) with a
    true/false-positive rate set by the verifier model's skill. A weak verifier
    that accepts a wrong answer ends the episode wrong — so verification is
    valuable but risky, and the policy must learn *when* to trust it.

The backend is deliberately behind a vectorized interface (PoolBackend) so the
same orchestrator engine runs on this synthetic world now and on live models
later (see backends.py seam), with real per-(model,role) calls swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TinyConfig:
    q: int = 6000
    n_domains: int = 8
    embed_dim: int = 80
    embed_noise: float = 0.4
    sharpness: float = 2.2
    specialization: float = 0.7
    difficulty_shift: float = 0.6      # >0 leaves headroom for composition to fill
    max_turns: int = 5
    think_base: float = 0.9            # thinker context added at n_think=0
    think_decay: float = 0.5          # diminishing returns per extra thinker
    seed: int = 0
    # extended pool: (name, role_label, base_skill, price)
    pool: tuple[tuple[str, str, float, float], ...] = (
        ("tiny-1b",     "cheap",  -0.6, 1.0),
        ("small-8b",    "cheap",  -0.2, 2.0),
        ("mid-32b",     "mid",     0.1, 6.0),
        ("large-70b",   "mid",     0.35, 12.0),
        ("xl-200b",     "strong",  0.6, 25.0),
        ("frontier-a",  "strong",  0.8, 45.0),
        ("frontier-b",  "strong",  0.85, 60.0),
    )


class SyntheticBackend:
    """Vectorized per-turn role effects over a fixed synthetic pool."""

    def __init__(self, cfg: TinyConfig, task_seed: int | None = None):
        # Two independent RNGs so the POOL (models: skills, prices, role competence,
        # the frozen-encoder projection) is fixed by cfg.seed across epochs, while
        # the TASKS (questions: domain, difficulty, per-turn randomness, embed noise)
        # are freshly sampled per epoch by task_seed. This mirrors the real subnet:
        # an owner-curated pool + a hardening/rotating task stream.
        pool_rng = np.random.default_rng(cfg.seed)
        task_rng = np.random.default_rng(cfg.seed if task_seed is None else task_seed)
        self.cfg = cfg
        pool = list(cfg.pool)
        self.names = [p[0] for p in pool]
        self.roles_label = [p[1] for p in pool]
        self.price = np.array([p[3] for p in pool], dtype=np.float64)
        self.K = len(pool)
        Dm = cfg.n_domains

        # --- POOL (fixed by cfg.seed) ---
        base = np.array([p[2] for p in pool])
        homes = pool_rng.integers(0, Dm, size=self.K)
        spec = np.zeros((self.K, Dm))
        for m in range(self.K):
            spec[m, homes[m]] = pool_rng.uniform(0.8, 1.6)
        self.skill = base[:, None] + cfg.specialization * spec        # (K, Dm)
        mean_skill = self.skill.mean(axis=1)
        self.think_delta = cfg.think_base * (0.5 + _sigmoid(mean_skill))       # (K,)
        self.verify_tpr = 0.60 + 0.35 * _sigmoid(1.5 * mean_skill)             # accept|correct
        self.verify_fpr = 0.35 - 0.22 * _sigmoid(1.5 * mean_skill)             # accept|wrong
        proj = pool_rng.normal(0, 1, size=(Dm + 1, cfg.embed_dim))             # frozen encoder

        # --- TASKS (fresh by task_seed) ---
        self.domain = task_rng.integers(0, Dm, size=cfg.q)
        self.difficulty = task_rng.normal(cfg.difficulty_shift, 1.0, size=cfg.q)
        self.u_work = task_rng.random((cfg.q, cfg.max_turns))
        self.u_verify = task_rng.random((cfg.q, cfg.max_turns))
        sig = np.zeros((cfg.q, Dm + 1))
        sig[np.arange(cfg.q), self.domain] = 1.0
        sig[:, Dm] = self.difficulty
        clean = sig @ proj
        clean /= clean.std() + 1e-8
        self.embeddings = (clean + task_rng.normal(0, cfg.embed_noise,
                                                   size=(cfg.q, cfg.embed_dim))).astype(np.float32)

    @property
    def Q(self) -> int:
        return self.cfg.q

    # --- vectorized role effects (called by the orchestrator engine) ---------
    def worker_prob(self, model_per_q: np.ndarray, thinker_context: np.ndarray) -> np.ndarray:
        sk = self.skill[model_per_q, self.domain]                     # (Q,)
        return _sigmoid(self.cfg.sharpness * (sk - self.difficulty + thinker_context))

    def worker_correct(self, model_per_q, thinker_context, turn) -> np.ndarray:
        return self.u_work[:, turn] < self.worker_prob(model_per_q, thinker_context)

    def thinker_increment(self, model_per_q, n_think) -> np.ndarray:
        return self.think_delta[model_per_q] * (self.cfg.think_decay ** n_think)

    def verifier_accept(self, model_per_q, working_correct, turn) -> np.ndarray:
        rate = np.where(working_correct == 1, self.verify_tpr[model_per_q],
                        self.verify_fpr[model_per_q])
        return self.u_verify[:, turn] < rate

    # baselines
    def single_model_accuracy(self) -> np.ndarray:
        """Accuracy of each model as a lone Worker (no thinker context), mean over Q."""
        accs = []
        for m in range(self.K):
            p = self.worker_prob(np.full(self.Q, m), np.zeros(self.Q))
            accs.append(p.mean())
        return np.array(accs)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
