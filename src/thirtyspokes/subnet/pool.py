"""Sanctioned pool + task stream (ARCHITECTURE.md §5, §6 — protocol/owner-owned).

The pool (models) is owner-curated and FIXED within a season; the task stream is
freshly sampled every epoch (hardening / rotating). In synthetic mode both come
from SyntheticBackend with the pool/task seed split. A `LivePool` seam runs the
same interface over real models + real tasks via a validator-controlled gateway
(no miner network access) — not exercised here (needs API keys).
"""

from __future__ import annotations

import numpy as np

from ..synth_tinyrouter import SyntheticBackend, TinyConfig


class SanctionedPool:
    """Owner-curated pool + task sampler. Backend = execution surface for the engine."""

    def __init__(self, cfg: TinyConfig):
        self.cfg = cfg
        ref = SyntheticBackend(cfg, task_seed=cfg.seed)  # fixed-pool reference
        self.embed_dim = ref.embeddings.shape[1]
        self.k_models = ref.K
        self.names = ref.names
        self.prices = ref.price

    def sample_tasks(self, epoch_seed: int) -> SyntheticBackend:
        """Fresh eval tasks for an epoch, over the fixed pool."""
        return SyntheticBackend(self.cfg, task_seed=epoch_seed)

    def probe_states(self, n: int, seed: int = 12345) -> np.ndarray:
        """A fixed probe set of task embeddings for fingerprint/sanity checks."""
        be = SyntheticBackend(self.cfg, task_seed=seed)
        idx = np.random.default_rng(seed).choice(be.Q, min(n, be.Q), replace=False)
        return be.embeddings[idx]


class LivePool:  # pragma: no cover — production seam, needs API access
    """Real-model gateway. Implements the same backend interface as
    SyntheticBackend but each (model, role) turn is a real API call executed by
    the validator (never the miner). Wire per-role prompts + verifiable grading;
    reuse backends.OpenAICompatibleBackend for the calls."""

    def __init__(self, *_a, **_k):
        raise NotImplementedError("LivePool needs provider API access; use SanctionedPool offline")
