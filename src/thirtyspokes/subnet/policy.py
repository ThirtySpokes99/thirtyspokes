"""Baseline policy + training (the enablement kit's reference miner).

Miners may replace this with any policy of any capacity, as long as it produces
a PolicyArtifact that loads into the fixed engine. This baseline trains a
RouterHead orchestrator with CMA-ES and is what ships so day-1 miners have a
competitive-but-beatable starting point.
"""

from __future__ import annotations

import numpy as np

from ..orchestrator import run_episodes, train_orchestrator
from .artifact import PolicyArtifact
from .protocol import CompetitionParams


def train_baseline(pool, params: CompetitionParams, hidden: int = 32,
                   iters: int = 140, seed: int = 0, miner_id: str = "baseline",
                   verbose: bool = False) -> PolicyArtifact:
    """Train a policy offline on freshly sampled tasks over the sanctioned pool."""
    backend = pool.sample_tasks(epoch_seed=1000 + seed)     # miner's own training tasks
    trained = train_orchestrator(backend, lam=params.lam, hidden=hidden,
                                 iters=iters, max_turns=params.max_turns,
                                 seed=seed, verbose=verbose)
    return PolicyArtifact(embed_dim=pool.embed_dim, k_models=pool.k_models,
                          hidden=hidden, theta=trained.theta, miner_id=miner_id)


def run_policy(art: PolicyArtifact, backend, max_turns: int):
    """Execute a policy artifact through the fixed engine on a task backend."""
    return run_episodes(art.head(), art.theta, backend, max_turns)
