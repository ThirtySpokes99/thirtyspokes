"""Scoring, sanity gate, and anti-copy fingerprint for orchestrator policies.

Objective (ARCHITECTURE.md §8): quality - lam*cost, cost summing every pool call
the workflow makes. Fingerprint: a policy's first-turn action distribution over a
fixed probe set (the v4 output-distribution dedup, lifted to the action space).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..orchestrator import N_ROLES, N_STATE, VERIFIER
from ..validate import tv_distance
from .artifact import PolicyArtifact
from .policy import run_policy
from .protocol import CompetitionParams, MIN_ACTION_ENTROPY


@dataclass
class Score:
    quality: float       # mean correctness
    norm_cost: float     # mean cost / (max_turns * max_price)
    objective: float     # quality - lam*norm_cost


def score(art: PolicyArtifact, backend, params: CompetitionParams) -> Score:
    r = run_policy(art, backend, params.max_turns)
    cost_ref = params.max_turns * backend.price.max()
    q = float(r.correct.mean())
    nc = float((r.cost / cost_ref).mean())
    return Score(quality=q, norm_cost=nc, objective=q - params.lam * nc)


def _first_turn_action_dist(art: PolicyArtifact, probe: np.ndarray, k: int) -> np.ndarray:
    """(n_probe, K*N_ROLES) softmax action distribution on the first turn."""
    feats = np.hstack([probe, np.zeros((len(probe), N_STATE), dtype=np.float32)])
    logits = art.head().logits(art.theta, feats).reshape(len(probe), k, N_ROLES)
    logits[:, :, VERIFIER] = -1e9          # verifier invalid before any answer
    flat = logits.reshape(len(probe), k * N_ROLES)
    flat = flat - flat.max(axis=1, keepdims=True)
    e = np.exp(flat)
    return e / e.sum(axis=1, keepdims=True)


def fingerprint(art: PolicyArtifact, probe: np.ndarray) -> np.ndarray:
    return _first_turn_action_dist(art, probe, art.k_models)


def sanity_gate(art: PolicyArtifact, probe: np.ndarray) -> tuple[bool, str]:
    dist = _first_turn_action_dist(art, probe, art.k_models)
    if not np.all(np.isfinite(dist)):
        return False, "non_finite_output"
    choice = dist.argmax(axis=1)
    frac = np.bincount(choice, minlength=dist.shape[1]) / len(choice)
    frac = frac[frac > 0]
    entropy = float(-(frac * np.log(frac)).sum())
    if entropy < MIN_ACTION_ENTROPY:
        return False, f"degenerate_policy:H={entropy:.3f}"
    return True, "ok"


def is_duplicate(fp_a: np.ndarray, fp_b: np.ndarray, threshold: float) -> bool:
    return tv_distance(fp_a, fp_b) < threshold
