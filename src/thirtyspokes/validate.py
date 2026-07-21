"""Anti-copy fingerprinting + pre-eval sanity gate (architecture 6 / eval-hardening).

Fingerprint: a router's soft output distributions over a fixed private probe set
of embeddings. Two heads whose distributions are near-identical across the probe
are duplicates — the earliest commit is scored, later copies get zero.

Critical caveat (small action space): with K=3-5 choices, *independently trained*
honest routers legitimately agree on most argmax decisions. So we compare the
FULL soft distributions via mean total-variation distance, never argmax
agreement, and set the duplicate threshold below the natural TV distance between
independent routers (measured by calibrate_threshold, run in Phase 0).

Sanity gate: cheap checks that keep degenerate/oversized heads out of scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .router import PARAM_CAP, RouterHead


def fingerprint(head: RouterHead, theta: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """(n_probe, K) soft distributions, the head's behavioral signature."""
    return head.distribution(theta, probe)


def tv_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Mean total-variation distance between two fingerprints (in [0, 1])."""
    return float(0.5 * np.abs(fp_a - fp_b).sum(axis=1).mean())


def is_duplicate(fp_a: np.ndarray, fp_b: np.ndarray, threshold: float) -> bool:
    return tv_distance(fp_a, fp_b) < threshold


def calibrate_threshold(fps: list[np.ndarray], margin: float = 0.5) -> dict:
    """Measure natural TV distance between independently trained routers.

    Takes precomputed fingerprints (each produced by its own head, so this is
    architecture-agnostic — miners may pick different hidden sizes). Returns the
    min/mean natural distance and a recommended duplicate threshold set safely
    below the natural minimum (threshold = min_natural * margin). Copies
    (near-zero TV) fall below it; honest independents sit above.
    """
    dists = [tv_distance(fps[i], fps[j])
             for i in range(len(fps)) for j in range(i + 1, len(fps))]
    dists = np.array(dists) if dists else np.array([1.0])
    return {
        "min_natural_tv": float(dists.min()),
        "mean_natural_tv": float(dists.mean()),
        "recommended_threshold": float(dists.min() * margin),
    }


@dataclass
class SanityResult:
    ok: bool
    reason: str


def sanity_gate(
    head: RouterHead,
    theta: np.ndarray,
    probe: np.ndarray,
    min_choice_entropy: float = 0.15,
) -> SanityResult:
    """Reject before expensive scoring: bad shape, NaNs, oversize, or degenerate.

    'Degenerate' = the head collapses to (almost) always choosing one model over
    the probe set; such a router is just a constant policy masquerading as a
    router and should not consume a scoring slot. Threshold 0.15 nats rejects
    ~97/3-or-worse choice mixes and accepts ~95/5 (review LOW: the old 0.05 let
    a 99/1 near-constant policy through). A pool-selection-entropy floor is also
    a live health metric in the architecture — same idea, applied pre-eval.
    """
    if head.n_params > PARAM_CAP:
        return SanityResult(False, f"param_cap_exceeded:{head.n_params}")
    if theta.shape != (head.n_params,):
        return SanityResult(False, "param_shape_mismatch")
    if not np.all(np.isfinite(theta)):
        return SanityResult(False, "non_finite_params")

    dist = head.distribution(theta, probe)
    if not np.all(np.isfinite(dist)):
        return SanityResult(False, "non_finite_output")

    # entropy of the *choice mix* across the probe (are all decisions one model?)
    choice = dist.argmax(axis=1)
    frac = np.bincount(choice, minlength=head.k) / len(choice)
    frac = frac[frac > 0]
    entropy = float(-(frac * np.log(frac)).sum())
    if entropy < min_choice_entropy:
        return SanityResult(False, f"degenerate_constant_policy:H={entropy:.3f}")
    return SanityResult(True, "ok")
