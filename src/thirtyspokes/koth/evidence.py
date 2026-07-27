"""Evidence accumulation (docs/DESIGN.md §5b) — the scoring engine that makes 32 noisy questions/epoch
enough to rank miners and makes decoupled validators agree.

Each epoch's *verified* per-benchmark result for a FIXED artifact is pooled into one decayed binomial;
the score is the closed-form **Wilson lower bound** on the pooled counts. Pooling is valid because the
artifact is hash-bound by the on-chain commit (same agent, i.i.d. draws). An EWMA (`half_life_epochs`)
gives liveness — stop submitting and your evidence bleeds away. **Re-committing a new artifact resets
the accumulator** (its key changes), which is what makes dethroning cost real, attested epochs.

Simulation (`scripts/scoring_v2_sim.py`) vs per-epoch scoring: validator divergence ↓5.5×, crown churn
↓20×, mis-crown 71%→6%, time-to-crown monotone in the true edge. Does NOT fix best-of-N grinding
(that needs an intra-epoch commit — docs/DESIGN.md §5b).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Z_DEFAULT = 1.645          # 95% one-sided


def wilson_lcb(k: float, n: float, z: float = Z_DEFAULT) -> float:
    """Lower confidence bound of a binomial rate `k/n` — exact, monotone in `n`, and valid on the
    *decayed* (fractional) counts the accumulator holds. `n<=0` → 0.0 (no evidence)."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


EvidenceKey = tuple  # (hotkey, source_hash, weights_hash, suite_version)

# A pooled frontier is only meaningful if it describes most of the evidence it is dividing. The
# numerator (`q_lcb`) pools EVERY scored epoch, but the frontier can only pool the epochs the owner
# actually published a reference for — so one referenced epoch out of two hundred would otherwise set
# the baseline for all of them. Below this fraction we decline to score frontier-relatively at all.
MIN_REF_COVERAGE = 0.5


@dataclass
class Evidence:
    """Decayed per-benchmark binomial counts + cost for one artifact, plus the pooled frontier."""
    n: dict[str, float] = field(default_factory=dict)      # bench -> decayed task count
    k: dict[str, float] = field(default_factory=dict)      # bench -> decayed correct count
    cost: dict[str, float] = field(default_factory=dict)   # bench -> decayed USD
    # --- the pooled pool-reference frontier (only epochs a reference covered) ---
    # Decayed on the SAME schedule as the counts above, so the baseline a miner is measured against
    # ages out at the same rate as the accuracy it is measured on.
    ref_n: float = 0.0        # decayed task count with reference coverage
    ref_z: float = 0.0        # Σ λ·n_e·zero_e    (featureless baseline at that epoch's price)
    ref_o: float = 0.0        # Σ λ·n_e·oracle_e  (per-ask oracle at that epoch's price)

    def decay(self, lam: float) -> None:
        for d in (self.n, self.k, self.cost):
            for b in list(d):
                d[b] *= lam
        self.ref_n *= lam
        self.ref_z *= lam
        self.ref_o *= lam

    def add(self, bench: str, n: float, k: float, cost: float) -> None:
        self.n[bench] = self.n.get(bench, 0.0) + n
        self.k[bench] = self.k.get(bench, 0.0) + k
        self.cost[bench] = self.cost.get(bench, 0.0) + cost

    def add_reference(self, n: float, zero: float, oracle: float) -> None:
        """Bank one epoch's frontier, weighted by the slice size it was measured over."""
        self.ref_n += n
        self.ref_z += n * zero
        self.ref_o += n * oracle

    def achievable_gap(self) -> float:
        """`oracle − zero`, pooled: how much headroom routing could POSSIBLY capture on this
        traffic. It is the denominator of the router scalar, and on its own it is the honest
        read on whether the suite is worth competing over — a saturated benchmark drives it to
        zero no matter how good the miners are."""
        return (self.ref_o - self.ref_z) / self.ref_n if self.ref_n > 0 else 0.0

    def headroom_lcb(self, weights: dict[str, float], *, min_gap: float,
                     z: float = Z_DEFAULT) -> float | None:
        """Frontier-relative headroom on the POOLED evidence, lower-bounded. None -> not scorable.

        `(q_lcb − zero) / (oracle − zero)` on the pooled quantities. Three properties matter:

        * IT KEEPS THE CONFIDENCE BOUND. The numerator is the same Wilson lower bound the legacy
          scalar uses, so the small-sample protection is not traded away for frontier-relativity:
          8/8 in one epoch still reads 0.747, not 1.0, and a lucky slice still cannot buy a crown.
          The denominator is measured by the OWNER, not the miner, so it is a known constant here
          and dividing by it preserves the bound.
        * IT POOLS THE RATIO, NOT THE RATIOS. Summing numerator and denominator separately across
          epochs is what makes a near-saturated epoch harmless: it contributes little to BOTH sums
          instead of contributing one wild ratio. A mean-of-per-epoch-ratios does the opposite.
        * IT REFUSES DEGENERATE TRAFFIC. When the pooled achievable gap is below `min_gap` the
          denominator is smaller than the noise in the numerator, and the ratio would amplify slice
          noise into a ranking. Returning None hands the decision back to the caller (which falls
          back to the absolute scalar) rather than inventing a number.
        """
        # Coverage is measured against the RANKED task count, not every task. The pool reference
        # only ever measures ranked benchmarks — a weight-0 eligibility floor carries row weight 0
        # in both frontiers, so measuring the pool on it would bill the owner for rows the scalar
        # discards. Comparing against `total_n()` therefore compares 8 ranked asks to 24 total on
        # the live suite (1 ranked + 2 floors), reads 0.33, and silently refuses to score
        # frontier-relatively FOREVER — the router scalar would never engage in production, which
        # is the exact failure this whole rework existed to remove.
        ranked_n = sum(self.n.get(b, 0.0) for b, w in weights.items() if w > 0)
        if self.ref_n <= 0 or ranked_n <= 0:
            return None
        if self.ref_n < MIN_REF_COVERAGE * ranked_n:
            return None                        # frontier describes too little of the evidence
        gap = self.achievable_gap()
        if gap < min_gap:
            return None                        # saturated traffic: nothing to measure
        return (self.q_lcb(weights, z) - self.ref_z / self.ref_n) / gap

    def q_lcb(self, weights: dict[str, float], z: float = Z_DEFAULT) -> float:
        """Σ_b w_b · wilson_lcb(k_b, n_b) — the accumulated reign scalar (bounded [0,1])."""
        return sum(w * wilson_lcb(self.k.get(b, 0.0), self.n.get(b, 0.0), z) for b, w in weights.items())

    def acc(self, weights: dict[str, float]) -> float:
        """Σ_b w_b · (k_b/n_b) — the accumulated point accuracy (display + per-benchmark floor)."""
        return sum(w * (self.k[b] / self.n[b] if self.n.get(b) else 0.0) for b, w in weights.items())

    def bench_acc(self, bench: str) -> float:
        return self.k[bench] / self.n[bench] if self.n.get(bench) else 0.0

    def total_n(self) -> float:
        return sum(self.n.values())

    def cost_per_task(self) -> float:
        tot = self.total_n()
        return sum(self.cost.values()) / tot if tot else 0.0


class EvidenceStore:
    """Per-artifact accumulators, keyed by `(hotkey, source_hash, weights_hash, suite_version)`.
    A hotkey that commits a NEW artifact (or a suite rotation) mints a fresh key → `n=0`, and the old
    accumulator is dropped (dethroning must be re-earned). EWMA decay = `0.5 ** (1/half_life)`."""

    def __init__(self, half_life_epochs: float = 200.0):
        self.lam = 0.5 ** (1.0 / half_life_epochs)
        self._ev: dict[EvidenceKey, Evidence] = {}
        self._key_of: dict[str, EvidenceKey] = {}     # hotkey -> its current key (re-commit detector)

    def decay_all(self) -> None:
        for e in self._ev.values():
            e.decay(self.lam)

    def for_artifact(self, hotkey: str, source_hash: str, weights_hash: str,
                     suite_version: str) -> Evidence:
        key = (hotkey, source_hash, weights_hash, suite_version)
        prev = self._key_of.get(hotkey)
        if prev is not None and prev != key:
            self._ev.pop(prev, None)                  # re-commit / suite rotation → drop old evidence
        self._key_of[hotkey] = key
        return self._ev.setdefault(key, Evidence())

    def drop(self, hotkey: str) -> None:
        """Deregistered hotkey → forget its evidence."""
        key = self._key_of.pop(hotkey, None)
        if key is not None:
            self._ev.pop(key, None)

    # --- restart-safety -------------------------------------------------------
    def snapshot(self) -> dict:
        return {"half_life_lam": self.lam,
                "ev": [{"key": list(k), "n": e.n, "k": e.k, "cost": e.cost,
                        "ref_n": e.ref_n, "ref_z": e.ref_z, "ref_o": e.ref_o}
                       for k, e in self._ev.items()],
                "key_of": {hk: list(k) for hk, k in self._key_of.items()}}

    def restore(self, snap: dict) -> None:
        self.lam = snap.get("half_life_lam", self.lam)
        self._ev = {tuple(r["key"]): Evidence(dict(r["n"]), dict(r["k"]), dict(r["cost"]),
                                              float(r.get("ref_n", 0.0)),
                                              float(r.get("ref_z", 0.0)),
                                              float(r.get("ref_o", 0.0)))
                    for r in snap.get("ev", [])}
        self._key_of = {hk: tuple(k) for hk, k in snap.get("key_of", {}).items()}
