"""The serving-side routing decision: prompt -> which pool model answers it.

This is the same decision the subnet scores, run for real traffic instead of for a benchmark. It
reuses the harness rather than reimplementing it (`harness.encode` -> `load_head` ->
`distribution().argmax()` -> `pool_models()`), because a head that scores well on-chain and a head
that serves here must be the SAME function of the prompt. Any divergence would make the subnet's
ranking a ranking of something the product does not do.

WHAT THIS DELIBERATELY DOES NOT DO. It does not grade, verify, or escalate. On exactly-gradable
traffic the measurements say cheapest-first + mechanical verification captures nearly all the
available value (with a perfect verifier, entering at the cheapest rung scores 0.9305 against a
0.9319 oracle — `docs/ROUTING_MEASUREMENTS.md (removed with v2, 2026-09-07)`), and that is a different product from a learned
router. Beta serves the learned router so its real-traffic value can be MEASURED against the
baseline; it does not assume the answer.

THE BASELINE IS THE POINT. Every request records what a fixed baseline policy would have been, and
the service can be run in `baseline` mode over the same traffic. The product claim is a comparison,
so the comparison is instrumented from the first request rather than retrofitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..koth import harness
from ..koth.harness import ROUTING_POOL, load_head, pool_models

# Cheapest by REALISED cost on the measured pool matrix, and the only rung that returned a non-empty
# answer on 100% of measured code problems (`docs/ROUTER_V2.md (removed with v2, 2026-09-07)` §2a-ter). Both properties matter for
# a baseline: it has to be the thing a customer would otherwise call, and it has to always answer.
# Not derived from `ROUTING_POOL` order — that is sorted by ADVERTISED blended price, which ranks
# differently from realised cost (Spearman 0.607) because completion volume varies 27x across the
# pool. Picking the ladder's rung 0 as "the cheap one" would pick a model that is not the cheapest.
DEFAULT_BASELINE = "openai/gpt-5.6-luna"

# The two serving tiers, and why exactly two. On the measured 112-ask matrix only these rungs
# produced a well-formed answer on 100% of asks — the other five are empty on 39-74% of code asks
# or blow the latency budget, and a tier that cannot answer is not a tier. Within the viable pair
# the roles assign themselves: luna is the cheapest model in the pool by REALISED cost ($0.0008/ask)
# and gemini-3.6-flash is best-single (0.7589). This is also where the field converged from the
# other direction — RouteLLM routes between exactly a weak and a strong tier — but here the pair is
# picked by measurement, not convention.
TIER_CHEAP = "openai/gpt-5.6-luna"
TIER_STRONG = "google/gemini-3.6-flash"


@dataclass(frozen=True)
class Decision:
    model: str
    rung: int
    source: str          # "head" | "baseline" | "tiered" | "tiered-escalate" | "tiered-explore"
    # Models to attempt IN ORDER: serving tries ladder[0], falls through on empty/error/timeout,
    # and 502s only when the whole ladder failed. Encoding the fallback here (instead of hardcoding
    # "then baseline" in the app) is what lets a tiered policy fall back to a DIFFERENT strong
    # model while the head policy falls back to the baseline, through one serving path.
    ladder: tuple[str, ...] = ()
    p_escalate: float | None = None   # the predictor's probability, logged for the training loop

    @property
    def routed(self) -> bool:
        return self.source != "baseline"


class RoutingPolicy:
    """Chooses a pool model per prompt, from a miner's head or from the fixed baseline.

    `weights=None` is `baseline` mode: the control arm of the A/B, and also the honest fallback for
    the first ten epochs of a fresh subnet when no head has cleared the crown floor
    (`koth/standings.py`) and there is nothing legitimate to serve.
    """

    def __init__(self, weights: bytes | None = None, *,
                 baseline: str = DEFAULT_BASELINE,
                 servable: set[str] | None = None):
        self.pool = pool_models()
        if baseline not in self.pool:
            raise ValueError(f"baseline {baseline!r} is not in ROUTING_POOL")
        self.baseline = baseline
        self.baseline_rung = self.pool.index(baseline)
        # None = serve the whole action space. Restricting it is available but NOT the default: the
        # head has one output per pool model, so an allowlist silently converts most of its choices
        # into baseline calls and the router looks broken when it is only being overridden. Let the
        # measured failures show up as fallbacks in `/stats` instead, where they are attributable.
        self.servable = servable
        self._head = None
        if weights is not None:
            self._head = load_head(weights, k=len(self.pool))

    def decide(self, prompt: str) -> Decision:
        if self._head is None:
            return Decision(self.baseline, self.baseline_rung, "baseline",
                            ladder=(self.baseline,))
        head, theta = self._head
        # Called through the module, not a bound import: the pinned encoder is the one piece of the
        # decision that needs a real model on disk, and tests substitute it here.
        features = harness.encode([prompt])
        rung = int(head.distribution(theta, features).argmax(axis=1)[0])
        model = self.pool[rung]
        if self.servable is not None and model not in self.servable:
            return Decision(self.baseline, self.baseline_rung, "baseline",
                            ladder=(self.baseline,))
        ladder = (model,) if model == self.baseline else (model, self.baseline)
        return Decision(model, rung, "head", ladder=ladder)


class TieredPolicy:
    """The pipeline's default policy: enter cheap, escalate when failure is predicted or observed.

    This is the architecture the measurements converge on. On the real matrix the cheap tier
    (`gpt-5.6-luna`) is ON the cost-quality frontier — no 7-way head measured anywhere, including
    the two earning on mainnet, beats entering there. The only decision with measured signal is
    binary: WILL THE CHEAP TIER FAIL THIS ASK (held-out AUC 0.8355/0.812 on production-shaped
    traffic). So:

      predictor says fine  -> ladder (cheap, strong)  — worst case: one wasted cheap call
      predictor says fail  -> ladder (strong, cheap)  — skip the cheap call's cost AND latency
      no predictor yet     -> ladder (cheap, strong)  — cheap-first + escalate-on-failure, which is
                              itself on the frontier; the predictor only sharpens it

    `explore_rate` keeps a fraction of predicted-escalate asks entering cheap anyway. Not a tuning
    knob: the training loop's labels are "what happened when the cheap tier was tried", so a
    predictor that always skips the cheap tier would starve its own successor of exactly the labels
    that could correct it. Exploration is the cost of staying trainable.
    """

    def __init__(self, escalate_weights: bytes | None = None, *,
                 threshold: float = 0.5, explore_rate: float = 0.05, rng=None):
        from . import escalate as E
        if TIER_CHEAP not in pool_models() or TIER_STRONG not in pool_models():
            raise ValueError("serving tiers must be pool models")
        self.threshold = threshold
        self.explore_rate = explore_rate
        self._rng = rng if rng is not None else np.random.default_rng()
        self._predict = E.load_predictor(escalate_weights) if escalate_weights else None
        self._pool = pool_models()

    def decide(self, prompt: str) -> Decision:
        features = harness.encode([prompt])
        p = float(self._predict(features)[0]) if self._predict is not None else None
        if p is not None and p >= self.threshold:
            if self._rng.random() < self.explore_rate:
                return Decision(TIER_CHEAP, self._pool.index(TIER_CHEAP), "tiered-explore",
                                ladder=(TIER_CHEAP, TIER_STRONG), p_escalate=p)
            return Decision(TIER_STRONG, self._pool.index(TIER_STRONG), "tiered-escalate",
                            ladder=(TIER_STRONG, TIER_CHEAP), p_escalate=p)
        return Decision(TIER_CHEAP, self._pool.index(TIER_CHEAP), "tiered",
                        ladder=(TIER_CHEAP, TIER_STRONG), p_escalate=p)


def price_of(model: str) -> tuple[float, float]:
    """Advertised ($/M input, $/M output). Reporting only — served cost is the provider's own
    metered figure, never this table."""
    for m, pin, pout in ROUTING_POOL:
        if m == model:
            return pin, pout
    return 0.0, 0.0
