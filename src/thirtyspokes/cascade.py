"""v1.5 cascade action space — try cheap, verify, escalate on reject.

The cascade is the headline cost lever (FrugalGPT / RouteLLM's money-maker, and
most of TinyRouter's verifier loop). Crucially it stays *precomputable*: given
the cached per-model correctness, cost, and pinned-verifier verdicts, the outcome
of "start at rung r and escalate up the cost ladder until the verifier accepts"
is fully determined. So the whole v1.5 competition reduces to single-shot over a
*derived* cache whose K actions are the K starting rungs — every existing
component (oracle, scorer, trainer, reign) works on it unchanged.

Execution model for a start rung r (rungs ordered cheap -> expensive):
  invoke model at rung r; if the pinned verifier accepts its answer, bank it and
  stop; else escalate to r+1; repeat; at the top rung, bank whatever comes out.
Cost = sum of every model invoked. Correct = correctness of the banked answer.
A perfect verifier makes cheap-first free; a noisy verifier makes over-eager
cheap starts risky — so the router must learn *where on the ladder to enter*.
"""

from __future__ import annotations

import numpy as np

from .cache import OutcomeCache, PoolModel


def to_cascade_cache(cache: OutcomeCache) -> OutcomeCache:
    """Derive a (Q, K) cascade-action cache: action r = 'start at cost-rung r'.

    Requires verifier_ok to be present in the source cache.
    """
    if cache.verifier_ok is None:
        raise ValueError("cascade requires a pinned-verifier cache (verifier_ok is None)")

    Q, K = cache.Q, cache.K
    order = np.argsort([m.price_per_token for m in cache.models])  # cheap -> expensive
    correct = cache.correct
    cost = cache.cost
    vok = cache.verifier_ok

    casc_correct = np.zeros((Q, K), dtype=bool)
    casc_cost = np.zeros((Q, K), dtype=np.float32)

    for r in range(K):
        banked = np.zeros(Q, dtype=bool)
        summed = np.zeros(Q, dtype=np.float32)
        done = np.zeros(Q, dtype=bool)
        rungs = order[r:]
        for pos, m in enumerate(rungs):
            active = ~done
            summed[active] += cost[active, m]
            is_last = pos == len(rungs) - 1
            accept = vok[:, m]
            newly = active & (accept | is_last)   # stop on accept, or forced at top rung
            banked[newly] = correct[newly, m]
            done[newly] = True
        casc_correct[:, r] = banked
        casc_cost[:, r] = summed

    # pseudo-models label the starting rung, priced by their realized mean cost
    casc_models = [
        PoolModel(name=f"start@{cache.models[order[r]].name}",
                  price_per_token=float(casc_cost[:, r].mean()),
                  role=f"rung{r}")
        for r in range(K)
    ]
    return OutcomeCache(
        models=casc_models,
        embeddings=cache.embeddings,
        correct=casc_correct,
        cost=casc_cost,
        verifier_ok=None,
        domain=cache.domain,
        question_ids=cache.question_ids,
        # Inherit the base cache's lambda normalizer (review M2): a cascade cache
        # normalized by its own (larger) max would flatter cascades in any
        # cross-action-space comparison. Cascade norm_cost may exceed 1.0 — that
        # is honest: an escalation chain can cost more than the priciest single call.
        cost_norm=cache.cost_max,
    )
