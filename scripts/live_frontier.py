#!/usr/bin/env python
"""Live real-pool frontier on GSM8K: run every pool tier on the SAME tasks, then
show the single-model quality/cost frontier AND the composition ceiling (per-sample
oracle over the pool). Needs OPENROUTER_API_KEY. Costs real money — keep n modest.
"""
from __future__ import annotations

import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from thirtyspokes.eval import config, math_tasks
from thirtyspokes.eval.agent import ReferenceAgent
from thirtyspokes.gateway.gateway import MeteringGateway, OpenRouterBackend
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.gateway.verify import verify_submission

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
LAM, COST_REF = 0.15, 0.05
_c = itertools.count()


def nonce():
    return f"n{next(_c)}"


def main():
    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, price_fn=config.price_for)
    tasks = math_tasks.load_gsm8k(N, seed=7)
    tiers = list(cfg.pool)
    correct = np.zeros((N, len(tiers)))
    cost = np.zeros((N, len(tiers)))

    for ti, tier in enumerate(tiers):
        agent = ReferenceAgent(Signer(), cfg.pool[tier], max_tokens=700)
        gw = MeteringGateway(be, {agent.hotkey})
        ok = 0
        for qi, task in enumerate(tasks):
            try:
                sub, rec = agent.solve(gw, task.task_id, task.prompt, nonce)
                v = verify_submission(sub, rec, gw.public_hex, task.gold, math_tasks.grade,
                                      lam=LAM, cost_ref=COST_REF)
                correct[qi, ti] = v.quality
                cost[qi, ti] = v.cost_usd
                ok += int(v.quality)
            except Exception as e:
                print(f"    [{tier} q{qi}] error: {type(e).__name__}: {str(e)[:60]}")
        print(f"  {tier:9s} {cfg.pool[tier]:32s} acc={correct[:,ti].mean():.3f} "
              f"cost=${cost[:,ti].sum():.4f}  ({ok}/{N})")

    print("\n" + "=" * 66)
    print(f"REAL GSM8K frontier  (n={N} tasks, pool of {len(tiers)} real models)")
    print("=" * 66)
    accs = correct.mean(axis=0)
    best = int(accs.argmax())
    print(f"  best single model      : {tiers[best]} acc={accs[best]:.3f}")
    # composition ceiling
    any_correct = (correct.max(axis=1) > 0).mean()
    # cheapest-correct oracle cost
    q = np.arange(N)
    ok_mask = correct > 0
    order = np.argsort(cost.mean(axis=0))  # cheap->exp
    cc_cost = 0.0
    cc_correct = 0
    for i in range(N):
        chosen = None
        for k in order:
            if ok_mask[i, k]:
                chosen = k; break
        if chosen is None:
            chosen = order[-1]
        cc_cost += cost[i, chosen]; cc_correct += int(correct[i, chosen])
    print(f"  per-sample oracle acc  : {any_correct:.3f}  (composition ceiling)")
    print(f"  composition gap        : {(any_correct-accs[best])*100:+.1f} pts over best single")
    print(f"  cheapest-correct oracle: acc={cc_correct/N:.3f} cost=${cc_cost:.4f} "
          f"(vs best-single cost ${cost[:,best].sum():.4f})")
    print(f"\n  total spend this run   : ${cost.sum():.4f}")


if __name__ == "__main__":
    main()
