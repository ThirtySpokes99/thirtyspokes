#!/usr/bin/env python
"""THE decisive experiment: does composition beat the best single model on HARD,
verifiable tasks (MATH-500 level 4-5)? Runs each pool tier live once, derives the
composition ceiling (per-sample oracle) + majority-vote from cached answers.

Uses a DIVERSE, clean (non-thinking) pool — thinking models (e.g. gemini-2.5-pro)
truncate before emitting \\boxed and must be excluded or given a reasoning budget.
Detects truncation (finish_reason=length) and flags it rather than grading 0.
Needs OPENROUTER_API_KEY. Costs real money — keep n modest.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from openai import OpenAI

from thirtyspokes.eval import config, hard_tasks

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MIN_LEVEL, MAX_TOKENS = 4, 2048

# Diverse, clean-output pool (4 different providers -> more independent errors).
POOL = {
    "cheap": "meta-llama/llama-3.1-8b-instruct",
    "mid": "deepseek/deepseek-v3.2-exp",
    "strong": "openai/gpt-4o-2024-11-20",
    "frontier": "anthropic/claude-opus-4.7",
}


def main():
    cfg = config.LiveConfig(); cfg.require_key()
    cli = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=60.0, max_retries=2)
    tasks = hard_tasks.load_math(N, min_level=MIN_LEVEL, seed=11)
    tiers = list(POOL); T = len(tiers)
    ans = {}
    correct = np.zeros((N, T)); cost = np.zeros((N, T)); trunc = np.zeros((N, T))

    print(f"MATH-500 (level>={MIN_LEVEL}), n={N}\n  pool={list(POOL.values())}\n")
    for ti, tier in enumerate(tiers):
        print(f"  running {tier}...", end="", flush=True)
        for qi, task in enumerate(tasks):
            try:
                r = cli.chat.completions.create(
                    model=POOL[tier], messages=[{"role": "user", "content": task.prompt}],
                    max_tokens=MAX_TOKENS, temperature=0.0,
                    extra_body={"usage": {"include": True}})
                txt = r.choices[0].message.content or ""
                ans[(qi, ti)] = txt
                correct[qi, ti] = hard_tasks.grade_hard(txt, task.gold)
                trunc[qi, ti] = 1.0 if r.choices[0].finish_reason == "length" else 0.0
                u = r.usage
                cost[qi, ti] = float((getattr(u, "model_extra", {}) or {}).get("cost") or 0.0)
                print("." if correct[qi,ti] else ("x" if not trunc[qi,ti] else "T"), end="", flush=True)
            except Exception as e:
                print(f"    [{tier} q{qi}] {type(e).__name__}: {str(e)[:50]}")
        print()
        flag = f"  ⚠ {int(trunc[:,ti].sum())} truncated" if trunc[:, ti].any() else ""
        print(f"  {tier:9s} {POOL[tier]:30s} acc={correct[:,ti].mean():.3f} "
              f"cost=${cost[:,ti].sum():.4f}{flag}")

    print("\n" + "=" * 66)
    print(f"COMPOSITION vs BEST-SINGLE on hard MATH (n={N})")
    print("=" * 66)
    accs = correct.mean(0); best = int(accs.argmax())
    oracle = (correct.max(1) > 0).mean()
    print(f"  per-model accuracy   : " + ", ".join(f"{t}={accs[i]:.2f}" for i, t in enumerate(tiers)))
    print(f"  best single model    : {tiers[best]} acc={accs[best]:.3f} cost=${cost[:,best].sum():.4f}")
    print(f"  per-sample oracle     : acc={oracle:.3f}  (composition CEILING)")
    print(f"  composition gap       : {(oracle-accs[best])*100:+.1f} pts achievable over best-single")

    vote_idx = list(range(1, T))  # mid, strong, frontier
    vc = np.array([hard_tasks.grade_hard(
        hard_tasks.majority_vote([ans.get((qi, ti), "") for ti in vote_idx]), tasks[qi].gold)
        for qi in range(N)])
    vcost = cost[:, vote_idx].sum()
    print(f"  majority-vote(top-3)  : acc={vc.mean():.3f} cost=${vcost:.4f}  "
          f"({(vc.mean()-accs[best])*100:+.1f} pts vs best-single, {vcost/max(cost[:,best].sum(),1e-9):.1f}x cost)")

    order = np.argsort(cost.mean(0))
    cc = sum(next((cost[i, k] for k in order if correct[i, k]), cost[i, order[-1]]) for i in range(N))
    ratio = cost[:, best].sum() / max(cc, 1e-9)
    print(f"  cheapest-correct route: acc={oracle:.3f} cost=${cc:.4f}  "
          f"({'{:.1f}x cheaper'.format(ratio) if ratio>=1 else '{:.1f}x MORE expensive'.format(1/ratio)} than best-single)")
    print(f"\n  total spend this run: ${cost.sum():.4f}")


if __name__ == "__main__":
    main()
