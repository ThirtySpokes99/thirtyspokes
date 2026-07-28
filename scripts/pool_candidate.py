#!/usr/bin/env python
"""Measure a CANDIDATE pinned pool before committing it on-chain.

The pinned pool is the single most consequential owner choice, and the current one
(llama-3.1-8b + gpt-4o) measures an achievable gap of 0.0000 -- it is a two-rung capability ladder
where the cheap model is simply worse at everything, so there is no routing decision to make.

What routing actually needs is ORTHOGONAL competence: a model that beats the others at something,
cheaply. Different labs train on different data toward different objectives, so a pool spanning many
families is the most plausible place to find it -- far more so than spanning price tiers of the same
lineage. This measures whether a candidate pool actually delivers that, using the same numbers the
validator itself computes:

  achievable gap  what a perfect per-ask router could add over randomising over the fixed pool
  nestedness      fraction of ordered model pairs where solved(a) is a SUBSET of solved(b).
                  High = a capability ladder = nothing to route.
  sole-correct    asks that exactly ONE model solves -- the raw routing signal, per model.
                  A model that is never sole-correct contributes nothing but cost and an extra
                  action for the router to get wrong.

Results are appended per slice as they land, so a kill costs one slice rather than the run.

    set -a && . ./.env && set +a
    python scripts/pool_candidate.py --slices 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.koth import reference as R
from thirtyspokes.koth.benchmarks import real_suite
from thirtyspokes.koth.epoch import current_epoch, epoch_nonce
from thirtyspokes.koth.verify import achievable_gap, row_weights_for
from thirtyspokes.subnet.chain import BittensorChain

# LATEST model in each family as of 2026-07-28, spanning 9 labs and a 333x price spread
# ($/M in -> $/M out). Ordered cheap -> expensive.
CANDIDATE = [
    "qwen/qwen3.7-flash",             # 0.03 /  0.13  Qwen        open   newest overall (07-27)
    "poolside/laguna-s-2.1",          # 0.10 /  0.20  Poolside    code-specialist, very cheap
    "deepseek/deepseek-v4-flash",     # 0.14 /  0.28  DeepSeek    open
    "kwaipilot/kat-coder-air-v2.5",   # 0.15 /  0.60  Kwaipilot   code-specialist, cheap
    "minimax/minimax-m3",             # 0.30 /  1.20  MiniMax     open   (m3 supersedes m2.7)
    "deepseek/deepseek-v4-pro",       # 0.43 /  0.87  DeepSeek    open   top tier, still cheap
    "openai/gpt-5.6-luna",            # 0.50 /  3.00  OpenAI      newest family, cheap tier
    "z-ai/glm-5.2",                   # 0.77 /  2.42  GLM         open
    "openai/gpt-5.6-terra",           # 1.25 /  7.50  OpenAI      newest family, mid tier
    "google/gemini-3.6-flash",        # 1.50 /  7.50  Google      newest (07-21)
    "x-ai/grok-4.5",                  # 2.00 /  6.00  xAI         newest (07-08)
    "moonshotai/kimi-k3",             # 3.00 / 15.00  Moonshot    newest (07-16)
]
# THE DELIBERATE INCLUSION: dedicated CODE SPECIALISTS (poolside/laguna-s-2.1 at $0.10/M,
# kwaipilot/kat-coder-air-v2.5 at $0.15/M). The ranked benchmark is code, and a purpose-built coder
# that beats general frontier models on code at a twentieth of their price is the textbook
# orthogonal-competence candidate -- the exact thing every measurement in this project found missing.
# No pool measured here has ever contained one: they were all general models at different price
# tiers, which is precisely why they were nested capability ladders with nothing to route.
#
# ANTHROPIC IS EXCLUDED by owner decision -- expensive for its performance relative to peers. Note
# the mechanism does not need the exclusion to protect itself: the scalar evaluates both baselines at
# the price actually paid, so a router reaching for $10/$50 tokens loses headroom automatically. The
# exclusion instead buys a real operational saving -- every model in the pool costs the OWNER
# n_per_bench calls per epoch forever in the reference build, so a family that would rarely be
# chosen is a permanent line item. Dropping the priciest model cuts that bill materially.

RESULTS = os.environ.get("RESULTS", "/root/pool_candidate.jsonl")


def nestedness(S: np.ndarray) -> float:
    B = S >= 0.5
    pairs = nested = 0
    for i in range(B.shape[1]):
        for j in range(B.shape[1]):
            if i == j:
                continue
            pairs += 1
            if not (B[:, i] & ~B[:, j]).any():
                nested += 1
    return nested / max(pairs, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="measure a candidate pinned pool")
    p.add_argument("--slices", type=int, default=3)
    p.add_argument("--n-per-bench", type=int, default=8)
    p.add_argument("--deadline-s", type=float, default=900)
    p.add_argument("--models", help="comma-separated override")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else CANDIDATE
    cfg = config.LiveConfig()
    cfg.require_key()
    chain = BittensorChain(int(os.environ["NETUID"]), os.environ["OWNER_WALLET"],
                           os.environ.get("NETWORK", "test"),
                           hotkey=os.environ.get("OWNER_HOTKEY1", "default"))
    backend = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=180.0, max_retries=2,
                                price_fn=config.price_for)
    suite = real_suite()
    weights = {b.name: b.weight for b in suite}
    cur = current_epoch(chain)

    print(f"{len(models)} models x {args.n_per_bench} asks x {args.slices} slices\n", flush=True)
    tot_num = tot_den = spend = 0.0
    solved = np.zeros(len(models))
    sole = np.zeros(len(models))
    rows_all = 0

    for ep in range(cur - args.slices, cur):
        nonce = epoch_nonce(ep, chain.beacon(ep))
        rec = R.build(suite, epoch=ep, nonce=nonce, n_per_bench=args.n_per_bench, models=models,
                      backend=backend, reasoning={"effort": "low"}, workers=16,
                      deadline_s=args.deadline_s,
                      progress=lambda d, t: print(f"    {d}/{t}", flush=True) if d % 24 == 0 else None)
        S = np.asarray(rec["scores"], float)
        C = np.asarray(rec["costs"], float)
        if not len(S):
            print(f"  epoch {ep}: no complete rows", flush=True)
            continue
        w = row_weights_for(rec["benchmarks"], weights)
        gap = achievable_gap(rec["scores"], rec["costs"], w)
        tot_num += gap * len(S)
        tot_den += len(S)
        spend += float(C.sum())
        B = S >= 0.5
        solved += B.sum(axis=0)
        sole += (B & (B.sum(axis=1, keepdims=True) == 1)).sum(axis=0)
        rows_all += len(S)
        print(f"  epoch {ep}: gap {gap:.4f}  nestedness {nestedness(S):.3f}  "
              f"solved-by-anyone {int(B.any(axis=1).sum())}/{len(S)}", flush=True)
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps({"epoch": ep, "gap": gap, "models": models,
                                 "scores": S.tolist(), "costs": C.tolist()}) + "\n")

    if not tot_den:
        print("\nNO DATA")
        sys.exit(1)
    pooled = tot_num / tot_den
    print(f"\nPOOLED achievable gap = {pooled:.4f} over {int(tot_den)} asks   (spend ${spend:.2f})")
    print(f"current pinned pool measured 0.0000; validator floor is 0.05\n")
    print(f"{'model':34s} {'acc':>6s} {'sole':>6s}")
    for i, m in enumerate(models):
        print(f"  {m:32s} {solved[i]/rows_all:6.3f} {int(sole[i]):6d}")
    contributors = [models[i] for i in range(len(models)) if sole[i] > 0]
    print(f"\nmodels EVER sole-correct: {len(contributors)}/{len(models)}")
    for m in contributors:
        print(f"  + {m}")
    print("\nA model never sole-correct adds only cost and an extra action to get wrong.")
    print("VERDICT: " + ("POOL HAS ROUTABLE STRUCTURE — worth pinning" if pooled >= 0.05
                         else "STILL BELOW THE 0.05 FLOOR — a broader pool did not create headroom"))


if __name__ == "__main__":
    main()
