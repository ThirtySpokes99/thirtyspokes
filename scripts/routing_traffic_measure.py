#!/usr/bin/env python
"""Is routing learnable on traffic that actually has a difficulty spread? The decisive rerun.

`difficulty_predictability.py` found that inside a single exam set, "can the cheap model handle
this?" is predictable at AUC 0.500 — chance — and that all fourteen prior measurements used exactly
that distribution. This runs the same question on `routing_traffic.py`: a production-shaped mix,
45% trivial through 10% hard, exactly graded.

Four numbers, in the order that makes each meaningful:

  1. LEGIBILITY   held-out AUC for "the cheap model solves this", from the prompt embedding alone.
                  This is the miner's actual product. Chance here means nothing downstream matters.
  2. BAND         achievable gap on the cost-quality frontier: what a per-ask oracle adds over the
                  best single model AT ITS PRICE. Room for a router to work in.
  3. CAPTURE      what a trained router actually gets, HELD OUT. The gap between 2 and 3 is the
                  difference between value existing and value being reachable, and every prior
                  measurement died in that gap.
  4. PARSE RATE   per model, per tier. Recorded because a reasoning model that burns its token
                  budget returns an empty string that scores as WRONG, which manufactured a fake
                  result twice in this project already.

Pre-committed readings:
    AUC ~0.5                     routing is unlearnable even here; the thesis is closed for good
    AUC high, capture ~0         the signal exists but a capped head cannot use it
    AUC high, capture > 0        routing works on real traffic and the exam sets were the confound
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import the sibling traffic module

from routing_traffic import build, grade                          # noqa: E402
from thirtyspokes.eval import config                              # noqa: E402
from thirtyspokes.gateway.gateway import OpenRouterBackend        # noqa: E402

# cheap -> expensive, ~50x price spread. Routing can only pay for itself when the ladder is steep.
POOL = ["qwen/qwen3.7-flash", "deepseek/deepseek-v4-flash", "openai/gpt-5.6-luna",
        "google/gemini-3.6-flash"]
RESULTS = os.environ.get("RESULTS", "data/routing_traffic_runs.jsonl")
TIERS = ["trivial", "easy", "medium", "medium-hard", "hard"]


def run_pool(tasks, models, max_tokens, workers):
    done = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line); done[(r["model"], r["id"])] = r
            except Exception:      # noqa: BLE001 — torn last line after a kill
                pass
        print(f"resuming: {len(done)} calls on disk", flush=True)
    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=180.0, max_retries=2,
                           price_fn=config.price_for)
    work = [(m, t) for t in tasks for m in models if (m, t["id"]) not in done]
    out = open(RESULTS, "a")
    n = [0]

    def one(job):
        m, t = job
        try:
            txt, _i, _o, c = be.complete(m, [{"role": "user", "content": t["prompt"]}],
                                         {"max_tokens": max_tokens, "temperature": 0.0,
                                          "reasoning": {"effort": "low"}})
            s = str(txt)
            rec = {"model": m, "id": t["id"], "tier": t["tier"], "correct": grade(s, t),
                   "cost": float(c), "parsed": bool(s.strip()), "empty": not s.strip()}
        except Exception as e:   # noqa: BLE001 — a provider failure is data, not a crash
            rec = {"model": m, "id": t["id"], "tier": t["tier"], "correct": 0.0, "cost": 0.0,
                   "parsed": False, "empty": True, "error": type(e).__name__}
        n[0] += 1
        if n[0] % 200 == 0:
            print(f"  {n[0]}/{len(work)} calls", flush=True)
        return rec

    if work:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(one, work):
                out.write(json.dumps(rec) + "\n"); out.flush()
                done[(rec["model"], rec["id"])] = rec
    out.close()
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="routing learnability on production-shaped traffic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(POOL))
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    tasks = build(args.n, args.seed)
    print(f"{len(tasks)} tasks x {len(models)} models = {len(tasks)*len(models)} calls", flush=True)
    done = run_pool(tasks, models, args.max_tokens, args.workers)

    ids = [t["id"] for t in tasks if all((m, t["id"]) in done for m in models)]
    by_id = {t["id"]: t for t in tasks}
    S = np.array([[done[(m, i)]["correct"] for m in models] for i in ids])
    C = np.array([[done[(m, i)]["cost"] for m in models] for i in ids])

    print("\n=== 4. PARSE RATES (an unanswered ask is not a wrong answer) ===")
    for m in models:
        rates = [np.mean([done[(m, i)]["parsed"] for i in ids if by_id[i]["tier"] == tr])
                 for tr in TIERS]
        flag = "  <-- SKEWED" if max(rates) - min(rates) > 0.10 else ""
        print(f"  {m:30s} " + " ".join(f"{t}={r:.2f}" for t, r in zip(TIERS, rates)) + flag)

    print("\n=== accuracy by tier (the shape that makes routing possible) ===")
    print(f"{'model':30s} " + "".join(t.rjust(10) for t in TIERS) + f"{'all':>9s}{'$/task':>10s}")
    for k, m in enumerate(models):
        row = f"{m:30s} "
        for tr in TIERS:
            sel = [j for j, i in enumerate(ids) if by_id[i]["tier"] == tr]
            row += f"{S[sel, k].mean():10.3f}"
        print(row + f"{S[:, k].mean():9.3f}{C[:, k].mean():10.5f}")

    # 1. LEGIBILITY
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from thirtyspokes.koth import harness as H
    emb = H.encode([by_id[i]["prompt"] for i in ids])
    y = (S[:, 0] >= 0.5).astype(int)              # does the CHEAPEST model solve it?
    rng = np.random.default_rng(0); perm = rng.permutation(len(ids)); cut = int(len(ids) * 0.7)
    tr_i, te_i = perm[:cut], perm[cut:]
    clf = LogisticRegression(max_iter=2000).fit(emb[tr_i], y[tr_i])
    a = roc_auc_score(y[te_i], clf.predict_proba(emb[te_i])[:, 1])
    print(f"\n=== 1. LEGIBILITY — 'cheapest model solves it', held out ===\n  AUC = {a:.3f}"
          f"   (exam-set baseline from difficulty_predictability.py: 0.500-0.543)")

    # 2/3. BAND and CAPTURE, using the subnet's own frontier machinery
    from thirtyspokes.koth.verify import achievable_gap
    band = achievable_gap(S, C)
    cmax = C.sum(axis=1).max() or 1.0
    obj = S - args.lam * C / (C.max() or 1.0)
    best_single = max(range(len(models)), key=lambda k: obj[:, k].mean())
    # the achievable router: train on the train split to pick a model per ask, score on held-out
    yk = obj.argmax(axis=1)
    clf2 = LogisticRegression(max_iter=2000).fit(emb[tr_i], yk[tr_i])
    pick = clf2.predict(emb[te_i])
    got = obj[te_i, pick].mean()
    bs = obj[te_i, best_single].mean()
    orc = obj[te_i].max(axis=1).mean()
    cap = (got - bs) / (orc - bs) if orc - bs > 1e-9 else float("nan")
    print(f"\n=== 2/3. BAND and CAPTURE (held out, cost-aware objective) ===")
    print(f"  best-single ({models[best_single].split('/')[-1]}) = {bs:+.4f}")
    print(f"  trained router                = {got:+.4f}")
    print(f"  per-ask oracle                = {orc:+.4f}")
    print(f"  band = {orc - bs:.4f}   CAPTURED = {cap:.1%}   (achievable_gap over pool = {band:.4f})")

    print("\nVERDICT")
    if a < 0.6:
        print("  Difficulty is NOT legible even on production-shaped traffic. Routing is closed.")
    elif cap <= 0.05:
        print("  The signal is legible but a router cannot convert it: capture ~0 despite AUC\n"
              f"  {a:.3f}. The bottleneck is the pool/cost structure, not prompt legibility.")
    else:
        print(f"  ROUTING WORKS HERE: AUC {a:.3f}, capture {cap:.1%} of a {orc - bs:.4f} band.\n"
              "  The exam benchmarks were the confound. Re-open the thesis on this traffic.")


if __name__ == "__main__":
    main()
