"""Counterfactual pool matrix over MIXED-DIFFICULTY MATH traffic (GSM8K + MATH-500 L5).

Builds the RouterBench-shaped substrate: every (prompt x model) cell populated with response,
realized cost, tokens, latency and graded score. Then computes the three numbers that decide whether
a routing subnet has anything to compete over, plus the two baselines a miner must beat.

Two deliberate departures from every prior ceiling experiment in this repo, both from the 2026-07-25
research pass:

  * MIXED DIFFICULTY. Prior runs scored each benchmark separately. Within one benchmark difficulty is
    roughly homogeneous, so "always call the best model" is optimal by construction and routing
    measures nothing -- which may be why composition measured ~= best-single five times. Here GSM8K
    (easy) and MATH-500 L4-5 (hard) are INTERLEAVED into one stream, so difficulty varies per query
    and is predictable from the prompt. That is the regime where a difficulty-router can earn.

  * COST IS THE AXIS, not a tiebreak. RouterBench is a dominator pool on accuracy (gpt-4 sole-correct
    on 2.03% of prompts) but not on cost (cheapest-correct oracle matches oracle quality 13.6-14.6x
    cheaper). So the gate reports the cost ratio as a first-class number.

The matrix is CACHED per (prompt, model) and resumable -- a cell is never paid for twice.

  set -a && . ./.env && set +a
  python scripts/pool_matrix_math.py --n-easy 50 --n-hard 50 --estimate      # free: cost estimate
  python scripts/pool_matrix_math.py --n-easy 50 --n-hard 50                 # spends money
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from thirtyspokes.eval import config, hard_tasks, math_tasks
from thirtyspokes.gateway.gateway import OpenRouterBackend

# Curated for CAPABILITY DIVERSITY across LABS, not for price tiers. The research is explicit that a
# cost ladder is the wrong shape (headroom collapses ~20x under a dominator; price/quality is not
# even monotone) and that the one material learned-router win came from an ORTHOGONAL-capability
# pool. Different labs = different training data/recipes = the best available proxy for orthogonality
# when every frontier model clusters on public benchmarks. ~100x input-price spread.
DEFAULT_MODELS = [
    "openai/gpt-5-nano",             # $0.05 / $0.40
    "deepseek/deepseek-v4-flash",    # $0.09 / $0.19
    "minimax/minimax-m2.7",          # $0.25 / $1.00
    "z-ai/glm-5.2",                  # $0.71 / $2.23
    "moonshotai/kimi-k3",            # $3.00 / $15.00
    "anthropic/claude-opus-5",       # $5.00 / $25.00
]

# Reasoning models spend their whole completion budget thinking and never reach an answer if this is
# small -- a measured failure in frontier_ceiling_experiment.py (gemini needed ~18k for one MATH-5).
# Silently scoring a frontier model ~0 would invent complementarity that does not exist.
MAX_TOKENS = 8192
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_math.jsonl"


def load_stream(n_easy: int, n_hard: int, seed: int, min_level: int = 5):
    """One interleaved mixed-difficulty stream. Both tiers are free-form numeric, so provenance is
    checkable (CRITICAL-2) and grading is exact."""
    easy = [dict(task_id=t.task_id, prompt=t.prompt, gold=t.gold, tier="easy", kind="gsm8k")
            for t in math_tasks.load_gsm8k(n_easy, seed=seed)]
    hard = [dict(task_id=t.task_id, prompt=t.prompt, gold=t.gold, tier="hard", kind="math500")
            for t in hard_tasks.load_math(n_hard, min_level=min_level, seed=seed)]
    stream = easy + hard
    np.random.default_rng(seed).shuffle(stream)
    return stream


def grade(row: dict, answer: str) -> float:
    if row["kind"] == "gsm8k":
        return math_tasks.grade(answer, row["gold"])
    return hard_tasks.grade_hard(answer, row["gold"])


def load_cache() -> dict:
    cells = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                cells[(c["task_id"], c["model"])] = c
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-easy", type=int, default=50)
    ap.add_argument("--n-hard", type=int, default=50)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--min-level", type=int, default=5,
                    help="MATH-500 difficulty floor for the HARD tier (5 = hardest)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--estimate", action="store_true", help="print a cost estimate and exit (free)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stream = load_stream(args.n_easy, args.n_hard, args.seed, args.min_level)
    cells = load_cache()
    todo = [(r, m) for r in stream for m in models if (r["task_id"], m) not in cells]

    if args.estimate:
        # ~700 prompt tokens, and assume half the completion budget is actually spent
        import urllib.request
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {config.LiveConfig().api_key}"})
        price = {m["id"]: m.get("pricing", {}) for m in
                 json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]}
        total = 0.0
        print(f"{'model':<34}{'cells':>7}{'est $':>10}")
        for m in models:
            p = price.get(m, {})
            pin, pout = float(p.get("prompt", 0)), float(p.get("completion", 0))
            n = sum(1 for r, mm in todo if mm == m)
            c = n * (700 * pin + MAX_TOKENS * 0.5 * pout)
            total += c
            print(f"{m:<34}{n:>7}{c:>10.2f}")
        print(f"{'TOTAL':<34}{len(todo):>7}{total:>10.2f}")
        print(f"\n({len(cells)} cells already cached and will not be re-paid)")
        return

    cfg = config.LiveConfig()
    cfg.require_key()
    backend = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=300.0, max_retries=3,
                                price_fn=config.price_for)
    print(f"stream: {len(stream)} prompts ({args.n_easy} easy + {args.n_hard} hard) x "
          f"{len(models)} models = {len(stream)*len(models)} cells; {len(todo)} to fetch",
          flush=True)

    fh = CACHE.open("a")
    done = {"n": 0}

    def run(job):
        row, model = job
        t0 = time.time()
        try:
            text, tin, tout, cost = backend.complete(
                model, [{"role": "user", "content": row["prompt"]}], {"max_tokens": MAX_TOKENS})
            cell = dict(task_id=row["task_id"], model=model, tier=row["tier"], kind=row["kind"],
                        response=text, tokens_in=tin, tokens_out=tout, cost_usd=float(cost),
                        latency_s=round(time.time() - t0, 2), score=grade(row, text))
        except Exception as e:  # noqa: BLE001 — a dead cell must not kill the run; it is retried next time
            cell = dict(task_id=row["task_id"], model=model, tier=row["tier"], kind=row["kind"],
                        error=f"{type(e).__name__}: {e}", cost_usd=0.0, score=None)
        done["n"] += 1
        if done["n"] % 25 == 0:
            print(f"  {done['n']}/{len(todo)}", flush=True)
        return cell

    # as_completed, NOT ex.map: map yields in INPUT order, so one slow reasoning call
    # head-of-line-blocks every finished cell behind it and nothing is persisted meanwhile.
    # On a paid, resumable job that is the difference between losing minutes and losing an hour.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for f in as_completed(futs):
            cell = f.result()
            fh.write(json.dumps(cell) + "\n")
            fh.flush()
            cells[(cell["task_id"], cell["model"])] = cell
    fh.close()

    spent = sum(c.get("cost_usd", 0.0) for c in cells.values())
    errs = sum(1 for c in cells.values() if c.get("error"))
    print(f"\nmatrix complete. spent ${spent:.2f} over {len(cells)} cells ({errs} errors)")
    print(f"cached at {CACHE}")


if __name__ == "__main__":
    sys.exit(main())
