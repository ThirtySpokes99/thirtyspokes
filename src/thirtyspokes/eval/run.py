"""Live-run driver — the one command to run the real evaluation with a key.

Sets up the metered gateway (real OpenRouter or --mock), loads a real task set
(GSM8K / procedural math, or SWE-bench Pro), runs the reference agent to produce
answer + proof per task, then verifies each proof (grade + receipts + binding)
and reports accuracy, real cost, and cost-aware score. This is the miner+validator
round trip on real data; swap the agent for a better one and the pool in config.

  python -m thirtyspokes.eval.run --domain math --n 20            # needs OPENROUTER_API_KEY
  python -m thirtyspokes.eval.run --domain math --n 5 --mock      # offline wiring check
  python -m thirtyspokes.eval.run --domain swe  --n 5             # needs key + Docker
"""

from __future__ import annotations

import argparse

import numpy as np

from ..gateway.gateway import MeteringGateway, MockBackend
from ..gateway.signing import Signer
from ..gateway.verify import verify_submission
from . import config, math_tasks, swe_tasks
from .agent import ReferenceAgent

_N = [0]


def _nonce():
    _N[0] += 1
    return f"n{_N[0]}"


def _make_gateway(cfg: config.LiveConfig, mock: bool, hotkey: str) -> MeteringGateway:
    if mock:
        backend = MockBackend()
    else:
        cfg.require_key()
        from ..gateway.gateway import OpenRouterBackend
        backend = OpenRouterBackend(cfg.api_key, cfg.base_url, cfg.request_timeout,
                                    cfg.max_retries, price_fn=config.price_for)
    return MeteringGateway(backend, {hotkey})


def run_math(args) -> None:
    cfg = config.LiveConfig()
    model = cfg.pool[args.tier]
    agent = ReferenceAgent(Signer(), model if not args.mock else args.tier, max_tokens=args.max_tokens)
    gw = _make_gateway(cfg, args.mock, agent.hotkey)
    tasks = (math_tasks.procedural(args.n, args.seed) if args.dataset == "procedural"
             else math_tasks.load_gsm8k(args.n, seed=args.seed))

    rows = []
    for t in tasks:
        sub, receipts = agent.solve(gw, t.task_id, t.prompt, _nonce)
        v = verify_submission(sub, receipts, gw.public_hex, t.gold, math_tasks.grade,
                              lam=args.lam, cost_ref=args.cost_ref)
        rows.append(v)
    _report("math", args, rows, model)


def run_swe(args) -> None:
    cfg = config.LiveConfig()
    model = cfg.pool[args.tier]
    agent = ReferenceAgent(Signer(), model if not args.mock else args.tier, max_tokens=args.max_tokens)
    gw = _make_gateway(cfg, args.mock, agent.hotkey)
    tasks = swe_tasks.load_swebench_pro(args.n, seed=args.seed)

    subs, patches = {}, {}
    for t in tasks:
        sub, receipts = agent.solve(gw, t.instance_id, swe_tasks.format_prompt(t), _nonce)
        subs[t.instance_id] = (sub, receipts)
        patches[t.instance_id] = sub.final_answer
    grader = swe_tasks.MockSWEGrader() if args.mock else swe_tasks.SWEGrader()
    results = grader.grade_batch(tasks, patches)   # {instance_id: 0/1}

    rows = []
    for t in tasks:
        sub, receipts = subs[t.instance_id]
        v = verify_submission(sub, receipts, gw.public_hex, t.instance_id,
                              lambda ans, iid: results[iid], lam=args.lam, cost_ref=args.cost_ref)
        rows.append(v)
    _report("swe-bench-pro", args, rows, model)


def _report(domain, args, rows, model) -> None:
    valid = [r for r in rows if r.valid]
    acc = float(np.mean([r.quality for r in valid])) if valid else 0.0
    cost = float(np.mean([r.cost_usd for r in valid])) if valid else 0.0
    score = float(np.mean([r.score for r in valid])) if valid else 0.0
    print("=" * 64)
    print(f"domain={domain}  model={model}  n={len(rows)}  "
          f"{'[MOCK]' if args.mock else '[LIVE]'}")
    print("=" * 64)
    print(f"  valid submissions : {len(valid)}/{len(rows)}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  mean cost / task  : ${cost:.5f}   (total ${sum(r.cost_usd for r in valid):.4f})")
    print(f"  cost-aware score  : {score:+.4f}   (lambda={args.lam}, cost_ref=${args.cost_ref})")
    if any(not r.valid for r in rows):
        from collections import Counter
        print("  rejected reasons  :", dict(Counter(r.reason for r in rows if not r.valid)))


def main() -> None:
    ap = argparse.ArgumentParser(prog="orchestra-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=["math", "swe"], default="math")
    ap.add_argument("--dataset", choices=["gsm8k", "procedural"], default="gsm8k")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tier", choices=list(config.DEFAULT_POOL), default="cheap")
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--cost_ref", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="offline wiring check (no API/Docker)")
    args = ap.parse_args()
    (run_swe if args.domain == "swe" else run_math)(args)


if __name__ == "__main__":
    main()
