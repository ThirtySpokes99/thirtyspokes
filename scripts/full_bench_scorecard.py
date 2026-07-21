"""Full-suite miner scorecard (Phase A/B).

Runs the test miner agents (routing model / inference / orchestration) over the FULL owner benchmarks
— GSM8K (full) + MMLU (capped) + GPQA (real, if HF_TOKEN) — with real OpenRouter metering, scores each
(accuracy + bootstrap LCB + real USD cost + aggregate Q_lcb), ranks by the cost-budgeted King, and
writes FULL_BENCH_SCORECARD.md. SWE is deferred (Phase C, Docker). Results are cached per (agent,bench)
so adding GPQA later (once the token lands) doesn't re-pay for math+MMLU.

  set -a && . ./.env && set +a
  python scripts/full_bench_scorecard.py --mmlu 2500 --gsm 1319 --gpqa 448 --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from thirtyspokes.eval import config, math_tasks
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.koth import gpqa, mmlu
from thirtyspokes.koth.benchmarks import BenchTask, grade_choice
from thirtyspokes.koth.verify import _bootstrap_lcb

CHEAP = "meta-llama/llama-3.1-8b-instruct"
STRONG = "openai/gpt-4o"
ALLOW = {CHEAP, STRONG}

AGENTS = {
    "cheap-only": ("def build_agent(w):\n def agent(p,cm):\n"
                   f"  return cm('{CHEAP}',[{{'role':'user','content':p}}],{{'max_tokens':512}})\n return agent", b"{}"),
    "strong-only": ("def build_agent(w):\n def agent(p,cm):\n"
                    f"  return cm('{STRONG}',[{{'role':'user','content':p}}],{{'max_tokens':512}})\n return agent", b"{}"),
    "length-router": ("""import json
def build_agent(w):
 c=json.loads(w.decode())
 HARD=('prove','integral','matrix','theorem','which of the following','probability')
 def agent(p,cm):
  hard=len(p)>c['thresh'] or any(k in p.lower() for k in HARD)
  return cm(c['strong'] if hard else c['cheap'],[{'role':'user','content':p}],{'max_tokens':512})
 return agent""", json.dumps({"thresh": 320, "cheap": CHEAP, "strong": STRONG}).encode()),
    "cascade-verify": ("""import json,re
def build_agent(w):
 c=json.loads(w.decode())
 def agent(p,cm):
  a=str(cm(c['cheap'],[{'role':'user','content':p}],{'max_tokens':512}))
  if len(a.strip())<2 or not re.search(r'[0-9A-D]',a):
   a=str(cm(c['strong'],[{'role':'user','content':p}],{'max_tokens':512}))
  return a
 return agent""", json.dumps({"cheap": CHEAP, "strong": STRONG}).encode()),
}


def build(src, w):
    ns: dict = {}
    exec(src, ns)  # noqa: S102 — our own trusted agent source
    return ns["build_agent"](w)


def load_suite(mmlu_n, gsm_n, gpqa_n, seed=1):
    benches = [("math", [BenchTask(t.task_id, t.prompt, float(t.gold)) for t in math_tasks.load_gsm8k(gsm_n, seed=seed)], math_tasks.grade),
               ("mmlu", mmlu.load_mmlu(mmlu_n, seed=seed), grade_choice)]
    if os.environ.get("HF_TOKEN"):
        try:
            benches.append(("gpqa", gpqa.load_gpqa(gpqa_n, seed=seed), grade_choice))
        except Exception as e:  # noqa: BLE001
            print(f"[gpqa] skipped ({type(e).__name__}: {str(e)[:80]}) — check HF access to Idavidrein/gpqa")
    else:
        print("[gpqa] no HF_TOKEN — running math+MMLU only; add the token + re-run to append GPQA")
    return benches


class Meter:
    def __init__(self, backend):
        self.b, self.cost, self.calls, self.lock = backend, 0.0, Counter(), threading.Lock()

    def call(self, model, messages, params=None):
        if model not in ALLOW:
            raise ValueError(f"model {model} not in pinned pool")
        content, _tin, _tout, cost = self.b.complete(model, messages, params or {})
        with self.lock:
            self.cost += cost
            self.calls[model] += 1
        return content


def run_agent_bench(agent, tasks, grade_fn, backend, workers):
    meter = Meter(backend)
    answers = {}

    def do(t):
        try:
            return t.task_id, str(agent(t.prompt, meter.call))
        except Exception:  # noqa: BLE001 — a failed call grades as wrong, cost as metered
            return t.task_id, ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, a in ex.map(do, tasks):
            answers[tid] = a
    correct = [grade_fn(answers[t.task_id], t.gold) for t in tasks]
    return {"acc": float(np.mean(correct)), "lcb": _bootstrap_lcb(correct, 0.05, 1000, 1234),
            "cost": meter.cost, "n": len(tasks), "calls": dict(meter.calls),
            "correct": {t.task_id: correct[i] for i, t in enumerate(tasks)},
            "model_of": {t.task_id: (list(meter.calls) and None) for t in tasks}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", type=int, default=2500)
    ap.add_argument("--gsm", type=int, default=1319)
    ap.add_argument("--gpqa", type=int, default=448)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cache", default="scripts/.full_bench_cache.json")
    ap.add_argument("--out", default="FULL_BENCH_SCORECARD.md")
    args = ap.parse_args()

    cfg = config.LiveConfig(); cfg.require_key()
    backend = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=90.0, max_retries=4, price_fn=config.price_for)
    cfgkey = f"m{args.mmlu}g{args.gsm}q{args.gpqa}"
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    print("loading suite…", flush=True)
    benches = load_suite(args.mmlu, args.gsm, args.gpqa)
    bench_names = [b[0] for b in benches]
    w = 1.0 / len(benches)

    results = {}
    for name, (src, wt) in AGENTS.items():
        agent = build(src, wt)
        row = {}
        for bname, tasks, gfn in benches:
            ck = f"{cfgkey}|{name}|{bname}|n{len(tasks)}"
            if ck in cache:
                row[bname] = cache[ck]; print(f"  [cache] {name}/{bname}", flush=True); continue
            r = run_agent_bench(agent, tasks, gfn, backend, args.workers)
            r.pop("correct", None); r.pop("model_of", None)
            row[bname] = r; cache[ck] = r
            json.dump(cache, open(args.cache, "w"))
            print(f"  {name:14s} {bname:5s} acc={r['acc']:.3f} lcb={r['lcb']:.3f} cost=${r['cost']:.4f} "
                  f"n={r['n']} routed={r['calls']}", flush=True)
        results[name] = row

    # aggregate + rank
    agg = {}
    for name, row in results.items():
        qlcb = sum(w * row[b]["lcb"] for b in bench_names)
        acc = sum(w * row[b]["acc"] for b in bench_names)
        cost = sum(row[b]["cost"] for b in bench_names)
        agg[name] = {"qlcb": qlcb, "acc": acc, "cost": cost}
    order = sorted(agg, key=lambda n: -agg[n]["qlcb"])

    lines = [f"# Full-Benchmark Miner Scorecard\n",
             f"Benchmarks: {', '.join(bench_names)} (weights {w:.3f} each). "
             f"Real OpenRouter, full metered cost. SWE deferred (Phase C).\n",
             "| Miner | " + " | ".join(f"{b} acc" for b in bench_names) + " | Q_lcb | avg acc | total cost |",
             "|---|" + "---|" * (len(bench_names) + 3)]
    for name in order:
        cells = " | ".join(f"{results[name][b]['acc']:.3f}" for b in bench_names)
        a = agg[name]
        lines.append(f"| {name} | {cells} | {a['qlcb']:.3f} | {a['acc']:.3f} | ${a['cost']:.4f} |")
    lines.append("\n## Ranking by attested Q_lcb (cost-budgeted)")
    maxc = max(a["cost"] for a in agg.values())
    for budget, label in ((maxc * 1.01, "unbounded"), (0.8 * maxc, "tight")):
        elig = [n for n in order if agg[n]["cost"] <= budget]
        lines.append(f"\n**Budget {label} (${budget:.3f}):** " +
                     " > ".join(f"{n} ({agg[n]['qlcb']:.3f})" for n in elig) +
                     (f"  — excluded: {[n for n in order if n not in elig]}" if len(elig) < len(order) else ""))

    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
