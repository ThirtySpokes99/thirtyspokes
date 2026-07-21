"""Scaffold-diversity ceiling — the one axis the KOTH mechanism actually competes on.

Every prior no-go read varied the MODEL with the scaffold fixed. But miners compete on the
ORCHESTRATION (the agent code), not the model. So the on-point question is: fix a cheap base model,
vary the SCAFFOLD, on HARD non-saturated tasks — and measure

  (a) does the best scaffold on a cheap model beat a NAIVE FRONTIER agent?  (orchestration ≥ model?)
  (b) do DIFFERENT scaffolds win DIFFERENT tasks (oracle-over-scaffolds ≫ best-single-scaffold)?
      — that solve-diversity is exactly the prize a KOTH orchestration competition would monetize.

5 scaffolds on one cheap base model (naive / self-consistency / self-refine / plan-solve /
code-execution), plus a naive frontier baseline, on MATH-500 L4-5 (math_verify-graded) + MBPP
(exec-graded). Real OpenRouter cost.

Run: set -a && . ./.env && set +a && python scripts/scaffold_diversity_experiment.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

from thirtyspokes.eval import config, hard_tasks
from thirtyspokes.gateway.gateway import OpenRouterBackend

BASE = "openai/gpt-4o-mini"       # the cheap model held fixed across scaffolds
FRONTIER = "openai/gpt-4o"        # naive baseline: "can a scaffold beat a raw frontier agent?"


# ---------------------------------------------------------------------------
# a per-run caller that accumulates (#calls, $cost)
# ---------------------------------------------------------------------------
class Caller:
    def __init__(self, be, model):
        self.be, self.model, self.n, self.cost = be, model, 0, 0.0

    def __call__(self, prompt, max_tokens=1024, temperature=0.0):
        txt, _, _, c = self.be.complete(self.model, [{"role": "user", "content": prompt}],
                                        {"max_tokens": max_tokens, "temperature": temperature})
        self.n += 1; self.cost += c
        return txt


def _py(resp: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", resp, re.S)
    return (m.group(1) if m else resp).strip()


def _run_py(program: str, timeout=10):
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
            f.write(program); f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or ""), (r.stderr or "")[-400:]
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, "", str(e)[-200:]


# ---------------------------------------------------------------------------
# scaffolds — each: (task, call) -> final_response_string   (call tracks cost/#calls)
# ---------------------------------------------------------------------------
def sc_naive(task, call):
    return call(task["prompt"], max_tokens=task["mt"])


def sc_self_consistency(task, call, K=5):
    rs = [call(task["prompt"], max_tokens=task["mt"], temperature=0.7) for _ in range(K)]
    if task["kind"] == "math":
        return hard_tasks.majority_vote(rs)
    for r in rs:                                   # code: first sample passing the visible test
        if task["visible"](r):
            return r
    return rs[0]


def sc_self_refine(task, call):
    r0 = call(task["prompt"], max_tokens=task["mt"])
    crit = call(f"{task['prompt']}\n\nCandidate solution:\n{r0}\n\n"
                "List concrete mistakes in the candidate (if none, say CORRECT).", max_tokens=400)
    return call(f"{task['prompt']}\n\nCandidate:\n{r0}\n\nIssues:\n{crit}\n\n"
                "Now give the corrected final solution.", max_tokens=task["mt"])


def sc_plan_solve(task, call):
    plan = call(f"Outline a concise plan to solve this — do NOT solve yet:\n{task['prompt']}",
                max_tokens=300)
    return call(f"{task['prompt']}\n\nFollow this plan:\n{plan}", max_tokens=task["mt"])


def sc_code_exec(task, call):
    if task["kind"] == "math":
        cp = (f"{task['problem']}\n\nWrite a short Python program that computes the answer and "
              "prints ONLY the final answer (use sympy if helpful).")
        code = _py(call(cp, max_tokens=700))
        ok, out, err = _run_py(code)
        if not ok:
            code = _py(call(f"{cp}\n\nYour code:\n{code}\n\nIt errored:\n{err}\n\nFix it.",
                            max_tokens=700))
            ok, out, err = _run_py(code)
        ans = out.strip().splitlines()[-1].strip() if out.strip() else ""
        return f"\\boxed{{{ans}}}"
    # code domain: generate → run visible test → repair once
    r0 = call(task["prompt"], max_tokens=task["mt"])
    if task["visible"](r0):
        return r0
    ok, out, err = task["run"](r0)
    return call(f"{task['prompt']}\n\nYour code:\n{_py(r0)}\n\nIt failed:\n{err}\n\nFix it.",
                max_tokens=task["mt"])


SCAFFOLDS = {"naive": sc_naive, "self_consistency": sc_self_consistency,
             "self_refine": sc_self_refine, "plan_solve": sc_plan_solve, "code_exec": sc_code_exec}


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def _grade_code(resp, setup, tests):
    prog = _py(resp) + "\n" + (setup or "") + "\n" + "\n".join(tests) + "\n"
    ok, _out, _err = _run_py(prog)
    return 1.0 if ok else 0.0


def load_tasks(n_per, seed):
    from datasets import load_dataset

    tasks = []
    for t in hard_tasks.load_math(n_per, min_level=4, seed=seed):
        problem = t.prompt.replace(hard_tasks._PROMPT_SUFFIX, "")
        tasks.append({"id": t.task_id, "domain": "math", "kind": "math", "prompt": t.prompt,
                      "problem": problem, "mt": 1024,
                      "grade": (lambda r, g=t.gold: hard_tasks.grade_hard(r, g)),
                      "visible": (lambda r, g=t.gold: hard_tasks.grade_hard(r, g))})  # math has no held test
    ds = load_dataset("google-research-datasets/mbpp", split="test").shuffle(seed=seed)
    for i, r in enumerate(ds.select(range(min(n_per, len(ds))))):
        setup, tests = r.get("test_setup_code", ""), r["test_list"]
        prompt = (f"{r['text']}\nYour function must pass:\n{tests[0]}\n"
                  "Return ONLY the Python function code.")
        vis = tests[0]

        def run(resp, s=setup, t0=vis):
            return _run_py(_py(resp) + "\n" + (s or "") + "\n" + t0 + "\n")
        tasks.append({"id": f"mbpp-{i}", "domain": "code", "kind": "code", "prompt": prompt,
                      "mt": 512, "grade": (lambda r, s=setup, t=tests: _grade_code(r, s, t)),
                      "visible": (lambda r, s=setup, t0=vis: _run_py(_py(r) + "\n" + (s or "") + "\n" + t0 + "\n")[0]),
                      "run": run})
    return tasks


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--frontier", default=FRONTIER)
    args = ap.parse_args()

    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=120.0, max_retries=3,
                           price_fn=config.price_for)
    tasks = load_tasks(args.n_per, args.seed)
    print(f"{len(tasks)} tasks (math L4-5 + MBPP); base={args.base}; scaffolds={list(SCAFFOLDS)}\n", flush=True)

    # per (scaffold, task): correctness + cost, on the fixed BASE model
    corr = defaultdict(dict); cost = defaultdict(dict)
    for ti, t in enumerate(tasks):
        for name, fn in SCAFFOLDS.items():
            call = Caller(be, args.base)
            try:
                resp = fn(t, call)
                corr[name][t["id"]] = t["grade"](resp)
            except Exception:  # noqa: BLE001
                corr[name][t["id"]] = 0.0
            cost[name][t["id"]] = call.cost
        # naive frontier baseline
        fcall = Caller(be, args.frontier)
        try:
            corr["_frontier"][t["id"]] = t["grade"](sc_naive(t, fcall))
        except Exception:  # noqa: BLE001
            corr["_frontier"][t["id"]] = 0.0
        cost["_frontier"][t["id"]] = fcall.cost
        if (ti + 1) % 10 == 0:
            print(f"  {ti + 1}/{len(tasks)}", flush=True)

    ids = [t["id"] for t in tasks]
    dom = {t["id"]: t["domain"] for t in tasks}
    def acc(name, subset=ids): return sum(corr[name][i] for i in subset) / max(len(subset), 1)
    def tcost(name, subset=ids): return sum(cost[name][i] for i in subset)
    math_ids = [i for i in ids if dom[i] == "math"]; code_ids = [i for i in ids if dom[i] == "code"]

    print("\n=== per-scaffold (base model), accuracy + total $cost ===")
    print("scaffold".ljust(18) + "  math   code  overall   $cost")
    for name in SCAFFOLDS:
        print(f"{name.ljust(18)}{acc(name, math_ids):6.2f}{acc(name, code_ids):7.2f}"
              f"{acc(name):9.2f}{tcost(name):8.3f}")
    print(f"{'NAIVE-FRONTIER'.ljust(18)}{acc('_frontier', math_ids):6.2f}{acc('_frontier', code_ids):7.2f}"
          f"{acc('_frontier'):9.2f}{tcost('_frontier'):8.3f}")

    best = max(SCAFFOLDS, key=lambda s: acc(s))
    # oracle over scaffolds (base): per task any scaffold correct; cost = cheapest correct scaffold
    oc_hits, oc_cost = 0, 0.0
    for i in ids:
        winners = [s for s in SCAFFOLDS if corr[s][i] >= 1.0]
        oc_hits += 1 if winners else 0
        oc_cost += min((cost[s][i] for s in winners), default=min(cost[s][i] for s in SCAFFOLDS))
    per_dom_best = {d: max(SCAFFOLDS, key=lambda s: acc(s, [i for i in ids if dom[i] == d]))
                    for d in ("math", "code")}

    print("\n=== verdict ===")
    print(f"  BEST-SCAFFOLD (cheap)   {best:18s} acc={acc(best):.3f}  cost=${tcost(best):.3f}")
    print(f"  NAIVE-FRONTIER          {args.frontier:18s} acc={acc('_frontier'):.3f}  cost=${tcost('_frontier'):.3f}")
    print(f"  ORACLE over scaffolds   {'(upper bound)':18s} acc={oc_hits/len(ids):.3f}  cost=${oc_cost:.3f}")
    print(f"  (a) orchestration vs frontier : best-scaffold − naive-frontier = {acc(best) - acc('_frontier'):+.3f} acc")
    print(f"  (b) scaffold solve-diversity  : oracle − best-scaffold        = {oc_hits/len(ids) - acc(best):+.3f} acc")
    print(f"      per-domain best scaffold  : {per_dom_best}   (same ⇒ no cross-domain diversity)")


if __name__ == "__main__":
    main()
