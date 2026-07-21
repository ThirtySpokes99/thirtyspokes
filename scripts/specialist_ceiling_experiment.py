"""Specialist-diverse ceiling demonstrator.

Tests the one condition that could reopen building a routing/composition subnet: a domain-mixed
query stream over models each strongest (and cheap) in a DIFFERENT domain, so no single model
dominates. Real questions, exact grading, real OpenRouter cost. Reports the decisive numbers:

  * BEST-SINGLE   — the single model with the highest accuracy (chosen on a train split).
  * DOMAIN-ROUTER — the *achievable* router: per domain pick the best model on the train split,
    route each test query by its (prompt-obvious) domain. This is what a real router can do.
  * ORACLE        — per-query best (an upper bound a router can't exceed).

If DOMAIN-ROUTER ≈ BEST-SINGLE on accuracy AND doesn't beat it on cost, the dominator-pool problem
holds even on deliberately specialist-diverse traffic → no moat. If it opens a real gap, that
quantifies the headroom and the traffic shape a launch would need.

Run: set -a && . ./.env && set +a && python scripts/specialist_ceiling_experiment.py
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

from thirtyspokes.eval import config, math_tasks
from thirtyspokes.gateway.gateway import OpenRouterBackend

# cost-varied, domain-SKEWED pool: a code specialist, cheap generalists, and one frontier model
DEFAULT_MODELS = [
    "qwen/qwen3-coder",                   # code specialist, cheap
    "deepseek/deepseek-chat",             # strong general + math, cheap
    "meta-llama/llama-3.3-70b-instruct",  # cheap generalist
    "openai/gpt-4o-mini",                 # cheap generalist
    "openai/gpt-4o",                      # frontier generalist, expensive (likely best-single)
]
DOMAINS = ["code", "math", "medicine", "law"]


# ---------------------------------------------------------------------------
# datasets → tasks: {id, domain, prompt, grade(response)->0/1, max_tokens}
# ---------------------------------------------------------------------------
def _mcq(rows, subject, n):
    tasks = []
    for i, r in enumerate(rows.select(range(min(n, len(rows))))):
        opts = "\n".join(f"{chr(65 + j)}) {c}" for j, c in enumerate(r["choices"]))
        gold = chr(65 + int(r["answer"]))
        prompt = (f"{r['question']}\n{opts}\n\nAnswer with the single letter (A/B/C/D) only.")
        tasks.append({"id": f"{subject}-{i}", "domain": subject, "prompt": prompt,
                      "grade": (lambda resp, g=gold: float(_extract_letter(resp) == g)),
                      "max_tokens": 8})
    return tasks


def _extract_letter(resp: str) -> str | None:
    hits = re.findall(r"\b([A-D])\b", resp.strip().upper())
    return hits[-1] if hits else None      # last standalone letter (robust to "the answer is X")


def _load_code(n, seed):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test").shuffle(seed=seed)
    tasks = []
    for i, r in enumerate(ds.select(range(min(n, len(ds))))):
        # standard MBPP protocol: show the tests so the required function name/signature is known
        prompt = (f"{r['text']}\nYour function must pass these tests:\n"
                  + "\n".join(r["test_list"][:2])
                  + "\nReturn ONLY the Python function code (no explanation, no test code).")
        setup, tests = r.get("test_setup_code", ""), r["test_list"]
        tasks.append({"id": f"mbpp-{i}", "domain": "code", "prompt": prompt,
                      "grade": (lambda resp, s=setup, t=tests: _grade_code(resp, s, t)),
                      "max_tokens": 400})
    return tasks


def _grade_code(resp: str, setup: str, tests: list[str]) -> float:
    m = re.search(r"```(?:python)?\s*(.*?)```", resp, re.S)
    code = (m.group(1) if m else resp).strip()
    program = code + "\n" + setup + "\n" + "\n".join(tests) + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
            f.write(program); f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
        return 1.0 if r.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, OSError):
        return 0.0


def load_tasks(n_per: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    tasks: list[dict] = []
    tasks += _load_code(n_per, seed)
    for t in math_tasks.load_gsm8k(n_per, seed=seed):        # numeric, exact gold
        tasks.append({"id": t.task_id, "domain": "math", "prompt": t.prompt,
                      "grade": (lambda resp, g=t.gold: math_tasks.grade(resp, g)), "max_tokens": 400})
    for subj in ("professional_medicine", "professional_law"):
        rows = load_dataset("cais/mmlu", subj, split="test").shuffle(seed=seed)
        dom = "medicine" if "medicine" in subj else "law"
        for t in _mcq(rows, subj, n_per):
            t["domain"] = dom
            tasks.append(t)
    return tasks


# ---------------------------------------------------------------------------
# run + analysis
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per", type=int, default=40, help="questions per domain")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=120.0, max_retries=2,
                           price_fn=config.price_for)
    tasks = load_tasks(args.n_per, args.seed)
    print(f"loaded {len(tasks)} tasks across {len(set(t['domain'] for t in tasks))} domains; "
          f"{len(models)} models → {len(tasks) * len(models)} calls\n", flush=True)

    corr: dict = defaultdict(dict)   # model -> {task_id: 0/1}
    cost: dict = defaultdict(dict)
    errors: dict = defaultdict(int)
    for ti, t in enumerate(tasks):
        for m in models:
            try:
                text, _tin, _tout, c = be.complete(m, [{"role": "user", "content": t["prompt"]}],
                                                   {"max_tokens": t["max_tokens"]})
                corr[m][t["id"]] = t["grade"](text)
                cost[m][t["id"]] = c
            except Exception:  # noqa: BLE001 — a model erroring is data, not a crash
                corr[m][t["id"]] = 0.0; cost[m][t["id"]] = 0.0; errors[m] += 1
        if (ti + 1) % 20 == 0:
            print(f"  {ti + 1}/{len(tasks)} tasks done", flush=True)

    models = [m for m in models if errors[m] < len(tasks)]        # drop fully-broken models
    by_domain: dict = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"]].append(t["id"])
    rng = random.Random(args.seed)
    train, test = set(), set()
    for ids in by_domain.values():                                # 50/50 split per domain
        s = ids[:]; rng.shuffle(s); k = len(s) // 2
        train.update(s[:k]); test.update(s[k:])

    def acc(m, ids):
        ids = [i for i in ids if i in corr[m]]
        return sum(corr[m][i] for i in ids) / max(len(ids), 1)

    def tcost(m, ids):
        return sum(cost[m][i] for i in ids if i in cost[m])

    # per-model per-domain accuracy (test) + overall
    print("\n=== per-model accuracy (test split) ===")
    hdr = "model".ljust(38) + "".join(d[:5].rjust(8) for d in DOMAINS) + "   overall   $cost"
    print(hdr)
    test_ids = list(test)
    for m in models:
        row = m.ljust(38)
        for d in DOMAINS:
            row += f"{acc(m, [i for i in by_domain[d] if i in test]):8.2f}"
        row += f"{acc(m, test_ids):10.2f}{tcost(m, test_ids):8.3f}"
        print(row)

    # BEST-SINGLE: chosen on TRAIN overall accuracy, measured on TEST
    best_single = max(models, key=lambda m: acc(m, list(train)))
    bs_acc, bs_cost = acc(best_single, test_ids), tcost(best_single, test_ids)

    # DOMAIN-ROUTER: per domain pick the best-TRAIN model (tie-break cheaper), route TEST by domain
    pick = {}
    for d in DOMAINS:
        tr = [i for i in by_domain[d] if i in train]
        pick[d] = max(models, key=lambda m: (acc(m, tr), -tcost(m, tr)))
    dr_correct = sum(corr[pick[t_dom]][tid]
                     for tid in test_ids for t_dom in [next(t["domain"] for t in tasks if t["id"] == tid)])
    dr_acc = dr_correct / max(len(test_ids), 1)
    dr_cost = sum(cost[pick[next(t["domain"] for t in tasks if t["id"] == tid)]][tid] for tid in test_ids)

    # ORACLE: per-query any-correct; cost = cheapest correct model (upper bound)
    oc_hits, oc_cost = 0, 0.0
    for tid in test_ids:
        winners = [m for m in models if corr[m].get(tid, 0.0) >= 1.0]
        if winners:
            oc_hits += 1
            oc_cost += min(cost[m][tid] for m in winners)
        else:
            oc_cost += min(cost[m][tid] for m in models)

    n = len(test_ids)
    print(f"\n=== ceiling (n={n} test queries) ===")
    print(f"  BEST-SINGLE   {best_single:34s} acc={bs_acc:.3f}  cost=${bs_cost:.3f}")
    print(f"  DOMAIN-ROUTER {'(' + ', '.join(f'{d}:{pick[d].split('/')[-1][:14]}' for d in DOMAINS) + ')':34s}"
          f" acc={dr_acc:.3f}  cost=${dr_cost:.3f}")
    print(f"  ORACLE        {'per-query best (upper bound)':34s} acc={oc_hits / max(n,1):.3f}  cost=${oc_cost:.3f}")
    print("\n=== verdict ===")
    print(f"  achievable accuracy gap (router − best-single): {dr_acc - bs_acc:+.3f}")
    print(f"  achievable cost saving at ≥ best-single acc   : "
          f"{'yes, $%.3f→$%.3f (%.0f%% cheaper)' % (bs_cost, dr_cost, 100*(1-dr_cost/bs_cost)) if dr_acc >= bs_acc - 1e-9 and dr_cost < bs_cost else 'none'}")
    print(f"  oracle headroom over best-single (upper bound): {oc_hits/max(n,1) - bs_acc:+.3f} acc")
    if errors:
        print(f"  note: model errors {dict(errors)}")


if __name__ == "__main__":
    main()
