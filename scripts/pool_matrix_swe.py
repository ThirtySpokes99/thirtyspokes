"""Counterfactual pool matrix over SWE-bench Pro (one-shot patching, oracle retrieval).

A probe on 3 tasks found the first per-task complementarity in this codebase (Opus 5 solved a
qutebrowser task DeepSeek failed) -- but it did NOT survive at n=20: every model scored 0/20 here.
One-shot patching is far harder than the published 62-80%, which come from AGENTIC scaffolds that
browse the repo and run tests. Kept because the harness is reusable if an agentic loop is built.

Two constraints discovered while building it, both load-bearing:
  * MAX_TOKENS 8192 -> 32768. Opus 5 emitted ZERO-length patches at exactly the 8k cap on 3 of 6
    cells -- pure reasoning, no diff. Scoring a frontier model on truncation would MANUFACTURE
    complementarity that does not exist.
  * pytest-only sampling. Roughly a third of SWE-bench Pro is JS/TS whose fail_to_pass format
    ("suite.js | test name") the grader cannot run; sampling blind would silently shrink n.

Output is written in the SAME schema as pool_matrix_math.jsonl, so routing_headroom.py reads it unchanged.

  set -a && . ./.env && set +a
  python scripts/pool_matrix_swe.py --generate          # API spend
  python scripts/pool_matrix_swe.py --grade             # Docker
  python scripts/routing_headroom.py --matrix data/pool_matrix_swe.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swe_pro_grader import as_list, grade_one, is_pytest_task, pull  # noqa: E402

from thirtyspokes.eval import config  # noqa: E402
from thirtyspokes.gateway.gateway import OpenRouterBackend  # noqa: E402

MODELS = [
    "openai/gpt-5-nano",
    "deepseek/deepseek-v4-flash",
    "minimax/minimax-m2.7",
    "z-ai/glm-5.2",
    "moonshotai/kimi-k3",
    "anthropic/claude-opus-5",
]
N_TASKS = 20
MAX_CTX_CHARS = 40_000
MAX_TOKENS = 16_384          # enough for a long patch once THINKING is bounded separately
# Reasoning models consume whatever max_tokens you give them: at 8k and again at 32k they
# thought until the budget ran out and emitted no diff (27% empty cells, 15 of 18 at the cap,
# $5.93 burned). Capping reasoning effort is the fix; raising max_tokens is not.
REASONING = {"effort": "low"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_swe.jsonl"
GEN = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_swe_generations.jsonl"

_PATCH_RE = re.compile(r"^diff --git a/(\S+) b/", re.M)


def fetch_file(repo, commit, path):
    try:
        return urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", timeout=25
        ).read().decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return None


def oracle_prompt(row) -> str:
    """Problem + the PRE-PATCH contents of the files the gold patch touches. Does not leak the fix;
    removes the needle-in-a-haystack of guessing which file to edit (SWE-bench's oracle setting)."""
    head = (f"Repository: {row['repo']} (language: {row['repo_language']})\n"
            f"You are at commit {row['base_commit']}.\n\n"
            f"# Problem\n{row['problem_statement']}\n\n# Requirements\n{row['requirements']}\n")
    chunks, used = [], 0
    for f in _PATCH_RE.findall(row["patch"]):
        body = fetch_file(row["repo"], row["base_commit"], f)
        if body is None:
            continue
        room = MAX_CTX_CHARS - used
        if room <= 500:
            break
        chunks.append(f"--- FILE: {f} ---\n{body[:room]}")
        used += min(len(body), room)
    ctx = ("\n# Current contents of the files you will need to change\n" + "\n\n".join(chunks)) if chunks else ""
    return (head + ctx + "\n\nReturn ONLY a unified git diff (patch) that fixes the problem. "
            "Start with 'diff --git'. No prose, no code fences.")


def extract_patch(text: str) -> str:
    t = str(text or "").replace("```diff", "```")
    if "```" in t:
        parts = [p for p in t.split("```") if "diff --git" in p]
        if parts:
            t = parts[0]
    i = t.find("diff --git")
    return (t[i:].strip() + "\n") if i >= 0 else ""


def load_tasks():
    """pytest-only, deterministic. Over-sample then filter, since ~1/3 of Pro is JS/TS."""
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    order = np.random.default_rng(7).permutation(len(ds))
    out = []
    for i in order:
        row = ds[int(i)]
        if is_pytest_task(row):
            out.append(row)
        if len(out) >= N_TASKS:
            break
    return out


def generate():
    rows = load_tasks()
    print(f"{len(rows)} pytest tasks selected", flush=True)
    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=900.0, max_retries=3,
                           price_fn=config.price_for)
    done = set()
    if GEN.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["model"])
                for l in GEN.read_text().splitlines() if l.strip()}
    prompts = {r["instance_id"]: oracle_prompt(r) for r in rows}
    jobs = [(r, m) for r in rows for m in MODELS if (r["instance_id"], m) not in done]
    print(f"{len(jobs)} cells to generate", flush=True)
    fh = GEN.open("a")
    n = {"i": 0}

    def run(job):
        r, model = job
        t0 = time.time()
        try:
            text, tin, tout, cost = be.complete(
                model, [{"role": "user", "content": prompts[r["instance_id"]]}],
                {"max_tokens": MAX_TOKENS, "reasoning": REASONING})
            return dict(task_id=r["instance_id"], model=model, repo=r["repo"],
                        patch=extract_patch(text), tokens_in=tin, tokens_out=tout,
                        cost_usd=float(cost), latency_s=round(time.time() - t0, 1))
        except Exception as e:                            # noqa: BLE001
            return dict(task_id=r["instance_id"], model=model, repo=r["repo"], patch="",
                        tokens_in=0, tokens_out=0, cost_usd=0.0, latency_s=0.0,
                        error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(run, j) for j in jobs]):
            c = f.result(); fh.write(json.dumps(c) + "\n"); fh.flush()
            n["i"] += 1
            if n["i"] % 10 == 0:
                print(f"  {n['i']}/{len(jobs)}", flush=True)
    fh.close()
    tot = sum(json.loads(l)["cost_usd"] for l in GEN.read_text().splitlines() if l.strip())
    empty = sum(1 for l in GEN.read_text().splitlines() if l.strip() and not json.loads(l)["patch"])
    print(f"\ngenerated. spent ${tot:.2f}; {empty} empty patches", flush=True)


def grade():
    rows = {r["instance_id"]: r for r in load_tasks()}
    gens = [json.loads(l) for l in GEN.read_text().splitlines() if l.strip()]
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["model"])
                for l in OUT.read_text().splitlines() if l.strip()}
    todo = [g for g in gens if (g["task_id"], g["model"]) not in done]
    print(f"{len(todo)} cells to grade", flush=True)
    for iid in {g["task_id"] for g in todo}:              # pull images once, serially
        pull(rows[iid])
    fh = OUT.open("a")

    def run(g):
        s = grade_one(rows[g["task_id"]], g["patch"])
        return dict(task_id=g["task_id"], model=g["model"], tier="hard", kind="swe",
                    response=g["patch"][:400], tokens_in=g.get("tokens_in", 0),
                    tokens_out=g.get("tokens_out", 0), cost_usd=g["cost_usd"],
                    latency_s=g.get("latency_s", 0.0),
                    score=(None if s is None else float(s)))

    n = {"i": 0}
    with ThreadPoolExecutor(max_workers=4) as ex:        # 4 containers at a time on 12 cores
        for f in as_completed([ex.submit(run, g) for g in todo]):
            c = f.result(); fh.write(json.dumps(c) + "\n"); fh.flush()
            n["i"] += 1
            print(f"  {n['i']}/{len(todo)}  {c['model']:<28}{c['score']}", flush=True)
    fh.close()
    print(f"\ngraded -> {OUT}\nnow: python scripts/routing_headroom.py --matrix {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()
    generate() if a.generate else grade() if a.grade else print(__doc__)
