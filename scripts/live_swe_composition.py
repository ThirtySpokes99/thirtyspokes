#!/usr/bin/env python
"""SWE composition ceiling on SWE-bench Verified (oracle-file, single-shot).

The economics-gate question: does composition beat the best single model on real,
hard, agentic-code tasks? We measure the CEILING — the per-sample oracle (any pool
model resolves the task) vs the best single model. If different models solve
different tasks, oracle >> best-single and a composition prize exists.

Setup (oracle-file): each model is given the issue + the current content of the
files the gold patch touches, and must emit a git diff. One call per model per task
(no agent loop). Patches are graded by the real swebench Docker harness (hidden
tests). Needs OPENROUTER_API_KEY + Docker + `pip install swebench`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
from datasets import load_dataset
from openai import OpenAI

from thirtyspokes.eval import config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_TOKENS = 4096
# pure-local-test repos (no network in the sandboxed containers; gold verified on pylint)
LIGHT = {"pylint-dev/pylint", "pytest-dev/pytest", "pydata/xarray", "mwaskom/seaborn"}
POOL = {
    "deepseek": "deepseek/deepseek-v3.2-exp",
    "gpt4o": "openai/gpt-4o-2024-11-20",
    "opus": "anthropic/claude-opus-4.7",
    "qwen-coder": "qwen/qwen-2.5-coder-32b-instruct",
}
DS = "princeton-nlp/SWE-bench_Verified"


def oracle_files(repo: str, base: str, patch: str) -> dict[str, str]:
    paths = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
    out = {}
    for p in paths:
        try:
            url = f"https://raw.githubusercontent.com/{repo}/{base}/{p}"
            out[p] = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")[:12000]
        except Exception:
            pass
    return out


def build_prompt(task) -> str:
    files = oracle_files(task["repo"], task["base_commit"], task["patch"])
    blocks = "\n".join(f"## {p}\n```\n{c}\n```" for p, c in files.items())
    return (f"You are fixing a bug in the `{task['repo']}` repository.\n\n"
            f"# Issue\n{task['problem_statement']}\n\n"
            f"# Relevant files (current content)\n{blocks}\n\n"
            "Produce a git diff (unified `diff --git` format, correct a/ and b/ paths) that "
            "resolves the issue. Output ONLY the diff inside a ```diff code block.")


def extract_patch(text: str) -> str:
    m = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
    body = m.group(1) if m else text
    i = body.find("diff --git")
    body = body[i:] if i >= 0 else body
    return body.rstrip() + "\n" if body.strip() else ""


def grade(model_slug: str, preds: list[dict], ids: list[str]) -> set[str]:
    path = f"/tmp/swe_preds_{model_slug}.jsonl"
    with open(path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    run_id = f"comp_{model_slug}"
    subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", DS, "--predictions_path", path, "--run_id", run_id,
                    "--instance_ids", *ids, "--max_workers", "4", "--cache_level", "instance"],
                   check=False)
    import glob
    for rep in glob.glob(f"*{run_id}*.json") + glob.glob(f"{model_slug}*.json"):
        try:
            d = json.load(open(rep))
            if "resolved_ids" in d:
                return set(d["resolved_ids"])
        except Exception:
            pass
    return set()


def main():
    cfg = config.LiveConfig(); cfg.require_key()
    cli = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=120.0, max_retries=2)
    ds = load_dataset(DS, split="test")
    import numpy as np
    light = [r for r in ds if r["repo"] in LIGHT]
    idx = np.random.default_rng(SEED).choice(len(light), size=min(N, len(light)), replace=False)
    tasks = [light[int(i)] for i in idx]
    ids = [t["instance_id"] for t in tasks]
    print(f"SWE-bench Verified oracle-file composition, n={len(tasks)}\n  pool={list(POOL.values())}")
    print(f"  instances={ids}\n")

    resolved: dict[str, set[str]] = {}
    for tier, model in POOL.items():
        preds = []
        print(f"  {tier:11s} generating...", end="", flush=True)
        for t in tasks:
            try:
                r = cli.chat.completions.create(model=model, temperature=0.0, max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": build_prompt(t)}])
                patch = extract_patch(r.choices[0].message.content or "")
            except Exception as e:
                print(f"[{t['instance_id']} {type(e).__name__}]", end="")
                patch = ""
            preds.append({"instance_id": t["instance_id"],
                          "model_name_or_path": tier, "model_patch": patch})
            print("+" if patch else ".", end="", flush=True)
        print(" grading...", flush=True)
        resolved[tier] = grade(tier, preds, ids)
        print(f"  {tier:11s} resolved {len(resolved[tier])}/{len(tasks)}: {sorted(resolved[tier])}")

    print("\n" + "=" * 64)
    print(f"COMPOSITION CEILING on SWE-bench Verified (n={len(tasks)})")
    print("=" * 64)
    per = {t: len(s) / len(tasks) for t, s in resolved.items()}
    best = max(per, key=per.get)
    oracle = set().union(*resolved.values()) if resolved else set()
    print("  per-model resolve : " + ", ".join(f"{t}={per[t]:.2f}" for t in POOL))
    print(f"  best single model : {best} = {per[best]:.2f} ({len(resolved[best])}/{len(tasks)})")
    print(f"  per-sample oracle  : {len(oracle)/len(tasks):.2f} ({len(oracle)}/{len(tasks)})  <- composition CEILING")
    print(f"  composition gap    : {(len(oracle)/len(tasks) - per[best])*100:+.1f} pts over best-single")
    uniq = {t: s - set().union(*[resolved[o] for o in resolved if o != t]) for t, s in resolved.items()}
    print("  uniquely solved    : " + ", ".join(f"{t}={len(uniq[t])}" for t in POOL) + "  (diversity = composition value)")


if __name__ == "__main__":
    main()
