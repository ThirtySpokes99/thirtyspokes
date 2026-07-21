#!/usr/bin/env python
"""Decisive SWE composition ceiling — AGENTIC (mini-swe-agent loop per model).

Fixes the cheap-read confound: each model runs a real agent loop (read/edit/test in
the task container) and emits a git diff that APPLIES cleanly, so we measure
problem-solving, not diff formatting. Per-sample oracle (any model resolves) vs
best-single answers whether a composition prize exists. Needs OPENROUTER_API_KEY +
Docker + mini-swe-agent + swebench.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

INSTANCES = [
    "pydata__xarray-4695", "pydata__xarray-2905", "pylint-dev__pylint-7080",
    "pylint-dev__pylint-4604", "pytest-dev__pytest-5809",
]
MODELS = {
    "deepseek": "openrouter/deepseek/deepseek-v3.2-exp",
    "gpt4o": "openrouter/openai/gpt-4o-2024-11-20",
    "opus": "openrouter/anthropic/claude-opus-4.7",
}
DS = "princeton-nlp/SWE-bench_Verified"
FILT = "(" + "|".join(INSTANCES) + ")"


def run_agent(tier: str, model: str) -> str:
    out = f"/tmp/agentic_{tier}"
    subprocess.run([sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
                    "--subset", DS, "--split", "test", "--filter", FILT,
                    "-m", model, "-w", "5", "--redo-existing",
                    "-c", "swebench.yaml", "-c", "agent.step_limit=60", "-c", "agent.cost_limit=1.0",
                    "-o", out], check=False)
    return f"{out}/preds.json"


def grade(tier: str, preds: str) -> set[str]:
    run_id = f"agentic_{tier}"
    subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", DS, "--predictions_path", preds, "--run_id", run_id,
                    "--instance_ids", *INSTANCES, "--max_workers", "5", "--cache_level", "instance"],
                   check=False)
    for rep in glob.glob(f"*{run_id}*.json"):
        try:
            d = json.load(open(rep))
            if "resolved_ids" in d:
                return set(d["resolved_ids"])
        except Exception:
            pass
    return set()


def main():
    resolved: dict[str, set[str]] = {}
    for tier, model in MODELS.items():
        print(f"\n===== agent: {tier} ({model}) =====", flush=True)
        preds = run_agent(tier, model)
        resolved[tier] = grade(tier, preds)
        print(f"  {tier} resolved {len(resolved[tier])}/{len(INSTANCES)}: {sorted(resolved[tier])}", flush=True)

    n = len(INSTANCES)
    per = {t: len(s) / n for t, s in resolved.items()}
    best = max(per, key=per.get)
    oracle = set().union(*resolved.values()) if resolved else set()
    print("\n" + "=" * 60)
    print(f"AGENTIC COMPOSITION CEILING — SWE-bench Verified (n={n})")
    print("=" * 60)
    print("  per-model resolve : " + ", ".join(f"{t}={per[t]:.2f}" for t in MODELS))
    print(f"  best single model : {best} = {per[best]:.2f} ({len(resolved[best])}/{n})")
    print(f"  per-sample oracle  : {len(oracle)/n:.2f} ({len(oracle)}/{n})  <- composition CEILING")
    print(f"  composition gap    : {(len(oracle)/n - per[best])*100:+.1f} pts over best-single")
    uniq = {t: s - set().union(*[resolved[o] for o in resolved if o != t]) for t, s in resolved.items()}
    print("  uniquely solved    : " + ", ".join(f"{t}={sorted(uniq[t])}" for t in MODELS))


if __name__ == "__main__":
    main()
