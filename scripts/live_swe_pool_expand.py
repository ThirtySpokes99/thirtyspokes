#!/usr/bin/env python
"""Power up the agentic composition read: does POOL DIVERSITY lift the ceiling?

n=5 with 3 general frontier models had opus dominate (oracle == best-single). But
composition value comes from diversity — different models solving different tasks.
This adds diverse/specialist models on the SAME 5 (cached) instances and recomputes
the oracle over the FULL pool. opus missed pylint-7080 + pylint-4604; if any added
model solves one, oracle > best-single and a composition prize exists.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys

INSTANCES = [
    "pydata__xarray-4695", "pydata__xarray-2905", "pylint-dev__pylint-7080",
    "pylint-dev__pylint-4604", "pytest-dev__pytest-5809",
]
# already run (from live_swe_agentic.py)
PRIOR = {
    "deepseek": {"pytest-dev__pytest-5809"},
    "gpt4o": set(),
    "opus": {"pydata__xarray-2905", "pydata__xarray-4695", "pytest-dev__pytest-5809"},
}
NEW = {
    "qwen-coder": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "codestral": "openrouter/mistralai/codestral-2501",
    "llama70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "grok-code": "openrouter/x-ai/grok-code-fast-1",
}
DS = "princeton-nlp/SWE-bench_Verified"
FILT = "(" + "|".join(INSTANCES) + ")"


def run_and_grade(tier: str, model: str) -> set[str]:
    out = f"/tmp/agentic_{tier}"
    subprocess.run([sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
                    "--subset", DS, "--split", "test", "--filter", FILT, "-m", model,
                    "-w", "5", "--redo-existing", "-c", "swebench.yaml",
                    "-c", "agent.step_limit=60", "-c", "agent.cost_limit=1.0", "-o", out], check=False)
    run_id = f"agentic_{tier}"
    subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", DS, "--predictions_path", f"{out}/preds.json",
                    "--run_id", run_id, "--instance_ids", *INSTANCES,
                    "--max_workers", "5", "--cache_level", "instance"], check=False)
    for rep in glob.glob(f"*{run_id}*.json"):
        try:
            d = json.load(open(rep))
            if "resolved_ids" in d:
                return set(d["resolved_ids"])
        except Exception:
            pass
    return set()


def main():
    resolved = dict(PRIOR)
    for tier, model in NEW.items():
        print(f"\n===== agent: {tier} ({model}) =====", flush=True)
        resolved[tier] = run_and_grade(tier, model)
        print(f"  {tier} resolved {len(resolved[tier])}/{len(INSTANCES)}: {sorted(resolved[tier])}", flush=True)

    n = len(INSTANCES)
    per = {t: len(s) / n for t, s in resolved.items()}
    best = max(per, key=per.get)
    oracle = set().union(*resolved.values())
    print("\n" + "=" * 62)
    print(f"FULL-POOL AGENTIC COMPOSITION CEILING — SWE-bench Verified (n={n}, {len(resolved)} models)")
    print("=" * 62)
    print("  per-model resolve : " + ", ".join(f"{t}={per[t]:.2f}" for t in resolved))
    print(f"  best single model : {best} = {per[best]:.2f}")
    print(f"  per-sample oracle  : {len(oracle)/n:.2f} ({len(oracle)}/{n})  <- composition CEILING")
    print(f"  composition gap    : {(len(oracle)/n - per[best])*100:+.1f} pts over best-single")
    for inst in INSTANCES:
        solvers = [t for t, s in resolved.items() if inst in s]
        print(f"    {inst:30s} solved by: {solvers or 'NONE'}")


if __name__ == "__main__":
    main()
