"""SWE-bench Pro evaluation domain (ScaleAI/SWE-bench_Pro, 731 real tasks).

Loading + prompt formatting + prediction packaging are fully built and offline-
testable. Grading a patch means applying it in the task's prebuilt Docker image
and running the hidden fail_to_pass / pass_to_pass tests — that needs Docker +
the SWE-bench Pro harness, so `SWEGrader` constructs and invokes it (ready to run
where Docker exists); `MockSWEGrader` exercises the wiring offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SWETask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    requirements: str
    interface: str
    repo_language: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_files: list[str]
    dockerhub_tag: str


def _as_list(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return [v] if v else []


def load_swebench_pro(n: int, split: str = "test", seed: int = 0) -> list[SWETask]:
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split=split)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        r = ds[int(i)]
        out.append(SWETask(
            instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
            problem_statement=r["problem_statement"], requirements=r.get("requirements", ""),
            interface=r.get("interface", ""), repo_language=r.get("repo_language", ""),
            fail_to_pass=_as_list(r["fail_to_pass"]), pass_to_pass=_as_list(r["pass_to_pass"]),
            test_files=_as_list(r.get("selected_test_files_to_run", [])),
            dockerhub_tag=r.get("dockerhub_tag", "")))
    return out


def format_prompt(task: SWETask) -> str:
    """The task shown to the miner's agent. It must return a unified git diff."""
    return (
        f"Repository: {task.repo} (language: {task.repo_language})\n"
        f"You are at commit {task.base_commit}.\n\n"
        f"# Problem\n{task.problem_statement}\n\n"
        f"# Requirements\n{task.requirements}\n\n"
        f"# Interface\n{task.interface}\n\n"
        "Return ONLY a unified git diff (patch) that resolves the problem. "
        "Do not include prose outside the diff.")


def to_prediction(instance_id: str, patch: str, model_name: str = "orchestra") -> dict:
    """The standard SWE-bench prediction record."""
    return {"instance_id": instance_id, "model_name_or_path": model_name,
            "model_patch": patch}


class SWEGrader:  # pragma: no cover — needs Docker + the SWE-bench Pro harness
    """Grade patches by running the hidden tests in each task's Docker image.

    `grade_batch` writes predictions.jsonl and invokes the harness; each task
    scores 1.0 iff all fail_to_pass now pass and all pass_to_pass still pass.
    Point `harness_cmd` at the SWE-bench Pro runner for your environment.
    """

    def __init__(self, harness_cmd: str = "python -m swebench.harness.run_evaluation",
                 max_workers: int = 4, run_id: str = "orchestra"):
        self.harness_cmd = harness_cmd
        self.max_workers = max_workers
        self.run_id = run_id

    def grade_batch(self, tasks: list[SWETask], patches: dict[str, str]) -> dict[str, float]:
        import shutil
        import subprocess
        import tempfile
        if shutil.which("docker") is None:
            raise RuntimeError("Docker not available; SWE grading requires Docker + the SWE-bench Pro harness")
        with tempfile.TemporaryDirectory() as d:
            pred_path = f"{d}/preds.jsonl"
            with open(pred_path, "w") as f:
                for t in tasks:
                    f.write(json.dumps(to_prediction(t.instance_id, patches.get(t.instance_id, ""))) + "\n")
            cmd = (f"{self.harness_cmd} --dataset_name ScaleAI/SWE-bench_Pro "
                   f"--predictions_path {pred_path} --run_id {self.run_id} "
                   f"--max_workers {self.max_workers}")
            subprocess.run(cmd, shell=True, check=True, cwd=d)
            # harness writes <run_id>.<model>.json with resolved instance_ids
            report = json.load(open(f"{d}/{self.run_id}.json"))
            resolved = set(report.get("resolved_ids", report.get("resolved", [])))
        return {t.instance_id: (1.0 if t.instance_id in resolved else 0.0) for t in tasks}


class MockSWEGrader:
    """Offline stand-in: deterministic pass/fail from the patch, for wiring tests."""

    def grade_batch(self, tasks, patches) -> dict[str, float]:
        out = {}
        for t in tasks:
            patch = patches.get(t.instance_id, "")
            out[t.instance_id] = 1.0 if patch.strip().startswith("diff --git") and len(patch) > 40 else 0.0
        return out
