"""Self-contained SWE-bench Pro grader.

The stock `swebench` harness CANNOT grade SWE-bench Pro: its MAP_REPO_VERSION_TO_SPECS only knows the
ORIGINAL SWE-bench repos, so every Pro instance dies with KeyError: '<repo>'. There is no
`swebench-pro` package on PyPI. But the dataset ships the entire evaluation contract, so the grader
is short:

    dockerhub_tag              -> jefzda/sweap-images:<tag>   (verified; ~1 GB compressed)
    before_repo_set_cmd        -> reset to base_commit, then check out the TEST files from the fix
    selected_test_files_to_run -> which test files to run
    fail_to_pass / pass_to_pass-> the pass criteria

Container conventions (verified by inspection): repo at /app, entrypoint bash, pytest on PATH.

Scoring is SWE-bench's: 1.0 iff every fail_to_pass test now passes AND every pass_to_pass test still
passes. A patch that does not apply scores 0.0 -- that is a real failure, not an error.

LIMITATION: Python/pytest repos only. SWE-bench Pro also contains JS/TS repos (tutanota, NodeBB)
whose fail_to_pass uses a different format ("suite.js | test name") and needs a different runner;
those return None (unsupported) rather than a misleading 0.0.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

REGISTRY = "jefzda/sweap-images"
REPO_DIR = "/app"
_PYTEST_LINE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)", re.M)


def as_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:                                    # noqa: BLE001
        try:
            return json.loads(v)
        except Exception:                                # noqa: BLE001
            return []


def is_pytest_task(row: dict) -> bool:
    """Pro mixes pytest repos with JS/TS ones. pytest node IDs contain '::'; the JS format is
    '<file> | <suite name>'. Grading a JS task with pytest would silently score 0."""
    f2p = as_list(row["fail_to_pass"])
    return bool(f2p) and all("::" in str(t) for t in f2p[:5])


def image_for(row: dict) -> str:
    return f"{REGISTRY}:{row['dockerhub_tag']}"


def pull(row: dict, timeout: int = 1800) -> bool:
    r = subprocess.run(["docker", "pull", image_for(row)], capture_output=True, timeout=timeout)
    return r.returncode == 0


def grade_one(row: dict, patch: str, *, timeout: int = 2400) -> float | None:
    """1.0 resolved / 0.0 not / None unsupported-or-infra-error."""
    if not is_pytest_task(row):
        return None
    f2p, p2p = as_list(row["fail_to_pass"]), as_list(row["pass_to_pass"])
    test_files = as_list(row.get("selected_test_files_to_run", "[]"))
    if not test_files:
        test_files = sorted({t.split("::")[0] for t in f2p})
    if not patch.strip():
        return 0.0                                       # no patch is a legitimate miss

    with tempfile.TemporaryDirectory() as d:
        Path(d, "fix.patch").write_text(patch)
        script = "\n".join([
            "set -o pipefail",
            f"cd {REPO_DIR} || exit 90",
            row["before_repo_set_cmd"],                  # reset to base + restore the fix's tests
            # try progressively looser application; a patch that will not apply is a real 0
            "if   git apply -v /koth/fix.patch 2>/dev/null; then echo KOTH_APPLIED=git",
            "elif git apply --3way /koth/fix.patch 2>/dev/null; then echo KOTH_APPLIED=3way",
            "elif patch -p1 --fuzz=5 -i /koth/fix.patch >/dev/null 2>&1; then echo KOTH_APPLIED=patch",
            "else echo KOTH_APPLIED=NONE; fi",
            "echo KOTH_TESTS_BEGIN",
            f"pytest {' '.join(shlex.quote(f) for f in test_files)} "
            "-v --tb=no -p no:cacheprovider --continue-on-collection-errors 2>&1 || true",
        ])
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "-v", f"{d}:/koth:ro",
                 "--entrypoint", "bash", image_for(row), "-c", script],
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        out = r.stdout + r.stderr

    if "KOTH_APPLIED=NONE" in out:
        return 0.0                                       # patch did not apply -> not resolved
    if "KOTH_TESTS_BEGIN" not in out:
        return None                                      # setup failed -> infra, not the model
    status = dict((m.group(1), m.group(2)) for m in _PYTEST_LINE.finditer(out.split("KOTH_TESTS_BEGIN")[-1]))
    if not status:
        return None                                      # nothing ran -> infra
    # SWE-bench semantics: every f2p must now pass, every p2p must still pass
    if any(status.get(t) != "PASSED" for t in f2p):
        return 0.0
    if any(status.get(t) not in (None, "PASSED", "SKIPPED") for t in p2p):
        return 0.0
    return 1.0


if __name__ == "__main__":                               # quick self-check on the GOLD patch
    import numpy as np
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    for i in np.random.default_rng(0).choice(len(ds), 3, replace=False):
        row = ds[int(i)]
        kind = "pytest" if is_pytest_task(row) else "JS/TS (unsupported)"
        print(f"{row['repo']:<28}{kind}")
        if not is_pytest_task(row):
            continue
        print("  pulling ...", flush=True)
        if not pull(row):
            print("  pull FAILED"); continue
        # the GOLD patch must score 1.0 — if it does not, the grader is wrong, not the model
        print(f"  gold patch -> {grade_one(row, row['patch'])}", flush=True)
