#!/usr/bin/env python
"""Pull the Docker images for one seeded R2E-Gym slice, so grading never pulls inside an episode.

`sandbox.run` passes `--pull never` on purpose: pulling 2.15 GB inside a graded episode spends the
arm's wall clock on the validator's housekeeping (§8b.2), and an arm that pauses to download is an
arm whose per-task timings mean nothing. So the slice is provisioned first, deliberately, and an
image still absent at grade time is a loud refusal rather than a silent stall.

THE DRAW MATCHES `m3a_spread.py` EXACTLY — same seed, same shuffle, same slice — because
provisioning a different set than the run uses is indistinguishable from a flaky benchmark: tasks
would drop at grade time and the arm would look worse than it is.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.v3.benchmarks import sandbox                    # noqa: E402
from thirtyspokes.v3.benchmarks.r2egym import R2EGym              # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=15)
    parser.add_argument("--seed", default="m3a")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    bench = R2EGym()
    loaded = list(bench.load())
    random.Random(f"{args.seed}:r2egym").shuffle(loaded)
    slice_ = loaded[:args.tasks]
    images = sorted({bench.image_for(t.task_id) for t in slice_})
    print(f"{len(slice_)} tasks -> {len(images)} distinct images")

    have = sandbox.images_present(images)
    todo = [i for i in images if i not in have]
    print(f"already present: {len(have)}   to pull: {len(todo)}")
    if args.dry_run:
        for image in todo:
            print(f"  would pull {image}")
        return 0

    # `sandbox._docker` rather than a command line of our own: `docker_host()` returns None for a
    # declared-local daemon, and "None means add no --host" is a rule that already lives in one
    # place. Rebuilding the line here got it wrong on the first run — `--host None`.
    for n, image in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] pulling {image}", flush=True)
        result = subprocess.run(sandbox._docker("pull", image), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr.strip()[:200]}")
    still = sandbox.images_present(images)
    print(f"present now: {len(still)}/{len(images)}")
    return 0 if len(still) == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
