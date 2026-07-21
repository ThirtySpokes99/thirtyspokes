"""Verifiable-answer graders + fresh task generation (anti-contamination).

Math is the MVP domain: grading needs no API and no sandbox — extract the final
number and compare to a computed gold. Tasks are generated procedurally and
block-seeded, so they are fresh every epoch and unmemorizable. SWE/GPQA graders
are production seams (Docker + hidden tests / sealed keys).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MathTask:
    task_id: str
    prompt: str
    gold: float


def generate_math_tasks(n: int, seed: int) -> list[MathTask]:
    """Fresh procedural multi-step arithmetic. Seed from a recent block hash in
    production so no one can precompute the epoch's tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        a, b, c, d = (int(x) for x in rng.integers(2, 40, size=4))
        t = i % 4
        if t == 0:
            gold, prompt = a * b + c, f"Compute {a} * {b} + {c}."
        elif t == 1:
            gold, prompt = a * b - c * d, f"Compute {a} * {b} - {c} * {d}."
        elif t == 2:
            gold, prompt = (a + b) * c, f"Compute ({a} + {b}) * {c}."
        else:
            gold, prompt = a * b * c - d, f"Compute {a} * {b} * {c} - {d}."
        tasks.append(MathTask(f"math-{seed}-{i}", prompt, float(gold)))
    return tasks


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(answer: str) -> float | None:
    matches = _NUM.findall(str(answer))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def grade_math(answer: str, gold: float, tol: float = 1e-6) -> float:
    v = extract_number(answer)
    return 1.0 if v is not None and abs(v - gold) <= tol else 0.0


class SWEGrader:  # pragma: no cover — production seam
    """Apply the miner's patch in a Docker sandbox and run the repo's hidden test
    suite; gold = tests pass. Tasks = GitHub issues created after a cutoff."""

    def grade(self, patch: str, task) -> float:
        raise NotImplementedError("SWEGrader needs a Docker sandbox + SWE-bench harness")
