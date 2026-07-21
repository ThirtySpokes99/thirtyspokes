"""Math evaluation domain: real GSM8K + fresh procedural tasks, numeric grading.

GSM8K gives real, human-written problems with numeric gold (verifiable, no judge).
The procedural generator gives *fresh* block-seeded problems for the
anti-contamination task stream. Both yield a unified `Task`; grading is numeric
extraction + compare — fully offline, no API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gateway.grader import extract_number, generate_math_tasks


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    gold: float
    source: str


PROMPT_SUFFIX = "\n\nReason step by step, then give the final answer as a single number."


def load_gsm8k(n: int, split: str = "test", seed: int = 0) -> list[Task]:
    """Real GSM8K problems (gold after '#### '). Sampled deterministically."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    tasks = []
    for i in idx:
        row = ds[int(i)]
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        tasks.append(Task(f"gsm8k-{int(i)}", row["question"] + PROMPT_SUFFIX, gold, "gsm8k"))
    return tasks


def procedural(n: int, seed: int) -> list[Task]:
    """Fresh procedural arithmetic (unmemorizable; seed from a block hash)."""
    return [Task(t.task_id, t.prompt + PROMPT_SUFFIX, t.gold, "procedural")
            for t in generate_math_tasks(n, seed)]


def grade(answer: str, gold: float, tol: float = 1e-4) -> float:
    v = extract_number(answer)
    return 1.0 if v is not None and abs(v - gold) <= tol else 0.0
