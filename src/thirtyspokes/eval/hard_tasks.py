"""Hard verifiable benchmark: MATH-500 (competition math, levels 1-5).

Unlike GSM8K (saturated), level-4/5 MATH is genuinely hard — frontier models are
far from 100% and weaker models much lower — so a composition gap can actually
appear. Answers are LaTeX; grading uses math_verify (the field-standard robust
checker), which also gives us answer-equivalence for majority voting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PROMPT_SUFFIX = ("\n\nSolve step by step. Put ONLY the final answer inside "
                  "\\boxed{} at the end.")


@dataclass(frozen=True)
class HardTask:
    task_id: str
    prompt: str
    gold: str
    level: int
    subject: str


def load_math(n: int, min_level: int = 4, seed: int = 0) -> list[HardTask]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = [i for i in range(len(ds)) if int(ds[i]["level"]) >= min_level]
    pick = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    out = []
    for i in pick:
        r = ds[int(i)]
        out.append(HardTask(f"math500-{int(i)}", r["problem"] + _PROMPT_SUFFIX,
                            r["answer"], int(r["level"]), r["subject"]))
    return out


def _mv():
    from math_verify import parse, verify
    return parse, verify


def _boxed(s: str) -> str:
    r"""math_verify extracts from a LaTeX answer *environment*; a bare gold string is parsed with the
    plain-text rules and silently mangled — `1\frac{4}{5}` comes back as 4/5, dropping the leading 1.
    MATH golds are stored bare, so every mixed-number/latex gold mis-parsed and grade_hard returned
    0.0 for even an exactly-correct answer (the blanket `except` below hid it). Wrap before parsing."""
    s = str(s).strip()
    return s if "\\boxed" in s else f"\\boxed{{{s}}}"


def grade_hard(answer: str, gold: str) -> float:
    # parsing_timeout=None is REQUIRED off the main thread: math_verify's default timeout uses
    # signal.alarm(), which raises in any worker thread. Graded inside a ThreadPoolExecutor (as every
    # parallel eval does), the raise was swallowed by the `except` below and EVERY answer scored 0.0 —
    # a silent, total zeroing that reads exactly like "the models all failed".
    parse, verify = _mv()
    try:
        return 1.0 if verify(parse(_boxed(gold), parsing_timeout=None),
                             parse(str(answer), parsing_timeout=None),
                             timeout_seconds=None) else 0.0
    except Exception:
        return 0.0


def answers_equal(a: str, b: str) -> bool:
    parse, verify = _mv()
    try:
        return bool(verify(parse(_boxed(a), parsing_timeout=None),
                           parse(_boxed(b), parsing_timeout=None),
                           timeout_seconds=None))
    except Exception:
        return False


def majority_vote(answers: list[str]) -> str:
    """Return the answer from the largest equivalence cluster (ties -> first)."""
    clusters: list[list[str]] = []
    for a in answers:
        for c in clusters:
            if answers_equal(a, c[0]):
                c.append(a); break
        else:
            clusters.append([a])
    return max(clusters, key=len)[0] if clusters else ""
