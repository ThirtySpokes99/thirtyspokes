"""GPQA benchmark — graduate-level multiple-choice, letter-match grading (docs/DESIGN.md §5).

`load_gpqa` is the real HuggingFace seam (`Idavidrein/gpqa`, a GATED dataset — set
`HF_TOKEN`). `make_gpqa` builds an offline computable pool (a harder expression than
MMLU so the two benchmarks are distinct) so the subnet runs with no network; grading
reuses `benchmarks.grade_choice`.
"""

from __future__ import annotations

import numpy as np

from .benchmarks import BenchTask, PooledBenchmark, grade_choice


def load_gpqa(n: int, subset: str = "gpqa_main", seed: int = 0) -> list[BenchTask]:  # pragma: no cover — gated HF seam
    """Real GPQA (needs HF_TOKEN). Options are shuffled deterministically; gold is
    the letter of the correct answer after the shuffle."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", subset, split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        r = ds[int(i)]
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = rng.permutation(4)
        shuffled = [choices[j] for j in order]
        gold = chr(65 + int(np.where(order == 0)[0][0]))
        lines = "\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(shuffled))
        prompt = f"[GPQA] {r['Question']}\n{lines}\n\nAnswer with the letter only."
        out.append(BenchTask(f"gpqa-{int(i)}", prompt, gold))
    return out


def _mc_pool(prefix: str, lo: int, hi: int, count: int, seed: int):
    """Offline computable MC: 'value of (a+b)*c - d?' with 4 int options."""
    rng = np.random.default_rng(seed)
    tasks, ans = [], {}
    for i in range(count):
        a, b, c, d = (int(x) for x in rng.integers(lo, hi, size=4))
        val = (a + b) * c - d
        distract: set[int] = set()
        while len(distract) < 3:
            cand = val + int(rng.integers(-15, 16))
            if cand != val:
                distract.add(cand)
        pos = int(rng.integers(0, 4))
        opts = list(distract)
        opts.insert(pos, val)
        gold = chr(65 + pos)
        lines = "\n".join(f"{chr(65+j)}) {o}" for j, o in enumerate(opts))
        tid = f"gpqa-{prefix}-{i}"
        tasks.append(BenchTask(
            tid, f"[GPQA] What is the value of ({a} + {b}) * {c} - {d}?\n{lines}\n\n"
                 "Answer with the letter only.", gold))
        ans[tid] = gold
    return tasks, ans


def make_gpqa() -> PooledBenchmark:
    pool, a1 = _mc_pool("pool", 2, 15, 40, seed=505)
    held, a2 = _mc_pool("held", 16, 30, 40, seed=606)   # disjoint operand range
    # weight 0 — floor only; multiple choice is undefendable by proof-inspection (see mmlu.py)
    return PooledBenchmark("gpqa", 0.0, pool, held, grade_choice, {**a1, **a2})
