"""MMLU benchmark — multiple-choice, letter-match grading (docs/DESIGN.md §5).

`load_mmlu` is the real HuggingFace seam (`cais/mmlu`, public answer keys — which
is exactly the "static public dataset" the KOTH binding + fresh-probe design makes
safe). `make_mmlu` builds an offline computable pool so the subnet runs with no
network; grading reuses `benchmarks.grade_choice`.
"""

from __future__ import annotations

import numpy as np

from .benchmarks import BenchTask, PooledBenchmark, grade_choice


def load_mmlu(n: int, split: str = "test", seed: int = 0) -> list[BenchTask]:  # pragma: no cover — HF seam
    """Real MMLU. Gold is the correct option letter; the prompt lists A-D options."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split=split)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        r = ds[int(i)]
        opts = "\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(r["choices"]))
        prompt = (f"[MMLU] {r['question']}\n{opts}\n\nAnswer with the letter only.")
        out.append(BenchTask(f"mmlu-{int(i)}", prompt, chr(65 + int(r["answer"]))))
    return out


def _mc_pool(prefix: str, lo: int, hi: int, count: int, seed: int):
    """Offline computable MC: 'value of a*b+c?' with 4 int options."""
    rng = np.random.default_rng(seed)
    tasks, ans = [], {}
    for i in range(count):
        a, b, c = (int(x) for x in rng.integers(lo, hi, size=3))
        val = a * b + c
        distract: set[int] = set()
        while len(distract) < 3:
            cand = val + int(rng.integers(-12, 13))
            if cand != val:
                distract.add(cand)
        pos = int(rng.integers(0, 4))
        opts = list(distract)
        opts.insert(pos, val)
        gold = chr(65 + pos)
        lines = "\n".join(f"{chr(65+j)}) {o}" for j, o in enumerate(opts))
        tid = f"mmlu-{prefix}-{i}"
        tasks.append(BenchTask(
            tid, f"[MMLU] What is the value of {a} * {b} + {c}?\n{lines}\n\n"
                 "Answer with the letter only.", gold))
        ans[tid] = gold
    return tasks, ans


def make_mmlu() -> PooledBenchmark:
    pool, a1 = _mc_pool("pool", 2, 15, 40, seed=303)
    held, a2 = _mc_pool("held", 16, 30, 40, seed=404)   # disjoint operand range
    # weight 0: multiple choice cannot be defended against memorization by proof-inspection (the
    # answer is in the prompt by construction), so it is an ELIGIBILITY FLOOR, never ranked.
    # `eligible` still requires acc >= f_min here, so a broken agent is caught; memorizing it
    # earns nothing. Same policy as the live suite — see benchmarks.real_suite.
    return PooledBenchmark("mmlu", 0.0, pool, held, grade_choice, {**a1, **a2})
