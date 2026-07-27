"""The owner-given benchmark suite the miner runs in its TEE (docs/DESIGN.md §5).

Each `Benchmark` exposes a fixed public POOL of tasks (the "static public dataset"
the design deliberately allows), a deterministic `sample(n, seed)` the runtime and
validator both re-derive from the epoch nonce (so the miner can't cherry-pick or
precompute which slice is scored), a disjoint HELD-OUT set `probe(n, seed)` used
only by the fresh-probe audit (the anti-memorization backstop), and a pure
`grade(answer, gold)` the validator applies to the attested answers.

The offline pools here are computable synthetic stand-ins so the whole subnet runs
with no network. The REAL loaders are seams: math -> `eval.math_tasks`/`eval.hard_tasks`,
SWE -> `eval.swe_tasks`, and MMLU/GPQA -> `koth.mmlu`/`koth.gpqa`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..gateway import signing
from ..gateway.grader import grade_math


@dataclass(frozen=True)
class BenchTask:
    task_id: str
    prompt: str
    gold: object          # float (math) | str (MC letter) | "" (SWE format-graded)


def bench_seed(nonce: str, epoch: int, name: str) -> int:
    """Deterministic per-(nonce, epoch, benchmark) sample seed. The validator issues
    the nonce, so no one knows the slice until after the run is committed to."""
    return int(signing.sha256_hex(f"{nonce}|{epoch}|{name}")[:8], 16)


# --- multiple-choice + patch graders (shared by MMLU/GPQA and SWE) -----------
def extract_choice(answer: str) -> str | None:
    """Parse the chosen option letter. Prefers an explicit 'answer: X', else the
    last standalone A-D. A more robust parser is a production seam."""
    up = str(answer).upper()
    m = re.search(r"ANSWER[:\s]*([A-D])", up)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-D])\b", up)
    return hits[-1] if hits else None


def grade_choice(answer: str, gold: str) -> float:
    c = extract_choice(answer)
    return 1.0 if c is not None and c == str(gold).upper() else 0.0


def grade_patch(answer: str, _gold) -> float:
    """SWE stand-in: a well-formed unified diff scores 1.0. The real grader runs
    the task's hidden tests in Docker (`eval.swe_tasks.SWEGrader`)."""
    a = str(answer).strip()
    return 1.0 if a.startswith("diff --git") and len(a) > 40 else 0.0


# --- a benchmark = fixed pool + disjoint held-out + a grader ------------------
class Benchmark(Protocol):
    name: str
    weight: float
    def sample(self, n: int, seed: int) -> list[BenchTask]: ...
    def probe(self, n: int, seed: int) -> list[BenchTask]: ...
    def grade(self, answer: str, gold) -> float: ...
    def answer(self, task_id: str) -> str: ...   # canonical correct answer (sim helper)


@dataclass
class PooledBenchmark:
    name: str
    weight: float
    _pool: list[BenchTask]
    _heldout: list[BenchTask]
    _grade: Callable[[str, object], float]
    _answers: dict[str, str] = field(default_factory=dict)   # task_id -> canonical answer

    def _draw(self, tasks: list[BenchTask], n: int, seed: int) -> list[BenchTask]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tasks), size=min(n, len(tasks)), replace=False)
        return [tasks[int(i)] for i in idx]

    def sample(self, n: int, seed: int) -> list[BenchTask]:
        return self._draw(self._pool, n, seed)

    def probe(self, n: int, seed: int) -> list[BenchTask]:
        return self._draw(self._heldout, n, seed)

    def grade(self, answer: str, gold) -> float:
        return self._grade(answer, gold)

    def answer(self, task_id: str) -> str:
        return self._answers[task_id]

    def pool_tasks(self) -> list[BenchTask]:
        return self._pool

    def heldout_tasks(self) -> list[BenchTask]:
        return self._heldout


# --- offline computable pools ------------------------------------------------
def _math_pool(prefix: str, lo: int, hi: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    tasks, ans = [], {}
    for i in range(count):
        a, b, c = (int(x) for x in rng.integers(lo, hi, size=3))
        gold = float(a * b + c)
        tid = f"math-{prefix}-{i}"
        tasks.append(BenchTask(tid, f"Compute {a} * {b} + {c}.", gold))
        ans[tid] = str(int(gold))
    return tasks, ans


def make_math() -> PooledBenchmark:
    pool, a1 = _math_pool("pool", 2, 40, 40, seed=101)
    held, a2 = _math_pool("held", 41, 80, 40, seed=202)   # disjoint operand range
    return PooledBenchmark("math", 0.60, pool, held, grade_math, {**a1, **a2})


_VALID_DIFF = "diff --git a/fix.py b/fix.py\n@@ -1 +1 @@\n-old\n+new  # resolves {tid}\n"


def _swe_pool(prefix: str, base: int, count: int):
    tasks, ans = [], {}
    for i in range(count):
        tid = f"swe-{prefix}-{i}"
        rid = base + i
        tasks.append(BenchTask(
            tid, f"[SWE] Repository: acme/pkg-{rid}. Fix the failing test. "
                 "Return ONLY a unified git diff (patch).", ""))
        ans[tid] = _VALID_DIFF.format(tid=tid)
    return tasks, ans


def make_swe() -> PooledBenchmark:
    pool, a1 = _swe_pool("pool", 0, 30)
    held, a2 = _swe_pool("held", 1000, 30)
    return PooledBenchmark("swe", 0.40, pool, held, grade_patch, {**a1, **a2})


def default_suite() -> list[Benchmark]:
    """The four owner-given benchmarks. RANKING weight sums to 1.0 across the free-form pair
    (math 0.60 / swe 0.40); MMLU + GPQA are weight 0 — eligibility floors only, because multiple
    choice cannot be defended against memorization by proof-inspection (the answer is in the prompt
    by construction). Same policy as the live suite, `real_suite`."""
    from . import gpqa, mmlu
    return [make_math(), mmlu.make_mmlu(), gpqa.make_gpqa(), make_swe()]


# --- REAL benchmarks: actual datasets, disjoint pool/held-out slices ---
class RealBenchmark:
    """Wraps a loaded real dataset; sample()/probe() draw from disjoint halves so the
    memorization audit's held-out slice is genuinely unseen by the scored slice."""

    def __init__(self, name: str, weight: float, pool: list[BenchTask],
                 held: list[BenchTask], grade_fn, answer_kind: str | None = None):
        self.name, self.weight = name, weight
        self._pool, self._held, self._grade = pool, held, grade_fn
        # declared, not inferred: `verify._bench_kind` otherwise guesses from the grader's identity
        # and falls through to "number", which would parse a program as a numeric answer.
        if answer_kind is not None:
            self.answer_kind = answer_kind

    def _draw(self, tasks: list[BenchTask], n: int, seed: int) -> list[BenchTask]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tasks), size=min(n, len(tasks)), replace=False)
        return [tasks[int(i)] for i in idx]

    def sample(self, n, seed): return self._draw(self._pool, n, seed)
    def probe(self, n, seed): return self._draw(self._held, n, seed)
    def grade(self, answer, gold): return self._grade(answer, gold)


def real_suite(n_load: int = 1000, seed: int = 0) -> list[Benchmark]:
    """The live scored suite: rank on LiveCodeBench; MMLU + GSM8K are weight-0 eligibility floors.

    WHY CODE CARRIES THE RANKING WEIGHT. The scalar this subnet exists to compute divides by the
    achievable gap `oracle − zero` — the quality a perfect per-ask router could add over a featureless
    one. Measured on real data across eight experiments, that gap is **+0.019 on GSM8K**: every pool
    model lands at 79-97%, and the best router beat the featureless baseline by +0.002. Ranking there
    does not merely fail to reward routing — it divides by a number smaller than the sampling noise
    in an 8-question slice, manufacturing a leaderboard out of nothing. LiveCodeBench is the one
    regime measured with models in the MIDDLE and disagreeing (scores 0.47-0.78, the first non-zero
    sole-correct rate, gap **+0.083**), which is above the validator's `min_headroom_gap` floor where
    math sits far below it. See `koth/lcb.py` for the honest caveat: this makes the competition
    measurable, it does not promise anyone can win it.

    GSM8K stays as a weight-0 FLOOR rather than being dropped: `eligible` requires `acc >= f_min` on
    every benchmark regardless of weight, so it remains a cheap sanity gate that catches a broken
    agent — and unlike code it grades with no container, so it costs the validator nothing.

    Loads real MMLU + GSM8K + LiveCodeBench, splitting each into disjoint pool/held-out.
    (SWE/GPQA are heavier seams — GPQA gated, SWE needs the Docker patch loop.)

    `n_load` MUST be >> the validator's `n_per_bench`, or the per-epoch "slice" is the whole pool
    and the epoch nonce selects nothing. It was 16 (-> 8 pool items, drawn 8-at-a-time), which made
    the scored set 16 FIXED public questions forever: the unpredictable-beacon defence was vacuous,
    the anti-grind commit window had no sample variance to protect, and the whole set was trivially
    memorizable. 1000 -> 500 pool / 500 held-out per benchmark, so an 8-of-500 draw genuinely
    varies per epoch. Both miner and validator derive the suite identically, so this MUST be changed
    in lockstep — hence the SUITE_VERSION bump (`runtime.py`).

    Upper bound: GSM8K's test split is ~1319 rows, so `n_load` cannot go far above 1000 without
    changing the split or the dataset."""
    from ..eval import math_tasks
    from . import lcb, mmlu
    mt = mmlu.load_mmlu(n_load, seed=seed)
    g = [BenchTask(t.task_id, t.prompt, t.gold) for t in math_tasks.load_gsm8k(n_load, seed=seed)]
    half = lambda xs: (xs[: len(xs) // 2], xs[len(xs) // 2:])  # noqa: E731
    mp, mh = half(mt)
    gp, gh = half(g)
    # WEIGHTS: the RANKING scalar is 100% CODE; MMLU and math are floor-only sanity gates (weight 0).
    #
    # Multiple choice cannot be defended against memorization by any proof-inspection rule: every
    # option is in the prompt by construction, so `extract_choice(prompt)` already yields the gold
    # and the provenance rule in `verify._grounded_one` would flag every honest agent. MMLU at 0.5
    # therefore meant HALF the score sat on a benchmark where memorizing the 500-item public pool
    # was undetectable in principle — and the memorizer is also the CHEAPEST miner, so it dominated
    # both axes of `S = Q_lcb − λ·cost`. Weight 0 keeps it as an eligibility floor (`eligible`
    # checks `acc >= f_min` on EVERY benchmark regardless of weight), so a broken agent is still
    # caught while memorizing it earns exactly nothing.
    #
    # `math` was the ranked benchmark until the achievable gap was measured on it: +0.019, which is
    # smaller than the sampling noise the router scalar would divide it by. It is retained at weight
    # 0 for the same reason as MMLU — `eligible` gates on EVERY benchmark's accuracy regardless of
    # weight, so it stays a cheap broken-agent detector that needs no container to grade.
    #
    # SWE remains the intended second RANKED benchmark and is deliberately still not wired —
    # `grade_patch` is a stand-in that scores any well-formed diff 1.0, so ranking on it today would
    # be trivially gameable. It needs `eval.swe_tasks.SWEGrader` (Docker + the SWE-bench Pro
    # harness) first, and one-shot SWE measured degenerate at the FLOOR (0/120) regardless.
    return [RealBenchmark("mmlu", 0.0, mp, mh, grade_choice),
            RealBenchmark("math", 0.0, gp, gh, math_tasks.grade),
            lcb.make_lcb(seed=seed, weight=1.0)]


# --- FRESH benchmarks (WS4): tasks generated per-epoch from the seed, so there is nothing
#     to memorize by construction (the memorization-proof option; static PooledBenchmark
#     instead relies on the fresh-probe audit). MockPool's solver grades these too. ---
class FreshBenchmark:
    def __init__(self, name: str, weight: float, gen, grade_fn):
        self.name, self.weight = name, weight
        self._gen, self._grade = gen, grade_fn

    def sample(self, n, seed): return self._gen(n, seed)
    def probe(self, n, seed): return self._gen(n, seed ^ 0x5EED)   # disjoint held-out slice
    def grade(self, answer, gold): return self._grade(answer, gold)


def _gen_math(n: int, seed: int) -> list[BenchTask]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b, c = (int(x) for x in rng.integers(2, 99, size=3))
        out.append(BenchTask(f"fmath-{seed}-{i}", f"Compute {a} * {b} + {c}.", float(a * b + c)))
    return out


def _gen_mc(n: int, seed: int) -> list[BenchTask]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b, c = (int(x) for x in rng.integers(2, 30, size=3))
        val = a * b + c
        distract: set[int] = set()
        while len(distract) < 3:
            cand = val + int(rng.integers(-12, 13))
            if cand != val:
                distract.add(cand)
        pos = int(rng.integers(0, 4))
        opts = list(distract); opts.insert(pos, val)
        lines = "\n".join(f"{chr(65+j)}) {o}" for j, o in enumerate(opts))
        out.append(BenchTask(f"fmc-{seed}-{i}",
                             f"[MMLU] What is the value of {a} * {b} + {c}?\n{lines}\n\n"
                             "Answer with the letter only.", chr(65 + pos)))
    return out


def fresh_suite() -> list[Benchmark]:
    """Freshly-generated (nonce-seeded) tasks — memorization impossible by construction."""
    return [FreshBenchmark("math", 0.5, _gen_math, grade_math),
            FreshBenchmark("mmlu", 0.5, _gen_mc, grade_choice)]
