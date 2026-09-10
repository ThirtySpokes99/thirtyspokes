"""The verdict — paired aggregate plus the two breadth rules (docs/WHITEPAPER.md
§5.2, §5.2b, §5.3; the build plan M8).

A challenger takes the crown iff ALL THREE hold:

  1. **Aggregate.** The paired delta in `final` exceeds `eps` AND the bootstrap lower bound on that
     paired difference is above zero.
  2. **Breadth, by leave-one-out.** `min over b of [ aggregate delta in final, computed WITHOUT
     benchmark b ] > 0` — drop the challenger's single best subject and it must still win.
  3. **Breadth, by median.** `median over b of [ per-benchmark delta in final ] > 0` — the typical
     benchmark must have moved the challenger's way.

This is the "Spirit" of the three words: what is rewarded is the part that generalises, because a
win concentrated in a few benchmarks is a lookup table wearing a policy's coat.

WHY CONDITION 3 EXISTS, AND WHY IT IS AN ADDITION RATHER THAN A REPLACEMENT (§5.2, §5.3, D16).
Conditions 1–2 were measured against the failure mode §5.3 names — a miner learning *"benchmark X →
model Y wins"* — and did not bind. Sweeping one fixed aggregate win over k benchmarks, leave-one-out
refused only k = 1: a challenger that beat the king on TWO of twelve and was byte-identical on the
other ten was crowned at delta +0.0525, `loo_min` +0.0286, sign test 2/12 at p = 0.9968. The same
shape survived being broadly *worse* — losing 0.02 on ten benchmarks and winning 0.50 on two is
crowned at delta +0.0667, `loo_min` +0.0273. Under D14 the king's weights are public, so "the king
plus two benchmark-specific overrides" is a buildable submission, and `eps` does not price it.

The median refuses both by construction — ten identical benchmarks put it at exactly 0, ten small
losses put it at −0.02 — and neither condition implies the other, which is why both are kept:

  * `[+0.315, +0.315, 0×10]` passes leave-one-out at +0.0286 and fails the median at exactly 0;
  * `[+0.90, +0.01×6, −0.10×5]` passes the median at +0.0100 and fails leave-one-out at −0.0400.

Leave-one-out asks *"is this win carried by one benchmark?"*; the median asks *"did more than half
the corpus move the right way?"*. A two-benchmark specialist passes the first and fails the second;
a one-benchmark landslide with the remaining eleven splitting six-to-five positive would pass the
second and fail the first.

WHAT THAT PARAGRAPH IS WORTH AT N = 3 — MEASURED 2026-09-01, `scripts/breadth_power.py`, and
this rule was KEPT on the strength of it.
"A two-benchmark specialist fails the median" is TRUE AT TWELVE BENCHMARKS AND FALSE AT THREE: two of
three is a majority, so `[+0.0825, +0.0825, 0]` has a median of +0.0825 and both conditions pass.
N = 3 is what this was parameterised at; what the corpus actually is, is an open owner decision
(the corpus record §4 decision 2 — its §2 currently makes the runnable corpus N = 2, and this
module must not pre-empt that; the N = 2 case is the paragraph below). Measured at
n_b = 83: that shape is crowned 55.5% against condition 1's own 56.5% — breadth does essentially
nothing — and **the rate rises with the slice** (62.5% at 250, 73.8% at 1,000, 99.2% at 16,000),
because the median of `[x, x, d₃]` is `x` for any `d₃` and larger slices only make condition 1 more
reliable at letting the shape through. *Buying tasks does not make this gate safer; buying benchmarks
does — N = 5 at the same total budget moves it 55.5% → 52.8% for nothing.*

AND AT N = 2 — MEASURED 2026-09-01, same harness, n_b = 125, 400 trials — ONE OF THE TWO BREADTH
CONDITIONS DIES AND THE OTHER GETS STRONGER, which is the opposite of the collapse the shape of the
arithmetic suggests. Condition 3 goes: the median of two deltas is their mean, so it IS the aggregate
(bit-identical to `delta` in 87–99% of trials, the residue being where `BREADTH_FLOOR` zeroes one),
and condition 1 already requires that above `eps`. Condition 2 becomes *"win BOTH benchmarks"* and
does more work than it does at N = 3: a one-benchmark specialist goes 88.2% → 51.2% against condition
1 alone, a predator genuinely −0.02 on the other goes 93.8% → 31.8%, and a genuine uniform +0.0825 is
refused 0.5% of the time. The two-of-three shape this section is about has no N = 2 analogue — at
N = 2 a specialist wins one of two, which is exactly what leave-one-out refuses. **So neither
condition may be dropped on "N is small": at N = 3 the median is what catches concentration and at
N = 2 it is leave-one-out.**

The rest of the same measurement is why nothing here changed. Against condition 1 alone, breadth
refuses **0.0–1.7%** of genuinely broad challengers and **26–57%** of one-benchmark specialists
(`specialist1_loud` 50.2% vs 82.5%; a predator genuinely −0.02 elsewhere, 24.8% vs 57.2%) — so it is
nearly free and it does real work on every concentrated shape except the one above. `eps`, not
breadth, is what refuses broad challengers. Four replacement families were measured and all four
fail; they are recorded in the corpus record §3 so nobody re-proposes them, and the closest one
— a bootstrap homogeneity test against "the true advantage is uniform, equal to the observed
aggregate" — is refused because tasks are NOT iid (R2E-Gym draws many per repository, so its
calibration guarantee fails by 5×), because two free miner knobs restore a pure lookup table to ~50%,
and because `max − median` at N = 3 is blind to the worst benchmark and crowns `[+0.183, +0.183,
−0.200]` at 51.5% where the rule below crowns it at 19.0%.

WHY THE BOOTSTRAP IS PAIRED. Both arms face the identical slice (§5.2), so each resample draws the
SAME tasks for both and scores both on them. Task difficulty then cancels *inside* every resample
instead of deciding it — slice luck is the thing that mis-crowned 71% of per-epoch reigns in this
project's own simulation, and pairing is what removes it. It is seeded from the window nonce so the
verdict is reproducible from the published reveal rather than from the validator's memory.

The resample is stratified WITHIN each benchmark. Two reasons, and the first is fatal rather than
stylistic: the score weights benchmarks equally (§5.1), so a global resample would hand each
benchmark a random task count — sometimes zero, which makes `quality_b` a mean of nothing — and
would inject between-benchmark count noise into a statistic defined to have none. The slice itself
is drawn stratified (§6.3b); the resample matches it.

AND THE RESAMPLE UNIT IS A CLUSTER, NOT A TASK — MEASURED 2026-09-01. Tasks are drawn uniformly
within a benchmark but they are not independent. R2E-Gym is 6,812 gradeable tasks over ELEVEN
repositories (`benchmarks/r2egym.py`), so an 83-task stratum holds ~7.5 tasks of each, and a
challenger that suits a repository is better on all of them at once. `scripts/breadth_power.py`
§4a found that structure breaking a candidate breadth rule's calibration by 5x; the measurement
below asks the prior question, about `lcb` — condition 1, the gate every crown passes through.

Under the null the challenger's TRUE advantage is 0 on every benchmark, so a calibrated 5th
percentile puts `lcb > 0` on 5% of duels. Tasks in blocks of `block`, each block carrying a mean-zero
advantage offset of size `tau`; N = 3, n_b = 83, 400 trials, the same data resampled both ways:

                     P(lcb>0)              bootstrap SE       true
    block   tau    iid  ->  clustered    iid  ->  clustered    SD(delta)
        1  0.00   0.055      0.055      0.0278    0.0278       0.0282   iid control
        1  0.20   0.055      0.055      0.0289    0.0289       0.0295   UNCLUSTERED control
        5  0.20   0.095      0.048      0.0290    0.0349       0.0349
        8  0.20   0.107      0.060      0.0293    0.0388       0.0419   R2E-Gym's block size
       10  0.20   0.177      0.098      0.0286    0.0403       0.0462
       20  0.30   0.265      0.090      0.0293    0.0566       0.0649

The second row is the control that names the cause: identical extra variance with no correlation
does not move the rate at all, so what breaks the interval is the CLUSTERING and not the noise. Read
the SE columns against the last one — the iid bootstrap's SE is flat at ~0.029 while the true SD
reaches 0.065. **The bootstrap could not see its own error**, so no published diagnostic,
`noise_floor` included, would have shown a validator this happening; resampling clusters is what puts
that number back in touch with reality (0.0293 -> 0.0566 on the last row).

Both control rows are unchanged to four decimals, which is the bit-for-bit claim below arriving as a
measurement.

WHAT THE FIX DOES NOT DO: restore exact nominal coverage when there are few clusters. A cluster
bootstrap's effective sample size is the CLUSTER count, not the task count — 83 tasks in blocks of 20
is FIVE clusters — and the percentile interval under-covers there for the ordinary small-sample
reason. Measured at 200 trials — 1.5pp of binomial noise, so these are a trend and not digits to hold
against the 400-trial table — holding the block size and raising the slice: block 8 goes 0.085 ->
0.075 -> 0.030 as the clusters go 11 -> 32 -> 100, and block 20 goes 0.100 -> 0.060 at 5 -> 25. So
the residual tracks the CLUSTER COUNT and not the resample unit. R2E-Gym has ELEVEN repositories,
which is the first of those columns: expect ~0.06 rather than 0.05, and read `lcb` on a clustered
benchmark as approximately rather than exactly a 5% bound.

`tau` IS ASSUMED AND THE 0.107 IS NOT A PRODUCTION NUMBER. Nothing in this repo has measured how much
of a challenger's advantage is per-repository, and until a clustered benchmark is admitted and duelled
nothing can. What is certain is the DIRECTION: clustering widens the true sampling distribution and
leaves the iid bootstrap exactly where it was, so the shipped `lcb` is anticonservative by an unknown
amount that is never negative. That is why the fix is worth taking at an unmeasured magnitude — it
errs the way `eps` errs. **A wider interval refuses crowns; it cannot manufacture one**, and only a
manufactured crown is irreversible in the ledger (§5.2b, D10).

So `BenchmarkPair.groups` names each task's cluster and the bootstrap draws whole clusters — within
the benchmark and paired, both unchanged. It is OPTIONAL, and absent means every task is its own
cluster, which is bit-for-bit the resample that shipped: the identical `rng.integers` call on the
identical stream, asserted rather than claimed
(`test_an_ungrouped_duel_is_bit_for_bit_the_resample_that_shipped`). LiveCodeBench and HLE tasks
really are independent and supply nothing. An invented grouping is worse than none — it would widen
an interval for a structure that is not there, and refuse real challengers for it.

WHY LEAVE-ONE-OUT AND NOT A SIGN TEST — A CORRECTION THIS MODULE MUST NOT LOSE. An earlier draft of
the spec gated breadth on "≥7 of 12 benchmarks, sign test p < 0.10". Those two clauses contradict
each other and the rule gated nothing: P(≥7 of 12) under the null is 0.387, so seven of twelve
happens by coin flip more than a third of the time. Raising the bar to ≥9 (p = 0.073) is the wrong
repair — with ~20 tasks per benchmark the per-benchmark win/loss bit is very noisy, so a genuinely
broad +3-point challenger can land 8–4 by luck and be refused. A sign test over twelve noisy bits is
weak in BOTH directions, because it throws away the graded scores it was computed from.

Leave-one-out keeps the magnitudes. It asks the question the failure mode actually poses — *is this
win carried by one benchmark?* — and demands nothing about winning everywhere. It is also legible to
a miner: **your win must survive losing your best subject.** The sign test is still computed and
published as a diagnostic (`sign_wins`, `sign_p`); it just must never gate.

THE MEDIAN IS NOT THAT SIGN TEST WEARING A NEW HAT, AND THE DISTINCTION IS NARROW ENOUGH TO STATE.
A median over twelve above zero is approximately *"wins a majority"* — near the ≥7-of-12 the table
above dismisses. What §5.3 rejects is demanding **statistical significance** from twelve noisy bits:
`p < 0.10` needs ≥9 of 12, and that refuses real challengers. The measured archetype duel is exactly
that case — the honest cascade beats the spendthrift 8–4 at one-sided p = 0.1938, so the draft rule
would have turned it away — and it clears the median comfortably at +0.1716. A bare majority is a
far weaker bar than significance, and it gates on a score rather than on a p-value, so nothing here
re-introduces the instrument §5.3 threw out. It also keeps the magnitudes where they matter: the
median is an order statistic *of the deltas*, so at an even benchmark count it is the mean of the
two middle ones and `median > 0` is NOT exactly ≥7 of 12 — a 6–6 split passes when the smallest win
outweighs the smallest loss. At an odd count (a narrowed window, §6.3b) it is exactly a strict
majority.

Note the deliberate asymmetry between the three conditions: the aggregate must clear `eps`, both
breadth conditions need only clear zero. `eps` is the anti-noise/anti-copy margin on the headline
comparison; breadth is a structural question about where the win came from. Requiring `eps` on all N
subsets as well would silently multiply the bar and refuse broad challengers for being merely broad.

WHAT `eps = 0.05` BUYS AND WHAT IT COSTS (§5.2b, D10). At ~250 tasks per arm the point estimate's
standard error is ≈0.028, so `eps = 0.05` behaves like a **~7–8 point bar**: a challenger genuinely
at +0.05 clears the point estimate about half the time and its lower bound sits barely above zero.
That is the conservative direction on purpose — a too-large `eps` refuses real challengers, a
too-small one crowns noise, and only the second is irreversible in the ledger. The number to watch
is `band / eps`: once a challenger sits within `eps` of the achievable ceiling every later duel
lands inside the margin and the crown ossifies, so **`band / eps` bounds how many coronations this
arena can ever have.** The band is unknown for twelve agentic benchmarks over ~300 models and cannot
be borrowed from this repo's single-turn measurements; production reference arms measure it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from typing import Protocol

import numpy as np

from ..gateway import signing
from .config import BREADTH_FLOOR, EPS, N_BOOT


class FinalB(Protocol):
    """The ONLY thing this module needs from `score.py`: §5.1b's per-benchmark score, as a function
    of `(benchmark, mean graded quality, mean per-task spend)`.

    A callable rather than a table of computed `final_b` values because the paired bootstrap
    re-scores both arms on every resample, and a number cannot be resampled. `λ_b` and `C_b` live
    behind it and the duel never sees them — which is also why `C_b` is free to be any pinned
    constant (§5.1b): under a paired comparison it only rescales `λ_b`.

    Called positionally, so the implementation may name its parameters however it likes.
    """

    def __call__(self, benchmark: str, quality: float, spend: float) -> float: ...


@dataclass(frozen=True)
class BenchmarkPair:
    """One benchmark's per-task outcomes for BOTH arms, row-aligned.

    Both arms live in one object because the duel is paired: row `i` is the same task for the king
    and for the challenger (§5.2). A structure that could carry two different task lists would let
    an asymmetric exclusion through unnoticed, and §6.3c is explicit that a grader failure drops the
    task from *both* arms — one arm scored on an easier slice is not a duel. The length check below
    is that invariant made checkable rather than assumed.
    """

    benchmark: str
    king_quality: np.ndarray        # (n_b,) graded per-task scores, partial credit where supported
    king_spend: np.ndarray          # (n_b,) dollars spent on that task
    challenger_quality: np.ndarray
    challenger_spend: np.ndarray
    # The cluster each task belongs to (`TaskSpec.group` — R2E-Gym's repository), which is what the
    # bootstrap resamples. None is the default and means every task is its own cluster: the resample
    # that shipped, bit for bit, and the only honest answer for a benchmark of independent tasks.
    groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.king_quality)
        if n == 0:
            raise ValueError(f"benchmark {self.benchmark!r} contributes no tasks")
        for name in ("king_spend", "challenger_quality", "challenger_spend"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"benchmark {self.benchmark!r}: {name} covers {len(getattr(self, name))} tasks "
                    f"but king_quality covers {n} — the arms are not paired")
        if self.groups is not None:
            if len(self.groups) != n:
                raise ValueError(
                    f"benchmark {self.benchmark!r}: groups covers {len(self.groups)} tasks but "
                    f"king_quality covers {n} — the labels are not row-aligned")
            if any(group is None for group in self.groups):
                # A half-labelled benchmark would be resampled as if the unlabelled tasks were
                # independent, which is the anticonservative interval arriving by the back door.
                # Either the benchmark has a grouping or it has none (see the module docstring).
                raise ValueError(
                    f"benchmark {self.benchmark!r}: {self.groups.count(None)} of {n} tasks carry no "
                    "group — a partial grouping is a wrong grouping; pass None for no grouping")

    @property
    def n(self) -> int:
        return len(self.king_quality)


@dataclass(frozen=True)
class DuelVerdict:
    """One duel's published record. Everything the reveal needs, and nothing stored twice.

    `challenger_wins`, `loo_min`, `median` and `benchmarks` are DERIVED rather than stored: a
    recorded verdict that disagreed with the numbers it was computed from would be undetectable, and
    this project has the render-determinism precedent for "a duplicated field is a field that can
    disagree".
    """

    delta: float                                  # challenger − king in `final`, point estimate
    lcb: float                                    # 5th percentile of the paired bootstrap delta
    by_benchmark: tuple[tuple[str, float], ...]   # per-benchmark delta, published every window
    loo: tuple[tuple[str, float], ...]            # (benchmark dropped, aggregate delta without it)
    sign_wins: int                                # diagnostic only — see the module docstring
    sign_p: float
    n_tasks: int                                  # per arm, across the whole slice
    eps: float
    # Per-benchmark bootstrap SE. A PUBLISHED DIAGNOSTIC, gated on by nothing — see
    # `config.BREADTH_FLOOR`. It is ~10x that floor at the slice the corpus can actually fill
    # (0.049 at 83 tasks/benchmark, measured), which is what makes a per-benchmark SIGN a noisy
    # read; publishing it every window is what keeps that visible rather than implicit. It does NOT
    # mean breadth is unjudgeable there — the module docstring's 2026-09-01 measurement is the
    # authority on what the rule does and does not catch at N = 3. Defaulted so a caller building a
    # verdict by hand (tests, replays of an older reveal) is not forced to invent one.
    noise_floor: tuple[float, ...] = ()

    @property
    def benchmarks(self) -> tuple[str, ...]:
        """The benchmarks actually PRESENT in this window's slice (§6.3b). A window built without a
        benchmark that could not supply its minimum task count is narrowed, and the reveal has to
        say so — otherwise leave-one-out silently ran over a different set and verdicts across
        windows stop being comparable."""
        return tuple(name for name, _ in self.by_benchmark)

    @property
    def counted(self) -> tuple[tuple[str, float], ...]:
        """Per-benchmark deltas with sub-noise movement zeroed — what BREADTH is judged on.

        *** WHY CONDITIONS 2 AND 3 CANNOT USE THE RAW DELTAS. ***

        Both gate at a strict `> 0`, and `final_b` is CONTINUOUS IN SPEND. So on a benchmark where a
        challenger routes identically to the king it can still manufacture a positive delta by
        spending a hair less. Measured by the red team: the king plus two benchmark-specific
        overrides plus a **1e-5 dollar** spend reduction on four others is CROWNED — delta +0.1000,
        loo_min +0.0545, median +3.03e-06. That is precisely the submission D16 was written to
        refuse, bought for a hundredth of a cent.

        The same defect appears as a RATE, which is how it would actually be met. Replace a fixture's
        bit-exact zeros with ordinary measurement noise — which is all real arms ever produce, since
        two policies agreeing on every routing decision still differ in tokens and therefore dollars
        — and a pure ONE-benchmark specialist is crowned **38-49% of the time**, against 0/200 at
        sigma = 0. `loo_min` and `median` had both become coin flips on the sign of noise.

        So a benchmark only counts toward breadth once its delta clears its own bootstrap noise.
        `delta` and `lcb` (condition 1) deliberately keep the RAW values: the aggregate should report
        what the challenger actually did, while breadth asks the different question of where it
        moved something real.

        **THE FLOOR PRICES THE MANUFACTURED ATTACK, NOT THE BOUGHT ONE** — measured 2026-09-01,
        `scripts/breadth_power.py` §4b. A hundredth of a cent is zeroed; spending genuinely LESS
        off-subject is not. A one-benchmark specialist that routes the other two identically to the
        king and spends half as much there lifts each untouched delta by ~0.035, thirty times this
        floor, and is crowned 56.5% against condition 1's 57.2% — the same hole, repriced from 1e-5
        dollars to ~$0.83 a window. Raising the floor to meet it is not the repair: at n_b = 83 the
        per-benchmark bootstrap SE is 0.049, so a floor large enough would refuse a genuine uniform
        +0.03 (`test_a_uniform_three_point_challenger_is_refused_by_eps_not_by_breadth`).
        """
        return tuple((name, d if abs(d) > BREADTH_FLOOR else 0.0)
                     for name, d in self.by_benchmark)

    @property
    def loo_min(self) -> float:
        """The worst leave-one-out aggregate, computed over `counted` (§5.2 condition 2)."""
        counted = [d for _, d in self.counted]
        if len(counted) < 2:
            return 0.0
        return min(float(np.mean(counted[:i] + counted[i + 1:])) for i in range(len(counted)))

    @property
    def loo_dropped(self) -> str:
        """The benchmark whose removal costs the challenger most — i.e. its best subject."""
        return min(self.loo, key=lambda item: item[1])[0]

    @property
    def median(self) -> float:
        """§5.2 condition 3: the TYPICAL benchmark's delta, published beside `loo_min` and the sign
        test so a disagreement between the three is visible rather than a matter of recollection.

        Taken over the benchmarks present (§6.3b), not a hard-coded twelve, exactly as leave-one-out
        is. At an even count this is the mean of the two middle deltas — which is why it is not the
        sign test's ≥7-of-12 in disguise: a 6–6 split passes when the smallest win outweighs the
        smallest loss.

        Computed over `counted`, not the raw deltas — see that property for the two measured attacks
        this closes. It is the median that does most of the work: with sub-noise movement zeroed, a
        challenger that genuinely moved only two benchmarks has a median of exactly 0.0 and is
        refused, whether the other benchmarks were untouched or merely noisy.
        """
        return float(np.median([d for _, d in self.counted]))

    @property
    def challenger_wins(self) -> bool:
        return (self.delta > self.eps and self.lcb > 0.0
                and self.loo_min > 0.0 and self.median > 0.0)


def sign_test(deltas: Sequence[float]) -> tuple[int, float]:
    """Benchmarks won, and the one-sided binomial p-value. **A DIAGNOSTIC. IT DOES NOT GATE.**

    Published because per-benchmark direction is worth reading, and computed here so the reveal
    carries the number that the earlier draft would have gated on — which makes it visible when it
    and the breadth rule disagree, rather than a matter of anyone's recollection.

    Ties count against the challenger (`> 0`, not `>= 0`) and stay in the denominator, which is the
    same direction as every other tie in this mechanism: a copy of the king ties on all N benchmarks
    and scores p = 1.0, no special case required.
    """
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0.0)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
    return wins, p


def _arm(pair: BenchmarkPair, challenger: bool) -> tuple[np.ndarray, np.ndarray]:
    if challenger:
        return pair.challenger_quality, pair.challenger_spend
    return pair.king_quality, pair.king_spend


def _finals(pairs: Sequence[BenchmarkPair], final_b: FinalB, rows: Sequence[np.ndarray],
            *, challenger: bool) -> list[float]:
    """`final_b` per benchmark, on the given per-benchmark row selection."""
    out = []
    for pair, sel in zip(pairs, rows):
        quality, spend = _arm(pair, challenger)
        out.append(final_b(pair.benchmark, float(quality[sel].mean()), float(spend[sel].mean())))
    return out


def _clusters(pair: BenchmarkPair) -> tuple[np.ndarray, ...] | None:
    """The rows of each distinct group, or None when every task is its own cluster.

    Computed once per duel rather than once per resample, and ordered by LABEL for the reason `duel`
    sorts benchmarks: the resample is then a function of the set of groups present and never of the
    order the tasks happened to land in the slice.
    """
    if pair.groups is None:
        return None
    labels = np.asarray(pair.groups)
    return tuple(np.flatnonzero(labels == group) for group in np.unique(labels))


def _resample(pair: BenchmarkPair, clusters: tuple[np.ndarray, ...] | None,
              rng: np.random.Generator) -> np.ndarray:
    """One benchmark's resampled rows: whole CLUSTERS, with replacement, within the benchmark.

    The ungrouped branch is the draw that shipped, unchanged — the same `rng.integers(0, n, size=n)`
    call, consuming the same stream in the same order — so adding a group id to one benchmark cannot
    move any verdict computed on the others.

    A clustered resample does NOT return `n` rows: clusters differ in size, so the row count varies
    between resamples. That is the cluster bootstrap rather than a defect — the statistic is a mean
    within the benchmark, and the varying composition is precisely the extra variance the iid draw
    was blind to.

    Its degenerate case is a benchmark drawn entirely from ONE cluster: every resample then returns
    that cluster, the benchmark contributes no bootstrap variance, and `lcb` is too narrow again. Not
    guarded, because it is not reachable at the corpus that exists — R2E-Gym's largest repository is
    34% of the usable pool, so an 83-task stratum landing wholly inside it has probability ~1e-39 —
    and a raise would refuse a live window for an arithmetic curiosity. The general statement of the
    same thing is in the module docstring: the interval's effective sample size is the CLUSTER count.
    """
    if clusters is None:
        return rng.integers(0, pair.n, size=pair.n)
    picked = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in picked])


def _delta(pairs: Sequence[BenchmarkPair], final_b: FinalB,
           rows: Sequence[np.ndarray]) -> tuple[float, list[float]]:
    """The paired aggregate delta, and the per-benchmark deltas it is the mean of.

    EQUAL WEIGHT PER BENCHMARK, NEVER PER TASK (§5.1). Averaging raw tasks would let whichever
    benchmark contributed the most tasks — or has the highest variance — silently dominate the
    total, which is an implicit weighting nobody chose and is directly at odds with rewarding
    generalisation.
    """
    king = _finals(pairs, final_b, rows, challenger=False)
    chal = _finals(pairs, final_b, rows, challenger=True)
    per_benchmark = [c - k for c, k in zip(chal, king)]
    return float(np.mean(per_benchmark)), per_benchmark


def duel(pairs: Sequence[BenchmarkPair], final_b: FinalB, *, eps: float = EPS,
         n_boot: int = N_BOOT, nonce: str | None = None, seed: int | None = None) -> DuelVerdict:
    """The three-condition verdict (§5.2).

    `nonce` (the window beacon) seeds the bootstrap so the verdict is reproducible from the reveal;
    `seed` exists for offline experiments only.

    Both breadth conditions run over the benchmarks PRESENT in `pairs`, never a hard-coded twelve
    (§6.3b):
    admission drops benchmarks that cannot carry weight and rotation adds them, so a mechanism
    pinned to 12 would break the first time a benchmark failed its gate.
    """
    # Sorted, for the reason `score.py::_by_benchmark` sorts: the aggregate is a mean of means, so
    # its last bits depend on summation order, and the bootstrap consumes the rng per benchmark in
    # sequence. Sorting makes the verdict a function of the admitted SET rather than of the order
    # results happened to land in.
    #
    # It does NOT make `delta` bit-identical to the difference of two separately-aggregated arm
    # scores: this module subtracts per benchmark and then averages (which is what leave-one-out
    # needs), and the two orders of operation agree to ~1e-16, measured. Identical in exact
    # arithmetic, so no verdict can turn on it at eps = 0.05 — but a reveal that publishes both
    # numbers should not claim they are the same bytes.
    pairs = tuple(sorted(pairs, key=lambda p: p.benchmark))
    if len(pairs) < 2:
        raise ValueError(
            f"breadth cannot be judged on {len(pairs)} benchmark(s) — leave-one-out would score the "
            "challenger on an empty slice, and a NaN must never decide a crown")
    names = [p.benchmark for p in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"benchmark appears twice in the slice: {names} — equal weight per "
                         "benchmark (§5.1) would silently double one benchmark's weight")
    if seed is None:
        if nonce is None:
            raise ValueError("pass the window nonce (production) or an explicit seed (offline)")
        seed = int(signing.sha256_hex(f"{nonce}|v3-duel-bootstrap")[:8], 16)

    all_rows = [np.arange(p.n) for p in pairs]
    delta, per_benchmark = _delta(pairs, final_b, all_rows)

    # Leave-one-out on `final`, not on quality (§5.1b): breadth is required in the TRADEOFF, so a
    # challenger that is broadly more accurate only by outspending fails here as it should.
    loo = tuple(
        (pairs[i].benchmark, _delta(pairs[:i] + pairs[i + 1:], final_b,
                                    all_rows[:i] + all_rows[i + 1:])[0])
        for i in range(len(pairs)))

    rng = np.random.default_rng(seed)
    clusters = [_clusters(p) for p in pairs]
    boot = np.empty(n_boot)
    boot_per_benchmark = np.empty((n_boot, len(pairs)))
    for b in range(n_boot):
        # Paired (the SAME rows for both arms), stratified within the benchmark, and drawn by
        # cluster where the benchmark has one — see the module docstring for all three.
        rows = [_resample(p, c, rng) for p, c in zip(pairs, clusters)]
        boot[b], boot_per_benchmark[b] = _delta(pairs, final_b, rows)
    lcb = float(np.percentile(boot, 5.0))

    # The per-benchmark bootstrap SE, from the resampling already running. PUBLISHED, NOT GATED ON:
    # a 1-SE gate was implemented and measured to refuse a genuine uniform +0.03 challenger, because
    # at ~20 tasks per benchmark that SE is order 0.11. Breadth gates on `BREADTH_FLOOR` instead;
    # this number is what makes the remaining sample-size gap visible every window.
    noise_floor = tuple(float(s) for s in boot_per_benchmark.std(axis=0, ddof=1))

    wins, p_value = sign_test(per_benchmark)
    return DuelVerdict(delta=delta, lcb=lcb,
                       by_benchmark=tuple(zip(names, per_benchmark)),
                       noise_floor=noise_floor, loo=loo,
                       sign_wins=wins, sign_p=p_value,
                       n_tasks=sum(p.n for p in pairs), eps=eps)


@dataclass(frozen=True)
class Contender:
    """A queued challenger and its verdict, for batch coronation."""

    hotkey: str
    commit_block: int
    verdict: DuelVerdict


def champion(contenders: Sequence[Contender]) -> Contender | None:
    """The window's crown: the largest paired delta among the challengers that WON (§8 step 4).

    **Ties break on earliest commit block**, then hotkey — exactly the `(commit_block, hotkey)`
    order the queue is evaluated in (§8b.1), so the tiebreak is a total order and the reveal is
    reproducible rather than dependent on the order results happened to land in.

    This selection is only meaningful because every duel in a window ran against ONE king arm
    (§5.2a, D13): re-running the king per challenger would compare A against king-run-1 and B
    against king-run-2, and run-to-run variance could then crown the worse of the two.

    A near-copy of the king never reaches here: it ties, ties fail the strict `eps` beat, and the
    crown stays where it is. That converts anti-copy from a detection problem (measurement 11 found
    copier agreement and honest convergence to be the same signal) into an economic one.
    """
    winners = [c for c in contenders if c.verdict.challenger_wins]
    if not winners:
        return None
    return min(winners, key=lambda c: (-c.verdict.delta, c.commit_block, c.hotkey))
