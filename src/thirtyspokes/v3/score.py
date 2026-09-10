"""The per-arm score: quality minus what that quality cost (§5.1, §5.1b, D8).

Two rules carry this module, and in both cases the obvious implementation is the wrong one.

EQUAL WEIGHT PER BENCHMARK, NEVER PER TASK (§5.1). `quality` is the mean over admitted benchmarks
of `quality_b`, not the mean over tasks. Averaging raw tasks would let whichever benchmark
contributed the most of them — or has the highest variance — silently dominate the total. That is an
implicit weighting nobody chose, and it is directly at odds with rewarding generalisation, which is
the "Spirit" property the whole architecture is named for. The two readings coincide when slices are
balanced (§6.3b); the aggregation must not depend on their coinciding.

LAMBDA IS MEASURED, NOT CHOSEN (§5.1b). The two obvious ways to price cost are both broken:
`quality / cost` is maximised at the bottom of the quality range, so it rewards frugal
incompetence; `quality − λ·cost` with a hand-picked λ buries the most important number in the
mechanism — the exchange rate between accuracy and dollars — in a constant somebody guessed.
Deriving λ_b from the fixed-policy sweep instead makes **always-cheapest and always-strongest score
exactly the same**: the score's iso-lines run parallel to the line joining them, so buying quality
at the pool's own rate earns nothing, and a policy scores above both extremes if and only if it
sits *above* that line. That is the definition of routing skill, and here it is the objective
rather than a commentary on it. It is also what defuses the miner-set allowance asymmetry of §4 —
outspending the king into a win now scores zero.

PER BENCHMARK, NOT GLOBAL. The accuracy-per-dollar exchange rate genuinely differs between a
terminal-ops benchmark and a cyber one — different task lengths, different model spreads. A single
global λ would price them all at one benchmark's rate, so `final_b` values would not be
commensurable and the equal-weight mean above would be adding unlike things.

`C_b` IS FREE, WHICH IS WHY IT IS PINNED. Under the paired duel of §5.2 the delta is
`(q_ch − q_king) − Σ_b (λ_b/C_b)(spend_ch,b − spend_king,b)`, so `C_b` only ever rescales `λ_b` and
any fixed positive choice is sound. An earlier draft normalised by King₀'s live spend, which would
have forced a full King₀ arm every window purely to obtain a denominator — and a live baseline
drifts where a pinned constant does not. `pinned_cost` measures it once, from the same sweep, as
the mean dollars per task; that choice puts `relspend ≈ 1` for a typical policy, so a published
`λ_b` reads as "quality forgone per typical task's worth of spend" and is comparable across
benchmarks. **`C_b` is free EVERYWHERE, including in the guard** — see `RELSPEND_SPREAD_FLOOR`,
which used to be the one place it was read as a scale, and was therefore a way to switch a guard
off by choosing a denominator.

A GUARDED BENCHMARK IS EXCLUDED, NOT SCORED ON PURE QUALITY (§5.1b). An earlier draft clamped `λ_b`
to 0 when a guard fired, which is arithmetically "score this benchmark on quality with spend
unpriced" — so the section's own claim, that outspending the king buys nothing, was false exactly
where the guards fire. Two such benchmarks satisfy the breadth rule (§5.2), so a challenger
byte-identical to the king on ten benchmarks and simply burning its allowance on the two flagged
ones was crowned at thousands of times the king's spend (measured: delta +0.0833). And the
dominator guard is not a corner — it fires on the condition this repository has measured five times
(ROUTING_MEASUREMENTS). A fired guard says the benchmark could not be priced from the sweep, which
is a benchmark §6.1's admission gate should have dropped, and a benchmark the admission gate should
have dropped must not price a crown.

WHY A MEAN OF PER-BENCHMARK MEANS IS SAFE HERE, GIVEN §17. This repository distrusts per-category
averaging for a measured reason: averaging per-fold capture *ratios* reported −2.7761 where the
pooled truth was −0.2638, because one intent's near-zero band turned two misplaced asks into a
capture of −20 (ROUTING_MEASUREMENTS §17). The failure is division by a measured per-category
denominator that can approach zero. Nothing aggregated here is a ratio: `quality_b` is a mean of
bounded graded scores and `final_b` subtracts a priced spend from one. The only division by a
measured quantity in this module happens once, at λ derivation — which is exactly what
`RELSPEND_SPREAD_FLOOR` guards.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .types import EpisodeResult, TaskSpec

# The fixed-policy spend spread below which λ_b is refused (§5.1b guard), as a FRACTION OF THE TWO
# SWEEP ARMS' OWN MEAN PER-TASK SPEND. The sweep that fits λ_b is a ~50-task probe (§6.1), so each
# arm's mean per-task spend carries a standard error worth a few percent of it; a spread under 5% is
# not separable from that noise, and λ_b divides by it. Since λ_b is pinned per corpus rather than
# recomputed per window, a noisy fit would be frozen into every score until the next re-derivation.
#
# MEASURED AGAINST THE SWEEP, NOT AGAINST `C_b`, and that is the whole point. An earlier draft
# expressed this floor in units of the caller's `C_b`, which made it the ONE place the otherwise-free
# normaliser was read as a scale — so `C_b` deflated by 1e-9 (a hand-pinned λ_b/C_b pair at M3b
# rather than one derived from `pinned_cost`) made `spread` grow without bound, the guard fell
# silent, λ_b was fitted on a spend difference of 1e-9, and the λ invariant degraded to 6e-8, past
# the 1e-9 bar. Silently: the published `Exchange` carried no flag. The ratio below is scale-free, so
# `C_b` now cancels out of the guard exactly as it cancels out of the score.
RELSPEND_SPREAD_FLOOR = 0.05

# The two guards, as the words the window publishes. A benchmark that changed what miners are
# optimising WITHOUT TELLING THEM would break the "Kind" property outright, so the exclusion carries
# its reason with it rather than the benchmark simply going missing from the score.
LAMBDA_UNDEFINED = ("lambda-undefined: fixed-policy relspend spread below floor; "
                    "this benchmark is excluded from scoring")
DOMINATOR = ("dominator: the best fixed policy buys no quality over the cheapest; "
             "this benchmark is excluded from scoring")


@dataclass(frozen=True)
class Exchange:
    """One benchmark's measured accuracy/dollars exchange rate — `λ_b` and its normaliser `C_b`.

    Pinned per corpus, not recomputed per window (§5.1b): it is the rate miners train against, and
    drifting it silently every window would make scores incomparable across windows and move the
    target under everyone. Re-derive when the corpus or catalog changes materially, and announce it.

    `flags` is empty on a healthy benchmark and otherwise names every guard that fired. **A
    benchmark with any flag is excluded from scoring** (`score_arm`), so its `lam` is not a price and
    is never read; it is left at 0 rather than at something arbitrary. The reason travels with the
    number because the exclusion has to be announced (§5.1b) — a benchmark that merely vanished from
    the mean would move a verdict invisibly.
    """

    benchmark: str
    lam: float
    c_per_task: float
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkScore:
    """`quality_b`, `spend_b` and `final_b` for one benchmark of one arm.

    `n_tasks` is carried but never enters the score — it is what lets a reader of the reveal check
    that the slice was balanced (§6.3b) and that a benchmark cleared its minimum count. The whole
    point of §5.1 is that this number does not weight anything.
    """

    benchmark: str
    n_tasks: int
    quality: float
    spend: float
    final: float


@dataclass(frozen=True)
class ArmScore:
    """One arm over one slice. `final` is the reported score (§5.1b); `quality` is published beside
    it because a win at equal quality and lower spend and a win on quality alone are different
    products and the reveal should say which happened.

    `per_benchmark` is ordered and complete: the leave-one-out breadth rule (§5.2) runs over exactly
    the benchmarks present here, so a narrowed slice is visible rather than silently scored over a
    different set (§6.3b).

    `excluded` names the benchmarks in this arm that a §5.1b guard removed, with the guards that
    fired. It is published rather than dropped for the same reason: exclusion changes the denominator
    of both means above and changes the set leave-one-out runs over, so a silent one would move a
    verdict where no reader could see it.
    """

    per_benchmark: tuple[BenchmarkScore, ...]
    quality: float
    final: float
    excluded: tuple[tuple[str, tuple[str, ...]], ...]


def pinned_cost(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """`C_b` per benchmark from a fixed-policy sweep: mean dollars per task (§5.1b, M3a).

    Measured over whatever arms are handed in — pool both fixed policies, so the constant is a
    property of the benchmark rather than of one policy's appetite. The value is free (any fixed
    positive choice gives identical verdicts, pinned by test), so what is being bought here is a
    definite, reproducible number and a readable scale for `λ_b`, not correctness.
    """
    return {benchmark: spend for benchmark, (_, _, spend) in _by_benchmark(results).items()}


def derive_exchange(cheapest: Sequence[EpisodeResult], top: Sequence[EpisodeResult],
                    c_per_task: Mapping[str, float]) -> dict[str, Exchange]:
    """`λ_b` from the fixed-policy sweep (§5.1b, §6.1):

        λ_b = (q_top,b − q_cheapest,b) / (relspend_top,b − relspend_cheapest,b)

    This is the slope of the benchmark's own cost/quality tradeoff, and fitting the score's price to
    it is what makes the two arms tie in `final`. `c_per_task` is an input rather than a
    re-measurement so that the pinned `C_b` and the fitted `λ_b` cannot drift apart: they are
    published as a pair and fixed together (§5.1b, "λ must be RE-DERIVED after M3b").

    WHICH TWO ARMS IS NOT THIS FUNCTION'S DECISION — `reference.fit_pool` makes it, once, for every
    caller. `top` is King₀, the best fixed policy by quality (§5.5), so the line the score's
    iso-lines run parallel to joins the floor to the best thing money buys on this pool, and a
    challenger scores above King₀ only by buying quality at a better rate than King₀ itself does.
    When King₀ is the cheapest policy there is nothing above the floor to fit to; `fit_pool` then
    hands in always-strongest and the dominator guard below fires, which is the finding.

    MEASURED 2026-09-06, AND IT IS WHY THE PAIR IS KING₀ RATHER THAN ALWAYS-STRONGEST. On the live
    probe pool the single strongest model bought nothing over the cheapest on either benchmark —
    paired difference +0.010 ± 0.054 on livecodebench, −0.051 ± 0.066 on hle, both straddling zero —
    while the cascade bought +0.148 ± 0.051 and +0.139 ± 0.071. Fitted to always-strongest, λ was a
    coin flip and the corpus could not launch; fitted to King₀ it is +0.19 and +0.17 with intervals
    clear of zero. The old pair measured "does a stronger single model help", which on this market
    is no; the mechanism's question is "does the best fixed policy beat the floor", which is yes.
    It is also the pair §5.6's power gate already measures the band on (`contrast_for`), so the
    price and the gate now describe one line rather than two.

    Either guard leaves λ at 0 and names itself in `flags`, and a flagged benchmark is EXCLUDED from
    scoring (`score_arm`, §5.1b) rather than scored on pure quality:

    * relative spread below `RELSPEND_SPREAD_FLOOR` — λ is undefined here. The spread is measured
      against the two sweep arms' own mean per-task spend, NOT against `C_b`: `C_b` is free under a
      paired duel, so a guard denominated in it could be switched off by pinning `C_b` small (see
      the constant). The comparison is signed on purpose, so a sweep where the top policy
      somehow spends *less* than the cheapest is refused too: a negative λ would score a bonus for
      spending money. Arms that both spend nothing have no scale to measure against, and are
      refused for the same reason rather than divided by zero.
    * `quality_top ≤ quality_cheapest` — the pool buys nothing with money, so λ ≤ 0. Flag the
      corpus, because that is the dominator condition this project has hit five times
      (ROUTING_MEASUREMENTS: the cheapest model dominating on quality-per-dollar is the finding that
      closed the routing thesis, and it is a fact about the model market rather than about the
      benchmark). Evaluated independently of the spread guard so the corpus is flagged either way.
    """
    cheap = _by_benchmark(cheapest)
    strong = _by_benchmark(top)
    exchange = {}
    for benchmark in sorted({*cheap, *strong}):
        if benchmark not in cheap or benchmark not in strong:
            raise ValueError(f"the fixed-policy sweep covers {benchmark!r} in only one arm: "
                             "lambda is a slope between two arms and needs both")
        cost = c_per_task.get(benchmark, 0.0)
        if cost <= 0.0:
            raise ValueError(f"benchmark {benchmark!r} has no positive pinned C_b: "
                             "a benchmark cannot be priced without one")
        _, q_cheap, spend_cheap = cheap[benchmark]
        _, q_strong, spend_strong = strong[benchmark]

        spread = (spend_strong - spend_cheap) / cost                 # §5.1b's denominator, fits λ
        pair_mean = (spend_cheap + spend_strong) / 2.0               # the sweep's own scale, guards
        relative_spread = (spend_strong - spend_cheap) / pair_mean if pair_mean > 0.0 else 0.0
        flags = []
        if relative_spread < RELSPEND_SPREAD_FLOOR:
            flags.append(LAMBDA_UNDEFINED)
        if q_strong <= q_cheap:
            flags.append(DOMINATOR)
        lam = 0.0 if flags else (q_strong - q_cheap) / spread
        exchange[benchmark] = Exchange(benchmark, lam, cost, tuple(flags))
    return exchange


def raw_quality(results: Sequence[EpisodeResult]) -> float:
    """§5.1's aggregate — the equal-weight mean over benchmarks of the mean graded score — with no
    λ in it. `reference.king_zero` chooses King₀ on this BEFORE λ exists, because λ is then fitted
    to King₀ and a choice made under the fitted table would be choosing between arms the table was
    built to make tie."""
    return _mean([mean_q for _, mean_q, _ in _by_benchmark(results).values()])


def score_arm(results: Sequence[EpisodeResult], exchange: Mapping[str, Exchange]) -> ArmScore:
    """One arm's `final = mean over admitted benchmarks of (quality_b − λ_b·spend_b/C_b)` (§5.1b).

    ADMITTED means unflagged. A benchmark whose guard fired leaves the mean entirely and is recorded
    in `excluded`; it is not scored at λ = 0, which prices spend at nothing and lets a challenger buy
    the crown with money alone (module docstring). It is not scored on a re-derived λ either — λ is
    pinned per corpus, so the admitted set is the same every window and leave-one-out stays
    comparable across them (§6.3b).

    A benchmark with no measured exchange rate is refused rather than scored at λ = 0. That is the
    silent fallback §5.1b forbids, and it is a live failure mode rather than a hypothetical one: M3a
    fits λ on three benchmarks and the other nine arrive at M3b, so a stale table would misprice
    three quarters of the corpus while looking like an ordinary score.

    An arm left with nothing admitted is refused rather than returned empty. §5.1b: if every
    benchmark fires a guard there is no corpus to launch on — that is a finding about the model
    market, and the response is to refuse, not to score the arena on unpriced quality.
    """
    rows, excluded = [], []
    for benchmark, (n_tasks, quality, spend) in _by_benchmark(results).items():
        if benchmark not in exchange:
            raise ValueError(f"benchmark {benchmark!r} has no measured exchange rate: "
                             "lambda must be re-derived over the admitted set before scoring")
        rate = exchange[benchmark]
        if rate.flags:
            excluded.append((benchmark, rate.flags))
            continue
        rows.append(BenchmarkScore(benchmark, n_tasks, quality, spend,
                                   quality - rate.lam * (spend / rate.c_per_task)))
    if not rows:
        raise ValueError("every benchmark in this arm is excluded by a fired guard "
                         f"({', '.join(name for name, _ in excluded)}): a benchmark the admission "
                         "gate should have dropped must not price a crown, and a corpus with none "
                         "left is one to refuse rather than to score on unpriced quality")
    return ArmScore(tuple(rows), _mean([r.quality for r in rows]), _mean([r.final for r in rows]),
                    tuple(excluded))


def _by_benchmark(results: Sequence[EpisodeResult]) -> dict[str, tuple[int, float, float]]:
    """benchmark -> (task count, mean graded score, mean spend), in sorted benchmark order.

    Sorted rather than first-seen. The admitted benchmarks are a set, but a float mean of means
    depends on summation order in its last bits, so first-appearance order would make the aggregate
    depend on which benchmark's task the slice happened to draw first. Sorting makes it a function
    of the admitted set alone. (Within a benchmark the order is the window's nonce-derived task
    order, which §4 already fixes as identical for both arms.)
    """
    if not results:
        raise ValueError("no episodes to score: an arm with no admitted tasks has no quality")
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.benchmark, []).append(result)
    return {benchmark: (len(episodes),
                        _mean([e.graded_score for e in episodes]),
                        _mean([e.spend_usd for e in episodes]))
            for benchmark, episodes in sorted(grouped.items())}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def best_possible_final(observed: Sequence[EpisodeResult], *, tasks: Sequence[TaskSpec],
                        exchange: Mapping[str, Exchange],
                        excluded: Collection[str] = frozenset()) -> float:
    """The highest `final` an arm can still reach, given what it has already scored (§5.4).

    An EXACT bound, not the reference implementation's quantile heuristic, and the difference is the
    whole reason this is safe to act on. `teutonic/`'s version assumes the remaining tasks will go
    as well as the observed 95th percentile and says of itself that it is "a tunable heuristic for
    unseen samples, not a mathematical bound" — fine where a stopped run is retried, and wrong here,
    where §7 gives a hotkey ONE submission ever and a false abort spends it forever. So this bounds
    rather than predicts: no continuation whatsoever can beat the figure it returns.

    Two facts make the bound exact and both are properties of the scoring rule, not assumptions
    about the model:

    * a graded score is refused unless it lies in [0, 1] (`scaffold._graded`), so an unrun task can
      contribute at most 1.0 to `quality_b`;
    * `final_b` subtracts `λ_b·spend_b/C_b` and spend only ever grows, so the most favourable
      remaining spend is zero.

    Scoring an unrun task 1.0 also dominates the possibility of it being EXCLUDED later (§6.3c),
    which would drop it from the denominator instead: `(q + u)/n >= q/(n − u)` reduces to
    `n − u >= q`, i.e. the run count is at least the sum of the run scores, which every score being
    at most 1 guarantees. So the bound covers a future exclusion of an unrun task without having to
    model one.

    `excluded` IS THE HALF OF THIS THAT WAS WRONG, and it was wrong in the one direction that
    matters. §6.3c drops a task whose grader or sandbox died from both arms and from the denominator
    — but `_GraderGuard` hands the episode back a placeholder 0.0 first, so a task that is on its way
    OUT of the comparison looks, to a bound that does not know about it, like a task this arm scored
    zero on. Two dead graders early in a slice therefore dragged the ceiling below a bar the arm went
    on to clear: measured, a ceiling of 0.50 against a bar of 1.00 aborting a challenger whose
    post-exclusion score was 1.00. A false abort spends a submission §7 only grants once.

    So excluded tasks leave the ceiling exactly as they leave the score: out of the numerator and out
    of the denominator. The caller passes the guard's live set, and because that set only grows, a
    ceiling computed with it is valid for every exclusion made so far and can only rise as more
    arrive — which is the safe direction.
    """
    # The slice is counted from the tasks themselves rather than handed in, so the denominator and
    # the exclusion set cannot drift apart in two places.
    observed = [result for result in observed if result.task_id not in excluded]
    slice_counts: dict[str, int] = {}
    for task in tasks:
        if task.task_id in excluded:
            continue
        slice_counts[task.benchmark] = slice_counts.get(task.benchmark, 0) + 1
    scored = _by_benchmark_sums(observed)
    rows = []
    for benchmark, total in sorted(slice_counts.items()):
        if benchmark not in exchange:
            raise ValueError(f"benchmark {benchmark!r} has no measured exchange rate")
        rate = exchange[benchmark]
        if rate.flags:
            continue                        # excluded from `final` entirely, exactly as `score_arm`
        run, quality_sum, spend = scored.get(benchmark, (0, 0.0, 0.0))
        if total < run:
            raise ValueError(f"benchmark {benchmark!r} scored {run} of {total} slice tasks")
        unrun = total - run
        # BOTH terms are per-task means, because `final_b` is: `_by_benchmark` hands `score_arm`
        # the MEAN spend, not the total. Pricing the sum here made the ceiling smaller than scores
        # the arm could actually reach — a bound wrong in the one direction that abandons a
        # challenger who would have won. Caught by the exhaustive continuation test, which is why it
        # enumerates rather than argues.
        rows.append((quality_sum + unrun) / total
                    - rate.lam * ((spend / total) / rate.c_per_task))
    if not rows:
        raise ValueError("every benchmark is excluded by a fired guard: there is no ceiling to "
                         "compute, and an arm with no admitted benchmark has no `final`")
    return _mean(rows)


def _by_benchmark_sums(results: Sequence[EpisodeResult]) -> dict[str, tuple[int, float, float]]:
    """benchmark -> (count, SUM of graded scores, SUM of spend).

    Sums rather than `_by_benchmark`'s means, because the ceiling divides by the SLICE's task count
    and not by the number run so far — that is the whole point of it.
    """
    grouped: dict[str, tuple[int, float, float]] = {}
    for result in results:
        count, quality, spend = grouped.get(result.benchmark, (0, 0.0, 0.0))
        grouped[result.benchmark] = (count + 1,
                                     quality + result.graded_score,
                                     spend + result.spend_usd)
    return grouped
