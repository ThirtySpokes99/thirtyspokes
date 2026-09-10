"""The verdict's three conditions (docs/WHITEPAPER.md §5.2, §5.3; the build plan M8).

The tests this architecture is named for are the two halves of one question — *how much of the
corpus did this win come from?* — and a mechanism can fail either half:

  * a **narrow specialist** moves the aggregate past `eps` on a landslide in one or two subjects and
    must still be refused, because "benchmark X → model Y wins" is twelve facts, not a policy;
  * a **broadly better challenger** that loses four benchmarks to noise must still be crowned. This
    is the half a sign test gets wrong: at 8–4 its one-sided p is 0.194, so the draft rule's
    `p < 0.10` would have refused a challenger that is uniformly better everywhere in expectation.

Breadth is therefore TWO rules, not one (§5.2 conditions 2 and 3), and the tests below measure that
neither implies the other: leave-one-out alone crowns a two-benchmark specialist, and the median
alone crowns a one-benchmark landslide whose remaining eleven happen to split six-to-five positive.

The rest of the file protects the properties that make those verdicts mean anything: the bootstrap
is paired (task difficulty cancels instead of deciding), it resamples whole CLUSTERS where the
benchmark has them (R2E-Gym's repositories — tasks that move together, which an iid draw prices as
independent evidence), the aggregation is equal weight per benchmark, and both breadth rules run
over the benchmarks *present* rather than a hard-coded twelve.

`final_b` here is a stand-in for `score.py`'s §5.1b score — linear in quality and in priced spend,
which is the only shape this module depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from thirtyspokes.v3.config import EPS, N_BOOT
from thirtyspokes.v3.duel import (BenchmarkPair, Contender, DuelVerdict, _clusters, _delta,
                                   _resample, champion, duel, sign_test)

# The twelve of §6.1, so a narrowed slice in the tests below reads as a real one.
NAMES = ("terminal-bench", "deepswe", "agents-last-exam", "automationbench", "hle-tools",
         "gdpval-aa", "frontier-swe", "programbench", "cybergym", "exploitbench", "nl2repo",
         "swe-marathon")

LAMBDA = 0.5   # stand-in exchange rate; `score.py` measures the real per-benchmark λ_b


def final_b(benchmark: str, quality: float, spend: float) -> float:
    return quality - LAMBDA * spend


def _slice(deltas, *, n=20, spend=0.04, ch_spend=None, noise_sd=0.15, names=NAMES):
    """A paired slice whose per-benchmark delta in `final` is EXACTLY `deltas[b]`.

    Two construction choices carry the tests. Per-task difficulty is drawn once and shared by both
    arms, which is what a paired duel actually faces. Per-task challenger noise is mean-centred, so
    the bootstrap sees real within-benchmark variation while the point estimate stays exactly as
    designed — otherwise every assertion below would be an assertion about a particular seed.
    """
    rng = np.random.default_rng(20260831)
    pairs = []
    for name, d in zip(names, deltas):
        king_q = 0.45 + rng.uniform(0.0, 0.30, n)          # difficulty, shared by both arms
        noise = rng.normal(0.0, noise_sd, n)
        king_spend = np.full(n, spend)
        pairs.append(BenchmarkPair(name, king_q, king_spend,
                                   king_q + d + (noise - noise.mean()),
                                   np.full(n, spend if ch_spend is None else ch_spend)))
    return tuple(pairs)


# 11 near-zero benchmarks that sum to -0.035: a challenger no better than the king off its subject.
OFF_SUBJECT = (0.02, -0.03, 0.01, -0.02, 0.015, -0.025, 0.005, -0.01, 0.02, -0.03, 0.01)


def test_a_one_benchmark_specialist_loses():
    """M8 exit 1 — *the test the whole architecture is named for.*

    The specialist wins one benchmark by a landslide, which is enough to carry the aggregate past
    `eps` with a positive lower bound. Condition 1 therefore PASSES, and the assertions say so: the
    refusal has to come from breadth, or the test would be passing for the wrong reason.
    """
    deltas = OFF_SUBJECT[:8] + (0.90,) + OFF_SUBJECT[8:]        # landslide on "cybergym"
    v = duel(_slice(deltas), final_b, seed=7)

    assert v.delta > v.eps and v.lcb > 0.0, "condition 1 must pass, or breadth is not what refused"
    assert not v.challenger_wins
    assert v.loo_min < 0.0
    assert v.loo_dropped == "cybergym"
    # Exactly one benchmark's removal flips it, and it is the one the win was concentrated in.
    assert [name for name, d in v.loo if d <= 0.0] == ["cybergym"]


def test_a_broadly_better_challenger_wins_despite_losing_four_benchmarks():
    """M8 exit 2 — the half a sign test gets wrong.

    Uniformly better in expectation, but per-benchmark noise sinks four of the twelve. An 8-4 record
    has one-sided p = 0.194, so the draft's `p < 0.10` clause would have refused it; leave-one-out
    keeps the magnitudes and crowns it.
    """
    deltas = (0.13, 0.12, 0.10, 0.14, 0.09, 0.11, 0.13, 0.10, -0.02, -0.04, -0.01, -0.03)
    v = duel(_slice(deltas), final_b, seed=7)

    assert v.challenger_wins
    assert v.loo_min > 0.0                       # survives losing its best subject
    # Condition 3 must not turn into the significance test §5.3 rejects: a bare majority is a far
    # weaker bar, and this 8-4 challenger clears it by an order of magnitude while the draft's
    # `p < 0.10` refuses it. The real archetype duel is the same shape (§5.3, median +0.1716).
    assert v.median == pytest.approx(0.10, abs=1e-12)
    assert v.sign_wins == 8 and v.sign_p > 0.10  # published, and pointing the other way
    assert v.n_tasks == 240


def test_neither_breadth_condition_implies_the_other():
    """§5.2 condition 3, and the reason it is an ADDITION to leave-one-out rather than a replacement.

    Keeping two breadth rules is only justified if each admits something the other refuses, so both
    directions are measured here rather than argued. The numbers are the ones §5.2 publishes:

      * `[+0.315, +0.315, 0x10]` — a two-benchmark specialist. Leave-one-out passes at +0.0286,
        because dropping either winner leaves the other; the median lands on exactly 0.0, because
        ten of its twelve subjects are byte-identical to the king. **Only the median refuses it.**
      * `[+0.90, +0.01x6, -0.10x5]` — a one-benchmark landslide whose remaining eleven happen to
        split six-to-five positive. The median passes at +0.0100; leave-one-out fails at -0.0400,
        because the landslide was carrying the whole win. **Only leave-one-out refuses it.**

    The specialist clears `eps`, so the median is the only thing standing between it and the crown.
    The published landslide does not (delta +0.0383) — the pair is a statement about the two
    PREDICATES, not about two live attacks — so the same shape scaled until it does clear `eps` is
    measured beside it, and leave-one-out alone still refuses that. The first case is the attack
    `test_invariant.py` sweeps end to end; here it is one half of the independence claim, which
    is a different property and the one that justifies keeping both rules rather than one.
    """
    specialist = duel(_slice((0.315, 0.315) + (0.0,) * 10, noise_sd=0.0), final_b, seed=7)
    landslide = duel(_slice((0.90,) + (0.01,) * 6 + (-0.10,) * 5, noise_sd=0.0), final_b, seed=7)
    bigger = duel(_slice((1.20,) + (0.01,) * 6 + (-0.10,) * 5, noise_sd=0.0), final_b, seed=7)

    assert specialist.delta > specialist.eps and specialist.lcb > 0.0   # condition 1 passed
    assert specialist.loo_min == pytest.approx(0.0286, abs=5e-5)        # leave-one-out says yes
    assert specialist.median == 0.0                                     # the median says no
    assert not specialist.challenger_wins

    assert landslide.median == pytest.approx(0.0100, abs=5e-5)          # the median says yes
    assert landslide.loo_min == pytest.approx(-0.0400, abs=5e-5)        # leave-one-out says no
    assert landslide.delta < landslide.eps                              # eps refuses it too
    assert bigger.delta > bigger.eps and bigger.lcb > 0.0               # this one it does not
    assert (bigger.median, bigger.loo_min) == (landslide.median, landslide.loo_min)
    assert not bigger.challenger_wins


def test_a_six_six_split_passes_the_median_because_it_keeps_the_magnitudes():
    """The median is not the sign test §5.3 threw out wearing a new hat, and this is the case that
    separates them. Twelve benchmarks split exactly 6-6, so the sign test reads 6 of 12 — not even a
    majority, p = 0.6128 — and the median is +0.0950, because it is an order statistic OF THE
    DELTAS: at an even count it is the mean of the two middle ones, and here the smallest win
    (+0.20) outweighs the smallest loss (-0.01).

    A challenger worth twenty times more where it wins than it costs where it loses is what this
    mechanism exists to sell, and a rule that counted bits instead of reading them would refuse it.
    So `median > 0` over twelve is only APPROXIMATELY the >=7-of-12 §5.3 dismisses (§5.2, D16).
    """
    v = duel(_slice((0.20,) * 6 + (-0.01,) * 6, noise_sd=0.0), final_b, seed=7)

    assert v.challenger_wins
    assert v.median == pytest.approx(0.0950, abs=5e-5)
    assert v.sign_wins == 6                       # a bit-counting majority rule would have refused


def test_at_an_odd_benchmark_count_the_median_is_exactly_a_strict_majority():
    """§6.3b: a window narrowed to eleven benchmarks, because one could not supply its minimum task
    count. With an odd count the median IS the middle delta rather than the mean of two, so
    `median > 0` is exactly *more than half moved the right way* — five wins of eleven is refused
    however large they are.

    Recorded because it means the bar moves with the window width, and a reader comparing verdicts
    across a narrowed window has to know that: the same challenger with a sixth win at twelve
    benchmarks is crowned (the test above). Refused by the median alone here — the aggregate clears
    `eps` at +0.0855 and leave-one-out clears zero at +0.0740.
    """
    narrowed = NAMES[:11]
    v = duel(_slice((0.20,) * 5 + (-0.01,) * 6, noise_sd=0.0, names=narrowed), final_b, seed=7)

    assert len(v.benchmarks) == 11
    assert v.delta > v.eps and v.lcb > 0.0
    assert v.loo_min == pytest.approx(0.0740, abs=5e-5)
    assert v.median == pytest.approx(-0.0100, abs=1e-12)   # the 6th of 11, not a mean of two
    assert not v.challenger_wins


def test_a_uniform_three_point_challenger_is_refused_by_eps_not_by_breadth():
    """the build plan M8 exit 2 says "uniformly +3 points ... must still be crowned", and under D10 it is
    NOT — `eps = 0.05` is five points on the `final` scale, so +0.03 cannot clear condition 1 at any
    task count. The plan's wording predates the pin; the arithmetic in §5.2b is the authority (at
    ~250 tasks `eps = 0.05` behaves like a ~7-8 point bar).

    Recorded as a test rather than a comment so the distinction survives: this challenger passes
    BREADTH on every benchmark and is refused purely by the aggregate bar. If a future reader
    lowers `eps` — the one direction D10 permits — this is the test that tells them what changed.
    """
    v = duel(_slice((0.03,) * 12), final_b, seed=7)

    assert not v.challenger_wins
    assert v.loo_min > 0.0                       # breadth is not the objection
    assert v.delta < v.eps                       # the bar is
    assert duel(_slice((0.03,) * 12), final_b, eps=0.02, seed=7).challenger_wins


def test_a_byte_identical_copy_ties_and_therefore_loses():
    """M8 exit 9. A copy of the king scores exactly what the king scores, on every task and so on
    every resample. Requiring a STRICT `eps` beat is what turns anti-copy from a detection problem —
    measurement 11 found copier agreement and honest convergence to be the same signal, i.e.
    undetectable — into an economic one: copying can tie, and ties keep the crown where it is.
    """
    king = _slice((0.0,) * 12, noise_sd=0.0)
    copy = tuple(BenchmarkPair(p.benchmark, p.king_quality, p.king_spend,
                               p.king_quality.copy(), p.king_spend.copy()) for p in king)
    v = duel(copy, final_b, seed=7)

    assert v.delta == 0.0 and v.lcb == 0.0
    assert v.loo_min == 0.0
    assert not v.challenger_wins
    assert (v.sign_wins, v.sign_p) == (0, 1.0)   # ties count against the challenger, no special case
    assert champion([Contender("hk_copy", commit_block=10, verdict=v)]) is None


def test_leave_one_out_runs_over_the_benchmarks_present_in_the_slice():
    """M8 exit 7b, §6.3b. A benchmark that cannot supply its minimum task count is left out of the
    window, and the verdict must run over — and RECORD — the eleven actually present. Nothing is
    hard-coded to twelve: admission drops benchmarks and rotation adds them, so a mechanism pinned
    to 12 breaks the first time a benchmark fails its gate.
    """
    present = tuple(name for name in NAMES if name != "hle-tools")
    deltas = (0.13, 0.12, 0.10, 0.14, 0.11, 0.13, 0.10, -0.02, -0.04, -0.01, -0.03)
    designed = dict(zip(present, deltas))
    v = duel(_slice(deltas, names=present), final_b, seed=7)

    assert v.benchmarks == tuple(sorted(present)) and "hle-tools" not in v.benchmarks
    assert len(v.loo) == 11
    assert v.delta == pytest.approx(sum(deltas) / 11, abs=1e-12)      # equal weight 1/11, not 1/12
    for name, without in v.loo:
        assert without == pytest.approx((sum(deltas) - designed[name]) / 10, abs=1e-12)


def test_task_difficulty_cancels_because_the_bootstrap_is_paired():
    """§5.2. Each resample draws the SAME tasks for both arms, so on a slice where the challenger is
    a constant better on every task, every resample must return that constant exactly and the lower
    bound must equal the point estimate. An unpaired bootstrap would resample the two arms
    independently, and the wild per-task difficulty spread below would then dominate the interval —
    slice luck deciding the crown, which is the failure paired scoring exists to remove.
    """
    rng = np.random.default_rng(11)
    pairs = tuple(
        BenchmarkPair(name, q := rng.uniform(0.0, 1.0, 20), np.full(20, 0.04),
                      q + 0.07, np.full(20, 0.04))
        for name in NAMES)
    v = duel(pairs, final_b, seed=7)

    assert v.delta == pytest.approx(0.07, abs=1e-12)
    assert v.lcb == pytest.approx(v.delta, abs=1e-12)


# --- the resample UNIT: whole clusters, not tasks -------------------------------------------------
#
# `duel.py`'s module docstring carries the measurement and the two controls. These tests protect the
# three properties it rests on: the ungrouped path is untouched, a grouping widens the interval and
# nothing else, and the widened resample is still paired.


def _clustered(deltas, *, block=8, n=80, tau=0.40, spend=0.04, seed=20260901, names=NAMES[:3]):
    """A paired slice whose tasks arrive in correlated BLOCKS, the way R2E-Gym's repositories do.

    Every task in a block shares one offset drawn for that block, so a block is better or worse than
    the benchmark as a whole and the rows inside it are not independent — which is the entire content
    of "a challenger that suits a repository is better on all of its tasks". The offsets are
    mean-centred within each benchmark, so the per-benchmark delta is EXACTLY `deltas[b]` and the
    grouped and ungrouped duels below differ in nothing but what the bootstrap draws.

    Three benchmarks and 80 tasks each, which is the corpus that exists (§6.3b, ~83 per stratum), not
    the twelve the rest of this file uses.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for name, d in zip(names, deltas):
        king_q = 0.45 + rng.uniform(0.0, 0.30, n)          # difficulty, shared by both arms
        offsets = rng.normal(0.0, tau, n // block)
        pairs.append(BenchmarkPair(name, king_q, np.full(n, spend),
                                   king_q + d + np.repeat(offsets - offsets.mean(), block),
                                   np.full(n, spend),
                                   groups=tuple(f"repo{i // block}" for i in range(n))))
    return tuple(pairs)


def _ungrouped(pairs):
    """The identical numbers with the labels removed — the slice as it would have been duelled."""
    return tuple(BenchmarkPair(p.benchmark, p.king_quality, p.king_spend,
                               p.challenger_quality, p.challenger_spend) for p in pairs)


def _iid_bootstrap_lcb(pairs, *, seed, n_boot=N_BOOT):
    """The resample AS IT SHIPPED, transcribed: `rng.integers(0, n, size=n)` per benchmark, in
    sorted benchmark order, off one generator seeded once. This is the baseline the bit-for-bit
    claim is made against — a verdict, not a recollection of one."""
    ordered = tuple(sorted(pairs, key=lambda p: p.benchmark))
    rng = np.random.default_rng(seed)
    boot = [_delta(ordered, final_b, [rng.integers(0, p.n, size=p.n) for p in ordered])[0]
            for _ in range(n_boot)]
    return float(np.percentile(boot, 5.0))


def test_an_ungrouped_duel_is_bit_for_bit_the_resample_that_shipped():
    """THE BACKWARD-COMPATIBILITY PROOF, and it is exact rather than approximate.

    `groups` is optional and absent means every task is its own cluster. That has to be the SAME
    draw the arena has been running, not a statistically equivalent one: a lower bound that moved in
    the last bits would re-decide every historical verdict sitting within `eps` of the bar, silently
    and for no measured reason. Both branches make the identical `rng.integers` call on the identical
    stream, so `==` is the right comparison and `approx` would be hiding the claim.

    The second half is what gives the first teeth: the same slice WITH its labels does not reproduce
    the transcription, so this test fails if `groups` is ever quietly ignored.
    """
    pairs = _ungrouped(_clustered((0.06,) * 3))

    assert duel(pairs, final_b, seed=7).lcb == _iid_bootstrap_lcb(pairs, seed=7)
    assert duel(pairs, final_b, seed=7).noise_floor == duel(pairs, final_b, seed=7).noise_floor
    assert duel(_clustered((0.06,) * 3), final_b, seed=7).lcb != _iid_bootstrap_lcb(pairs, seed=7)


def test_clustering_widens_the_lower_bound_and_the_iid_resample_crowns_what_it_should_refuse():
    """§5.2 condition 1, measured 2026-09-01 (see `duel.py`). Tasks in blocks of eight, each block
    carrying its own mean-zero advantage offset: the true per-benchmark advantage is exactly +0.06
    either way, but the estimate can wander much further than an iid draw believes.

    The gate's outcome flips, and in the safe direction. Resampling tasks, the challenger is CROWNED
    — delta +0.06, lower bound +0.0285. Resampling repositories, the same numbers give −0.0233 and
    it is refused, which is right: with ten blocks per benchmark this challenger is genuinely not
    separable from zero. A wider interval can only refuse a crown; it can never manufacture one,
    which is why the fix is worth taking at an unmeasured cluster effect size.

    The per-benchmark bootstrap SE (`noise_floor`, published every window) moves with it — but only
    once the labels are there. That is the measurement's sharpest point: before the fix that
    diagnostic was flat while the true spread climbed, so the arena had no way to see this.
    """
    grouped = _clustered((0.06,) * 3)
    clustered_v, iid = duel(grouped, final_b, seed=7), duel(_ungrouped(grouped), final_b, seed=7)

    assert iid.lcb > 0.0 and iid.challenger_wins
    assert clustered_v.lcb < 0.0 and not clustered_v.challenger_wins
    assert clustered_v.delta - clustered_v.lcb > 2.0 * (iid.delta - iid.lcb)
    assert min(clustered_v.noise_floor) > max(iid.noise_floor)


def test_a_grouping_moves_the_interval_and_nothing_else():
    """Everything except `lcb` and `noise_floor` is computed from the slice itself, so labelling the
    tasks must leave it untouched to the last bit. The aggregate, every per-benchmark delta, both
    breadth conditions and the sign test are all point estimates over all the rows.

    This is what keeps the fix inside condition 1: `challenger_wins` may now say no where it said
    yes, but only through the lower bound — the breadth rules four rule families were measured to
    protect (`scripts/breadth_power.py`) see exactly the numbers they saw before.
    """
    grouped = _clustered((0.09, 0.06, -0.01))
    clustered_v, iid = duel(grouped, final_b, seed=7), duel(_ungrouped(grouped), final_b, seed=7)

    assert clustered_v.delta == iid.delta
    assert clustered_v.by_benchmark == iid.by_benchmark
    assert clustered_v.loo == iid.loo
    assert (clustered_v.median, clustered_v.loo_min) == (iid.median, iid.loo_min)
    assert (clustered_v.sign_wins, clustered_v.sign_p) == (iid.sign_wins, iid.sign_p)
    assert clustered_v.n_tasks == iid.n_tasks == 240


def test_the_clustered_resample_is_still_paired_and_still_within_the_benchmark():
    """The two properties §5.2 rests on, re-asserted on the new draw because it is a new draw.

    PAIRED: the challenger is a constant better on every task, so every resample — whatever bag of
    repositories it drew — must return that constant exactly and the lower bound must equal the point
    estimate. STRATIFIED: the three benchmarks carry different task counts and different numbers of
    repositories, so a resample that took its rows from the wrong benchmark's clusters would score
    the wrong arm's tasks and move the constant off +0.07.
    """
    rng = np.random.default_rng(11)
    pairs = tuple(
        BenchmarkPair(name, q := rng.uniform(0.0, 1.0, n), np.full(n, 0.04),
                      q + 0.07, np.full(n, 0.04),
                      groups=tuple(f"repo{i % repos}" for i in range(n)))
        for name, n, repos in (("deepswe", 40, 5), ("hle-tools", 24, 3), ("cybergym", 33, 11)))
    v = duel(pairs, final_b, seed=7)

    assert v.delta == pytest.approx(0.07, abs=1e-12)
    assert v.lcb == pytest.approx(v.delta, abs=1e-12)
    assert v.n_tasks == 97


def test_the_grouped_verdict_does_not_depend_on_the_order_the_tasks_landed_in():
    """`_clusters` orders by LABEL, and this is that claim asserted rather than commented.

    The clusters are drawn by index, so if they were collected in first-appearance order instead
    then which repository is index 0 would depend on how the slice happened to be shuffled — and two
    validators replaying one reveal from the same nonce over the same tasks would publish different
    lower bounds. That is the same failure `test_the_verdict_does_not_depend_on_the_order_the_
    benchmarks_arrive_in` covers one level up, and it was uncovered one level down: a `_clusters`
    keyed on first appearance passes every other test in this file.

    Reversing the rows preserves every cluster's membership and inverts their first appearance, so
    under label ordering the verdict is bit-identical and under any positional ordering it is not.
    """
    pairs = _clustered((0.09, 0.06, -0.01))
    reversed_rows = tuple(
        BenchmarkPair(p.benchmark, p.king_quality[::-1], p.king_spend[::-1],
                      p.challenger_quality[::-1], p.challenger_spend[::-1],
                      groups=tuple(reversed(p.groups)))
        for p in pairs)

    assert duel(reversed_rows, final_b, seed=7).lcb == duel(pairs, final_b, seed=7).lcb


def test_a_resample_draws_as_many_clusters_as_the_benchmark_has():
    """The cluster COUNT is what sets the interval's width — a cluster bootstrap's effective sample
    size is the number of clusters (module docstring) — so drawing the wrong number of them silently
    reprices `lcb`, which decides crowns. Nothing else here pins it: every other grouped test asserts
    that the interval MOVED, and it still moves if the draw is short by one.

    With equal-sized clusters `len(clusters)` draws is exactly `n` rows, so the count is checkable
    without restating the loop.
    """
    pair = _clustered((0.09,), names=NAMES[:1])[0]
    clusters = _clusters(pair)
    rng = np.random.default_rng(3)

    assert len(clusters) == 10 and {len(c) for c in clusters} == {8}
    assert {len(_resample(pair, clusters, rng)) for _ in range(200)} == {pair.n}


def test_a_partial_or_misaligned_grouping_is_refused():
    """A wrong grouping is worse than none, so both ways of writing one down fail at construction.

    Half-labelled is the dangerous one: `duel` would treat the unlabelled tasks as their own
    clusters, which is the too-narrow interval this whole change removes, arriving by the back door
    and looking like a clustered duel in the reveal. A benchmark either has a grouping or it does
    not — `LiveCodeBench` and HLE pass None and are resampled exactly as they always were.
    """
    with pytest.raises(ValueError, match="not row-aligned"):
        BenchmarkPair("deepswe", np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4),
                      groups=("sympy", "sympy", "pandas"))
    with pytest.raises(ValueError, match="partial grouping"):
        BenchmarkPair("deepswe", np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                      groups=("sympy", None, "pandas"))


def test_the_bootstrap_is_seeded_from_the_window_nonce():
    """The verdict must be reproducible from the published reveal rather than from the validator's
    memory, so the lower bound is a function of the window nonce and nothing else. An unseeded
    bootstrap is refused outright: a duel that cannot be re-derived cannot be audited."""
    pairs = _slice((0.06,) * 12)
    assert duel(pairs, final_b, nonce="w7").lcb == duel(pairs, final_b, nonce="w7").lcb
    assert duel(pairs, final_b, nonce="w7").lcb != duel(pairs, final_b, nonce="w8").lcb
    with pytest.raises(ValueError, match="nonce"):
        duel(pairs, final_b)


def test_doubling_one_benchmarks_task_count_does_not_move_the_verdict():
    """M8 exit 7. Equal weight per BENCHMARK, never per task (§5.1): pooling tasks would let
    whichever benchmark contributed the most rows silently dominate the total — an implicit
    weighting nobody chose, and one directly at odds with rewarding generalisation."""
    deltas = (0.13, 0.12, 0.10, 0.14, 0.09, 0.11, 0.13, 0.10, -0.02, -0.04, -0.01, -0.03)
    pairs = _slice(deltas)
    doubled = (BenchmarkPair(pairs[0].benchmark,
                             *(np.tile(a, 2) for a in (pairs[0].king_quality, pairs[0].king_spend,
                                                       pairs[0].challenger_quality,
                                                       pairs[0].challenger_spend))),) + pairs[1:]

    assert duel(doubled, final_b, seed=7).delta == pytest.approx(duel(pairs, final_b, seed=7).delta,
                                                                 abs=1e-12)
    assert duel(doubled, final_b, seed=7).by_benchmark == duel(pairs, final_b, seed=7).by_benchmark


def test_the_verdict_does_not_depend_on_the_order_the_benchmarks_arrive_in():
    """The aggregate is a mean of means, so its last bits depend on summation order, and the
    bootstrap draws per benchmark in sequence. A verdict must be a function of the admitted SET —
    the same slice handed over in a different order is the same duel, down to the published lower
    bound."""
    pairs = _slice((0.13, 0.12, 0.10, 0.14, 0.09, 0.11, 0.13, 0.10, -0.02, -0.04, -0.01, -0.03))

    assert duel(tuple(reversed(pairs)), final_b, seed=7) == duel(pairs, final_b, seed=7)


def test_the_verdict_is_computed_on_final_not_on_quality():
    """§5.1b. Breadth is required in the TRADEOFF, not in accuracy alone. This challenger is more
    accurate on every single benchmark and buys it at the pool's own exchange rate, so its `final`
    is worse everywhere — it must lose on the aggregate AND on every leave-one-out subset."""
    v = duel(_slice((0.10,) * 12, spend=0.04, ch_spend=0.34), final_b, seed=7)

    assert v.delta == pytest.approx(0.10 - LAMBDA * 0.30, abs=1e-12)
    assert v.delta < 0.0
    assert not v.challenger_wins
    assert max(d for _, d in v.loo) < 0.0


def test_the_sign_test_reproduces_the_correction_and_is_only_a_diagnostic():
    """§5.3's table, the reason breadth is not gated on a sign test. "≥7 of 12" happens by coin flip
    more than a third of the time, so the draft's `≥7 and p < 0.10` clauses contradicted each other
    and the rule gated nothing; ≥9 is the wrong repair, because per-benchmark win/loss over ~20
    tasks is noisy enough to sink a genuinely broad challenger."""
    assert sign_test([1.0] * 7 + [-1.0] * 5) == (7, pytest.approx(0.3872, abs=5e-5))
    assert sign_test([1.0] * 9 + [-1.0] * 3) == (9, pytest.approx(0.0730, abs=5e-5))
    assert sign_test([0.0] * 12) == (0, 1.0)


def test_a_slice_that_cannot_carry_the_breadth_rule_is_refused_loudly():
    """Leave-one-out on a single benchmark would score the challenger on an empty slice, and the
    resulting NaN compares false — the crown would be refused for a reason nobody could read. A
    benchmark appearing twice is the mirror image: it silently doubles that benchmark's weight in an
    aggregation defined to be equal per benchmark."""
    pairs = _slice((0.06,) * 12)
    with pytest.raises(ValueError, match="breadth"):
        duel(pairs[:1], final_b, seed=7)
    with pytest.raises(ValueError, match="twice"):
        duel(pairs + pairs[:1], final_b, seed=7)


def test_the_arms_must_cover_the_same_tasks():
    """§6.3c. Both arms run the identical task list, and a grader failure drops the task from BOTH.
    An arm holding a different number of rows is an asymmetric exclusion — one arm scored on an
    easier slice — so it fails at construction rather than becoming a silently different duel."""
    with pytest.raises(ValueError, match="not paired"):
        BenchmarkPair("cybergym", np.zeros(20), np.zeros(20), np.zeros(19), np.zeros(19))


def test_the_crown_goes_to_the_largest_delta_and_ties_break_on_earliest_commit_block():
    """§5.2, §8 step 4. Batch coronation ranks every winning challenger against the ONE king arm the
    window ran (§5.2a), so the comparison between challengers is well defined; remaining ties break
    on earliest commit block, then hotkey — the same total order the queue is evaluated in (§8b.1),
    so the reveal is reproducible rather than dependent on the order results landed in."""
    def won(delta):
        return DuelVerdict(delta=delta, lcb=delta / 2, by_benchmark=(("a", delta), ("b", delta)),
                           loo=(("a", delta), ("b", delta)), sign_wins=2, sign_p=0.25,
                           n_tasks=240, eps=EPS)

    late_and_best = Contender("hk_a", commit_block=200, verdict=won(0.10))
    tied_later = Contender("hk_b", commit_block=50, verdict=won(0.07))
    tied_earlier = Contender("hk_c", commit_block=40, verdict=won(0.07))

    assert champion([tied_later, late_and_best, tied_earlier]) is late_and_best
    assert champion([tied_later, tied_earlier]) is tied_earlier


# --- what the verdict is worth at N = 2 (MEASURED 2026-09-05/06, not argued) -----------------------


def test_at_two_benchmarks_the_median_is_the_aggregate_and_tests_nothing_new():
    """`the spend plan` §1.2 asserts this; here it is arithmetic, and it costs a whole condition.

    The median of two numbers IS their mean, and the aggregate is the mean of the per-benchmark
    deltas — so at N = 2 condition 3 returns condition 1's number to four decimals and cannot refuse
    anything condition 1 admitted. Three conditions become two.
    """
    v = duel(_slice((0.1329, -0.0147), noise_sd=0.0, names=("livecodebench", "hle")),
             final_b, seed=7)

    assert v.median == pytest.approx(v.delta, abs=1e-12)


def test_at_two_benchmarks_only_breadth_stands_between_noise_and_a_crown():
    """MEASURED on the real M3a arms: `always_cheapest` duelled against a SECOND DRAW OF ITSELF.

    Two runs of the identical policy over the identical 131 tasks, differing only in the worker's
    own nondeterminism (six identical calls to one HLE task returned six distinct answers at
    `temperature: 0.0`, from a single pinned endpoint). The observed per-benchmark deltas were
    livecodebench +0.1329 and hle -0.0147, and the verdict cleared THREE of its four conditions:
    delta +0.0591 > eps, lcb +0.0105 > 0, median +0.0591 > 0. Only leave-one-out refused it.

    The deltas below are those measurements. What the test pins is how little separated an identical
    policy from the throne — and the second half pins the consequence: the hle delta is far inside
    its own paired noise (SE ~0.054 on 68 binary tasks), so its SIGN is a coin flip, and flipping it
    crowns a challenger that routes exactly as the king does.
    """
    refused = duel(_slice((0.1329, -0.0147), noise_sd=0.0, names=("livecodebench", "hle")),
                   final_b, seed=7)

    assert refused.delta > EPS and refused.lcb > 0.0 and refused.median > 0.0
    assert refused.loo_min <= 0.0
    assert not refused.challenger_wins, "breadth is the only condition doing any work here"

    crowned = duel(_slice((0.1329, +0.0147), noise_sd=0.0, names=("livecodebench", "hle")),
                   final_b, seed=7)

    assert crowned.challenger_wins, (
        "flipping a sub-noise sign on the second benchmark crowns a policy identical to the king's "
        "— which is what N = 2 costs, and why the corpus needs benchmarks before it needs miners")
