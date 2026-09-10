"""What the score has to mean (docs/WHITEPAPER.md §5.1/§5.1b, the build plan M8 exits 3/3b/7).

Three properties here are load-bearing for the mechanism rather than for the module:

  * **The λ invariant** (exit 3). Always-cheapest and always-strongest must score identically, per
    benchmark and in aggregate. It is the single line of defence against both degenerate policies at
    once, and a regression turns the subnet silently into either a frugality contest or a spending
    contest — silently, because every other test would still pass and the leaderboard would still
    look like a leaderboard.
  * **`C_b` is free** (exit 3b). The normaliser cancels out of a paired comparison, so it is pinned
    rather than measured live. Pinned as a test so nobody later "fixes" it into the score by
    normalising against the king's actual spend, which is the draft this design already rejected.
  * **Equal weight per benchmark** (exit 7). The fixture is built so the task-pooled reading gives
    the *opposite* ranking; the test asserts both, so it fails if the implicit-weighting bug ever
    comes back rather than merely passing for the right value.
  * **A guarded benchmark is EXCLUDED, not scored on pure quality** (§5.1b, adversarial finding 3).
    Clamping lambda to 0 left spend unpriced exactly where the guards fire, and the dominator guard
    fires on the condition this repository has measured five times — so two flagged benchmarks let a
    challenger byte-identical to the king elsewhere buy the crown at 5000x its spend. Its twin is
    finding 5: the guard must not be silenceable by choosing `C_b`, or the exclusion can be switched
    off from the pinned constants with nothing published to say so.

The sweep fixture below deliberately gives the two benchmarks exchange rates that differ by ~1.8x.
A corpus with one uniform rate would let a global λ pass the invariant test, and the per-benchmark
rule (§5.1b) would then be untested.
"""

from __future__ import annotations

import random

import pytest

from thirtyspokes.v3.config import EPS
from thirtyspokes.v3.score import (
    DOMINATOR,
    LAMBDA_UNDEFINED,
    RELSPEND_SPREAD_FLOOR,
    Exchange,
    best_possible_final,
    derive_exchange,
    pinned_cost,
    score_arm,
)
from thirtyspokes.v3.types import EpisodeResult, TaskSpec

TB = "terminal-bench"
CG = "cyber-gym"


def _episodes(benchmark: str, scores: tuple[float, ...],
              spends: tuple[float, ...]) -> list[EpisodeResult]:
    return [EpisodeResult(task_id=f"{benchmark}-{i}", benchmark=benchmark, steps=(),
                          graded_score=score, spend_usd=spend, stopped_reason="stop")
            for i, (score, spend) in enumerate(zip(scores, spends, strict=True))]


def _flat(benchmark: str, quality: float, spend: float, n: int = 4) -> list[EpisodeResult]:
    return _episodes(benchmark, (quality,) * n, (spend,) * n)


# The §6.1 fixed-policy sweep. Benchmarks are listed cyber-gym-first so the sorted grouping is doing
# work rather than agreeing with the input order by accident.
CHEAPEST = [*_episodes(CG, (0.0, 0.1, 0.1, 0.2), (0.04, 0.05, 0.05, 0.06)),
            *_episodes(TB, (0.1, 0.3, 0.3, 0.5), (0.01, 0.02, 0.02, 0.03))]
STRONGEST = [*_episodes(CG, (0.4, 0.5, 0.5, 0.6), (0.40, 0.45, 0.45, 0.50)),
             *_episodes(TB, (0.5, 0.6, 0.64, 0.74), (0.18, 0.20, 0.20, 0.22))]

C_PINNED = pinned_cost([*CHEAPEST, *STRONGEST])   # {TB: 0.11, CG: 0.25}
RATES = derive_exchange(CHEAPEST, STRONGEST, C_PINNED)


def test_c_b_is_the_sweeps_mean_dollars_per_task_pooled_over_both_fixed_policies():
    """Pooled, so the pinned constant is a property of the benchmark rather than of one policy's
    appetite — and the value is free anyway (see the C_b test below), so what is bought here is a
    definite reproducible number and a readable scale for lambda."""
    assert C_PINNED[TB] == pytest.approx(0.11)
    assert C_PINNED[CG] == pytest.approx(0.25)


def test_lambda_is_the_measured_slope_of_the_pools_own_cost_quality_line():
    """§5.1b's formula, pinned. Measured rather than chosen is the whole point: a hand-picked lambda
    buries the exchange rate between accuracy and dollars in a constant somebody guessed."""
    # TB: (0.62 - 0.30) / ((0.20 - 0.02) / 0.11)
    assert RATES[TB].lam == pytest.approx(0.32 * 0.11 / 0.18)
    # CG: (0.50 - 0.10) / ((0.45 - 0.05) / 0.25)
    assert RATES[CG].lam == pytest.approx(0.25)
    assert RATES[TB].flags == () and RATES[CG].flags == ()


def test_always_cheapest_and_always_strongest_score_the_same_per_benchmark_and_in_aggregate():
    """M8 exit 3 — THE LAMBDA INVARIANT.

    The score's iso-lines run parallel to the line joining the two extremes, so buying quality at
    the pool's own exchange rate earns exactly nothing. This is what stops the arena becoming a
    frugality contest (quality/cost) or a spending contest (a lambda picked too low), and it is what
    defuses §4's miner-set allowance asymmetry: a richer challenger cannot outspend the king into a
    win when the extra quality is priced at the rate the pool gives it away.
    """
    cheap = score_arm(CHEAPEST, RATES)
    strong = score_arm(STRONGEST, RATES)

    for row_cheap, row_strong in zip(cheap.per_benchmark, strong.per_benchmark, strict=True):
        assert row_cheap.benchmark == row_strong.benchmark
        assert row_cheap.final == pytest.approx(row_strong.final, abs=1e-9)
    assert cheap.final == pytest.approx(strong.final, abs=1e-9)

    # ...and the tie is not an artefact of a corpus with one uniform exchange rate, which is the
    # case a single global lambda would also have survived (§5.1b: per benchmark, not global).
    assert RATES[TB].lam / RATES[TB].c_per_task > 1.7 * RATES[CG].lam / RATES[CG].c_per_task
    # The extremes really are far apart in both dimensions; the tie is priced, not degenerate.
    assert strong.quality - cheap.quality > 0.3


def test_a_policy_above_the_pool_line_scores_above_both_extremes():
    """The definition of routing skill (§5.1b), and the half of it the product claim rests on: more
    quality per dollar than the pool gives away for free. Its mirror — the strongest models' bill
    for the cheapest models' quality — must land below both."""
    above = [*_flat(TB, 0.62, 0.02), *_flat(CG, 0.50, 0.05)]     # strong quality, cheap spend
    below = [*_flat(TB, 0.30, 0.20), *_flat(CG, 0.10, 0.45)]     # cheap quality, strong spend
    extreme = score_arm(CHEAPEST, RATES).final

    assert score_arm(above, RATES).final > extreme + 0.3
    assert score_arm(below, RATES).final < extreme - 0.3


def test_a_policy_exactly_on_the_pool_line_ties_both_extremes():
    """The other side of the same property: the score is flat along the pool's own tradeoff, so
    landing halfway between the two fixed policies earns nothing over either."""
    midpoint = [*_flat(TB, 0.46, 0.11), *_flat(CG, 0.30, 0.25)]
    assert score_arm(midpoint, RATES).final == pytest.approx(score_arm(CHEAPEST, RATES).final,
                                                             abs=1e-9)


# Two arms that spend identically (C_b per task, so the priced term is the same for both) and differ
# only in where their quality sits: the honest arm is better on terminal-bench, the specialist is
# better on cyber-gym and better overall *per task* once cyber-gym contributes more tasks.
HONEST = [*_episodes(TB, (0.55, 0.65), (0.11, 0.11)),
          *_episodes(CG, (0.35, 0.45), (0.25, 0.25))]
SPECIALIST = [*_episodes(TB, (0.25, 0.35), (0.11, 0.11)),
              *_episodes(CG, (0.63, 0.73), (0.25, 0.25))]
HONEST_WIDE = [*_episodes(TB, (0.55, 0.65), (0.11, 0.11)),
               *_episodes(CG, (0.30, 0.40, 0.40, 0.50), (0.25,) * 4)]
SPECIALIST_WIDE = [*_episodes(TB, (0.25, 0.35), (0.11, 0.11)),
                   *_episodes(CG, (0.58, 0.68, 0.68, 0.78), (0.25,) * 4)]


def test_scaling_c_b_rescales_lambda_and_changes_no_score():
    """M8 exit 3b. `C_b` cancels out of a paired comparison — it only ever rescales `lambda_b` — so
    it is pinned rather than measured from King₀'s live spend, which would have forced a full King₀
    arm every window purely to obtain a denominator. Pinned as a test because the tempting "fix" is
    to normalise by something real, and nothing else would fail if someone did.

    (Exact while lambda is defined. The relspend floor is deliberately expressed in units of `C_b`,
    so a `C_b` pinned orders of magnitude off that convention moves that guard with it.)
    """
    scaled_c = {benchmark: cost * 10.0 for benchmark, cost in C_PINNED.items()}
    scaled = derive_exchange(CHEAPEST, STRONGEST, scaled_c)

    for benchmark, rate in RATES.items():
        assert scaled[benchmark].lam == pytest.approx(rate.lam * 10.0)
        assert scaled[benchmark].flags == ()

    for arm in (CHEAPEST, STRONGEST, HONEST, SPECIALIST):
        assert score_arm(arm, scaled).final == pytest.approx(score_arm(arm, RATES).final, abs=1e-12)
    assert score_arm(HONEST, scaled).final > score_arm(SPECIALIST, scaled).final


def test_doubling_one_benchmarks_task_count_does_not_change_the_ranking():
    """M8 exit 7, and the bug it guards: averaging raw tasks lets whichever benchmark contributed
    the most of them silently dominate — an implicit weighting nobody chose, directly at odds with
    rewarding generalisation.

    The fixture is built so the wrong aggregation gives the OPPOSITE verdict, asserted below. A test
    that only checked the right answer would pass under both implementations here.
    """
    assert score_arm(HONEST, RATES).final > score_arm(SPECIALIST, RATES).final
    assert score_arm(HONEST_WIDE, RATES).final > score_arm(SPECIALIST_WIDE, RATES).final
    # Same score, not merely the same ordering: cyber-gym's mean is unchanged, so equal weighting
    # cannot notice that it now carries twice the tasks.
    assert score_arm(HONEST_WIDE, RATES).final == pytest.approx(score_arm(HONEST, RATES).final)

    def pooled(arm):
        return sum(e.graded_score for e in arm) / len(arm)

    assert pooled(HONEST) > pooled(SPECIALIST)               # balanced: the readings coincide
    assert pooled(HONEST_WIDE) < pooled(SPECIALIST_WIDE)     # unbalanced: pooling flips the verdict


def test_quality_is_the_mean_of_benchmark_means_not_of_tasks():
    """§5.1 stated directly, on a slice that is as unbalanced as §6.3b's minimum-count rule allows a
    window to get."""
    arm = [*_flat(TB, 0.9, 0.11, n=1), *_flat(CG, 0.1, 0.25, n=5)]
    assert score_arm(arm, RATES).quality == pytest.approx(0.5)
    assert sum(e.graded_score for e in arm) / len(arm) == pytest.approx(0.2333333, abs=1e-6)


def test_final_is_quality_minus_the_priced_spend_per_benchmark():
    """The objective itself (D8), pinned against hand arithmetic so a refactor cannot quietly change
    what miners are optimising."""
    row = score_arm([*_flat(TB, 0.50, 0.22)], RATES).per_benchmark[0]
    assert row.benchmark == TB and row.n_tasks == 4
    assert row.quality == pytest.approx(0.50) and row.spend == pytest.approx(0.22)
    assert row.final == pytest.approx(0.50 - RATES[TB].lam * (0.22 / 0.11))


def test_a_spread_below_the_floor_excludes_the_benchmark_and_says_so():
    """M8 exit 10, §5.1b's first guard. lambda is a ratio whose denominator is measured on a
    ~50-task probe, so a spread this small is not separable from the sweep's own noise — and lambda
    is pinned per corpus, so a noisy fit would be frozen into every score until the next
    re-derivation.

    ANNOUNCED, never silent: an exclusion that changed what miners are optimising without telling
    them is the "Kind" property failing, and a benchmark that merely went missing from the mean is
    arithmetically indistinguishable from one that was never in the slice.
    """
    cheap = _flat(TB, 0.30, 0.100)
    strong = _flat(TB, 0.50, 0.101)
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))

    assert (0.101 - 0.100) / ((0.100 + 0.101) / 2) < RELSPEND_SPREAD_FLOOR   # the sweep's own scale
    assert LAMBDA_UNDEFINED in rates[TB].flags
    assert rates[TB].lam == 0.0

    # Scored beside a healthy benchmark, terminal-bench does not appear in the mean at all — and the
    # arm says which guard removed it.
    scored = score_arm([*strong, *_flat(CG, 0.40, 0.25)], rates | {CG: RATES[CG]})
    assert [row.benchmark for row in scored.per_benchmark] == [CG]
    assert scored.excluded == ((TB, (LAMBDA_UNDEFINED,)),)
    assert scored.final == pytest.approx(0.40 - RATES[CG].lam * (0.25 / RATES[CG].c_per_task))


def test_a_sweep_whose_strongest_policy_spends_less_is_refused_rather_than_priced_negatively():
    """The floor is compared signed on purpose. A negative slope would make `final` pay a BONUS for
    spending money, which is the spending contest the measured lambda exists to prevent — and it is
    reachable in practice, since the cheapest fixed policy retries where the strongest answers once.
    """
    cheap = _flat(CG, 0.30, 0.30)
    strong = _flat(CG, 0.50, 0.10)
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong])) | {TB: RATES[TB]}

    assert LAMBDA_UNDEFINED in rates[CG].flags
    assert rates[CG].lam == 0.0
    # Excluded, so neither thrift nor extravagance on cyber-gym reaches the score in EITHER
    # direction — where a clamped lambda would have made the two identical by not pricing at all.
    thrifty = score_arm([*_flat(CG, 0.40, 0.01), *_flat(TB, 0.50, 0.11)], rates)
    lavish = score_arm([*_flat(CG, 0.40, 9.99), *_flat(TB, 0.50, 0.11)], rates)
    assert thrifty.final == pytest.approx(lavish.final)
    assert thrifty.final == pytest.approx(score_arm(_flat(TB, 0.50, 0.11), rates).final)


def test_the_dominator_condition_excludes_the_benchmark_and_flags_the_corpus():
    """§5.1b's second guard: the pool buys nothing with money, so lambda <= 0. Flagged rather than
    silently excluded because it is the finding this project has now hit five times — the cheapest
    model dominating on quality-per-dollar is a fact about the MODEL MARKET, not the router or the
    benchmark, and it is the condition that closed the routing thesis (ROUTING_MEASUREMENTS)."""
    cheap = _flat(TB, 0.50, 0.02)
    strong = _flat(TB, 0.40, 0.20)
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong])) | {CG: RATES[CG]}

    assert rates[TB].flags == (DOMINATOR,)
    assert rates[TB].lam == 0.0
    scored = score_arm([*cheap, *_flat(CG, 0.30, 0.25)], rates)
    assert scored.excluded == ((TB, (DOMINATOR,)),)
    assert [row.benchmark for row in scored.per_benchmark] == [CG]


def test_two_guarded_benchmarks_cannot_buy_the_crown_with_money_alone():
    """**THE ATTACK EXCLUSION EXISTS TO CLOSE** (§5.1b, adversarial finding 3).

    With a fired guard clamping lambda to 0, the flagged benchmark was scored on quality with spend
    UNPRICED — so §5.1b's own claim ("the extra quality bought at the pool's own rate scores zero")
    was false exactly there. Two flagged benchmarks satisfy the breadth rule, so a challenger
    byte-identical to the king on the other ten and simply burning its allowance on the two took the
    crown at 5000x the king's spend (measured: delta +0.0833, loo_min +0.0455).

    Not an exotic corner: the DOMINATOR guard fires precisely when the strongest fixed policy buys
    no quality over the cheapest, the condition this repository has measured five times
    (ROUTING_MEASUREMENTS). Excluded, the two benchmarks are not in either arm's mean, so the
    spendthrift scores exactly what the king scores — a tie, which §5.2's strict `eps` beat refuses.
    """
    def sweep(quality: float, spend: float) -> list[EpisodeResult]:
        """Twelve benchmarks; on the first two the strongest policy buys nothing (0.50 either way),
        which is what fires DOMINATOR there."""
        return [ep for i in range(12)
                for ep in _flat(f"bench{i:02d}", 0.50 if i < 2 else quality, spend)]

    cheap, strong = sweep(0.30, 0.01), sweep(0.90, 1.00)
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert rates["bench00"].flags == (DOMINATOR,) and rates["bench02"].flags == ()

    king = [ep for i in range(12)
            for ep in _flat(f"bench{i:02d}", 0.50 if i < 2 else 0.30, 0.01)]
    burner = [ep for i in range(12)                     # +0.50 quality at 5000x spend, on the two
              for ep in (_flat(f"bench{i:02d}", 1.00, 50.0) if i < 2
                         else _flat(f"bench{i:02d}", 0.30, 0.01))]

    scored_king, scored_burner = score_arm(king, rates), score_arm(burner, rates)
    assert scored_burner.final == pytest.approx(scored_king.final)      # bought nothing
    assert [name for name, _ in scored_burner.excluded] == ["bench00", "bench01"]

    # ...and it is the exclusion that did it, not the fixture. Scored the way the clamp scored —
    # lambda = 0 on the flagged pair, which is the same arithmetic as an unpriced benchmark — the
    # burner leads by the +0.0833 the red team measured, and that delta clears `eps`.
    clamped = {b: Exchange(b, 0.0, 1.0) for b in rates}
    assert score_arm(burner, clamped).final - score_arm(king, clamped).final \
        == pytest.approx(0.0833, abs=5e-4)


def test_an_arm_whose_every_benchmark_is_guarded_is_refused_rather_than_scored_unpriced():
    """§5.1b: if every benchmark fires a guard there is no corpus to launch on. That is a finding
    about the model market (ROUTING_MEASUREMENTS), and the response is to refuse — scoring the whole
    arena on unpriced quality is the frugality-free spending contest the measured lambda exists to
    prevent, run over every benchmark at once.

    Reached, not hypothetical: `simulate.priced` scores ONE benchmark at a time for the duel's
    bootstrap, so a flagged benchmark that reached a duel refuses here rather than pricing a crown.
    """
    cheap = [*_flat(TB, 0.50, 0.02), *_flat(CG, 0.30, 0.100)]
    strong = [*_flat(TB, 0.40, 0.20), *_flat(CG, 0.50, 0.101)]
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert rates[TB].flags == (DOMINATOR,) and rates[CG].flags == (LAMBDA_UNDEFINED,)

    with pytest.raises(ValueError, match="excluded by a fired guard"):
        score_arm(cheap, rates)
    with pytest.raises(ValueError, match=CG):                    # names them, never silent
        score_arm(_flat(CG, 0.40, 9.99), rates)


def test_a_c_b_pinned_off_the_measured_convention_cannot_move_the_spread_guard():
    """**Adversarial finding 5.** `C_b` is free under a paired duel — but the spread guard used to be
    measured in units of it, which made it the one place `C_b` was read as a scale. Deflate `C_b` by
    1e-9 and a benchmark whose two fixed arms differ in spend by 1e-9 went from LAMBDA_UNDEFINED to
    UNFLAGGED, lambda was fitted on noise, and the lambda invariant degraded to 6e-8 — past the 1e-9
    bar — with the published `Exchange` carrying no flag to say so.

    That is the failure mode of pinning lambda_b/C_b BY HAND at M3b instead of deriving them from
    `pinned_cost`, and the inflating direction is the mirror of it: a `C_b` pinned far ABOVE the
    convention would have fired the guard on a healthy benchmark and excluded it from the corpus.
    The guard is now a ratio measured against the sweep's own spend, so both are inert.
    """
    cheap = [*_flat("noise", 0.0, 1.0), *_flat("real", 0.0, 1.0)]
    strong = [*_flat("noise", 1.0, 1.0 + 1e-9), *_flat("real", 1.0, 2.0)]
    honest = pinned_cost([*cheap, *strong])
    expected = {"noise": (LAMBDA_UNDEFINED,), "real": ()}

    for scale in (1e-9, 1.0, 1e9):
        rates = derive_exchange(cheap, strong, {b: c * scale for b, c in honest.items()})
        assert {b: rate.flags for b, rate in rates.items()} == expected, f"at C_b x {scale}"
        # And the invariant the guard protects still holds over what is left to score.
        assert score_arm(cheap, rates).final == pytest.approx(score_arm(strong, rates).final,
                                                              abs=1e-9)


def test_both_guards_report_independently_when_both_conditions_hold():
    """They answer different questions — "can lambda be fitted here?" and "does this pool buy
    anything with money?" — and the second is a fact about the corpus worth publishing even when the
    first has already excluded the benchmark."""
    cheap = _flat(TB, 0.50, 0.100)
    strong = _flat(TB, 0.40, 0.101)
    rates = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert set(rates[TB].flags) == {LAMBDA_UNDEFINED, DOMINATOR}


def test_a_benchmark_with_no_measured_exchange_rate_is_refused_not_scored_at_zero():
    """Scoring it at lambda = 0 is the silent fallback §5.1b forbids, and it is a live failure mode:
    M3a fits lambda on three benchmarks and the other nine arrive at M3b, so a stale table would
    misprice three quarters of the corpus while looking like an ordinary score."""
    with pytest.raises(ValueError, match="frontier-swe"):
        score_arm([*_flat(TB, 0.5, 0.11), *_flat("frontier-swe", 0.5, 0.30)], RATES)


def test_the_score_runs_over_the_benchmarks_present_rather_than_a_fixed_twelve():
    """§6.3b: admission drops benchmarks and rotation adds them, so nothing may be hard-coded to
    twelve. A narrowed slice is scored over what it contains — visibly, via `per_benchmark`, which
    is also the set the leave-one-out breadth rule will run over."""
    narrowed = score_arm(_flat(CG, 0.30, 0.25), RATES)
    assert [row.benchmark for row in narrowed.per_benchmark] == [CG]
    assert narrowed.final == pytest.approx(0.30 - RATES[CG].lam)


def test_benchmark_rows_are_ordered_by_name_whatever_order_the_slice_arrived_in():
    """A mean of means depends on summation order in its last bits, so first-appearance order would
    make the aggregate depend on which benchmark the slice happened to draw first — a difference of
    1e-16 inside a verdict that also compares at 1e-9."""
    forwards = score_arm([*_flat(TB, 0.6, 0.11), *_flat(CG, 0.3, 0.25)], RATES)
    backwards = score_arm([*_flat(CG, 0.3, 0.25), *_flat(TB, 0.6, 0.11)], RATES)
    assert [row.benchmark for row in forwards.per_benchmark] == [CG, TB]
    assert forwards.final == backwards.final


def test_an_arm_with_no_admitted_tasks_is_refused():
    """Reachable rather than hypothetical: §8b.3 excludes a task from BOTH arms when the grader or
    sandbox fails, and a window can lose every one of them."""
    with pytest.raises(ValueError, match="no admitted tasks"):
        score_arm([], RATES)


def test_a_sweep_covering_a_benchmark_in_only_one_arm_is_refused():
    """lambda is a slope between two arms. Taking the intersection instead would drop the benchmark
    from the pinned table silently, and it would resurface as a scoring refusal windows later."""
    with pytest.raises(ValueError, match=CG):
        derive_exchange(CHEAPEST, _flat(TB, 0.6, 0.2), C_PINNED)


# --- §5.4's futility ceiling ----------------------------------------------------------------------

def _episode(benchmark: str, task_id: str, score: float, spend: float) -> EpisodeResult:
    return EpisodeResult(task_id=task_id, benchmark=benchmark, steps=(),
                         graded_score=score, spend_usd=spend, stopped_reason="stop")


_RATE = {"alpha": Exchange(benchmark="alpha", lam=0.1, c_per_task=1.0, flags=()),
         "beta": Exchange(benchmark="beta", lam=0.1, c_per_task=1.0, flags=())}


def _slice(**counts) -> list[TaskSpec]:
    """A slice of the given shape — the ceiling counts its denominator from the tasks now."""
    return [TaskSpec(task_id=f"{b}-{i}", benchmark=b, prompt="", tools=())
            for b, n in counts.items() for i in range(n)]



def test_the_ceiling_is_reached_when_every_remaining_task_scores_full_marks():
    """The bound has to be TIGHT as well as valid, or it never fires: a ceiling nothing can attain
    would abandon nobody and save no money."""
    observed = [_episode("alpha", "a1", 0.0, 0.0)]

    ceiling = best_possible_final(observed, tasks=_slice(alpha=4), exchange=_RATE)

    assert ceiling == pytest.approx(0.75)          # (0 + 3 x 1.0) / 4, at zero further spend


def test_no_continuation_can_beat_the_ceiling(): 
    """The property the whole check rests on, asserted over every continuation of a small slice
    rather than argued: a false abort spends a miner's one submission forever (§7)."""
    observed = [_episode("alpha", "a1", 0.3, 0.02)]
    ceiling = best_possible_final(observed, tasks=_slice(alpha=3), exchange=_RATE)

    for second in (0.0, 0.25, 0.5, 0.75, 1.0):
        for third in (0.0, 0.25, 0.5, 0.75, 1.0):
            for spend in (0.0, 0.01, 0.5):
                complete = observed + [_episode("alpha", "a2", second, spend),
                                       _episode("alpha", "a3", third, spend)]
                assert score_arm(complete, _RATE).final <= ceiling + 1e-12


def test_the_ceiling_falls_as_an_arm_scores_badly_and_never_rises():
    """Checked at batch boundaries, so a check that does not fire cannot have fired earlier."""
    running, previous = [], None
    for index in range(1, 5):
        running.append(_episode("alpha", f"a{index}", 0.0, 0.0))
        ceiling = best_possible_final(running, tasks=_slice(alpha=4), exchange=_RATE)
        if previous is not None:
            assert ceiling <= previous
        previous = ceiling
    assert previous == pytest.approx(0.0)


def test_spend_already_made_lowers_the_ceiling_because_spend_only_grows():
    """`final_b` prices spend, and money spent cannot come back — so the most favourable remaining
    spend is zero, not a refund."""
    thrifty = [_episode("alpha", "a1", 1.0, 0.0)]
    spendthrift = [_episode("alpha", "a1", 1.0, 2.0)]

    assert (best_possible_final(spendthrift, tasks=_slice(alpha=2), exchange=_RATE)
            < best_possible_final(thrifty, tasks=_slice(alpha=2), exchange=_RATE))


def test_the_ceiling_weights_benchmarks_equally_like_the_score_it_bounds():
    """§5.1 aggregates per BENCHMARK, so a ceiling that averaged tasks would bound a different
    quantity than the verdict computes — and could bound it too low."""
    observed = [_episode("alpha", f"a{i}", 0.0, 0.0) for i in range(9)]

    ceiling = best_possible_final(observed, tasks=_slice(alpha=9, beta=1), exchange=_RATE)

    assert ceiling == pytest.approx(0.5)     # alpha pinned at 0.0, beta still able to reach 1.0


def test_a_flagged_benchmark_leaves_the_ceiling_as_it_leaves_the_score():
    flagged = {"alpha": _RATE["alpha"],
               "beta": Exchange(benchmark="beta", lam=0.0, c_per_task=1.0, flags=("dominated",))}
    observed = [_episode("alpha", "a1", 0.0, 0.0)]

    ceiling = best_possible_final(observed, tasks=_slice(alpha=2, beta=8), exchange=flagged)

    assert ceiling == pytest.approx(0.5)     # beta excluded entirely, exactly as `score_arm` does


def test_the_ceiling_ignores_tasks_a_dead_grader_took_out_of_the_comparison():
    """The half of the "exact bound" that was not exact, and it failed in the one direction that
    costs a miner their submission.

    §6.3c drops a task whose grader or sandbox died from both arms and from the denominator — but
    `_GraderGuard` hands the episode back a placeholder 0.0 first. To a bound that does not know
    about the exclusion, a task on its way OUT of the comparison looks exactly like a task this arm
    scored zero on, so two dead graders early in a slice drag the ceiling down and abort an arm that
    would have won. Measured: ceiling 0.50 against a bar of 0.60, on a challenger whose
    post-exclusion score was 1.00 against a king's 0.11.
    """
    tasks = _slice(alpha=4)
    dead = frozenset({"alpha-0", "alpha-1"})
    placeholders = [_episode("alpha", "alpha-0", 0.0, 0.0), _episode("alpha", "alpha-1", 0.0, 0.0)]

    blind = best_possible_final(placeholders, tasks=tasks, exchange=_RATE)
    aware = best_possible_final(placeholders, tasks=tasks, exchange=_RATE, excluded=dead)

    assert blind == pytest.approx(0.5), "the placeholder zeros were being counted as scores"
    assert aware == pytest.approx(1.0), "an excluded task leaves numerator and denominator alike"


def test_an_excluded_task_leaves_the_ceiling_exactly_as_it_leaves_the_score():
    """The property that makes the bound valid again: the ceiling and `score_arm` must agree about
    which tasks are in the comparison."""
    tasks = _slice(alpha=4)
    dead = frozenset({"alpha-3"})
    run = [_episode("alpha", f"alpha-{i}", 0.5, 0.1) for i in range(4)]

    ceiling = best_possible_final(run, tasks=tasks, exchange=_RATE, excluded=dead)
    actual = score_arm([r for r in run if r.task_id not in dead], _RATE).final

    assert actual <= ceiling + 1e-12


def test_no_continuation_beats_the_ceiling_across_many_random_worlds():
    """The property the futility abort rests on, searched rather than illustrated.

    The single-world test above enumerates continuations of ONE observed episode, and that is too
    thin to see the error it most needs to catch: pricing the SUM of spend where `final_b` prices the
    mean is invisible until a benchmark has more than one observation. That bug was in the first
    version of this bound and made the ceiling smaller than scores the arm could actually reach —
    wrong in the one direction that abandons a challenger who would have won.

    Seeded, so a failure is reproducible rather than a rumour.
    """
    rng = random.Random(20260905)
    checked = 0

    for _ in range(300):
        names = ["alpha", "beta", "gamma"][:rng.randint(1, 3)]
        exchange = {n: Exchange(benchmark=n, lam=rng.uniform(0.0, 2.0),
                                c_per_task=rng.uniform(0.01, 3.0), flags=()) for n in names}
        counts = {n: rng.randint(2, 5) for n in names}
        tasks = [TaskSpec(task_id=f"{n}-{i}", benchmark=n, prompt="", tools=())
                 for n, total in counts.items() for i in range(total)]

        observed = [_episode(n, f"{n}-{i}", rng.random(), rng.uniform(0.0, 0.5))
                    for n, total in counts.items()
                    for i in range(rng.randint(0, total - 1))]      # several per benchmark
        if not observed:
            continue
        ceiling = best_possible_final(observed, tasks=tasks, exchange=exchange)

        for _ in range(20):
            complete = list(observed)
            seen = {r.task_id for r in observed}
            for task in tasks:
                if task.task_id not in seen:
                    complete.append(_episode(task.benchmark, task.task_id,
                                             rng.choice([1.0, rng.random()]),
                                             rng.choice([0.0, rng.uniform(0.0, 0.3)])))
            checked += 1
            assert score_arm(complete, exchange).final <= ceiling + 1e-9, (
                f"a continuation beat the ceiling: {score_arm(complete, exchange).final} > {ceiling}")

    assert checked > 1_000, "the search has to actually search"
