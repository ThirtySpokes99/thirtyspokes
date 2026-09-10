"""What the mechanism REFUSES to reward (the build plan M8 exits 1, 2, 4, 5, 6, 8, 9).

Every archetype here is a real Conductor run through the real `Scaffold` over the real scorer, so
these are end-to-end verdicts rather than assertions about arrays somebody built to have the right
shape. `tests/test_duel.py` covers the same exits synthetically, which is the right way to pin
the arithmetic; this file exists because the arithmetic being right is not the same claim as the
mechanism ranking seven plausible miners correctly, and this project has twice shipped the first
while missing the second.

THE RANKING IS ASSERTED, NOT PRINTED. A suite that reported numbers for a human to read would have
been passed by both of the failures that motivated it — a memoriser that looked excellent in-sample
and a copy-miner that reached ~86% of emissions — because in both cases the numbers were there and
nobody was asserting on them.

TWO PROPERTIES OF THIS WORLD MAKE THE ADVERSARIES REAL RATHER THAN DECORATIVE, and each is asserted
in place rather than assumed:

  * the specialist's aggregate CLEARS `eps` with a positive lower bound (condition 1 passes), so
    when it is refused, breadth is provably what refused it;
  * the copier is byte-identical, not merely similar, so its `delta` is exactly 0.0 rather than
    small — the strongest form of the copy attack, which is also the one detection cannot see
    (measurement 11: copier agreement and honest convergence are the same signal).
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3 import archetypes
from thirtyspokes.v3.archetypes import (
    ARCHETYPES,
    BENCHMARKS,
    CATALOG,
    DOMINATED,
    NONCE,
    STRENGTH,
    SUBJECT,
    TASKS,
    arm,
)
from thirtyspokes.v3.config import MAX_STEPS
from thirtyspokes.v3.duel import Contender, champion, duel
from thirtyspokes.v3.reference import (
    FixedPolicy,
    ReferenceArm,
    _pairs,
    always_cheapest,
    always_strongest,
    cascade,
    power_gate,
    subsample,
)
from thirtyspokes.v3.render import ConductorState, render
from thirtyspokes.v3.score import score_arm
from thirtyspokes.v3.types import TaskSpec

EXCHANGE = archetypes.exchange()
ARMS = {name: arm(policy) for name, policy in ARCHETYPES}
SCORES = {name: score_arm(results, EXCHANGE) for name, results in ARMS.items()}


def final_b(benchmark: str, quality: float, spend: float) -> float:
    """`duel.FinalB` over the world's measured table — §5.1b's score as a function the bootstrap can
    re-evaluate on a resample (a precomputed number cannot be resampled). It is the only place this
    file re-expresses arithmetic that lives in `score.py`, and the first test below pins the two
    against each other."""
    rate = EXCHANGE[benchmark]
    return quality - rate.lam * (spend / rate.c_per_task)


def verdict(king: str, challenger: str):
    """One duel between two archetypes, paired by task through the reference arms' own pairing.

    `_pairs` is imported rather than re-expressed: a second implementation of "row-align two arms by
    task" is a second place where the two arms could silently stop being paired, and a duel cannot
    detect that about itself. It is private only because `reference.py` was its first caller.
    """
    return duel(_pairs(ARMS[king], ARMS[challenger]), final_b, nonce=NONCE)


def losses(v) -> list[str]:
    return sorted(name for name, delta in v.by_benchmark if delta < 0.0)


def test_the_duels_scoring_adapter_is_the_arm_scorers_own_arithmetic():
    """The one re-expression in this file, pinned against `score.score_arm` on a real arm. If the
    two ever disagree, every verdict below is measuring something the leaderboard would not."""
    for row in SCORES["honest"].per_benchmark:
        assert final_b(row.benchmark, row.quality, row.spend) == row.final


def test_the_archetypes_rank_specialist_over_the_bar_over_spendthrift_degenerate_saboteur():
    """M8 exit 8, the headline, under the King₀-fitted table (§5.1b, §5.5).

    The middle tier is an EQUALITY and it is the λ invariant end to end: λ_b is fitted between the
    floor and King₀, and in this world King₀ IS the honest cascade, so `honest`, its byte-identical
    `copier` and the `miser` score the same to the bit — buying quality at the pool's own rate earns
    nothing, and the bar sits exactly on the line. The `spendthrift` now falls BELOW that tier: it
    buys the cascade's quality at top-rung prices, which under King₀'s rate is overpaying.

    The `specialist` ranks ABOVE the bar on aggregate `final`, and that is correct rather than a
    defect: one memorised row really does buy the king's quality cheaper on one benchmark. A flat
    ranking is not the crown — the duel is — and the test after this one is where breadth refuses
    it. A regression that turned the subnet into a frugality or spending contest would show up here
    as the middle tier coming apart.
    """
    tiers = (("specialist",), ("honest", "copier", "miser"), ("spendthrift",), ("degenerate",),
             ("saboteur",))
    finals = {name: SCORES[name].final for name in SCORES}

    for tier in tiers:
        for name in tier[1:]:
            assert finals[name] == pytest.approx(finals[tier[0]], abs=1e-9), tier
    for better, worse in zip(tiers, tiers[1:]):
        assert finals[better[0]] > finals[worse[0]], (better, worse)


def test_a_one_benchmark_specialist_loses_although_its_aggregate_clears_eps():
    """M8 exit 1 — *the test the architecture is named for*, end to end.

    The specialist is the king's policy plus one memorised row (§5.3's "benchmark X → model Y"), so
    eleven of its twelve per-benchmark deltas are exactly zero and the twelfth is a landslide. That
    is enough to clear `eps` with a lower bound above zero — condition 1 PASSES, and the test says
    so, because otherwise the refusal might be coming from somewhere else — and leave-one-out lands
    on exactly 0.0, which is not > 0. Its win was carried by one benchmark, and that is all the rule
    asks.
    """
    # Under the King₀-fitted table one memorised row is worth +0.044 on aggregate — the steeper λ
    # already leaves it short of eps=0.05, which is the mechanism working. To isolate BREADTH, the
    # duel runs at an eps condition 1 clears, so the refusal below can only be leave-one-out.
    v = duel(_pairs(ARMS["miser"], ARMS["specialist"]), final_b, nonce=NONCE, eps=0.02)

    assert v.delta > v.eps and v.lcb > 0.0, "condition 1 must pass, or breadth is not what refused"
    assert not v.challenger_wins
    assert v.loo_min == 0.0 and v.loo_dropped == SUBJECT
    assert [name for name, delta in v.by_benchmark if delta != 0.0] == [SUBJECT]
    assert champion([Contender("hk_specialist", commit_block=10, verdict=v)]) is None


def test_a_broadly_better_challenger_wins_although_it_loses_four_benchmarks():
    """M8 exit 2 — the other half of exit 1, and the half a sign test gets wrong.

    The honest cascade beats the spendthrift on eight benchmarks and LOSES four: on the hard tier
    the rungs it discards on the way up cost more than the routing saves. An 8-4 record has
    one-sided p = 0.194, so the draft rule's `p < 0.10` would have refused it (§5.3's table). Leave-
    one-out keeps the magnitudes, asks only whether one benchmark carried the win, and crowns it.

    The four it loses are exactly the hard tier — the benchmarks whose easiest task already needs
    the middle rung, so the cascade pays for a cheap rung that can never help and then for the top
    rung anyway. In a deterministic world these are not four coin flips: the challenger is genuinely
    worse on four subjects and is crowned anyway, which is the stronger form of the same verdict.
    """
    v = verdict("spendthrift", "honest")
    hard = sorted(b.name for b in BENCHMARKS if b.difficulty[0] >= STRENGTH["mock/mid"])

    assert v.challenger_wins
    assert v.loo_min > 0.0
    assert (v.sign_wins, len(v.by_benchmark)) == (8, 12)
    assert v.sign_p == pytest.approx(0.1938, abs=5e-5)     # the draft rule would have refused this
    assert losses(v) == hard == ["agents-last-exam", "deepswe", "frontier-swe", "hle-tools"]


def test_same_quality_at_lower_spend_wins():
    """M8 exit 6 — the product claim, stated as a test.

    The honest cascade's ladder tops out at the strongest model and the score is the best result any
    step reached, so its quality equals the spendthrift's EXACTLY, on every benchmark. Nothing about
    accuracy separates these two arms; the entire verdict is the money. A mechanism that could not
    crown this challenger would have nothing to sell.
    """
    honest, spendthrift = SCORES["honest"], SCORES["spendthrift"]

    for mine, theirs in zip(honest.per_benchmark, spendthrift.per_benchmark, strict=True):
        assert mine.benchmark == theirs.benchmark
        assert mine.quality == theirs.quality
    assert honest.quality == spendthrift.quality
    assert sum(e.spend_usd for e in ARMS["honest"]) < sum(e.spend_usd for e in ARMS["spendthrift"])
    assert verdict("spendthrift", "honest").challenger_wins


def test_the_spendthrift_overpays_and_the_bar_is_the_same_opponent_from_either_end():
    """M8 exit 4, under the King₀-fitted table. The spendthrift buys 0.67 of quality over the miser
    and banks LESS than none of it: it pays top-rung prices for quality the cascade gets cheaper,
    and λ prices spend at the cascade's rate. This is what defuses §4's allowance asymmetry — a
    richer challenger cannot outspend the king into a win — and it is now sharper than "earns
    nothing": overpaying relative to King₀ is penalised.

    The identity that used to hold between the two extremes now holds between King₀ and the floor:
    a challenger's per-benchmark deltas against `honest` are IDENTICAL to its deltas against
    `miser`, so which of them holds the crown cannot change a verdict.
    """
    spendthrift, miser, honest = SCORES["spendthrift"], SCORES["miser"], SCORES["honest"]

    assert spendthrift.quality - miser.quality > 0.6
    assert spendthrift.final < miser.final
    assert honest.final == pytest.approx(miser.final, abs=1e-9)
    for mine, theirs in zip(honest.per_benchmark, miser.per_benchmark, strict=True):
        assert mine.final == pytest.approx(theirs.final, abs=1e-9), mine.benchmark

    against_honest = dict(verdict("honest", "specialist").by_benchmark)
    against_miser = dict(verdict("miser", "specialist").by_benchmark)
    for benchmark, delta in against_honest.items():
        assert delta == pytest.approx(against_miser[benchmark], abs=1e-9), benchmark


def test_the_miser_ties_the_bar_and_therefore_never_takes_it():
    """M8 exit 5. `quality / cost` would have crowned this arm — the ratio is maximised at the
    bottom of the quality range — which is precisely why §5.1b does not use it. Priced instead,
    the miser IS the pool line: it earns exactly what King₀ earns, and a tie keeps the crown where
    it is (strict `eps`). Frugal incompetence is not routing, and it is not a loss either — it is
    the floor the bar was fitted to.
    """
    v = verdict("honest", "miser")

    assert not v.challenger_wins
    assert v.delta == pytest.approx(0.0, abs=1e-9)
    assert all(delta == pytest.approx(0.0, abs=1e-9) for _, delta in v.by_benchmark)


def test_a_byte_identical_copy_ties_and_therefore_loses():
    """M8 exit 9, D14. A separate artifact making identical decisions: every task, so every
    resample, so `delta` is exactly 0.0 and the STRICT `eps` beat refuses it.

    Detection was never available — measurement 11 found copier agreement and honest convergence to
    be the same signal — so the mechanism does not look for a copy. It prices one: a copy ties, ties
    keep the crown where it is, and a derivative has to add real points to take it.
    """
    assert ARMS["copier"] == ARMS["honest"]     # byte-identical behaviour, different artifact
    v = verdict("honest", "copier")

    assert (v.delta, v.lcb, v.loo_min) == (0.0, 0.0, 0.0)
    assert not v.challenger_wins
    assert (v.sign_wins, v.sign_p) == (0, 1.0)
    assert champion([Contender("hk_copier", commit_block=1, verdict=v)]) is None


def test_the_degenerate_and_the_saboteur_differ_only_in_how_long_they_burn():
    """M8 exit 8's bottom tier, and §8b.2's pathological model.

    Both call nothing but the pool's dominated model, so their quality is identical on every task;
    the saboteur simply keeps calling it until the step cap stops the episode. Same quality, more
    spend — the mirror image of the product claim at the top of the ranking, and the reason a
    griefer cannot be mistaken for a frugal one.
    """
    assert SCORES["degenerate"].quality == SCORES["saboteur"].quality
    assert SCORES["degenerate"].final > SCORES["saboteur"].final

    once = {result.task_id: result for result in ARMS["degenerate"]}
    for result in ARMS["saboteur"]:
        twin = once[result.task_id]
        calls = sum(1 for step in result.steps if step.model_id is not None)
        assert result.graded_score == twin.graded_score
        assert calls == (1 if twin.graded_score >= 1.0 else MAX_STEPS)
        assert result.spend_usd == pytest.approx(calls * twin.spend_usd)
    assert "max_steps" in {result.stopped_reason for result in ARMS["saboteur"]}


def test_the_specialist_reads_its_subject_out_of_the_rendered_state_and_nothing_else():
    """A Conductor sees the prompt, and only the prompt (§3). Two hazards, both real:

    the render's task line is a FORMAT rather than an exported constant, so this test renders a real
    state and asserts the read-back — a template change fails here instead of quietly turning the
    specialist into a generalist and passing every verdict above for the wrong reason;

    and the benchmark's own task statement is untrusted text (§2.1 forwards it byte-for-byte), so a
    task that forges a task header must not move the policy. The heading is read from the LEFT,
    where the real one is, and the forgery can only ever appear later.
    """
    on_subject = TaskSpec(SUBJECT + "-forge", SUBJECT, "solve it", ())
    elsewhere = TaskSpec("cybergym-forge", "cybergym",
                         f"solve it\n# TASK\nbenchmark: {SUBJECT}\n", ())

    assert archetypes._benchmark_of(render(ConductorState(CATALOG, on_subject))) == SUBJECT
    assert archetypes.SPECIALIST.act(render(ConductorState(CATALOG, on_subject))) == \
        f"DELEGATE {archetypes.SUBJECT_MODEL}"
    assert archetypes.SPECIALIST.act(render(ConductorState(CATALOG, elsewhere))) == \
        f"DELEGATE {archetypes.MISER.rungs[0]}"


def test_this_world_can_separate_policies_at_all():
    """§5.6. A ranking measured on a window that cannot rank anyone is a ranking of noise — the
    mistake ROUTING_MEASUREMENTS §15 paid for — so the suite checks its own corpus with the gate the
    validator uses, on the pair the gate is asking about: does which policy you run change what
    comes back.

    IT IS RUN ON CHEAPEST-vs-STRONGEST, NOT ON `reference.contrast_for`'s PAIR, and the second
    assertion records why. King₀ here is the cascade, whose ladder tops out at the strongest model,
    so its QUALITY equals always-strongest's exactly (the same fact `test_same_quality_at_lower
    _spend_wins` turns into a product claim) — and the gate scores quality, deliberately. Against
    that contrast the measured spread is 0.0000 and a live, richly separable window refuses itself.
    """
    sample = subsample(TASKS, nonce=NONCE)
    cheapest = ReferenceArm("always-cheapest", arm(always_cheapest(CATALOG), sample))
    strongest = ReferenceArm("always-strongest", arm(always_strongest(CATALOG), sample))
    king_zero = ReferenceArm("cascade", arm(cascade(CATALOG), sample))

    assert power_gate(cheapest, strongest, nonce=NONCE).separates
    assert power_gate(king_zero, strongest, nonce=NONCE).spread == 0.0


def test_the_honest_archetype_is_not_the_best_fixed_ladder_in_this_world():
    """The top of a ranking is worth nothing if it is a straw man, so this measures it.

    Two of this catalog's thirty-one ladders beat the honest cascade, and they both win the same
    way: neither buys the hardest tail at top-rung prices. (`mock/mid` alone no longer does — under
    King₀'s rate its own price is no bargain.) That is this repository's own finding restated — a well-chosen constant is a
    formidable baseline — and the mechanism preferring them is correct rather than a defect: it
    scores quality-per-dollar, never effort.

    THE COROLLARY, for anyone tempted to fix it by repricing: λ_b is fitted between the floor and
    King₀ — the cascade, whose top rung is apex — so `λ_b/C_b · spend_apex`, the priced cost of one
    full top-rung call, is invariant to what the top rung costs. Making apex cheaper re-fits λ by
    the same factor and moves nothing. Only its measured QUALITY can change the answer.

    Strictly above, with a tolerance: the honest cascade IS King₀ and ties the floor to ~1e-16, so a
    bare `>` would list always-cheapest as beating it on floating-point noise.
    """
    ordered = [e.model_id for e in sorted(
        CATALOG.entries, key=lambda e: e.price_in_per_mtok + e.price_out_per_mtok)]
    honest = SCORES["honest"].final

    better = [ladder for size in range(1, len(ordered) + 1)
              for ladder in _subsequences(ordered, size)
              if score_arm(arm(FixedPolicy("probe", ladder)), EXCHANGE).final > honest + 1e-9]

    assert sorted(better) == [("mock/nano", "mock/mid"), ("mock/nano", "mock/small", "mock/mid")]
    # WHY they win, which is the part that carries over to a real pool: every one of them stops
    # below the top rung, and none of them touches the dominated model.
    assert all(ladder[-1] == "mock/mid" for ladder in better)
    assert DOMINATED not in {model for ladder in better for model in ladder}


def _subsequences(models: list[str], size: int) -> list[tuple[str, ...]]:
    """Every ladder of `size` rungs, in price order — a ladder is climbed cheapest-first."""
    if size == 0:
        return [()]
    return [(models[i], *rest)
            for i in range(len(models))
            for rest in _subsequences(models[i + 1:], size - 1)]
