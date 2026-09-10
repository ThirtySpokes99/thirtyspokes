"""The end-to-end window (the build plan M9 exit 1) and the operational rules it has to hold.

Each rule below is a property of the ORDER the validator does things in, not of any one module, so
none of them can be tested where the module lives. That is the whole reason this file exists: the
king's arm being computed once is `simulate`'s business, and a green `test_duel.py` says nothing
about it.

The world these tests run in is built so that routing skill EXISTS and is not the obvious thing —
`mock/premium` is the most expensive model and only the second strongest, so the price-ordered
cascade never tries `mock/mid`, which is strong enough for everything at half the price of the rung
the cascade does climb to. Without that, a window in which the cheapest model dominates would make
every one of these tests pass for the wrong reason (ROUTING_MEASUREMENTS: five times, the blocker
was the pool).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from thirtyspokes.v3 import simulate
from thirtyspokes.v3.admission import describe
from thirtyspokes.v3.archetypes import Copier
from thirtyspokes.v3.benchmarks.mock import MockBenchmark, worker_answers
from thirtyspokes.v3.conductor import Conductor
from thirtyspokes.v3.config import MAX_DUELS_PER_WINDOW, MAX_QUEUE_DEPTH
from thirtyspokes.v3.devkit import suite_grade
from thirtyspokes.v3.emissions import KING0
from thirtyspokes.v3.score import score_arm
from thirtyspokes.v3.simulate import (DEFERRED, DUELLED, REFUSED, SKIPPED, UNQUEUED, Submission,
                                       Validator, format_window, learned_router, pin_corpus,
                                       priced, run_simulation)
from thirtyspokes.v3.tools import PROTOCOL
from thirtyspokes.v3.types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from thirtyspokes.v3.window import Paired
from thirtyspokes.v3.worker import MockWorker
from thirtyspokes.subnet.chain import MockChain

# Prices are deliberately NOT a capability ranking (see the module docstring).
CATALOG = Catalog(entries=(
    CatalogEntry("mock/cheap", 0.20, 0.80, 128_000),
    CatalogEntry("mock/mid", 0.40, 1.60, 200_000),
    CatalogEntry("mock/heavy", 1.00, 4.00, 300_000),
    CatalogEntry("mock/premium", 1.20, 4.80, 400_000),
))
STRENGTH = {"mock/cheap": 1, "mock/mid": 3, "mock/heavy": 3, "mock/premium": 2}
COST = {"mock/cheap": 0.04, "mock/mid": 0.08, "mock/heavy": 0.16, "mock/premium": 0.20}

# One benchmark the priciest model cannot do at all (so the gate has a quality spread to measure)
# and two where the middle rung is the whole prize (so there is a band a router can capture).
LIVE = (MockBenchmark("frontier-swe", STRENGTH, n_tasks=4, difficulty=(3, 3)),
        MockBenchmark("nl2repo", STRENGTH, n_tasks=4, difficulty=(2, 2)),
        MockBenchmark("programbench", STRENGTH, n_tasks=4, difficulty=(2, 2)))
ROUTABLE = ("nl2repo", "programbench")

# A window the gate must refuse. Every model is ONE SUBGOAL IN TWENTY-FIVE apart here, not equal:
# a world where which policy you run changes literally nothing is a world where the strongest fixed
# policy buys no quality over the cheapest, which is §5.1b's DOMINATOR condition, so every benchmark
# is excluded and the corpus is refused before a window can be opened at all. The gate's other half
# is what catches this one — a spread of 0.04 that is real, perfectly separable from zero, and below
# `POWER_SPREAD_FLOOR`. That is exactly the case the floor was written for: ROUTING_MEASUREMENTS §15
# ran an epoch whose oracle beat best-single by +0.0078 and the mechanism paid out anyway.
DEAD = (MockBenchmark("flat-a", STRENGTH, n_tasks=4, difficulty=(2, 2), subgoals=25),
        MockBenchmark("flat-b", STRENGTH, n_tasks=4, difficulty=(2, 2), subgoals=25))


@dataclass
class Spy:
    """A Conductor that counts the turns it was asked for — how "once per window" is measured."""

    inner: Conductor
    calls: int = 0

    def act(self, prompt: str) -> str:
        self.calls += 1
        return self.inner.act(prompt)


def build(root: Path, benchmarks=LIVE, *, grade=None,
          per_benchmark: int = 3, minimum: int = 2) -> tuple[Validator, Path]:
    """A validator over a small offline corpus, plus the model tree its submissions pass.

    The worker is not injectable: a sweep run against a broken pool measures the outage rather than
    the pool (see `test_a_worker_failure_is_an_outcome_and_stays_in_the_score`), so a test that wants
    a dead model swaps it into `pins.world` AFTER the corpus is pinned, exactly as `grade` does.
    """
    worker = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
    tasks = tuple(task for benchmark in benchmarks for task in benchmark.load())
    pins = pin_corpus(CATALOG, tasks, suite_grade(benchmarks), worker)
    if grade is not None:
        # Fitted with the honest grader, then made flaky: λ_b is pinned per corpus (§5.1b) and a
        # sweep that hit the dead sandbox would refuse before any window opened.
        pins = replace(pins, world=replace(pins.world, grade=grade(pins.world.grade)))
    tree = simulate._write_tree(root / "valid")
    chain = MockChain()
    chain.register("burn")                        # uid 0
    validator = Validator(pins=pins, reference=describe(tree), chain=chain, windows=(1, 2, 3),
                          per_benchmark=per_benchmark, minimum=minimum)
    return validator, tree


def router(validator: Validator, learned=ROUTABLE) -> Conductor:
    return learned_router(validator.pins.king_zero, learned, "mock/mid")


def submission(validator: Validator, hotkey: str, block: int, conductor: Conductor, tree: Path,
               *, budget_usd: float = 100.0) -> Submission:
    validator.chain.register(hotkey)
    return Submission(hotkey=hotkey, commit_block=block, model_tree=tree,
                      serve=lambda: conductor, budget_usd=budget_usd)


def duelled(report, hotkey: str):
    return next(o for o in report.outcomes if o.hotkey == hotkey)


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    """The shipped `orchestra-sim` cast, run once: three windows, two coronations."""
    return run_simulation(tmp_path_factory.mktemp("sim"), verbose=False)


# --- the whole thing ------------------------------------------------------------------------------


def test_a_full_window_runs_offline_end_to_end_and_crowns_the_better_conductor(simulation):
    first = simulation[0]
    assert first.scored and first.power.separates
    assert first.crowned == "router-b"
    assert duelled(first, "router-a").won and duelled(first, "router-b").won
    assert not duelled(first, "copycat").won
    assert duelled(first, "dud").status == REFUSED
    # The traces D15 publishes as the entry ramp: the king's arm and the reference arms, with the
    # per-step record a miner trains on.
    assert len(first.reference) == 2
    assert all(arm.results for arm in first.reference)
    assert any(result.steps for result in first.king_results)


def test_every_arm_reaches_the_read_channel_through_the_guard(tmp_path, monkeypatch):
    """The offline half of the wiring §8b.3 needs at the OTHER container (`validator._arm` has the
    live one). `benchmarks.base.inspector` catches nothing by design, so if an arm holds the
    adapter's own inspector rather than `guard.inspecting`, one dead container abandons the whole
    window instead of dropping one task from both arms — and a miner who routes to a model they
    control can produce that container on demand. Asserted on what the arm CONSTRUCTS, because the
    offline worlds declare no tools and so never enter the loop."""
    seen = []
    built = simulate.Scaffold

    def capture(*args, **kwargs):
        seen.append(kwargs.get("inspect"))
        return built(*args, **kwargs)

    monkeypatch.setattr(simulate, "Scaffold", capture)
    run_simulation(tmp_path / "wiring", verbose=False)

    arms = [f for f in seen if f is not None]
    assert len(arms) >= 3, "the reference arms, the king's arm and every challenger arm"
    assert all(getattr(f, "__func__", None) is simulate._GraderGuard.inspecting for f in arms)


def test_a_window_whose_prompts_invite_a_read_the_tasks_do_not_declare_is_refused_unspent(tmp_path):
    """The read channel's third fact, which no signature and no hash covers: `tools.PROTOCOL` is
    appended by the ADAPTER (that is what keeps §2.1's audit sentence byte-true), so a corpus where
    the protocol and the declared verbs disagree verifies perfectly and runs a whole arm. Here the
    prompt invites a read and the task declares no verb, so `Scaffold._read` would refuse the call
    and the worker's `READ` line would be graded as its answer. Refused at window open, before the
    power gate — the money is the point, so the witness is that nothing reached the pool."""
    tasks = tuple(task for benchmark in LIVE for task in benchmark.load())
    pins = pin_corpus(CATALOG, tasks, suite_grade(LIVE),
                      MockWorker(answers=worker_answers(STRENGTH), costs=COST))
    # Every task, so that whichever slice the nonce draws carries one (§6.3b draws after commits).
    pool = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
    pins = replace(pins, world=replace(
        pins.world, worker=pool,
        tasks=tuple(replace(t, prompt=t.prompt + PROTOCOL) for t in tasks)))
    chain = MockChain()
    chain.register("burn")
    validator = Validator(pins=pins, reference=describe(simulate._write_tree(tmp_path / "valid")),
                          chain=chain, windows=(1,), per_benchmark=3, minimum=2)

    with pytest.raises(ValueError, match="read protocol"):
        validator.run_window(1, ())

    assert not pool.seen, "the window was refused before the reference arms spent anything"


def test_assembling_the_offline_arena_runs_the_channel_check_over_its_own_benchmarks(
        tmp_path, monkeypatch):
    """The other half of the same gate, at the other in-repo corpus assembly.

    `check_protocol` above is caught by any corpus that gets it wrong, because a broken corpus is
    constructible in a test. `check_channel` is not: the shipped cast agrees, so the arena builds
    identically whether the line is there or not, and its removal would be invisible to every other
    assertion in this file. The witness therefore has to be the call — asked, and asked about this
    arena's OWN benchmarks rather than about an empty tuple."""
    asked = []
    monkeypatch.setattr(simulate, "check_channel", lambda bs: asked.append(tuple(bs)))

    validator, _ = simulate.mock_arena(tmp_path)

    assert asked, "the arena assembled a corpus without checking the read channel"
    assert asked[0] == simulate._BENCHMARKS
    assert {t.benchmark for t in validator.pins.world.tasks} == {b.name for b in asked[0]}


def test_the_window_is_reproducible_from_the_chain_nonce_alone(tmp_path):
    """Two runs of the same schedule must publish the same reveal, byte for byte.

    Not a nicety: the reveal is what a third party re-derives a crown from, and everything the
    validator does is seeded from the chain beacon — the slice, the task order, the bootstrap. A
    single set iteration anywhere on the path (`koth/matrix.py::scoreable_ids`, the bug this repo
    has already paid for) would show up here and nowhere else.
    """
    first = run_simulation(tmp_path / "a", verbose=False)
    second = run_simulation(tmp_path / "b", verbose=False)
    assert [format_window(r) for r in first] == [format_window(r) for r in second]


# --- §5.2a / D13: one king arm per window ---------------------------------------------------------


def test_the_kings_arm_is_computed_once_per_window_and_shared_by_every_duel(tmp_path):
    """The correctness rule, measured in turns: N challengers must cost the king ONE arm.

    Re-running per duel would compare challenger A against king-run-1 and B against king-run-2, so
    run-to-run variance could crown the worse of the two — and §5.2's batch coronation ranks
    challengers against each other, which is only meaningful against one baseline. It also collapses
    the griefing ratio from ~1:1 to N:1.
    """
    def window_with(n: int, root: Path):
        validator, tree = build(root)
        spy = Spy(validator.pins.king_zero)
        # A window that opens with a miner on the throne — the ordinary case (§5.2).
        validator.king = submission(validator, "incumbent", 1, spy, tree)
        queue = [submission(validator, f"ch-{i}", 10 + i, router(validator), tree)
                 for i in range(n)]
        return validator.run_window(1, queue), spy

    one, one_spy = window_with(1, tmp_path / "one")
    two, two_spy = window_with(2, tmp_path / "two")

    assert one_spy.calls > 0                       # the counter would have seen a second arm
    assert two_spy.calls == one_spy.calls          # ...and there was not one
    # Every turn the king's model was asked for appears in the trace the window publishes: one
    # `act` call is one `StepRecord`, so a second arm would be money spent that no published trace
    # accounts for — which is what a re-run per duel is, whether or not the duels then agree.
    assert one_spy.calls == sum(len(result.steps) for result in one.king_results)
    assert two_spy.calls == sum(len(result.steps) for result in two.king_results)
    assert len([o for o in two.outcomes if o.verdict is not None]) == 2
    # Both duels were judged against the same numbers, so ranking them against each other is
    # well defined.
    assert two.king == one.king
    assert {o.verdict.delta for o in two.outcomes} == {duelled(one, "ch-0").verdict.delta}


def test_admission_runs_before_the_kings_arm_so_a_broken_challenger_costs_the_king_nothing(
        tmp_path):
    """§8b.2: an invalid submission must not be able to spend the king's allowance.

    The load check is only worth anything if the king's arm has not already started, so a window
    whose entire queue is invalid runs no king arm at all.
    """
    validator, tree = build(tmp_path)
    broken = simulate._write_tree(tmp_path / "broken", pickled=True)
    spy = Spy(validator.pins.king_zero)
    validator.king = submission(validator, "incumbent", 1, spy, tree)

    refused = validator.run_window(1, [submission(validator, "dud", 5, router(validator), broken)])
    assert spy.calls == 0
    assert refused.king is None and refused.king_results == ()
    assert duelled(refused, "dud").status == REFUSED

    # The control: a valid challenger DOES buy the king an arm, so the assertion above is about
    # admission and not about a spy that never fires.
    scored = validator.run_window(2, [submission(validator, "ok", 6, router(validator), tree)])
    assert spy.calls > 0
    assert scored.king is not None


# --- §7: one shot per hotkey ----------------------------------------------------------------------


def test_an_invalid_artifact_spends_the_shot(tmp_path):
    validator, tree = build(tmp_path)
    broken = simulate._write_tree(tmp_path / "broken", pickled=True)
    report = validator.run_window(
        1, [submission(validator, "dud", 5, validator.pins.king_zero, broken)])

    outcome = duelled(report, "dud")
    assert outcome.status == REFUSED and outcome.shot_spent
    assert "dud" in validator.spent
    # The refusal is the validator's own words, not a paraphrase: it is what the miner greps for.
    assert "pickle archive refused" in outcome.detail


def test_a_spent_hotkey_is_skipped_and_never_rejudged(tmp_path):
    """A hotkey with any prior entry is refused forever (§7) — skipped, not re-run, not re-charged.

    Skipping is not the same as refusing: nothing is spent, no arm runs, and — with nothing else in
    the queue — the king is not touched either.
    """
    validator, tree = build(tmp_path)
    broken = simulate._write_tree(tmp_path / "broken", pickled=True)
    spy = Spy(validator.pins.king_zero)
    validator.king = submission(validator, "incumbent", 1, spy, tree)
    dud = submission(validator, "dud", 5, validator.pins.king_zero, broken)

    validator.run_window(1, [dud])
    again = validator.run_window(2, [dud])

    outcome = duelled(again, "dud")
    assert outcome.status == SKIPPED
    assert not outcome.shot_spent and outcome.verdict is None
    assert again.king is None and spy.calls == 0


# --- §5.6: the power gate -------------------------------------------------------------------------


def test_a_window_that_cannot_separate_two_fixed_policies_runs_no_duels_and_spends_no_shot(
        tmp_path):
    """The measurement-15 rule: a window nobody can be ranked on must cost a delay, not a shot.

    Our inability to measure is never charged to a miner, so challengers roll over with their one
    submission intact (§8 step 2) — and because the gate runs two FIXED policies, a near-optimal
    king can never stall the arena by being good.
    """
    validator, tree = build(tmp_path, DEAD)
    spy = Spy(validator.pins.king_zero)
    validator.king = submission(validator, "incumbent", 1, spy, tree)
    challenger = submission(validator, "hopeful", 5, router(validator), tree)
    already = submission(validator, "used-up", 6, router(validator), tree)
    validator.spent.add("used-up")

    report = validator.run_window(1, [challenger, already])

    assert not report.scored
    assert duelled(report, "hopeful").status == DEFERRED
    assert not duelled(report, "hopeful").shot_spent
    # A hotkey with nothing left to spend is told so, not told it is still queued.
    assert duelled(report, "used-up").status == SKIPPED
    assert validator.spent == {"used-up"}          # nothing NEW was spent
    assert report.king is None and report.crowned is None and spy.calls == 0
    assert "CANNOT separate policies" in report.power.reason
    # Refused on the FLOOR, not on the lower bound: the spread here is real and perfectly separable
    # from zero, and still too small to host a coronation (§5.2b). A gate that was only a
    # significance test would have scored this window.
    assert report.power.lcb > 0.0 and report.power.spread < report.power.floor
    # The schedule is unchanged rather than absent: a refused window is not owner downtime (§8b.6),
    # and the king keeps earning while the corpus is re-measured next window.
    by_hotkey = {report.metagraph.get(uid, "burn"): share
                 for uid, share in report.weights.items()}
    assert by_hotkey["incumbent"] == pytest.approx(0.85)


def test_a_deferred_challenger_is_still_judgeable_afterwards(tmp_path):
    """Rolling over is only meaningful if the shot is still there when a window can rank it."""
    dead_validator, dead_tree = build(tmp_path / "dead", DEAD)
    dead_validator.run_window(
        1, [submission(dead_validator, "hopeful", 5, router(dead_validator), dead_tree)])
    assert "hopeful" not in dead_validator.spent

    # The same hotkey and artifact, carried into a window whose corpus can rank it.
    alive, tree = build(tmp_path / "alive")
    alive.spent = set(dead_validator.spent)
    report = alive.run_window(1, [submission(alive, "hopeful", 5, router(alive), tree)])
    assert duelled(report, "hopeful").status == DUELLED


# --- §8b.1: the queue -----------------------------------------------------------------------------


def test_a_window_runs_at_most_max_duels_and_the_remainder_roll_over_with_their_shot_unspent(
        tmp_path):
    """§8b.1 / M9 exit 10: the queue is capped by the validator's wall clock, never dropped.

    Challenger arms are serial and each is bounded by the two-hour per-duel clock, so a window holds
    a fixed number of them. What the cap must NOT do is spend the overflow's one submission (§7) on
    a window that never looked at it, so the remainder are deferred with their shot intact — and
    deferred before the ~70 GB pull, so a full window costs them the wait and nothing else.
    """
    validator, tree = build(tmp_path)
    queue = [submission(validator, f"ch-{i:02d}", 10 + i, router(validator), tree)
             for i in range(MAX_DUELS_PER_WINDOW + 2)]
    overflow = [sub.hotkey for sub in queue[MAX_DUELS_PER_WINDOW:]]

    report = validator.run_window(1, queue)

    judged = [o for o in report.outcomes if o.status == DUELLED]
    rolled = [o for o in report.outcomes if o.status == DEFERRED]
    assert len(judged) == MAX_DUELS_PER_WINDOW
    # The overflow is the BACK of the (commit_block, hotkey) queue — seniority decides who waits.
    assert [o.hotkey for o in rolled] == overflow
    assert not any(o.shot_spent for o in rolled)
    assert validator.spent == {o.hotkey for o in judged}

    # Rolling over is only worth anything if the shot is still there when a window has room.
    later = validator.run_window(2, [sub for sub in queue if sub.hotkey in overflow])
    assert [o.status for o in later.outcomes] == [DUELLED] * len(overflow)


def test_a_commit_beyond_the_queue_depth_is_refused_rather_than_accepted(tmp_path):
    """§8b.1 / M9 exit 10: the deregistration trap, closed at the door.

    A challenger who registers, funds a key, uploads ~70 GB and then waits longer than the immunity
    period is deregistered before ever being evaluated — it paid a registration burn and got
    nothing, which is the "Kind" property broken at its most expensive point. The cap is therefore a
    drain time, and a commit past it is turned away rather than taken: its shot is untouched, so it
    is still there to spend once the queue has drained.
    """
    validator, tree = build(tmp_path)
    # Every queued artifact is invalid, so the depth cap is measured without paying for arms: the
    # claim here is about who gets INTO the queue, which admission never reaches.
    broken = simulate._write_tree(tmp_path / "broken", pickled=True)
    queue = [submission(validator, f"ch-{i:02d}", 10 + i, validator.pins.king_zero, broken)
             for i in range(MAX_QUEUE_DEPTH + 1)]
    last = queue[-1].hotkey

    report = validator.run_window(1, queue)

    assert len([o for o in report.outcomes if o.status == REFUSED]) == MAX_QUEUE_DEPTH
    turned_away = duelled(report, last)
    assert turned_away.status == UNQUEUED
    assert not turned_away.shot_spent and last not in validator.spent
    assert f"MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH}" in turned_away.detail

    # It kept its one submission: a later window judges it, which is what "refused, not spent"
    # means — turned away at the door costs the wait and nothing else.
    assert duelled(validator.run_window(2, [queue[-1]]), last).status == REFUSED


# --- §4 / §5.8: money -----------------------------------------------------------------------------


def test_budget_exhaustion_zeroes_the_same_tail_for_both_arms(tmp_path):
    """Both arms run the identical nonce-derived order, so exhaustion cuts them at the same place.

    If the order differed, one arm would be zeroed on a different — possibly easier — set of tasks
    and the duel would stop being paired. The zeroed tasks are asserted to be a SUFFIX of the arm's
    run order, which is what makes "the same count" mean "the same tasks".
    """
    validator, tree = build(tmp_path)
    starved = 0.1                                  # ~2.5 tasks' worth at this world's prices
    validator.king = submission(validator, "incumbent", 1, validator.pins.king_zero, tree,
                                budget_usd=starved)
    report = validator.run_window(
        1, [submission(validator, "twin", 5, validator.pins.king_zero, tree,
                       budget_usd=starved)])

    outcome = duelled(report, "twin")
    assert report.king_exhausted > 0
    assert outcome.exhausted == report.king_exhausted
    reasons = [result.stopped_reason for result in report.king_results]
    cut = len(reasons) - report.king_exhausted
    assert reasons[cut:] == ["budget_exhausted"] * report.king_exhausted
    assert "budget_exhausted" not in reasons[:cut]


def test_a_verdict_decided_by_funding_is_visible_in_the_reveal(tmp_path):
    """§5.8: a richer challenger against a starved king is legitimate under D5 — and must not read
    as skill. The exhaustion point is published for both arms rather than buried in an aggregate."""
    validator, tree = build(tmp_path)
    validator.king = submission(validator, "incumbent", 1, validator.pins.king_zero, tree,
                                budget_usd=0.1)
    report = validator.run_window(
        1, [submission(validator, "rich", 5, validator.pins.king_zero, tree, budget_usd=100.0)])

    assert report.king_exhausted > 0
    assert duelled(report, "rich").exhausted == 0
    assert f"tasks the king's allowance never reached: {report.king_exhausted}" in \
        format_window(report)


# --- §5.2 / §8 step 4: the verdict and batch coronation -------------------------------------------


def test_the_crown_goes_to_the_largest_delta_winner_of_the_window(simulation):
    """Batch coronation: both routers beat King₀, and the better one takes it (§8 step 4)."""
    first = simulation[0]
    a, b = duelled(first, "router-a").verdict, duelled(first, "router-b").verdict
    assert a.challenger_wins and b.challenger_wins
    assert b.delta > a.delta
    assert first.crowned == "router-b"


def test_a_coronation_tie_breaks_on_the_earliest_commit_block(tmp_path):
    """Two identical policies produce identical deltas; seniority decides, not dict order."""
    validator, tree = build(tmp_path)
    late = submission(validator, "aaa-late", 99, router(validator), tree)
    early = submission(validator, "zzz-early", 7, router(validator), tree)

    report = validator.run_window(1, [late, early])

    assert duelled(report, "aaa-late").verdict.delta == duelled(report, "zzz-early").verdict.delta
    assert report.crowned == "zzz-early"


def test_a_copy_of_the_king_ties_and_the_crown_does_not_move(tmp_path):
    """D14 prices derivatives instead of detecting them: a copy ties, and a tie is not a win.

    Measurement 11 found copier agreement and honest convergence to be the same signal, so a
    detector was never available. `eps` is what makes a derivative have to add real points.
    """
    validator, tree = build(tmp_path)
    validator.king = submission(validator, "incumbent", 1, validator.pins.king_zero, tree)
    # A DIFFERENT artifact that behaves identically — the claim is about behaviour, not identity.
    report = validator.run_window(
        1, [submission(validator, "copycat", 5, Copier(validator.pins.king_zero), tree)])

    verdict = duelled(report, "copycat").verdict
    assert verdict.delta == pytest.approx(0.0, abs=1e-12)
    assert not verdict.challenger_wins
    assert report.crowned is None
    assert "does not clear eps" in duelled(report, "copycat").detail


def test_a_win_that_lives_in_one_benchmark_is_refused_by_breadth(tmp_path):
    """§5.2 condition 2, the "Spirit" property: your win must survive losing your best subject.

    A router that has learned exactly one benchmark clears the aggregate margin comfortably and is
    still refused — which is the whole anti-lookup-table mechanism, since without it a policy of
    twelve facts would be enough to tie every competent miner and hand the crown to seniority.
    """
    validator, tree = build(tmp_path)
    narrow = submission(validator, "specialist", 5, router(validator, ("nl2repo",)), tree)

    verdict = duelled(validator.run_window(1, [narrow]), "specialist").verdict

    assert verdict.delta > verdict.eps and verdict.lcb > 0.0        # condition 1 PASSED
    assert verdict.loo_min <= 0.0                                   # condition 2 refused it
    assert verdict.loo_dropped == "nl2repo"
    assert not verdict.challenger_wins


def test_the_duel_runs_over_exactly_the_benchmarks_5_1b_admits(tmp_path):
    """§5.1b: a benchmark whose guard fired is excluded from SCORING, so it is not a duel row.

    `score_arm` drops it from the published arm, so handing it to the duel anyway would price a
    crown on a benchmark whose λ is not a price — and `loo` and the median would run over a set the
    published arm does not have, which is §6.3b's comparability rule broken from the other side.
    """
    # A benchmark no model in the pool can do (strength tops out at 3), so it flags under the
    # King0-fitted table; `frontier-swe` at difficulty 3 now PRICES, since the cascade reaches
    # `mock/heavy`. See `test_invariant2.DEAD` for the same fixture and why.
    dead = MockBenchmark("frontier-swe", STRENGTH, n_tasks=4, difficulty=(4, 4))
    validator, tree = build(tmp_path, (dead,) + LIVE[1:])
    excluded = {name for name, rate in validator.pins.world.exchange.items() if rate.flags}
    assert excluded, "this world has no flagged benchmark, so it proves nothing"

    report = validator.run_window(
        1, [submission(validator, "hopeful", 5, router(validator), tree)])

    verdict = duelled(report, "hopeful").verdict
    assert set(verdict.benchmarks) == set(report.benchmarks) - excluded
    # The same set the published arm was scored over — one definition of "admitted", not two.
    assert set(verdict.benchmarks) == {row.benchmark for row in report.king.per_benchmark}


def test_a_benchmarks_grouping_reaches_the_duel_and_a_benchmark_without_one_supplies_none():
    """The wiring behind `duel`'s clustered resample: `TaskSpec.group` -> `BenchmarkPair.groups`.

    The label is a fact about the TASK, so it is read from the window's pinned slice rather than
    copied into every `EpisodeResult` — two copies of "which repository this row came from" is two
    things that can disagree about what the bootstrap resampled. This is the only place the two are
    joined, and it is joined by task ID rather than by position.

    A benchmark whose adapter supplies no group must land on None, which is every-task-its-own-
    cluster and bit-for-bit the resample that shipped (`test_duel.py`). An invented grouping
    would widen that benchmark's interval for a structure it does not have.
    """
    tasks = (TaskSpec("r2egym-sympy-aa", "r2egym", "issue", (), "sympy"),
             TaskSpec("r2egym-pandas-bb", "r2egym", "issue", (), "pandas"),
             TaskSpec("r2egym-sympy-cc", "r2egym", "issue", (), "sympy"),
             TaskSpec("lcb-1", "livecodebench", "problem", ()),
             TaskSpec("lcb-2", "livecodebench", "problem", ()))

    def arm(score):
        return tuple(EpisodeResult(task_id=t.task_id, benchmark=t.benchmark, steps=(),
                                   graded_score=score, spend_usd=0.01, stopped_reason="stop")
                     for t in tasks)

    pairs = simulate._pairs(Paired(king=arm(0.4), challenger=arm(0.5), excluded=()), {}, tasks)

    assert {p.benchmark: p.groups for p in pairs} == {
        "livecodebench": None, "r2egym": ("sympy", "pandas", "sympy")}


def test_the_reveal_names_the_median_when_it_is_the_condition_that_refused(tmp_path):
    """§5.2 condition 3, in the words the miner reads.

    The median refuses wins that leave-one-out lets through — Phase 1 measured the pair: a big win
    on two benchmarks of twelve and nothing on the other ten clears LOO at +0.0286 and has a median
    of exactly zero. That is the shape built here. If `_why` still knew only the first two
    conditions, this challenger would be refused with an EMPTY explanation.
    """
    twelve = tuple(MockBenchmark(f"bench-{i:02d}", STRENGTH, n_tasks=2, difficulty=(2, 2))
                   for i in range(12))
    validator, tree = build(tmp_path, twelve, per_benchmark=2, minimum=2)
    narrow = submission(validator, "two-subjects", 5,
                        router(validator, ("bench-00", "bench-01")), tree)

    report = validator.run_window(1, [narrow])

    outcome = duelled(report, "two-subjects")
    verdict = outcome.verdict
    assert verdict.delta > verdict.eps and verdict.lcb > 0.0    # condition 1 passed
    assert verdict.loo_min > 0.0                                # condition 2 passed
    assert verdict.median == pytest.approx(0.0)                 # condition 3 refused it
    assert not verdict.challenger_wins and report.crowned is None
    assert "median benchmark's delta +0.0000 is not positive" in outcome.detail
    # And the number itself is in the reveal, beside the two conditions that let it through.
    assert "median +0.0000" in format_window(report)


# --- §8b.3: worker failures are data, grader failures are not -------------------------------------


def test_a_grader_failure_drops_its_task_from_both_arms_and_from_the_denominator(tmp_path):
    """Docker dying is a fact about the validator, not about the miner (§8b.3, §6.3c).

    Scoring it zero would inject the owner's infrastructure noise into a miner's result, and an
    exclusion applied to one arm only would hand the other an easier slice.
    """
    class Sandbox:
        """Kills the first task it ever grades, in every arm — flaky infra, not a policy."""

        def __init__(self, grade):
            self.grade, self.dead = grade, None

        def __call__(self, task, answer):
            self.dead = self.dead or task.task_id
            if task.task_id == self.dead:
                raise RuntimeError("sandbox died")
            return self.grade(task, answer)

    sandbox: list[Sandbox] = []

    def flaky(grade):
        sandbox.append(Sandbox(grade))
        return sandbox[-1]

    validator, tree = build(tmp_path, grade=flaky)
    report = validator.run_window(
        1, [submission(validator, "hopeful", 5, router(validator), tree)])

    assert report.graders_failed == (sandbox[0].dead,)
    # The denominator is the ADMITTED slice, not the whole one: a benchmark whose §5.1b guard fired
    # never enters a score, so it is `report.n_tasks` minus the excluded benchmarks' share that the
    # dead task has to come out of. Measured against the slice's own shape (§6.3b: every admitted
    # benchmark appears, `per_benchmark` tasks each) rather than against a hard-coded number.
    admitted = {b for b, rate in validator.pins.world.exchange.items() if not rate.flags}
    assert any(sandbox[0].dead.startswith(f"{b}-") for b in admitted), \
        "the dead task is on an excluded benchmark, so it was never in a denominator to leave"
    full = len(admitted) * validator.per_benchmark
    # Out of the denominator for BOTH arms: one task fewer than a full admitted slice, in each.
    verdict = duelled(report, "hopeful").verdict
    assert verdict.n_tasks == full - 1
    assert sum(row.n_tasks for row in report.king.per_benchmark) == full - 1
    assert sum(row.n_tasks for row in duelled(report, "hopeful").arm.per_benchmark) == full - 1


def test_a_worker_failure_is_an_outcome_and_stays_in_the_score(tmp_path):
    """The other half of §8b.3: a model erroring is information about that rung.

    A Conductor that keeps delegating to a flaky model should be scored for it, so the step is
    recorded, the episode continues, and the task stays in the denominator.

    The model is killed AFTER the corpus is pinned, for the reason the `grade=flaky` fixture states:
    λ_b is pinned per corpus (§5.1b), and a fixed-policy sweep run against a dead model measures
    that model's outage rather than the pool's cost/quality slope — every benchmark would trip the
    DOMINATOR guard and the corpus would be refused before a window could open.
    """
    validator, tree = build(tmp_path)
    dead = MockWorker(answers=worker_answers(STRENGTH), costs=COST,
                      failures=frozenset({"mock/cheap"}))
    validator.pins = replace(validator.pins, world=replace(validator.pins.world, worker=dead))

    report = validator.run_window(
        1, [submission(validator, "hopeful", 5, router(validator), tree)])

    # King₀ — always-cheapest — delegates to the dead model on every one of its tasks.
    king_zero = next(arm for arm in report.reference if arm.name == validator.pins.king_zero.name)
    dead_rungs = [step for result in king_zero.results for step in result.steps
                  if step.model_id == "mock/cheap"]
    assert dead_rungs and not any(step.success for step in dead_rungs)
    assert all(step.cost_usd == 0.0 for step in dead_rungs)     # a provider that errored charged us
    # Recorded and moved on: the episode reached its own STOP rather than dying, and nothing was
    # excluded — a dead rung is data about that rung, so it belongs in the score.
    assert {result.stopped_reason for result in king_zero.results} == {"stop"}
    assert report.graders_failed == ()
    admitted = {b for b, rate in validator.pins.world.exchange.items() if not rate.flags}
    assert sum(row.n_tasks for row in report.king.per_benchmark) == \
        len(admitted) * validator.per_benchmark


# --- §5.5: the crown reverts when the king deregisters --------------------------------------------


def test_the_crown_reverts_to_king_zero_when_the_king_is_no_longer_registered(tmp_path):
    """§5.5: a king that has left the metagraph does not keep the throne — or defend it.

    From the reversion on it is King₀ that is duelled, on the owner's allowance (§8b.8), and the
    first challenger to clear the verdict against it takes the crown exactly as at cold start. A
    deposed king that kept the throne would instead defend it with a model nobody can pay, which is
    a live opponent conjured out of a deregistration.
    """
    validator, tree = build(tmp_path)
    deposed = Spy(router(validator))
    validator.king = submission(validator, "gone", 1, deposed, tree)
    validator.coronations.append("gone")
    validator.chain.deregister("gone")

    report = validator.run_window(
        1, [submission(validator, "hopeful", 5, router(validator), tree)])

    assert report.king_hotkey == KING0
    assert deposed.calls == 0                     # the deposed king did not defend
    # The throne was vacant, so the challenger duelled King₀ and won it, as at cold start (§8b.8).
    # Against the deposed king — the same router — it would have tied and nothing would have moved.
    assert report.crowned == "hopeful"


def test_a_deposed_king_holds_its_pension_rank_so_the_survivors_are_not_paid_one_rank_too_high(
        tmp_path):
    """§5.5 + §5.7: the payout bug a reversion closes, which the burn total cannot see.

    §5.7 excludes the REIGNING hotkey from its own lineage — one hotkey, one share. A king recorded
    as reigning after it has left the metagraph is therefore excluded from a lineage it belongs at
    the top of, and every surviving pensioner moves up one rank: predecessor #1 draws 0.05 where the
    reversion puts it at 0.04. The slate still sums to 1.0 and still burns 0.85 either way, so the
    error is invisible in the aggregate and wrong in every payment.
    """
    validator, tree = build(tmp_path)
    validator.chain.register("first")
    validator.king = submission(validator, "second", 1, validator.pins.king_zero, tree)
    validator.coronations.extend(["first", "second"])
    validator.chain.deregister("second")

    report = validator.run_window(1, [])

    # The deposed king takes the rank-1 slot it was dethroned into, and the rank behind it does not
    # move up: the pension is paid by position in the LINEAGE, not among those who resolved (§5.7).
    assert report.pensioners == ("second", "first")
    by_hotkey = {report.metagraph.get(uid, "burn"): share for uid, share in report.weights.items()}
    assert by_hotkey["first"] == pytest.approx(0.04)
    assert "second" not in by_hotkey                       # its 0.05 burns; it resolves to no uid
    assert by_hotkey["burn"] == pytest.approx(0.96)        # 0.85 + 0.05 + the three empty slots
    assert sum(report.weights.values()) == 1.0


# --- §5.7: emissions ------------------------------------------------------------------------------


def test_king_zeros_share_burns_and_is_never_reflowed_to_anyone(tmp_path):
    """§8b.8: until somebody beats the best fixed policy, 0.85 burns and nobody is paid.

    Folding it into the crown would hand the first winner 100% of emissions, which is exactly the
    outcome the pension exists to soften.
    """
    validator, tree = build(tmp_path)
    report = validator.run_window(
        1, [submission(validator, "copycat", 5, validator.pins.king_zero, tree)])

    assert validator.king_hotkey == KING0
    assert report.crowned is None
    assert report.weights == {0: pytest.approx(1.0)}
    assert sum(report.weights.values()) == 1.0


def test_the_dethroned_king_draws_the_first_pension_slot_and_the_rest_burns(simulation):
    """§5.7: 0.85 to the reigning king, 0.05 to its predecessor, the four unfilled slots burn."""
    last = simulation[-1]
    assert last.crowned == "router-c"
    assert last.pensioners == ("router-b",)
    by_hotkey = {last.metagraph.get(uid, "burn"): share for uid, share in last.weights.items()}
    assert by_hotkey["router-c"] == pytest.approx(0.85)
    assert by_hotkey["router-b"] == pytest.approx(0.05)
    assert by_hotkey["burn"] == pytest.approx(0.10)
    assert sum(last.weights.values()) == 1.0


def test_the_emission_slate_is_resolved_against_the_live_metagraph(simulation):
    """Risk register #22: Bittensor recycles UIDs, so the slate is rebuilt every window from
    `uid -> hotkey` rather than remembered. The reveal publishes what it was resolved against."""
    last = simulation[-1]
    assert last.metagraph                                   # published: uid -> hotkey, as resolved
    assert set(last.weights) <= set(last.metagraph)         # nothing paid to a uid nobody holds


# --- the price the duel uses ----------------------------------------------------------------------


def test_the_price_the_duel_pays_is_score_arms_own_and_not_a_second_copy_of_it(tmp_path):
    """`priced` must be `score_arm`, not a re-typing of §5.1b's formula.

    Two copies of the formula would price the duel while `score_arm` prices the published arm, and
    the day the rule changes the crown and the leaderboard would disagree with nothing able to see
    it. Asserted with `==`, not `approx`: the day this needs a tolerance is the day a
    reimplementation crept in.
    """
    validator, tree = build(tmp_path)
    report = validator.run_window(
        1, [submission(validator, "hopeful", 5, router(validator), tree)])

    final_b = priced(validator.pins.world.exchange)
    for row in report.king.per_benchmark:
        assert final_b(row.benchmark, row.quality, row.spend) == row.final
    assert score_arm(report.king_results, validator.pins.world.exchange).final == report.king.final


def test_a_benchmark_with_no_measured_exchange_rate_is_refused_rather_than_priced_at_zero(tmp_path):
    """§5.1b's forbidden silent fallback survives the adapter: it raises there as it does in
    `score_arm`, because it IS `score_arm`."""
    validator, _ = build(tmp_path)
    with pytest.raises(ValueError, match="no measured exchange rate"):
        priced(validator.pins.world.exchange)("a-benchmark-nobody-fitted", 1.0, 0.0)


# --- the reveal -----------------------------------------------------------------------------------


def test_the_report_publishes_the_per_benchmark_breakdown_and_the_emission_schedule(simulation):
    text = format_window(simulation[0])
    for benchmark in simulation[0].benchmarks:
        assert benchmark in text
    assert "POWER GATE — separates policies" in text
    assert "# KING ARM" in text and "# QUEUE" in text and "# EMISSIONS" in text
    assert "reigning king" in text and "BURN" in text
    assert "per-benchmark delta" in text
    # The narrowing and exclusion numbers a reader needs to compare two windows (§6.3b, §8b.3).
    assert "grader/sandbox failures excluded from BOTH arms: 0" in text


def test_the_reveal_says_which_condition_refused_a_challenger(simulation):
    """A challenger that loses on breadth and one that loses on the margin need different work."""
    assert "does not clear eps" in duelled(simulation[0], "copycat").detail
    assert "REFUSED: delta +0.0000 does not clear eps" in format_window(simulation[0])


# --- offline --------------------------------------------------------------------------------


def test_the_whole_simulation_runs_with_no_network_and_no_credentials(tmp_path, monkeypatch):
    """M9 exit 1: mock workers, mock chain, and nothing that could reach out for anything."""
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "R2_ACCESS_KEY_ID",
                 "AWS_ACCESS_KEY_ID", "BT_WALLET_NAME"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("the offline simulation opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    simulate.main(["--root", str(tmp_path)])


def test_the_offline_mechanism_withholds_the_crown_from_an_arm_that_bought_nothing(tmp_path):
    """§4's crown rule, mirrored here rather than left to the daemon.

    An arm that never reaches its slice scores 0 quality at ~0 spend, which `final` puts at exactly
    0.0 — above any king whose priced spend exceeds its quality. The daemon learned to withhold the
    crown from such an arm; this module did not, and this module IS the mechanism, so the rule
    existing in only one of them is the drift its own docstring exists to forbid. Measured before
    this: `crowned: penniless, weights {1: 0.85, 0: 0.15}`.
    """
    validator, tree = build(tmp_path)
    report = validator.run_window(
        1, [submission(validator, "penniless", 5, router(validator), tree, budget_usd=0.0)])

    row = duelled(report, "penniless")
    assert report.crowned is None, "the offline mechanism crowned an arm that bought nothing"
    assert "crown is withheld" in row.detail
