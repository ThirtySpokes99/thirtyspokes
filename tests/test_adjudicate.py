"""`simulate.adjudicate` — the one function that turns a window's arms into verdicts and a crown.

The daemon and the offline mechanism both call it, so these are the rules that place a crown,
tested on arms small enough to read:

* a task an arm never reached before its PER-DUEL clock ran out leaves both arms of that duel
  (§8b.2), rather than scoring zero for the arm that ran out;
* an arm that answered under half its slice may still not take the crown (§4, `MIN_SLICE_REACHED`);
* several winners are crowned in COMMIT ORDER, each later one only by clearing the incumbent on the
  identical slice (§5.2, §8 step 4).

λ is zero in every exchange below, so `final` is plain quality and every margin can be read straight
off the scores.
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3.config import DUEL_WALL_CLOCK_REASON, EPS
from thirtyspokes.v3.score import Exchange
from thirtyspokes.v3.simulate import _king_too_thin, adjudicate
from thirtyspokes.v3.types import EpisodeResult, TaskSpec

BENCHMARKS = ("alpha", "beta")
PER_BENCHMARK = 20
TASKS = tuple(TaskSpec(f"{b}-{i}", b, "solve it", ())
              for b in BENCHMARKS for i in range(PER_BENCHMARK))
EXCHANGE = {b: Exchange(b, 0.0, 1.0) for b in BENCHMARKS}


def arm(score: float, *, cut_from: int | None = None, reason: str = DUEL_WALL_CLOCK_REASON,
        only: str | None = None) -> tuple[EpisodeResult, ...]:
    """Every task at `score`, except the tail from position `cut_from` in each benchmark (or in
    `only`), which `reason` stopped instead at 0 quality and $0."""
    rows = []
    for task in TASKS:
        position = int(task.task_id.rsplit("-", 1)[1])
        cut = (cut_from is not None and position >= cut_from
               and (only is None or task.benchmark == only))
        rows.append(EpisodeResult(task.task_id, task.benchmark, (), 0.0 if cut else score,
                                  0.0 if cut else 0.01, reason if cut else "stop"))
    return tuple(rows)


def judge(king, *challengers):
    return adjudicate(king, challengers, failed=(), exchange=EXCHANGE, tasks=TASKS, nonce="window")


def by(adjudged, hotkey: str):
    return next(j for j in adjudged.judgements if j.hotkey == hotkey)


# --- commit-order succession -------------------------------------------------------------------------


def test_a_later_winner_inside_eps_of_an_earlier_clock_cut_winner_leaves_it_the_crown():
    """`early` committed first and ran out of clock on 30% of its slice: scored as zeros it would have
    lost to the king (0.49 < 0.50); judged on the tasks both arms answered it clears the king by 0.20.
    `late` committed after it and clears the king by MORE (0.22) — but over `early`'s answered tasks
    it is only 0.02 better, inside `eps`, so the crown stays with the earlier commit."""
    adjudged = judge(arm(0.50), ("early", 1, arm(0.70, cut_from=14)), ("late", 2, arm(0.72)))
    early, late = by(adjudged, "early"), by(adjudged, "late")

    assert early.clocked == 12 and early.eligible
    assert early.verdict.challenger_wins and early.verdict.n_tasks == 28
    assert early.verdict.delta == pytest.approx(0.20)
    assert "judged on the 28 tasks both arms answered" in early.detail
    assert late.verdict.challenger_wins and late.verdict.delta == pytest.approx(0.22)

    assert adjudged.succession.crowned.hotkey == "early"
    assert early.rematch is None
    assert late.rematch.incumbent == "early" and not late.rematch.clears
    assert late.rematch.verdict.delta == pytest.approx(0.02) and late.rematch.verdict.delta < EPS
    assert "early committed earlier and holds the crown ahead of it" in late.detail


def test_a_later_winner_that_clears_the_earlier_one_takes_the_crown_and_both_rows_say_so():
    adjudged = judge(arm(0.50), ("early", 1, arm(0.60)), ("late", 2, arm(0.80)))
    early, late = by(adjudged, "early"), by(adjudged, "late")

    assert adjudged.succession.crowned.hotkey == "late"
    assert late.rematch.clears and late.rematch.verdict.delta == pytest.approx(0.20)
    assert "and clears early, which committed earlier" in late.detail
    assert "held the crown until late, which committed later, cleared it" in early.detail


# --- the per-duel clock ------------------------------------------------------------------------------


def test_an_arm_that_answered_under_half_its_slice_is_scored_but_may_not_take_the_crown():
    """§4 and §8b.2 together. The tail the clock took leaves the comparison, so without a floor an arm
    that answered a handful of tasks and stalled would be judged on that handful. It is scored and
    published; it may not win."""
    adjudged = judge(arm(0.50), ("staller", 1, arm(0.90, cut_from=8)))
    staller = by(adjudged, "staller")

    assert staller.verdict.challenger_wins              # on the 16 tasks it answered, it beats the king
    assert staller.reached == 16 and staller.clocked == 24 and not staller.eligible
    assert adjudged.succession.crowned is None
    assert "crown is withheld" in staller.detail and "running out the clock" in staller.detail


def test_an_episode_that_stalled_to_its_own_clock_is_still_scored():
    """The per-EPISODE clock is a model stalling on a task it started — its doing, not the clock's —
    so those rows stay in the score as zeros, and `clocked` does not count them."""
    adjudged = judge(arm(0.50), ("slow", 1, arm(0.90, cut_from=14, reason="wall_clock")))
    slow = by(adjudged, "slow")

    assert slow.verdict.n_tasks == 40 and slow.clocked == 0 and slow.reached == 40
    assert slow.verdict.delta == pytest.approx(0.90 * 14 / 20 - 0.50)


def test_the_kings_own_clock_tail_leaves_every_duel_and_the_published_king_arm():
    king = arm(0.50, cut_from=16)
    adjudged = judge(king, ("hopeful", 1, arm(0.80)))
    hopeful = by(adjudged, "hopeful")

    assert _king_too_thin(king) is None                                   # 32 of 40 answered
    assert hopeful.verdict.n_tasks == 32 and hopeful.verdict.delta == pytest.approx(0.30)
    assert sum(row.n_tasks for row in adjudged.king.per_benchmark) == 32


def test_a_king_arm_the_clock_cut_below_half_is_too_thin_to_price_anything():
    reason = _king_too_thin(arm(0.50, cut_from=8))
    assert reason is not None and "answered 16 of 40" in reason and "no shot is spent" in reason
    assert _king_too_thin(arm(0.50, cut_from=10)) is None                # exactly half answered
    assert _king_too_thin(()) is None


def test_a_duel_the_clock_left_unpriceable_withholds_the_crown_rather_than_crashing_the_window():
    """Breadth needs two benchmarks. When the clock took all of one, the duel cannot be priced; that
    is a thin slice rather than a broken corpus, so the arm is published with no verdict and no
    crown."""
    adjudged = judge(arm(0.50), ("half", 1, arm(0.90, cut_from=0, only="beta")))
    half = by(adjudged, "half")

    assert half.verdict is None and half.arm is None
    assert "too little of the slice" in half.detail
    assert adjudged.succession.crowned is None


def test_a_corpus_that_cannot_carry_a_verdict_still_fails_loudly_when_the_clock_is_not_why():
    """Only a slice the CLOCK thinned may go unpriced. Any other refusal from `duel` is a
    misconfigured corpus, and must not be swallowed into a quiet non-verdict."""
    one = tuple(TaskSpec(f"alpha-{i}", "alpha", "solve it", ()) for i in range(PER_BENCHMARK))
    king = tuple(EpisodeResult(t.task_id, "alpha", (), 0.5, 0.01, "stop") for t in one)
    hopeful = tuple(EpisodeResult(t.task_id, "alpha", (), 0.9, 0.01, "stop") for t in one)

    with pytest.raises(ValueError, match="breadth"):
        adjudicate(king, [("hopeful", 1, hopeful)], failed=(),
                   exchange={"alpha": EXCHANGE["alpha"]}, tasks=one, nonce="window")
