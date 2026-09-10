"""The pins that only make sense in terms of ANOTHER pin (docs/WHITEPAPER.md §8b.1, §8b.2).

`config.py` is mostly independent numbers, and those need no test — a value nobody derived cannot be
derived wrongly. The three below are different: each is a RATIO or a CONTRAST against a constant
that lives beside it, so editing either half in isolation breaks a rule silently while every
individual value still looks reasonable. `test_action.py` already pins one of these
(`MAX_CONDUCTOR_TOKENS` against `REASON_TOKEN_CAP`); this file is the same idea for the queue caps
and the two wall clocks.
"""

from __future__ import annotations

from thirtyspokes.v3.config import (
    DUEL_WALL_CLOCK_REASON,
    DUEL_WALL_CLOCK_SECONDS,
    EPISODE_WALL_CLOCK_SECONDS,
    MAX_DUELS_PER_WINDOW,
    MAX_QUEUE_DEPTH,
)


def test_the_queue_drains_within_two_windows_so_immunity_can_cover_it():
    """§8b.1's deregistration trap is a DRAIN TIME problem, not a headcount one: a challenger who
    waits longer than Bittensor's immunity period is deregistered before it is ever evaluated,
    having paid a registration burn and uploaded ~70 GB for nothing.

    The cap is therefore expressed as windows-to-drain, and the owner's chain-side obligation is to
    set `immunity_period` to cover three windows — one to wait for the next window to open, two to
    drain. Raising `MAX_QUEUE_DEPTH` without raising `MAX_DUELS_PER_WINDOW` lengthens that wait past
    what the owner configured on chain, and nothing on chain would say so.
    """
    assert MAX_QUEUE_DEPTH % MAX_DUELS_PER_WINDOW == 0
    assert MAX_QUEUE_DEPTH // MAX_DUELS_PER_WINDOW == 2


def test_a_capped_queue_still_defers_rather_than_drops():
    """The cap only means "roll over" if the queue can hold more than one window's duels. At
    `MAX_QUEUE_DEPTH == MAX_DUELS_PER_WINDOW` no challenger ever rolls over, so §8b.1's "a queue is
    never dropped, only deferred" would be satisfied by a queue that never defers — and the refusal
    at the commit boundary would be doing all the work the rollover was written for."""
    assert MAX_QUEUE_DEPTH > MAX_DUELS_PER_WINDOW


def test_the_two_wall_clocks_are_told_apart_in_the_published_trace():
    """§8b.2 has two clocks with two different meanings: an episode that stalled is a fact about the
    challenger's model, a task the arm never reached is a fact about the validator's clock. §5.8
    requires a verdict decided by something other than routing to be visible in the reveal rather
    than buried, and one `stopped_reason` for both would bury exactly that distinction — a
    pathological model and an overrunning validator would render as the same row.

    The per-duel clock must also outlast the per-episode one, or the arm is abandoned before its
    first episode could time out and the episode clock is dead code."""
    assert DUEL_WALL_CLOCK_REASON != "wall_clock"
    assert DUEL_WALL_CLOCK_SECONDS > EPISODE_WALL_CLOCK_SECONDS


def test_the_price_probe_wait_is_bounded_by_the_clock_it_sits_inside():
    """Pinned from below only, it could be raised to 3,000 s — past the episode clock its own comment
    names as the outer bound, so a caller waiting for a first price would outlive the episode making
    the call it is waiting for."""
    from thirtyspokes.v3.config import EPISODE_WALL_CLOCK_SECONDS, PRICE_PROBE_WAIT_SECONDS

    assert 0 < PRICE_PROBE_WAIT_SECONDS < EPISODE_WALL_CLOCK_SECONDS
