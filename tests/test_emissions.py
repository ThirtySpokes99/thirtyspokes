"""The emission schedule's burns (docs/WHITEPAPER.md §5.7, the build plan M8b exits 1-7).

Almost every test here pins something that BURNS, because every way this schedule can be distorted
is a way a share reaches somebody who did not earn it. Two of them are the ones to keep:

  * A recycled UID pays a stranger. Bittensor reassigns the UID of a deregistered hotkey, so a slate
    that stored `{uid: share}` would pay whoever inherited an ex-king's UID a pension they never
    earned — silently, every window, forever (risk register #22). The test recycles UID 5 and
    asserts nobody is paid at it.
  * With no ex-kings the king takes 0.85 and 0.15 BURNS, not 1.00. Reflowing an empty pension into
    the crown would hand the first winner every token, which is the exact outcome the pension exists
    to soften.

The metagraph is taken as `uid -> hotkey` (what `chain.hotkeys()` returns), and the fixtures always
register a hotkey at the burn UID, so "the burn address is an ordinary registered neuron" is never
an untested assumption.
"""

from __future__ import annotations

import math

import pytest

from thirtyspokes.v3.config import EMISSION_KING, EMISSION_PENSION
from thirtyspokes.v3.emissions import KING0, Lineage, emission_weights

BURN_UID = 0


def metagraph(*hotkeys: str) -> dict[int, str]:
    """`uid -> hotkey`, live, with the burn address at UID 0 and miners numbered from 1."""
    return {BURN_UID: "hk_burn_address", **dict(enumerate(hotkeys, start=1))}


def uid_of(mg: dict[int, str], hotkey: str) -> int:
    return next(uid for uid, hk in mg.items() if hk == hotkey)


# --- exit 1: the slate always accounts for the whole emission ------------------------------------

@pytest.mark.parametrize("n_past", [0, 1, 3, 5, 6, 9])
@pytest.mark.parametrize("king", ["hk_king", KING0])
def test_the_slate_sums_to_exactly_one_in_every_state(n_past: int, king: str):
    """0, 3 and 6+ past kings, under a real king and under King₀ — nothing created, nothing lost."""
    past = tuple(f"hk_{i}" for i in range(n_past))
    mg = metagraph(*past, "hk_king")

    weights = emission_weights(Lineage.from_coronations(past), king, mg)

    assert sum(weights.values()) == 1.0


def test_an_unfilled_mid_lineage_slot_still_leaves_the_slate_summing_to_one():
    """The state that breaks a `1.0 - sum(paid)` residual, so it is pinned rather than rediscovered.

    King₀ reigns and only the 1st and 4th pensioners are still registered, so the slate pays exactly
    0.05 and 0.02 — a pair whose accumulated sum is a ulp off its own complement. An off-by-a-ulp
    slate is emission quietly created or destroyed by rounding, every window.
    """
    lineage = Lineage.from_coronations(("hk_e", "hk_d", "hk_c", "hk_b", "hk_a"))
    mg = metagraph("hk_a", "hk_d")          # hk_b, hk_c and hk_e have deregistered

    weights = emission_weights(lineage, KING0, mg)

    assert weights[uid_of(mg, "hk_a")] == EMISSION_PENSION[0]
    assert weights[uid_of(mg, "hk_d")] == EMISSION_PENSION[3]
    assert sum(weights.values()) == 1.0


def test_a_slate_nobody_can_be_resolved_for_burns_everything():
    weights = emission_weights(Lineage.from_coronations(("hk_a",)), "hk_king", metagraph())

    assert weights == {BURN_UID: 1.0}


# --- exit 1b: pay by hotkey, resolved to a UID at weight-setting time ----------------------------

def test_a_recycled_uid_pays_the_stranger_nothing_and_burns_the_slot():
    """Risk register #22, and the reason nothing in this module ever remembers a UID.

    `hk_a` is crowned, dethroned, then deregisters, and the chain hands its UID 5 to an unrelated
    `hk_stranger`. Both halves are asserted: while `hk_a` holds UID 5 the pension IS paid there, so
    a schedule that simply paid nobody could not pass this test either.
    """
    lineage = Lineage.from_coronations(("hk_old", "hk_a", "hk_king"))
    while_registered = {BURN_UID: "hk_burn_address", 5: "hk_a", 6: "hk_old", 7: "hk_king"}
    after_recycling = {BURN_UID: "hk_burn_address", 5: "hk_stranger", 6: "hk_old", 7: "hk_king"}

    before = emission_weights(lineage, "hk_king", while_registered)
    assert before[5] == EMISSION_PENSION[0]

    after = emission_weights(lineage, "hk_king", after_recycling)
    assert 5 not in after                              # the stranger inherited a UID, not a pension
    assert after[6] == EMISSION_PENSION[1]             # and hk_old did not move up into the slot
    assert after[BURN_UID] == pytest.approx(1.0 - EMISSION_KING - EMISSION_PENSION[1])
    assert sum(after.values()) == 1.0


# --- exit 2: unfilled slots burn -----------------------------------------------------------------

def test_with_no_ex_kings_the_king_takes_085_and_015_burns():
    """Not 1.00. The first winner must not be paid the tail its predecessors would have drawn."""
    mg = metagraph("hk_king")

    weights = emission_weights(Lineage(), "hk_king", mg)

    assert weights[uid_of(mg, "hk_king")] == EMISSION_KING
    assert weights[BURN_UID] == pytest.approx(sum(EMISSION_PENSION))
    assert sum(weights.values()) == 1.0


# --- exit 3: a deregistered ex-king's share burns -------------------------------------------------

def test_a_deregistered_ex_kings_share_burns_without_shifting_the_others_up():
    """Shares are paid by position in the LINEAGE, not by position among those who resolved.

    Promoting the survivors would pay them a rank nobody was dethroned into, and would make
    deregistering a pensioner profitable for everyone behind it.
    """
    lineage = Lineage.from_coronations(("hk_a", "hk_b", "hk_c", "hk_d", "hk_e", "hk_f"))
    mg = metagraph("hk_a", "hk_b", "hk_c", "hk_d", "hk_f", "hk_king")   # hk_e has deregistered

    weights = emission_weights(lineage, "hk_king", mg)

    assert weights[uid_of(mg, "hk_f")] == EMISSION_PENSION[0]
    assert uid_of(mg, "hk_a") not in weights                # still the 6th, still off the end
    assert weights[uid_of(mg, "hk_d")] == EMISSION_PENSION[2]   # not promoted into hk_e's 0.04
    assert weights[BURN_UID] == pytest.approx(EMISSION_PENSION[1])
    assert sum(weights.values()) == 1.0


# --- exit 4: one hotkey, one slot -----------------------------------------------------------------

def test_a_hotkey_crowned_twice_holds_one_slot_at_its_most_recent_position():
    """Otherwise an operator alternating coronations between its own hotkeys occupies several."""
    lineage = Lineage.from_coronations(("hk_a", "hk_b", "hk_c", "hk_a"))
    mg = metagraph("hk_a", "hk_b", "hk_c", "hk_king")

    weights = emission_weights(lineage, "hk_king", mg)

    assert lineage.pensioners("hk_king") == ("hk_a", "hk_c", "hk_b")
    assert weights[uid_of(mg, "hk_a")] == EMISSION_PENSION[0]   # its most recent position, once
    assert weights[uid_of(mg, "hk_c")] == EMISSION_PENSION[1]   # hk_a took no second, later slot
    assert weights[uid_of(mg, "hk_b")] == EMISSION_PENSION[2]
    assert sum(weights.values()) == 1.0


def test_a_repeat_coronation_does_not_spend_a_slot_twice():
    """Dedup before the cut to five: seven coronations by six hotkeys still fill all five slots."""
    lineage = Lineage.from_coronations(
        ("hk_a", "hk_b", "hk_c", "hk_b", "hk_d", "hk_e", "hk_f"))

    assert lineage.pensioners("hk_king") == ("hk_f", "hk_e", "hk_d", "hk_b", "hk_c")


# --- exit 5: the king does not also draw a pension ------------------------------------------------

def test_the_reigning_king_never_also_draws_a_pension():
    """A king that kept its old tail would earn 0.90 out of what its predecessors are owed."""
    lineage = Lineage.from_coronations(("hk_king", "hk_b", "hk_king"))
    mg = metagraph("hk_b", "hk_king")

    weights = emission_weights(lineage, "hk_king", mg)

    assert weights[uid_of(mg, "hk_king")] == EMISSION_KING
    assert weights[uid_of(mg, "hk_b")] == EMISSION_PENSION[0]
    assert weights[BURN_UID] == pytest.approx(sum(EMISSION_PENSION[1:]))
    assert sum(weights.values()) == 1.0


# --- exit 6: King₀ -------------------------------------------------------------------------------

def test_king0_reigning_burns_the_crown_while_pensioners_are_still_paid():
    """Nobody has ever won, or the crown reverted (§5.5). The pensioners earned theirs against a
    live opponent, so an empty throne must not stop their tail."""
    lineage = Lineage.from_coronations(("hk_a", "hk_b", "hk_c"))
    mg = metagraph("hk_a", "hk_b", "hk_c")

    weights = emission_weights(lineage, KING0, mg)

    assert weights[uid_of(mg, "hk_c")] == EMISSION_PENSION[0]
    assert weights[uid_of(mg, "hk_b")] == EMISSION_PENSION[1]
    assert weights[uid_of(mg, "hk_a")] == EMISSION_PENSION[2]
    assert weights[BURN_UID] == pytest.approx(EMISSION_KING + sum(EMISSION_PENSION[3:]))
    assert sum(weights.values()) == 1.0


def test_king0_is_never_paid_even_if_a_hotkey_is_registered_at_the_burn_uid():
    """King₀ is the best fixed policy, not a neuron — its 0.85 has nowhere to go but the burn."""
    weights = emission_weights(Lineage(), KING0, metagraph("hk_a"))

    assert weights == {BURN_UID: 1.0}


# --- exit 7: dethroning shifts the lineage by one -------------------------------------------------

def test_dethroning_shifts_the_lineage_by_one_and_drops_the_sixth_oldest():
    six = ("hk_1", "hk_2", "hk_3", "hk_4", "hk_5", "hk_6")
    mg = metagraph(*six, "hk_7")
    before = emission_weights(Lineage.from_coronations(six), "hk_6", mg)

    after = emission_weights(Lineage.from_coronations((*six, "hk_7")), "hk_7", mg)

    assert before[uid_of(mg, "hk_6")] == EMISSION_KING
    assert after[uid_of(mg, "hk_7")] == EMISSION_KING
    assert after[uid_of(mg, "hk_6")] == EMISSION_PENSION[0]   # dethroned: it takes the top tail
    for position, hotkey in enumerate(("hk_5", "hk_4", "hk_3", "hk_2")):
        assert before[uid_of(mg, hotkey)] == EMISSION_PENSION[position]
        assert after[uid_of(mg, hotkey)] == EMISSION_PENSION[position + 1]   # shifted down one
    assert uid_of(mg, "hk_1") not in after                  # the sixth oldest fell off the end
    # Every slot filled, so nothing burns — up to the ~1e-17 that 0.85/0.05/... cannot represent in
    # binary. That residue is the pinned decimals, not a share the schedule failed to pay.
    assert after[BURN_UID] == pytest.approx(0.0, abs=1e-15)
    assert sum(after.values()) == 1.0


# --- the lineage type ----------------------------------------------------------------------------

def test_the_lineage_is_derivable_from_an_append_only_coronation_history():
    """It is the crowned hotkeys in the order they were crowned, oldest first — nothing else.

    The pension keeps no state of its own, so it cannot drift from the history third parties audit.
    """
    history = [("hk_a", "won"), ("hk_b", "lost"), ("hk_c", "won")]

    lineage = Lineage.from_coronations(hk for hk, verdict in history if verdict == "won")

    assert lineage.coronations == ("hk_a", "hk_c")
    assert lineage.pensioners("hk_c") == ("hk_a",)          # most recent first, king excluded


def test_the_pension_is_five_deep_and_the_ladder_is_the_pinned_one():
    """A sixth pensioner would be paid nothing, and the ladder is what miners were promised."""
    assert EMISSION_PENSION == (0.05, 0.04, 0.03, 0.02, 0.01)
    assert EMISSION_KING == 0.85
    assert math.fsum((EMISSION_KING, *EMISSION_PENSION)) == 1.0

    lineage = Lineage.from_coronations(tuple(f"hk_{i}" for i in range(9)))
    assert len(lineage.pensioners("hk_king")) == len(EMISSION_PENSION)


# --- the owner's opt-out: King₀ paid at a UID instead of burned -----------------------------------

def test_king_zero_uid_pays_the_crown_at_that_uid_instead_of_burning_it():
    """§5.5's burn is the DEFAULT, not the only option. An owner who wants the genesis king visible
    as a reigning king names the UID it reigns at, and the 0.85 is paid there."""
    lineage = Lineage.from_coronations(("hk_a", "hk_b"))
    mg = metagraph("hk_a", "hk_b")

    weights = emission_weights(lineage, KING0, mg, king_zero_uid=uid_of(mg, "hk_a"))

    # The crown lands on the named UID, on top of the pension that UID had already earned.
    assert weights[uid_of(mg, "hk_a")] == pytest.approx(EMISSION_KING + EMISSION_PENSION[1])
    assert weights[uid_of(mg, "hk_b")] == EMISSION_PENSION[0]
    assert sum(weights.values()) == 1.0


def test_king_zero_uid_equal_to_the_burn_uid_leaves_the_slate_bit_identical():
    """The case the owner of netuid 99 actually runs. The burn UID was already collecting King₀'s
    share, so re-attributing it there must not move a single number on chain — what changes is the
    reveal's claim about who reigns, and nothing else."""
    lineage = Lineage.from_coronations(("hk_a",))
    mg = metagraph("hk_a")

    burned = emission_weights(lineage, KING0, mg)
    attributed = emission_weights(lineage, KING0, mg, king_zero_uid=BURN_UID)

    assert attributed == burned
    assert sum(attributed.values()) == 1.0


def test_king_zero_uid_left_unset_still_burns():
    """The default is unchanged: absent the flag, §5.5 stands exactly as before."""
    assert emission_weights(Lineage(), KING0, metagraph("hk_a")) == {BURN_UID: 1.0}
