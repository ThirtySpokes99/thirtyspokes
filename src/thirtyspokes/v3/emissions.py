"""Emissions — the crown, a five-deep pension, and everything that burns (§5.7, the build plan M8b).

Winner-take-all was v1's rule and it has a real cost: a miner who expects to be dethroned inside a
window has almost no reason to enter, and a dethroned king has every reason to grief the new one. So
v3 pays 0.85 to the reigning king and a decaying 0.05/0.04/0.03/0.02/0.01 tail to the five most
recently crowned ex-kings. It is a soft landing and not an income on purpose: at 0.05 a former king
earns about a seventeenth of the crown, so defending is worth vastly more than having once won.

WHY EVERY UNFILLED SLOT BURNS, AND IS NEVER REFLOWED TO THE KING. Early on there are no ex-kings.
Folding their 0.15 into the crown would hand the first winner 100% of emissions — exactly the
outcome the pension exists to soften — so this schedule is a subtraction and never a
redistribution: a slot with nobody in it, or with somebody who can no longer be resolved, pays
`burn_uid`. The same holds for King₀ (§5.5): while the best fixed policy holds the throne its 0.85
burns, and the existing pensioners are still paid, because they earned their tail against a live
opponent.

WHY THE SLATE IS REBUILT FROM THE LIVE METAGRAPH EVERY WINDOW (risk register #22). Bittensor
recycles UIDs. If ex-king `hk_A` held UID 5 and deregistered, UID 5 is handed to an unrelated
`hk_B` — and a stored `{5: 0.04}` slate would then pay a stranger a pension they never earned,
silently, every window, forever. Nothing here remembers a UID: the lineage stores HOTKEYS, and the
metagraph is taken in the direction that verifies (`uid -> hotkey`, as `chain.hotkeys()` returns
it). A hotkey that no longer holds its old UID simply is not in the inverted map, and its slot
burns.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .config import EMISSION_KING, EMISSION_PENSION

KING0 = ""
"""The best fixed policy holds the crown at genesis and whenever a king vacates it (§5.5).

Not a registrable hotkey: the empty string resolves to no UID, which is what makes "King₀'s share
burns" fall out of the same resolution step that burns a deregistered pensioner's.
"""


@dataclass(frozen=True)
class Lineage:
    """Every coronation there has ever been, OLDEST FIRST — the pension's only input.

    Derived from the validator's append-only history (the hotkeys of the entries that took the
    crown, in the order they took it), so the pension holds no state of its own that could drift out
    of step with the record third parties audit.

    It deliberately does NOT record who reigns now. The crown reverts to King₀ when a king
    deregisters (§5.5), and a reversion is not a coronation — a lineage that had to answer "who is
    on the throne" would have to carry a record of something that never happened. The reigning
    hotkey is passed in beside it.
    """

    coronations: tuple[str, ...] = ()

    @classmethod
    def from_coronations(cls, hotkeys: Iterable[str]) -> Lineage:
        """Oldest first — the order an append-only history yields, made explicit and immutable."""
        return cls(tuple(hotkeys))

    def pensioners(self, king_hotkey: str) -> tuple[str, ...]:
        """The ex-kings drawing a tail, most recent coronation first, at most five.

        Three rules, and the order they apply in is load-bearing:

        * The REIGNING king is dropped first. One hotkey, one share — a king also holding the top
          pension would earn 0.90 rather than 0.85, out of the tail its own predecessors are owed.
        * A hotkey crowned more than once holds ONE slot, at its most recent position. Otherwise an
          operator alternating coronations between hotkeys it controls occupies several slots at
          once, and the pension pays for turnover rather than for having been dethroned.
        * Deduplication happens BEFORE the cut to five, so a repeat coronation never spends a slot
          twice; what drops off the end is the sixth most recent DISTINCT hotkey.

        Dethroning therefore shifts everyone down one position and drops the sixth off the end, with
        no bookkeeping of its own: the new king prepends, and the list is recomputed from the
        history.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for hotkey in reversed(self.coronations):
            if hotkey == king_hotkey or hotkey in seen:
                continue
            seen.add(hotkey)
            ordered.append(hotkey)
            if len(ordered) == len(EMISSION_PENSION):
                break
        return tuple(ordered)


def emission_weights(
    lineage: Lineage,
    king_hotkey: str,
    hotkey_of: Mapping[int, str],
    *,
    burn_uid: int = 0,
) -> dict[int, float]:
    """The window's weight slate: `uid -> share`, summing to 1.0 with the burn included.

    `hotkey_of` is the live metagraph, `uid -> hotkey` (`chain.hotkeys()`), and it must be re-read
    each window rather than remembered — see the module docstring for the recycled-UID failure this
    argument's direction exists to prevent. It is inverted here, so resolution and verification are
    the same step: a pensioner appears in the slate only at a UID the chain currently says is its
    own.

    Pension shares are paid by POSITION IN THE LINEAGE, not by position among those who resolved: an
    unresolvable pensioner's share burns and the ones behind it do not move up. Shifting them up
    would pay a rank nobody was dethroned into.

    The burn share is always present, even at 0.0. What burns is the number the owner and every
    miner want to see, and a slate that expressed it by absence would report the same thing as a
    schedule that simply forgot a slot.
    """
    uid_of = {hotkey: uid for uid, hotkey in hotkey_of.items()}
    weights: dict[int, float] = {}

    king_uid = uid_of.get(king_hotkey) if king_hotkey != KING0 else None
    if king_uid is not None:
        weights[king_uid] = EMISSION_KING

    for share, hotkey in zip(EMISSION_PENSION, lineage.pensioners(king_hotkey)):
        pensioner_uid = uid_of.get(hotkey)
        if pensioner_uid is not None:
            weights[pensioner_uid] = share

    # The correctly-rounded complement of what was paid, rather than `1.0 - sum(paid)`. None of
    # 0.85/0.05/... is representable in binary, and the residual of an accumulated sum leaves the
    # slate off 1.0 by an ulp in states that really occur (a mid-lineage pensioner deregisters,
    # leaving 0.05 and 0.02 paid, is one) — which is emission quietly created or destroyed by
    # rounding. This form is exact in every state, and the exactness is pinned by test.
    weights[burn_uid] = weights.get(burn_uid, 0.0) + math.fsum(
        [1.0, *(-share for share in weights.values())])
    return weights
