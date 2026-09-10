"""The action grammar — what a Conductor may say, and what the scaffold will act on (§2).

    REASON    <free text>     zero or more leading lines; internal only, DISCARDED here
    DELEGATE  <model_id>      the scaffold sends the ORIGINAL task to this model
    RETRY     <model_id>      the same, after observing an outcome
    STOP                      submit the best result so far

Exactly one action per turn, and it is the last thing in the output.

WHY THE PARSER IS THE SECURITY BOUNDARY. §2.1: *the only things that reach a worker model are the
pinned task text and a validated model ID.* Everything the Conductor generates passes through here
and leaves as an `Action`, which has room for a kind and a catalog slug and nothing else — so there
is no channel through which a miner can smuggle a memorised solution into the answer path, and no
need for a detector to look for one. That is the property that permits public benchmarks to be used
at all, and it is what makes `koth/verify.py::grounding_check` (removed with v2, 2026-09-07) unnecessary rather than merely
stronger. Widening the pool to all ~300 OpenRouter models does not reopen the channel, because an
unrecognised ID is refused rather than forwarded.

REFUSED, NEVER COERCED. A near-miss model ID is not repaired, fuzzy-matched or prefix-matched to
the catalog: a miner gets one submission ever, and a parser that guesses would silently score a
policy other than the one submitted. A parse failure is cheap and legible — the step is refused,
counted, and the episode continues (§2) — while a wrong guess is invisible. Same reason a REASON
block is discarded rather than summarised: whatever is in it, nothing derived from it reaches a
worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_CONSECUTIVE_PARSE_FAILURES
from .types import Action, Catalog

_REASON = "REASON"
_WITH_MODEL = {"DELEGATE": "delegate", "RETRY": "retry"}
_STOP = Action(kind="stop", model_id=None)


class ActionParseError(Exception):
    """A turn that is not exactly one valid action. Counted by the scaffold, never coerced."""


def parse(raw: str, catalog: Catalog) -> Action:
    """One Conductor turn -> the action the scaffold will run, or `ActionParseError`.

    The shape enforced is: every non-blank line before the last must be a `REASON` line, and the
    last non-blank line must be the action. That single rule delivers three things at once —
    reasoning is discarded, free text that is not tagged as reasoning cannot pass for it, and
    nothing may follow a valid action (so trailing content is refused rather than ignored, which is
    the injection an "ends with a valid action" rule would wave through).

    Keywords are matched case-insensitively; MODEL IDs are matched exactly. The asymmetry is
    deliberate. The verb comes from a closed four-word vocabulary where `Delegate` can mean nothing
    else, and refusing it would spend a miner's step on our formatting taste. A model ID is an
    identity: case-folding one is the coercion this parser exists not to do.
    """
    lines = [stripped for line in raw.splitlines() if (stripped := line.strip())]
    if not lines:
        raise ActionParseError("empty output: no action")

    *reasoning, last = lines
    for line in reasoning:
        if line.split(maxsplit=1)[0].upper() != _REASON:
            raise ActionParseError(f"untagged text where only REASON may appear: {line!r}")
    return _action(last, catalog)


def _action(line: str, catalog: Catalog) -> Action:
    parts = line.split()
    verb = parts[0].upper()

    if verb == "STOP":
        if len(parts) != 1:
            raise ActionParseError(f"trailing content after STOP: {line!r}")
        return _STOP

    kind = _WITH_MODEL.get(verb)
    if kind is None:
        raise ActionParseError(f"not an action: {line!r}")
    if len(parts) != 2:
        raise ActionParseError(f"{verb} takes exactly one model id: {line!r}")

    model_id = parts[1]
    if model_id not in catalog.ids:
        raise ActionParseError(f"{model_id!r} is not in this window's catalog snapshot")
    return Action(kind=kind, model_id=model_id)


@dataclass
class ParseFailureTracker:
    """Counts CONSECUTIVE parse failures and forces STOP at the limit (§2, §8b.2).

    A model that cannot emit a parseable action is not routing, and §8b.2's pathological case — one
    that emits max-token REASON at every step and never reaches an action — would otherwise burn the
    whole step budget on every task while deciding nothing, draining the king's allowance for no
    information. Consecutive rather than cumulative: a policy that fumbles one turn and recovers is
    still routing, and charging it for an old mistake would refuse a real challenger.

    `resolve` returns None for a failure the episode should carry on past — the step is refused and
    counted, and the scaffold re-prompts. `consecutive > 0` after the call is exactly "this turn
    failed", which is what `StepRecord.parse_failed` records.
    """

    limit: int = MAX_CONSECUTIVE_PARSE_FAILURES
    consecutive: int = 0

    def resolve(self, raw: str, catalog: Catalog) -> Action | None:
        try:
            action = parse(raw, catalog)
        except ActionParseError:
            self.consecutive += 1
            return _STOP if self.consecutive >= self.limit else None
        self.consecutive = 0
        return action
