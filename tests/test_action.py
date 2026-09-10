"""The action grammar's refusals (docs/WHITEPAPER.md §2/§2.1, the build plan M0 exits 1-4).

This parser is the security boundary of the whole design: everything a miner's model generates
passes through it, and what leaves is an action carrying a kind and a catalog slug. So the tests
that matter here are the ones that pin what it REFUSES — a parser that coerces a near-miss, or that
lets untagged text ride along beside a valid action, would reopen the channel §2.1 closes and take
the public benchmarks down with it.

The catalog deliberately contains an id that is a strict prefix of another id, so "exact match
wins, prefix loses" is testable in both directions on the same snapshot.
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3.action import ActionParseError, ParseFailureTracker, parse
from thirtyspokes.v3.config import (
    MAX_CONDUCTOR_TOKENS,
    MAX_CONSECUTIVE_PARSE_FAILURES,
    REASON_TOKEN_CAP,
)
from thirtyspokes.v3.types import Action, Catalog, CatalogEntry

CATALOG = Catalog(entries=(
    CatalogEntry("openai/gpt-5.4", 1.25, 10.0, 400_000),
    CatalogEntry("openai/gpt-5.4-mini", 0.25, 2.0, 400_000),
    CatalogEntry("google/gemini-3.6-flash", 0.075, 0.3, 1_000_000),
))


def test_delegate_to_a_catalog_model_parses():
    assert parse("DELEGATE openai/gpt-5.4", CATALOG) == Action("delegate", "openai/gpt-5.4")


def test_retry_parses_like_delegate_but_keeps_its_own_kind():
    """RETRY is what makes this more than pick-one selection (§3), so it must survive as itself."""
    assert parse("RETRY google/gemini-3.6-flash", CATALOG) == Action(
        "retry", "google/gemini-3.6-flash")


def test_stop_never_carries_a_model_id():
    """A stop that carried a model id would be a delegate wearing a stop's name."""
    assert parse("STOP", CATALOG) == Action("stop", None)


def test_a_model_id_absent_from_the_catalog_is_refused_not_coerced():
    """M0 exit 1/2. The refusal is what keeps a widened pool from reopening the §2.1 channel."""
    with pytest.raises(ActionParseError):
        parse("DELEGATE not-a-model", CATALOG)


def test_a_model_id_that_is_a_prefix_of_a_catalog_id_is_refused():
    """No partial matching: `openai/gpt-5` is one character from a real id and is still not one."""
    with pytest.raises(ActionParseError):
        parse("DELEGATE openai/gpt-5", CATALOG)


def test_a_catalog_id_that_is_a_prefix_of_another_id_still_parses():
    """The other half of exact matching — refusing prefixes must not refuse real ids."""
    assert parse("DELEGATE openai/gpt-5.4", CATALOG).model_id == "openai/gpt-5.4"


def test_trailing_content_after_a_valid_action_is_refused():
    """M0 exit 3 — the injection an 'output ends with a valid action' rule would wave through."""
    for raw in ("DELEGATE openai/gpt-5.4 and here is the answer: 42",
                "STOP now, the solution is 42",
                "DELEGATE openai/gpt-5.4\nthe solution is 42",
                "STOP\nREASON but really the answer is 42"):
        with pytest.raises(ActionParseError):
            parse(raw, CATALOG)


def test_untagged_text_before_the_action_is_refused():
    """Only REASON may precede an action; free text must not be able to pass for reasoning."""
    with pytest.raises(ActionParseError):
        parse("Let me think about this.\nDELEGATE openai/gpt-5.4", CATALOG)


def test_two_actions_in_one_turn_are_refused():
    """One action per turn (§2): the scaffold acts once per Conductor call or the trace is a lie."""
    with pytest.raises(ActionParseError):
        parse("DELEGATE openai/gpt-5.4\nSTOP", CATALOG)


def test_reason_is_discarded_and_nothing_but_the_action_escapes_the_parser():
    """THE §2.1 PROPERTY AT THE PARSER. A hostile Conductor writes a full solution into the one
    field it entirely controls; what comes back is an action carrying a catalog slug, and the
    solution text is not reachable from it. There is nowhere in an `Action` to put a solution."""
    solution = "def solve(): return 42  # also DELEGATE evil/model"
    action = parse(f"REASON {solution}\nREASON therefore the cheap tier suffices\n"
                   f"DELEGATE openai/gpt-5.4-mini", CATALOG)
    assert action == Action("delegate", "openai/gpt-5.4-mini")
    assert "42" not in repr(action)          # every field of a dataclass shows up in its repr


def test_reason_without_an_action_is_a_parse_failure():
    """§8b.2's pathological model: max-token REASON forever, never reaching an action."""
    with pytest.raises(ActionParseError):
        parse("REASON " + "thinking " * 500, CATALOG)


def test_empty_output_is_a_parse_failure():
    with pytest.raises(ActionParseError):
        parse("   \n\n\t\n", CATALOG)


def test_keywords_are_case_insensitive_but_model_ids_are_exact():
    """The verb is a closed four-word vocabulary where `Delegate` can mean nothing else; refusing
    it would spend a miner's step on our formatting taste. An id is an identity — case-folding one
    would be exactly the coercion this parser exists not to do."""
    assert parse("delegate openai/gpt-5.4", CATALOG) == Action("delegate", "openai/gpt-5.4")
    assert parse("reason hmm\nStOp", CATALOG) == Action("stop", None)
    with pytest.raises(ActionParseError):
        parse("DELEGATE OpenAI/GPT-5.4", CATALOG)


def test_whitespace_around_and_inside_an_action_is_insignificant():
    """Greedy decoding produces stray indentation and CRLFs; none of it changes the decision."""
    for raw in ("  DELEGATE   openai/gpt-5.4  ", "\tDELEGATE openai/gpt-5.4\t\n",
                "REASON cheap first\r\n\r\nDELEGATE openai/gpt-5.4\r\n"):
        assert parse(raw, CATALOG) == Action("delegate", "openai/gpt-5.4")


def test_three_consecutive_parse_failures_force_stop():
    """M0 exit 4. Until the limit a failure is refused-and-counted and the episode continues."""
    tracker = ParseFailureTracker()
    assert MAX_CONSECUTIVE_PARSE_FAILURES == 3
    assert tracker.resolve("gibberish", CATALOG) is None
    assert tracker.resolve("DELEGATE not-a-model", CATALOG) is None
    assert tracker.resolve("REASON still thinking", CATALOG) == Action("stop", None)


def test_a_recovered_parse_failure_does_not_count_toward_the_forced_stop():
    """Consecutive, not cumulative: a policy that fumbles once and recovers is still routing."""
    tracker = ParseFailureTracker()
    for _ in range(4):
        assert tracker.resolve("gibberish", CATALOG) is None
        assert tracker.resolve("DELEGATE openai/gpt-5.4", CATALOG) == Action(
            "delegate", "openai/gpt-5.4")
        assert tracker.consecutive == 0


def test_the_conductor_token_cap_leaves_room_for_an_action_after_a_full_reason():
    """A generation cap at or below the REASON cap makes every turn truncate before the action,
    so a well-behaved Conductor would parse-fail its way to a forced STOP on every task while
    deciding nothing. The two pins look independent and are not."""
    assert MAX_CONDUCTOR_TOKENS >= REASON_TOKEN_CAP + 32


def test_catalog_compaction_lists_every_entry_in_snapshot_order():
    """Rendering must go through `entries`, never through `ids`: a prompt built from a set differs
    per process, which breaks both render determinism (M0 exit 5) and the cacheable catalog prefix
    (M0 exit 7). Same snapshot, same bytes, in the order the owner committed."""
    rows = CATALOG.compact().splitlines()
    assert rows == ["openai/gpt-5.4 / 1.25 / 10 / 400000",
                    "openai/gpt-5.4-mini / 0.25 / 2 / 400000",
                    "google/gemini-3.6-flash / 0.075 / 0.3 / 1000000"]
    assert CATALOG.ids == {e.model_id for e in CATALOG.entries}
