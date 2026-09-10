"""GDPval's refusal, kept honest (§6.1 check 2, the build plan M2a).

THESE TESTS PROTECT A NEGATIVE RESULT, WHICH FAILS DIFFERENTLY FROM AN ADAPTER. An adapter breaks
loudly — a load raises, a grade comes back wrong. A refusal breaks silently, in two ways, and there
is one test here for each:

  * **Someone writes the adapter anyway.** `benchmarks/__init__.py` names a shipped grader as a
    pre-adapter veto, and a class quietly appearing in `gdpval.py` would satisfy `base.Benchmark`
    structurally with nothing to stop it. So the module is inspected for one.
  * **The evidence rots.** `openai/gdpval` can gain rows, and a verdict whose numbers were true in
    September is a verdict nobody can re-derive in March. `MEASURED` is therefore re-derived from
    the real parquet, and that test fails if the dataset moves — which is the correct outcome, not a
    flake: it means the refusal needs re-reading.

NO MODEL IS EVER CALLED, and no deliverable file is downloaded — only the 1.9 MB parquet. The whole
of GDPval's grading instrument is `rubric_json`, and every claim made about it is checked here.
"""

from __future__ import annotations

import inspect
import json

import pytest

from thirtyspokes.v3 import benchmarks
from thirtyspokes.v3.benchmarks import gdpval, real
from thirtyspokes.v3.benchmarks.base import Grade


def _cached():
    """The real rows if they are fetchable, else None, so a test skips rather than hit the net."""
    try:
        return gdpval.load_rows()
    except real.BenchmarkUnavailable:
        return None


ROWS = _cached()
needs_gdpval = pytest.mark.skipif(ROWS is None, reason="GDPval is not cached")


def _row(*, refs: list[str], deliverables: list[str], scores: list[int],
         checked: bool = False) -> dict:
    """A GDPval-shaped row this file owns, so the verdict's arithmetic is testable offline."""
    return {
        "reference_files": refs,
        "deliverable_files": deliverables,
        "rubric_json": json.dumps([
            {"score": score, "criterion": "prose a person reads", "rubric_item_id": str(n),
             "author_type": "human", "tags": ["true"],
             "required": True if checked else None,
             "read_only": None, "form_content": None}
            for n, score in enumerate(scores)]),
    }


def test_gdpval_supplies_no_benchmark_and_is_in_no_adapter_list():
    """The veto is that its grader is a judge; nothing here may quietly become an adapter anyway."""
    assert gdpval.ADMITTED is False
    assert gdpval.GRADEABLE_TASKS == 0
    assert gdpval.PROVENANCE.grader == gdpval.GRADER_BY_JUDGE
    assert gdpval.PROVENANCE.grader not in (real.GRADER_AS_SHIPPED, real.GRADER_REIMPLEMENTED)
    assert gdpval.PROVENANCE.usable_tasks == 0

    # `base.Benchmark` is a Protocol, so conformance is structural: a class with these three methods
    # IS a benchmark to every caller in the package, whether or not it says so.
    implementors = [name for name, obj in inspect.getmembers(gdpval, inspect.isclass)
                    if obj.__module__ == gdpval.__name__
                    and all(callable(getattr(obj, m, None)) for m in ("load", "tools", "grade"))]
    assert implementors == [], f"{implementors} would make a refused benchmark scoreable"
    assert all(cls.__module__ != gdpval.__name__ for cls in benchmarks.ADAPTERS)


def test_the_refusal_is_pinned_to_a_commit_like_every_admitted_benchmark():
    """A verdict about a moving dataset is not a verdict; `Provenance` enforces the same rule here."""
    assert gdpval.PROVENANCE.revision == gdpval.REVISION
    with pytest.raises(ValueError):
        real.Provenance(repo=gdpval.REPO, revision="main", licence="", redistribution="",
                        grader=gdpval.GRADER_BY_JUDGE, grader_source="", deviation="",
                        granularity="", usable_tasks=0)


def test_a_rubric_total_with_penalties_is_not_a_grade():
    """94 criteria carry negative scores, so even a perfect judge can produce what `Grade` refuses.

    This is the disqualification that survives every optimistic assumption: grant the model judge,
    grant it perfect agreement, and the instrument still is not `Grade(completed, total)` — it is a
    weighted sum that can go below zero, and `quality_b` is a mean of these.
    """
    rows = [_row(refs=[], deliverables=["a.xlsx"], scores=[2, 2, -85])]
    assert gdpval.census(rows).criteria_with_a_negative_score == 1
    with pytest.raises(ValueError):
        Grade(2 + 2 - 85, 4)


def test_the_generous_text_filter_still_separates_a_file_from_an_answer():
    """`TEXT_SUFFIXES` is wide on purpose, so "3 of 220 survive" is not an artefact of a narrow set.

    A completion returns a string. A task whose gold is an .xlsx cannot be answered by one, however
    good the string is, and a task with no deliverable at all has no artifact to compare against —
    both are counted apart from the textual case rather than folded into it.
    """
    rows = [_row(refs=[], deliverables=["notebook.ipynb"], scores=[1]),
            _row(refs=[], deliverables=["deck.pptx"], scores=[1]),
            _row(refs=["input.xlsx"], deliverables=["query.overpassql"], scores=[1]),
            _row(refs=[], deliverables=[], scores=[1])]
    counted = gdpval.census(rows)
    assert counted.with_a_file_deliverable == 3
    assert counted.with_a_binary_deliverable == 1
    assert counted.self_contained_and_textual == 1      # the .overpassql needs its attachment
    assert counted.needing_reference_files == 1


def test_a_criterion_that_carried_an_executable_check_would_be_counted():
    """The claim is that all 10,453 are prose; the counter must be able to say otherwise.

    A census that returns 0 because it can only return 0 is not evidence. `required` is the field
    that would hold a check, so a row carrying one has to move the number.
    """
    prose = [_row(refs=[], deliverables=["a.pdf"], scores=[1, 1])]
    assert gdpval.census(prose).criteria_with_a_machine_readable_check == 0
    assert gdpval.census(
        [_row(refs=[], deliverables=["a.pdf"], scores=[1, 1], checked=True)]
    ).criteria_with_a_machine_readable_check == 2


@needs_gdpval
def test_the_census_behind_the_refusal_still_holds_at_the_pinned_revision():
    """Re-derives every number the module docstring argues from, against the real parquet.

    A failure here is not a flake and must not be edited away: it means `openai/gdpval` is not what
    the refusal was written about, and the refusal has to be re-read.
    """
    assert gdpval.census(ROWS) == gdpval.MEASURED


@needs_gdpval
def test_no_gdpval_task_can_be_scored_without_a_judge():
    """The headline: 220 rows, 0 gradeable — "loads 220 rows" is not "moved N" (spend plan §1.2).

    Both halves are checked at once because either alone would be misleading: every task's grading
    instrument is prose, and the tasks whose *shape* the scaffold could handle number three.
    """
    criteria = [item for row in ROWS for item in json.loads(row["rubric_json"])]
    assert criteria, "the rubric is the only grading instrument GDPval ships"
    assert all(item.get("required") is None and item.get("read_only") is None
               and item.get("form_content") is None for item in criteria)
    assert all(isinstance(item["criterion"], str) and item["criterion"].strip() for item in criteria)
    assert gdpval.GRADEABLE_TASKS == 0 < gdpval.MEASURED.self_contained_and_textual < len(ROWS)
