"""HLE — the non-code adapter, and what a corpus of code benchmarks cannot check (§6.1, M2a).

WHAT THESE TESTS PROTECT. Three of them are about HLE and the rest are about the *protocol*: this is
the only benchmark in the corpus with no program to run, so it is the only place where "the sandbox
is not compulsory", "a `Grade` total need not be a test count" and "an adapter need declare no tools"
can be stated as tests rather than assumed. If a change to `base.py` ever makes a container or a test
count structural, these fail and nothing else does.

The other half is the grader. It is a REIMPLEMENTATION — HLE ships an LLM judge that cannot be used
here (a model call per graded task, charged to a miner through `final_b`, and non-deterministic under
a paired duel) — and a home-grown grader is exactly the thing §3's "the scaffold cancels" argument
does not cover. So every rule it applies has a test named for the mistake it prevents, and two of
them are about *not* scoring: a prose answer must not match a letter gold by accident, and a
multi-answer gold must never be admitted at all.

NO MODEL IS EVER CALLED AND NO CONTAINER IS EVER STARTED. Submissions are fixtures — a letter, a
sentence, an echoed instruction. The one test that needs the network is the census, and it SKIPS:
`cais/hle` is gated and the account behind `OWNER_HF_TOKEN` has not accepted its terms (measured
2026-09-01), so the count that would move `N` from 2 to 3 has not been taken.
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3.benchmarks import hle
from thirtyspokes.v3.benchmarks.base import Grade, grader
from thirtyspokes.v3.types import TaskSpec


def _row(id="q1", answer="B", answer_type=hle.HLE_ANSWER_TYPE, image="", question="Which one?"):
    """One dataset row, in the schema the pinned card declares (`HLE_COLUMNS`)."""
    return {"id": id, "question": question, "answer": answer, "answer_type": answer_type,
            "image": image}


def _cached():
    """The task set if the gate is open, else None, so the census test skips rather than fail."""
    try:
        return hle.HumanitysLastExam()._loaded()
    except hle.BenchmarkUnavailable:
        return None


HLE_ROWS = _cached()
needs_hle = pytest.mark.skipif(
    HLE_ROWS is None,
    reason=f"cais/hle is gated and the terms are not accepted for this token. {hle.HLE_GATE_NOTE}")


# --- the grader, which is ours and therefore owes a test per rule ---------------------------------
def test_a_correct_option_scores_one_and_a_wrong_one_zero():
    """M2a exit 2 for a benchmark with no sandbox, no image and no subprocess."""
    assert hle.hle_grade("Explanation: because.\nAnswer: C", "C") == Grade(1, 1)
    assert hle.hle_grade("Explanation: because.\nAnswer: D", "C") == Grade(0, 1)


def test_the_option_letter_is_read_even_when_the_answer_repeats_the_option_text():
    """The shipped judge accepts `C) Paris`; a letter-only comparison would score the answer's
    FORMATTING, and since the Conductor chooses which worker answers, that noise would land straight
    on the routing decision the duel is measuring."""
    for answered in ("C", "(C)", "C.", "C) Paris", "(c). Paris", "c", "**C**", "C: Paris"):
        assert hle.hle_grade(f"Answer: {answered}", "C") == Grade(1, 1), answered


def test_prose_after_the_marker_scores_zero_rather_than_matching_a_letter_by_accident():
    """The leniency above has an edge, and `A` is the letter it opens on: an answer that BEGINS with
    a bare option letter is prose, not a choice. `A common misconception...` must not score the
    question whose gold is A — a grader that gave a free point for a sentence's first word would
    inflate one benchmark's quality and move a crown."""
    assert hle.hle_grade("Answer: A common misconception is that it is B.", "A") == Grade(0, 1)
    assert hle.hle_grade("Answer: none of these", "A") == Grade(0, 1)
    assert hle.hle_grade("", "A") == Grade(0, 1)


def test_the_answer_after_the_last_marker_is_read_and_the_line_below_it_is_not():
    """A model that echoes the format instruction before answering would otherwise be graded on the
    instruction; the confidence line HLE's own format asks for sits BELOW the answer, so the line
    read is the first after the last marker rather than the last line of the reply."""
    assert hle.hle_grade("Answer: {the letter of your chosen option}\nAnswer: B", "B") == Grade(1, 1)
    assert hle.hle_grade("Answer: B\nConfidence: 90%", "B") == Grade(1, 1)


def test_an_unformatted_reply_is_read_from_its_last_line():
    """With no marker at all a reply reasons first and answers at the end, so reading its FIRST line
    would grade the reasoning. Both ends are needed and they are not the same end."""
    assert hle.hle_grade("Both look plausible.\nB", "B") == Grade(1, 1)
    assert hle.hle_grade("B", "B") == Grade(1, 1)


def test_a_binary_grader_publishes_its_own_coarseness_rather_than_hiding_it():
    """`base.Grade` carries the granularity the benchmark was capable of precisely so a benchmark
    that can only report a bit is VISIBLE as one: §5.4's ~225-tasks-per-arm table is computed for
    binary scoring, and a corpus cannot be sized without knowing which of its benchmarks are."""
    assert hle.hle_grade("Answer: A", "A").total == 1
    assert hle.HLE_PROVENANCE.granularity.startswith("1/1")


# --- admission: the rows this seam can honestly score ---------------------------------------------
def test_only_the_mechanically_gradeable_rows_are_admitted():
    """Each filter drops rows this seam cannot score, not rows it finds inconvenient: an
    `exactMatch` row needs the LLM judge back, an image-bearing row is a question the text-only
    scaffold never fully asks, and a prose gold is free text to compare."""
    rows = [_row(id="ok"),
            _row(id="free", answer_type="exactMatch"),
            _row(id="visual", image="https://example.invalid/x.png"),
            _row(id="prose", answer="the second one")]
    assert list(hle.hle_rows(rows)) == [f"{hle.HLE_NAME}-ok"]


def test_a_multi_answer_gold_is_not_admitted_even_though_the_reader_would_find_a_letter_in_it():
    """The two rules are deliberately different strengths, and this is why. `hle_option` would read
    `C, D` as `c`, so a submission naming only half the answer would score full marks; admission is a
    fullmatch, so the row never reaches the grader at all."""
    assert hle.hle_rows([_row(id="multi", answer="C, D")]) == {}
    assert hle.hle_option("C, D") == "c"


def test_every_admitted_gold_is_one_the_grader_can_decide():
    """The property tying the filter to the grader: submitting an admitted row's own gold must score
    1. A row that passed admission and then could not be graded would be an invisible constant added
    to one benchmark's quality."""
    rows = [_row(id="plain", answer="B"), _row(id="parens", answer="(c)"),
            _row(id="dotted", answer="D.")]
    for task_id, row in hle.hle_rows(rows).items():
        assert hle.hle_grade(row["answer"], row["answer"]) == Grade(1, 1), task_id


def test_the_task_id_is_the_dataset_s_own_id_so_a_refetch_names_the_same_task():
    """Slices are drawn by ID and the reveal publishes them (§6.2); an ID that renumbers between two
    loads makes two windows incomparable and the published traces unmatchable to their tasks."""
    assert list(hle.hle_rows([_row(id="66f1a"), _row(id="66f1b")])) == ["hle-66f1a", "hle-66f1b"]


# --- what the non-code benchmark proves about the protocol itself ---------------------------------
def test_grading_needs_no_container_because_nothing_from_the_worker_is_executed():
    """The package checklist asks every adapter to grade "inside a `--network none` container", which
    is a code-shaped rule: here the submission is a letter and there is nothing to execute. Stated as
    a test because a future change that made a sandbox structural would otherwise break this
    benchmark for a reason that has nothing to do with it — `sandbox` is not even imported."""
    bench = hle.HumanitysLastExam()
    bench._rows = hle.hle_rows([_row(id="q1", answer="B")])
    task = bench.load()[0]
    assert not hasattr(hle, "sandbox")
    assert bench.grade("Answer: B", task) == Grade(1, 1)


def test_the_adapter_declares_no_tools_because_a_knowledge_question_has_no_action_space():
    """`scaffold._call_worker` makes ONE completion call and executes no tool, so any declaration
    would be a prompt that lies to the Conductor. On the code benchmarks that emptiness is a
    limitation being recorded; here it is simply the truth, which is the half of `tools()` a corpus
    of code benchmarks could not show."""
    assert hle.HumanitysLastExam().tools() == ()


def test_the_adapter_satisfies_the_benchmark_protocol_the_scaffold_is_built_from():
    """`grader(benchmark)` is the seam `Scaffold` is constructed with, so building one is the
    cheapest complete statement that this adapter is usable at all."""
    bench = hle.HumanitysLastExam()
    bench._rows = hle.hle_rows([_row(id="q1", answer="B")])
    task = TaskSpec(task_id="hle-q1", benchmark=hle.HLE_NAME, prompt="", tools=())
    assert grader(bench)(task, "Answer: B") == 1.0


def test_the_prompt_carries_the_question_verbatim_and_adds_only_the_answer_format():
    """§2.1: the benchmark's task statement reaches the worker byte for byte. The lines under it are
    owner text — identical for every miner and both arms, as LiveCodeBench's "output ONLY the program
    source" is — and asking for the LETTER is what makes a judge-free grader honest."""
    prompt = hle.hle_prompt(_row(question="Which one?\n\nA. this\nB. that"))
    assert prompt.startswith("Which one?\n\nA. this\nB. that\n\n")
    assert prompt.endswith("Answer: {the letter of your chosen option}")


# --- provenance and the gate ----------------------------------------------------------------------
def test_the_adapter_pins_a_commit_and_records_a_redistribution_it_does_not_have():
    """§6.1.1 and M2a exit 4. The card asks in terms that the dataset not be re-uploaded, so a window
    cannot be made self-contained from an R2 copy of it — "probably fine" is how a subnet acquires a
    takedown mid-window, and the field is an enumerated verdict rather than prose for that reason."""
    prov = hle.HumanitysLastExam().provenance
    assert len(prov.revision) == 40
    assert prov.redistribution == hle.REDISTRIBUTION_REFUSED
    assert prov.grader == hle.GRADER_REIMPLEMENTED


def test_a_load_failure_names_the_gate_and_the_action_that_opens_it():
    """The measured failure of this round: `dataset_info` succeeds on a gated repo without
    acceptance, so metadata reachability is not access. An error that only reported a 403 would send
    the next reader to look at the token rather than at the terms.

    The refusal is a sentinel rather than a real `GatedRepoError` so that the assertion cannot pass
    on a live 403: the message must carry BOTH the fetch's own failure and the gate note, which is
    only true if the patched call is the one that ran."""
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise RuntimeError("no-network-in-this-test")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(huggingface_hub, "hf_hub_download", refuse)
        with pytest.raises(hle.BenchmarkUnavailable, match="no-network-in-this-test") as refused:
            hle.hle_parquet()
    assert "must accept its terms" in str(refused.value)
    assert f"https://huggingface.co/datasets/{hle.HLE_REPO}" in str(refused.value)


def test_the_owner_s_token_is_passed_to_the_fetch_under_the_name_this_repo_keeps_it_under():
    """`hf_hub_download` knows `HF_TOKEN` and nothing about `OWNER_HF_TOKEN`, which is where this
    repository keeps the owner's credential (`koth/neuron.py`). A gated benchmark fetched without it
    401s *after* the terms are accepted, for a reason that looks exactly like the gate."""
    import huggingface_hub

    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here — the fetch is what is under test, not the file")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OWNER_HF_TOKEN", "not-a-real-token")
        patch.setattr(huggingface_hub, "hf_hub_download", capture)
        with pytest.raises(hle.BenchmarkUnavailable):
            hle.hle_parquet()
    assert seen["token"] == "not-a-real-token"
    assert seen["revision"] == hle.HLE_REVISION


@needs_hle
def test_the_recorded_census_is_the_count_that_actually_loads():
    """SKIPPED WHILE THE GATE IS CLOSED, AND THAT IS THE POINT. `usable_tasks` is None because the
    census could not be taken, and M3a exit 4 sizes the arena from that field — so the moment the
    owner accepts the terms this test starts failing and keeps failing until the number is counted
    and written down. A remembered number and a counted one are indistinguishable in that field, and
    only this test can tell them apart."""
    assert hle.HLE_PROVENANCE.usable_tasks == len(hle.HumanitysLastExam().load())
