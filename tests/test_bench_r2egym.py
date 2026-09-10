"""The R2E-Gym adapter: the census that moves N, and the grading path that makes it real.

WHAT THESE TESTS PROTECT THAT `test_real_benchmarks.py` DOES NOT. That module's three adapters are
either cheap to grade (LiveCodeBench) or effectively ungradeable on this box (SWE-bench Verified: 7 of
500 instances have their multi-gigabyte image). This one is the third case — 6812 tasks, each with a
~0.5 GB image and a 2-second grade — and its risks are its own:

  * **The census IS the deliverable** (§1.2 of the spend plan: at N = 2 leave-one-out collapses, the
    median is redundant, and a 125-task stratum is impossible over a 112-task benchmark). Three
    filters decide it; each has a test naming what it drops and why.
  * **The obvious partial credit is wrong here by a factor of fifty.** The median task ships 45
    already-passing tests against a median of one fail-to-pass, so the fraction of ALL tests hands a
    do-nothing patch ~0.98. The same trap `real.swe_grade` documents, deeper.
  * **The grader is reimplemented** (`r2egym` is not installable — it pins `swebench==3.0.2` against
    the 4.1.0 SWE-bench Verified grades with), so §3's "the scaffold cancels" argument is weaker here
    and the transcription has to be pinned against the shipped code's actual behaviour, including the
    parts of it that look like bugs.
  * **The tests live outside the repository**, which is the only thing stopping a patch from editing
    the tests that grade it. That is a security property and it is asserted by trying it.

NO MODEL IS EVER CALLED AND NOTHING IS PULLED. Submissions are fixtures: the dataset's own gold patch
reconstructed from `parsed_commit_content`, an empty string, a patch aimed at the test file. The one
image these tests can use is whichever is already provisioned, so a box with none skips.
"""

from __future__ import annotations

import difflib
import json

import pytest

from thirtyspokes.v3.benchmarks import r2egym, sandbox
from thirtyspokes.v3.benchmarks.base import Grade
from thirtyspokes.v3.tools import PROTOCOL
from thirtyspokes.v3.types import TOOL_NAMES, Environment, TaskSpec


def _cached():
    """The corpus if it is already fetchable, else None, so a test skips rather than hit the net."""
    try:
        return r2egym.R2EGym()._loaded()
    except r2egym.BenchmarkUnavailable:
        return None


ROWS = _cached()

# The task the Docker path is measured on: the first one whose image is already provisioned, so the
# suite grades whatever the operator happens to have rather than naming an instance nobody has. The
# measured example is `r2egym-sympy-2a1c9aa2…` — 13 tests, 3 fail-to-pass, 1.9 s unpatched.
PROVISIONED = next((task_id for task_id in (ROWS or ())
                    if sandbox.available() and sandbox.image_present(ROWS[task_id]["docker_image"])),
                   None)


def _corpus() -> r2egym.R2EGym:
    """The corpus loaded once, at import, rather than once per test.

    A full `load()` reads thirteen parquet files and costs ~6 seconds; eleven tests want it, and a
    suite that spends a minute re-decoding the same 6812 rows is a suite people stop running.
    """
    bench = r2egym.R2EGym()
    bench._rows = ROWS
    return bench


needs_docker = pytest.mark.skipif(
    not sandbox.available(),
    reason=f"grading needs a docker client AND a declared host ({sandbox.DOCKER_HOST_ENV})")
needs_corpus = pytest.mark.skipif(ROWS is None, reason="R2E-Gym V1 is not cached")
# The reason distinguishes the two facts that arrive here as one None: no image, or no declared host
# (`image_present` answers False without asking a daemon). Measured: images ARE provisioned on the
# owner's box and this still fired, so a fixed "none is provisioned" would be a false statement.
needs_image = pytest.mark.skipif(
    PROVISIONED is None,
    reason=("no R2E-Gym instance image is provisioned on this box" if sandbox.declared() else
            f"{sandbox.DOCKER_HOST_ENV} is unset, so no daemon was asked about the images"))

# A row shaped like the dataset's, so the filters and the grading arithmetic can be exercised on
# statuses whose verdict is known. `two_pass` failed before the fix and passes after; `one_pass` and
# `three_fail` are the rest of the module the commit did not touch. The keys are the dataset's own
# shape — pytest's `::` path already joined with dots, which is what the shipped parser produces.
EXPECTED = {"one_pass": "PASSED", "two_pass": "PASSED", "three_fail": "FAILED"}


def _summary(**statuses: str) -> str:
    """A pytest `-rA` tail carrying `statuses`, which is the only part of a log the parser reads."""
    lines = "\n".join(f"{status} r2e_tests/test_1.py::{name}" for name, status in statuses.items())
    return f"+ bash run_tests.sh\n=== {r2egym.R2E_SUMMARY_MARKER} ===\n{lines}\n=== 1 failed ===\n"


def _row(problem: str = "[ISSUE]\nit is broken\n[/ISSUE]", *, pre_fix: str | None = None,
         post_fix: str | None = None) -> dict:
    return {"repo_name": "sympy", "docker_image": "namanjain12/sympy_final:cafe1234",
            "problem_statement": problem,
            "expected_output_json": json.dumps(EXPECTED),
            "execution_result_content": json.dumps({
                "old_commit_res_stdout": pre_fix if pre_fix is not None else _summary(
                    one_pass="PASSED", two_pass="FAILED", three_fail="FAILED"),
                "new_commit_res_stdout": post_fix if post_fix is not None else _summary(
                    one_pass="PASSED", two_pass="PASSED", three_fail="FAILED")})}


def _fixture() -> tuple[r2egym.R2EGym, TaskSpec]:
    bench = r2egym.R2EGym()
    bench._rows = r2egym.r2e_rows([_row()])
    return bench, bench.load()[0]


# --- the census: what moves N, and what each filter costs (§1.2) -----------------------------------
@needs_corpus
def test_the_corpus_is_the_one_admitted_benchmark_larger_than_a_window_can_use():
    """THE POINT OF THE WHOLE ADAPTER. At N = 2 a ~250-task slice needs 125 per stratum and
    LiveCodeBench has 112 tasks in total, so the slice is every task every window and "drawn after
    commits" protects nothing. 6812 is the first census in this corpus where the draw is a draw."""
    bench = _corpus()
    tasks = bench.load()
    assert len(tasks) == r2egym.R2E_USABLE_TASKS == r2egym.R2E_PROVENANCE.usable_tasks == 6812
    assert all(task.benchmark == r2egym.R2E_NAME for task in tasks)


@needs_corpus
def test_a_row_with_no_problem_statement_is_dropped_because_there_is_nothing_to_ask():
    """1192 of the 8101 rows carry an empty `problem_statement`, and they are not scattered: they are
    every matplotlib row and every moto row, so this filter costs the corpus two of its thirteen
    repositories outright. Keeping them would ask a worker to fix an issue it was never shown and
    score every model zero for a reason that has nothing to do with routing."""
    assert r2egym.r2e_rows([_row(problem="")]) == {}
    assert r2egym.r2e_rows([_row(problem="   \n ")]) == {}
    repos = {row["repo_name"] for row in _corpus()._loaded().values()}
    assert "matplotlib" not in repos and "moto" not in repos and len(repos) == 11


def test_a_row_whose_own_post_fix_log_does_not_reproduce_its_expected_map_is_dropped():
    """The self-consistency filter, 97 rows. `fail_to_pass` is derived by running the shipped parser
    over the dataset's own logs, so a row where that parser cannot reproduce the dataset's own
    `expected_output_json` is a row whose granularity would be derived from a known disagreement."""
    disagreeing = _row(post_fix=_summary(one_pass="PASSED", two_pass="PASSED"))
    assert r2egym.r2e_rows([disagreeing]) == {}


def test_a_row_where_nothing_was_broken_before_the_fix_is_dropped_as_ungradeable():
    """A task whose fail-to-pass set is empty has no denominator: `Grade(0, 0)` is refused by the
    protocol, and scoring it out of the whole map would add a near-constant to this benchmark's
    quality. Zero rows in the corpus hit this — it is asserted because the filter is what makes that
    claim safe to rely on, not because it fires."""
    already_fixed = _row(pre_fix=_summary(one_pass="PASSED", two_pass="PASSED",
                                          three_fail="FAILED"))
    assert r2egym.r2e_rows([already_fixed]) == {}


@needs_corpus
def test_the_task_id_is_the_commit_so_it_survives_a_refetch():
    """§6.2 draws the slice by ID and the reveal publishes it, so an ID that renumbers between two
    loads makes two windows incomparable. The image's tag is the task's commit and is unique across
    all 8101 rows, which a row index is not."""
    rows = _corpus()._loaded()
    assert len(set(rows)) == len(rows)
    for task_id, row in rows.items():
        assert task_id.endswith(row["docker_image"].rsplit(":", 1)[-1])
        assert task_id.startswith(f"{r2egym.R2E_NAME}-{row['repo_name']}-")


@needs_corpus
def test_the_repository_label_survives_loading_because_the_draw_has_to_stratify():
    """34% of the usable pool is sympy and 21% is pandas, so a uniform draw is mostly two libraries.
    `repo_of` answers by ID, which is what the draw needs, the same arrangement
    `LiveCodeBench.difficulty_of` uses for the same reason.

    It is also on every loaded task as `TaskSpec.group`, which is what the DUEL needs: `duel`
    resamples whole repositories, because ~7.5 tasks of each in an 83-task stratum move together
    and an iid bootstrap prices them as independent evidence. The two must be the same label —
    a slice stratified by one grouping and resampled by another would be neither.
    """
    bench = _corpus()
    labels = {bench.repo_of(task.task_id) for task in bench.load()}
    assert "sympy" in labels and "pandas" in labels and len(labels) == 11
    assert all(task.group == bench.repo_of(task.task_id) for task in bench.load())


# --- granularity: what a duel on this benchmark actually costs (§5.4) ------------------------------
@needs_corpus
def test_the_denominator_is_the_fail_to_pass_set_and_it_is_binary_on_most_tasks():
    """THE HONEST GRANULARITY, PUBLISHED RATHER THAN IMPLIED (`base.Grade`). The 141-test status map
    makes this look like the richest benchmark in the corpus; the fail-to-pass set is median 1, mean
    1.88, and binary on 72.1% of tasks — barely better than SWE-bench Verified's 69%. §5.4's
    partial-credit multiplier does not arrive here, and a reader of `provenance.granularity` must not
    be told otherwise."""
    sizes = [len(row["fail_to_pass"]) for row in _corpus()._loaded().values()]
    binary = sum(1 for size in sizes if size == 1)
    assert min(sizes) >= 1
    assert sum(sizes) / len(sizes) == pytest.approx(1.88, abs=0.01)
    assert binary / len(sizes) == pytest.approx(0.721, abs=0.001)


def test_partial_credit_is_the_fail_to_pass_fraction_with_the_rest_of_the_map_as_a_gate():
    """M2a exit 3 for the 27.9% of tasks with more than one fail-to-pass test — the only place this
    adapter is finer than the shipped reward, which is all-or-nothing over the whole map."""
    expected = {"f_1": "PASSED", "f_2": "PASSED", "p_1": "PASSED", "p_2": "FAILED"}
    f2p = ("f_1", "f_2")
    observed = dict(expected)
    assert r2egym.r2e_grade(observed, expected, f2p) == Grade(2, 2)
    assert r2egym.r2e_grade(observed | {"f_2": "FAILED"}, expected, f2p) == Grade(1, 2)
    none_fixed = observed | {"f_1": "ERROR", "f_2": "FAILED"}
    assert r2egym.r2e_grade(none_fixed, expected, f2p) == Grade(0, 2)


def test_breaking_a_test_that_used_to_pass_scores_zero_however_much_else_was_fixed():
    """The rest of the map is a gate, not a summand. A patch that resolves the issue and breaks the
    library is not a partial solution to the task the benchmark asked."""
    expected = {"f_1": "PASSED", "p_1": "PASSED", "p_2": "PASSED"}
    broken = {"f_1": "PASSED", "p_1": "PASSED", "p_2": "FAILED"}
    assert r2egym.r2e_grade(broken, expected, ("f_1",)) == Grade(0, 1)


def test_a_test_that_was_expected_to_fail_and_now_passes_also_fails_the_gate():
    """Harsh, and deliberately the shipped reward's own harshness: `_calculate_reward_r2e` compares
    the maps for EQUALITY, so a submission that incidentally fixes an expected failure scores 0 there.
    11.6% of this corpus's statuses are FAILED or ERROR, so the alternative is not hypothetical — it
    is a v3 score of 1.0 on submissions the benchmark's own grader scores 0.0."""
    expected = {"f_1": "PASSED", "p_2": "FAILED"}
    assert r2egym.r2e_grade({"f_1": "PASSED", "p_2": "PASSED"}, expected, ("f_1",)) == Grade(0, 1)


def test_scoring_the_fraction_of_all_tests_would_hand_a_do_nothing_patch_nearly_full_marks():
    """WHY THE OBVIOUS PARTIAL CREDIT IS WRONG HERE, as arithmetic. The median usable task carries 45
    already-passing tests against a median of one fail-to-pass, so a submission that changes nothing
    matches 44 of 45. Every task would sit near 1.0, two policies would differ by a rounding error,
    and §6.1.3's admission probe would drop the benchmark for having no signal — for a reason entirely
    internal to how the score was defined."""
    expected = {f"p_{i}": "PASSED" for i in range(44)} | {"f_1": "PASSED"}
    unchanged = expected | {"f_1": "FAILED"}
    naive = sum(unchanged[k] == v for k, v in expected.items()) / len(expected)
    assert naive == pytest.approx(0.978, abs=1e-3)
    assert r2egym.r2e_grade(unchanged, expected, ("f_1",)).score == 0.0


def test_a_full_score_means_exactly_what_the_shipped_reward_means_by_one_point_zero():
    """§6.1.2 will not get an as-shipped grader here, so what it gets instead is a REFINEMENT whose
    top is the shipped top: `_calculate_reward_r2e` returns 1.0 iff the two maps have equal length and
    agree on every key, which is the key-set check plus the gate plus a full fail-to-pass set. A
    reimplementation that scored 1.0 anywhere the benchmark scores 0.0 would move this benchmark's
    level while still looking like a score."""
    expected = {"f_1": "PASSED", "p_1": "PASSED", "p_2": "ERROR"}
    for observed, shipped_reward in [({"f_1": "PASSED", "p_1": "PASSED", "p_2": "ERROR"}, 1.0),
                                     ({"f_1": "FAILED", "p_1": "PASSED", "p_2": "ERROR"}, 0.0),
                                     ({"f_1": "PASSED", "p_1": "PASSED"}, 0.0),
                                     ({"f_1": "PASSED", "p_1": "PASSED", "p_2": "ERROR",
                                       "extra": "PASSED"}, 0.0)]:
        full = r2egym.r2e_grade(observed, expected, ("f_1",)).score == 1.0
        assert full is (shipped_reward == 1.0)


# --- the transcribed parser: pinned against the shipped one's actual behaviour ---------------------
def test_only_the_summary_block_is_read_so_a_traceback_cannot_be_mistaken_for_a_verdict():
    """The shipped parser discards everything before `short test summary info`, and that is what makes
    it safe to run over a container log that also carries our `set -x` trace: a traceback quoting the
    word PASSED, or a shell line naming a test, is not a verdict."""
    log = ("+ git apply -v /w/patch.diff\nFAILED because PASSED appeared in a traceback\n"
           f"=== {r2egym.R2E_SUMMARY_MARKER} ===\nPASSED r2e_tests/t.py::only\n")
    assert r2egym.r2e_parse_log(log) == {"only": "PASSED"}


def test_a_collection_error_keeps_its_empty_name_because_dropping_it_would_hide_a_deleted_test():
    """`ERROR r2e_tests/test_1.py - ImportError` has no `::`, so the shipped parser stores it under
    the empty key and its length check then refuses the whole map. Normalising that away here would
    let a patch that stops a test being COLLECTED score as if the test had never existed."""
    log = (f"=== {r2egym.R2E_SUMMARY_MARKER} ===\nERROR r2e_tests/test_1.py - ImportError: no\n")
    assert r2egym.r2e_parse_log(log) == {"": "ERROR"}
    assert r2egym.r2e_grade({"": "ERROR"}, EXPECTED, ("two_pass",)) == Grade(0, 1)


def test_both_sides_of_the_comparison_are_normalised_by_the_same_function():
    """The shipped reward decolours and truncates at ` - ` on BOTH maps before comparing, because the
    dataset ships some expected keys with ANSI escapes baked in. Two normalisers would be two chances
    to drift, and the drift would look like every test failing."""
    log = (f"=== {r2egym.R2E_SUMMARY_MARKER} ===\n"
           "FAILED r2e_tests/t.py::\x1b[31mred\x1b[0m - AssertionError: boom\n")
    assert r2egym.r2e_parse_log(log) == {"red": "FAILED"}
    assert r2egym.r2e_key("\x1b[31mred\x1b[0m - AssertionError: boom") == "red"


def test_no_summary_block_at_all_is_the_graders_failure_and_not_a_zero():
    """§8b.3's line. Once the apply marker has been ruled out, a log with no summary means the
    container died before pytest wrote one — an OOM kill, a truncated run, a daemon restart — and such
    a task leaves BOTH arms rather than charging the miner for the validator's machine."""
    assert r2egym.r2e_parse_log("Killed\n") == {}
    assert r2egym.r2e_parse_log("") == {}


def test_the_fail_to_pass_set_is_derived_from_the_datasets_own_pre_fix_log():
    """R2E-Gym labels no fail-to-pass set — its reward is all-or-nothing — so the split has to come
    from somewhere. It comes from the log of the expected map being produced at the parent commit,
    which every row ships: a fact about the dataset rather than a judgement about the task."""
    pre_fix = _summary(one_pass="PASSED", two_pass="FAILED", three_fail="FAILED")
    expected = {"one_pass": "PASSED", "two_pass": "PASSED", "three_fail": "FAILED"}
    assert r2egym.r2e_fail_to_pass(pre_fix, expected) == ("two_pass",)


# --- the prompt: what actually reaches a worker (§2.1) ---------------------------------------------
def test_the_issue_body_is_forwarded_the_way_the_benchmarks_own_harness_forwards_it():
    """`DockerRuntime.get_task_instruction` verbatim, including its lack of a strip: this is the text
    §2.1 forwards byte for byte, so trimming it would make a v3 episode ask a different question from
    the one R2E-Gym's own agent is asked."""
    assert r2egym.r2e_issue("noise[ISSUE]\nbody\n[/ISSUE]noise") == "\nbody\n"
    assert r2egym.r2e_issue("no tags here") == "no tags here"


def test_the_prompt_names_the_repository_because_a_blind_worker_cannot_guess_a_file_path():
    """The scaffold makes one completion call with no tools, so the worker cannot list the tree it is
    patching. The repository name is owner text, identical for both arms and every miner — exactly as
    `real.swe_prompt`'s is — and without it a correct diff is unguessable for a reason that has
    nothing to do with routing. No COMMIT is named: the checkout is the parent of the tagged one."""
    prompt = r2egym.r2e_prompt(_row())
    assert prompt.startswith("Repository: sympy")
    assert "it is broken" in prompt and "[ISSUE]" not in prompt
    assert "unified git diff" in prompt


def test_nothing_the_conductor_produces_could_reach_the_prompt():
    """§2.1 as a property of this adapter: `load` builds every prompt from the row alone, so the only
    strings in it are the dataset's, this module's own template, and the pinned read protocol —
    which is inside `task.prompt` precisely so that `WorkerRequest.text == task.prompt` stays true on
    the first turn of every delegate and so the protocol enters the window's own task hash."""
    _, task = _fixture()
    assert task.prompt == r2egym.r2e_prompt(_row())
    assert task.tools == TOOL_NAMES
    assert task.prompt.endswith(PROTOCOL)


def test_the_declared_verbs_have_an_environment_and_it_is_rooted_at_the_repository():
    """THE OTHER HALF OF `test_no_adapter_declares_a_tool_the_scaffold_cannot_run`, which can only
    check the declares-nothing direction without loading a corpus. Here the fixture supplies a row,
    so the declaring direction is checkable: three verbs, and somewhere to run them.

    THE ROOT IS THE CONFINEMENT. `/r2e_tests` sits outside `/testbed`, which this module's docstring
    records as what makes the benchmark safe to grade; rooting the read channel at the repository is
    the same property applied to the other door, so a miner driving a worker's reads still cannot
    reach the file their patch will be graded against."""
    bench, task = _fixture()
    environment = bench.environment(task)

    assert bench.tools() == TOOL_NAMES
    assert environment == Environment(image=_row()["docker_image"], root=r2egym.R2E_REPO_PATH)
    assert not environment.root.startswith(r2egym.R2E_TESTS_PATH)
    assert not r2egym.R2E_TESTS_PATH.startswith(environment.root)


# --- the grading path, on a real image -------------------------------------------------------------
def test_the_runner_applies_the_patch_before_the_tests_are_reachable_at_all():
    """THE SECURITY PROPERTY, read off the script that carries it. `/r2e_tests` is outside `/testbed`,
    `git apply` refuses to leave the repository or write through a symlink, and the `rm -rf` removes
    any `r2e_tests/` the patch created inside the repository before the real one is linked in. This is
    what SWE-bench's harness needs a gold-test-patch re-apply for."""
    runner = r2egym._r2e_runner()
    apply_at = runner.index("git apply")
    assert apply_at < runner.index("rm -rf") < runner.index("ln -s") < runner.index("run_tests.sh")
    assert "patch --" not in runner          # `patch` accepts `../`; `git apply` does not


def test_an_empty_answer_scores_zero_without_starting_a_container(monkeypatch):
    """An empty submission is a genuine failure to solve the task, not a grading failure — and the
    fixture asserts it costs no container, because 6812 tasks times one wasted 0.5 GB image start is
    the difference between a window that fits in §8b.2's clock and one that does not."""
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: pytest.fail("started a container"))
    bench, task = _fixture()
    assert bench.grade("", task) == Grade(0, 1)
    assert bench.grade("   \n ", task) == Grade(0, 1)


@needs_corpus
def test_usable_ids_are_the_tasks_whose_image_is_already_provisioned(monkeypatch):
    """`--pull never` makes an absent image a refusal at grading time, which is the right failure —
    a 0.6 GB pull inside a graded episode spends the arm's wall clock on housekeeping (§8b.2) — but
    only a kind one if the window can be checked before it opens. This is the number that binds: ~115
    images fit in the 70 GB free against a corpus of 6812.

    ASKED IN ONE BATCH, WHICH ON THIS CORPUS IS THE DIFFERENCE BETWEEN A CENSUS AND A BILL: the
    per-image route pays two docker round trips per task, because `image_present` calls `available()`
    and `available()` probes the daemon whenever the seam names one — ~13,600 invocations for 6812
    tasks, 2000-4000 s over the deployed ssh transport. Detonating the single-name entry point is the
    check, because what regresses is the ROUTE and it regresses silently: the answers stay right."""
    bench = r2egym.R2EGym()
    bench._rows = {"a": {"docker_image": "img-a"}, "b": {"docker_image": "img-b"}}
    monkeypatch.setattr(sandbox, "image_present",
                        lambda image: pytest.fail("the census asked one image at a time"))
    monkeypatch.setattr(sandbox, "images_present", lambda names: {n for n in names if n == "img-a"})
    assert bench.usable_ids() == ("a",)


@needs_corpus
def test_the_image_a_task_grades_in_is_pinned_to_that_tasks_own_commit():
    """The one place this benchmark is strictly better provenanced than SWE-bench Verified, which
    grades in `…:latest` — a dataset pinned to a commit whose environment floats (`real.py` finding
    2). Here the tag IS the commit, one image per task, so the environment a score was computed in can
    be named exactly (§6.1.1)."""
    rows = _corpus()._loaded()
    images = [row["docker_image"] for row in rows.values()]
    assert len(set(images)) == len(images)
    assert not any(image.endswith(":latest") for image in images)


@needs_docker
@needs_image
def test_the_datasets_own_gold_patch_scores_full_marks_and_an_empty_one_scores_zero():
    """M2a exit 2 on a real image, the shipped runner and the shipped test command. The gold patch is
    reconstructed from the dataset's own before-and-after file contents, so a grader that scores it
    below 1.0 is broken in a way no fixture would reveal — and the empty submission is the other end,
    which fixes the scale rather than merely the top of it. Measured on `r2egym-sympy-2a1c9aa2…`:
    13 tests, 3 fail-to-pass, 0/3 unpatched and 3/3 patched, 2.4 s."""
    bench = _corpus()
    task = next(t for t in bench.load() if t.task_id == PROVISIONED)
    assert bench.grade(_gold_patch(PROVISIONED), task).score == 1.0
    assert bench.grade("", task).score == 0.0


@needs_docker
@needs_image
def test_a_patch_that_will_not_apply_is_the_miners_zero_and_not_an_exclusion():
    """The discrimination §8b.3 requires and the shipped code cannot make for us: a malformed patch
    and a container that died both produce no summary block, and they belong on opposite sides of the
    line. `R2E_APPLY_FAIL` is ours because `r2egym`'s agent loop never applies a patch and so ships no
    marker to import."""
    bench = _corpus()
    task = next(t for t in bench.load() if t.task_id == PROVISIONED)
    nonsense = ("diff --git a/does_not_exist.py b/does_not_exist.py\n"
                "--- a/does_not_exist.py\n+++ b/does_not_exist.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert bench.grade(nonsense, task).score == 0.0


@needs_docker
@needs_image
def test_a_patch_aimed_at_the_graded_tests_cannot_reach_them():
    """The attack the layout defends against, run rather than argued. A patch that rewrites
    `r2e_tests/` to assert nothing would score full marks on every task at once; here it either fails
    to apply or is deleted before the real tests are linked in, and the fail-to-pass tests still
    fail."""
    bench = _corpus()
    task = next(t for t in bench.load() if t.task_id == PROVISIONED)
    hijack = ("diff --git a/r2e_tests/test_1.py b/r2e_tests/test_1.py\n"
              "new file mode 100644\n--- /dev/null\n+++ b/r2e_tests/test_1.py\n"
              "@@ -0,0 +1,2 @@\n+def test_everything():\n+    assert True\n")
    assert bench.grade(hijack, task).score == 0.0


def _gold_patch(task_id: str) -> str:
    """The commit's non-test changes, rebuilt from `parsed_commit_content`'s before-and-after texts.

    Built here rather than shipped by the dataset, which stores the commit as structured hunks and not
    as a diff — and restricted to the non-test files because the test changes are already inside the
    image as `/r2e_tests`. Reading the column costs a shard, so this is called only by the two tests
    that grade for real, never at import.
    """
    import pyarrow.parquet as pq

    from thirtyspokes.v3.benchmarks.real import _hf_file

    commit = task_id.rsplit("-", 1)[-1]
    for filename in r2egym.R2E_FILES:
        path = _hf_file(r2egym.R2E_REPO, filename, r2egym.R2E_REVISION)
        table = pq.read_table(path, columns=["docker_image", "parsed_commit_content"])
        for row in table.to_pylist():
            if not row["docker_image"].endswith(commit):
                continue
            patch = ""
            for diff in json.loads(row["parsed_commit_content"])["file_diffs"]:
                target = diff["header"]["file"]["path"]
                if "test" in target.rsplit("/", 1)[-1] or "/tests/" in target:
                    continue
                patch += "".join(difflib.unified_diff(
                    diff["old_file_content"].splitlines(keepends=True),
                    diff["new_file_content"].splitlines(keepends=True),
                    fromfile=f"a/{target}", tofile=f"b/{target}"))
            return patch
    raise AssertionError(f"{task_id} is in the census but in none of the shards")
