"""SWE-bench Pro, the adapter whose whole point is a task COUNT (the spend plan §1.2, §6.1).

WHAT THESE TESTS PROTECT. The corpus is N = 2 and the mechanism breaks there — leave-one-out
collapses to "win each benchmark separately", the §5.2 median is arithmetically the aggregate, and a
125-task stratum does not exist. Only gradeable tasks move N, so the property that matters most here
is not that the code runs but that a task the adapter offers can actually be scored:

  * **The dataset's own gold patch resolves.** Run over every provisioned instance it is the only
    check that separates "loads 731 rows" from "can grade 23 tasks", which is the distinction this
    round exists to make. Measured today: 23 of 23 gold patches score 1.0, in 2-18 s each.
  * **A failing submission is scored and a broken grader is not.** §8b.3, and it is sharper here than
    on the sibling adapter because the shipped parser reports "no tests" for both a container that
    died and a run that collected nothing.
  * **The partial credit is the fail_to_pass fraction with pass_to_pass as a gate.** The published
    metric is one bit; §5.4's task counts are computed for binary, and Pro's median instance carries
    three fail_to_pass tests, so this is where the corpus can most cheaply buy resolution.

NO MODEL IS EVER CALLED. Every submission is a fixture — the dataset's own gold patch, a diff against
a file that does not exist, an empty string.

TESTS THAT NEED SOMETHING SKIP CLEANLY. The dataset, the shipped grader scripts (GitHub, cached on
first fetch) and Docker are each checked once at import, and the instance-image tests are guarded on
one image being provisioned rather than on Docker alone.
"""

from __future__ import annotations

import json

import pytest

from thirtyspokes.v3.benchmarks import real, sandbox
from thirtyspokes.v3.benchmarks import swebench_pro as swep
from thirtyspokes.v3.benchmarks.base import Grade, grader

# The instance every Docker test runs on: the smallest image in the corpus (0.45 GB compressed,
# 1.47 GB unpacked) and the fastest measured (2 s for the gold patch). It also carries SIX
# fail_to_pass tests, so the partial credit here is a real fraction and not a disguised bit.
INSTANCE = ("instance_ansible__ansible-9a21e247786ebd294dafafca1105fcd770ff46c6"
            "-v67cdaa49f89b34e42b69d5b7830b3c3ad3d8803f")
TASK_ID = f"{swep.SWEP_NAME}-{INSTANCE}"


def _cached(load):
    """A resource if it is already fetchable, else None, so a test skips rather than hit the net."""
    try:
        return load()
    except real.BenchmarkUnavailable:
        return None


ROWS = _cached(lambda: swep.SweBenchPro()._loaded())
SCRIPTS = _cached(lambda: swep._script(f"run_scripts/{INSTANCE}/parser.py"))

needs_docker = pytest.mark.skipif(
    not sandbox.available(),
    reason=f"grading needs a docker client AND a declared host ({sandbox.DOCKER_HOST_ENV})")
needs_rows = pytest.mark.skipif(ROWS is None, reason="SWE-bench Pro is not cached")
needs_scripts = pytest.mark.skipif(
    SCRIPTS is None, reason="the shipped per-instance grader scripts are not cached")
# The reason names which of the two facts is true: an absent image, or an undeclared host, where
# `image_present` returns False without asking any daemon. Measured on the owner's box — the image is
# provisioned (`usable_ids()` is 23 once declared, 0 before) and this skip still fired, so a fixed
# "not provisioned" reason would make the suite state something false about the machine it ran on.
needs_image = pytest.mark.skipif(
    ROWS is None or not sandbox.image_present(
        f"{swep.SWEP_REGISTRY}:{ROWS[TASK_ID]['dockerhub_tag']}"),
    reason=(f"the {INSTANCE} instance image is not provisioned" if sandbox.declared() else
            f"{sandbox.DOCKER_HOST_ENV} is unset, so no daemon was asked about the image"))

# A Pro-shaped row this file owns, so the pure functions can be exercised without the dataset. Every
# field is the shape the parquet stores it in, including the two that are Python reprs rather than
# JSON, because that encoding is the thing `swep_list` exists for.
FIXTURE = {
    "instance_id": "instance_fixture__fixture-abc-vdef",
    "repo": "fixture/fixture",
    "repo_language": "python",
    "base_commit": "b" * 40,
    "problem_statement": "The widget does not spin.",
    "requirements": "spin() must return True.",
    "interface": "Type: Method\nName: Widget.spin",
    "patch": "",
    "fail_to_pass": "['t.py::test_spin', \"t.py::test_doesn't_wobble\"]",
    "pass_to_pass": "['t.py::test_exists']",
    "selected_test_files_to_run": '["t.py"]',
    "before_repo_set_cmd": f"git reset --hard {'b' * 40}\ngit clean -fd\ngit checkout f00 -- t.py",
    "dockerhub_tag": "fixture.fixture-fixture__fixture-abc-vdef",
}


def fixture_bench():
    bench = swep.SweBenchPro()
    bench._rows = swep.swep_rows([FIXTURE])
    return bench, bench.load()[0]


def report(*named_statuses: tuple[str, str]) -> str:
    """A container log carrying the shipped parser's own output shape, after the verdict marker."""
    tests = [{"name": name, "status": status} for name, status in named_statuses]
    return f"noise from the test run\n{swep.SWEP_VERDICT}\n{json.dumps({'tests': tests})}\n"


# --- the count that moves N ------------------------------------------------------------------------
@needs_docker
@needs_scripts
@needs_image
def test_the_datasets_own_gold_patch_resolves_and_an_empty_one_does_not():
    """THE TEST THIS ADAPTER EXISTS TO PASS (M2a exit 2). A gold patch that does not score 1.0 means
    the adapter offers tasks it cannot grade, which is the failure mode of the sibling SWE-bench
    Verified adapter at scale — 500 rows, 7 gradeable — and the reason this round was commissioned.
    Run across every provisioned instance today it is 23 of 23."""
    bench = swep.SweBenchPro()
    task = next(t for t in bench.load() if t.task_id == TASK_ID)
    assert bench.grade(bench._loaded()[TASK_ID]["patch"], task).score == 1.0
    assert bench.grade("", task).score == 0.0


@needs_docker
@needs_scripts
@needs_image
def test_a_patch_that_will_not_apply_is_the_miners_zero_and_not_an_exclusion():
    """§8b.3, at the place a validator crosses it most often. The shipped entry script runs
    `git apply -v` with no fallback and does not stop when it fails, so the tests run against
    unpatched code and the shipped parser still writes a report — which is a SCORE of zero, not the
    grader failing. An adapter that raised here would drop the task from both arms and quietly
    forgive every malformed patch a miner produced."""
    bench = swep.SweBenchPro()
    task = next(t for t in bench.load() if t.task_id == TASK_ID)
    nonsense = ("diff --git a/does_not_exist.py b/does_not_exist.py\n"
                "--- a/does_not_exist.py\n+++ b/does_not_exist.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert bench.grade(nonsense, task).score == 0.0


@needs_rows
def test_the_pinned_census_loads_with_stable_ids_that_survive_a_refetch():
    """M2a exit 1. Stable across two loads because the window's task order comes from the nonce alone
    and the reveal publishes the IDs (§6.2): an ID that renumbers makes two windows incomparable and
    the published traces unmatchable to the tasks they came from."""
    first, second = swep.SweBenchPro().load(), swep.SweBenchPro().load()
    assert len(first) == swep.SWEP_USABLE_TASKS == swep.SWEP_PROVENANCE.usable_tasks
    assert [t.task_id for t in first] == [t.task_id for t in second]
    assert all(t.benchmark == swep.SWEP_NAME for t in first)


@needs_docker
@needs_rows
def test_the_loadable_census_is_far_larger_than_the_gradeable_one_and_says_so():
    """THE ARITHMETIC, HELD AS A PROPERTY. The corpus needs one multi-gigabyte image per task and
    ~997.5 GB compressed for all 731, against ~76 GB free — so `usable_tasks` is a census and
    `usable_ids` is the truth. Asserting the gap rather than a magic number is what stops the census
    from being quietly redefined as the provisioned count, which would make a disk limit look like a
    benchmark size.

    `needs_docker` because both sides of the inequality are: `usable_ids` asks the GRADING daemon
    what it has, and on a box that has not declared one (§8b.9) the honest answer is none at all."""
    bench = swep.SweBenchPro()
    usable = bench.usable_ids()
    if not usable:
        # The property needs at least one provisioned image to hold. The benchmark is outside the
        # launch corpus (the corpus record) and its images were removed from the grading daemon on
        # 2026-09-07, so a daemon that holds none is the expected state, reported rather than failed.
        pytest.skip("no SWE-bench Pro instance image is provisioned on the grading daemon")
    assert 0 < len(usable) < len(bench.load())


def test_usable_ids_are_the_tasks_whose_image_is_already_provisioned(monkeypatch):
    """`--pull never` makes an absent image a refusal at grading time, which is the right failure —
    pulling gigabytes inside a graded episode would spend the arm's wall clock (§8b.2) on the
    validator's housekeeping — but it is only kind (§10) if a window can be checked before it opens.
    """
    bench = swep.SweBenchPro()
    bench._rows = {"a": dict(FIXTURE, dockerhub_tag="a"), "b": dict(FIXTURE, dockerhub_tag="b")}
    monkeypatch.setattr(sandbox, "images_present",
                        lambda names: {n for n in names if n.endswith(":a")})
    assert bench.usable_ids() == ("a",)


def test_naming_a_tasks_image_reads_a_dataset_column_and_fetches_nothing(monkeypatch):
    """`usable_ids` calls `image_for` once per task over 731 tasks. The sibling adapter measured what
    happens when that route touches the network: naming a SWE-bench Verified image went through the
    shipped spec builder and hit raw.githubusercontent.com 500 times, and an offline validator could
    not take the census at all. Here the tag is a dataset column, so detonating the script fetcher is
    the check — the format is upstream's and not this module's to assert on."""
    monkeypatch.setattr(swep, "_script", lambda path: pytest.fail("image_for fetched a script"))
    bench, task = fixture_bench()
    assert bench.image_for(task.task_id) == f"{swep.SWEP_REGISTRY}:{FIXTURE['dockerhub_tag']}"


# --- what the score is made of ----------------------------------------------------------------------
def test_partial_credit_is_the_fail_to_pass_fraction_with_the_regression_suite_as_a_gate():
    """M2a exit 3. The published metric is `(f2p | p2p) <= passed` — one bit. This agrees with it at
    0, at 1, and whenever there is a single fail_to_pass test, and is finer in between: only 209 of
    731 tasks are binary here against 345 of 500 on SWE-bench Verified, so this is the benchmark
    where §5.4's ~670-task arms actually come down."""
    assert swep.swep_grade([{"name": "f1", "status": "PASSED"},
                            {"name": "f2", "status": "FAILED"},
                            {"name": "p1", "status": "PASSED"}], ["f1", "f2"], ["p1"]) == Grade(1, 2)
    assert swep.swep_grade([{"name": "f1", "status": "PASSED"},
                            {"name": "p1", "status": "PASSED"}], ["f1"], ["p1"]) == Grade(1, 1)


def test_breaking_a_test_that_used_to_pass_scores_zero_however_much_else_was_fixed():
    """The regression suite is a GATE and not a summand. A patch that fixes the issue and breaks the
    library is not a partial solution to the task the benchmark asked — and scoring the fraction of
    ALL tests would instead hand a do-nothing patch nearly full marks, which `real.swe_grade`
    measured at 0.997 on a Verified instance and is worse here (pass_to_pass runs to 3509 tests)."""
    assert swep.swep_grade([{"name": "f1", "status": "PASSED"},
                            {"name": "p1", "status": "FAILED"}], ["f1"], ["p1"]) == Grade(0, 1)


def test_a_task_with_no_regression_suite_is_graded_on_its_own_tests_alone():
    """Half the corpus — 372 of 731 rows — ships an EMPTY pass_to_pass, where the sibling benchmark
    has a median of 51. The gate is then vacuous and the score is the fail_to_pass fraction, which is
    worth pinning: a gate implemented as "at least one regression test passed" would score every one
    of those 372 tasks zero."""
    assert swep.swep_grade([{"name": "f1", "status": "PASSED"}], ["f1"], []) == Grade(1, 1)


def test_a_parser_that_wrote_no_report_raises_instead_of_scoring_zero():
    """The discriminator between a submission that failed and a container that never ran (§8b.3). It
    is the marker and not an exit code, because the shipped run scripts swallow their own test
    failures (`|| true`, `|| echo '{"tests":[]}'`) and exit 0 either way."""
    with pytest.raises(sandbox.SandboxError, match="no V3_SWEP_REPORT line"):
        swep.swep_verdict("Traceback (most recent call last):\nModuleNotFoundError: no pytest\n")
    with pytest.raises(sandbox.SandboxError, match="no readable report"):
        swep.swep_verdict(f"{swep.SWEP_VERDICT}\n")
    assert swep.swep_verdict(report(("t", "PASSED"))) == [{"name": "t", "status": "PASSED"}]


def test_an_empty_answer_scores_zero_without_starting_a_container(monkeypatch):
    """A worker that returned nothing is a genuine failure, not a grading incident — and paying for a
    multi-gigabyte container to learn that, ~250 times a window, is the arm's wall clock spent on
    nothing. The short-circuit is also what keeps this test offline: it returns before the shipped
    grader scripts are fetched."""
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: pytest.fail("started a container"))
    bench, task = fixture_bench()
    assert bench.grade("   \n", task) == Grade(0, 2)


# --- the shipped grader, and the glue around it -----------------------------------------------------
def test_the_list_columns_are_python_reprs_and_not_json_so_they_are_decoded_as_such():
    """MEASURED, not assumed: `json.loads` fails on 722 of the 2193 list-shaped column values at this
    revision, because test names contain apostrophes and the repr mixes quote styles. The official
    evaluator calls `eval()` on them; `ast.literal_eval` is the same decoding without executing the
    dataset on the machine that holds the subnet's scoring authority (D1)."""
    assert swep.swep_list(FIXTURE["fail_to_pass"]) == ["t.py::test_spin", "t.py::test_doesn't_wobble"]
    assert swep.swep_list('["a"]') == ["a"]
    assert swep.swep_list(["a"]) == ["a"]


def test_the_entry_script_restores_the_benchmarks_tests_after_the_models_patch():
    """The ordering upstream depends on and a rewrite would silently lose. `before_repo_set_cmd`'s
    last line checks the test files out from the FIX commit, and it must run after `git apply` — a
    patch that rewrote a test it is judged by would otherwise be graded with its own rewrite."""
    script = swep.swep_entryscript(FIXTURE, "", "")
    assert script.index("git apply") < script.index("git checkout f00 -- t.py")
    # Only the LAST line of the column, because the reset and the base checkout above it are already
    # in the script; replaying them after the patch would revert it.
    assert "git clean -fd" not in script


def test_the_entry_script_re_exports_the_image_env_the_shipped_grader_reads():
    """`PYTEST_ADDOPTS` carries `--continue-on-collection-errors` and `--reruns=3` in these images, so
    it is not decoration: dropping it changes which tests are reported and how flaky ones resolve.
    Upstream re-exports both dockerfiles' ENV lines even though Docker already sets them, and this is
    transcribed rather than trimmed for that reason."""
    script = swep.swep_entryscript(FIXTURE, 'ENV PYTEST_ADDOPTS="--reruns=3"', "ENV GOFLAGS=-mod=mod")
    assert 'export PYTEST_ADDOPTS="--reruns=3"' in script
    assert "export GOFLAGS=-mod=mod" in script


def test_the_entry_script_sends_the_verdict_back_through_the_log_not_the_mount():
    """The payload mount is READ-ONLY, so a graded patch cannot edit the tests it is scored against —
    which means the shipped parser's `output.json`, written to a read-write `/workspace` upstream,
    has no shared directory to come back through. It goes to /tmp and is echoed after the marker."""
    script = swep.swep_entryscript(FIXTURE, "", "")
    assert f"{swep.SWEP_VERDICT}\ncat /tmp/output.json" in script
    assert "/tmp/stdout.log" in script and "/tmp/stderr.log" in script


@needs_scripts
def test_the_run_script_and_the_parser_are_the_benchmarks_own_files_at_a_pinned_commit():
    """§6.1.2 and §6.1.1 together. The grader is per-instance and lives in a GitHub repository with no
    PyPI package, so it is fetched and cached rather than imported — and the commit is pinned because
    the run scripts have been revised at least twice since release (the repo's own 1/7 and 2/9 news
    entries), which would silently reprice `λ_b` and `C_b` (§5.1b)."""
    parser = swep._script(f"run_scripts/{INSTANCE}/parser.py")
    assert "def parse_test_output" in parser and "class TestStatus" in parser
    assert len(swep.SWEP_SCRIPTS_COMMIT) == 40
    assert (swep.SWEP_SCRIPTS_CACHE / swep.SWEP_SCRIPTS_COMMIT).is_dir()


@needs_scripts
def test_a_shipped_grader_file_that_cannot_be_fetched_raises_instead_of_grading_without_it(
        monkeypatch):
    """A benchmark that quietly graded with a missing run script would report every task failed, and
    §6.3b requires a narrowed window to be visible rather than silent. `BenchmarkUnavailable` is the
    right class: it means this benchmark cannot be in the window at all, as against a grader failure
    mid-arm, which drops one task from both arms (§6.3c)."""
    with pytest.raises(real.BenchmarkUnavailable, match="shipped grader file"):
        swep._script("run_scripts/no-such-instance/run_script.sh")


# --- what every adapter owes, whatever it grades ---------------------------------------------------
def test_the_adapter_satisfies_the_benchmark_protocol():
    """`grader(benchmark)` is the seam `Scaffold` is constructed with, so building one is the cheapest
    complete statement that the adapter is usable at all."""
    bench = swep.SweBenchPro()
    assert bench.name == swep.SWEP_NAME and callable(grader(bench))
    assert callable(bench.load) and callable(bench.tools) and callable(bench.grade)


def test_no_tool_is_declared_that_the_scaffold_cannot_run():
    """Still `()`, and the REASON CHANGED with the read channel, so it is restated rather than left.

    It is no longer that the scaffold cannot run a tool — `v3/tools.py` runs three of them, and
    R2E-Gym declares them. It is that this benchmark's instance images float on `:latest`, which is
    not a pin: a read against one is not reproducible across a re-pull, so an observation a score was
    computed from could not be re-executed by a third party. The honest consequence is unchanged and
    is recorded rather than hidden — this repository measured SWE-bench Pro one-shot at 0/120, and
    §6.1.3's admission probe is what decides whether it contributes signal or only cost."""
    bench = swep.SweBenchPro()
    _, task = fixture_bench()

    assert bench.tools() == ()
    assert bench.environment(task) is None


def test_the_prompt_is_the_benchmarks_own_statement_including_the_requirements_block():
    """§2.1: what reaches a worker is the pinned task text. `helper_code/create_problem_statement.py`
    composes the official prompt from three columns, and Pro's issue text is deliberately
    underspecified — the acceptance criteria live in `requirements`. Forwarding only
    `problem_statement`, as the sibling adapter does for Verified, would be asking a harder question
    than the leaderboard asks and would make a v3 number unreadable against it."""
    _, task = fixture_bench()
    assert FIXTURE["problem_statement"] in task.prompt
    assert FIXTURE["requirements"] in task.prompt and FIXTURE["interface"] in task.prompt


def test_the_adapter_pins_a_commit_and_not_a_branch():
    """§6.1.1. `λ_b` and `C_b` are fitted per corpus and pinned (§5.1b), so a benchmark that moves
    under the arena reprices every score computed against it — silently, and after the miners who
    trained on the published traces trained against the old one."""
    assert len(swep.SWEP_PROVENANCE.revision) == 40


def test_the_record_names_the_licence_question_admission_will_not_answer_for_it():
    """M2a exit 4. "Probably fine" is how a subnet acquires a takedown mid-window, and this benchmark
    is the corpus's hardest case: the dataset card declares no licence, the eleven source
    repositories are GPL-3.0 and AGPL-3.0, and the grading images live in a third party's personal
    Docker Hub namespace with no terms at all. That is a decision for the owner, recorded as an
    enumerated verdict rather than as prose nobody re-reads."""
    assert swep.SWEP_PROVENANCE.redistribution == real.REDISTRIBUTION_UNRESOLVED
    assert "AGPL" in swep.SWEP_PROVENANCE.licence


def test_the_record_says_which_parts_of_the_grader_are_the_shipped_ones():
    """§3's argument that scaffold quality CANCELS between the arms holds because both are graded by
    something neither of them chose. Here the run script and the parser are upstream's, verbatim,
    but the entry script around them is TRANSCRIBED — the eval repository is not packaged, so unlike
    `swebench` 4.1.0 an upstream change does not reach this code by itself. That is exactly the kind
    of thing that must be in the record rather than in someone's memory."""
    assert swep.SWEP_PROVENANCE.grader == real.GRADER_AS_SHIPPED
    assert "transcribed" in swep.SWEP_PROVENANCE.grader_source
    assert swep.SWEP_PROVENANCE.deviation and swep.SWEP_PROVENANCE.granularity
