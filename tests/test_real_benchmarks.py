"""The three real adapters and the Docker seam under them (the build plan M2a, §6.1, §8b.3).

WHAT THESE TESTS PROTECT, AND WHY IT IS NOT THE SAME THING `test_benchmarks.py` PROTECTS. There
the world is a hash function and grading is arithmetic; here grading is *executing a program a model
wrote* on the machine that holds the subnet's scoring authority (D1). Three properties carry that,
and each one has a test named for it:

  * **A sandbox failure raises and never becomes a zero** (§8b.3, §6.3c). The failure this guards
    against is silent and permanent: a task scored 0 because Docker was down is indistinguishable
    afterwards from a task the miner failed, and the miner is charged for the owner's infrastructure.
  * **The line between the two is drawn where it can actually be drawn.** A non-zero exit code is
    ordinary — `git apply` refusing a malformed patch is a fact about the answer. A missing verdict
    marker is not. Every one of the four discrimination tests below fixes one case on one side of
    that line.
  * **Partial credit means the thing that helps** (§5.1, §5.4). Two of these tests exist because the
    obvious partial credit is wrong on SWE-bench in a way that would have looked fine: fraction of
    ALL tests passing hands a do-nothing patch 0.997.

NO MODEL IS EVER CALLED. Submissions are fixtures — a program written here, the dataset's own gold
patch, a letter. The only money-shaped thing in the module is Docker, which is free.

TESTS THAT NEED SOMETHING SKIP CLEANLY. Docker, and the two ungated datasets, are checked once at
import; HLE is gated, so nothing here can load it and the tests that would have are replaced by
tests of the pure functions plus a recorded refusal.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import pkgutil
import subprocess
import threading
import time

import pytest

from thirtyspokes.v3 import benchmarks, window
from thirtyspokes.v3.benchmarks import real, sandbox
from thirtyspokes.v3.benchmarks.base import Grade, grader
from thirtyspokes.v3.conductor import MockConductor
from thirtyspokes.v3.scaffold import Scaffold
from thirtyspokes.v3.types import TOOL_NAMES, Catalog, CatalogEntry, TaskSpec
from thirtyspokes.v3.worker import MockWorker

# The one instance whose image is expected locally; the whole SWE-bench Docker path is measured on
# it (2.2 s for the gold patch). Any of the 500 would do — this one is small and fast.
SWE_INSTANCE = "pylint-dev__pylint-4604"


def _cached(load):
    """A dataset if it is already fetchable, else None, so a test skips rather than hit the net."""
    try:
        return load()
    except real.BenchmarkUnavailable:
        return None


LCB_ROWS = _cached(lambda: real.LiveCodeBench()._loaded())
SWE_ROWS = _cached(lambda: real.SweBenchVerified()._loaded())

# `swebench` IS SWE-bench Verified's grader and it is in no pyproject extra, so a clean environment
# has the dataset and not the thing that scores it. Guarded here rather than left to fail, because
# an unimportable grader must SKIP these tests and not error them: erroring hides the one benchmark
# whose grading path is the shipped one behind an environment problem.
HAS_SWEBENCH = importlib.util.find_spec("swebench") is not None


def _swe_image_ready() -> bool:
    return (SWE_ROWS is not None and HAS_SWEBENCH and sandbox.image_present(
        real.SweBenchVerified().image_for(f"{real.SWE_NAME}-{SWE_INSTANCE}")))


needs_docker = pytest.mark.skipif(
    not sandbox.available(),
    reason=f"grading needs a docker client AND a declared host ({sandbox.DOCKER_HOST_ENV})")
needs_lcb = pytest.mark.skipif(LCB_ROWS is None, reason="LiveCodeBench is not cached")
needs_swe = pytest.mark.skipif(SWE_ROWS is None, reason="SWE-bench Verified is not cached")
needs_swebench = pytest.mark.skipif(not HAS_SWEBENCH,
                                    reason="the swebench package ships this benchmark's grader")
# Two different facts reach this predicate as one False, so the reason has to say WHICH: an absent
# image, or an undeclared host, where `image_present` returns False without asking any daemon.
# Measured on the owner's box — the image IS provisioned there and the skip still fired, so a fixed
# "not provisioned" reason would have the suite reporting a falsehood about the machine it ran on.
needs_swe_image = pytest.mark.skipif(
    not _swe_image_ready(),
    reason=(f"the {SWE_INSTANCE} instance image is not provisioned" if sandbox.declared() else
            f"{sandbox.DOCKER_HOST_ENV} is unset, so no daemon was asked about the image"))

# A LiveCodeBench-shaped row this file owns, so the Docker grading path can be exercised against
# programs whose verdict is known. A real competition problem cannot serve here: nothing in this
# repository can write a correct solution to one on demand, so a "known-good submission" (M2a exit 2)
# would be unobtainable and the exit criterion untestable.
DOUBLE_IT = {"question_id": "fixture-double", "platform": real.LCB_PLATFORM, "starter_code": "",
             "difficulty": "easy", "question_content": "Read n and print 2n.",
             "public_test_cases": json.dumps([
                 {"input": "3\n", "output": "6\n", "testtype": "stdin"},
                 {"input": "4\n", "output": "8\n", "testtype": "stdin"},
                 {"input": "5\n", "output": "10\n", "testtype": "stdin"}]),
             "private_test_cases": ""}

GOOD_PROGRAM = "import sys\nprint(int(sys.stdin.readline()) * 2)\n"
BAD_PROGRAM = "print('nope')\n"
# Right on 3 and 4, wrong on 5 — two of three cases, which is a score no pass/fail grader can report.
PARTIAL_PROGRAM = ("import sys\nn = int(sys.stdin.readline())\nprint(n * 2 if n < 5 else 0)\n")


def fixture_lcb(row: dict = DOUBLE_IT) -> tuple[real.LiveCodeBench, TaskSpec]:
    bench = real.LiveCodeBench()
    bench._rows = real.lcb_rows([row])
    return bench, bench.load()[0]


# --- the sandbox: what is the validator's failure and what is the miner's (§8b.3) -----------------
@needs_docker
def test_a_grader_container_cannot_reach_the_network_at_all():
    """`--network none` is the load-bearing flag, and not only for hygiene: a graded program that
    could reach the network could fetch the expected outputs and score 1.0 without solving anything.
    Asserted by running the thing it forbids rather than by reading the argument list back."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        result = sandbox.run(
            real.LCB_IMAGE,
            ["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 80), 5)"],
            mount=work / "payload", log_path=logs / "log", timeout=60)
    assert result.exit_code != 0
    assert "Error" in result.output or "error" in result.output


@needs_docker
def test_a_missing_image_is_a_sandbox_failure_and_not_a_zero():
    """The §8b.3 line, at the place a validator actually crosses it: an unprovisioned image under
    `--pull never` is the owner's omission, so the task leaves BOTH arms rather than scoring the
    miner zero — and the alternative, pulling gigabytes inside a graded episode, would spend the
    arm's wall clock (§8b.2) on the validator's housekeeping."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        with pytest.raises(sandbox.SandboxError, match="could not start"):
            sandbox.run("v3-no-such-image:never", ["true"], mount=work / "payload",
                        log_path=logs / "log", timeout=60)


@needs_docker
def test_a_container_that_overruns_its_clock_is_excluded_rather_than_scored():
    """A container-level timeout mixes "the program hung" with "the validator's clock was tight",
    and from outside the box they are one event. §8b.3 breaks the tie one way only. `koth/lcb.py`
    scored this 0.0; that is the change this module makes deliberately, and the repair for the case
    that IS the miner's lives inside the container (see the hanging-program test below)."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        with pytest.raises(sandbox.SandboxError, match="excluded from both arms"):
            sandbox.run(real.LCB_IMAGE, ["sleep", "120"], mount=work / "payload",
                        log_path=logs / "log", timeout=3)


@needs_docker
def test_a_killed_container_is_actually_killed_and_not_merely_abandoned(monkeypatch):
    """`docker run` under a client-side timeout kills the CLIENT; the container keeps its memory and
    its CPU share for as long as the workload wants. On a validator running ~500 graded episodes a
    window, that is how one hung task becomes an unusable machine — and `--rm` does not help,
    because it fires when the container exits.

    THE NAME IS PINNED SO THE ASSERTION IS ABOUT THIS CONTAINER. An earlier version asked whether
    ANY `v3-` container was running, which is a question about the host: it passed for two runs and
    failed on the third, and a test that fails for something another test left behind reports a bug
    nobody wrote. Scoping it to one name also made `_kill`'s real hole visible — one removal loses
    the race against a container the daemon had not finished creating when the client was killed.
    """
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda n: "deadbeef" * 2)
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        with pytest.raises(sandbox.SandboxError):
            sandbox.run(real.LCB_IMAGE, ["sleep", "120"], mount=work / "payload",
                        log_path=logs / "log", timeout=3)
    # `_docker`, not the bare client: the container lives on the daemon `V3_DOCKER_HOST` names, and a
    # bare `docker ps` asks the ambient one — which on the controller is a different machine, so the
    # check would pass vacuously with the container still running where the grades are.
    survivors = subprocess.run(
        sandbox._docker("ps", "-a", "--format", "{{.Names}}", "--filter",
                        f"name=^v3-{'deadbeef' * 2}$"), capture_output=True, text=True).stdout
    assert survivors.split() == []


@needs_docker
def test_a_container_the_daemon_creates_after_the_kill_is_still_reaped():
    """THE RACE THE FULL SUITE FOUND, made deterministic. `subprocess.run`'s timeout kills the docker
    CLIENT; if the daemon had not finished creating the container by then, a single `docker rm -f`
    answers "No such container" and the container starts afterwards, unattended, holding its memory
    for as long as the workload wants. Creation is slowest exactly when the machine is loaded, which
    is when a validator is 400 episodes into a window — so the leak arrives at the worst moment and
    looks like nothing at all. Here the late creation is done on purpose, one second after the kill
    begins; a `_kill` that asks once fails this and the shipped one does not."""
    name = f"v3-{'feedface' * 2}"

    def late_creation():
        time.sleep(1.0)
        # On the daemon `_kill` reaps from (`_docker` adds `--host`); with the bare client the late
        # container would be created on the ambient daemon — on the controller, the shared
        # production one — and `_kill` would be reaping a different machine. Measured 2026-09-07.
        subprocess.run(sandbox._docker("run", "-d", "--name", name, "--network", "none",
                                       real.LCB_IMAGE, "sleep", "120"), capture_output=True)

    creator = threading.Thread(target=late_creation)
    creator.start()
    try:
        sandbox._kill(name)
        creator.join()
        survivors = subprocess.run(
            sandbox._docker("ps", "-a", "--format", "{{.Names}}", "--filter", f"name=^{name}$"),
            capture_output=True, text=True).stdout
        assert survivors.split() == []
    finally:
        creator.join()
        subprocess.run(sandbox._docker("rm", "-f", name), capture_output=True)


@needs_docker
def test_a_non_zero_exit_code_is_data_because_a_failing_submission_is_an_outcome():
    """The other side of the same line. A worker model's answer crashing is information about that
    rung, so the sandbox returns it and the adapter decides; only an absent verdict marker means the
    grader itself did not run."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        result = sandbox.run(real.LCB_IMAGE, ["python", "-c", "raise SystemExit(3)"],
                             mount=work / "payload", log_path=logs / "log", timeout=60)
    assert result.exit_code == 3


def test_no_docker_client_at_all_is_a_sandbox_failure_rather_than_a_crash(tmp_path, monkeypatch):
    """The first thing that happens on a validator whose docker socket moved. It must arrive as the
    exclusion §8b.3 describes, not as a `FileNotFoundError` two frames below the grader."""
    monkeypatch.setattr(sandbox, "DOCKER", "v3-no-such-docker-binary")
    # A grading host has to be DECLARED before `run` executes anything (§8b.9), and this test is
    # about the failure AFTER that point — so it says `local`, which is what a single-box deployment
    # says, and gets the missing-client refusal rather than the undeclared one.
    monkeypatch.setenv(sandbox.DOCKER_HOST_ENV, sandbox.DOCKER_HOST_LOCAL)
    (tmp_path / "payload").mkdir()
    with pytest.raises(sandbox.SandboxError, match="on PATH"):
        sandbox.run("img", ["true"], mount=tmp_path / "payload", log_path=tmp_path / "log",
                    timeout=5)


@needs_docker
def test_the_mount_is_read_only_so_a_graded_program_cannot_edit_its_own_test_cases():
    """The mount is the one directory the host and the container share. A program that could write
    to it could replace the expected outputs it is being graded against."""
    with sandbox.workspace() as (work, logs):
        payload = work / "payload"
        payload.mkdir()
        (payload / "tests.json").write_text("original")
        sandbox.run(real.LCB_IMAGE,
                    ["python", "-c", f"open('{sandbox.MOUNT}/tests.json', 'w').write('rewritten')"],
                    mount=payload, log_path=logs / "log", timeout=60)
        assert (payload / "tests.json").read_text() == "original"


@needs_docker
def test_the_memory_cap_reaches_the_container_so_one_task_cannot_take_the_host_down():
    """Nothing bounds what a submitted program tries to allocate. The cap is what stops the ~500
    graded episodes of a window from being one `bytearray` away from a dead validator — and because
    a killed container writes no verdict, breaching it produces an exclusion, never a score."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        result = sandbox.run(real.LCB_IMAGE,
                             ["python", "-c", "b = bytearray(512 * 1024 * 1024); print(len(b))"],
                             mount=work / "payload", log_path=logs / "log", timeout=120,
                             memory="64m")
    assert result.exit_code != 0


@needs_docker
def test_the_captured_output_is_capped_while_the_whole_log_is_kept_on_disk():
    """A test suite's output routinely runs to megabytes, and that is exactly the size that must
    never be materialised into an exception message — while SWE-bench's shipped parser needs every
    byte of it, so the file cannot be truncated either."""
    with sandbox.workspace() as (work, logs):
        (work / "payload").mkdir()
        result = sandbox.run(
            real.LCB_IMAGE, ["python", "-c", "print('x' * 4_000_000)"],
            mount=work / "payload", log_path=logs / "log", timeout=120)
        assert len(result.output.encode()) <= sandbox.MAX_CAPTURE_BYTES
        assert result.log_path.stat().st_size > sandbox.MAX_CAPTURE_BYTES


def test_the_workspace_is_allocated_where_the_docker_daemon_can_also_see_it(tmp_path, monkeypatch):
    """MEASURED on testnet 526 (`koth/lcb.py`): `docker run -v` is resolved by the DAEMON, so when
    the validator is itself containerised, a path inside it does not exist on the host — the bind
    mounts nothing, the driver is absent, and not one code task grades. The client reported a
    healthy server throughout."""
    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    with sandbox.workspace() as (work, _):
        assert work.parent == tmp_path


def test_the_container_log_is_never_allocated_on_the_shared_grade_directory(tmp_path, monkeypatch):
    """THE OTHER HALF, AND IT IS BOTH A COST AND A CONTAINMENT PROPERTY. `V3_GRADE_DIR` is the one
    directory the daemon must also see, and under the deployed arrangement that is a shared mount:
    one read issued ~30 filesystem round trips across the link, about half of them the log's own
    (create / write / stat / tail / the grader's full re-read), measured at 1652 ms per read against
    0.24 ms local. The log is a host-side `stdout=` redirect no daemon ever opens — only `mount` is
    bind-mounted — so it belongs on local storage, and moving it there also leaves strictly less of
    what grading writes on the filesystem the other machine can read.

    Asserted as "not under the grade directory", not as "in /tmp": the root is configurable, because
    an operator sets `V3_GRADE_DIR` when the default filesystem is too small and a test suite's log
    runs to megabytes."""
    monkeypatch.setenv(sandbox.GRADE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(sandbox.LOG_DIR_ENV, raising=False)
    with sandbox.workspace() as (work, logs):
        assert tmp_path not in logs.parents and logs != tmp_path
        assert logs.is_dir() and not any(logs.iterdir())
        assert work.parent == tmp_path      # ...while the payload still resolves for the daemon

    elsewhere = tmp_path.parent / "v3-logs"
    elsewhere.mkdir()
    monkeypatch.setenv(sandbox.LOG_DIR_ENV, str(elsewhere))
    with sandbox.workspace() as (_, logs):
        assert logs.parent == elsewhere


# --- LiveCodeBench ---------------------------------------------------------------------------------
@needs_docker
def test_a_known_good_submission_scores_one_and_a_known_bad_one_scores_zero():
    """M2a exit 2, through the real Docker grader on a program this file wrote."""
    bench, task = fixture_lcb()
    assert bench.grade(GOOD_PROGRAM, task) == Grade(3, 3)
    assert bench.grade(BAD_PROGRAM, task) == Grade(0, 3)


@needs_docker
def test_partial_credit_is_the_fraction_of_test_cases_passing_not_a_rounded_pass_fail():
    """M2a exit 3. LiveCodeBench's own metric is all-or-nothing over every case, which would report
    this program and the one that fails everything as the same number — and §5.4's task counts are
    computed for exactly that loss of information."""
    bench, task = fixture_lcb()
    grade = bench.grade(PARTIAL_PROGRAM, task)
    assert grade == Grade(2, 3)
    assert grade.score == pytest.approx(2 / 3)


@needs_docker
def test_a_hanging_program_is_scored_zero_rather_than_excluded_from_both_arms():
    """The repair that makes `sandbox.run`'s timeout safe to treat as the validator's fault. The
    driver bounds each case, so a program that never terminates still comes back with a verdict and
    is scored as the failure it is; without this the miner's own hang would leave both arms."""
    one_case = dict(DOUBLE_IT, public_test_cases=json.dumps(
        [{"input": "3\n", "output": "6\n", "testtype": "stdin"}]))
    bench, task = fixture_lcb(one_case)
    assert bench.grade("while True:\n    pass\n", task) == Grade(0, 1)


def test_an_empty_answer_scores_zero_without_starting_a_container(monkeypatch):
    """A worker that returned nothing is a genuine failure, not a grading incident — and paying for
    a container to learn that, ~500 times a window, is the arm's wall clock spent on nothing."""
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: pytest.fail("started a container"))
    bench, task = fixture_lcb()
    assert bench.grade("   \n", task) == Grade(0, 3)


def test_a_driver_that_printed_no_verdict_raises_instead_of_scoring_zero():
    """The discriminator between a submission that failed and a container that never ran. It is a
    marker and not an exit code because a program crashing on every case exits non-zero and is an
    ordinary outcome (§8b.3)."""
    with pytest.raises(sandbox.SandboxError, match="no V3_CASES line"):
        real.lcb_verdict("Traceback (most recent call last):\n  ImportError: no json\n")
    assert real.lcb_verdict(f"noise\n{real.LCB_VERDICT} 7 12\n") == (7, 12)


def test_the_usable_task_set_is_the_one_grading_convention_this_seam_implements():
    """stdin/stdout only. LeetCode-style problems use a functional calling convention that needs a
    second harness, and mixing the two silently changes what "solved" means (`koth/lcb.py`)."""
    rows = real.lcb_rows([
        DOUBLE_IT,
        dict(DOUBLE_IT, question_id="lc", platform="leetcode"),
        dict(DOUBLE_IT, question_id="starter", starter_code="class Solution:")])
    assert list(rows) == [f"{real.LCB_NAME}-fixture-double"]


def test_extract_code_pulls_the_program_out_of_a_fenced_response():
    """Reused verbatim from `koth/lcb.py` so a score here means what it meant there; pinned by test
    because "improving" it silently rescores every measurement taken with it."""
    fenced = "Here you go:\n```python\nimport sys\nprint(input())\n```\nHope that helps."
    assert real.extract_code(fenced) == "import sys\nprint(input())\n"


def test_the_per_case_timeout_lives_inside_the_container_where_the_two_failures_differ():
    """Structural, and it is the premise of two other tests in this file: the sandbox's clock cannot
    tell a hung submission from a tight validator timeout, so the bound that CAN is the driver's."""
    assert f"timeout={real.LCB_PER_CASE_TIMEOUT}" in real._LCB_DRIVER


@needs_lcb
def test_livecodebench_loads_its_pinned_census_with_stable_ids():
    """M2a exit 1 and the census M3a exit 4 needs. Stable across two loads because the window's task
    order is derived from the nonce alone and the reveal publishes the IDs: an ID that renumbers
    makes two windows incomparable and the published traces unmatchable to their tasks."""
    first, second = real.LiveCodeBench().load(), real.LiveCodeBench().load()
    assert len(first) == real.LCB_USABLE_TASKS == real.LCB_PROVENANCE.usable_tasks
    assert [t.task_id for t in first] == [t.task_id for t in second]
    assert all(t.benchmark == real.LCB_NAME for t in first)


@needs_lcb
def test_the_difficulty_label_survives_loading_because_the_draw_has_to_stratify():
    """§6.3b. The usable pool is 26 easy / 26 medium / 60 hard, so a uniform draw lands hard-heavy —
    which `koth/lcb.py` measured as the least trustworthy tier (35 of 72 reference cells empty from
    truncation; the achievable gap falls from +0.083 to +0.042 once it is dropped). `TaskSpec` has
    no room for a label, so the benchmark keeps it."""
    bench = real.LiveCodeBench()
    labels = {bench.difficulty_of(t.task_id) for t in bench.load()}
    assert labels == {"easy", "medium", "hard"}


# --- LiveCodeBench release files: the tag, the draw, and the default that must not move ------------
# The pool digest of the DEFAULT corpus, computed through the pre-change load path (one file,
# `lcb_rows`, no release stamp, no group) and pinned here. It is the whole "nothing moved" claim in
# one number: `window.pool_digest` binds every field of every task — benchmark, ID, prompt, tools and
# `group` — and it is what a schedule commits to on chain, so a digest that still reads this after
# the release work is a corpus in which not one committed byte changed.
LCB_DEFAULT_POOL_DIGEST = "2e70d6c6f1a6da81114bce8bd81f9e852fc7ab040d2a597e6ab14c395a8f9b68"


def _all_releases_cached() -> bool:
    """True only when all six release files are ALREADY on disk, and never a download.

    The five older files are 4.35 GB. A test suite may not fetch that to decide whether to run a
    test, so the cache is asked directly rather than through `_hf_file`, which would fetch."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:                                          # pragma: no cover — env-dependent
        return False
    return all(isinstance(try_to_load_from_cache(real.LCB_REPO, name, repo_type="dataset",
                                                 revision=real.LCB_REVISION), str)
               for name in real.LCB_RELEASES)


needs_lcb_releases = pytest.mark.skipif(
    not _all_releases_cached(),
    reason="the five older LiveCodeBench release files (4.35 GB) are not in the HuggingFace cache")


def fixture_releases(rows_per_release: dict[str, int]) -> real.LiveCodeBench:
    """A LiveCodeBench whose corpus is synthetic rows spread across the named releases.

    The tag and the draw are pure functions of the labels, so they are testable without the 4.35 GB
    the older releases cost — which is exactly what a clean checkout does not have. The stamped
    `release` key is the one `_loaded` writes; `lcb_rows` stays a pure filter and does not."""
    bench = real.LiveCodeBench(tuple(rows_per_release))
    bench._rows = {f"{real.LCB_NAME}-{name}-{i}":
                   dict(DOUBLE_IT, question_id=f"{name}-{i}", release=name)
                   for name, count in rows_per_release.items() for i in range(count)}
    return bench


def test_the_default_corpus_is_the_one_file_that_shipped_and_this_work_did_not_move_it(
        tmp_path, monkeypatch):
    """THE POINT OF THE WHOLE CHANGE IS THAT THIS TEST PASSES. Being able to draw across the six
    releases is not enabling them: whether the older files are worth their contamination is the
    owner's call on a measurement that has not run (the LCB census §6), so `LiveCodeBench()`
    still opens exactly one file and still records the census of that one file.

    Asserted on the fetch itself rather than on the corpus, because it is the fetch that would give
    the enlargement away — five extra `_hf_file` calls and 4.35 GB."""
    asked = []
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(DOUBLE_IT) + "\n")
    monkeypatch.setattr(real, "_hf_file",
                        lambda repo, filename, revision: (asked.append(filename), rows)[1])
    bench = real.LiveCodeBench()
    tasks = bench.load()
    assert bench.releases == (real.LCB_FILE,) == ("test6.jsonl",)
    assert asked == ["test6.jsonl"]
    assert bench.provenance == real.LCB_PROVENANCE      # the record, per instance, still the default
    assert real.LCB_USABLE_TASKS == real.LCB_RELEASES[real.LCB_FILE] == 112
    assert [t.group for t in tasks] == [None]


@needs_lcb
def test_the_default_pool_digest_is_bit_for_bit_the_one_the_schedule_already_committed_to():
    """`window.pool_digest` is what a schedule entry pins so the pool cannot move under a window, so
    this is the strongest available statement that the default corpus is untouched: 112 tasks, every
    prompt, every ID, every tool list and every (absent) group hashing to the number the pre-change
    adapter produced. A prompt is what the scaffold forwards byte-for-byte (§2.1) — if this test ever
    fails, a commitment moved and every miner who committed against it is scored on something else."""
    assert window.pool_digest(real.LiveCodeBench().load()) == LCB_DEFAULT_POOL_DIGEST


def test_putting_the_release_in_group_would_move_a_commitment_that_nothing_else_moved():
    """THE MEASUREMENT BEHIND `load`'S FOURTH REASON, and the cheapest of the four to check.

    `group` is the duel's resample unit, so `pool_digest` binds it — which means the tag-versus-
    cluster question is not a free stylistic one. Had the release gone into `group`, the default
    corpus's on-chain commitment would have changed although not one prompt, ID or tool did, and
    `lcb` would have widened on every LiveCodeBench stratum for a correlation nobody has measured.
    The tag lives on the adapter instead, which costs a commitment nothing."""
    bench = fixture_releases({real.LCB_FILE: 3})
    tasks = bench.load()
    tagged = [dataclasses.replace(t, group=bench.release_of(t.task_id)) for t in tasks]
    assert window.pool_digest(tagged) != window.pool_digest(tasks)
    assert {t.group for t in tasks} == {None}


def test_the_release_a_task_came_from_is_carried_so_a_score_can_be_reported_per_release():
    """The reporting tag the census asks for (§6): the per-release score span across the arms is the
    read that decides the enable question, and it cannot be taken from a corpus that has forgotten
    which file each task came from. `test6` must stay readable as its own line, because the numbers
    that admitted this benchmark were all measured on it alone."""
    bench = fixture_releases({"test.jsonl": 2, "test6.jsonl": 2})
    assert {bench.release_of(t.task_id) for t in bench.load()} == {"test.jsonl", "test6.jsonl"}


def test_two_release_files_merge_and_each_row_is_stamped_with_the_file_it_came_from(
        tmp_path, monkeypatch):
    """WHO WRITES THE TAG, checked without the 4.35 GB — `fixture_releases` hands `_rows` the stamped
    shape it hopes for, so every other test above takes `_loaded` on trust, and the one test that does
    not is gated on a corpus a clean checkout has never downloaded. Mutation-measured: dropping the
    stamp, stamping the wrong file, or overwriting instead of merging all survive the suite on a
    machine without the older releases.

    The merge is the load-bearing half. `_loaded` keys six files into one dict with no collision rule
    because the census found no shared question ID; an `=` where the `update` is would keep only the
    last file, and a stratified draw would then be short by every earlier release without saying so.
    `usable_tasks` is asserted here too, because a per-instance census is only a census if it follows
    the corpus rather than remembering 112."""
    files = {}
    for release, ids in (("test.jsonl", ("old-a", "old-b")), ("test6.jsonl", ("new-a",))):
        files[release] = tmp_path / release
        files[release].write_text("".join(
            json.dumps(dict(DOUBLE_IT, question_id=i)) + "\n" for i in ids))
    monkeypatch.setattr(real, "_hf_file", lambda repo, filename, revision: files[filename])

    bench = real.LiveCodeBench(("test6.jsonl", "test.jsonl"))
    tasks = bench.load()
    assert {t.task_id: bench.release_of(t.task_id) for t in tasks} == {
        f"{real.LCB_NAME}-old-a": "test.jsonl", f"{real.LCB_NAME}-old-b": "test.jsonl",
        f"{real.LCB_NAME}-new-a": "test6.jsonl"}
    assert bench.provenance.usable_tasks == (real.LCB_RELEASES["test.jsonl"]
                                             + real.LCB_RELEASES["test6.jsonl"]) == 322


def test_an_unknown_or_empty_release_set_is_refused_before_anything_is_fetched():
    """A mistyped file name would otherwise be a 404 inside `_hf_file` at load time, and an empty one
    a benchmark that loads zero tasks — which `window.draw` narrows out silently, dropping N to 1 and
    gating the duel on a single benchmark. Both are caller mistakes and both are refused up front."""
    with pytest.raises(ValueError, match="non-empty subset"):
        real.LiveCodeBench(("test7.jsonl",))
    with pytest.raises(ValueError, match="non-empty subset"):
        real.LiveCodeBench(())


def test_the_enabled_releases_are_ordered_by_the_census_and_not_by_the_caller():
    """The `window.draw`/`pool_digest` principle applied one level down: what loads depends only on
    WHICH releases are enabled, never on the order somebody listed them in, so two callers who asked
    for the same corpus differently get the same corpus."""
    assert (real.LiveCodeBench(("test6.jsonl", "test.jsonl")).releases
            == real.LiveCodeBench(("test.jsonl", "test6.jsonl")).releases
            == ("test.jsonl", "test6.jsonl"))


def test_the_draw_is_balanced_across_releases_and_the_remainder_goes_to_the_oldest():
    """§6's measurement in one call: ~112 tasks stratified across the six releases instead of all of
    `test6`, for the same money and the same wall clock. 112 over six is 18 with four left over, and
    the leftovers go to the earliest releases in census order — a rule, not an accident, because a
    draw whose composition wandered would make the per-release spans incomparable between runs."""
    bench = fixture_releases({name: 40 for name in real.LCB_RELEASES})
    drawn = bench.draw_across_releases(112, seed="window-7")
    counts = [sum(1 for t in drawn if bench.release_of(t.task_id) == name)
              for name in real.LCB_RELEASES]
    assert counts == [19, 19, 19, 19, 18, 18]
    assert len(drawn) == 112 and len({t.task_id for t in drawn}) == 112


def test_the_draw_is_reproducible_from_its_seed_and_moves_when_the_seed_does():
    """Seeded by a string ranked through sha256, like `window._draw_key` and `scaffold.task_order`:
    reproducible from the published record by anyone holding the pool, with no library RNG stream in
    the loop. A draw that could not be re-derived would make a reported per-release span unauditable."""
    bench = fixture_releases({name: 40 for name in real.LCB_RELEASES})
    first = [t.task_id for t in bench.draw_across_releases(112, seed="window-7")]
    assert first == [t.task_id for t in bench.draw_across_releases(112, seed="window-7")]
    assert set(first) != {t.task_id for t in bench.draw_across_releases(112, seed="window-8")}


def test_a_release_too_small_for_its_share_does_not_silently_shrink_the_draw():
    """`test3` holds 53 usable tasks, so a 400-task draw across the six cannot take 67 from it. The
    short release contributes everything it has and the shortfall spreads over the others, which is
    the same round-robin rule the remainder follows — and it is why the count a caller asked for is
    the count it is billed for. Asking for more than the corpus holds returns the corpus, which is
    the same guard seen from the other end."""
    bench = fixture_releases({"test.jsonl": 40, "test3.jsonl": 5, "test6.jsonl": 40})
    drawn = bench.draw_across_releases(60, seed="window-7")
    counts = {name: sum(1 for t in drawn if bench.release_of(t.task_id) == name)
              for name in ("test.jsonl", "test3.jsonl", "test6.jsonl")}
    assert counts == {"test.jsonl": 28, "test3.jsonl": 5, "test6.jsonl": 27}
    assert len(bench.draw_across_releases(500, seed="window-7")) == 85


@needs_lcb_releases
def test_the_census_recorded_beside_the_constant_is_what_the_six_files_actually_load():
    """The counts in `LCB_RELEASES` are the enable decision's whole evidence base, so they are
    checked against the files rather than trusted: the LCB census measured them with a
    streaming script that never built an adapter, and this is the same numbers arrived at through
    `LiveCodeBench` itself.

    The duplicate check is load-bearing rather than tidy — `_loaded` merges six files into one dict
    with no collision rule, which is only safe because the census found no shared question ID. If
    that ever stops being true the pool silently shrinks and a release tag silently becomes wrong.

    IT COSTS ~9 s AND ~4.4 GB OF RSS, measured, which is why it is gated on the files already being
    present. That figure is also a number the enable decision should see: it is what every process
    that loads all six releases pays, against ~129 MB for `test6.jsonl` alone."""
    everything = real.LiveCodeBench(tuple(real.LCB_RELEASES))
    tasks = everything.load()                            # 4.35 GB parsed once, so once is all it is
    per_release = {name: sum(1 for t in tasks if everything.release_of(t.task_id) == name)
                   for name in real.LCB_RELEASES}
    assert per_release == real.LCB_RELEASES
    assert len(tasks) == sum(real.LCB_RELEASES.values()) == 602
    assert everything.provenance.usable_tasks == 602      # the record follows the corpus it describes
    assert all(t.group is None for t in tasks)


# --- SWE-bench Verified ----------------------------------------------------------------------------
def _swe_spec(instance: str = SWE_INSTANCE):
    """The shipped test spec, or a clean skip — because BUILDING one fetches, by upstream's design.

    `make_test_spec` builds the env script eagerly, and for a repo whose install spec is a
    `requirements.txt` that means `requests.get` against raw.githubusercontent.com at the pinned
    `environment_setup_commit`. Nothing here can remove it without reimplementing the grader, which
    §6.1.2 refuses. So the dependency is named and skipped on, rather than left to arrive as a
    connection error from six frames inside an assertion about partial credit.

    Called lazily, never at module scope: `image_for` used to route through this and the whole module
    therefore hit GitHub at COLLECTION, which made an offline run a collection error instead of a
    skip. `image_for` no longer builds scripts, and this must not put the call back.
    """
    try:
        return real.SweBenchVerified()._spec(f"{real.SWE_NAME}-{instance}")
    except Exception as exc:                                     # noqa: BLE001 — offline, 404, gate
        pytest.skip(f"the shipped spec builder could not reach this instance's environment: {exc}")


@needs_docker
@needs_swe_image
def test_the_datasets_own_gold_patch_resolves_and_an_empty_one_does_not():
    """M2a exit 2 for the hard case: a real instance image, the shipped eval script, the shipped log
    parser. The gold patch is the dataset's own reference solution, so a grader that scores it below
    1.0 is broken in a way no fixture would reveal."""
    _swe_spec()          # precondition: `grade` builds this same spec, and building one fetches
    bench = real.SweBenchVerified()
    task_id = f"{real.SWE_NAME}-{SWE_INSTANCE}"
    task = next(t for t in bench.load() if t.task_id == task_id)
    gold = bench._loaded()[task_id]["patch"]
    assert bench.grade(gold, task).score == 1.0
    assert bench.grade("", task).score == 0.0


@needs_docker
@needs_swe_image
def test_a_patch_that_will_not_apply_is_the_miners_zero_and_not_an_exclusion():
    """The one case the shipped parser cannot tell apart on its own: `get_logs_eval` reports the
    same "no statuses" for a patch that would not apply and for a container that died before writing
    its markers, and §8b.3 puts those on opposite sides of the line. Reading the shipped
    APPLY_PATCH_FAIL marker first is what separates them."""
    _swe_spec()          # precondition: `grade` builds this same spec, and building one fetches
    bench = real.SweBenchVerified()
    task_id = f"{real.SWE_NAME}-{SWE_INSTANCE}"
    task = next(t for t in bench.load() if t.task_id == task_id)
    nonsense = ("diff --git a/does_not_exist.py b/does_not_exist.py\n"
                "--- a/does_not_exist.py\n+++ b/does_not_exist.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert bench.grade(nonsense, task).score == 0.0


@needs_swe
@needs_swebench
def test_a_log_with_no_test_output_at_all_is_the_graders_failure_and_not_a_zero(tmp_path):
    """The other half of the same discrimination. Once the apply marker has been ruled out, a log
    the shipped parser cannot read means the container died before writing its markers — an OOM
    kill, a truncated run, a daemon restart — and §8b.3 excludes such a task from both arms rather
    than charging the miner for it."""
    empty = tmp_path / "log"
    empty.write_text("Killed\n")
    spec = _swe_spec()
    with pytest.raises(sandbox.SandboxError, match="no test output"):
        real._swe_report(spec, empty)


def _report(f2p_pass: int, f2p_fail: int, p2p_pass: int, p2p_fail: int) -> dict:
    """A shipped report over a fabricated status map — the real `get_eval_tests_report`, so the
    partial-credit arithmetic is checked against the grader that will produce it, not a stand-in."""
    from swebench.harness.grading import get_eval_tests_report

    f2p = [f"f2p_{i}" for i in range(f2p_pass + f2p_fail)]
    p2p = [f"p2p_{i}" for i in range(p2p_pass + p2p_fail)]
    statuses = {name: ("PASSED" if i < f2p_pass else "FAILED") for i, name in enumerate(f2p)}
    statuses |= {name: ("PASSED" if i < p2p_pass else "FAILED") for i, name in enumerate(p2p)}
    return get_eval_tests_report(statuses, {"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p})


@needs_swebench
def test_partial_credit_is_the_fail_to_pass_fraction_with_the_regression_suite_as_a_gate():
    """M2a exit 3 for the 31% of Verified tasks that carry more than one FAIL_TO_PASS test. It
    agrees with the published `resolved` metric exactly at 0 and 1, and whenever there is a single
    such test — 345 of 500, measured at this revision."""
    assert real.swe_grade(_report(3, 1, 40, 0), 4) == Grade(3, 4)
    assert real.swe_grade(_report(4, 0, 40, 0), 4) == Grade(4, 4)
    assert real.swe_grade(_report(0, 1, 40, 0), 1) == Grade(0, 1)


@needs_swebench
def test_breaking_a_test_that_used_to_pass_scores_zero_however_much_else_was_fixed():
    """The regression suite is a gate rather than a summand. A patch that fixes the issue and breaks
    the library is not a partial solution to the task the benchmark asked."""
    assert real.swe_grade(_report(4, 0, 39, 1), 4) == Grade(0, 4)


@needs_swebench
def test_scoring_the_fraction_of_all_tests_would_hand_a_do_nothing_patch_nearly_full_marks():
    """WHY THE OBVIOUS PARTIAL CREDIT IS WRONG HERE, as arithmetic rather than as an opinion. The
    measured instance carries 1 FAIL_TO_PASS against 364 PASS_TO_PASS, so a submission that changes
    nothing passes 364 of 365. Every task would sit near 1.0, two policies would differ by a
    rounding error, and §6.1.3's admission probe would drop the benchmark for having no signal — for
    a reason entirely internal to how the score was defined."""
    report = _report(0, 1, 364, 0)
    naive = (len(report["FAIL_TO_PASS"]["success"]) + len(report["PASS_TO_PASS"]["success"])) / 365
    assert naive == pytest.approx(0.9973, abs=1e-4)
    assert real.swe_grade(report, 1).score == 0.0


@needs_swebench
def test_the_runner_is_built_from_the_shipped_apply_sequence_rather_than_a_copy_of_it():
    """§6.1.2: the grader is used AS SHIPPED. `run_evaluation` tries three apply commands in order
    and, if all three fail, logs APPLY_PATCH_FAIL and never runs the tests. Importing both means a
    harness upgrade that changes either changes this too, instead of leaving a copy that silently
    disagrees about what "the patch applied" means."""
    from swebench.harness.constants import APPLY_PATCH_FAIL, DOCKER_WORKDIR
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS

    runner = real._swe_runner()
    assert all(cmd in runner for cmd in GIT_APPLY_CMDS)
    assert APPLY_PATCH_FAIL in runner and f"cd {DOCKER_WORKDIR}" in runner


def test_extract_patch_pulls_the_diff_out_of_a_fenced_response_and_ends_it_with_a_newline():
    """`git apply` refuses a patch that does not end in a newline, so an answer would be rejected for
    its last byte and scored as a failure to solve the problem."""
    body = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"
    assert real.extract_patch(f"Sure:\n```diff\n{body}\n```\n") == body + "\n"
    assert real.extract_patch(body) == body + "\n"
    assert real.extract_patch("   ") == ""


def test_usable_ids_are_the_tasks_whose_image_is_already_provisioned(monkeypatch):
    """`--pull never` makes an absent image a refusal at grading time, which is the right failure —
    but it is only kind (§10, "Kind") if the window can be checked before it opens."""
    bench = real.SweBenchVerified()
    bench._rows = {"a": {"instance_id": "a"}, "b": {"instance_id": "b"}}
    monkeypatch.setattr(bench, "image_for", lambda task_id: f"img-{task_id}")
    monkeypatch.setattr(sandbox, "images_present", lambda names: {n for n in names if n == "img-a"})
    assert bench.usable_ids() == ("a",)


@needs_swe
def test_swebench_verified_loads_its_pinned_census_with_the_test_lists_decoded():
    """M2a exit 1. The parquet stores both test lists as JSON strings and the shipped spec builder
    wants lists; decoding at load is the difference between reusing a grader and reusing most of
    one."""
    bench = real.SweBenchVerified()
    tasks = bench.load()
    assert len(tasks) == real.SWE_USABLE_TASKS == real.SWE_PROVENANCE.usable_tasks
    assert all(t.benchmark == real.SWE_NAME for t in tasks)
    row = bench._loaded()[f"{real.SWE_NAME}-{SWE_INSTANCE}"]
    assert isinstance(row["FAIL_TO_PASS"], list) and row["FAIL_TO_PASS"]


@needs_swe
@needs_swebench
def test_the_image_a_task_grades_in_is_the_published_one_for_that_instance():
    """Finding 2: one multi-gigabyte image per task, which is the tractability criterion M2a names
    and rules out. Pinned by test because the namespace is what decides whether the validator pulls
    the published images or is expected to have built 500 of them."""
    image = real.SweBenchVerified().image_for(f"{real.SWE_NAME}-{SWE_INSTANCE}")
    assert image.startswith(f"{real.SWE_IMAGE_NAMESPACE}/sweb.eval.")
    assert SWE_INSTANCE.replace("__", "_1776_") in image


@needs_swe
@needs_swebench
def test_naming_a_tasks_image_does_not_build_that_tasks_scripts(monkeypatch):
    """`usable_ids` calls `image_for` once per task, and `_spec` fetches a `requirements.txt` from
    raw.githubusercontent.com for every repo whose install spec names one. Routed through `_spec`,
    the local census §6.3b wants was 500 GitHub round trips and an offline validator could not take
    it at all — measured: this module could not even be COLLECTED under an egress guard, because the
    `needs_swe_image` marker calls `image_for` at import.

    Detonating `_spec` is the check, rather than asserting on the string: the name is already pinned
    by the test above, and what regresses is the route, not the format."""
    bench = real.SweBenchVerified()
    monkeypatch.setattr(type(bench), "_spec",
                        lambda self, task_id: pytest.fail("image_for went through the spec builder"))
    assert bench.image_for(f"{real.SWE_NAME}-{SWE_INSTANCE}")


@needs_swe
@needs_swebench
def test_the_census_and_the_grader_name_the_same_image_for_a_task():
    """The two are now built on separate routes — `image_for` from the key alone, `grade` from the
    shipped spec — so they can disagree, and the way they would is silent: `usable_ids` reports a
    task runnable, then grading asks for an image under a different arch that `--pull never` refuses.
    Held here against the shipped property, which is why `SWE_ARCH` is passed to both."""
    task_id = f"{real.SWE_NAME}-{SWE_INSTANCE}"
    assert real.SweBenchVerified().image_for(task_id) == _swe_spec().instance_image_key


# --- HLE, the non-code benchmark ------------------------------------------------------------------
def test_a_correct_option_scores_one_and_a_wrong_one_zero():
    """M2a exit 2 for a benchmark with no sandbox, no image and no subprocess — which is the point of
    it being in the corpus: a code-shaped assumption cannot hide in a protocol that also carries
    this."""
    assert real.hle_grade("Explanation: because.\nAnswer: C", "C") == Grade(1, 1)
    assert real.hle_grade("Explanation: because.\nAnswer: D", "C") == Grade(0, 1)


def test_the_last_answer_line_is_the_one_read_so_a_repeated_instruction_does_not_score():
    """A model that echoes the format instruction before answering would otherwise be graded on the
    instruction. The fallback is the last non-empty line, which is what a bare one-token reply is."""
    echoed = "Answer: {your chosen answer}\nExplanation: it is B.\nAnswer: B"
    assert real.hle_grade(echoed, "B") == Grade(1, 1)
    assert real.hle_grade("B", "B") == Grade(1, 1)
    assert real.hle_grade("(b).", "B") == Grade(1, 1)
    # The marker also decides which END to read from. After one, the answer is the line that follows
    # it, so the confidence line HLE's own format asks for below it is not read as the answer; with
    # no marker at all the reply reasons first and answers last, so reading its first line would
    # grade the reasoning.
    assert real.hle_grade("Answer: B\nConfidence: 90%", "B") == Grade(1, 1)
    assert real.hle_grade("Both look plausible.\nB", "B") == Grade(1, 1)


def test_a_binary_grader_publishes_its_own_coarseness_rather_than_hiding_it():
    """`base.Grade` carries the granularity the benchmark was capable of precisely so a benchmark
    that can only report a bit is VISIBLE as one — §5.4's task counts are computed for binary, and a
    corpus cannot be sized without knowing which of its benchmarks are."""
    assert real.hle_grade("Answer: A", "A").total == 1
    assert real.HLE_PROVENANCE.granularity.startswith("1/1")


def test_only_the_mechanically_gradeable_rows_are_admitted():
    """Three filters, each dropping rows this seam cannot honestly score. The shipped grader is an
    LLM judge, which costs a model call per graded task — charged to a miner through `final_b`
    (§5.1b) — and is non-deterministic, which is the one thing a PAIRED comparison cannot survive."""
    rows = [{"id": "ok", "answer_type": "multipleChoice", "answer": "B", "image": "",
             "question": "q"},
            {"id": "free", "answer_type": "exactMatch", "answer": "B", "image": "", "question": "q"},
            {"id": "prose", "answer_type": "multipleChoice", "image": "", "question": "q",
             "answer": "the second one"},
            {"id": "visual", "answer_type": "multipleChoice", "answer": "B", "question": "q",
             "image": "https://example.invalid/x.png"}]
    assert list(real.hle_rows(rows)) == [f"{real.HLE_NAME}-ok"]


def test_a_task_set_that_cannot_be_fetched_raises_instead_of_returning_an_empty_corpus():
    """A benchmark that quietly loaded zero tasks would leave the window's admitted set short with
    no record of why — and §6.3b requires a narrowed window to be visible rather than silent."""
    with pytest.raises(real.BenchmarkUnavailable, match="v3/no-such-dataset"):
        real._hf_file("v3/no-such-dataset", "test.jsonl", "0" * 40)


# --- what every adapter owes, whatever it grades --------------------------------------------------
# THE REGISTRY, WRITTEN OUT AGAIN HERE, because every check below takes its SUBJECT from
# `benchmarks.ADAPTERS`: a `for` over it cannot notice that it is short and an `all()` over it is
# True when it is empty. That is not hypothetical — the registry named 2 of the 6 adapters that
# ship, and with it set to `()` in-process the protocol, the tools and the pin checks all passed
# while covering nothing. An expectation read from the same tuple would have passed too, so this one
# is a literal. `registered_adapters` is what turns a shrunken registry red, and
# `test_an_empty_registry_turns_every_invariant_red` runs these checks under `()` to prove it does.
REGISTERED = ("livecodebench", "swebench-verified", "hle", "r2egym", "swebench-pro")


def registered_adapters():
    """`benchmarks.ADAPTERS`, refused unless it is still the whole registry.

    Sorted lists and not sets, so that the same adapter registered twice fails here as well:
    `score.py` groups by `TaskSpec.benchmark` and leave-one-out drops by it (§5.2), so two entries
    answering to one `name` split a benchmark in two and double its weight in an aggregate defined
    to weight benchmarks equally (§5.1).
    """
    names = [adapter().name for adapter in benchmarks.ADAPTERS]
    assert sorted(names) == sorted(REGISTERED), f"the registry is {names}, not {list(REGISTERED)}"
    return benchmarks.ADAPTERS


def test_every_adapter_the_package_ships_is_registered_or_refused_in_writing():
    """The defect these checks were rewritten after: six adapters shipped, the registry named two,
    and the other four were imported nowhere in `src/` — so no window could load them and nothing
    below covered them.

    `base.Benchmark` is a Protocol, so conformance is structural: a class with these three methods
    IS a benchmark to every caller in the package whether or not anything registers it. Hence
    discovery over the package's own modules rather than a list — a seventh adapter file that
    nobody registers fails here on the day it lands, which is the only moment the omission is cheap.
    """
    unregistered = {("base", "Benchmark"),          # the contract itself, not an adapter
                    ("mock", "MockBenchmark"),      # the offline world, no task set and no grader
                    ("real", "HumanitysLastExam")}  # superseded by `hle.py`; same `name = "hle"`
    shipped = set()
    for info in pkgutil.iter_modules(benchmarks.__path__):
        module = importlib.import_module(f"{benchmarks.__name__}.{info.name}")
        shipped |= {(info.name, name) for name, obj in inspect.getmembers(module, inspect.isclass)
                    if obj.__module__ == module.__name__
                    and all(callable(getattr(obj, m, None)) for m in ("load", "tools", "grade"))}
    registered = {(a.__module__.rsplit(".", 1)[-1], a.__name__) for a in registered_adapters()}
    assert shipped - registered == unregistered, (
        f"{sorted(shipped - registered - unregistered)} ship as benchmarks and are in no registry")


def test_every_adapter_satisfies_the_benchmark_protocol():
    """`grader(benchmark)` is the seam `Scaffold` is constructed with, so building one is the
    cheapest complete statement that an adapter is usable at all."""
    for adapter in registered_adapters():
        bench = adapter()
        assert isinstance(bench.name, str) and bench.name
        assert callable(grader(bench))
        assert callable(bench.load) and callable(bench.tools) and callable(bench.grade)


def test_no_adapter_declares_a_tool_the_scaffold_cannot_run():
    """THE SENTENCE IS UNCHANGED AND STAYS TRUE FOREVER; WHAT MOVED IS WHAT THE SCAFFOLD CAN RUN.

    `render._task` puts `task.tools` into the prompt as text, so a declaration the harness cannot
    honour is a prompt that lies to the Conductor about the worker's action space. Until the read
    channel existed the only honest declaration was `()`, and the honest consequence — recorded
    rather than hidden — was that the agentic benchmarks were being asked one-shot. `v3/tools.py`
    is what changed that, and only for the three read-only verbs: anything outside `TOOL_NAMES` is
    still a declaration nothing executes.

    The other direction is asserted for every adapter that declares nothing, because that is the
    half that can be checked with no corpus loaded; R2E-Gym's own environment is asserted against a
    fixture row in `test_bench_r2egym.py`, where `image_for` has a row to read.
    """
    probe = TaskSpec(task_id="probe", benchmark="probe", prompt="p", tools=())
    for adapter in registered_adapters():
        bench = adapter()
        assert set(bench.tools()) <= set(TOOL_NAMES), (
            f"{bench.name} declares a verb the scaffold cannot run")
        if not bench.tools():
            assert bench.environment(probe) is None, (
                f"{bench.name} has an environment and declares no tools, so nothing would use it")


def test_every_adapter_pins_a_commit_and_not_a_branch():
    """§6.1.1. `λ_b` and `C_b` are fitted per corpus and pinned (§5.1b), so a benchmark that moves
    under the arena reprices every score computed against it — silently, and after the miners who
    trained on the published traces trained against the old one."""
    for adapter in registered_adapters():
        assert len(adapter().provenance.revision) == 40
    with pytest.raises(ValueError, match="not a commit id"):
        real.Provenance(repo="r", revision="main", licence="l", grader=real.GRADER_AS_SHIPPED,
                        redistribution=real.REDISTRIBUTION_PERMITTED, grader_source="s",
                        deviation="d", granularity="g", usable_tasks=1)


def test_every_adapter_records_the_licence_question_admission_will_not_answer_for_it():
    """M2a exit 4. "Probably fine" is how a subnet acquires a takedown mid-window, so the field is
    an enumerated verdict rather than free text. Only R2E-Gym's says `permitted`, and it is the one
    whose card actually grants it; the rest are decisions waiting on the owner rather than facts an
    adapter can establish."""
    verdicts = {a().name: a().provenance.redistribution for a in registered_adapters()}
    assert set(verdicts.values()) <= {real.REDISTRIBUTION_PERMITTED, real.REDISTRIBUTION_UNRESOLVED,
                                      real.REDISTRIBUTION_REFUSED}
    # HLE's verdict is `refused` and it is carried by `benchmarks/hle.py`, which is the HLE this
    # registry holds; `real.HumanitysLastExam` is the superseded copy and is registered nowhere,
    # because two adapters answering to one `name` split a benchmark in two and double its weight
    # in the equal-weight aggregate (§5.1) — a silent scoring bug, not a tidiness one.


def test_every_adapter_says_whether_its_grader_is_the_shipped_one():
    """§3's argument that scaffold quality CANCELS between the two arms holds because both arms are
    graded by something neither of them chose. A reimplementation still cancels within a duel; what
    it costs is the ability to read a v3 number against the published leaderboard, which is how a
    broken adapter is normally caught. So it is recorded, per benchmark, with what differs."""
    for adapter in registered_adapters():
        prov = adapter().provenance
        assert prov.grader in (real.GRADER_AS_SHIPPED, real.GRADER_REIMPLEMENTED)
        assert prov.grader_source and prov.deviation and prov.granularity
    assert real.SWE_PROVENANCE.grader == real.GRADER_AS_SHIPPED


def test_an_empty_registry_turns_every_invariant_red(monkeypatch):
    """The vacuous pass, asserted rather than remembered — this repository's worst failure mode.

    These are the real check functions above, called with the registry emptied: each one must fail.
    A check that survives this is one that would have gone on reporting a corpus it stopped
    covering, which is exactly what happened while `ADAPTERS` named two of six.
    """
    monkeypatch.setattr(benchmarks, "ADAPTERS", ())
    for check in (test_every_adapter_the_package_ships_is_registered_or_refused_in_writing,
                  test_every_adapter_satisfies_the_benchmark_protocol,
                  test_no_adapter_declares_a_tool_the_scaffold_cannot_run,
                  test_every_adapter_pins_a_commit_and_not_a_branch,
                  test_every_adapter_records_the_licence_question_admission_will_not_answer_for_it,
                  test_every_adapter_says_whether_its_grader_is_the_shipped_one):
        with pytest.raises(AssertionError, match="the registry is"):
            check()


@needs_docker
def test_a_real_adapter_drops_into_the_shipped_scaffold_and_the_worker_sees_the_task_verbatim():
    """The §2.1 invariant, end to end, against a grader that really executes what comes back. The
    catalog, the parser and the episode loop are the shipped ones; only the Conductor and the pool
    are mocked, because calling either for real would spend money."""
    bench, task = fixture_lcb()
    catalog = Catalog(entries=(CatalogEntry("cheap/model", 0.05, 0.20, 128_000),))
    worker = MockWorker(answers={"cheap/model": GOOD_PROGRAM}, costs={"cheap/model": 0.01})
    scaffold = Scaffold(catalog, MockConductor(("DELEGATE cheap/model", "STOP")), worker,
                        grader(bench))
    episode = scaffold.run_episode(task, budget_remaining=1.0)
    assert episode.result.graded_score == 1.0
    assert [text for _, text, _ in worker.seen] == [task.prompt]
    assert all(request.text == task.prompt for request in episode.requests)
