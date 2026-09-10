"""R2E-Gym V1 — DeepSWE's own training environment, and the adapter that moves N.

WHY THIS ONE WAS WRITTEN. The corpus was N = 2 runnable (the spend plan §1.1), and at N = 2
three parts of the mechanism break: leave-one-out (§5.2 condition 2) collapses to "win each benchmark
separately", the median (condition 3, D16) is arithmetically the aggregate, and a ~250-task slice
needs 125 per stratum where LiveCodeBench has 112 tasks in total, so "drawn after commits" stops
being a draw. This is the third benchmark, and it is the one that is not task-starved.

**THE ANSWER TO THE QUESTION THIS ADAPTER WAS WRITTEN TO ASK — does R2E-Gym package its environments
differently from SWE-bench? — IS: NO IN KIND, AND THE DISK MARGIN IS ~2.4x, NOT ~10x.** It is the
same shape (one prebuilt image per task, `namanjain12/<repo>_final:<commit>` on Docker Hub,
`--pull never`, an operator provisioning step), so it is disk-bound in exactly the way SWE-bench
Verified is. What changed is every constant in that bound:

| | SWE-bench Verified | **R2E-Gym V1** |
|---|---|---|
| rows | 500 | 8101 |
| gradeable | **7** (disk) | **6812** (of 8101; disk then caps what a window can use) |
| image, on disk | 3.59-7.26 GB, mean 5.24 (n=7) | **min 1.28 / median 1.88 / mean 2.15 / max 6.82** (n=25) |
| images per ~70 GB | ~13 more | **~32** |
| grading wall clock | minutes | **1.9 s unpatched, 2.4 s with the gold patch** (sympy, 13 tests) |
| environment pin | `:latest` — not a pin | **the tag is the task's own commit** |

**THE DISK ROWS WERE WRONG BY ~3.6x AND THE TABLE WAS COMPARING TWO DIFFERENT MEASUREMENTS.** They
read *"0.29 / 0.37 / 0.48 measured"* and *"~115 images per 70 GB"*, which is
`docker image inspect --format '{{.Size}}'` — the image's *content* size, not the footprint it
occupies. The SWE-bench column was never in that convention, so the ten-fold margin was an artefact
of reading the two columns with two different commands. Re-measured 2026-09-01 on the 25 images
provisioned for the one-shot run, with `docker image ls` for BOTH columns: the ratio between the two
commands is 3.58x on R2E-Gym's 25 and 3.64x on SWE-bench Verified's 7, so correcting one column
without the other is what produced the error. Consequence is provisioning arithmetic, not the pin
argument below: a 250-task slice is ~540 GB, not the ~152 GB the old figure implies.

That last row is not a footnote. §6.1.1 requires a pinned corpus and `real.py` records that SWE-bench
pins its dataset revision while its images float on `:latest`; here the tag IS the commit hash, one
image per task, 8101 distinct, so the environment a score was computed in can be named exactly.

WHAT IS ACTUALLY GRADEABLE, AND WHAT EACH FILTER COSTS (`r2e_rows`, measured over all 8101 rows):

  * **1192 rows carry no problem statement at all** — every `matplotlib` (808) and every `moto` (384)
    row. There is nothing to ask a worker, so the two repositories leave the corpus entirely and the
    13 repositories become 11.
  * **97 more fail their own self-consistency check**: the shipped parser, run over the dataset's own
    `new_commit_res_stdout`, does not reproduce the dataset's own `expected_output_json`. The
    fail-to-pass set below is derived from those two logs, so a row where they disagree is a row
    whose granularity would be derived from something already known to be wrong.
  * That leaves **6812**, every one of which has a non-empty fail-to-pass set.

PARTIAL CREDIT IS THE POINT OF THIS ADAPTER AND IT ONLY HALF-DELIVERS. THE NUMBER IS 1.88, NOT 141.
Each task ships a status map averaging 141 tests (median 49), which looks like the richest
granularity in the corpus — and scoring the fraction of THOSE is the same mistake `real.swe_grade`
documents, because the tests are one pytest module and most of them already pass before the fix. The
honest denominator is the fail-to-pass set, and measured over the 6812 usable tasks it is **median 1,
mean 1.88, max 268 — binary on 72.1% (4909) and graded on 27.9% (1903)**. That is barely better than
SWE-bench Verified's 69% binary, so §5.4's task-count multiplier does NOT arrive here; what arrives
is 6812 tasks instead of 7, which is the other multiplier and the larger one.

THE GRADER IS REIMPLEMENTED, AND §3's "THE SCAFFOLD CANCELS" ARGUMENT IS WEAKENED ACCORDINGLY.
`r2egym` is not on PyPI and cannot be installed here: its `pyproject.toml` pins `swebench==3.0.2`,
while SWE-bench Verified's adapter grades through `swebench` 4.1.0, so installing R2E-Gym's harness
would silently repoint the other adapter's shipped grader at a different major version.
`r2e_parse_log` is therefore a transcription of
`r2egym.repo_analysis.execution_log_parser.parse_log_pytest` — 35 lines, one function, the same one
for all 13 repositories — and `r2e_grade` refines rather than reproduces
`DockerRuntime._calculate_reward_r2e`. Both deviations are recorded in `R2E_PROVENANCE`
and the refinement is exact: `Grade.score == 1.0` **iff** the shipped reward is 1.0 (see `r2e_grade`).

THE TEST DIRECTORY IS NOT INSIDE THE REPOSITORY, AND THAT IS WHAT MAKES THIS SAFE TO GRADE.
`/r2e_tests` sits at the filesystem root, `/testbed` is the git repository, and the patch is
applied with `git apply` inside `/testbed` before the tests are linked in (`_r2e_runner`). A patch
that edits its own tests to pass is the same class of attack as a grader container with a network,
and it is the reason `run_evaluation` re-applies the gold test patch in SWE-bench's harness; here the
tests are simply out of reach.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable, Iterator

from ..tools import PROTOCOL
from ..types import TOOL_NAMES, Environment, TaskSpec
from . import sandbox
from .base import Grade
from .real import (
    GRADER_REIMPLEMENTED,
    REDISTRIBUTION_PERMITTED,
    BenchmarkUnavailable,
    Provenance,
    _hf_file,
    extract_patch,
)

R2E_NAME = "r2egym"
R2E_REPO = "R2E-Gym/R2E-Gym-V1"
R2E_REVISION = "903d405799ac435061c41e72260c81ca5100f964"

# All thirteen shards, because the corpus is the whole split: the tasks are not ordered by repository
# and a subset of the files is a subset of the repositories. 1.6 GB in the HuggingFace cache, fetched
# once; the disk that actually binds is the images (see the module docstring), not this.
R2E_FILES = tuple(f"data/train-{i:05d}-of-00013.parquet" for i in range(13))

# The four columns a task needs, out of thirteen. The projection is not tidiness: the split is 6.1 GB
# uncompressed and `parsed_commit_content` (222 KB/row, the full before-and-after text of every file
# the commit touched) plus `prompt` (22 KB/row, the issue-writing prompt given to the model that
# GENERATED the problem statement) are most of it, and `real._parquet_rows` reads every column. Both
# are unused here, and `prompt` is one a careless adapter would mistake for the task text: it contains
# the gold patch.
R2E_COLUMNS = ("repo_name", "docker_image", "problem_statement", "expected_output_json",
               "execution_result_content")

# Rows decoded at a time. Measured peak RSS for a full `load()`: 2003 MB whole-shard, 1074 MB at 64,
# 1006 MB at 16 — so the win is nearly all in leaving whole-shard behind, and 64 is where it flattens.
# It is worth a constant because the largest `execution_result_content` runs to megabytes (two full
# pytest logs plus a build transcript) and the validator loads this on the box that also runs the
# graded containers.
R2E_BATCH_ROWS = 64

R2E_USABLE_TASKS = 6812

# Where the image puts things — identical in the three repositories checked (sympy, coveragepy,
# scrapy): the git repository at `/testbed`, the graded tests at `/r2e_tests` OUTSIDE it, and the
# shipped runner at `/testbed/run_tests.sh` containing one line, byte-identical across all three:
# `PYTHONWARNINGS=... .venv/bin/python -W ignore -m pytest -rA r2e_tests`. It is invoked rather than
# copied, so the pytest flags stay the benchmark's; `-rA` is what produces the summary block
# `r2e_parse_log` reads, and an adapter that wrote its own command could drop it and silently grade
# every task as unparseable.
R2E_REPO_PATH = "/testbed"
R2E_TESTS_PATH = "/r2e_tests"
R2E_RUNNER = "run_tests.sh"

# Ours, not the benchmark's: `r2egym`'s agent loop never applies a patch, so there is no shipped
# marker to import the way `real._swe_runner` imports `APPLY_PATCH_FAIL`. It separates a malformed
# patch (the worker's output, scored 0) from a container that produced nothing (the validator's
# problem, excluded from both arms) — §8b.3.
R2E_APPLY_FAIL = "R2E_APPLY_PATCH_FAIL"

# A repository test suite: same host-protection bounds as SWE-bench Verified, for the same reason.
# Generous on purpose — breaching one yields no verdict, which is an exclusion rather than a zero.
# The clock is far off the measured cost (2.4 s for a 13-test task) because the corpus reaches 4087
# tests on one task and pandas suites are minutes, not seconds.
R2E_MEMORY = "8g"
R2E_PIDS = 2048
R2E_CPUS = "4"
R2E_TIMEOUT = 1800.0

# The shipped parser's own landmarks. `PASSED` is also the status the fail-to-pass set is defined by.
R2E_SUMMARY_MARKER = "short test summary info"
R2E_PASSED = "PASSED"
R2E_STATUSES = (R2E_PASSED, "FAILED", "ERROR")

# `DockerRuntime.run_tests` strips exactly this from the container's output before parsing, and
# `decolor_dict_keys` strips the colour half again from the expected map's KEYS — the dataset ships
# some of them with escape codes baked in. Both are applied here, to both sides, by `r2e_key`.
_ANSI = re.compile(r"\x1b\[[0-9;]*m|\r")

R2E_PROVENANCE = Provenance(
    repo=R2E_REPO, revision=R2E_REVISION,
    licence="apache-2.0, declared on the dataset card",
    # The one of the four adapters whose licence question has an actual answer rather than an
    # inference. THE PERMISSION IS SCOPED, AND THE SCOPE IS THE POINT: what a window would commit and
    # a reveal would publish is the task id, the generated problem statement, the image name and two
    # lists of test names, all of which are the dataset's own Apache-2.0 content. It does NOT extend
    # to (a) the grading environments, which are one third party's Docker Hub account under no
    # declared licence — an image cannot be mirrored to the owner's R2, so §6.2's "self-contained
    # window" holds for the tasks and not for the environments, exactly as it fails to hold for
    # SWE-bench; or (b) `parsed_commit_content`, which embeds upstream repository source verbatim,
    # including GPL-3.0 orange3 under an Apache-2.0 label. This adapter never reads that column
    # (`R2E_COLUMNS`), which is what keeps the permission clean.
    redistribution=REDISTRIBUTION_PERMITTED,
    grader=GRADER_REIMPLEMENTED,
    grader_source="`r2e_parse_log` transcribes `r2egym.repo_analysis.execution_log_parser."
                  "parse_log_pytest`; `r2e_grade` refines `DockerRuntime._calculate_reward_r2e`. The "
                  "shipped package is NOT importable here: it is absent from PyPI and pins "
                  "`swebench==3.0.2`, which would repoint SWE-bench Verified's as-shipped grader "
                  "(4.1.0) at a different major version. The test commands, the runner script and "
                  "the expected statuses are all still the benchmark's own — what is reimplemented "
                  "is the log parsing and the arithmetic over it.",
    deviation="the shipped reward is BINARY: 1.0 iff the observed status map equals "
              "`expected_output_json` exactly, 0.0 otherwise. This reports the fail-to-pass fraction "
              "with the rest of the map as a gate, which scores 1.0 on exactly the same submissions "
              "and is finer only in between — see `r2e_grade`.",
    granularity="1/|fail-to-pass|, where fail-to-pass is derived from the dataset's own pre-fix log. "
                "Measured over the 6812 usable tasks: median 1, mean 1.88, max 268 — BINARY on 72.1% "
                "(4909) and graded on 27.9% (1903). The 141-test status map is not the denominator "
                "and must not become one (see the module docstring)",
    usable_tasks=R2E_USABLE_TASKS)


def _projected_rows(path: pathlib.Path, columns: tuple[str, ...]) -> Iterator[dict]:
    """`columns` of a parquet file, a batch at a time. `R2E_COLUMNS` and `R2E_BATCH_ROWS` say why.

    A generator rather than a list because both halves of the memory cost are avoidable: the columns
    nobody reads are never decoded, and the ones that are decoded are released as soon as `r2e_rows`
    has reduced them to a record.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        raise BenchmarkUnavailable("pyarrow is required to read a parquet task set "
                                   "(`uv pip install -e '.[benchmarks]'`)") from exc
    for batch in pq.ParquetFile(path).iter_batches(batch_size=R2E_BATCH_ROWS,
                                                   columns=list(columns)):
        yield from batch.to_pylist()


def r2e_issue(problem_statement: str) -> str:
    """The benchmark's own task instruction: the body of the `[ISSUE]` block, or the whole statement.

    `DockerRuntime.get_task_instruction` verbatim, including its lack of a `.strip()` — this is the
    text §2.1 forwards byte for byte, and trimming it here would make a v3 episode ask a different
    question from the one R2E-Gym's own agent is asked. 6849 of the 8101 rows carry the tags; the
    fallback is the shipped one and covers the remaining 60 usable rows.
    """
    match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", problem_statement, re.DOTALL)
    return match.group(1) if match else problem_statement


def r2e_prompt(row: dict) -> str:
    """The task statement as it reaches a worker (§2.1), plus the owner's answer-format instruction.

    THE REPOSITORY NAME IS OWNER TEXT AND IT IS HERE BECAUSE THE ANSWER IS A FILE PATH, and it is
    what a worker orients from before its first read: without knowing which of the eleven
    repositories it is in, a correct diff is unguessable for a reason that has nothing to do with
    routing. `real.swe_prompt` says the same thing for the same reason.

    THE READ PROTOCOL IS APPENDED HERE RATHER THAN ADDED BY THE SCAFFOLD, and that is what keeps
    §2.1's audit sentence byte-true: `WorkerRequest.text == task.prompt` on the first turn of every
    delegate, because the protocol is INSIDE `task.prompt`. It therefore also enters the on-chain
    window commitment with the task (`window.py` hashes `t.prompt`), so what a worker was shown is
    pinned with the corpus rather than living in a module the owner could edit mid-flight. One
    constant, from `tools.py`, exactly as this line's own answer-format instruction is one string.

    NO COMMIT IS NAMED, WHICH IS WHERE THIS DIFFERS FROM `swe_prompt`. The image's tag is the commit
    that FIXED the issue and the checkout is its parent with the fix removed, so quoting either would
    point a worker at a tree it is not looking at.
    """
    return (f"Repository: {row['repo_name']}\n\n"
            f"# Problem\n{r2e_issue(row['problem_statement'])}\n\n"
            "Return ONLY a unified git diff (patch) that resolves the problem, rooted at the "
            "repository root so it applies with `git apply -p1`. Do not include prose outside the "
            "diff." + PROTOCOL)


def r2e_key(name: str) -> str:
    """One test's name, normalised the way the shipped reward normalises BOTH sides before comparing.

    Two steps, both `_calculate_reward_r2e`'s: strip the ANSI escapes some expected keys ship with,
    then drop pytest's ` - <reason>` suffix. Applied to the observed map and the expected map by the
    same function on purpose — the comparison is only meaningful if the two were normalised
    identically, and two functions is two chances for them to drift apart.
    """
    return _ANSI.sub("", name).split(" - ")[0]


def r2e_parse_log(log: str) -> dict[str, str]:
    """`{test name: PASSED|FAILED|ERROR}` from a pytest `-rA` summary block.

    A transcription of the shipped `parse_log_pytest` (see `R2E_PROVENANCE.grader_source` for why it
    cannot be imported), including the two behaviours that look like bugs and are load-bearing:

      * **Everything before `short test summary info` is discarded**, so the failure tracebacks — and
        our own `set -x` trace, and the container's noise — cannot be mistaken for verdicts.
      * **A summary line with no `::` yields the empty key**, which is what a collection error looks
        like (`ERROR r2e_tests/test_1.py - ImportError`). The shipped reward turns that into a zero
        through its length check; `r2e_grade` turns it into a zero through its key-set check. Dropping
        it here instead would let a patch that stops a test from being COLLECTED score as if the test
        had never existed.

    Returns `{}` when there is no summary block at all, which is the discrimination `grade` needs:
    a container that never reached pytest is the validator's failure, not the miner's (§8b.3).
    """
    text = _ANSI.sub("", log or "")
    if R2E_SUMMARY_MARKER not in text:
        return {}
    statuses = {}
    for line in text.split(R2E_SUMMARY_MARKER)[1].strip().splitlines():
        for status in R2E_STATUSES:
            if status in line:
                statuses[r2e_key(".".join(line.split("::")[1:]))] = status
                break
    return statuses


def r2e_fail_to_pass(pre_fix_log: str, expected: dict[str, str]) -> tuple[str, ...]:
    """The tests the issue is about: expected to pass, and not passing before the fix.

    THIS IS DERIVED, NOT SHIPPED, AND IT IS DERIVED FROM THE BENCHMARK'S OWN EVIDENCE. R2E-Gym labels
    no fail-to-pass set — its reward is all-or-nothing over the whole map — but every row carries the
    log of that map being produced at the parent commit (`execution_result_content.old_commit_res_-
    stdout`), so the split is a fact about the dataset rather than a judgement about the task. It is
    what makes a graded score possible at all: without it the only honest denominator is 1.
    """
    pre_fix = r2e_parse_log(pre_fix_log)
    return tuple(name for name, status in expected.items()
                 if status == R2E_PASSED and pre_fix.get(name) != R2E_PASSED)


def r2e_grade(observed: dict[str, str], expected: dict[str, str],
              fail_to_pass: tuple[str, ...]) -> Grade:
    """The observed status map, scored against the expected one — a refinement of the shipped reward.

    THE REFINEMENT IS EXACT, AND THAT IS THE WHOLE JUSTIFICATION FOR NOT USING THE SHIPPED REWARD.
    `_calculate_reward_r2e` returns 1.0 iff the two maps have the same length and agree on every key.
    Here the same-key-set check and the gate together say precisely that, so `score == 1.0` holds on
    exactly the submissions the shipped grader would have scored 1.0 — and on the 27.9% of tasks with
    more than one fail-to-pass test, a submission that fixes some of them is now distinguishable from
    one that fixes none, which under the shipped reward it is not.

    THE REST OF THE MAP IS A GATE AND NOT A SUMMAND, for the reason `real.swe_grade` measured on
    SWE-bench and which is worse here: the median task ships 45 already-passing tests against a median
    of ONE fail-to-pass, so scoring the fraction of all of them would hand a do-nothing patch ~0.98,
    pin every task near 1.0, and leave §6.1.3's admission probe measuring rounding error.

    Expected-to-FAIL tests are inside the gate, not outside it (11.6% of all statuses in this corpus
    are FAILED or ERROR). A patch that incidentally fixes one scores zero — which is harsh, and is
    what the shipped reward does, and the alternative is a score of 1.0 where the benchmark's own
    grader says 0.0.
    """
    total = len(fail_to_pass)
    if set(observed) != set(expected):
        return Grade(0, total)
    gate = set(expected) - set(fail_to_pass)
    if any(observed[name] != expected[name] for name in gate):
        return Grade(0, total)
    return Grade(sum(observed[name] == R2E_PASSED for name in fail_to_pass), total)


def _r2e_runner() -> str:
    """Apply the worker's patch, link the tests in, run the shipped runner. The ORDER is the grader.

    `git apply` and no `patch --fuzz` fallback, which is where this deliberately departs from
    SWE-bench's shipped apply sequence: `patch` accepts `../` in a target path and `git apply` refuses
    to leave the repository or to write through a symlink. `/r2e_tests` is outside `/testbed`, so with
    `git apply` alone the tests are unreachable from a patch; with the fallback they would not be. The
    cost is that a patch needing fuzz scores zero, which is a miner's loss on a malformed answer
    rather than a miner's win on an unsolved task.

    The `rm -rf` before the symlink closes the other half: a patch is free to CREATE `r2e_tests/` as
    an ordinary directory inside the repository, and if it survived, `run_tests.sh` — whose argument
    is the relative path `r2e_tests` — would run the worker's own tests and grade the worker against
    them.
    """
    return ("#!/bin/bash\n"
            "set -uxo pipefail\n"
            f"cd {R2E_REPO_PATH}\n"
            f"git apply -v {sandbox.MOUNT}/patch.diff || git apply -v -p0 {sandbox.MOUNT}/patch.diff "
            f"|| {{ echo '{R2E_APPLY_FAIL}'; exit 1; }}\n"
            f"rm -rf {R2E_REPO_PATH}/r2e_tests\n"
            f"ln -s {R2E_TESTS_PATH} {R2E_REPO_PATH}/r2e_tests\n"
            f"bash {R2E_RUNNER}\n")


def r2e_rows(rows: Iterable[dict]) -> dict[str, dict]:
    """The gradeable subset, keyed by stable task ID, carrying only what a score needs.

    THE ROW IS REBUILT RATHER THAN ANNOTATED, because `execution_result_content` is ~200 KB per row
    (two full pytest logs and a build transcript) and holding 6812 of them would cost more than a
    gigabyte of the validator's memory to keep evidence that has already been reduced to a tuple of
    test names.

    The ID is `<name>-<repo>-<commit>` and the commit is the image's tag, which is unique across all
    8101 rows — so it survives a re-fetch, which a row index would not (§6.2: the slice is drawn by
    ID and the reveal publishes it).

    Three filters, and the module docstring records what each one costs. All three drop rows that
    cannot be scored honestly rather than rows that are inconvenient.
    """
    usable = {}
    for row in rows:
        if not (row["problem_statement"] or "").strip():
            continue
        execution = json.loads(row["execution_result_content"])
        expected = {r2e_key(name): status
                    for name, status in json.loads(row["expected_output_json"]).items()}
        # The dataset's own post-fix log, parsed by the same parser, must reproduce the dataset's own
        # expected map — otherwise the fail-to-pass set below is derived from a disagreement.
        if r2e_parse_log(execution.get("new_commit_res_stdout") or "") != expected:
            continue
        fail_to_pass = r2e_fail_to_pass(execution.get("old_commit_res_stdout") or "", expected)
        if not fail_to_pass:
            continue
        image = row["docker_image"]
        usable[f"{R2E_NAME}-{row['repo_name']}-{image.rsplit(':', 1)[-1]}"] = {
            "repo_name": row["repo_name"], "docker_image": image,
            "problem_statement": row["problem_statement"],
            "expected": expected, "fail_to_pass": fail_to_pass}
    return usable


class R2EGym:
    """6812 executable repository tasks, graded by running the commit's own tests in its own image.

    The benchmark DeepSWE trains against, and the reason the corpus can be stratified at all: it is
    the only admitted benchmark with more tasks than a window can use, so §6.3b's "drawn after
    commits" is a real draw here rather than an enumeration.

    ASKED AS ONE COMPLETION IT MEASURED A DEAD STRATUM, AND THAT IS WHY IT NOW HAS A READ CHANNEL.
    the sandbox record §7: 3 models over 25 tasks, every model 0.0000, routable band +0.0000, and 0
    of 75 episodes produced an appliable patch — against a gold-patch control of 24 of 25 at exactly
    1.0 through this same `grade`. The failure was at the apply step, not the reasoning step, so
    `tools()` declares the three read-only verbs and `environment` roots them at the repository. The
    open question is unchanged and is NOT answered by this adapter: the arena needs a between-model
    SPREAD, not a solve rate, and §6.1.3's admission probe is what decides. This adapter's job is to
    make that probe cheap enough to run — 2.4 s and no dollars per grade.
    """

    name = R2E_NAME
    provenance = R2E_PROVENANCE

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        """The tasks, each carrying its REPOSITORY as `TaskSpec.group`.

        THE ONLY ADMITTED-CORPUS BENCHMARK THAT HAS AN HONEST GROUPING, and the label is the
        dataset's own `repo_name` column rather than anything this adapter inferred. `duel` resamples
        whole groups: 6812 tasks over eleven repositories means an 83-task stratum holds ~7.5 of
        each, a challenger that suits a repository is better on all of them at once, and an iid
        bootstrap publishes a lower bound too narrow for that (see `duel.py`). LiveCodeBench and HLE
        pass no group, which is the every-task-its-own-cluster default and the resample that shipped.
        """
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=r2e_prompt(row),
                              tools=self.tools(), group=row["repo_name"])
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        """The three read-only verbs (`tools.py`) — THE ONE ADAPTER THAT DECLARES ANY, AND WHY.

        MEASURED, and it is the whole reason the channel was built: 3 models x 25 of these tasks,
        every model 0.0000, routable band **+0.0000**, and **0 of 75 episodes produced a patch
        `git apply` would accept** — while the benchmark's own fix scored exactly 1.0 on **24 of 25**
        of those same tasks through this adapter's own `grade` (the sandbox record §7.1, §7.2). So
        the harness is sound and the failure is the ACTION SPACE: every answer was a well-formed
        diff with an invented blob hash and invented context lines, necessarily invented because the
        worker had never seen the file, and `git apply` matches context.

        It is this adapter and no other because the reads must be reproducible: R2E-Gym's image tag
        IS the task's own commit, whereas both SWE-bench variants float on `:latest`, which is not a
        pin and so not a fixed thing to read (`real.py`).

        WHAT THIS DOES NOT CLAIM. The arena needs a BETWEEN-MODEL spread, not a high solve rate: a
        channel that moves all three models to 0.4 together still measures a band of +0.0000.
        Whether apply rates differ by model is NOT MEASURED, and §6.1.3's admission probe — the same
        harness plus the gold control, about $0.50 — is what decides whether this stratum is
        admitted. Registration is not admission.
        """
        return TOOL_NAMES

    def environment(self, task: TaskSpec) -> Environment:
        """The task's own image, rooted at the repository — WHICH IS THE CONFINEMENT (`tools.py`).

        `/r2e_tests` sits OUTSIDE `/testbed`, and the module docstring records that this is exactly
        what makes the benchmark safe to grade. Rooting the read channel at the repository is the
        same property applied to the other door: a miner who can drive a worker's reads still cannot
        reach the file their patch will be graded against.

        THE OTHER DOOR IS INSIDE THE ROOT: `/testbed` IS THE GIT REPOSITORY, so its history is within
        the boundary, and a clone that kept its upstream refs holds the very commit the task asks the
        worker to re-derive. This adapter cannot verify what a third party's image contains, so
        `tools.HIDDEN` excludes `.git` from the channel rather than trusting the build. Grading is
        untouched — it runs in its own container from the same image and never goes through the
        reader — and no source file becomes unreadable.
        """
        return Environment(image=self.image_for(task.task_id), root=R2E_REPO_PATH)

    def repo_of(self, task_id: str) -> str:
        """Which of the eleven repositories a task is in, for the stratified draw (§6.3b).

        Exposed for the same reason `LiveCodeBench.difficulty_of` is: the usable pool is 34% sympy and
        21% pandas, so a uniform draw is mostly two libraries. It answers by task ID, from the corpus
        rather than from a loaded slice, which is what a draw needs; `load` puts the same label on
        each `TaskSpec.group`, which is what the duel's resample needs.
        """
        return self._loaded()[task_id]["repo_name"]

    def image_for(self, task_id: str) -> str:
        """The published image for one task — read from the row, not built.

        No package computes this name (contrast `real.SweBenchVerified.image_for`, which must borrow
        swebench's key property): the dataset ships `docker_image` per row, so the census below costs
        one local `docker image inspect` per task and no network at all.
        """
        return self._loaded()[task_id]["docker_image"]

    def usable_ids(self) -> tuple[str, ...]:
        """The task IDs whose image is present locally, so a window can be built from them.

        `sandbox.run` passes `--pull never`, so an absent image is a refusal at grading time — the
        right failure, because pulling 2.15 GB inside a graded episode spends the arm's wall clock on
        the validator's housekeeping (§8b.2), but only a kind one if it can be checked beforehand.
        This is the number that binds in practice: ~32 images fit in the 70 GB the box has free,
        against a corpus of 6812 (see the module docstring — the ~115 this used to claim was the
        `docker image inspect` content size, not the footprint).

        BATCHED, and on this corpus that is the difference between a census and a bill: asked one
        image at a time these 6812 tasks are ~13,600 docker round trips, which over the deployed ssh
        transport is 2000-4000 s before a window opens (`sandbox.images_present`).
        """
        images = {task_id: self.image_for(task_id) for task_id in self._loaded()}
        present = sandbox.images_present(images.values())
        return tuple(task_id for task_id, image in images.items() if image in present)

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Apply the patch in the task's own image, run the shipped runner, score its summary."""
        row = self._loaded()[task.task_id]
        fail_to_pass = row["fail_to_pass"]
        patch = extract_patch(submission)
        if not patch.strip():
            return Grade(0, len(fail_to_pass))    # an empty answer is a failure, not an error
        with sandbox.workspace() as (work, logs):
            payload = work / "payload"
            payload.mkdir()
            (payload / "patch.diff").write_text(patch)
            (payload / "run.sh").write_text(_r2e_runner())
            result = sandbox.run(row["docker_image"], ["/bin/bash", f"{sandbox.MOUNT}/run.sh"],
                                 mount=payload, log_path=logs / "sandbox.log", timeout=R2E_TIMEOUT,
                                 memory=R2E_MEMORY, pids=R2E_PIDS, cpus=R2E_CPUS)
            log = result.log_path.read_text(errors="replace")
        # THE APPLY MARKER IS READ FIRST, AND THE ORDER IS THE POINT — the same discrimination
        # `real.SweBenchVerified.grade` draws. A patch that will not apply produces no summary block
        # and so does a container that died before pytest started, but §8b.3 puts those on opposite
        # sides of the line: the first is the worker's output and scores 0, the second is the
        # validator's problem and leaves both arms.
        if R2E_APPLY_FAIL in log:
            return Grade(0, len(fail_to_pass))
        observed = r2e_parse_log(log)
        if not observed:
            raise sandbox.SandboxError(
                f"no pytest summary block for {task.task_id}; the patch did apply, so this is the "
                f"grader rather than the submission: {log.strip()[-400:]}")
        return r2e_grade(observed, row["expected"], fail_to_pass)

    def _loaded(self) -> dict[str, dict]:
        if self._rows is None:
            rows: dict[str, dict] = {}
            # A shard at a time, and a batch at a time within one, so that none of the 6.1 GB split
            # is held longer than it takes `r2e_rows` to reduce it to a record.
            for filename in R2E_FILES:
                path = _hf_file(R2E_REPO, filename, R2E_REVISION)
                rows |= r2e_rows(_projected_rows(path, R2E_COLUMNS))
            self._rows = rows
        return self._rows
