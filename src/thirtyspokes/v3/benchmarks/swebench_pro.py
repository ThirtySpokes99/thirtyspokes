"""SWE-bench Pro — 731 enterprise issues, graded by the benchmark's own per-instance scripts.

WHY THIS ADAPTER EXISTS AND WHAT IT DOES NOT FIX. The corpus is N = 2 runnable, and N = 2 breaks the
mechanism in three measured places (the spend plan §1.2): leave-one-out collapses to "win each
benchmark separately", the §5.2 median is arithmetically the aggregate, and a 125-task stratum is
impossible against LiveCodeBench's 112 usable tasks. Only task COUNT moves N, so the number this
module is judged on is how many of the 731 rows can actually be graded — not how many load.

**THE ARITHMETIC, MEASURED TODAY, AND IT IS THE DELIVERABLE.** Every instance grades inside its own
published image `jefzda/sweap-images:<dockerhub_tag>`, one per row, 731 distinct tags with no reuse.
Summed from the registry's own manifest sizes: **997.5 GB compressed for the corpus**, which unpacks
to **~3.6 TB** at the ratio measured over the 23 images present here (median 3.64x, range 3.24-4.43;
`ansible-9a21e247` is 0.453 GB compressed and 1.47 GB unpacked). The volume had 79 GB free when this
work started and 71 GB when it ended, on a box shared with other projects. So the corpus is
disk-bound by a factor of ~50, exactly as SWE-bench Verified is — this adapter does not repeal that.

What it does change is the size of the number, for three reasons that are all arithmetic:

  * **23 instances are already provisioned on this box** (ansible x10, openlibrary x8,
    qutebrowser x5) and **all 23 grade their own gold patch to 1.0, in 2-18 s each**. SWE-bench
    Verified delivers 7. That is a free 3x before anything is pulled.
  * **The images are ~3x smaller.** Verified's instances measure 3.59-7.26 GB unpacked; Pro's Python
    tier measures 1.47-1.84 GB (ansible), 2.05-3.47 GB (qutebrowser), 2.74-4.75 GB (openlibrary).
  * **Instances sharing a `-v<env-sha>` tag suffix share their environment layers**, and this is the
    real lever. `docker system df -v` reports 985.6 MB SHARED against 834-858 MB UNIQUE across the
    three local `ansible ... -v390e508d` instances, while the two singleton ansible families cost
    1.38-1.50 GB each. A family of k ansible instances therefore costs about **0.99 + 0.85k GB**, not
    1.8k. The corpus has 66 such families over 731 instances, and ansible's 96 instances sit in 10.

So the answer depends entirely on whether the slice is drawn family-first. Spending ~55 GB on
ansible's six largest families buys **~60 more instances** (6 x 0.99 + 60 x 0.85 = 57 GB); spending
the same 55 GB uniformly at the corpus median (0.99 GB compressed, ~3.6 GB unpacked) buys **~15**.
Total gradeable is therefore **~80 tasks drawn cheaply and ~40 drawn representatively**, against 23
today at zero cost. That is short of §6.3b's 125 per stratum and it is 11% of the benchmark, but it
is not the single-digit answer the brief warned about.

A family-first draw buys tasks by concentrating them, and that cost is the one the spend plan §1.3
already names for SWE-bench Verified, whose 7 provisioned instances come from three repositories and
are "not a representative stratum". Sixty ansible tasks is one repository, one test runner and one
issue style; the honest middle is a few families across two or three repositories, which is what the
23 already here happen to be.

**RECOMMENDATION: INCLUDE IT.** The counter-argument
is on the record and it is serious — this repository measured SWE-bench Pro **one-shot at 0/120**
(`koth/lcb.py`), and §6.1.3's admission probe exists to drop exactly that: a benchmark pinned at the
floor cancels between the arms and contributes cost with no signal. That is a measurement to run, not
a reason to skip the adapter, because the floor is a property of the REGIME this benchmark is asked
in (one completion call, no tools, no repository access — `real.py` finding 1) rather than of this
dataset, and the same argument condemns SWE-bench Verified. The regime is now escapable — the read
channel of `v3/tools.py` is what R2E-Gym declares — but not HERE, because these instance images
float on `:latest` and a read against an unpinned image is not reproducible (see `tools`). The adapter is cheap, the census is now known, and the probe
can be run against 23 provisioned tasks for the price of the completions.

**RUN 2026-09-07, AND THE FLOOR CANCELLED EXACTLY AS §6.1.3 PREDICTS.** The spread smoke drew 21 of
the provisioned tasks and ran the three fixed policies under the King₀ pair (floor qwen3-30b, top
claude-haiku-4.5, cascade via qwen3-coder): 63 episodes, 63 zeros, $0.68; every episode was one
`DELEGATE` then `STOP` with no observations, while the same images grade their gold patch to 1.0.
λ fits to 0.0 and DOMINATOR fires in 100% of resamples, so the table excludes the benchmark and the
launch corpus stays at N = 2. The adapter stays; the benchmark waits for a read channel against pinned
images. Record: the M3a record §10, `runs/swep-probe/`.

TWO FINDINGS THAT ARE NOT ABOUT DISK.

1. **`--network none` IS NOT WHAT THE OFFICIAL HARNESS RUNS.** `swe_bench_pro_eval.py` takes
   `--block_network` as an opt-in flag and defaults it OFF, and some shipped run scripts rely on
   that: the NodeBB script runs `npm install --production=false` and `npm install lodash underscore
   async` *inside the test step*. §8b/M2a make the network cut mandatory, so those instances will
   fail their setup here and produce no verdict — an exclusion, not a zero, which is the right
   outcome but also means the js/ts tier may be much smaller than its row count. The Python tier's
   scripts (ansible, qutebrowser, openlibrary) install nothing at test time and are unaffected. This
   is a per-instance property and the honest way to learn it is to run the gold patch, which is what
   `test_bench_swebench_pro.py` does for the provisioned ones.

2. **THE IMAGES ARE ONE PERSON'S DOCKER HUB ACCOUNT, AND THE TAGS ARE MUTABLE.** `jefzda` is a
   personal namespace, not a Scale org one, and nothing here pins a digest. §6.1.1 pins the dataset
   revision and the script commit; the environment the tests run in is pinned only by convention.
   That is the same hole `real.py` records for SWE-bench Verified's `:latest` tags, one namespace
   less durable.
"""

from __future__ import annotations

import ast
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request

from ..types import TaskSpec
from . import sandbox
from .base import Grade
from .real import (
    GRADER_AS_SHIPPED,
    REDISTRIBUTION_UNRESOLVED,
    BenchmarkUnavailable,
    Provenance,
    _hf_file,
    _parquet_rows,
    extract_patch,
)

SWEP_NAME = "swebench-pro"
SWEP_REPO = "ScaleAI/SWE-bench_Pro"
SWEP_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
SWEP_FILE = "data/test-00000-of-00001.parquet"
SWEP_USABLE_TASKS = 731

# The published per-instance images. `dockerhub_tag` is a dataset column, so the image name needs no
# construction — the benchmark's own README says to read it, which is why `helper_code/image_uri.py`
# (with its two hard-coded element-web special cases) is not reimplemented here.
SWEP_REGISTRY = "jefzda/sweap-images"

# The grader is a GitHub repository, not a PyPI package: `run_scripts/<instance_id>/run_script.sh`
# and `parser.py` are per-instance and there is nothing to `pip install`. Pinned to a commit for the
# same reason `Provenance.revision` is (§6.1.1) — the run scripts have been revised twice since
# release (the repo's own news entries for 1/7 and 2/9), so an unpinned fetch reprices the corpus.
SWEP_SCRIPTS_REPO = "scaleapi/SWE-bench_Pro-os"
SWEP_SCRIPTS_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
SWEP_SCRIPTS_CACHE = pathlib.Path.home() / ".cache" / "thirtyspokes" / "swebench-pro"

# Where the shipped `parser.py` writes its verdict, echoed to the container's stdout after this line.
# The official runner has it write into a read-WRITE /workspace mount; `sandbox.run` mounts read-only
# so that a graded program cannot edit the tests it is scored against, so the file goes to /tmp and
# comes back through the log. The marker is what tells a failed submission from a container that
# never ran (§8b.3) — the parser writing nothing is the grader's failure, not the miner's.
SWEP_VERDICT = "V3_SWEP_REPORT"

# A repository test suite in a distribution image: minutes, not seconds. The official harness gives
# each instance 3600 s and up to 30 GB on Modal; these are the sibling adapter's HOST-PROTECTION
# bounds, and breaching one produces no verdict and therefore an exclusion, never a score (§8b.3).
SWEP_MEMORY = "8g"
SWEP_PIDS = 4096
SWEP_CPUS = "4"
SWEP_TIMEOUT = 2700.0

SWEP_PROVENANCE = Provenance(
    repo=SWEP_REPO, revision=SWEP_REVISION,
    licence="none declared on the dataset card. The eval repository (scaleapi/SWE-bench_Pro-os), "
            "which is where the grader lives, is MIT. The eleven source repositories are unusually "
            "copyleft-heavy for a benchmark — ansible, qutebrowser, navidrome, tutanota, vuls and "
            "webclients are GPL-3.0, openlibrary and element-web and teleport are AGPL-3.0 — so the "
            "patches and test files carry those terms, and the images are a third party's Docker "
            "Hub account with no licence stated at all",
    redistribution=REDISTRIBUTION_UNRESOLVED,
    grader=GRADER_AS_SHIPPED,
    grader_source="the benchmark's own per-instance `run_script.sh` (which tests run, and how) and "
                  "`parser.py` (which of them passed), fetched verbatim from "
                  f"{SWEP_SCRIPTS_REPO}@{SWEP_SCRIPTS_COMMIT} and executed unmodified. The entry "
                  "script around them is transcribed from `swe_bench_pro_eval.create_entryscript` "
                  "rather than imported, because that repository is not packaged — so unlike "
                  "SWE-bench Verified, where `swebench` 4.1.0 is a dependency, a harness change "
                  "upstream does not reach this code by itself.",
    deviation="the published metric is `(fail_to_pass | pass_to_pass) <= passed` — one bit. This "
              "reports the fail_to_pass fraction with pass_to_pass as a gate, identical to the "
              "published metric at 0, at 1, and whenever |fail_to_pass| == 1, and finer in between. "
              "`strip_binary_hunks` is not transcribed: a one-shot text worker does not emit binary "
              "diff hunks, and a patch that carries one fails to apply and scores 0 as it should.",
    granularity="1/|fail_to_pass|. Measured at this revision: median 3, mean 14.4, max 1869, and "
                "only 209 of 731 tasks are binary — against 345 of 500 on SWE-bench Verified. This "
                "is the corpus's finest-grained agentic benchmark, which is what §5.4's task counts "
                "are most sensitive to",
    usable_tasks=SWEP_USABLE_TASKS)


def _script(path: str) -> str:
    """One file of the shipped grader, at the pinned commit, from the cache when it is already there.

    Cached on disk rather than re-fetched because §8b bans network access from the grading container
    and M2a's tests must pass offline once the data is cached: the fetch belongs to provisioning, in
    the same place `usable_ids` belongs, and never inside a graded episode.
    """
    cached = SWEP_SCRIPTS_CACHE / SWEP_SCRIPTS_COMMIT / path
    if cached.exists():
        return cached.read_text()
    url = (f"https://raw.githubusercontent.com/{SWEP_SCRIPTS_REPO}/{SWEP_SCRIPTS_COMMIT}/"
           f"{urllib.parse.quote(path)}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read().decode()
    except (urllib.error.URLError, OSError) as exc:
        raise BenchmarkUnavailable(f"could not fetch the shipped grader file {path} from "
                                   f"{SWEP_SCRIPTS_REPO}@{SWEP_SCRIPTS_COMMIT}: {exc}") from exc
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(body)
    return body


def swep_list(value) -> list[str]:
    """One of the dataset's list-shaped string columns, decoded.

    The parquet stores `fail_to_pass`, `pass_to_pass` and `selected_test_files_to_run` as the repr of
    a Python list, not as JSON — the test names contain apostrophes, so single and double quotes are
    mixed within one column and `json.loads` fails on about half the rows. The official evaluator
    calls `eval()` on these; `ast.literal_eval` is the same decoding without executing the dataset.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        return [str(item) for item in ast.literal_eval(str(value))]
    except (ValueError, SyntaxError):
        return [str(item) for item in json.loads(str(value))]


def swep_prompt(row: dict) -> str:
    """The task statement forwarded to a worker byte for byte (§2.1).

    The first three paragraphs are the benchmark's own: `helper_code/create_problem_statement.py`
    composes exactly this from `problem_statement`, `requirements` and `interface`, and it is what
    the official SWE-agent scaffold puts in front of a model. Reusing it rather than the raw
    `problem_statement` matters here — Pro's issue text is deliberately underspecified and the
    requirements block is where the acceptance criteria live, so a worker given only the first field
    is being asked a harder question than the leaderboard asks.

    The closing instruction is owner text, identical for every miner and both arms, and identical to
    the one `real.swe_prompt` uses: this benchmark declares no tools, so its delegates are ONE
    completion call with no repository access and a patch is the only artifact it can grade.
    """
    return (f"Repository: {row['repo']}\n"
            f"You are at commit {row['base_commit']}.\n\n"
            f"# Problem\n{row['problem_statement']}\n\n"
            f"Requirements:\n{row['requirements']}\n\n"
            f"New interfaces introduced:\n{row['interface']}\n\n"
            "Return ONLY a unified git diff (patch) that resolves the problem, rooted at the "
            "repository root so it applies with `git apply -p1`. Do not include prose outside the "
            "diff.")


def swep_entryscript(row: dict, base_dockerfile: str, instance_dockerfile: str) -> str:
    """The shipped entry sequence: re-export the image's ENV, reset, apply, restore tests, run, parse.

    Transcribed from `swe_bench_pro_eval.create_entryscript` line for line, and the two places it
    reads oddly are both upstream's:

      * only the LAST line of `before_repo_set_cmd` is used, because the earlier lines are the reset
        and the base checkout that this script has already done — what remains is the
        `git checkout <fix commit> -- <test files>` that restores the tests the patch is judged by,
        and it must run AFTER the model's patch so a patch that rewrote a test cannot score with it.
      * the ENV lines of both dockerfiles are re-exported although Docker already sets them. Kept
        because dropping it is a change to the shipped environment, and `PYTEST_ADDOPTS` — which
        carries `--continue-on-collection-errors` and `--reruns=3` — is one of the variables it
        carries, so a difference here is a difference in what the grader sees.

    Two deliberate departures from upstream, both forced by `sandbox.run` and neither touching what
    decides the score: the logs go to /tmp because the payload mount is read-only (a graded patch
    must not be able to edit the tests it is scored against), and the parser's `output.json` is
    echoed to stdout after `SWEP_VERDICT` because the read-only mount is also the only directory the
    host shares with the container.
    """
    env = "\n".join(line.strip().replace("ENV", "export", 1)
                    for dockerfile in (base_dockerfile, instance_dockerfile)
                    for line in dockerfile.split("\n") if line.strip().startswith("ENV"))
    tests = ",".join(swep_list(row["selected_test_files_to_run"]))
    return (f"#!/bin/bash\n{env}\n"
            "cd /app\n"
            f"git reset --hard {row['base_commit']}\n"
            f"git checkout {row['base_commit']}\n"
            f"git apply -v {sandbox.MOUNT}/patch.diff\n"
            f"{row['before_repo_set_cmd'].strip().splitlines()[-1]}\n"
            f"bash {sandbox.MOUNT}/run_script.sh {tests} > /tmp/stdout.log 2> /tmp/stderr.log\n"
            f"python {sandbox.MOUNT}/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json\n"
            f"echo {SWEP_VERDICT}\n"
            "cat /tmp/output.json\n")


def swep_verdict(log: str) -> list[dict]:
    """The shipped parser's own `{"tests": [...]}`, taken from the container's log.

    Raises rather than returning an empty list when the marker or the JSON is missing, and the
    distinction is §8b.3's: a patch that would not apply still reaches the parser and comes back as
    a report in which the fail_to_pass tests failed — a score. A parser that produced nothing means
    the container died before reaching it, which is the validator's problem and excludes the task
    from both arms.
    """
    if SWEP_VERDICT not in log:
        raise sandbox.SandboxError(f"the SWE-bench Pro grader produced no {SWEP_VERDICT} line, so "
                                   f"the shipped parser never ran: {log.strip()[-400:]}")
    body = log.split(SWEP_VERDICT, 1)[1].strip()
    try:
        return list(json.loads(body)["tests"])
    except (ValueError, KeyError, TypeError) as exc:
        raise sandbox.SandboxError("the SWE-bench Pro parser wrote no readable report: "
                                   f"{body[:400]}") from exc


def swep_grade(tests: list[dict], fail_to_pass: list[str], pass_to_pass: list[str]) -> Grade:
    """The shipped report, turned into a graded score — the sibling adapter's rule, for its reason.

    `swe_bench_pro_eval.main` scores `(f2p | p2p) <= passed`, one bit per task. Scoring the fraction
    of ALL tests instead is wrong here for the same measured reason it is wrong on SWE-bench Verified
    (`real.swe_grade`): the regression suite dwarfs the problem's own tests, so a do-nothing patch
    would score near 1.0 and two policies would differ by a rounding error. So pass_to_pass is a
    GATE and fail_to_pass is the score.

    The partial credit is worth more here than there. Verified is binary on 69% of its tasks; Pro's
    median instance carries 3 fail_to_pass tests and only 209 of 731 carry one, so this is where
    §5.4's task counts actually come down.
    """
    passed = {test["name"] for test in tests if test.get("status") == "PASSED"}
    wanted, regression = set(fail_to_pass), set(pass_to_pass)
    if not regression <= passed:
        return Grade(0, len(wanted))
    return Grade(len(wanted & passed), len(wanted))


def swep_rows(rows: list[dict]) -> dict[str, dict]:
    """Keyed by the dataset's own `instance_id`, in file order.

    No filtering, and that is the difference from this repository's earlier
    an earlier standalone grader (removed with v2, 2026-09-07), which admitted pytest instances only because it reimplemented the
    runner and could not parse the JS/TS convention (`<file> | <suite name>`). Every instance here
    ships its own parser, so the language is upstream's problem and all four are gradeable — what
    narrows the set is disk, and `usable_ids` is where that shows up.
    """
    return {f"{SWEP_NAME}-{row['instance_id']}": row for row in rows}


class SweBenchPro:
    """731 long-horizon issues across 11 repositories and 4 languages, graded as shipped.

    Everything that decides the score is the benchmark's: `run_script.sh` chooses and runs the tests,
    `parser.py` reads their statuses, and the pass criterion is the published one with pass_to_pass
    kept as a gate. What is ours is the container invocation, because `--network none` and the
    resource caps are §8b requirements that upstream's runner takes as an optional flag.
    """

    name = SWEP_NAME
    provenance = SWEP_PROVENANCE

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=swep_prompt(row),
                              tools=self.tools())
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        """Empty, and the cost is at its highest here: this repository measured SWE-bench Pro
        one-shot at 0/120 against a published 62-80% agentic.

        IT IS NOT THE READ CHANNEL `r2egym.py` DECLARES, AND THE REASON IS THE PIN. These instance
        images float on `:latest`, so a read against one is not reproducible across a re-pull and the
        observation a score was computed from could not be re-executed by a third party. R2E-Gym's
        tag IS the task's own commit, which is why it is the only adapter that gets the channel."""
        return ()

    def environment(self, task: TaskSpec) -> None:
        """None, exactly because `tools()` is empty (`base.Benchmark.environment`)."""
        return None

    def usable_ids(self) -> tuple[str, ...]:
        """The task IDs whose instance image is present locally — the count that moves N.

        Measured on this box today: 23 of 731, against SWE-bench Verified's 7. The rest are a disk
        problem and not a code one; see the module docstring for the arithmetic. `sandbox.run` passes
        `--pull never`, so an absent image is a refusal at grading time rather than a multi-gigabyte
        pull spending an arm's wall clock (§8b.2), and this is how a window checks before it opens.

        Batched (`sandbox.images_present`): 731 tasks asked one at a time are ~1,460 docker round
        trips, and over the deployed ssh transport that is a census costing minutes.
        """
        images = {task_id: self.image_for(task_id) for task_id in self._loaded()}
        present = sandbox.images_present(images.values())
        return tuple(task_id for task_id, image in images.items() if image in present)

    def image_for(self, task_id: str) -> str:
        """The published image for one task, read from the dataset's own column.

        No network and no construction: `dockerhub_tag` is a column, so `usable_ids` — one call per
        task over 731 tasks — stays the cheap local census §6.3b wants. That is the bug `real.py`
        records having measured on SWE-bench Verified, where naming an image went through the spec
        builder and fetched from GitHub 500 times.
        """
        return f"{SWEP_REGISTRY}:{self._loaded()[task_id]['dockerhub_tag']}"

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Apply the patch in the task's image, run the shipped scripts, read the shipped report."""
        row = self._loaded()[task.task_id]
        fail_to_pass = swep_list(row["fail_to_pass"])
        patch = extract_patch(submission)
        if not patch.strip():
            return Grade(0, len(set(fail_to_pass)))  # an empty answer is a failure, not an error
        instance = row["instance_id"]
        entry = swep_entryscript(row,
                                 _script(f"dockerfiles/base_dockerfile/{instance}/Dockerfile"),
                                 _script(f"dockerfiles/instance_dockerfile/{instance}/Dockerfile"))
        with sandbox.workspace() as (work, logs):
            payload = work / "payload"
            payload.mkdir()
            (payload / "patch.diff").write_text(patch)
            (payload / "run_script.sh").write_text(_script(f"run_scripts/{instance}/run_script.sh"))
            (payload / "parser.py").write_text(_script(f"run_scripts/{instance}/parser.py"))
            entry_path = payload / "entry.sh"
            entry_path.write_text(entry)
            # Executable and shebanged because the two image shapes invoke it differently: 21 of the
            # 23 provisioned images declare `ENTRYPOINT ["/bin/bash"]`, which makes the argv an
            # argument to bash, and the other two declare only `CMD ["bash"]`, which makes the argv
            # the command itself. `sandbox.run` has no `--entrypoint`, and it should not grow one for
            # this; a script that is executable in its own right satisfies both.
            entry_path.chmod(0o755)
            result = sandbox.run(self.image_for(task.task_id), [f"{sandbox.MOUNT}/entry.sh"],
                                 mount=payload, log_path=logs / "sandbox.log",
                                 timeout=SWEP_TIMEOUT, memory=SWEP_MEMORY, pids=SWEP_PIDS,
                                 cpus=SWEP_CPUS)
            tests = swep_verdict(result.log_path.read_text(errors="replace"))
        return swep_grade(tests, fail_to_pass, swep_list(row["pass_to_pass"]))

    def _loaded(self) -> dict[str, dict]:
        if self._rows is None:
            self._rows = swep_rows(_parquet_rows(_hf_file(SWEP_REPO, SWEP_FILE, SWEP_REVISION)))
        return self._rows
