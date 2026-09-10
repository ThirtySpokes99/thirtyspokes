"""The real adapters — three benchmarks that exist, at pinned revisions, with their own graders.

WHAT WAS ACTUALLY REACHABLE, checked against HuggingFace on 2026-09-01 rather than taken from §6.1's
list. **Four of the spec's twelve resolve there**: DeepSWE v1.1 (`R2E-Gym/R2E-Gym-V1`, Apache 2.0),
HLE (`cais/hle`, gated), GDPval (`openai/gdpval`) and CyberGym (`sunblaze-ucb/cybergym`).
Terminal-Bench 3.0 is real but GitHub-distributed. Agents' Last Exam, AutomationBench, FrontierSWE,
ProgramBench, ExploitBench, NL2Repo and SWE-Marathon did not resolve at any obvious path. Two of the
three adapters here — LiveCodeBench and SWE-bench Verified — are therefore **not on §6.1's list at
all**; they were chosen for being tractable and real, which is what M2a asks for and what four
unverified names cannot supply.

So the reachable pool is nearer seven than twelve, and the admitted set will be smaller again once
§6.1.3's probe runs. Nothing here hard-codes a count (§6.3b), and nothing here is the registry
either: `benchmarks/__init__.py` holds it, and of the three adapters below it registers two. HLE
left for `benchmarks/hle.py`, which supersedes the block at the foot of this module.

THE RECORD FOR EACH, since M2a exit 4 and the package checklist both demand it in writing and
"probably fine" is how a subnet acquires a takedown mid-window. Every field is in `Provenance` so it
is data rather than prose, and `test_real_benchmarks.py` asserts each adapter carries one.

| | LiveCodeBench | SWE-bench Verified | HLE |
|---|---|---|---|
| usable tasks | 112 of 175 | 500 | unknown — gated |
| licence | `cc`, variant unstated | none declared | MIT, but see below |
| redistribution | unresolved | unresolved | **refused by the publisher** |
| grader | reimplemented (stdin/stdout) | **as shipped** (`swebench` 4.1.0) | reimplemented (subset) |
| granularity | up to 1/12 | 1/|F2P|; binary on 69% of tasks | binary |

**ALL THREE FAIL §6.1.2 AS WRITTEN, AND THE OWNER SHOULD RE-READ THE REQUIREMENT BEFORE TREATING
THAT AS A VETO.** §6.1.2 asks for a licence to redistribute via the owner's R2, so that a window is
self-contained (§6.2). Two things narrow what that actually needs. There is one validator (D1), so
no third party ever fetches the slice — a committed list of task IDs plus the owner's own copy makes
the window as reproducible as a single-validator design can be. And the published traces (D15) carry
`StepRecord.rendered_state_digest`, a hash, rather than the rendered state, so publishing them does
not publish a single task statement. What the owner needs from these three is therefore a licence to
USE, which all three plainly grant, plus a decision recorded next to D15 that traces stay digested.
If a future reveal ever publishes rendered states verbatim, that decision reverses and HLE in
particular must leave the corpus: its card asks, in terms, that the dataset not be re-uploaded or
distributed.

TWO STRUCTURAL FINDINGS THAT ARE NOT ABOUT ANY ONE BENCHMARK.

1. **NO ADAPTER MAY DECLARE A TOOL THE SCAFFOLD CANNOT RUN.** `render._task` renders `task.tools`
   into the prompt as text, so a benchmark that declared `("bash", "editor")` would be telling the
   Conductor about an action space the worker does not have — the exact "prompt that lies" the
   package checklist forbids. What the scaffold can run is the three read-only verbs of
   `v3/tools.py` (`types.TOOL_NAMES`) and nothing else; **all three adapters in this module still
   return `()`**, so everything here is run one-shot, and this repository has measured that regime:
   SWE-bench Pro one-shot scored **0/120** where published agentic scaffolds run 62-80%
   (`koth/lcb.py`), and R2E-Gym measured a routable band of **+0.0000** with 0 of 75 patches applying
   (the sandbox record §7). §3 argues a pinned scaffold cancels between arms, and it does — but a
   benchmark pinned at the floor cancels to nothing, which is precisely what §6.1.3's admission probe
   exists to catch. The read channel is declared by `r2egym.py` alone, for a reason that is about the
   PIN and not about the work: these instance images float on `:latest`, so a read against one is not
   reproducible across a re-pull, whereas R2E-Gym's tag is the task's own commit.

2. **SWE-bench NEEDS A MULTI-GIGABYTE DOCKER IMAGE PER TASK**, which is the one tractability
   criterion the build plan M2a names and rules out ("no multi-GB per-task images"). Each instance grades
   inside `swebench/sweb.eval.x86_64.<instance_id>:latest`, so a 500-task corpus is 500 images. The
   validator must provision the slice's images before the window opens — `sandbox.run` passes
   `--pull never` so that a missing one is a loud operator failure rather than a pull inside a graded
   episode — and `usable_ids` reports which of them are actually present. A second consequence has no
   fix here: those images are tagged `latest`, so the dataset revision is pinned (§6.1.1) while the
   environment the tests run in is not.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..types import TaskSpec
from . import sandbox
from .base import Grade

# --- the record every adapter carries (M2a exit 4, §6.1) ------------------------------------------

REDISTRIBUTION_PERMITTED = "permitted"
REDISTRIBUTION_UNRESOLVED = "unresolved"
REDISTRIBUTION_REFUSED = "refused"

# `as-shipped` means the benchmark's own published grading code decided the score. `reimplemented`
# means this repository decided it. The distinction is not bookkeeping: §3's argument that scaffold
# quality CANCELS between the two arms holds because both arms are graded identically by something
# neither of them chose, and a home-grown grader moves the level of one benchmark while still looking
# like a score. It cancels within a duel either way; what a reimplementation costs is the ability to
# read a v3 number against the benchmark's published leaderboard, which is how a broken adapter is
# normally caught.
GRADER_AS_SHIPPED = "as-shipped"
GRADER_REIMPLEMENTED = "reimplemented"


@dataclass(frozen=True)
class Provenance:
    """Everything §6.1 requires to be known about a benchmark before it can carry weight.

    Carried as a field rather than written in a docstring because the checks are per benchmark and
    the corpus rotates: a benchmark whose licence is unresolved is a benchmark the owner must decide
    about, and a decision that lives only in prose is one that is made again, differently, by the
    next person to add an adapter.
    """

    repo: str                   # where it comes from
    revision: str               # A COMMIT, NEVER A BRANCH — see below
    licence: str
    redistribution: str         # one of the REDISTRIBUTION_* constants
    grader: str                 # one of the GRADER_* constants
    grader_source: str          # the exact code that decides the score
    deviation: str              # how the v3 score differs from the benchmark's published metric
    granularity: str            # what one `Grade`'s `total` counts, and how coarse it really is
    usable_tasks: int | None    # the census (M3a exit 4, M7a); None where it could not be taken

    def __post_init__(self) -> None:
        # A revision that is a branch name is not a pin. `λ_b` and `C_b` are fitted per corpus and
        # pinned (§5.1b), so a benchmark that moves under the arena silently reprices every score
        # computed against it — and the miners who trained on the published traces trained against
        # the old one. 40 hex characters is a git object id and nothing else is.
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(f"{self.repo} is pinned to {self.revision!r}, which is not a commit id")


class BenchmarkUnavailable(RuntimeError):
    """The task set could not be obtained: no network, a missing optional dependency, or a gate.

    Distinct from `sandbox.SandboxError` because it happens at a different time and means a different
    thing. A load failure means this benchmark cannot be part of the window at all, which the window
    builder handles by narrowing the admitted set and recording it (§6.3b invariant 3). A grader
    failure happens mid-arm, on one task that both arms must now drop (§6.3c).
    """


def _hf_token() -> str | None:
    """The Hub credential, under BOTH names this repo and the Hub each use.

    `hf_hub_download` reads `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` from the environment and knows
    nothing about `OWNER_HF_TOKEN`, which is the name this repository keeps the credential under
    (`.env`, `koth/neuron.py`). A gated benchmark loaded without it 401s **even after the terms have
    been accepted**, and — this is why it is worth a function — the failure is indistinguishable
    from not having accepted them, so the operator re-accepts a gate that was never the problem.
    """
    import os
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OWNER_HF_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _hf_file(repo: str, filename: str, revision: str) -> pathlib.Path:
    """Fetch one file at a pinned revision, from the cache when it is already there.

    `hf_hub_download` and not `datasets.load_dataset`: the loader resolves a *config*, which for
    LiveCodeBench means a Python build script whose `ALLOWED_FILES` table maps release names onto
    file lists — a second, upstream-controlled indirection between a pinned revision and the rows
    that come back. A file name at a commit has no such layer.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        raise BenchmarkUnavailable("huggingface_hub is required to load a real benchmark "
                                   "(`uv pip install -e '.[benchmarks]'`)") from exc
    try:
        return pathlib.Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                            token=_hf_token()))
    except Exception as exc:                                     # noqa: BLE001 — gate, 404, offline
        raise BenchmarkUnavailable(
            f"could not fetch {filename} from {repo}@{revision}: {exc}") from exc


def _parquet_rows(path: pathlib.Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        raise BenchmarkUnavailable("pyarrow is required to read a parquet task set "
                                   "(`uv pip install -e '.[benchmarks]'`)") from exc
    return pq.read_table(path).to_pylist()


# --- LiveCodeBench --------------------------------------------------------------------------------

LCB_NAME = "livecodebench"
LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

# THE SIX RELEASE FILES THIS REVISION SHIPS, in the order `code_generation_lite.py`'s ALLOWED_FILES
# concatenates them, with the census's usable count beside each. A release tag names a PREFIX of the
# list (`release_v4` is files 1-4), so every file is an INCREMENT rather than a snapshot:
#
#   file          usable  contest dates (usable rows)  easy/med/hard
#   test.jsonl       210  2023-05-13 .. 2024-03-02        76/76/58
#   test2.jsonl       57  2024-03-09 .. 2024-05-25        23/15/19
#   test3.jsonl       53  2024-06-01 .. 2024-08-03        19/17/17
#   test4.jsonl       65  2024-08-04 .. 2024-09-28        15/16/34
#   test5.jsonl      105  2024-09-22 .. 2024-12-28        24/24/57
#   test6.jsonl      112  2025-01-04 .. 2025-04-06        26/26/60
#
# Measured by `scripts/lcb_census.py` (the LCB census §1) and reproduced through this
# module's own `lcb_rows` on 2026-09-02. ZERO question IDs are shared across the six, so the counts
# ADD to 602 and merging releases is a concatenation rather than a union — which is why `_loaded`
# can key them into one dict without a collision rule.
LCB_RELEASES = {"test.jsonl": 210, "test2.jsonl": 57, "test3.jsonl": 53,
                "test4.jsonl": 65, "test5.jsonl": 105, "test6.jsonl": 112}

# THE DEFAULT CORPUS, AND NOTHING HERE MOVES IT. `LiveCodeBench()` reads this one file, exactly as it
# always has; the other five load only for a caller that names them. Same file `koth/lcb.py` pinned,
# so the two measurements stay comparable.
#
# WHY THE OTHER FIVE ARE LOADABLE AND STILL NOT ENABLED. §5.4 wants ~250 tasks per arm over N = 2
# benchmarks, so a LiveCodeBench stratum of 125 — and `window.draw`'s `ranked[:125]` over 112 tasks
# is the IDENTITY. The draw is not a draw, and M7a's protection (the slice is chosen after
# challengers commit, so nobody can know what will be scored) protects nothing when the draw is
# everything. 602 tasks fixes that. What it is bought with is contamination: the older releases are
# 1-2 years deeper into every worker's training window, and the cost of that is SPREAD, not cheating
# — §2.1 stops the Conductor emitting answer text, so a memorised worker helps the king's arm and
# the challenger's identically. A task every pool model has memorised scores ~1.0 under every routing
# decision, contributes zero paired variance, and is a task the owner paid for that resolves nothing.
# Enabling is therefore the owner's call on a measurement that has not run (census §6: draw test 3's
# ~112 tasks stratified across the six and report the per-release span). This module supplies the
# draw and the tag; it does not supply the decision, and `LCB_FILE` is what says so.
LCB_FILE = "test6.jsonl"

# stdin-style problems only, and no starter code: they grade by running the program and comparing
# stdout, one uniform convention. LeetCode-style problems use a functional calling convention that
# would need a second harness, and mixing the two silently changes what "solved" means
# (`koth/lcb.py`). Measured at this revision: 112 of 175 rows survive — 26 easy, 26 medium, 60 hard.
LCB_PLATFORM = "atcoder"
# Read out of the table rather than repeated under it: the default corpus's census IS its one file's
# census, and two places to write 112 is one place for them to disagree.
LCB_USABLE_TASKS = LCB_RELEASES[LCB_FILE]

# Test cases per problem, capped. Reused from `koth/lcb.py`, where the cap exists because one problem
# ships 2305 cases; at this revision the median problem has 43. The cap is also the grading
# GRANULARITY (§5.1): twelve cases is a score in twelfths, where LiveCodeBench's own metric is
# all-or-nothing over every case. The first cases in the list are the problem's public samples — 1 to
# 5 of them, measured — so a 12-case check is roughly three samples and nine hidden cases.
LCB_MAX_TESTS = 12
LCB_PER_CASE_TIMEOUT = 10
LCB_IMAGE = "python:3.11-slim"
LCB_VERDICT = "V3_CASES"

# The driver, reused from `koth/lcb.py` except that it reports `ok` AND `len(cs)` as a fraction
# rather than collapsing them to 1.0-iff-all-passed. Partial credit is one of only two multipliers
# §5.4 has against ~670 tasks per arm, and this is the benchmark that can supply it most cheaply.
#
# THE PER-CASE TIMEOUT IS WHAT KEEPS A HANGING PROGRAM SCOREABLE. It bounds the workload INSIDE the
# container, so a program that never terminates comes back as an honest `0 of 12` instead of hitting
# `sandbox.run`'s clock — which cannot tell a hung submission from a validator whose timeout was too
# tight, and therefore excludes the task from both arms (§8b.3).
_LCB_DRIVER = (
    "import json,subprocess,sys\n"
    "cs=json.load(open('/w/tests.json'))\n"
    "ok=0\n"
    "for c in cs:\n"
    "    try:\n"
    "        r=subprocess.run([sys.executable,'/w/sol.py'],input=c['input'],\n"
    f"                         capture_output=True,text=True,timeout={LCB_PER_CASE_TIMEOUT})\n"
    "        if r.stdout.split()==c['output'].split(): ok+=1\n"
    "    except Exception: pass\n"
    f"print('{LCB_VERDICT}',ok,len(cs))\n")

LCB_PROVENANCE = Provenance(
    repo=LCB_REPO, revision=LCB_REVISION,
    licence="cc (the card names no variant; problem statements are AtCoder's)",
    redistribution=REDISTRIBUTION_UNRESOLVED,
    grader=GRADER_REIMPLEMENTED,
    grader_source="_LCB_DRIVER: run the program per case, compare stdout by whitespace-split "
                  "tokens. LiveCodeBench's own grader is not in the dataset repo — it lives in the "
                  "LiveCodeBench GitHub package — so this is the same convention re-expressed, as "
                  "`koth/lcb.py` already does, which is what makes the two comparable.",
    deviation="published pass@1 is all-or-nothing over EVERY case; this is the fraction over the "
              f"first {LCB_MAX_TESTS}, so v3 numbers run higher than the leaderboard's and are "
              "finer. Comparable across arms, not against the leaderboard.",
    granularity=f"1/n for n = min(stdin cases, {LCB_MAX_TESTS}); at this revision every usable "
                f"problem has at least 2 cases, so n is {LCB_MAX_TESTS} for nearly all of them",
    usable_tasks=LCB_USABLE_TASKS)


def lcb_prompt(row: dict) -> str:
    """The task statement as it will be forwarded to a worker, byte for byte (§2.1).

    The trailing instruction is owner text, not the Conductor's: LiveCodeBench's `question_content`
    describes the problem but never says what artifact to return, and a grader that runs a program
    against stdin cannot grade prose. It is identical for every miner and both arms, so it is part of
    the pinned task exactly as the scaffold's own template is. Reused verbatim from `koth/lcb.py`, so
    a score here means what it meant there.
    """
    return (f"{row['question_content']}\n\n"
            "Write a complete Python 3 program that reads from standard input and writes the answer "
            "to standard output. Output ONLY the program source, no prose and no code fences.")


def lcb_cases(row: dict) -> list[dict]:
    """This problem's stdin test cases, public then private, capped at `LCB_MAX_TESTS`.

    The private cases arrive base64'd over zlib over a pickle — and the pickle yields the JSON
    *string*, not the list, which is the trap `koth/lcb.py` documents (a 2305-character string read
    as 2305 cases). A row whose private cases will not decode still has its public ones, and a
    public-only check is a weaker check rather than a broken one, so that failure narrows the grade
    instead of dropping the task.
    """
    import base64
    import pickle
    import zlib

    cases = json.loads(row["public_test_cases"])
    try:
        raw = pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"])))
        cases += json.loads(raw) if isinstance(raw, str) else raw
    except Exception:                    # noqa: BLE001 — public-only is still a valid check
        pass
    return [c for c in cases if c.get("testtype") == "stdin"][:LCB_MAX_TESTS]


def extract_code(text: str) -> str:
    """Pull the program out of a worker's response.

    Reused verbatim from `koth/lcb.py`, which took it from this repository's pool-matrix experiments.
    It is not a parser this module gets to improve: changing how a response becomes a program changes
    every score measured with it, and the reason to keep it is that the earlier measurements are the
    only calibration this benchmark has.
    """
    t = str(text or "")
    if "```" in t:
        for b in (b for b in t.split("```") if b.strip()):
            b = b[len("python"):] if b.lstrip().lower().startswith("python") else b
            if "input" in b or "print" in b:
                return b.strip() + "\n"
    return t.strip() + "\n"


def lcb_verdict(output: str) -> tuple[int, int]:
    """`(cases passed, cases run)` from the driver's own line.

    The marker is the discriminator between a submission that failed and a container that never ran
    (§8b.3). `sandbox.run` deliberately returns a non-zero exit code as data, because a program
    crashing on every case is an ordinary outcome; what is NOT ordinary is a driver that printed
    nothing, and only this function can tell the difference.
    """
    for line in reversed(output.splitlines()):
        if line.startswith(LCB_VERDICT):
            _, ok, total = line.split()
            return int(ok), int(total)
    raise sandbox.SandboxError(f"the LiveCodeBench driver produced no {LCB_VERDICT} line: "
                               f"{output.strip()[-400:]}")


class LiveCodeBench:
    """Self-contained competition problems, graded by running the program against its test cases.

    The one regime this repository measured that is worth routing over: scores span 0.47-0.78, 13.9%
    of problems are unsolvable by anyone, and the oracle gap is +0.083 where GSM8K's is +0.019
    (`koth/lcb.py`). It is also the cheapest of the three to grade — one throwaway `python:3.11-slim`
    container per submission, no per-task image, sub-second on the measured cases.
    """

    name = LCB_NAME

    def __init__(self, releases: Sequence[str] = (LCB_FILE,)) -> None:
        """One release file by default, and it is `LCB_FILE` — the corpus that shipped.

        Normalised to census order rather than the caller's, for the reason `window.draw` ranks by
        hash and `pool_digest` sorts: what loads then depends only on WHICH releases are enabled and
        never on the order somebody happened to list them in.

        `provenance` is per instance because `usable_tasks` is a census and a census is of a corpus.
        A six-release instance carrying "112" would be exactly the remembered number the module
        docstring refuses; the default instance's record is `LCB_PROVENANCE` unchanged, which the
        tests assert rather than assume.
        """
        chosen = set(releases)
        if not releases or chosen - set(LCB_RELEASES):
            raise ValueError(f"LiveCodeBench releases must be a non-empty subset of "
                             f"{tuple(LCB_RELEASES)}, not {tuple(releases)!r}")
        self.releases = tuple(name for name in LCB_RELEASES if name in chosen)
        self.provenance = replace(
            LCB_PROVENANCE, usable_tasks=sum(LCB_RELEASES[name] for name in self.releases))
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        """The enabled releases' tasks — WITH NO `group`, WHICH IS AN ARGUMENT AND NOT AN OMISSION.

        `TaskSpec.group` is the duel's RESAMPLE UNIT: `duel` draws whole groups because tasks in one
        group are correlated and an iid resample on clustered data publishes an `lcb` that is too
        narrow (`duel.py`). R2E-Gym sets it from the repository. So a release — a date-bounded batch
        of AtCoder contest problems, `test6` being 2025-01-04 .. 2025-04-06 — has to be one of two
        things: a cluster, or a reporting tag. **It is a tag**, on four counts, and the fourth is the
        one that decides it.

        1. WHAT A RELEASE IS. R2E-Gym's group is a REPOSITORY, and two tasks in it are edits to the
           same source tree: same code, same conventions, same test framework, so a challenger that
           suits sympy is better on all of them at once by a mechanism you can name. Two AtCoder
           problems from adjacent weekends share a DATE and nothing else — different setters,
           different topics, self-contained statements, their own test data. The release boundary is
           a publication batch drawn over a contest calendar, not a shared substrate. `duel.py`
           already recorded the conclusion: *"LiveCodeBench and HLE tasks really are independent and
           supply nothing. An invented grouping is worse than none."*
        2. THE ONE CHANNEL FROM RELEASE TO ADVANTAGE RUNS THROUGH A LABEL THAT IS ALREADY PER TASK.
           Releases 1-3 are 28-33% hard against 4-6's 52-54% (census §4), and `koth/lcb.py` measured
           the achievable gap falling from +0.083 to +0.042 once the hard tier is dropped — so
           advantage really does move with DIFFICULTY, and a release is a coarse proxy for
           `difficulty_of`. An iid resample over a mixed pool already carries that heterogeneity;
           what clustering would add on top is the correlation left over once difficulty is held
           fixed. That residual has never been measured — UNMEASURED, not measured small, which is
           itself the argument for leaving `lcb` where it is rather than for moving it blind.
        3. THE ONE GENUINELY RELEASE-LEVEL EFFECT — age, so contamination — MOVES THE LEVEL AND NOT
           THE DELTA. §2.1 means a memorised worker helps both arms identically, so a saturated old
           release pushes both arms toward 1.0 and its PAIRED deltas toward zero with almost no
           variance. A cluster of null tasks does not need a wider interval; it needs not to be
           bought, which is the decision `LCB_FILE` is waiting on. Clustering there would price a
           correlation the pairing has already cancelled.
        4. AND `group` IS NOT FREE TO SET. `window.pool_digest` binds it, so tagging even the
           DEFAULT corpus would move the on-chain commitment of 112 tasks in which not one prompt,
           tool or ID changed (`test_putting_the_release_in_group_would_move_a_commitment...`
           measures exactly that). It would also widen `lcb` on every LiveCodeBench stratum at SIX
           clusters, which is where `duel.py`'s own calibration is measured worst — the residual
           under-coverage tracks the CLUSTER COUNT, 0.100 at five clusters against 0.060 at
           twenty-five. The asymmetry `duel.py` takes the cluster fix on, *a wider interval refuses
           crowns and cannot manufacture one*, runs the other way for a structure nobody has
           measured: it would refuse real challengers for it.

        WHAT WOULD REVERSE THIS, so the next person does not re-derive it: the same test-3
        measurement, read on the per-release DELTAS between arms rather than on the per-release
        SCORES. Scores differing by release is contamination and is expected. A challenger's
        ADVANTAGE differing by release beyond its own noise is a cluster — and then `group=` here is
        one line, at the price of re-committing every pool digest.
        """
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=lcb_prompt(row),
                              tools=self.tools())
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        """Empty — see finding 1 in the module docstring. A self-contained programming question has
        no tree to read, so there is nothing here for the worker read channel to expose."""
        return ()

    def environment(self, task: TaskSpec) -> None:
        """None, exactly because `tools()` is empty (`base.Benchmark.environment`)."""
        return None

    def difficulty_of(self, task_id: str) -> str:
        """The dataset's own `easy`/`medium`/`hard` label, for the stratified draw (§6.3b).

        Exposed because the usable pool is 26/26/60 and a uniform draw therefore lands hard-heavy —
        which `koth/lcb.py` measured as the worst place to take a signal from (35 of 72 reference
        cells empty from token-cap truncation, and the achievable gap falling from +0.083 to +0.042
        once the hard tier is dropped). `TaskSpec` has no room for a label, so the label stays here.
        """
        return self._loaded()[task_id]["difficulty"]

    def release_of(self, task_id: str) -> str:
        """Which release file a task came from — A REPORTING TAG, and deliberately not a cluster.

        Here for the same reason `difficulty_of` is here: `TaskSpec` has no room for a label. The one
        field it could have borrowed is `group`, and `load` records at length why it must not.

        It is what makes the census's own recommendation runnable — draw across the six, report the
        per-release score span, and read `test6` as its own line so the enlarged corpus stays
        comparable to the 112-task measurement that admitted this benchmark.
        """
        return self._loaded()[task_id]["release"]

    def draw_across_releases(self, n: int, *, seed: str) -> tuple[TaskSpec, ...]:
        """`n` tasks spread as evenly across the ENABLED releases as those releases can supply.

        The instrument the census asks for (§6) and nothing more: test 3's spread smoke already plans
        to run ~112 LiveCodeBench tasks, its cost is driven by the task COUNT and not by which tasks,
        so drawing them balanced across the six releases and tagging each with `release_of` answers
        the enable question for money that was going to be spent anyway. **It is not the window's
        draw** — `window.draw` is nonce-derived and stratified by benchmark; this is a probe's.

        Ranked by `sha256(seed|task_id)` rather than by a seeded RNG stream, which is
        `window._draw_key`'s and `scaffold.task_order`'s choice for the reason it is also right here:
        the result depends only on which tasks are eligible, never on the order they were listed in
        nor on any library's generator, so it is reproducible from the published record alone.

        TWO RULES ABOUT THE COUNTS, both deterministic and neither silent. Round-robin over the
        releases in CENSUS ORDER, so when `n` does not divide evenly the remainder goes to the
        EARLIEST releases — 112 over six is 19/19/19/19/18/18, oldest first. And a release too small
        for its share contributes everything it has while the shortfall spreads over the others, so
        the draw returns `n` whenever `n` tasks exist at all rather than silently shrinking. Asking
        for more than the corpus holds returns the corpus.
        """
        ranked: dict[str, list[TaskSpec]] = {name: [] for name in self.releases}
        for task in self.load():
            ranked[self.release_of(task.task_id)].append(task)
        for tasks in ranked.values():
            tasks.sort(key=lambda t: hashlib.sha256(f"{seed}|{t.task_id}".encode()).hexdigest())

        picked: list[TaskSpec] = []
        while len(picked) < n and any(ranked.values()):
            for name in self.releases:
                if ranked[name] and len(picked) < n:
                    picked.append(ranked[name].pop(0))
        return tuple(picked)

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Fraction of the problem's test cases the submitted program passes."""
        cases = lcb_cases(self._loaded()[task.task_id])
        if not cases:
            # Not a zero: a problem with no runnable case cannot distinguish a good submission from
            # a bad one, so scoring it would add a constant to one benchmark's quality (§8b.3).
            raise sandbox.SandboxError(f"{task.task_id} has no stdin test cases at this revision")
        code = extract_code(submission)
        if not code.strip():
            return Grade(0, len(cases))          # an empty answer is a genuine failure, not an error
        with sandbox.workspace() as (work, logs):
            payload = work / "payload"
            payload.mkdir()
            (payload / "sol.py").write_text(code)
            (payload / "tests.json").write_text(json.dumps(cases))
            (payload / "drv.py").write_text(_LCB_DRIVER)
            result = sandbox.run(
                LCB_IMAGE, ["python", f"{sandbox.MOUNT}/drv.py"],
                mount=payload, log_path=logs / "sandbox.log",
                # The driver bounds each case; this bounds the driver, with room for interpreter
                # start-up on every one of them. It should never fire, and if it does the task is
                # excluded rather than zeroed.
                timeout=LCB_PER_CASE_TIMEOUT * len(cases) + 60)
        passed, total = lcb_verdict(result.output)
        return Grade(passed, total)

    def _loaded(self) -> dict[str, dict]:
        """The enabled releases, merged in census order, each row stamped with the file it came from.

        Stamped here rather than inside `lcb_rows`, which stays a pure filter over rows: it is the
        one definition of "usable" that `scripts/lcb_census.py` calls rather than restates, and a
        census that disagreed with the adapter would be one bug rather than two definitions.

        The merge needs no collision rule because the census measured that it needs none — zero
        question IDs are shared across the six files (`LCB_RELEASES`).
        """
        if self._rows is None:
            rows: dict[str, dict] = {}
            for release in self.releases:
                with open(_hf_file(LCB_REPO, release, LCB_REVISION)) as handle:
                    parsed = lcb_rows([json.loads(line) for line in handle])
                rows.update({task_id: dict(row, release=release)
                             for task_id, row in parsed.items()})
            self._rows = rows
        return self._rows


def lcb_rows(rows: list[dict]) -> dict[str, dict]:
    """The usable subset, keyed by stable task ID and in file order.

    File order rather than a shuffle, and the dataset's own `question_id` rather than an index: the
    window's task order comes from the nonce alone (`scaffold.task_order`), and an ID that renumbers
    between two loads makes two windows incomparable and the published traces unmatchable to the
    tasks they came from.
    """
    return {f"{LCB_NAME}-{row['question_id']}": row for row in rows
            if row["platform"] == LCB_PLATFORM and not row["starter_code"].strip()}


# --- SWE-bench Verified ---------------------------------------------------------------------------

SWE_NAME = "swebench-verified"
SWE_REPO = "princeton-nlp/SWE-bench_Verified"
SWE_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SWE_FILE = "data/test-00000-of-00001.parquet"
SWE_USABLE_TASKS = 500

# The published instance images, one per task. `make_test_spec(..., namespace=...)` turns this into
# `swebench/sweb.eval.x86_64.<instance_id>:latest` — see finding 2 in the module docstring for what
# 500 of those cost and for the `:latest` pin that is not one.
SWE_IMAGE_NAMESPACE = "swebench"
# Written down rather than left to `make_test_spec`'s default, because `image_for` now names the
# image without calling it and the two must agree: an arch that differed between the census and the
# grader would report a task usable and then run it in an image that is not there.
SWE_ARCH = "x86_64"

# A repository test suite, not a single program: more memory, more processes, and minutes rather than
# seconds. Generous on purpose — these are HOST-PROTECTION bounds, and a suite killed by one produces
# no verdict, which `grade` turns into an exclusion rather than a zero (§8b.3).
SWE_MEMORY = "8g"
SWE_PIDS = 2048
SWE_CPUS = "4"
SWE_TIMEOUT = 1800.0

SWE_PROVENANCE = Provenance(
    repo=SWE_REPO, revision=SWE_REVISION,
    licence="none declared on the dataset card (the SWE-bench harness is MIT; the issue text and "
            "patches carry the twelve source repositories' own licences)",
    redistribution=REDISTRIBUTION_UNRESOLVED,
    grader=GRADER_AS_SHIPPED,
    grader_source="swebench 4.1.0: `make_test_spec` builds the eval script, `get_logs_eval` applies "
                  "the per-repository log parser, `get_eval_tests_report` and `compute_pass_to_pass` "
                  "turn it into per-test statuses. Only the container invocation is ours, because "
                  "`--network none` and the resource caps are v3 requirements the harness's own "
                  "runner does not impose.",
    deviation="the published metric is `resolved` — every FAIL_TO_PASS passes AND every PASS_TO_PASS "
              "still passes. This reports the FAIL_TO_PASS fraction with PASS_TO_PASS as a gate, "
              "which is identical to `resolved` whenever |FAIL_TO_PASS| == 1 (345 of 500 tasks) and "
              "at either extreme, and finer in between.",
    granularity="1/|FAIL_TO_PASS|. Measured at this revision: 345 of 500 tasks have exactly one, so "
                "the benchmark is BINARY on 69% of its tasks and graded on 31% (mean 3.03, max 438)",
    usable_tasks=SWE_USABLE_TASKS)


def swe_prompt(row: dict) -> str:
    """The task statement forwarded to a worker (§2.1).

    The worker is asked for a patch and nothing else because this benchmark declares no tools, so its
    delegates are ONE completion call with no repository access (finding 1). It cannot read the code
    it is patching, which is why this regime measured 0/120 on SWE-bench Pro; Verified is the easier
    set and M3a is what says whether the floor here is off zero.
    """
    return (f"Repository: {row['repo']}\n"
            f"You are at commit {row['base_commit']}.\n\n"
            f"# Problem\n{row['problem_statement']}\n\n"
            "Return ONLY a unified git diff (patch) that resolves the problem, rooted at the "
            "repository root so it applies with `git apply -p1`. Do not include prose outside the "
            "diff.")


def extract_patch(submission: str) -> str:
    """Pull the unified diff out of a worker's response, fenced or not.

    A patch must end in a newline or `git apply` refuses it, so the newline is added here rather than
    left to whether a model happened to emit one — an answer rejected for its last byte would be
    scored as a failure to solve the problem.
    """
    text = str(submission or "")
    if "```" in text:
        for block in (b for b in text.split("```") if b.strip()):
            body = block.split("\n", 1)[-1] if block.lstrip().lower().startswith("diff") else block
            if "--- " in body and "+++ " in body:
                return body.strip("\n") + "\n"
    return text.strip("\n") + "\n" if text.strip() else ""


def swe_grade(report: dict, n_fail_to_pass: int) -> Grade:
    """The shipped report, turned into a graded score.

    THE OBVIOUS PARTIAL CREDIT — fraction of ALL tests passing — IS WRONG HERE, AND MEASURABLY SO.
    A Verified instance carries a median of 51 PASS_TO_PASS regression tests against a median of one
    FAIL_TO_PASS, so a submission that changes nothing at all passes 364 of 365 on the measured
    instance and scores 0.997. Every task would then be pinned near 1.0, the spread between two
    policies would be a rounding error, and §6.1.3's admission probe would drop the benchmark for
    having no signal — for a reason that is entirely an artefact of how the score was defined.

    So the regression suite is a GATE and the problem-specific tests are the score: full marks needs
    every FAIL_TO_PASS, breaking anything that used to pass scores zero however much else was fixed.
    That preserves the shipped metric's meaning (it agrees with `resolved` exactly when there is one
    FAIL_TO_PASS test, and at 0 and 1 always) while giving the 31% of tasks with more than one
    something finer than a bit.
    """
    from swebench.harness.grading import compute_pass_to_pass

    if compute_pass_to_pass(report) < 1.0:
        return Grade(0, n_fail_to_pass)
    return Grade(len(report["FAIL_TO_PASS"]["success"]), n_fail_to_pass)


class SweBenchVerified:
    """500 human-validated GitHub issues, graded by the SWE-bench harness's own test verdicts.

    The whole grading path except the container invocation is `swebench` 4.1.0's: this class builds
    the shipped `TestSpec`, runs the shipped eval script, and hands the shipped log parser the
    container's output. What it does not reuse is `run_evaluation.run_instance`, which starts
    containers with the daemon's defaults — no network isolation and no resource caps — and writes
    its verdict into a log tree keyed by a run id. §8b's requirements are not optional and a grader
    that phones home is not a grader, so the invocation is ours and the judgement is theirs.
    """

    name = SWE_NAME
    provenance = SWE_PROVENANCE

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=swe_prompt(row),
                              tools=self.tools())
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        """Empty — finding 1, and here it costs the most: SWE-bench is agentic work asked as a single
        completion. It is NOT the read channel `r2egym.py` declares, and the reason is the pin rather
        than the shape of the work: these instance images float on `:latest`, so a read against one
        is not reproducible across a re-pull and an observation would not be re-executable."""
        return ()

    def environment(self, task: TaskSpec) -> None:
        """None, exactly because `tools()` is empty (`base.Benchmark.environment`)."""
        return None

    def usable_ids(self) -> tuple[str, ...]:
        """The task IDs whose instance image is present locally, so a window can be built from them.

        `sandbox.run` passes `--pull never`, so an absent image is a refusal at grading time. That is
        the right failure — pulling a multi-gigabyte image inside a graded episode spends the arm's
        wall clock on the validator's housekeeping (§8b.2) — but it is only kind if it can be checked
        beforehand, which is what this is for.

        One batched `docker image inspect` per chunk rather than one per task (`images_present`):
        the per-task shape costs two round trips each, and over the deployed ssh transport a census
        is then minutes of a window's clock spent on housekeeping again.
        """
        images = {task_id: self.image_for(task_id) for task_id in self._loaded()}
        present = sandbox.images_present(images.values())
        return tuple(task_id for task_id, image in images.items() if image in present)

    def image_for(self, task_id: str) -> str:
        """The published image for one task, named WITHOUT building that task's scripts.

        `make_test_spec` eagerly builds the env script, and for every repo whose install spec is a
        `requirements.txt` or an `environment.yml` that fetches the file from
        raw.githubusercontent.com. Measured under an egress guard: a bare `image_for` opened a
        GitHub connection, so `usable_ids` — one call per task — turned the cheap local census
        §6.3b wants into 500 network round trips, and an offline validator could not take it at all.

        The key reads only `arch`, `instance_id`, the tag and the namespace; none of the three
        script lists reach it. So they are omitted and swebench's own property still does the
        formatting, which keeps the lowercasing and the `__` -> `_1776_` substitution theirs rather
        than a copy here that can drift from the images the tag actually names. Grading still goes
        through `_spec`, because there the eval script IS the shipped grader.
        """
        try:
            from swebench.harness.constants import MAP_REPO_TO_EXT
            from swebench.harness.test_spec.test_spec import TestSpec
        except ImportError as exc:                               # pragma: no cover — env-dependent
            raise BenchmarkUnavailable("the `swebench` package names this benchmark's images; "
                                       "reimplementing that name is refused (§6.1.2)") from exc
        row = self._loaded()[task_id]
        return TestSpec(instance_id=row["instance_id"], repo=row["repo"], version=row["version"],
                        repo_script_list=[], eval_script_list=[], env_script_list=[],
                        arch=SWE_ARCH, FAIL_TO_PASS=[], PASS_TO_PASS=[],
                        language=MAP_REPO_TO_EXT[row["repo"]], docker_specs={},
                        namespace=SWE_IMAGE_NAMESPACE).instance_image_key

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Apply the patch in the task's own image, run the shipped eval script, parse its log."""
        spec = self._spec(task.task_id)      # first, so a missing `swebench` names itself (§6.1.2)
        from swebench.harness.constants import APPLY_PATCH_FAIL

        total = len(spec.FAIL_TO_PASS)
        patch = extract_patch(submission)
        if not patch.strip():
            return Grade(0, total)               # an empty answer is a genuine failure, not an error
        with sandbox.workspace() as (work, logs):
            payload = work / "payload"
            payload.mkdir()
            (payload / "patch.diff").write_text(patch)
            (payload / "eval.sh").write_text(spec.eval_script)
            (payload / "run.sh").write_text(_swe_runner())
            log_path = logs / "sandbox.log"
            result = sandbox.run(spec.instance_image_key,
                                 ["/bin/bash", f"{sandbox.MOUNT}/run.sh"],
                                 mount=payload, log_path=log_path, timeout=SWE_TIMEOUT,
                                 memory=SWE_MEMORY, pids=SWE_PIDS, cpus=SWE_CPUS)
            # THE APPLY-FAILURE CHECK COMES FIRST, AND THE ORDER IS THE POINT. `get_logs_eval`
            # reports the same `(no statuses, False)` for a patch that would not apply and for a
            # container that died before writing its markers, but §8b.3 puts those on opposite sides
            # of the line: a malformed patch is the worker's output and scores 0, while a container
            # that produced nothing is the validator's problem and the task leaves both arms. Reading
            # the shipped marker first is what separates them.
            log = result.log_path.read_text(errors="replace")
            if APPLY_PATCH_FAIL in log:
                return Grade(0, total)
            report = _swe_report(spec, log_path)
        return swe_grade(report, total)

    def _spec(self, task_id: str):
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError as exc:                               # pragma: no cover — env-dependent
            raise BenchmarkUnavailable("the `swebench` package supplies this benchmark's grader; "
                                       "reimplementing it is refused (§6.1.2)") from exc
        return make_test_spec(self._loaded()[task_id], namespace=SWE_IMAGE_NAMESPACE,
                              arch=SWE_ARCH)

    def _loaded(self) -> dict[str, dict]:
        if self._rows is None:
            self._rows = swe_rows(_parquet_rows(_hf_file(SWE_REPO, SWE_FILE, SWE_REVISION)))
        return self._rows


def _swe_runner() -> str:
    """Apply the model's patch, then run the shipped eval script — the harness's own sequence.

    `run_evaluation.run_instance` tries three apply commands in order and, if all three fail, logs
    `APPLY_PATCH_FAIL` and never runs the tests. Reproduced here rather than imitated: the three
    commands and the marker are imported from the package, so a harness upgrade that changes either
    changes this too instead of leaving a copy that silently disagrees about what "the patch applied"
    means.
    """
    from swebench.harness.constants import APPLY_PATCH_FAIL, DOCKER_WORKDIR
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS

    attempts = " || ".join(f"{cmd} {sandbox.MOUNT}/patch.diff" for cmd in GIT_APPLY_CMDS)
    return ("#!/bin/bash\n"
            "set -uxo pipefail\n"
            f"cd {DOCKER_WORKDIR}\n"
            f"{attempts} || {{ echo '{APPLY_PATCH_FAIL}'; exit 1; }}\n"
            f"/bin/bash {sandbox.MOUNT}/eval.sh\n")


def _swe_report(spec, log_path: pathlib.Path) -> dict:
    """The shipped parser's per-test verdict, or a refusal if it could not read one."""
    from swebench.harness.grading import get_eval_tests_report, get_logs_eval

    statuses, parsed = get_logs_eval(spec, str(log_path))
    if not parsed:
        raise sandbox.SandboxError(
            f"the swebench log parser found no test output for {spec.instance_id}; the patch did "
            "apply, so this is the grader rather than the submission")
    return get_eval_tests_report(statuses, {"FAIL_TO_PASS": spec.FAIL_TO_PASS,
                                            "PASS_TO_PASS": spec.PASS_TO_PASS})


def swe_rows(rows: list[dict]) -> dict[str, dict]:
    """Keyed by the dataset's own `instance_id`, with the two test lists decoded.

    The parquet stores `FAIL_TO_PASS` and `PASS_TO_PASS` as JSON strings, and `make_test_spec` wants
    lists. Decoding here rather than at the call site means the shipped spec builder is handed
    exactly the shape it documents, which is the difference between reusing a grader and reusing most
    of one.
    """
    decoded = {}
    for row in rows:
        row = dict(row)
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if isinstance(row[key], str):
                row[key] = json.loads(row[key])
        decoded[f"{SWE_NAME}-{row['instance_id']}"] = row
    return decoded


# --- Humanity's Last Exam -------------------------------------------------------------------------

HLE_NAME = "hle"
HLE_REPO = "cais/hle"
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_FILE = "data/test-00000-of-00001.parquet"

# Multiple-choice rows only, and only where the gold answer is a single option token. The other half
# of HLE is `exactMatch` short answers, and the benchmark's SHIPPED grader for those is an LLM judge
# (`hle_eval/run_judge_results.py` prompts a frontier model per question). That is disqualifying
# twice over for this seam: it costs a model call per graded task on top of the episode's own spend
# — which §5.1b prices into `final_b` and would therefore charge a miner for the grader — and it is
# non-deterministic, so the same submission can score differently in the king's arm and the
# challenger's, which is the one thing a PAIRED comparison cannot survive (§5.2).
HLE_ANSWER_TYPE = "multipleChoice"

# The answer line HLE's own system prompt asks for. Kept verbatim so that a response formatted for
# the shipped judge is formatted for this grader too.
HLE_ANSWER_MARKER = "answer:"

HLE_PROVENANCE = Provenance(
    repo=HLE_REPO, revision=HLE_REVISION,
    licence="MIT on the card, and the card also asks in terms that the dataset not be publicly "
            "shared, re-uploaded or distributed; the repo is gated behind accepted terms",
    redistribution=REDISTRIBUTION_REFUSED,
    grader=GRADER_REIMPLEMENTED,
    grader_source="`hle_grade`: the option token after HLE's own `Answer:` line, matched against the "
                  "gold. The shipped grader is an LLM judge and cannot be used here — see "
                  "`HLE_ANSWER_TYPE` for the two reasons.",
    deviation="published accuracy is judge-scored over the whole set; this is exact match over the "
              "multiple-choice subset. On that subset the judge's task is a letter comparison, which "
              "is what makes the substitution defensible — and it is why the subset is the corpus.",
    granularity="1/1 — binary, and unavoidably so. This is the benchmark that proves the protocol "
                "carries a binary grader honestly (`Grade(0, 1)` publishes its own coarseness) "
                "rather than the one that helps §5.4's task count",
    usable_tasks=None)          # the census needs accepted terms; see `HumanitysLastExam.load`


def hle_prompt(row: dict) -> str:
    """The question plus HLE's own answer-format instruction (§2.1: forwarded byte for byte)."""
    return (f"{row['question']}\n\n"
            "Respond in the following format:\n"
            "Explanation: {your reasoning}\n"
            "Answer: {your chosen answer}")


def hle_option(text: str) -> str:
    """The answer token a response or a gold carries, normalised.

    TWO RULES, AND THEY POINT AT OPPOSITE ENDS OF THE TEXT ON PURPOSE. With HLE's own `Answer:`
    marker present, the answer is the FIRST non-empty line after the LAST such marker: last, because
    a model that echoes the format instruction before answering would otherwise be graded on the
    instruction; first after it, because the answer is what follows the marker and anything below is
    the confidence line the format also asks for. With no marker at all, it is the LAST non-empty
    line — an unformatted reply reasons first and answers at the end, so reading its first line would
    grade the reasoning.

    Normalisation is case-folding and stripping surrounding punctuation and nothing else: `hle_rows`
    admits only rows whose gold is a single option token, so there is no free-text equivalence to
    judge here — that is the job this seam refuses to do without a model.
    """
    body = str(text or "")
    lowered = body.lower()
    marked = HLE_ANSWER_MARKER in lowered
    if marked:
        body = body[lowered.rfind(HLE_ANSWER_MARKER) + len(HLE_ANSWER_MARKER):]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    return (lines[0] if marked else lines[-1]).strip(" .*:()[]").casefold()


def hle_grade(submission: str, gold: str) -> Grade:
    """Binary, and `Grade(_, 1)` says so — §5.4's cost of a benchmark that cannot do better."""
    return Grade(1 if hle_option(submission) == hle_option(gold) else 0, 1)


def hle_rows(rows: list[dict]) -> dict[str, dict]:
    """The mechanically gradeable subset: multiple choice, a single-token gold, and no image.

    Three filters, each removing rows this seam cannot honestly score rather than rows it finds
    inconvenient. A gold that is not one option token would need the judge back (`hle_option` would
    be comparing free text). An image-bearing question is one the worker never sees — the scaffold
    forwards `TaskSpec.prompt`, a string, so a multimodal row would be graded on a question that was
    never fully asked, which is a task every model fails for a reason that has nothing to do with
    routing.
    """
    usable = {}
    for row in rows:
        if row.get("answer_type") != HLE_ANSWER_TYPE:
            continue
        if str(row.get("image") or "").strip():
            continue
        if not re.fullmatch(r"[A-Za-z]", str(row.get("answer", "")).strip().strip(" .*:()[]")):
            continue
        usable[f"{HLE_NAME}-{row['id']}"] = row
    return usable


class HumanitysLastExam:
    """Knowledge questions — the non-code benchmark, and the one that tests the protocol's shape.

    It is here to answer a question about `base.Benchmark` rather than about routing: a corpus of
    three code benchmarks would let a code-shaped assumption hide in the protocol (a sandbox that is
    always needed, a `Grade` whose total is always a test count, tools that are always a shell). This
    adapter needs no Docker, no image and no subprocess, and its `Grade` is out of 1.

    IT CANNOT BE LOADED IN THIS ENVIRONMENT AND THAT IS RECORDED RATHER THAN WORKED AROUND. `cais/hle`
    is gated: fetching it needs an authenticated HuggingFace token belonging to an account that has
    accepted the terms. So the row schema below is taken from the benchmark's own evaluation code
    (`hle_eval/`) and not verified against the data, `load` raises `BenchmarkUnavailable` naming the
    gate, and `usable_tasks` is None — a census nobody took, said out loud, rather than a number
    somebody remembered. Everything that decides a score is a module-level function over rows, so the
    grading contract is fully tested offline; what is untested is the fetch and the field names.
    """

    name = HLE_NAME
    provenance = HLE_PROVENANCE

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=hle_prompt(row),
                              tools=self.tools())
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        return ()

    def environment(self, task: TaskSpec) -> None:
        return None

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        return hle_grade(submission, self._loaded()[task.task_id]["answer"])

    def _loaded(self) -> dict[str, dict]:
        if self._rows is None:
            self._rows = hle_rows(_parquet_rows(_hf_file(HLE_REPO, HLE_FILE, HLE_REVISION)))
        return self._rows
