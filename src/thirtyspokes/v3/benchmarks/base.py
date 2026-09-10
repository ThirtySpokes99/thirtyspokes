"""The benchmark contract: a pinned task set, the tools it exposes, and its own grader (M2a).

GRADED, NOT PASS/FAIL — AND THE TYPES SAY SO, BECAUSE A FLOAT WOULD NOT (§5.1, §5.4).

Binary outcomes carry maximum variance, and here that is a bill rather than a statistical footnote.
§5.4's table — ~225 tasks per arm to detect +10 points, ~670 for +5 — is computed assuming binary
scoring, over tasks that each cost real money on someone else's machine. Partial credit is one of
only two multipliers the plan has against those counts, so the granularity of a grader is a direct
lever on what the arena can resolve at all, and a benchmark that reports 0/1 where it could have
reported 7/9 has made every duel on it more expensive for nothing.

That is why `grade` returns `Grade(completed, total)` and not a bare float. A float lets an adapter
ship `1.0 if passed else 0.0` invisibly: the number looks the same as a graded one, nothing
downstream can tell, and §5.4's task count is silently wrong for that benchmark. With `total` in the
record the granularity is *published* — a benchmark that only ever reports out of 1 is visibly
binary, and the disagreement rate the plan currently assumes at 20% can be re-derived from what the
graders actually reported.

`Grade.score` is the graded score itself. the build plan M2a sketches it as a fourth protocol member
(`load`, `tools`, `grade`, `graded_score`); computing it once here instead is deliberate — twelve
adapters each deriving their own fraction is twelve chances to derive it differently, and one of
them to round it back to 0/1.

THE SCORE IS BOUNDED IN [0, 1] BY CONSTRUCTION. `score.py` means these into `quality_b` and
`final_b` subtracts a priced spend from that mean, so a grader returning 1.7 does not fail anywhere
— it inflates one benchmark's quality and moves a crown. `Grade` refuses to hold such a value.

THREE FACTS MUST AGREE PER BENCHMARK, AND THEY ARE CHECKED AT ASSEMBLY, NOT MID-WINDOW. `tools()`
declares a runnable verb, `environment()` is not None, and `task.prompt` ends in `tools.PROTOCOL`:
each is half of the read channel and any two without the third is a defect nothing downstream can
see. This module used to claim the first two were "asserted against each other" while the only
assertion was `inspector`'s ValueError — which fires after money is spent and is then swallowed into
a §6.3c exclusion, so a wiring mistake read as infrastructure noise. `check_channel` and
`check_protocol` refuse both disagreements for free, before a window opens;
`sandbox.check_grading_host` is the precedent for refusing at launch rather than spending a window.

A GRADER FAILURE MUST RAISE, NEVER RETURN 0.0 (§8b.3, §6.3c). A worker model erroring is an outcome
— data about that rung. Docker dying is a fact about the validator, and such a task is excluded from
BOTH arms and from the denominator, which the caller can only do if the failure reaches it. An
adapter that catches its own sandbox error and returns a zero converts the owner's infrastructure
noise into a miner's score, and nothing downstream can tell the difference afterwards. `grader`
therefore catches nothing, mirroring `scaffold._call_worker`, which keeps grading outside the one
`except` it has.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .. import tools
from ..types import TOOL_NAMES, Environment, Observation, TaskSpec, ToolCall


@dataclass(frozen=True)
class Grade:
    """What the benchmark's own grader saw: `completed` of `total` checks passed.

    `total` is the granularity the benchmark was capable of, so it is carried rather than divided
    away at the source: `Grade(0, 1)` and `Grade(0, 9)` are the same score and very different
    evidence about how many tasks a duel on this benchmark will need (§5.4).
    """

    completed: int
    total: int

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"a grade out of {self.total} checks is not a grade")
        if not 0 <= self.completed <= self.total:
            raise ValueError(f"{self.completed} of {self.total} checks passed is not a score in "
                             "[0, 1], and quality_b is a mean of these")

    @property
    def score(self) -> float:
        """The graded score in [0, 1] — what `quality_b` averages (§5.1)."""
        return self.completed / self.total


class Benchmark(Protocol):
    """One admitted benchmark. Adapters live outside this package; see `benchmarks/__init__.py`."""

    name: str

    def load(self) -> tuple[TaskSpec, ...]:
        """The pinned task set, with stable IDs (M2a exit 1).

        A tuple, and ordered: the window's task order is derived from the nonce
        (`scaffold.task_order`), which is only a *re-derivable* order if what it sorts is itself
        stable across two loads of the same pinned version.
        """

    def tools(self) -> tuple[str, ...]:
        """The tool surface this benchmark exposes to a worker, stamped onto every task it loads.

        A subset of `types.TOOL_NAMES`, because those are the verbs the scaffold can actually run;
        `()` is the ordinary answer and keeps the one-completion path exactly as it was.
        """

    def environment(self, task: TaskSpec) -> Environment | None:
        """Where this task's tool calls run, or None where the benchmark exposes no tools.

        It is `None` EXACTLY WHEN `tools()` declares no runnable verb, and the two are asserted
        against each other by `check_channel` at assembly rather than trusted: a benchmark that
        declares a verb it has nowhere to run would be a prompt that lies to the Conductor about the
        worker's action space, and one that has an environment and declares nothing would simply
        never use it.
        """

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """The benchmark's own grader, at its finest granularity. Raises if grading failed."""


def grader(benchmark: Benchmark) -> Callable[[TaskSpec, str], float]:
    """The `Scaffold.grade` seam: `(task, submission) -> graded score`.

    The argument order flips because the two contracts are written from different ends — a benchmark
    grades *a submission* against a task, and the scaffold holds *a task* and receives an answer for
    it. Nothing else happens here on purpose: no clamping (`Grade` already bounds the score), no
    default for a missing task, and above all no `except`. §8b.3 requires a grader failure to reach
    the caller, which drops the task from both arms (§6.3c); swallowing it here would look like a
    routing failure and be scored as one.
    """
    return lambda task, submission: benchmark.grade(submission, task).score


def inspector(benchmark: Benchmark, *,
              execute: Callable[[Environment, Mapping[str, object]], str] | None = None
              ) -> Callable[[TaskSpec, ToolCall, str], Observation]:
    """The `Scaffold.inspect` seam: `(task, worker's call, digest of the reply) -> Observation`.

    Beside `grader` because it is the same kind of thing — the adapter's own half of an episode,
    handed to the scaffold as a callable so that module stays offline — and it catches nothing for
    the same reason: a container that could not run is a fact about the validator, so it must reach
    the caller and drop the task from both arms (§8b.3, §6.3c) rather than become an observation
    saying so, which would put the owner's infrastructure into a worker's prompt.

    `execute` exists so the offline suite can drive the whole loop with no Docker against the SAME
    `tools.READER` source a container runs (`mock.in_process`); None means the real sandbox.
    """
    def inspect(task: TaskSpec, call: ToolCall, response_sha256: str) -> Observation:
        environment = benchmark.environment(task)
        if environment is None:
            raise ValueError(f"{benchmark.name} declared {call.name} for {task.task_id} and has no "
                             "environment to run it in")
        return tools.observe(environment, call, response_sha256, execute=execute)
    return inspect


def check_channel(benchmarks: Sequence[Benchmark]) -> None:
    """`tools()` and `environment()`, asserted against each other BEFORE a window spends anything.

    `Benchmark.environment` has always *claimed* the two agree. Nothing checked it: the only
    assertion was `inspector`'s `ValueError`, which fires mid-window, after money is spent, and is
    then caught by `_GraderGuard.inspecting` and turned into a §6.3c exclusion — so a wiring mistake
    left its task out of both arms and the denominator wearing infrastructure noise's clothes, which
    is the one confusion §8b.3 exists to prevent. Refused here instead, where it costs nothing and
    names the adapter. The precedent is `sandbox.check_grading_host`: refuse at launch rather than
    spend the window on a fact that was knowable for free.

    THE AGREEMENT IS OVER THE RUNNABLE VERBS, not over every name an adapter declares, because what
    needs an environment is exactly what `Scaffold._read` will dispatch — a `TOOL_NAMES` verb the
    benchmark declared. `TaskSpec.tools` also carries the surface `render._task` shows the
    Conductor, and the offline worlds legitimately name one the scaffold cannot run
    (`mock.MockBenchmark` declares `bash`/`editor` to be RENDERED, and refuses a *real* verb without
    a root for this same reason). A REAL adapter may not do that at all — `real.py` states it as a
    rule — but the rule there is about prompts that lie, not about wiring, and the two failures
    deserve their own refusals.

    Costs nothing and needs no Docker: `load()` is the pinned task set the corpus was assembled from
    and `environment` builds a record rather than starting a container.
    """
    for benchmark in benchmarks:
        declared = tuple(name for name in benchmark.tools() if name in TOOL_NAMES)
        environment = benchmark.environment(benchmark.load()[0])
        if bool(declared) != (environment is not None):
            raise ValueError(
                f"{benchmark.name} declares runnable verbs {declared} and its environment is "
                f"{environment!r}: a verb with nowhere to run is a prompt lying to the Conductor "
                f"about the worker's action space, and an environment no verb reaches is a "
                f"container nobody starts")


def check_protocol(tasks: Sequence[TaskSpec]) -> None:
    """A task invites a read exactly when it declares one — checked at window open, per task.

    THE THIRD FACT OF THE SAME AGREEMENT, AND THE ONE NOTHING CHECKED AT ALL. `tools.PROTOCOL` is
    appended by the ADAPTER (`r2egym.r2e_prompt`), which is what keeps §2.1's audit sentence
    byte-true — the protocol is INSIDE `task.prompt` — and it is also what makes forgetting it
    invisible: either half alone still loads, still hashes into the window commitment, and still
    runs an arm to completion.

    Both directions are wrong, and differently. A benchmark that declares a verb and forgets the
    protocol ships a worker never told it may read, which is the measured R2E-Gym failure (0 of 75
    appliable patches, `tools.py`) shipped again with the fix wired but unreachable. The reverse is
    worse: `Scaffold._read` refuses a call whose verb the task does not declare, so a worker that
    took the invitation has its `READ <path>` line graded as its patch.

    Over the runnable verbs and at window open, for `check_channel`'s two reasons — that is the set
    `_read` dispatches on, and this is the last moment before the slice costs anybody money.
    """
    for task in tasks:
        invited = task.prompt.endswith(tools.PROTOCOL)
        if invited != any(name in TOOL_NAMES for name in task.tools):
            raise ValueError(
                f"{task.task_id} ({task.benchmark}) {'ends' if invited else 'does not end'} in the "
                f"read protocol and declares tools {task.tools}: a worker invited to read a task "
                f"that declares no verb has its READ line graded as its answer, and one that "
                f"declares a verb without the protocol is never told it may read")
