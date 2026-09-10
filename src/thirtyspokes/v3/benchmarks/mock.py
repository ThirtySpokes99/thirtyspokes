"""Deterministic offline benchmark worlds — no network, no Docker, no key, no money (M2a).

Everything downstream needs a world to run a policy in: the episode loop's own tests, the duel's
adversarial archetypes (honest / specialist / spendthrift / miser / copier, M8), and the end-to-end
simulation (M9). The real adapters cannot serve that purpose — they need containers and dollars —
so without this module the mechanism would be testable only where it is most expensive to test.

A WORLD IS `TASK DIFFICULTY x MODEL STRENGTH`, NOT A TABLE OF CANNED OUTCOMES. The shape is what
buys the properties:

  * **Partial credit is a consequence, not a special case.** A model one point short of a task's
    difficulty completes all but one of its subgoals, two points short all but two. So a graded
    benchmark is `subgoals=4` rather than a second grading path, and binary — the expensive default
    (§5.4) — is the degenerate `subgoals=1`.
  * **The four worlds M2a requires are one argument apart.** With `strengths={"cheap": 1,
    "strong": 3}`: `difficulty=(1, 1)` a cheap model suffices; `(3, 3)` only the strong model
    succeeds; `(99, 99)` nothing succeeds at any price; `(2, 4), subgoals=4` graded partial credit.
    A mixture — `(1, 7)` against a top strength of 6 — puts roughly a seventh of the tasks out of
    reach of every model, which is the case where stopping early is **pure savings**: the score is
    already 0, so every further call is spend that `final_b` charges for and quality that cannot
    arrive (§5.1b). That a policy can learn it is exactly what a world with unsolvable tasks is for.

DETERMINISM IS BY sha256, NEVER BY `hash()`. Python randomises string hashing per process, so a
world built on `hash()` would deal a different difficulty in every run and the fixtures for a
paired duel would stop being paired — `koth/matrix.py::scoreable_ids` again, in the fixtures rather
than in the mechanism. The seed is a field so two worlds can differ while every other input matches.

WHAT THE MOCK DELIBERATELY DOES NOT MODEL: the content of an answer. A submission carries only
*which model wrote it*, because under §2.1 that is the only thing that can vary — the task text
going out is the benchmark's own, byte for byte. A submission no model in this world produced scores
0, so a Conductor that writes the answer itself gets nothing here, which is the same verdict the
real seam gives it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..tools import READER
from ..types import TOOL_NAMES, Environment, TaskSpec
from .base import Grade

# The mock transcript. A worker's entire contribution in this world is its identity, so that is all
# the "answer" carries; `MockBenchmark.grade` reads it back out.
_MARK = "solution attempt by "


def attempt(model_id: str) -> str:
    """What a worker model returns in this world."""
    return _MARK + model_id


def worker_answers(models: Iterable[str]) -> dict[str, str]:
    """A `worker.MockWorker(answers=...)` script for a world's models.

    The existing `MockWorker` is keyed by model ID and that is exactly enough: since a submission
    only has to say who wrote it, no second mock worker is needed and the §2.1 seam under test stays
    the shipped one. A model scripted here but absent from `strengths` answers with a strength of 0.
    """
    return {model_id: attempt(model_id) for model_id in models}


def answered_by(submission: str) -> str | None:
    """The model that produced this submission, or None if no model in any world did."""
    return submission[len(_MARK):] if submission.startswith(_MARK) else None


@dataclass(frozen=True)
class MockBenchmark:
    """A benchmark-shaped world: `n_tasks` tasks of hashed difficulty against models of a strength.

    `difficulty` is an inclusive range and each task draws from it deterministically, so a world can
    be uniform (`(3, 3)`) or a mixture (`(1, 7)`) without a second mechanism. Task IDs and prompts
    depend on the name alone, so re-seeding moves the difficulty of a task without renaming it —
    which is what makes two seeds two worlds over the *same* task set.
    """

    name: str
    strengths: Mapping[str, int]
    n_tasks: int = 20
    difficulty: tuple[int, int] = (1, 1)
    subgoals: int = 1
    tool_names: tuple[str, ...] = ()
    seed: str = "mock"
    root: str = ""

    def __post_init__(self) -> None:
        low, high = self.difficulty
        if high < low:
            # Not defensive: a reversed range makes the modulo in `difficulty_of` negative, so the
            # world would come out quietly easier than the fixture says rather than failing.
            raise ValueError(f"difficulty range {self.difficulty} is empty")
        if set(self.tool_names) & set(TOOL_NAMES) and not self.root:
            # A world may name tools the scaffold cannot run — the shipped stand-in declares `bash`
            # and `editor`, which exist to be RENDERED, since `render._task` is what the Conductor
            # reads. Declaring a real verb without a root is different: the default root is the
            # empty string, i.e. the process's own directory, so the fixture would read the
            # repository it is being tested from.
            raise ValueError(f"tools {self.tool_names} against root {self.root!r}: a world that "
                             "declares a runnable verb needs a root to run it against")

    def load(self) -> tuple[TaskSpec, ...]:
        """The pinned task set: IDs and prompts are a function of the name and the index alone."""
        return tuple(
            TaskSpec(task_id=f"{self.name}-{i:04d}", benchmark=self.name,
                     prompt=f"[{self.name}] task {i}: make the failing case pass.",
                     tools=self.tool_names)
            for i in range(self.n_tasks))

    def tools(self) -> tuple[str, ...]:
        return self.tool_names

    def environment(self, task: TaskSpec) -> Environment | None:
        """A directory on this machine, read by `in_process` — no image, no daemon, no network."""
        return Environment(image=f"{self.name}:mock", root=self.root) if self.root else None

    def difficulty_of(self, task_id: str) -> int:
        """This task's difficulty — a pure function of `(seed, name, task_id)`."""
        low, high = self.difficulty
        return low + _digest(f"{self.seed}|{self.name}|{task_id}") % (high - low + 1)

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Subgoals completed: one lost per point by which the model falls short of the task.

        A model at or above the task's difficulty completes all of them, and one absent from this
        world's `strengths` is the weakest possible — an unscripted rung is a useless one.

        A submission NO model produced scores 0 outright rather than being graded as a strength-0
        attempt. On an easy graded world the second reading would hand three of four subgoals to a
        string nobody generated, so a Conductor that smuggled its own answer into the record — or a
        worker that returned nothing — would collect partial credit for it (§2.1).
        """
        model_id = answered_by(submission)
        if model_id is None:
            return Grade(0, self.subgoals)
        shortfall = max(0, self.difficulty_of(task.task_id) - int(self.strengths.get(model_id, 0)))
        return Grade(max(0, self.subgoals - shortfall), self.subgoals)


def in_process(env: Environment, request: Mapping[str, object]) -> str:
    """`tools.in_sandbox` with the container replaced by THIS interpreter — the offline read seam.

    IT RUNS THE SHIPPED `tools.READER` SOURCE, not a second implementation of it. That is the whole
    point: the confinement the security story rests on — `realpath` after symlink resolution, the
    refusal of anything outside the root, the re-check of every file `FIND` walks to — is then
    exercised adversarially by the offline suite against a fake root on the filesystem, rather than
    only by a test that needs Docker and is therefore skipped on most machines.

    Returns the reader's stdout, markers and all, because that is what the seam's caller parses out
    of a container log; a mock that returned the payload already unwrapped would leave `observe`'s
    missing-marker path (§8b.3's exclusion) untested.
    """
    with tempfile.TemporaryDirectory() as work:
        call = pathlib.Path(work) / "call.json"
        call.write_text(json.dumps(request, sort_keys=True))
        out, argv = io.StringIO(), sys.argv
        sys.argv = ["reader.py", str(call)]
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(READER, "<v3-reader>", "exec"), {"__name__": "__main__"})  # noqa: S102
        finally:
            sys.argv = argv
        return out.getvalue()


def _digest(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
