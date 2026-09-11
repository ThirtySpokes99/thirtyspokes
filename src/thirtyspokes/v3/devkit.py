"""The miner's local dev kit — `thirtyspokes-dev` (the build plan M6b; the "Kind" property).

A miner gets ONE submission per hotkey (§7) and an invalid artifact spends it. Without a local kit
the only way to discover that a model fails admission, emits unparseable actions, or loops until the
step cap is to burn a registration finding out. the retired first-generation dev kit existed for exactly this reason
in the previous design and its promise has to survive into v3:

    what I see locally is what the validator scores.

THE ONE RULE THAT MAKES THAT PROMISE TRUE: this module CALLS THE VALIDATOR'S OWN FUNCTIONS. `admit`,
`Scaffold` and `score_arm` are imported and invoked, never re-expressed. A reimplementation drifts —
and it drifts precisely for the miners who most need it, since the ones checking locally are
the ones who cannot afford a surprise. Everything here is a thin wiring layer plus a report; if
this file ever grows arithmetic of its own, the promise above has quietly become a claim.

WHAT IT REPORTS THAT A SCORE ALONE CANNOT SHOW. `final` is one number, and three very different
models produce a low one: a bad router, a model whose turns do not parse, and a model that never
reaches STOP. `types.EpisodeResult.stopped_reason` exists to separate them, so the report leads with
the parse-failure rate and the endings breakdown, then prints the decisions themselves — which model
at which step, and why each episode ended.

WHAT IT DOES NOT DO: run the miner's 70 GB of weights. That needs their serving stack, so the
Conductor arrives through the same `conductor.Conductor` seam the validator uses (`--conductor
module:attr`). Nothing here touches R2, the chain, or a network at all — a miner must be able to
iterate before registering anything, and the offline stand-in world exists so that the first run
needs no key.
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .admission import AdmissionError, Reference, admit, describe
from .benchmarks.base import Benchmark, check_channel, grader, inspector
from .benchmarks.mock import MockBenchmark, worker_answers
from .conductor import Conductor
from .config import DUEL_WALL_CLOCK_REASON, EPISODE_CONCURRENCY
from .reference import always_cheapest, always_strongest, cascade, fit_pool
from .scaffold import Episode, OutcomeTable, Scaffold
from .score import ArmScore, Exchange, score_arm
from .types import (Catalog, CatalogEntry, Environment, EpisodeResult, Observation, StepRecord,
                    TaskSpec, ToolCall)
from .worker import MockWorker, Worker

# `EpisodeResult.stopped_reason`'s vocabulary (types.py), in the order a miner reads it: the one
# healthy ending first, then the five that mean something went wrong. Every one is printed even at
# zero — expressing "this never happened" by absence reads identically to a report that forgot to
# count it, which is exactly the failure mode this section exists to close.
#
# BOTH CLOCKS ARE NAMED. `duel_wall_clock` is the per-ARM bound (§8b.2), and a local run reaches it
# for exactly the reason a scored one does: `dry_run` goes through `Scaffold.run_window`, which
# enforces it. A kit that named only the episode clock would show a miner an ending its own report
# has no row for — local behaviour diverging from scored behaviour, which is the one thing this
# module exists to prevent.
ENDINGS = ("stop", "max_steps", "parse_failures", "wall_clock", DUEL_WALL_CLOCK_REASON,
           "budget_exhausted")


@dataclass(frozen=True)
class LocalWorld:
    """One local run's world — what §6.2 says the owner commits per window, minus the nonce.

    `exchange` is the owner's PUBLISHED `λ_b`/`C_b` table (§5.1b), not something a miner derives:
    it is pinned per corpus precisely so scores are comparable across windows, so a locally fitted λ
    would make the local `final` a different number from the scored one and break this kit's whole
    promise. `mock_world` derives its own only because a stand-in corpus has no published table.
    """

    catalog: Catalog
    tasks: tuple[TaskSpec, ...]
    grade: Callable[[TaskSpec, str], float]
    worker: Worker
    exchange: Mapping[str, Exchange]
    # The benchmarks' read channel (`benchmarks.base.inspector`), beside `grade` because it is the
    # other half of the same thing — what the adapters contribute to an episode. None is a world
    # whose benchmarks declare no tools, which is every world offline, and it makes the local run
    # and the scored run the same run either way (§2.1's audit is unchanged at `observations == ()`).
    inspect: Callable[[TaskSpec, ToolCall, str], Observation] | None = None


@dataclass(frozen=True)
class DevReport:
    """One dry run. Everything else is derived from the episodes, so nothing can disagree with them.

    `arm` is the only stored derivative, because scoring needs the exchange table as well as the
    episodes; it comes straight out of `score.score_arm` and is not post-processed here.
    """

    episodes: tuple[Episode, ...]
    arm: ArmScore
    # The outcome table's stats for this run (§5.2c): what was bought live and what was replayed.
    table: dict | None = None

    @property
    def spend_usd(self) -> float:
        """What this run cost on the key — the number to hold against the allowance (§4)."""
        return sum(e.result.spend_usd for e in self.episodes)

    @property
    def turns(self) -> int:
        return sum(len(e.result.steps) for e in self.episodes)

    @property
    def parse_failures(self) -> int:
        return sum(step.parse_failed for e in self.episodes for step in e.result.steps)

    @property
    def parse_failure_rate(self) -> float:
        """Refused turns over all turns (§2). A rate, not a count: it is the miner's decode-format
        yield, and it is invisible in `final` — a model whose every turn is refused scores like one
        that routes badly, and the two need entirely different fixes."""
        return self.parse_failures / self.turns if self.turns else 0.0

    @property
    def endings(self) -> dict[str, int]:
        """How the episodes ended, by `stopped_reason`. A reason outside `ENDINGS` is still counted
        rather than dropped — this section's job is that no ending goes unreported."""
        counts = dict.fromkeys(ENDINGS, 0)
        for episode in self.episodes:
            reason = episode.result.stopped_reason
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def check_admission(root: Path, reference: Reference) -> str | None:
    """The §1.1 gate, run by the validator's own `admit`. `None` if admitted, else its refusal.

    THE STRING IS RETURNED VERBATIM, never reworded or summarised. A refusal is what the miner
    greps for and what the operator sees in the validator's log, and a local paraphrase would be a
    divergence between local and scored behaviour dressed as a nicety — the exact thing the "Kind"
    property forbids, and the thing a miner would only discover after spending their one shot.
    """
    try:
        admit(root, reference)
    except AdmissionError as exc:
        return str(exc)
    return None


def dry_run(world: LocalWorld, conductor: Conductor, *, budget_usd: float, nonce: str = "dev",
            clock: Callable[[], float] = time.monotonic,
            concurrency: int = EPISODE_CONCURRENCY) -> DevReport:
    """Run the window the way the validator runs it, and score it the way the validator scores it.

    Three lines on purpose: `Scaffold` is the pinned harness (§3), `run_window` enforces the budget
    and the nonce-derived order (§4), and `score_arm` prices spend (§5.1b). The kit's contribution
    is the wiring and the report, never the arithmetic.

    `clock` is the same wall-clock seam `Scaffold` exposes (§8b.2), passed through so a miner — and
    this module's tests — can see the timeout ending in microseconds rather than in fifteen minutes.

    CONCURRENCY DEFAULTS TO THE VALIDATOR'S, and that is the whole point of this module rather than
    a performance nicety. The scored arm runs `EPISODE_CONCURRENCY` episodes at once, which is what
    lets ~250 tasks finish inside `DUEL_WALL_CLOCK_SECONDS`; a local run left at 1 takes hours on the
    same slice and ends in `duel_wall_clock`. A miner would then be shown a failure the validator
    would never have given them, and would tune against it — local behaviour diverging from scored
    behaviour, which is the one thing this module exists to prevent. It is an argument rather than a
    constant so a miner debugging one episode can still pin it to 1 and read a serial trace.
    """
    # The same outcome table the validator runs an arm against (§5.2c), fresh: within one arm a
    # key repeats only as a RETRY of the same model, which is the next attempt and a new draw, so a
    # single arm sees few hits — what the kit shows is the accounting the validator computes.
    table = OutcomeTable()
    scaffold = Scaffold(world.catalog, conductor, world.worker, world.grade, clock=clock,
                        inspect=world.inspect, table=table)
    episodes = scaffold.run_window(world.tasks, nonce=nonce, budget_usd=budget_usd,
                                   concurrency=concurrency)
    return DevReport(episodes, score_arm([e.result for e in episodes], world.exchange),
                     table=table.stats())


def format_report(report: DevReport) -> str:
    """The three things a miner cannot get from a number: the score's parts, the failure modes, the
    decisions."""
    return "\n".join([*_score_section(report), "",
                      *_failure_section(report), "",
                      *_decision_section(report)])


def suite_grade(benchmarks: Sequence[Benchmark]) -> Callable[[TaskSpec, str], float]:
    """`Scaffold.grade` over a slice spanning several benchmarks: dispatch by tag, then their own.

    A dispatcher, not a grader — the scoring is `benchmarks.base.grader`'s, unchanged. Public
    because a miner supplying their own `--world` needs the same dispatch, and the alternative is
    that the one thing that must not be reimplemented is the first thing they reimplement.

    Dispatch is by `TaskSpec.benchmark` because that is the tag `score.py` groups by and the breadth
    rule drops by (§5.2), so a task graded by the wrong benchmark's grader would also be *priced*
    under the wrong `λ_b`. A tag no benchmark claims raises instead of scoring 0: that is a slice
    built wrong, not a task the policy failed, and §8b.3 keeps the two apart.
    """
    graders = {benchmark.name: grader(benchmark) for benchmark in benchmarks}
    return lambda task, answer: graders[task.benchmark](task, answer)


def suite_inspect(benchmarks: Sequence[Benchmark], *,
                  execute: Callable[[Environment, Mapping[str, object]], str] | None = None
                  ) -> Callable[[TaskSpec, ToolCall, str], Observation]:
    """`Scaffold.inspect` over a slice spanning several benchmarks: dispatch by tag, then their own.

    THE EXACT MIRROR OF `suite_grade`, and it exists because `grade` had a dispatcher and `inspect`
    did not. A window slice is multi-benchmark by construction, so a bare `inspector(one_benchmark)`
    carried into every Scaffold is correct only while exactly ONE adapter declares tools and
    `Scaffold._read` short-circuits on every other task's empty `task.tools`. The second adapter to
    declare a verb turns it into `inspector(X)` handed a task from Y, and that fails two ways, both
    silent: `R2EGym.environment` looks the task up by ID, so it is a `KeyError` inside
    `_GraderGuard.inspecting`'s broad except and the task leaves BOTH arms and the denominator; an
    adapter whose environment ignores the task instead answers from the WRONG image, and the worker
    is shown another benchmark's bytes with nothing raising at all. A dispatch bug arriving dressed
    as infrastructure noise is exactly the confusion §8b.3 exists to prevent.

    Dispatch is by `TaskSpec.benchmark` for `suite_grade`'s reason and one more: that tag is pinned
    CORPUS data, so keying on it opens no §2.1 path. It is not absent from every prompt — `render`
    shows it to the Conductor, which is the whole point of naming the benchmark — but §2.1 is a
    sentence about WORKER prompts, and a worker's text is `compose(task.prompt, observations)`, into
    which the tag never enters. A tag no benchmark claims RAISES rather than scoring or returning
    None — a slice built wrong is not a read that did not happen, and `Scaffold._read` reads None as
    "the reply is the submission".

    `execute` is `inspector`'s own seam, threaded through unchanged so the offline suite can drive a
    multi-benchmark read world against the shipped `tools.READER` with no Docker
    (`mock.in_process`).
    """
    inspectors = {benchmark.name: inspector(benchmark, execute=execute)
                  for benchmark in benchmarks}
    return lambda task, call, digest: inspectors[task.benchmark](task, call, digest)


# --- the offline stand-in world -------------------------------------------------------------------
# Deliberately fake model IDs: nothing here is a claim about a real provider's price or skill.
_MOCK_CATALOG = Catalog(entries=(
    CatalogEntry("mock/cheap", 0.10, 0.40, 128_000),
    CatalogEntry("mock/mid", 0.60, 2.40, 200_000),
    CatalogEntry("mock/strong", 3.00, 12.00, 400_000),
))

_MOCK_STRENGTH = {"mock/cheap": 1, "mock/mid": 2, "mock/strong": 3}
_MOCK_COST = {"mock/cheap": 0.01, "mock/mid": 0.05, "mock/strong": 0.20}

# Two `benchmarks.mock` worlds rather than a corpus of this module's own, so the kit runs the same
# offline world the mechanism's own tests do. They differ in two ways on purpose: their difficulty
# mixes differ, so λ_b differs between them (§5.1b is per benchmark, and a world with one slope
# would demonstrate a global λ just as well); and one grades out of four subgoals while the other is
# binary, so a miner sees both granularities §5.1 talks about.
_MOCK_BENCHMARKS = (
    MockBenchmark("terminal-bench", _MOCK_STRENGTH, n_tasks=6, difficulty=(1, 2), subgoals=4,
                  tool_names=("bash", "editor")),
    MockBenchmark("cybergym", _MOCK_STRENGTH, n_tasks=6, difficulty=(1, 3)),
)

_MOCK_TASKS = tuple(task for benchmark in _MOCK_BENCHMARKS for task in benchmark.load())
_MOCK_GRADE = suite_grade(_MOCK_BENCHMARKS)


def mock_world() -> LocalWorld:
    """A synthetic corpus and pool, so the first local run needs no key and no network (M6b exit 5).

    WHAT A GOOD NUMBER HERE DOES AND DOES NOT MEAN. Difficulty in this world is a hash of the task
    ID (`benchmarks.mock`), so it is not legible in the prompt at all: no amount of reading the task
    says which model will solve it, and the only edge available is §3's — delegate, observe the
    failure, escalate. A real benchmark may well carry a signal this world does not; measurement 15
    found difficulty predictable at AUC 0.812 on production-shaped traffic against 0.500 on exam
    sets (ROUTING_MEASUREMENTS §15), which is the range this stand-in cannot stand in for. So a good
    score here means THE KIT WORKS, never that the artifact will score anything in particular.

    `λ_b` is derived here rather than hardcoded, from the same three fixed policies §6.1 sweeps and
    through the same `reference.fit_pool` the owner runs — so the stand-in world has the property the
    real one is built to have (the floor and King₀ score exactly the same) instead of a constant
    somebody guessed, and a miner's local score is priced the way the validator's is.
    """
    # The same gate the owner's `--world` module owes its corpus, run on the stand-in for the kit's
    # own reason: a miner who copies this function as the template for their `--world` copies the
    # check with it, and a world assembled here is assembled the way one is assembled there.
    check_channel(_MOCK_BENCHMARKS)
    fit = fit_pool({policy.name: _sweep(policy) for policy in
                    (always_cheapest(_MOCK_CATALOG), always_strongest(_MOCK_CATALOG),
                     cascade(_MOCK_CATALOG))})
    return LocalWorld(catalog=_MOCK_CATALOG, tasks=_MOCK_TASKS, grade=_MOCK_GRADE,
                      worker=_mock_worker(), exchange=fit.exchange)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="thirtyspokes-dev",
        description="Score your Conductor locally with the validator's own code, before you spend "
                    "your one submission. No R2 credential, no chain connection, no network.")
    parser.add_argument("--model-tree", type=Path, metavar="DIR",
                        help="your model tree — run the §1.1 admission checks over it")
    parser.add_argument("--reference", type=Path, metavar="DIR",
                        help="the owner's published reference tree, to check --model-tree against")
    parser.add_argument("--conductor", metavar="module:attr",
                        help="a zero-argument callable returning YOUR Conductor (act(prompt) -> "
                             "str). Runs on your machine, not the validator's, which never runs "
                             "miner code (§2.1). Default: a stand-in fixed policy.")
    parser.add_argument("--world", metavar="module:attr",
                        help="a zero-argument callable returning a LocalWorld — your own "
                             "benchmarks, graders, catalog and the owner's published λ table. "
                             "Default: the offline stand-in world.")
    parser.add_argument("--budget-usd", type=float, default=1.0,
                        help="the allowance this run may spend (§4). Default: 1.0")
    parser.add_argument("--concurrency", type=int, default=EPISODE_CONCURRENCY, metavar="N",
                        help=f"episodes in flight, as the validator runs them (default "
                             f"{EPISODE_CONCURRENCY}); pass 1 to read a serial trace of one episode "
                             f"at a time")
    parser.add_argument("--nonce", default="dev",
                        help="the window nonce the task order derives from (§4). Default: dev")
    args = parser.parse_args(argv)

    if args.model_tree:
        if not args.reference:
            raise SystemExit("--model-tree needs --reference: the checks compare your tree against "
                             "the owner's pinned architecture, and there is nothing to compare to")
        refusal = check_admission(args.model_tree, describe(args.reference))
        print("ADMISSION: " + (refusal or "admitted"))
        if refusal is not None:
            # A refused artifact is never scored, so printing a score under a refusal would show a
            # miner a number the validator will never compute for them.
            raise SystemExit(1)

    world = _load(args.world) if args.world else mock_world()
    if args.conductor:
        conductor: Conductor = _load(args.conductor)
    else:
        # King₀'s natural analogue (§5.5): the bar a Conductor has to clear is the best FIXED
        # policy, not an untrained base model, because a fixed cascade is what the owner could
        # deploy for free. Announced, so nobody reads a stand-in's number as their model's.
        conductor = always_cheapest(world.catalog)
        print(f"CONDUCTOR: stand-in reference policy '{conductor.name}' "
              f"{conductor.rungs} — pass --conductor module:attr to score YOUR model")
    print(format_report(dry_run(world, conductor, budget_usd=args.budget_usd, nonce=args.nonce,
                                concurrency=args.concurrency)))


def _load(spec: str) -> object:
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected module:attr, got {spec!r}")
    return getattr(importlib.import_module(module), attr)()


def _mock_worker() -> MockWorker:
    """A fresh worker per world: `MockWorker` records what it saw, and a shared instance would hand
    a miner the λ sweep's calls mixed in with their own run's."""
    return MockWorker(answers=worker_answers(_MOCK_STRENGTH), costs=_MOCK_COST)


def _sweep(policy: Conductor) -> list[EpisodeResult]:
    """One fixed policy over the whole stand-in corpus — §6.1's admission sweep, in miniature.

    Unbudgeted on purpose: this measures the pool's cost/quality slope, and a slope fitted on an arm
    that ran out of money halfway would be fitted on zeroes.
    """
    scaffold = Scaffold(_MOCK_CATALOG, policy, _mock_worker(), _MOCK_GRADE)
    return [e.result for e in scaffold.run_window(_MOCK_TASKS, nonce="sweep",
                                                  budget_usd=float("inf"))]


def _score_section(report: DevReport) -> list[str]:
    rows = [f"{'benchmark':<18}{'n':>4}{'quality':>10}{'$/task':>12}{'final':>10}"]
    for row in report.arm.per_benchmark:
        rows.append(f"{row.benchmark:<18}{row.n_tasks:>4}{row.quality:>10.4f}"
                    f"{row.spend:>12.6f}{row.final:>10.4f}")
    # Every column of the ALL row is a mean over BENCHMARKS, never over tasks (§5.1): a row that
    # pooled tasks would report a different number from the one the duel runs on, and would do it
    # most visibly on exactly the unbalanced slices the equal weighting exists for.
    spend = sum(r.spend for r in report.arm.per_benchmark) / len(report.arm.per_benchmark)
    rows.append(f"{'ALL':<18}{sum(r.n_tasks for r in report.arm.per_benchmark):>4}"
                f"{report.arm.quality:>10.4f}{spend:>12.6f}{report.arm.final:>10.4f}")
    return ["# SCORE — final_b = quality_b - lambda_b * spend_b / C_b, equal weight per benchmark "
            "(§5.1, §5.1b)", *rows,
            f"total spend this run: ${report.spend_usd:.6f}"]


def _failure_section(report: DevReport) -> list[str]:
    endings = "  ".join(f"{reason}={count}" for reason, count in report.endings.items())
    lines = ["# FAILURE MODES — a model that never parses, one that never stops and one that",
             "  simply routes badly all look alike in the score. These are what tell them apart.",
             f"parse failures: {report.parse_failures} of {report.turns} turns "
             f"({report.parse_failure_rate:.1%})",
             f"endings: {endings}"]
    if report.table is not None:
        lines.append(f"worker delegates: {report.table['fills']} bought live, "
                     f"{report.table['hits']} replayed from the outcome table (§5.2c)")
    return lines


def _decision_section(report: DevReport) -> list[str]:
    lines = ["# DECISIONS — which model at which step, and why each episode ended"]
    for episode in report.episodes:
        result = episode.result
        lines.append(f"{result.task_id} [{result.benchmark}] score {result.graded_score:.3f} "
                     f"${result.spend_usd:.6f} ended={result.stopped_reason}")
        lines += [f"  {_decision(step)}" for step in result.steps]
        if not result.steps and result.stopped_reason == "budget_exhausted":
            lines.append("  (the allowance never reached this task — §5.8)")
    return lines


def _decision(step: StepRecord) -> str:
    """One step, numbered from 1 as the prompt numbered it for the Conductor (`render._step`)."""
    if step.parse_failed:
        return f"step {step.step_index + 1}: REFUSED, unparseable — no model was called"
    if step.model_id is None:
        return f"step {step.step_index + 1}: STOP"
    return (f"step {step.step_index + 1}: {step.action.kind.upper()} {step.model_id} -> "
            f"{'succeeded' if step.success else 'failed'} (${step.cost_usd:.6f})")


# `python -m thirtyspokes.v3.devkit` must run the kit, not exit silently. Before the console script
# is installed that module path is the only way in, and a miner who gets no output cannot tell "the
# kit found nothing wrong" from "the kit never ran" — which is the ambiguity spending a shot that
# this whole module exists to prevent. `koth/devkit.py` and `v3/simulate.py` both carry this.
if __name__ == "__main__":
    main()
