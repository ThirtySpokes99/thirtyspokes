"""The episode loop — owner code, and the only place a worker model is ever called (§3).

THE INVARIANT THIS MODULE EXISTS TO HOLD (§2.1):

    The only things that reach a worker model are the pinned task text, a validated model ID, and
    bytes the benchmark's own environment returned in answer to that same worker's own read-only
    requests. The Conductor still contributes exactly one thing: the model ID.

Everything else here is arithmetic; that sentence is the mechanism. It is what permits public
benchmarks to be used at all — a miner cannot smuggle a memorised solution into the answer path, so
no detector has to look for one, and `koth/verify.py::grounding_check` (removed with v2, 2026-09-07) becomes unnecessary rather
than merely stronger. Five structural choices carry it, and they are structural precisely because a
convention would decay:

  * **`_call_worker` is the only call site**, and its inputs are `(task, model_id)`. There is no
    parameter through which Conductor output could flow: the model ID is re-validated against the
    window's committed snapshot on the way in (`pool.validate`), the text is composed from
    `task.prompt`, and the params are the pinned module constant `worker.WORKER_PARAMS`.
  * **It logs the exact bytes it sends**, into `Episode.requests`, before sending them — so
    equality with the benchmark's own statement is an assertion anyone can run over a real window's
    record, not a claim about the source code. Under the read channel that assertion becomes
    `request.text == tools.compose(task.prompt, request.observations)`, a RECOMPUTATION from the
    published benchmark and the published record; and because `compose(p, ()) == p` byte for byte,
    it is the old sentence unchanged on the first turn of every delegate.
  * **The observation transcript is a local variable inside one `_call_worker` call**, so it is
    DELEGATE-SCOPED by construction and there is no code path by which a Conductor decision — a
    RETRY, or its timing — could extend it or hand one model another model's reads. That is §2.1
    one level down, expressed as a scope rather than as a rule (`tools.py`).
  * **`Observation` has exactly one constructor**, `tools.observe`, whose inputs are an environment
    the BENCHMARK named and a call a WORKER wrote. Nothing here accepts a caller-supplied
    observation, output or transcript string.
  * **The raw Conductor output is still recorded** in `StepRecord.raw_output`. The invariant is not
    achieved by scrubbing the trace: a hostile Conductor's smuggled answer is preserved verbatim in
    the published record (D15) and never appears in a request.

WHERE THE OTHER OPERATIONAL RULES LIVE, since each is a way a duel could be stalled or made unfair:

  * §4  — the window budget is a *value* passed in, snapshotted by the caller at window open. There
    is deliberately no way for the loop to re-read a miner's allowance, because a mid-duel raise
    once spend climbs would otherwise be free. Exhaustion zeroes every remaining task.
  * §4  — task order is derived from the window nonce alone (`task_order`), so both arms are zeroed
    on the same tail.
  * §8b.2 — `MAX_STEPS` and the per-episode wall clock bound a pathological model; three consecutive
    parse failures force STOP (`action.ParseFailureTracker`). The per-duel clock bounds the *arm*
    and is therefore checked in `run_window`, between tasks.
  * §8b.3 — a worker failure is an *outcome*: recorded, and the episode continues. A grader failure
    is not, and the try/except is confined to `_call_worker` so that grading — which happens after
    it returns — can never be silently swallowed into a "bad rung".
  * §5.1 — the two numbers that cross this seam from outside are refused when they are outside their
    contract (`ScaffoldError`), because after this module they are only arithmetic.
"""

from __future__ import annotations

import math

import concurrent.futures

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from . import tools
from .action import ParseFailureTracker
from .config import (
    DUEL_WALL_CLOCK_REASON,
    DUEL_WALL_CLOCK_SECONDS,
    EPISODE_WALL_CLOCK_SECONDS,
    MAX_READS_PER_DELEGATE,
    MAX_STEPS,
)
from .conductor import Conductor
from .pool import validate
from .render import ConductorState, render
from .types import Catalog, EpisodeResult, Observation, OutcomeRow, StepRecord, TaskSpec, ToolCall
from .worker import WORKER_PARAMS, NotAnOutcome, Worker


class ScaffoldError(Exception):
    """A number crossed this seam outside its contract, so the arm cannot be scored.

    Raised rather than clamped, and raised rather than recorded as a bad rung, because of where this
    module sits: after it, a cost and a graded score are only arithmetic (`score.final_b` subtracts
    `λ_b · spend / C_b` from a mean of scores) and nothing downstream re-checks either. A silent
    clamp would keep the window alive at the price of scoring a quantity nobody measured.

    The trade is deliberate and it is not the `_call_worker` trade. A provider exception is caught
    there because losing a window to one flaky rung is worse than recording it; here the cost of
    refusing is at most one window (§8b.6: a refused window is not downtime, the king simply keeps
    earning) while the cost of accepting is a coronation, and §7 gives a hotkey one shot and appends
    the lineage forever. Between losing a window and crowning on a corrupt number, only one is
    reversible.
    """


@dataclass(frozen=True)
class WorkerRequest:
    """One worker call, exactly as it left the scaffold — the §2.1 audit row.

    `text` is what was passed to the provider. The owner's audit is

        text == tools.compose(task.prompt, observations)

    byte for byte, over every row of every window — a recomputation rather than a claim about source
    code, and identical to the old `text == task.prompt` wherever `observations` is empty, which is
    every row of every benchmark that declares no tools and the first turn of every delegate.

    `observations` is the transcript this call was composed from, and it is on the ROW because that
    is what makes the audit self-contained. The response commitment lives on `Observation` instead of
    here, because this row is written into the log BEFORE its call is made (which is what keeps the
    failure path auditable) and a reply cannot be committed to before it exists.
    """

    task_id: str
    model_id: str
    text: str
    observations: tuple[Observation, ...] = ()


@dataclass(frozen=True)
class Episode:
    """One task, one arm: the score, and every worker request the episode made.

    The requests are returned rather than logged optionally because an episode whose worker calls
    were not recorded is an episode whose §2.1 compliance cannot be checked afterwards — and the
    check is the thing that makes public benchmarks usable.
    """

    result: EpisodeResult
    requests: tuple[WorkerRequest, ...]
    # How many of this episode's delegates the window's outcome table answered, and how many it
    # bought live (§5.2c). Zero and zero without a table, which is every construction that predates
    # it. Per episode so an arm can be metered for what it replayed against what it paid for.
    hits: int = 0
    fills: int = 0


@dataclass(frozen=True)
class Delegate:
    """What one delegate came back with: the last audit row, the reply, the price, and the receipts
    the seam minted for it (empty under a seam that mints none). `outcome` is False for a failure
    that said nothing about the rung — the meter refused (`worker.NotAnOutcome`) — which the table
    must not remember."""

    request: WorkerRequest
    text: str | None
    cost_usd: float
    receipts: tuple = ()
    outcome: bool = True
    failure: str | None = None       # why `text` is None (`types.StepRecord.failure`)


class OutcomeTable:
    """The window's shared worker outcomes (§5.2c, D17): a memo at the delegate, filled live once.

    ONE PER WINDOW, SHARED BY EVERY ARM THE WINDOW RUNS — reference, king, challengers, in that
    order, so the owner's fills cover the cascade's rungs and a challenger that agrees with the king
    pays for its agreement without the provider being asked twice. A miss is a live call through
    the existing seam, graded, then stored; a hit returns the stored row and calls no provider and
    no grader. Sharing draws is a correctness gain before it is a saving: M3a measured
    always-cheapest duelled against a second draw of ITSELF clearing three of four verdict
    conditions, and under replay identical policies tie exactly.

    A memo, never a model: every row is a real draw bought through the gateway. Predicting a
    worker's outcome is what this project has measured to be either wrong or the router itself,
    and it stays refused.

    DURABLE WHEN GIVEN A PATH. Every fill is appended as one JSON line before `put` returns, and a
    table opened on an existing file starts from its rows, so a restart re-buys nothing (§8b.5). A
    torn last line is dropped, exactly as `Checkpoint` drops one. Nothing carries across windows:
    a row's price is the catalog snapshot it was bought under, and re-pricing its tokens against a
    later window's snapshot is the follow-up this v1 leaves a note for, not a thing it does.

    Thread-safe for the shape the validator drives it in: `EPISODE_CONCURRENCY` episodes of ONE
    arm at once, each on a different task, so no key is ever in flight twice, and a later arm's
    hits read rows a finished arm wrote.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._rows: dict[tuple[str, str, int], OutcomeRow] = {}
        self._hits: dict[tuple[str, str, int], int] = {}
        self._lock = threading.Lock()
        self._path = None if path is None else Path(path)
        self.hits = 0
        self.fills = 0
        if self._path is not None and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = row_from_json(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    break                  # a torn tail is the process that died, not a bad row
                self._rows[row.key] = row

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: tuple[str, str, int]) -> OutcomeRow | None:
        """The stored draw for this key, counting the hit — or None, which means buy it."""
        with self._lock:
            row = self._rows.get(key)
            if row is not None:
                self.hits += 1
                self._hits[key] = self._hits.get(key, 0) + 1
            return row

    def hit_count(self, key: tuple[str, str, int]) -> int:
        """How many times this key was replayed — the drift audit samples among keys with one."""
        with self._lock:
            return self._hits.get(key, 0)

    def put(self, row: OutcomeRow) -> None:
        """Store a draw the caller just bought and graded. The first writer of a key wins: two
        arms cannot buy one key in this table's threading shape, and if they ever did the earlier
        draw is the one every later arm replays."""
        with self._lock:
            if row.key in self._rows:
                return
            self._rows[row.key] = row
            self.fills += 1
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row_json(row), sort_keys=True) + "\n")
                    handle.flush()

    def rows(self) -> tuple[OutcomeRow, ...]:
        with self._lock:
            return tuple(self._rows.values())

    def stats(self) -> dict:
        """What the reveal publishes about the table — never the table itself (D15)."""
        rows = self.rows()
        failed = sum(1 for row in rows if row.failed)
        return {"rows": len(rows), "fills": self.fills, "hits": self.hits,
                "failed_rows": failed,
                "failure_rate": (failed / len(rows)) if rows else 0.0,
                "provider_usd": sum(row.cost_usd for row in rows)}


def row_json(row: OutcomeRow) -> dict:
    return {"task_id": row.task_id, "model_id": row.model_id, "attempt": row.attempt,
            "text": row.text, "cost_usd": row.cost_usd, "tokens_in": row.tokens_in,
            "tokens_out": row.tokens_out, "grade": row.grade, "filled_at": row.filled_at,
            "receipt_ids": list(row.receipt_ids), "failure": row.failure,
            "observations": [{"call": {"name": o.call.name, "arg": o.call.arg,
                                       "start": o.call.start},
                              "response_sha256": o.response_sha256, "output": o.output,
                              "truncated": o.truncated} for o in row.observations]}


def row_from_json(body: dict) -> OutcomeRow:
    return OutcomeRow(
        task_id=str(body["task_id"]), model_id=str(body["model_id"]), attempt=int(body["attempt"]),
        text=None if body["text"] is None else str(body["text"]),
        observations=tuple(Observation(call=ToolCall(name=o["call"]["name"], arg=o["call"]["arg"],
                                                     start=int(o["call"]["start"])),
                                       response_sha256=o["response_sha256"], output=o["output"],
                                       truncated=bool(o["truncated"]))
                           for o in body.get("observations", ())),
        cost_usd=float(body["cost_usd"]), tokens_in=int(body["tokens_in"]),
        tokens_out=int(body["tokens_out"]),
        grade=None if body["grade"] is None else float(body["grade"]),
        filled_at=float(body["filled_at"]), receipt_ids=tuple(body.get("receipt_ids", ())),
        failure=body.get("failure"))


def task_order(tasks: Iterable[TaskSpec], nonce: str) -> tuple[TaskSpec, ...]:
    """The window's task order, derived from its nonce alone (§4).

    Both arms run the identical order so that budget exhaustion zeroes the identical tail; if the
    order differed, one arm would be zeroed on a different — possibly easier — set of tasks and the
    duel would no longer be paired.

    Ordering is by a hash of `(nonce, task_id)` rather than by a seeded shuffle of the input list,
    so the result depends only on *which* tasks are in the slice and not on the order the caller
    happened to list them in. A shuffle would make the two arms' orders agree only as long as
    everything upstream also agreed, which is the `scoreable_ids` failure mode wearing a new hat.
    """
    return tuple(sorted(tasks, key=lambda t: _order_key(nonce, t.task_id)))


@dataclass(frozen=True)
class Scaffold:
    """The pinned harness (§3): one per arm, identical for king and challenger but for `conductor`.

    `grade` is the benchmark's own grader, `(task, answer) -> graded score in [0, 1]`. It is called
    on every worker response, not only at the end, because that verdict is what a `RETRY` decision
    is about: §3's whole argument for a sequential policy is that the Conductor *observes that a
    model failed* and escalates, which is strictly more information than choosing blind. The
    benchmark protocol that supplies it lands with the adapters in M2a; here it is a callable so
    that M1 stays offline.

    `clock` is the wall-clock seam (§8b.2), injected so the timeout can be tested in microseconds
    rather than in fifteen minutes.

    `inspect` is the benchmark's read channel (`benchmarks.base.inspector`), and None — the default —
    is today's behaviour bit for bit: no loop, one completion per delegate, `text == task.prompt`. It
    is LAST so that every existing positional construction still compiles, and it is a callable
    rather than a benchmark for the same reason `grade` is: this module stays offline and knows
    nothing about Docker.

    IT MAY ALSO ANSWER `None`, meaning "this read did not happen", and the loop then treats the reply
    as the submission. That is the guarded seam every production arm actually runs on
    (`simulate._GraderGuard.inspecting`), and it is why the return type is optional: an unguarded
    `SandboxError` from a read leaves this module, leaves `_arm`, and abandons the whole WINDOW —
    where the identical failure while grading drops one task from both arms. A miner can drive reads
    (they choose the model, and can route to one they control), so the unguarded shape was a
    permanent denial of service for the price of one registration.
    """

    catalog: Catalog
    conductor: Conductor
    worker: Worker
    grade: Callable[[TaskSpec, str], float]
    clock: Callable[[], float] = time.monotonic
    inspect: Callable[[TaskSpec, ToolCall, str], Observation | None] | None = None
    # The window's outcome table (§5.2c), shared by every arm of the window. None — the default at
    # every construction that predates it — buys every delegate live, which is the old behaviour
    # bit for bit. LAST, so every positional construction still compiles.
    table: OutcomeTable | None = None

    def run_window(self, tasks: Iterable[TaskSpec], *, nonce: str, budget_usd: float,
                   on_episode: Callable[[Episode], None] | None = None,
                   concurrency: int = 1) -> tuple[Episode, ...]:
        """Every task in the slice, in nonce order, until the allowance or the clock runs out.

        `budget_usd` is a number, not a source to poll: §4 requires the allowance to be read at one
        definite instant at window open, and the cheapest way to guarantee that is a signature with
        nothing to re-read. Exhaustion is visible rather than buried — a zeroed task carries
        `stopped_reason="budget_exhausted"` and no steps, which is what §5.8 asks the reveal to
        publish, since a verdict decided by funding rather than by routing must not read as skill.

        The episode that exhausts the allowance overruns it by at most one DELEGATE — which under
        the read channel is up to `MAX_READS_PER_DELEGATE + 1` provider calls, not one, because the
        allowance is checked between steps and a delegate's read loop runs inside a step. Measured:
        four $1.00 calls against a $0.50 remainder. The real hard stop is the miner's own key, which
        OpenRouter enforces. What this accounting is for is making exhaustion happen at the *same
        point* for both arms and be legible afterwards.

        THE PER-DUEL WALL CLOCK IS ENFORCED HERE, and here is the only place it can be (§8b.2). It
        bounds **one arm**, and one arm is exactly this call: §5.4 sizes ~250 tasks at ~15 min to
        about two hours, and under §5.2a the king's arm is a separate measurement shared by every
        duel in the window rather than half of any one duel. Without it the per-episode clock is not
        a bound at all — a model that stalls to `EPISODE_WALL_CLOCK_SECONDS` on all ~250 tasks holds
        the validator for ~62 hours, which is the denial of service §8b.2 exists to refuse.

        Checked BETWEEN TASKS and ahead of the allowance, as `run_episode` checks its own clock
        ahead of the budget: an episode already running may therefore overrun by up to the
        per-episode clock, so the honest bound on an arm is duel + episode, and
        `DUEL_WALL_CLOCK_SECONDS` must exceed `EPISODE_WALL_CLOCK_SECONDS` or an arm is abandoned
        before its first episode could ever time out (tests/test_config.py). An abandoned task
        carries `DUEL_WALL_CLOCK_REASON`, not the episode clock's `"wall_clock"`: §5.8 wants a
        verdict decided by something other than routing visible in the reveal, and one string for
        both clocks would render a pathological model and an overrunning validator as the same row.

        `on_episode` is called with each episode as it lands, for callers that must not lose an
        arm's work to a crash. It is observation and changes nothing: it runs after the episode is
        recorded and the allowance decremented, so it cannot alter what was spent or scored.

        `concurrency` RUNS TASKS IN CHUNKS, AND IS WHY AN ARM FITS ITS CLOCK AT ALL. §5.4 sizes ~250
        tasks at ~15 min each to about two hours *with parallelism*; measured serially these
        episodes take 45 s to 1.4 min, which is 3-6 h per arm against `DUEL_WALL_CLOCK_SECONDS` —
        so a serial arm cannot finish a full slice and every window would end on the wall clock.

        IT IS NOT A ONE-WAY DOOR, unlike the constants in `config.py`'s header. An episode is
        independent of every other episode: same state, same catalog, same history, same grader. So
        concurrency changes how many are in flight and nothing a miner trains against — the episode
        shape is untouched, and `concurrency=1` reproduces the serial run exactly (pinned by test).

        WHAT IT DOES CHANGE is the allowance overrun. Serially the arm overruns by at most one
        DELEGATE; with `k` in flight, up to `k` episodes can be dispatched against the same
        `remaining` before any of them reports, so the bound is `k` DELEGATEs. That is a widening of
        a bound this method already documents rather than a new failure: exhaustion is still
        recorded per task, still legible in the reveal (§5.8), and the real hard stop is still the
        miner's own key, which OpenRouter enforces.

        THE CALLER STILL OWES ONE THING (§8b.2, last paragraph): if the truncated arm is the KING's,
        §5.2a makes the zeroed tail shared by every duel in the window, so the owner's own slowness
        would decide all of them at once. Such a window is to be treated as the power gate treats a
        dead window — no duels, no shot spent — rather than scored. The scaffold makes that
        detectable and does not decide it: the check is
        `any(e.result.stopped_reason == DUEL_WALL_CLOCK_REASON for e in king_episodes)`.
        """
        remaining = budget_usd
        deadline = self.clock() + DUEL_WALL_CLOCK_SECONDS
        ordered = list(task_order(tasks, nonce))
        # Filled BY POSITION, never appended, so a zeroed tail and a chunk of finished episodes
        # cannot interleave: §4 requires the published order to be the nonce order, and a list built
        # by append would put tasks zeroed during a scan ahead of the chunk still in flight.
        results: list[Episode | None] = [None] * len(ordered)
        zeroed: list[int] = []
        runs = spent = 0.0
        index = 0
        while index < len(ordered):
            # THE ALLOWANCE SIZES THE BATCH, exactly as `Validator._arm` does it. Sizing from
            # `concurrency` alone and handing every member the same `remaining` means `k` episodes
            # start against a budget that covers one — so a batch wider than the remaining slice
            # enforces no budget at all and never records the exhausted tail §4 requires and §5.8
            # publishes. The first batch is one episode because before it there is no price.
            # An UNBOUNDED allowance imposes no bound, so it sizes nothing: `inf // mean` is not a
            # batch width, and `int()` of it raises rather than returning something large. The
            # archetype sweep runs with `budget_usd=inf` deliberately, so this is a live path.
            mean = (spent / runs) if runs else None
            if not math.isfinite(remaining):
                width = max(1, concurrency)
            elif mean is None or mean <= 0.0:
                width = 1
            else:
                width = max(1, min(max(1, concurrency), int(remaining // mean)))
            chunk: list[int] = []
            while index < len(ordered) and len(chunk) < width:
                task = ordered[index]
                if self.clock() >= deadline:
                    results[index] = _unreached(task, DUEL_WALL_CLOCK_REASON)
                    zeroed.append(index)
                elif remaining <= 0.0:
                    results[index] = _unreached(task, "budget_exhausted")
                    zeroed.append(index)
                else:
                    chunk.append(index)
                index += 1
            # Reported too. `on_episode` is how a caller persists an arm, and a row it never sees is
            # a row missing from the record — M3a's episode files silently omitted every zeroed task
            # until this was fixed, so an arm cut short read as an arm that was never cut.
            for position in zeroed:
                if on_episode is not None:
                    on_episode(results[position])
            zeroed.clear()
            if not chunk:
                continue
            if len(chunk) == 1:
                results[chunk[0]] = self.run_episode(ordered[chunk[0]],
                                                     budget_remaining=remaining)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                    for position, episode in zip(chunk, pool.map(
                            lambda i: self.run_episode(ordered[i], budget_remaining=remaining),
                            chunk)):
                        results[position] = episode
            # Accounted and reported IN NONCE ORDER once the chunk has drained, so `remaining` and
            # the callback see the same sequence a serial run would produce.
            for position in chunk:
                episode = results[position]
                remaining -= episode.result.spend_usd
                spent += episode.result.spend_usd
                runs += 1
                # OBSERVATION ONLY, and after the episode is already counted, so a raising callback
                # cannot lose money that was spent or leave `remaining` disagreeing with the results.
                # It exists because an arm is hours of paid model calls whose results would
                # otherwise be written once at the end: §8b.5 checkpoints per task for the same
                # reason, and a crash without this buys every finished episode again.
                if on_episode is not None:
                    on_episode(episode)
        return tuple(episode for episode in results if episode is not None)

    def run_episode(self, task: TaskSpec, *, budget_remaining: float) -> Episode:
        """One task: render, ask, act, grade, repeat — up to `MAX_STEPS` (§3).

        The score is the best result any step achieved, because STOP means "submit the best result
        so far" (§3): a Conductor that finds an answer and then explores further is not punished for
        the exploring, only charged for it.
        """
        tracker = ParseFailureTracker()
        history: list[StepRecord] = []
        requests: list[WorkerRequest] = []
        deadline = self.clock() + EPISODE_WALL_CLOCK_SECONDS
        stopped_reason = "max_steps"
        best = 0.0
        spent = 0.0
        hits = fills = 0

        for step_index in range(MAX_STEPS):
            if self.clock() >= deadline:
                # §8b.2: abandon the episode and score the task 0 for this arm. The spend stands —
                # the money was spent — so a model that stalls is charged for stalling.
                stopped_reason, best = "wall_clock", 0.0
                break
            if spent >= budget_remaining:
                stopped_reason = "budget_exhausted"
                break

            prompt = render(ConductorState(self.catalog, task, tuple(history)))
            raw = self.conductor.act(prompt)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            action = tracker.resolve(raw, self.catalog)

            if tracker.consecutive:
                # Refused and counted (§2). The step is recorded — a Conductor that cannot see it
                # fumbled has no reason to stop fumbling — and `resolve` returns STOP on the third
                # consecutive failure, which ends the episode having decided nothing.
                history.append(_refused(step_index, digest, raw))
                if action is None:
                    continue
                stopped_reason = "parse_failures"
                break

            if action.kind == "stop":
                # Recorded as a step even though it invokes nothing: the published traces (D15) are
                # what miners train on, and a trajectory truncated before its final decision cannot
                # teach a model when to stop. Always terminal, so it is never rendered into a later
                # prompt.
                history.append(StepRecord(step_index=step_index, rendered_state_digest=digest,
                                          raw_output=raw, action=action, model_id=None,
                                          success=False, cost_usd=0.0, parse_failed=False))
                stopped_reason = "stop"
                break

            # THE DELEGATE'S KEY (§5.2c): the validated model, and which attempt at it this is
            # within the episode — a RETRY of the same model is attempt 2, a fresh draw, never the
            # first draw handed back. Validated HERE as well as in `_call_worker`, because the key
            # must name the catalog's id and not the parser's.
            model_id = validate(self.catalog, action.model_id)
            attempt = 1 + sum(1 for step in history if step.model_id == model_id)
            row = None if self.table is None else self.table.get((task.task_id, model_id, attempt))
            if row is not None:
                # A HIT: the stored draw, the stored grade, the stored price — no provider, no
                # grader. The arm is still charged the price (`_replay`), so §4 does not know the
                # table exists; the owner's outlay is what changed.
                request, text, cost, replayed, failure = self._replay(task, row, requests)
                score = (0.0 if row.grade is None else row.grade) if replayed else 0.0
                hits += 1 if replayed else 0
            else:
                delegate = self._call_worker(task, model_id, requests)
                request, text, cost = delegate.request, delegate.text, delegate.cost_usd
                failure = delegate.failure
                # OUTSIDE `_call_worker`, and therefore outside its except (§8b.3): a worker failing
                # is data about that rung, but a grader or sandbox failing is a fact about the
                # validator's infrastructure. Swallowing the second into a "failed step" would inject
                # the owner's Docker problems into a miner's score; it must reach the caller, which
                # drops the task from BOTH arms (§6.3c). And because it is raised BEFORE the `put`
                # below, a grader failure is never a row: the table stores only what was graded.
                score = 0.0 if text is None else _graded(self.grade(task, text), task)
                if self.table is not None and delegate.outcome:
                    self.table.put(OutcomeRow(
                        task_id=task.task_id, model_id=model_id, attempt=attempt, text=text,
                        observations=request.observations, cost_usd=cost,
                        tokens_in=sum(int(getattr(r, "tokens_in", 0)) for r in delegate.receipts),
                        tokens_out=sum(int(getattr(r, "tokens_out", 0)) for r in delegate.receipts),
                        grade=None if text is None else score, filled_at=time.time(),
                        receipt_ids=tuple(str(getattr(r, "call_id", "")) for r in delegate.receipts),
                        failure=delegate.failure))
                    fills += 1
            spent += cost
            best = max(best, score)
            # `model_id` is taken from the logged request, not from the action, so the trace records
            # what was INVOKED next to what was ASKED FOR and the §2.1 audit can assert they match
            # rather than assume it (`types.StepRecord`). `success` means SOLVED, not "the provider
            # answered": it is the signal a RETRY decision is made on (§3), and partial credit is a
            # variance-reduction device for the duel (§5.1), not a report that the work is done.
            # `request` is the LAST row of the delegate's loop, so its transcript is the whole one:
            # every earlier row is one of its prefixes, which is what makes the §2.1 sentence
            # recomputable from the published trace alone rather than from `Episode.requests`, which
            # the daemon that replaces `run_window` does not keep (`validator._arm`).
            history.append(StepRecord(step_index=step_index, rendered_state_digest=digest,
                                      raw_output=raw, action=action, model_id=request.model_id,
                                      success=score >= 1.0, cost_usd=cost, parse_failed=False,
                                      observations=request.observations, failure=failure))

        result = EpisodeResult(task_id=task.task_id, benchmark=task.benchmark,
                               steps=tuple(history), graded_score=best, spend_usd=spent,
                               stopped_reason=stopped_reason)
        return Episode(result=result, requests=tuple(requests), hits=hits, fills=fills)

    def _replay(self, task: TaskSpec, row: OutcomeRow,
                log: list[WorkerRequest]) -> tuple[WorkerRequest, str | None, float, bool,
                                                   str | None]:
        """A hit (§5.2c): the row's reply and price, no provider and no grader — but the same debit.

        THE AUDIT ROWS ARE STILL LOGGED, one per turn of the delegate the row records, with
        `text == tools.compose(task.prompt, observations[:k])` exactly as the live loop logs them —
        so the §2.1 sentence holds over a replayed arm by the same recomputation, and a reader of
        `Episode.requests` cannot tell a hit from a miss, which is the point: the arm made the same
        choice and is accountable for the same bytes.

        THE DEBIT GOES THROUGH THE SEAM'S `charge`, where the seam has one, and its refusal is the
        live call's refusal: an exhausted allowance makes this a dead step at zero cost, counted in
        `unfunded_calls`, and NOT the row's outcome — a hit a miner could not pay for is not a
        model's answer they got for free. The fourth element says which happened; the fifth is
        the reason the step has no answer, if it has none: the refusal's, or the fill's.
        """
        request = None
        for k in range(len(row.observations) + 1):
            observations = row.observations[:k]
            request = WorkerRequest(task_id=task.task_id, model_id=row.model_id,
                                    text=tools.compose(task.prompt, observations),
                                    observations=observations)
            log.append(request)
        charge = getattr(self.worker, "charge", None)
        if charge is not None:
            try:
                charge(row.model_id, request.text, row.cost_usd, tokens_in=row.tokens_in,
                       tokens_out=row.tokens_out, response_text=row.text)
            except Exception as exc:  # noqa: BLE001 — the meter refused: a dead step, not the row
                return request, None, 0.0, False, f"refused: {exc}"
        return request, row.text, row.cost_usd, True, row.failure

    def _call_worker(self, task: TaskSpec, model_id: str,
                     log: list[WorkerRequest]) -> Delegate:
        """THE ONLY PLACE A WORKER MODEL IS INVOKED (§2.1). Returns the last row, text and cost.

        TWO INPUTS, NEITHER OF WHICH CAN CARRY CONDUCTOR OUTPUT: `task` is the benchmark's own
        record, and `model_id` is re-validated against the window's committed snapshot here rather
        than trusted from the parser. (`log` is where the audit rows go, not an input.) The params
        are the pinned module constant. The logged row IS the argument that is sent, so the audit
        records what left rather than what was intended.

        ONE DELEGATE IS A BOUNDED READ LOOP, NOT NECESSARILY ONE COMPLETION (`tools.py`). If the
        reply's last non-blank line parses as one of the benchmark's declared read-only verbs, it is
        executed in the benchmark's own environment, appended to the transcript, and the SAME model
        is asked again with `compose(task.prompt, observations)`. Anything else is the submission.
        A benchmark that declares no tools, or a worker that never emits a call, therefore behaves
        byte-identically to before this loop existed — `compose(p, ()) == p`.

        THE TRANSCRIPT IS A LOCAL VARIABLE, WHICH IS THE ENFORCEMENT (§2.1). It is born and dies
        inside this call, so it cannot outlive a delegate and no Conductor decision can extend it;
        `self.inspect` receives a task, a worker-authored call and a digest of the worker's own
        reply, and there is nothing in that signature a Conductor could reach.

        TWO DETAILS THE LOOP WOULD GET WRONG IF WRITTEN NAIVELY, both load-bearing:

          * **`self.inspect` sits OUTSIDE the `except`.** A worker failing is data about that rung; a
            Docker failure is a fact about the validator, and swallowing the second into a bad rung
            would inject the owner's infrastructure into a miner's score — the one thing the narrow
            scoping of this `except` exists to prevent (§8b.3, §6.4a).
          * **a mid-loop failure returns the ACCUMULATED cost, not zero.** `_priced` is applied per
            turn, so money already spent on turns 1-2 stays in `spend_usd`; returning 0.0 would price
            an arm that answered partly for free.

        `text` is None when the provider errored, refused or timed out — §8b.3's outcome, which the
        caller records as a failed step and continues past. The except is broad on purpose: a
        validator two hours into a 500-episode window must not die because a provider invented a new
        exception class, and narrowing it to `WorkerError` would make that property depend on every
        seam remembering to wrap. It is scoped to this one call for exactly that reason — nothing
        else, least of all grading or a read, is inside it.
        """
        model_id = validate(self.catalog, model_id)
        observations: tuple[Observation, ...] = ()
        spent = 0.0
        # A seam that mints receipts hands back the ones THIS THREAD minted during the delegate
        # (`gateway.GatewayWorker.minted`), so the row the caller stores can name them. Cleared
        # first so a receipt from an earlier delegate on this thread is never attributed here.
        minted = getattr(self.worker, "minted", None)
        if minted is not None:
            minted()
        while True:
            request = WorkerRequest(task_id=task.task_id, model_id=model_id,
                                    text=tools.compose(task.prompt, observations),
                                    observations=observations)
            log.append(request)
            try:
                answer, cost = self.worker.complete(request.model_id, request.text,
                                                    dict(WORKER_PARAMS))
            except NotAnOutcome as exc:
                # The meter refused (§4), which says nothing about the rung: a dead step for this
                # arm, and never a row in the window's table (§5.2c).
                return Delegate(request, None, spent, minted() if minted else (), outcome=False,
                                failure=f"refused: {exc}")
            except Exception as exc:  # noqa: BLE001 — a dead rung is data about it, not a dead window
                return Delegate(request, None, spent, minted() if minted else (),
                                failure=f"{type(exc).__name__}: {exc}")
            spent += _priced(cost, request.model_id)
            observed = self._read(task, answer, observations)
            if observed is None:
                return Delegate(request, answer, spent, minted() if minted else ())
            observations += (observed,)

    def _read(self, task: TaskSpec, answer: str,
              observations: tuple[Observation, ...]) -> Observation | None:
        """One read, or None meaning `answer` is the submission. The bound is checked here, once.

        Four things have to be true before a container is started, and each refuses a different
        mistake: the harness has a read channel at all (`inspect`), the BENCHMARK declared this verb
        (`task.tools`, which is also what `render._task` shows the Conductor, so the declaration and
        the action space are one fact), the reply really is a call, and the delegate has reads left.
        """
        if self.inspect is None or len(observations) >= MAX_READS_PER_DELEGATE:
            return None
        call = tools.parse_tool(answer)
        if call is None or call.name not in task.tools:
            return None
        return self.inspect(task, call, hashlib.sha256(answer.encode()).hexdigest())


def _priced(cost: object, model_id: str) -> float:
    """The provider's dollars, refused unless they are dollars (§4, §5.1b).

    THE ATTACK THIS CLOSES: a negative cost is paid twice. `final_b` subtracts `λ_b · spend / C_b`,
    so a minus sign is an unbounded score bonus; and `run_window` does `remaining -= spend`, so the
    same figure REFILLS the allowance and §4's exhaustion never fires. Measured before it was
    closed: three episodes at −$60 against a $1.00 window all ended `max_steps`, having spent
    nothing the budget could see. Not miner-controlled today and not reachable from a well-behaved
    provider, which is exactly why it needed a comparison rather than a remark — nothing downstream
    would have made a noise.

    Written as `not >= 0` rather than `< 0` so the one comparison also refuses NaN, which poisons
    the arm's mean the same way and by the same silence.
    """
    dollars = float(cost)
    if not dollars >= 0.0:
        raise ScaffoldError(
            f"{model_id} reported a cost of {dollars}: spend is subtracted from the window "
            "allowance and priced into `final`, so a figure that is not dollars corrupts both")
    return dollars


def _graded(score: object, task: TaskSpec) -> float:
    """The benchmark's verdict, refused unless it is in [0, 1] (§5.1, §8b.3).

    `benchmarks.base.Grade` already refuses an out-of-range score at construction and every adapter
    returns one — but `Scaffold.grade` is a bare callable, so that guarantee holds only for as long
    as every grader goes through `Grade`. One that returns 1.7 fails nowhere: `quality_b` is a mean
    of these, so it lands in `final` as a win that no routing produced and moves a crown.

    Refused, not clamped: a clamp would silently rescale one benchmark's quality against the others,
    which is the same corrupted verdict with the evidence removed. The refusal is loud on purpose —
    an adapter that returns 1.7 once will do it on every task it grades, so the window is already
    lost, and losing it is cheaper than a coronation nobody can take back (`ScaffoldError`).
    """
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ScaffoldError(
            f"the grader for {task.benchmark} scored {task.task_id} at {value}, outside [0, 1]; "
            "`quality_b` is a mean of these and nothing downstream re-checks it")
    return value


def _order_key(nonce: str, task_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{nonce}|{task_id}".encode()).hexdigest(), task_id


def _refused(step_index: int, digest: str, raw: str) -> StepRecord:
    return StepRecord(step_index=step_index, rendered_state_digest=digest, raw_output=raw,
                      action=None, model_id=None, success=False, cost_usd=0.0, parse_failed=True)


def _unreached(task: TaskSpec, reason: str) -> Episode:
    """A task the arm never reached: 0, no steps, no calls, and it says WHICH limit stopped it.

    Two limits produce this row — the allowance (§4) and the per-duel clock (§8b.2) — and they are
    two different facts: one is about the miner's funding, the other about the validator's clock.
    §5.8 requires both in the reveal rather than in an aggregate, so the reason is a parameter
    instead of a constant."""
    return Episode(result=EpisodeResult(task_id=task.task_id, benchmark=task.benchmark, steps=(),
                                        graded_score=0.0, spend_usd=0.0, stopped_reason=reason),
                   requests=())
