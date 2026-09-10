"""The v3 contract — the objects every other module codes against (docs/WHITEPAPER.md).

Frozen throughout, and small on purpose. Four of these types are load-bearing for the invariant the
whole design rests on (§2.1: *the only things that reach a worker model are the pinned task text, a
validated model ID, and bytes the benchmark's own environment returned in answer to that same
worker's own read-only requests*):

  * `TaskSpec.prompt` is the benchmark's own task statement, and it is what the scaffold forwards
    BYTE-FOR-BYTE. Nothing the Conductor produces is ever concatenated with it.
  * `Action` has room for a kind and a model ID and nothing else. A solution cannot be smuggled
    through a type that cannot carry one — which is why the parser returns this rather than, say,
    a dict of whatever it found.
  * `ToolCall` is the same guard on the other side: it is worker-authored, one line, and validated
    against `TOOL_NAMES`, so the read channel cannot carry a Conductor byte either.
  * `Observation` has exactly one constructor in the system, `tools.observe`. Nothing anywhere
    accepts a caller-supplied observation, so what a worker sees is composable only from the pinned
    prompt and what the environment itself answered.

`Catalog` is a per-window snapshot rather than a live fetch because model availability and prices
move: king and challenger evaluated minutes apart would otherwise face different action spaces and
the duel would not be paired (§6.2).
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("delegate", "retry", "stop")

# The read-only verbs a WORKER may issue (`tools.py`). Here rather than in `tools.py` for the reason
# `KINDS` is here: `ToolCall` validates against it at construction, exactly as `Action` validates
# against `KINDS`, and a type that imported its own validator's module would be a cycle.
TOOL_NAMES = ("READ", "LIST", "FIND")


@dataclass(frozen=True)
class CatalogEntry:
    """One model in the window's committed OpenRouter snapshot (§6.2).

    Prices are dollars per million tokens, as OpenRouter reports them. They are in the contract —
    and in the prompt — because a budget-aware policy has to see prices to be budget-aware, and
    because `final_b` prices spend (§5.1b): a Conductor that cannot see what a call costs cannot be
    scored on quality-per-dollar.
    """

    model_id: str
    price_in_per_mtok: float
    price_out_per_mtok: float
    context_length: int


@dataclass(frozen=True)
class Catalog:
    """The frozen catalog snapshot — the window's entire action space.

    ORDER IS PART OF THE SNAPSHOT. `entries` is a tuple and everything that renders or iterates the
    catalog must go through it, never through `ids`: set iteration order differs per process, and a
    prompt built from a set is a prompt that differs per process. That is the `scoreable_ids` class
    of bug this project has already paid for, and it would break both the render determinism
    requirement (M0 exit 5) and the cacheable catalog prefix (M0 exit 7).
    """

    entries: tuple[CatalogEntry, ...]

    @property
    def ids(self) -> frozenset[str]:
        """Membership only — derived, so it can never go stale against `entries`."""
        return frozenset(e.model_id for e in self.entries)

    def compact(self) -> str:
        """`id / $in / $out / ctx` per model, one per line, in snapshot order (§3).

        Compaction is what makes ~300 models affordable in context: the catalog is constant within a
        window, so a stable prefix is prefilled once per window instead of once per Conductor call.
        Rows only — the units header and the surrounding template belong to the render, which owns
        the `[catalog | task | history]` order the caching depends on.
        """
        return "\n".join(
            f"{e.model_id} / {e.price_in_per_mtok:g} / {e.price_out_per_mtok:g} / "
            f"{e.context_length}"
            for e in self.entries)


@dataclass(frozen=True)
class Action:
    """One Conductor turn, after parsing. Never executed, never forwarded (§2).

    The two invariants below are enforced here rather than at the call sites because they are what
    keeps a STOP from reaching a worker: a stop that carried a model ID would be a delegate wearing
    a stop's name, and the scaffold has no reason to look twice at a validated action.
    """

    kind: str                     # one of KINDS
    model_id: str | None          # None for "stop", a catalog member otherwise

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.kind!r} is not one of {KINDS}")
        if (self.model_id is None) != (self.kind == "stop"):
            raise ValueError(f"{self.kind!r} action with model_id={self.model_id!r}")


@dataclass(frozen=True)
class ToolCall:
    """One read-only request, after parsing — issued by a WORKER, never by the Conductor (§2.1).

    The same guard `Action` gives on the other side of the seam. `name` is validated here rather than
    at the call sites so that a `ToolCall` is constructible only from something `tools.parse_tool`
    accepted; and `arg` is held to ONE line, non-empty, with no NUL, so the type cannot carry a
    multi-line payload into the transcript that a later prompt is composed from. `str.splitlines`
    rather than a scan for `\\n`: it breaks on eleven characters, which is the wider and therefore
    safer direction, and it is the same fact `action.parse` already relies on.
    """

    name: str                     # one of TOOL_NAMES
    arg: str                      # a repository-relative path, or a fixed search string
    start: int = 1                # READ only, 1-based

    def __post_init__(self) -> None:
        if self.name not in TOOL_NAMES:
            raise ValueError(f"{self.name!r} is not one of {TOOL_NAMES}")
        if self.arg.splitlines() != [self.arg] or "\x00" in self.arg:
            raise ValueError(f"a tool argument must be one non-empty line: {self.arg!r}")
        if self.start < 1:
            raise ValueError(f"{self.name} from line {self.start}: lines are numbered from 1")


@dataclass(frozen=True)
class Observation:
    """What the benchmark's own environment answered one worker's own read with (§2.1).

    Constructed in exactly one place — `tools.observe` — which is what makes "no Conductor byte can
    be in here" a property of the type rather than a convention at the call sites.

    `response_sha256` is the digest of the worker reply the call was parsed from, so the chain from
    a recorded observation back to the reply that asked for it is checkable by anyone holding the
    replies (a greedy replay re-derives them; `TEMPERATURE` is pinned at 0). It lives here, on an
    object built AFTER a reply, rather than on `WorkerRequest`, which is deliberately written into
    the log BEFORE its call is made.

    `truncated` is carried rather than derived because a replay must be able to tell a whole file
    from the first 200 lines of one, and silence would be a lie about what the worker was shown.
    """

    call: ToolCall
    response_sha256: str
    output: str                   # what the environment returned, capped at OBSERVATION_BYTES
    truncated: bool


@dataclass(frozen=True)
class Environment:
    """Where a benchmark's read-only tool calls run: its own per-task image, and the root they see.

    The root is the confinement boundary and it is the benchmark's to name — R2E-Gym answers
    `/testbed`, which is precisely what puts the graded `/r2e_tests` out of reach (`r2egym.py`).
    """

    image: str
    root: str


@dataclass(frozen=True)
class StepRecord:
    """One step of an episode — and one row of the decision traces the owner publishes (D15).

    `action` is what the Conductor ASKED for; `model_id` is what the scaffold actually invoked.
    They are equal by construction, and recording both is what lets the §2.1 audit assert it rather
    than assume it. Both are None on a parse failure, and `model_id` is also None on STOP.

    `rendered_state_digest` rather than the rendered state itself: the state is thousands of tokens
    dominated by a catalog that is constant across every step of the window, so storing it per step
    would multiply the published traces by two orders of magnitude for no information. The digest is
    what proves a replay saw the same state.

    `observations` is the delegate's whole read transcript, IN FULL AND NOT AS A DIGEST, and it is
    the one place a state is stored rather than hashed. The reason is that it is the only new input
    §2.1 has: a digest lets a holder of a claimed transcript check it, and nobody would hold one, so
    the sentence

        request.text == tools.compose(task.prompt, step.observations[:k])

    would stop being a recomputation and become an assertion. It also makes the step legible in the
    published traces (D15) — a delegate that read three files and one that read none are otherwise
    the same row — and it is what re-derives the `prompt_hash` in every gateway receipt this step
    paid for, since one delegate is now up to `MAX_READS_PER_DELEGATE + 1` provider calls collapsed
    into one row. Empty on every benchmark that declares no tools, which is every row today.
    """

    step_index: int
    rendered_state_digest: str
    raw_output: str               # verbatim, including any REASON the parser discarded
    action: Action | None
    model_id: str | None
    success: bool
    cost_usd: float
    parse_failed: bool
    observations: tuple[Observation, ...] = ()
    # WHY a delegate came back with no answer: the provider's error as the seam reported it, or
    # the meter's refusal — None when it answered, and on every step that is not a delegate. Added
    # after the testnet rehearsal of 2026-09-08 published a window whose live buys failed 39% of
    # the time and whose trace could not say whether that was a 429 storm, a routing refusal or a
    # timeout. Rendered nowhere (§2.1: the Conductor sees `failed`, not the reason); published in
    # the traces (D15) and stored in the outcome table's failed rows, so a replayed failure
    # carries the reason the fill saw.
    failure: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """One task, one arm. `graded_score` is partial credit wherever the benchmark supports it.

    Graded, not pass/fail (§5.1): binary outcomes carry maximum variance, and at the ~250 tasks per
    arm a duel can afford (§5.4) that variance is the difference between a resolvable comparison and
    an unresolvable one.

    `stopped_reason` is one of `stop` (the Conductor said so), `max_steps`, `parse_failures`,
    `wall_clock`, `duel_wall_clock`, `budget_exhausted`, `futile` (§5.4: the arm could no longer
    win) or `excluded` (§6.3c had already dropped the task, so it was not bought). It is recorded rather than derived
    because the last four are the failure modes invisible in a score — a challenger that times out
    and one that routes badly both just look like a low number (the build plan M4 exit 4). The two clocks
    are kept apart on purpose (§8b.2): `wall_clock` is an episode that stalled, which is a fact
    about the challenger, while `duel_wall_clock` is a task the arm never reached, which is a fact
    about the validator's own clock — one string for both would render a pathological model and an
    overrunning validator as the same row.
    """

    task_id: str
    benchmark: str
    steps: tuple[StepRecord, ...]
    graded_score: float
    spend_usd: float
    stopped_reason: str


@dataclass(frozen=True)
class TaskSpec:
    """A benchmark task as the scaffold will forward it.

    `prompt` is the benchmark's own task statement and is what every worker request must byte-equal
    (§2.1). It is a plain immutable string precisely so that "the scaffold constructs every worker
    request from this" is checkable: there is nothing here for a Conductor to have influenced.

    `group` is the cluster this task belongs to — R2E-Gym's repository, and None wherever the
    benchmark's tasks are genuinely independent. It reaches no worker and no prompt; it is an input
    to the VERDICT, because `duel` resamples whole groups (see that module: tasks from one
    repository move together, and an iid bootstrap publishes an interval too narrow for it). None is
    the default and means "its own cluster", which is the resample that shipped — an adapter that
    has no honest grouping must supply none rather than invent one.
    """

    task_id: str
    benchmark: str
    prompt: str
    tools: tuple[str, ...]
    group: str | None = None


@dataclass(frozen=True)
class OutcomeRow:
    """One worker outcome the window bought once and every arm replays (§5.2c, D17).

    The unit is one DELEGATE — one `Scaffold._call_worker`, read loop included — keyed by
    `(task_id, model_id, attempt)`, where `attempt` is 1 for the first delegate to that model on that
    task within an episode and counts up on each RETRY of the same model. Across arms the n-th
    attempt of a key is one draw, which is the whole point: a Conductor never changes what a worker
    answers, only which keys get drawn and in what order, so a table of real draws is a sufficient
    statistic for scoring any Conductor on the slice.

    `text` is None for a provider failure — an OUTCOME (§8b.3), stored so every arm sees the same
    dead rung rather than a fresh coin flip on the provider's availability; `grade` is None on the
    same rows because nothing was graded. A grader failure is never a row: the caller stores only
    after the benchmark's own grader returned (§6.3c), so the table can hold no zero that was really
    the owner's dead sandbox.

    `receipt_ids` name the gateway receipts the fill minted (one per turn of the read loop) and the
    token counts are their sums; `filled_at` is a wall-clock second, the drift audit's input.
    """

    task_id: str
    model_id: str
    attempt: int
    text: str | None
    observations: tuple[Observation, ...]
    cost_usd: float
    tokens_in: int
    tokens_out: int
    grade: float | None
    filled_at: float
    receipt_ids: tuple[str, ...] = ()
    failure: str | None = None       # the fill's reason when `text` is None (`StepRecord.failure`)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.model_id, self.attempt)

    @property
    def failed(self) -> bool:
        """A stored provider failure (§8b.3)."""
        return self.text is None
