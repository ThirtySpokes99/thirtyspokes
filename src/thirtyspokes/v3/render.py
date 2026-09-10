"""The pinned state -> prompt template (docs/WHITEPAPER.md §3, the build plan M0 exits 5-7).

THE SECTION ORDER IS `[catalog | task | history]` AND IT IS A SERVING-COST DECISION, NOT A LAYOUT
ONE. The catalog is constant for a whole window (§6.2 freezes it precisely so king and challenger
face the same action space), while the task and the history change on every single step. Putting the
constant part first makes the prompt's leading ~15-20k tokens a stable prefix, so a serving stack
prefills it once per window instead of once per Conductor call. The Conductor runs ~5,000-15,000
times per duel (§D7), so across a duel this is the difference between a serving bill that is
negligible and one that dominates the mechanism's economics. Putting the varying state first would
forfeit it silently — nothing would break, the bill would just be two orders of magnitude larger —
which is why `cacheable_prefix` is a function with a test on it rather than a convention.

DETERMINISM IS THE OTHER LOAD-BEARING PROPERTY (M0 exit 5). Identical state must render to identical
bytes *across processes*, because the render is what a miner trains against and what
`StepRecord.rendered_state_digest` proves a replay saw. This repo has already paid for the failure
mode: `koth/matrix.py::scoreable_ids` iterated a set, so per-process string-hash randomisation made
two validators disagree while every seed was identical. Three rules keep it out of here:

  * nothing is iterated that is not already ordered — the catalog comes from `Catalog.entries` (a
    tuple, order is part of the snapshot), tools from `TaskSpec.tools` (a tuple), history from a
    tuple. No set, no dict, anywhere on this path.
  * every float is formatted at a pinned precision. `repr` of a float carries the low digits of
    accumulated arithmetic noise, so two runs that spent the same money would otherwise render
    different prompts.
  * spend is DERIVED from the history rather than passed alongside it, so the total shown can never
    disagree with the steps shown above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .config import MAX_STEPS
from .types import Catalog, StepRecord, TaskSpec

# Pinned (§1: "pinned system prompt"), and one-way for the same reason `MAX_STEPS` is: every miner
# trains a model to act under these exact words, so rewriting them re-scores everyone against
# instructions they never saw. It states the §2.1 invariant in the second person because that is the
# behaviour being asked for -- the Conductor never answers, it chooses who answers -- and states the
# grammar because a parse failure is a step spent deciding nothing (§8b.2).
SYSTEM_PROMPT = (
    "You are a Conductor. You never answer the task yourself. You choose which model answers it.\n"
    "The task statement below is forwarded to that model verbatim: you cannot edit it, add to it,\n"
    "or contribute any part of the answer. Your entire job is choosing where the work goes and\n"
    "when to stop.\n"
    "\n"
    "End every reply with exactly one action, on its own final line:\n"
    "  DELEGATE <model_id>   send the task to this model\n"
    "  RETRY <model_id>      send the task to this model after seeing an outcome\n"
    "  STOP                  submit the best result so far\n"
    "\n"
    "Lines before it may begin with REASON; they are internal to you and are discarded. Anything\n"
    "else, including text after the action, is refused and the step is wasted. <model_id> must\n"
    "appear in the catalog below, spelled exactly as it is there.\n"
)

_CATALOG_HEADER = (
    "# CATALOG\n"
    "The models available this window: id / $ per million input tokens / "
    "$ per million output tokens / context length.\n"
)

# Cost per Conductor call is what a budget-aware policy is choosing against, and the cheap tier is
# priced in millionths of a dollar per call while a frontier episode runs to dollars. Six decimals
# spans that range without ever rendering a real call as free, and a FIXED precision is what makes
# two runs that spent the same money render the same bytes.
_USD = ".6f"

# The catalog's share of the reference model's 262K context (§M0 criterion 4 sizes the snapshot at
# ~15-20k tokens for ~300 models). 32K leaves the catalog room to roughly double as OpenRouter grows
# while still spending under an eighth of the window, so task text, trajectory history and the
# generation itself are never squeezed by it. Enforced by measurement, not assumption: M0 exit 6.
CATALOG_TOKEN_BUDGET = 32_000

# THE TOKENISER SEAM. The real count needs the reference model's tokeniser, which is pinned at M0
# but not vendored here (M0 runs offline, and `config.REFERENCE_REVISION` is still the UNPINNED
# sentinel). Until it lands, `token_estimate` is a deliberately PESSIMISTIC chars-per-token ratio:
# BPE averages ~4 chars/token on English prose, but a catalog row is vendor slugs, digits and
# punctuation, which split far more finely. 2.0 is chosen to err high, because the only failure this
# guard exists to catch is an underestimate letting an oversized catalog through.
CHARS_PER_TOKEN = 2.0


@dataclass(frozen=True)
class ConductorState:
    """Everything the Conductor sees on one step (§3's state, minus what is derivable from it).

    `spend so far` and `step index` are in §3's list but are not fields here: both are functions of
    `history`, and a field that duplicates the history is a field that can contradict the history.
    """

    catalog: Catalog
    task: TaskSpec
    history: tuple[StepRecord, ...] = ()


def cacheable_prefix(catalog: Catalog) -> str:
    """The part of the prompt that is constant for the whole window — everything before the task.

    Split out so the property can be tested rather than trusted: `render` must begin with exactly
    this for every state built on the same catalog, on every step of every episode of the window.
    """
    return f"{SYSTEM_PROMPT}\n{_CATALOG_HEADER}{catalog.compact()}\n"


def render(state: ConductorState) -> str:
    """The pinned state -> prompt template. Identical state, identical bytes, in any process."""
    return cacheable_prefix(state.catalog) + _task(state.task) + _history(state.history)


def token_estimate(text: str) -> int:
    """Conservative token count. The seam for the reference tokeniser once it is pinned."""
    return ceil(len(text) / CHARS_PER_TOKEN)


def prefix_tokens(catalog: Catalog) -> int:
    """What the cacheable prefix costs, so M0 exit 6 is a measurement rather than a belief."""
    return token_estimate(cacheable_prefix(catalog))


def _task(task: TaskSpec) -> str:
    tools = ", ".join(task.tools) if task.tools else "none"
    return (f"\n# TASK\n"
            f"benchmark: {task.benchmark}\n"
            f"tools available to the worker: {tools}\n"
            f"---\n{task.prompt}\n---\n")


def _history(history: tuple[StepRecord, ...]) -> str:
    lines = [_step(i, step) for i, step in enumerate(history, 1)] or ["(no steps yet)"]
    spend = sum(step.cost_usd for step in history)
    return ("\n# HISTORY\n" + "\n".join(lines) + "\n"
            f"\nspent so far: ${spend:{_USD}}\n"
            f"this is step {len(history) + 1} of at most {MAX_STEPS}. "
            f"Reply with exactly one action.\n")


def _step(number: int, step: StepRecord) -> str:
    """One history row: what was asked for, what came back, what it cost (§3).

    A refused step is shown rather than hidden. It is the one piece of history the Conductor can
    act on immediately — a model that cannot see it fumbled has no reason to stop fumbling, and
    three consecutive fumbles end the episode (§2).
    """
    if step.parse_failed:
        return f"step {number}: refused, unparseable — no model was called"
    verb = step.action.kind.upper()
    outcome = "succeeded" if step.success else "failed"
    return f"step {number}: {verb} {step.model_id} -> {outcome} (${step.cost_usd:{_USD}})"
