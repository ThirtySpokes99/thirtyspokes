"""The prompt template's two properties (docs/WHITEPAPER.md §3, the build plan M0 exits 5-7).

Both are properties nothing else would notice breaking, which is why they are tested rather than
left to convention:

  * **Byte-identical renders across processes.** A prompt built from a set differs per process under
    string-hash randomisation, and the two validators would disagree while every seed matched — the
    `koth/matrix.py::scoreable_ids` bug, in the one place where it would silently re-score every
    miner. So the determinism test actually spawns subprocesses under different `PYTHONHASHSEED`s.
  * **A stable catalog prefix.** Nothing fails if the order goes wrong; the serving bill just grows
    by two orders of magnitude, because the ~15-20k-token catalog stops being prefill-cacheable and
    is paid on each of ~5,000-15,000 Conductor calls per duel.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

from thirtyspokes.v3.config import MAX_STEPS
from thirtyspokes.v3.render import (
    CATALOG_TOKEN_BUDGET,
    SYSTEM_PROMPT,
    ConductorState,
    cacheable_prefix,
    prefix_tokens,
    render,
)
from thirtyspokes.v3.types import Action, Catalog, CatalogEntry, StepRecord, TaskSpec

CATALOG = Catalog(entries=(
    CatalogEntry("openai/gpt-5.4", 1.25, 10.0, 400_000),
    CatalogEntry("openai/gpt-5.4-mini", 0.25, 2.0, 400_000),
    CatalogEntry("google/gemini-3.6-flash", 0.075, 0.3, 1_000_000),
))

TASK = TaskSpec(task_id="tb-0031", benchmark="terminal-bench",
                prompt="The test suite fails on a fresh checkout. Make it pass.",
                tools=("bash", "editor"))


def _step(number: int, model_id: str, success: bool, cost: float) -> StepRecord:
    return StepRecord(step_index=number, rendered_state_digest="deadbeef", raw_output="REASON ...",
                      action=Action("delegate" if number == 0 else "retry", model_id),
                      model_id=model_id, success=success, cost_usd=cost, parse_failed=False)


def _failed_step(number: int) -> StepRecord:
    return StepRecord(step_index=number, rendered_state_digest="deadbeef", raw_output="thinking...",
                      action=None, model_id=None, success=False, cost_usd=0.0, parse_failed=True)


def test_the_prompt_order_is_catalog_then_task_then_history():
    """M0 exit 7. The order is what makes the catalog a cacheable prefix; varying state first would
    forfeit the prefill saving silently, since nothing else would break."""
    prompt = render(ConductorState(CATALOG, TASK, (_step(0, "openai/gpt-5.4", False, 0.012),)))
    assert prompt.index("# CATALOG") < prompt.index("# TASK") < prompt.index("# HISTORY")


def test_the_catalog_prefix_is_byte_identical_on_every_step_of_a_window():
    """M0 exit 7, the property itself: one window, many tasks, growing histories, one prefix."""
    other_task = TaskSpec("swe-9", "frontier-swe", "Fix the regression in `parse_url`.", ())
    history: tuple[StepRecord, ...] = ()
    prefix = cacheable_prefix(CATALOG)
    for task in (TASK, other_task, TASK):
        for _ in range(3):
            assert render(ConductorState(CATALOG, task, history)).startswith(prefix)
            history = (*history, _step(len(history), "openai/gpt-5.4-mini", False, 0.0004))


def test_a_different_catalog_gives_a_different_prefix():
    """The other half: the prefix must actually depend on the snapshot it claims to cache."""
    smaller = Catalog(entries=CATALOG.entries[:2])
    assert cacheable_prefix(smaller) != cacheable_prefix(CATALOG)


def test_identical_state_renders_identical_bytes():
    state = ConductorState(CATALOG, TASK, (_step(0, "openai/gpt-5.4", True, 0.012),))
    assert render(state) == render(ConductorState(CATALOG, TASK, state.history))


def test_render_is_byte_identical_across_processes_under_different_hash_seeds():
    """M0 exit 5, and the reason it is spawned rather than asserted in-process: string hashing is
    randomised per process, so a set anywhere on this path renders a prompt that differs between
    two validators while every seed and every input matches. That is `scoreable_ids`, in the one
    place where the consequence is silently re-scoring every miner."""
    digests = {_render_digest_in_subprocess(seed) for seed in ("0", "1", "12345")}
    state = ConductorState(CATALOG, TASK, (_step(0, "openai/gpt-5.4", False, 0.012),))
    assert digests == {hashlib.sha256(render(state).encode()).hexdigest()}


def test_costs_render_at_a_pinned_precision_so_float_noise_cannot_change_the_prompt():
    """`repr` of a float carries the low digits of accumulated arithmetic, so two runs that spent
    the same money would otherwise render — and digest — different prompts."""
    noisy = _step(0, "openai/gpt-5.4", True, 0.1 + 0.2)
    exact = _step(0, "openai/gpt-5.4", True, 0.3)
    assert 0.1 + 0.2 != 0.3
    assert render(ConductorState(CATALOG, TASK, (noisy,))) == render(
        ConductorState(CATALOG, TASK, (exact,)))


def test_a_sub_cent_call_never_renders_as_free():
    """The cheap tier is the floor of every routing decision; rendering its calls as $0.00 would
    tell the Conductor that the pool's cheapest models cost nothing at all."""
    prompt = render(ConductorState(CATALOG, TASK, (_step(0, "google/gemini-3.6-flash",
                                                         True, 0.000021),)))
    assert "$0.000021" in prompt
    assert "$0.000000" not in prompt


def test_a_three_hundred_model_catalog_fits_the_pinned_context_budget():
    """M0 exit 6 — measured, not assumed. ~300 models is the whole of OpenRouter (D4), which is the
    action space the design commits to putting in context."""
    catalog = Catalog(entries=tuple(
        CatalogEntry(f"some-vendor-{i}/a-fairly-long-model-name-{i}-instruct",
                     0.15 + i / 100, 0.6 + i / 50, 128_000 + i)
        for i in range(300)))
    assert prefix_tokens(catalog) <= CATALOG_TOKEN_BUDGET
    # ...and the estimate is not passing by being trivially small.
    assert prefix_tokens(catalog) > 1_000


def test_every_catalog_model_and_its_prices_reach_the_prompt():
    """A budget-aware policy has to see prices to be budget-aware (§3), and it can only choose
    models it was shown."""
    prompt = render(ConductorState(CATALOG, TASK))
    for entry in CATALOG.entries:
        assert entry.model_id in prompt
    assert "0.075 / 0.3 / 1000000" in prompt


def test_the_task_statement_is_rendered_verbatim():
    """The Conductor decides about this exact text — the same bytes the scaffold forwards (§2.1)."""
    assert TASK.prompt in render(ConductorState(CATALOG, TASK))


def test_the_benchmark_tools_are_shown_and_an_empty_tool_list_is_not_a_blank():
    prompt = render(ConductorState(CATALOG, TASK))
    assert "bash, editor" in prompt
    toolless = render(ConductorState(CATALOG, TaskSpec("x", "hle", "What is a quine?", ())))
    assert "tools available to the worker: none" in toolless


def test_the_system_prompt_says_the_conductor_never_answers_only_delegates():
    """§2.1 as an instruction: the Conductor chooses where the work goes, it never does the work."""
    assert "never answer the task yourself" in SYSTEM_PROMPT
    assert "verbatim" in SYSTEM_PROMPT
    for verb in ("DELEGATE", "RETRY", "STOP", "REASON"):
        assert verb in SYSTEM_PROMPT
    assert SYSTEM_PROMPT in render(ConductorState(CATALOG, TASK))


def test_an_empty_history_renders_as_the_first_step_not_as_a_blank_section():
    """Step one is where every episode starts, so it is the most-rendered prompt of the duel."""
    prompt = render(ConductorState(CATALOG, TASK))
    assert "(no steps yet)" in prompt
    assert f"step 1 of at most {MAX_STEPS}" in prompt
    assert "spent so far: $0.000000" in prompt


def test_the_history_shows_what_was_asked_for_what_came_back_and_what_it_cost():
    """§3's history is (model, outcome, cost): a Conductor that cannot see an outcome cannot react
    to it, and reacting to an observed failure is the difference between RETRY and pick-one."""
    prompt = render(ConductorState(CATALOG, TASK, (
        _step(0, "google/gemini-3.6-flash", False, 0.0004),
        _step(1, "openai/gpt-5.4", True, 0.031))))
    assert "step 1: DELEGATE google/gemini-3.6-flash -> failed ($0.000400)" in prompt
    assert "step 2: RETRY openai/gpt-5.4 -> succeeded ($0.031000)" in prompt
    assert "spent so far: $0.031400" in prompt
    assert f"step 3 of at most {MAX_STEPS}" in prompt


def test_a_refused_step_is_shown_to_the_conductor_rather_than_hidden():
    """It is the one piece of history the Conductor can act on immediately, and three consecutive
    refusals end the episode (§2) — a model that cannot see it fumbled cannot stop fumbling."""
    prompt = render(ConductorState(CATALOG, TASK, (_failed_step(0),)))
    assert "step 1: refused, unparseable — no model was called" in prompt
    assert f"step 2 of at most {MAX_STEPS}" in prompt


_DIGEST_SCRIPT = """
import hashlib
from thirtyspokes.v3.render import ConductorState, render
from thirtyspokes.v3.types import Action, Catalog, CatalogEntry, StepRecord, TaskSpec

catalog = Catalog(entries=(
    CatalogEntry("openai/gpt-5.4", 1.25, 10.0, 400_000),
    CatalogEntry("openai/gpt-5.4-mini", 0.25, 2.0, 400_000),
    CatalogEntry("google/gemini-3.6-flash", 0.075, 0.3, 1_000_000),
))
task = TaskSpec(task_id="tb-0031", benchmark="terminal-bench",
                prompt="The test suite fails on a fresh checkout. Make it pass.",
                tools=("bash", "editor"))
step = StepRecord(step_index=0, rendered_state_digest="deadbeef", raw_output="REASON ...",
                  action=Action("delegate", "openai/gpt-5.4"), model_id="openai/gpt-5.4",
                  success=False, cost_usd=0.012, parse_failed=False)
print(hashlib.sha256(render(ConductorState(catalog, task, (step,))).encode()).hexdigest())
"""


def _render_digest_in_subprocess(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    done = subprocess.run([sys.executable, "-c", _DIGEST_SCRIPT], env=env, check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()
