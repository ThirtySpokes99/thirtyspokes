"""Eval-harness tests: offline wiring (mock backend) + network-guarded real loaders."""

from __future__ import annotations

import pytest

from thirtyspokes.eval import math_tasks, swe_tasks
from thirtyspokes.eval.agent import ReferenceAgent
from thirtyspokes.gateway.gateway import MeteringGateway, MockBackend
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.gateway.verify import verify_submission

_NC = [0]


def _nonce():
    _NC[0] += 1
    return f"e{_NC[0]}"


# --- offline wiring (no network, no API) -------------------------------------
def test_math_agent_mock_end_to_end():
    """Agent -> gateway -> proof -> verify, on the mock backend. Flow + cost +
    binding must hold even though the mock LLM can't actually solve."""
    task = math_tasks.procedural(1, seed=0)[0]
    agent = ReferenceAgent(Signer(), "small")
    gw = MeteringGateway(MockBackend(), {agent.hotkey})
    sub, receipts = agent.solve(gw, task.task_id, task.prompt, _nonce)
    v = verify_submission(sub, receipts, gw.public_hex, task.gold, math_tasks.grade,
                          lam=0.15, cost_ref=0.05)
    assert v.valid                          # receipts verify + answer↔trace binding holds
    assert v.cost_usd > 0                    # real metered cost accrued
    assert len(sub.trace) >= 1


def test_math_grader():
    assert math_tasks.grade("the final answer is 18", 18.0) == 1.0
    assert math_tasks.grade("42", 18.0) == 0.0
    assert math_tasks.grade("no number here", 18.0) == 0.0


def test_mock_swe_grader():
    class T:
        instance_id = "x"
    diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@\n+fix = True\n"
    g = swe_tasks.MockSWEGrader()
    assert g.grade_batch([T], {"x": diff})["x"] == 1.0
    assert g.grade_batch([T], {"x": "not a patch"})["x"] == 0.0


# --- real datasets (skipped if HF is unreachable) ----------------------------
def _load_or_skip(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # network / dataset unavailable
        pytest.skip(f"dataset unavailable: {type(e).__name__}")


def test_gsm8k_real_load_and_gold():
    tasks = _load_or_skip(math_tasks.load_gsm8k, 3, seed=1)
    assert len(tasks) == 3
    for t in tasks:
        assert isinstance(t.gold, float) and t.source == "gsm8k"
        assert "final answer" in t.prompt.lower()


def test_swebench_pro_real_load():
    tasks = _load_or_skip(swe_tasks.load_swebench_pro, 2, seed=3)
    assert len(tasks) == 2
    for t in tasks:
        assert t.instance_id and t.repo and t.dockerhub_tag
        assert isinstance(t.fail_to_pass, list)
        assert t.problem_statement in swe_tasks.format_prompt(t)


# --- the MATH grader must not silently score every answer 0.0 -----------------------------------
# Two bugs, both of which zeroed the grader while looking like "the models all failed":
#   1. math_verify parses a BARE gold with plain-text rules: `1\frac{4}{5}` -> 4/5 (leading 1 dropped),
#      so even an exactly-correct answer scored 0.0. Golds must be wrapped in \boxed{}.
#   2. math_verify's default timeout uses signal.alarm(), which RAISES in any worker thread — and
#      every parallel eval grades inside a ThreadPoolExecutor. grade_hard's `except` swallowed it.

def test_grade_hard_accepts_correct_answers_in_any_form():
    from thirtyspokes.eval.hard_tasks import grade_hard
    gold = r"1\frac{4}{5}"
    assert grade_hard(r"The answer is \boxed{1\frac{4}{5}}", gold) == 1.0
    assert grade_hard(r"\boxed{1\tfrac{4}{5}}", gold) == 1.0
    assert grade_hard(r"\boxed{\frac{9}{5}}", gold) == 1.0      # improper form is equivalent
    assert grade_hard(r"\boxed{1.8}", gold) == 1.0              # decimal form is equivalent
    assert grade_hard(r"\boxed{2\tfrac{3}{11}}", gold) == 0.0   # genuinely wrong stays wrong
    assert grade_hard(r"so \boxed{42}", "42") == 1.0            # bare integer gold


def test_grade_hard_works_off_the_main_thread():
    """Graded in a worker (as every parallel eval does), math_verify's signal-based timeout raises."""
    from concurrent.futures import ThreadPoolExecutor
    from thirtyspokes.eval.hard_tasks import grade_hard
    with ThreadPoolExecutor(2) as ex:
        got = list(ex.map(lambda a: grade_hard(a, r"1\frac{4}{5}"),
                          [r"\boxed{1\frac{4}{5}}", r"\boxed{9/5}", r"\boxed{7}"]))
    assert got == [1.0, 1.0, 0.0], f"grader zeroed in a worker thread: {got}"
