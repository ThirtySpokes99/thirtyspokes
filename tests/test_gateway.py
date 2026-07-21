"""End-to-end linchpin test: metered gateway + verifiable grading.

Proves the trust flow catches every threat in the gateway threat model while rewarding
honest cost-efficient work — all offline, no provider, no chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from thirtyspokes.gateway.gateway import MeteringGateway, MockBackend, gateway_call
from thirtyspokes.gateway.grader import generate_math_tasks, grade_math
from thirtyspokes.gateway.receipts import Receipt, TaskSubmission
from thirtyspokes.gateway.signing import Signer, sha256_hex
from thirtyspokes.gateway.verify import verify_submission

LAM, COST_REF = 0.5, 0.02
_NONCE = [0]


def _nonce() -> str:
    _NONCE[0] += 1
    return f"nonce-{_NONCE[0]}"      # globally unique (nonces are single-use)


def _agent(gw, miner, task, model, n_calls):
    """A toy agent: n_calls 'thinking' calls on `model`, then a finalize call that
    emits the (correct) answer through the gateway. Returns (submission, receipts)."""
    receipts = {}
    for _ in range(n_calls):
        _, r = gateway_call(gw, miner, model, [{"role": "user", "content": task.prompt}],
                            {"max_tokens": 64}, nonce=_nonce())
        receipts[r.call_id] = r
    answer = str(int(task.gold))
    _, rf = gateway_call(gw, miner, model, [{"role": "user", "content": answer}],
                         {"finalize": True, "max_tokens": 8}, nonce=_nonce())
    receipts[rf.call_id] = rf
    sub = TaskSubmission(miner.public_hex, task.task_id, answer, tuple(receipts))
    return sub, receipts


@pytest.fixture(scope="module")
def setup():
    task = generate_math_tasks(1, seed=42)[0]
    gw = MeteringGateway(MockBackend(), registered=set())
    miners = {name: Signer() for name in ["cheap", "expensive", "bypass", "fab", "thief"]}
    for m in miners.values():
        gw.register(m.public_hex)
    return task, gw, miners


def _verify(sub, receipts, gw, task):
    return verify_submission(sub, receipts, gw.public_hex, task.gold, grade_math,
                             lam=LAM, cost_ref=COST_REF)


def test_honest_cheap_beats_honest_expensive(setup):
    """Both correct; the cheap agent scores higher — cost is real and priced in."""
    task, gw, miners = setup
    sub_c, rc = _agent(gw, miners["cheap"], task, "small", n_calls=1)
    sub_e, re = _agent(gw, miners["expensive"], task, "frontier", n_calls=4)
    vc, ve = _verify(sub_c, rc, gw, task), _verify(sub_e, re, gw, task)
    assert vc.valid and ve.valid
    assert vc.quality == 1.0 and ve.quality == 1.0     # both correct
    assert ve.cost_usd > vc.cost_usd                   # expensive really cost more
    assert vc.score > ve.score                         # cheaper wins on quality/cost


def test_bypasser_gets_no_credit(setup):
    """Solve off-gateway, submit answer with no backing gateway work -> invalid."""
    task, gw, miners = setup
    m = miners["bypass"]
    sub = TaskSubmission(m.public_hex, task.task_id, str(int(task.gold)), trace=())
    v = verify_submission(sub, {}, gw.public_hex, task.gold, grade_math, LAM, COST_REF)
    assert not v.valid and v.reason == "answer_not_backed_by_gateway"


def test_fabricated_receipt_rejected(setup):
    """A self-signed fake receipt claiming near-zero cost fails signature check."""
    task, gw, miners = setup
    m = miners["fab"]
    forged = Receipt("call-x", m.public_hex, "frontier", "ph", sha256_hex(str(int(task.gold))),
                     1, 1, 0.00001, 1).signed_by(m)            # signed by MINER, not gateway
    sub = TaskSubmission(m.public_hex, task.task_id, str(int(task.gold)), ("call-x",))
    v = verify_submission(sub, {"call-x": forged}, gw.public_hex, task.gold,
                          grade_math, LAM, COST_REF)
    assert not v.valid and v.reason == "bad_gateway_signature"


def test_stolen_receipt_rejected(setup):
    """Reusing another miner's real (gateway-signed) receipts -> hotkey mismatch."""
    task, gw, miners = setup
    victim_sub, victim_receipts = _agent(gw, miners["cheap"], task, "small", n_calls=1)
    thief = miners["thief"]
    sub = TaskSubmission(thief.public_hex, task.task_id, str(int(task.gold)),
                         tuple(victim_receipts))
    v = verify_submission(sub, victim_receipts, gw.public_hex, task.gold,
                          grade_math, LAM, COST_REF)
    assert not v.valid and v.reason == "receipt_hotkey_mismatch"


def test_unregistered_and_bad_sig_blocked_at_gateway(setup):
    task, gw, _ = setup
    outsider = Signer()   # never registered
    with pytest.raises(PermissionError):
        gateway_call(gw, outsider, "small", [{"role": "user", "content": "x"}], {}, "n1")


def test_option2_subnet_end_to_end():
    """Full Option-2 subnet loop: arbitrary agents -> gateway -> verify -> reign.
    Honest cost-efficient agent wins; forger + bypasser are disqualified."""
    from thirtyspokes.gateway.subnet import run_simulation
    out = run_simulation(epochs=3, verbose=False)
    em, dq = out["emissions"], out["disqualifications"]
    assert dq.get("forger") == "bad_gateway_signature"
    assert dq.get("bypasser") == "answer_not_backed_by_gateway"
    assert em.get("cheap", 0) > em.get("expensive", 0) > 0     # cheaper-honest wins
    assert em.get("cheap", 0) > em.get("wrong", 0)             # quality matters too


def test_wrong_answer_scores_zero_quality(setup):
    task, gw, miners = setup
    m = miners["cheap"]
    _, r = gateway_call(gw, m, "small", [{"role": "user", "content": "999999"}],
                        {"finalize": True}, nonce="wrong-final")
    sub = TaskSubmission(m.public_hex, task.task_id, "999999", (r.call_id,))
    v = verify_submission(sub, {r.call_id: r}, gw.public_hex, task.gold,
                          grade_math, LAM, COST_REF)
    assert v.valid and v.quality == 0.0                # graded honestly wrong
