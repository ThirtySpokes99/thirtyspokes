"""Option-A (TEE) trust-flow tests: attestation verifies, cost is metered by the
runtime (not the agent), and every adversary that beat the gateway now fails."""

from __future__ import annotations

from dataclasses import replace

import pytest

from thirtyspokes.gateway.gateway import MockBackend
from thirtyspokes.gateway.grader import grade_math
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.tee.attestation import AttestationReport, Platform
from thirtyspokes.tee.runtime import TEERuntime, runtime_measurement
from thirtyspokes.tee.subnet import calling_agent, run_simulation
from thirtyspokes.tee.verify import verify_attestation

LAM, CR = 0.5, 0.02


@pytest.fixture
def env():
    platform = Platform()
    rt = TEERuntime(MockBackend(), platform)
    approved = {runtime_measurement()}
    return platform, rt, approved


def _verify(rep, platform, approved, nonce="n1", tid="t", epoch=1, gold=42.0):
    return verify_attestation(rep, approved_measurements=approved,
                              platform_public_hex=platform.public_hex, expect_task_id=tid,
                              expect_epoch=epoch, expect_nonce=nonce, gold=gold,
                              grade_fn=grade_math, lam=LAM, cost_ref=CR)


def test_cost_metered_by_runtime_not_agent(env):
    """The agent's only channel is call_model; the runtime meters real cost. An
    agent cannot under-report — this is the property the gateway lacked."""
    _, rt, _ = env
    rep = rt.run(calling_agent("frontier", 5, "42"), task_id="t", epoch=1, nonce="n1", prompt="q")
    assert rep.n_calls == 5 and rep.total_cost_usd > 0     # metered, not agent-asserted
    # a cheaper run of the same agent shape costs strictly less (real metering)
    rep2 = rt.run(calling_agent("tiny", 1, "42"), task_id="t", epoch=1, nonce="n2", prompt="q")
    assert rep2.total_cost_usd < rep.total_cost_usd


def test_valid_attestation_verifies_and_scores(env):
    platform, rt, approved = env
    rep = rt.run(calling_agent("tiny", 1, "42"), task_id="t", epoch=1, nonce="n1", prompt="q")
    v = _verify(rep, platform, approved)
    assert v.valid and v.quality == 1.0 and v.cost_usd > 0


def test_fake_quote_rejected(env):
    """Self-signed quote (no real TEE) fails the hardware-root check."""
    platform, rt, approved = env
    attacker = Signer()
    rep = AttestationReport("t", 1, "n1", "42", 0.0, 0, "x", runtime_measurement())
    rep = rep.attested_by(Platform(attacker))              # miner's key, not the vendor root
    v = _verify(rep, platform, approved)
    assert not v.valid and v.reason == "bad_platform_quote"


def test_unapproved_runtime_rejected(env):
    """A modified runtime reporting zero cost has a non-approved measurement."""
    platform, rt, approved = env
    rep = AttestationReport("t", 1, "n1", "42", 0.0, 0, "x", "MODIFIED-runtime")
    rep = rep.attested_by(platform)                        # genuine hardware, unapproved image
    v = _verify(rep, platform, approved)
    assert not v.valid and v.reason == "unapproved_runtime"


def test_replay_rejected(env):
    """An attestation sealed with a different nonce than the validator issued."""
    platform, rt, approved = env
    rep = rt.run(calling_agent("tiny", 1, "42"), task_id="t", epoch=1, nonce="STALE", prompt="q")
    v = _verify(rep, platform, approved, nonce="n1")       # validator issued n1, not STALE
    assert not v.valid and v.reason == "task_nonce_mismatch"


def test_tampered_report_rejected(env):
    """Mutating any field after attestation breaks the quote's report_data binding."""
    platform, rt, approved = env
    rep = rt.run(calling_agent("frontier", 4, "42"), task_id="t", epoch=1, nonce="n1", prompt="q")
    tampered = replace(rep, total_cost_usd=0.0)            # keep old quote, cheat cost
    v = _verify(tampered, platform, approved)
    assert not v.valid and v.reason == "report_data_mismatch"


def test_simulation_adversaries_all_fail():
    out = run_simulation(epochs=4, verbose=False)
    dq = out["disqualifications"]
    assert dq.get("lying-rt") == "unapproved_runtime"
    assert dq.get("fake-quote") == "bad_platform_quote"
    assert dq.get("replayer") == "task_nonce_mismatch"
    em = out["emissions"]
    assert em.get("honest-cheap", 0) > em.get("honest-exp", 0) > 0   # cost-efficiency wins
