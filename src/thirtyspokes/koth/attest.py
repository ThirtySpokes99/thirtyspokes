"""The attestation gate, standalone — the part of this project that outlives the subnet thesis.

    "This exact artifact produced this exact result, on real hardware — and you can verify that
     without re-running it or trusting whoever ran it."

`verify_proof` in `verify.py` used to do two unrelated jobs in one function: check that claim, and
then grade benchmark answers to produce an economic score. The first depends on nothing but the
hardware quote and the payload; the second is the routing economics, which `docs/ROUTING_MEASUREMENTS.md`
closed. Keeping them fused meant the surviving asset could not be imported without dragging in
frontiers, regret, Wilson bounds and a benchmark suite. `docs/ASSET.md` flagged that as the
outstanding extraction cost; this is it.

WHAT THE GATE ESTABLISHES, in order, each step meaningless without the ones before it:
  1. a quote exists and is real — a genuine Intel TDX DCAP quote chaining to the Intel root, or (dev
     only) a mock vendor signature;
  2. it came from an image the owner approved — MRTD + boot RTMR1/2 + runtime RTMR3, under a TCB
     policy, so "which code ran" is a hardware fact rather than a claim;
  3. the payload is the one that was attested — `report_data` is a hash over every proof field, so
     any post-hoc edit breaks the quote;
  4. it answers the challenge that was issued — the (epoch, nonce) the verifier chose, which is what
     makes replay and best-of-N grinding fail;
  5. it is bound to a specific artifact — the source and weights hashes must equal what the verifier
     independently recomputed from the published bundle, so the thing measured is the thing shipped.

`enforce=True` is the production posture and it is fail-CLOSED: a missing gate is a configuration
error that would silently admit a forged runtime, so it is rejected rather than skipped. That
asymmetry is deliberate — every other check here can only reject a bad proof, but an unset gate would
accept one.

No benchmarks, no scoring, no chain, no numpy. Returns a failure reason or None.
"""

from __future__ import annotations

from ..tee.attestation import verify_quote
from .proof import Proof


def verify_attestation(
    proof: Proof,
    *,
    approved_measurements: set[str],
    platform_public_hex: str,
    expect_epoch: int,
    expect_nonce: str,
    expect_hotkey: str,
    expect_source_hash: str,
    expect_weights_hash: str,
    approved_mrtd: set[str] | None = None,
    approved_rtmr: dict[int, str] | None = None,
    tcb_accept: frozenset[str] | None = None,
    collateral: str | None = None,
    pccs_url: str | None = None,
    now: int | None = None,
    enforce: bool = False,
) -> str | None:
    """None if the proof is authentic and bound as expected, else the failure reason.

    Reasons are returned rather than raised because every one of them is a normal outcome for an
    untrusted submitter — a forged proof is data about that submitter, not an exceptional condition.
    """
    q = proof.quote
    if q is None:
        return "bad_platform_quote"
    is_tdx = q.platform_sig.startswith("tdx:")
    if enforce:
        # PRODUCTION FAIL-CLOSED (docs/DESIGN.md §8). The miner's own CVM samples AND runs the
        # benchmark, so the score is trustworthy ONLY if the audited, measured runtime
        # provably ran — the one thing a trustless miner cannot forge. A tampered runtime
        # can forge every payload field (source/weights hash, cost, answers, and the
        # self-reported `measurement` string), so the sole real anchor is the hardware
        # quote gated on an owner-pinned measured image (MRTD + boot RTMR1/2 + runtime
        # RTMR3) under a TCB policy. A missing gate is a config error that would silently
        # admit a forged runtime — so it DQs here rather than skipping the check.
        if not is_tdx:
            return "mock_quote_rejected"                  # mock vendor key = zero security
        if not approved_mrtd:
            return "mrtd_gate_unset"
        if not approved_rtmr or not {1, 2, 3} <= set(approved_rtmr):
            return "rtmr_gate_unset"                      # need boot RTMR1/2 + runtime RTMR3
        if tcb_accept is None:
            return "tcb_policy_unset"                     # enforce ⇒ full DCAP (TCB/CRL/QE)
        if not proof.confined:
            # The agent ran with network egress. The measured-image gates above prove WHICH image
            # booted, not what it did inside — and an unconfined agent can reach an off-allow-list
            # model using a key embedded in its own weights, voiding the pool pinning, the metered
            # cost and the budget ceiling while still satisfying `no_pool_call` with one token call.
            # `run_agent_confined` degrades silently when the namespace probe fails, so this is
            # gated on the ATTESTED fact rather than on the miner's configuration.
            return "unconfined_agent"
    if is_tdx:                                            # WS7: real Intel-TDX hardware quote
        # `tcb_accept` set -> full DCAP (TCB status/CRL/QE-identity via dcap-qvl, H1);
        # else the offline crypto-chain-only path (cert chain to Intel root + binding).
        if tcb_accept is not None:
            from .tdx import verify_tdx_quote_field_full
            vq = verify_tdx_quote_field_full(
                q, approved_mrtd=approved_mrtd, approved_rtmr=approved_rtmr,
                tcb_accept=tcb_accept, collateral=collateral, pccs_url=pccs_url, now=now)
        else:
            from .tdx import verify_tdx_quote_field     # lazy: needs `cryptography` on real HW
            vq = verify_tdx_quote_field(q, approved_mrtd=approved_mrtd)
        if not vq.ok:
            if vq.reason == "mrtd_not_approved" or vq.reason.startswith("rtmr"):
                return "unapproved_runtime"            # hardware image not owner-approved
            if vq.reason.startswith("tcb_"):
                return vq.reason                       # surface the TCB status distinctly
            return "bad_platform_quote"
    elif not verify_quote(q, platform_public_hex):     # offline/dev mock vendor key
        return "bad_platform_quote"
    if q.measurement != proof.measurement:
        return "measurement_mismatch"
    if q.report_data != proof.report_data():
        return "report_data_mismatch"                  # payload tampered post-attestation
    if proof.measurement not in approved_measurements:
        return "unapproved_runtime"
    if (proof.epoch, proof.nonce) != (expect_epoch, expect_nonce):
        return "epoch_nonce_mismatch"                  # replay / stale / best-of-N
    if proof.hotkey != expect_hotkey:
        return "hotkey_mismatch"                       # a copier resubmitting someone's proof
    if proof.source_hash != expect_source_hash or proof.weights_hash != expect_weights_hash:
        return "artifact_binding_mismatch"             # not the downloaded/committed artifact
    c = proof.total_cost_usd
    if not (c == c) or c < 0:                          # NaN / negative
        return "bad_cost"
    return None
