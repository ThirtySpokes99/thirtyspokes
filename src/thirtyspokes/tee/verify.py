"""Validator-side attestation verification.

Cheap and re-execution-free: verify the hardware quote, that it attests an
*approved* runtime measurement, that the payload is bound to the quote, and that
the run answered the *task+nonce the validator issued* (anti-replay / anti-best-
of-N). Then grade the answer and read the runtime-metered cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attestation import AttestationReport, verify_quote


@dataclass
class Verdict:
    valid: bool
    reason: str
    quality: float
    cost_usd: float
    score: float


def verify_attestation(
    report: AttestationReport,
    *,
    approved_measurements: set[str],
    platform_public_hex: str,
    expect_task_id: str,
    expect_epoch: int,
    expect_nonce: str,
    gold,
    grade_fn,
    lam: float,
    cost_ref: float,
) -> Verdict:
    def bad(reason: str) -> Verdict:
        return Verdict(False, reason, 0.0, 0.0, 0.0)

    q = report.quote
    if q is None or not verify_quote(q, platform_public_hex):
        return bad("bad_platform_quote")                 # not hardware-rooted → forged
    if q.measurement != report.measurement:
        return bad("measurement_mismatch")               # quote must attest this runtime
    if q.report_data != report.report_data():
        return bad("report_data_mismatch")               # payload tampered post-attestation
    if report.measurement not in approved_measurements:
        return bad("unapproved_runtime")                 # a lying/modified runtime
    if (report.task_id, report.epoch, report.nonce) != (expect_task_id, expect_epoch, expect_nonce):
        return bad("task_nonce_mismatch")                # replay / stale / best-of-N
    c = report.total_cost_usd
    if not (c == c) or c < 0:                            # NaN / negative
        return bad("bad_cost")

    quality = float(grade_fn(report.answer, gold))
    score = quality - lam * (c / cost_ref)
    return Verdict(True, "ok", quality, c, score)
