"""Validator-side proof verification — cheap, no agent re-execution.

Given a miner's TaskSubmission and the receipts backing it: grade the answer,
verify every receipt (gateway signature + hotkey binding), require the answer to
be produced through the gateway (answer↔trace binding — closes the "work
off-gateway and claim zero cost" hole), sum the real cost, and score.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import signing
from .receipts import Receipt, TaskSubmission


@dataclass
class Verdict:
    task_id: str
    valid: bool
    reason: str
    quality: float
    cost_usd: float
    score: float


def verify_submission(
    sub: TaskSubmission,
    receipts: dict[str, Receipt],   # call_id -> receipt (miner-supplied or gateway-fetched)
    gateway_public_hex: str,
    gold,
    grade_fn,
    lam: float,
    cost_ref: float,
    require_binding: bool = True,
) -> Verdict:
    def bad(reason: str) -> Verdict:
        return Verdict(sub.task_id, False, reason, 0.0, 0.0, 0.0)

    total_cost = 0.0
    response_hashes = set()
    for cid in sub.trace:
        r = receipts.get(cid)
        if r is None:
            return bad("missing_receipt")
        if not r.verify(gateway_public_hex):     # forged / self-signed receipt
            return bad("bad_gateway_signature")
        if r.hotkey != sub.hotkey:               # stolen / another miner's receipt
            return bad("receipt_hotkey_mismatch")
        total_cost += r.cost_usd
        response_hashes.add(r.response_hash)

    # answer must have been produced through the gateway (else off-gateway free work)
    if require_binding and signing.sha256_hex(sub.final_answer) not in response_hashes:
        return bad("answer_not_backed_by_gateway")

    quality = float(grade_fn(sub.final_answer, gold))
    score = quality - lam * (total_cost / cost_ref)
    return Verdict(sub.task_id, True, "ok", quality, total_cost, score)
