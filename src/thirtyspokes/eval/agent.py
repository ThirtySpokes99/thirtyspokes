"""A reference agent — the *miner's* side, run to produce answer + proof.

Deliberately simple (reason once, then emit the answer through the gateway); it is
what miners replace with their own arbitrary orchestration. Every model call goes
through the metered gateway, so the run is fully attested. `final_answer` is the
raw response of the last gateway call, so the answer↔trace binding holds by
construction (no need to know the gold answer).
"""

from __future__ import annotations

from ..gateway.gateway import MeteringGateway, gateway_call
from ..gateway.receipts import TaskSubmission
from ..gateway.signing import Signer


class ReferenceAgent:
    def __init__(self, signer: Signer, model: str, reason: bool = True,
                 max_tokens: int = 1024):
        self.signer = signer
        self.hotkey = signer.public_hex
        self.model = model
        self.reason = reason
        self.max_tokens = max_tokens

    def solve(self, gw: MeteringGateway, task_id: str, prompt: str, nonce_fn):
        receipts = {}
        ctx = [{"role": "user", "content": prompt}]
        if self.reason:
            resp, r = gateway_call(gw, self.signer, self.model, ctx,
                                   {"max_tokens": self.max_tokens}, nonce_fn())
            receipts[r.call_id] = r
            ctx += [{"role": "assistant", "content": resp}]
        # final answer-only call: its raw response IS the submitted answer
        ctx += [{"role": "user", "content": "Output only the final answer, nothing else."}]
        final, rf = gateway_call(gw, self.signer, self.model, ctx,
                                 {"max_tokens": 256, "finalize": True}, nonce_fn())
        receipts[rf.call_id] = rf
        sub = TaskSubmission(self.hotkey, task_id, final, tuple(receipts))
        return sub, receipts
