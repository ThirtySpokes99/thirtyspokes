"""The Receipt — the un-forgeable unit of attested work.

A gateway-signed record binding a model call to a hotkey with its REAL cost.
Miners collect receipts and cite them (by call_id) in task submissions; validators
verify the gateway signature and sum cost — without re-running anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from . import signing
from .signing import Signer


@dataclass(frozen=True)
class Receipt:
    call_id: str
    hotkey: str            # the caller (bound — copies fail for other hotkeys)
    model: str
    prompt_hash: str
    response_hash: str     # pins the output; can't be swapped after the fact
    tokens_in: int
    tokens_out: int
    cost_usd: float        # REAL cost from the provider, not miner-supplied
    ts: int
    gw_sig: str = ""       # gateway ed25519 signature over the payload

    def _payload(self) -> bytes:
        d = asdict(self)
        d.pop("gw_sig")
        return signing.canonical(d)

    def signed_by(self, gateway: Signer) -> "Receipt":
        return replace(self, gw_sig=gateway.sign(self._payload()))

    def verify(self, gateway_public_hex: str) -> bool:
        return signing.verify(gateway_public_hex, self._payload(), self.gw_sig)


@dataclass(frozen=True)
class TaskSubmission:
    hotkey: str
    task_id: str
    final_answer: str
    trace: tuple[str, ...]   # call_ids backing this task
