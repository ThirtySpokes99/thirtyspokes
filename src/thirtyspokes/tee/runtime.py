"""The measured enclave runtime — the trusted metering + sandbox layer.

The runtime is what the hardware attests (its `MEASUREMENT` is on the subnet's
approved list). It runs the miner's arbitrary agent but hands it *only* a
`call_model` function — no backend, no network. All calls flow through the
MeteringProxy, which records the REAL provider cost, so the agent can neither make
an unmetered call nor lie about cost. In production the enclave's no-egress
sandbox enforces the "only call_model" property; here the interface models it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..gateway import signing
from ..gateway.gateway import ModelBackend
from .attestation import AttestationReport, Platform

# Bump on any runtime change → new measurement → must be re-approved on-chain.
RUNTIME_VERSION = "orchestra-tee-runtime-1"
# Whitelist of params the runtime forwards to the provider (rejects stream/n/etc.)
ALLOWED_PARAMS = {"max_tokens", "temperature", "top_p", "stop"}


def runtime_measurement() -> str:
    """Hash standing in for the enclave image measurement (MRENCLAVE)."""
    return signing.sha256_hex({"runtime": RUNTIME_VERSION})


@dataclass
class MeteringProxy:
    """The agent's ONLY channel to models. Meters real cost; the agent cannot bypass it."""

    backend: ModelBackend
    total_cost_usd: float = 0.0
    calls: list = field(default_factory=list)

    def call_model(self, model: str, messages: list, params: dict | None = None) -> str:
        p = {k: v for k, v in (params or {}).items() if k in ALLOWED_PARAMS}
        p.setdefault("max_tokens", 1024)
        p.setdefault("temperature", 0.0)
        text, tin, tout, cost = self.backend.complete(model, messages, p)
        self.total_cost_usd += float(cost)
        self.calls.append({"model": model, "tin": tin, "tout": tout, "cost": cost})
        return text

    def call_log_hash(self) -> str:
        return signing.sha256_hex(self.calls)


class TEERuntime:
    """Runs an agent under attestation. `agent(prompt, call_model) -> answer`."""

    def __init__(self, backend: ModelBackend, platform: Platform):
        self.backend = backend
        self.platform = platform

    def run(self, agent, *, task_id: str, epoch: int, nonce: str, prompt: str) -> AttestationReport:
        proxy = MeteringProxy(self.backend)
        answer = agent(prompt, proxy.call_model)     # agent gets ONLY prompt + call_model
        report = AttestationReport(
            task_id=task_id, epoch=epoch, nonce=nonce, answer=str(answer),
            total_cost_usd=proxy.total_cost_usd, n_calls=len(proxy.calls),
            call_log_hash=proxy.call_log_hash(), measurement=runtime_measurement(),
        )
        return report.attested_by(self.platform)
