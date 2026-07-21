"""Attestation core — the hardware-rooted proof of what ran.

`Platform` mocks the hardware attestation root (Intel DCAP / AMD SEV / NVIDIA CC):
it signs a `Quote` binding a runtime `measurement` to the hash of an execution's
outputs. In production this is replaced by a real TEE quote + the vendor's
verification service; the report/verify shape is identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from ..gateway import signing
from ..gateway.signing import Signer


@dataclass(frozen=True)
class Quote:
    measurement: str      # hash of the runtime image that ran (MRENCLAVE analog)
    report_data: str      # sha256 of the attested payload (binds all report fields)
    platform_sig: str     # signature by the hardware attestation root

    def _payload(self) -> bytes:
        return signing.canonical({"measurement": self.measurement,
                                  "report_data": self.report_data})


class Platform:
    """Mock hardware attestation root. Real: Intel/AMD/NVIDIA attestation service."""

    def __init__(self, signer: Signer | None = None):
        self._signer = signer or Signer()

    @property
    def public_hex(self) -> str:
        return self._signer.public_hex

    def quote(self, measurement: str, report_data: str) -> Quote:
        q = Quote(measurement, report_data, "")
        return replace(q, platform_sig=self._signer.sign(q._payload()))


def verify_quote(quote: Quote, platform_public_hex: str) -> bool:
    return signing.verify(platform_public_hex, quote._payload(), quote.platform_sig)


@dataclass(frozen=True)
class AttestationReport:
    task_id: str
    epoch: int
    nonce: str               # validator-issued; sealed here (anti-replay / anti-best-of-N)
    answer: str              # graded by the domain checker
    total_cost_usd: float    # metered by the trusted runtime — un-forgeable
    n_calls: int
    call_log_hash: str
    measurement: str         # runtime image hash; must be on the approved list
    quote: Quote | None = None

    def _payload(self) -> dict:
        d = asdict(self)
        d.pop("quote")
        return d

    def report_data(self) -> str:
        return signing.sha256_hex(self._payload())

    def attested_by(self, platform: Platform) -> "AttestationReport":
        return replace(self, quote=platform.quote(self.measurement, self.report_data()))
