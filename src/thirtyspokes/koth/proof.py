"""The multi-benchmark attested proof — the unit validators verify (docs/DESIGN.md §2).

Extends the single-answer `tee.AttestationReport` two ways:
  * BINDING: the payload commits to `source_hash` + `weights_hash` + `model_id`, so
    the hardware quote certifies *this exact public artifact produced this score* —
    no secret code, no swapped weights, no off-enclave computation. This is the
    load-bearing piece: because the artifact is public AND bound, cheating is
    publicly detectable rather than something we must prevent by hiding data.
  * MULTI-BENCHMARK: `results` is a vector of per-benchmark answers+cost instead of
    one answer, so a single attested run covers the whole owner-given suite.

The fixed-runtime-attests-variable-payload machinery is unchanged: one hardware
signature over {measurement, report_data} asserts both "this runtime image ran"
and "it produced this payload". Tampering any field recomputes `report_data()` and
breaks the quote (caught in `verify.py`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from ..gateway import signing
from ..tee.attestation import Platform, Quote


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str        # which benchmark this answer belongs to
    task_id: str          # the sampled task (validator re-derives the gold from the nonce)
    answer: str           # graded validator-side against the public gold
    cost_usd: float       # metered by the runtime for this task — un-forgeable


@dataclass(frozen=True)
class Proof:
    epoch: int
    nonce: str            # validator-issued per (miner, epoch); anti-replay / anti-best-of-N
    hotkey: str           # the miner identity; bound so a copier can't resubmit it
    source_hash: str      # sha256 of the public inference/orchestration source
    weights_hash: str     # sha256 of the public routing-model weights
    model_id: str         # the routing model / pool identifier
    results: tuple[BenchmarkResult, ...]
    total_cost_usd: float
    n_calls: int
    call_log_hash: str
    measurement: str      # runtime image hash; must be on the approved list
    quote: Quote | None = None

    def _payload(self) -> dict:
        d = asdict(self)          # recurses into the BenchmarkResult tuple
        d.pop("quote")
        return d

    def report_data(self) -> str:
        return signing.sha256_hex(self._payload())

    def attested_by(self, platform: Platform) -> "Proof":
        return replace(self, quote=platform.quote(self.measurement, self.report_data()))

    # --- convenience for the validator -------------------------------------
    def by_benchmark(self) -> dict[str, list[BenchmarkResult]]:
        out: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            out.setdefault(r.benchmark, []).append(r)
        return out

    # --- serialization for the decoupled miner->store->validator flow -------
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Proof":
        d = json.loads(s)
        q = d.get("quote")
        return cls(
            epoch=d["epoch"], nonce=d["nonce"], hotkey=d["hotkey"],
            source_hash=d["source_hash"], weights_hash=d["weights_hash"], model_id=d["model_id"],
            results=tuple(BenchmarkResult(**r) for r in d["results"]),
            total_cost_usd=d["total_cost_usd"], n_calls=d["n_calls"],
            call_log_hash=d["call_log_hash"], measurement=d["measurement"],
            quote=Quote(**q) if q else None,
        )
