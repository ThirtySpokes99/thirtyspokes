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
    # Was the untrusted agent actually run under no-egress confinement (koth/confine.py)?
    # ATTESTED, because the guarantee used to be unobservable: `run_agent_confined` silently
    # degrades to a plain subprocess when `confinement_available()` is False (it returns False on
    # ANY exception, including its own 10s probe timeout, and lru_caches that for the boot). An
    # unconfined agent has network egress and can call an off-allow-list model with a key embedded
    # in its own weights.bin — bypassing PinnedBackend, the MeteringProxy, the budget ceiling and
    # the cost tiebreak at once — while `no_pool_call` is satisfied by one token call. The
    # MRTD/RTMR gates prove WHICH IMAGE booted, not what it did inside, so without this field a
    # fully-enforcing validator could not tell the two runs apart. In the payload ⇒ covered by
    # `report_data()` ⇒ covered by the quote ⇒ un-forgeable.
    confined: bool = False
    # Latency and tokens are part of the product promise ("best answer at the lowest price" is also
    # a speed claim) and were measured by the metering proxy but discarded before the proof. In the
    # payload => covered by report_data => covered by the quote, so they are as un-forgeable as cost.
    latency_s: float = 0.0          # wall-clock across every pool call in the run
    tokens_in: int = 0
    tokens_out: int = 0
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
            # absent ⇒ False: an old proof that never asserted confinement must not be read as
            # having had it. Fails closed under a validator that gates on it.
            confined=bool(d.get("confined", False)),
            latency_s=float(d.get("latency_s", 0.0)),
            tokens_in=int(d.get("tokens_in", 0)), tokens_out=int(d.get("tokens_out", 0)),
            quote=Quote(**q) if q else None,
        )
