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
from dataclasses import asdict, dataclass, field, replace

from ..gateway import signing
from ..tee.attestation import Platform, Quote


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str        # which benchmark this answer belongs to
    task_id: str          # the sampled task (validator re-derives the gold from the nonce)
    answer: str           # graded validator-side against the public gold
    cost_usd: float       # metered by the runtime for this task — un-forgeable
    # --- routing decision (fixed-harness architecture; empty on the legacy free-agent path) ---
    # Under the harness the miner's whole contribution is the DECISION, so the decision has to be in
    # the attested payload or a validator can only grade the outcome. Both fields are inside
    # `report_data` ⇒ inside the quote ⇒ as un-forgeable as the cost.
    chosen_rung: int = -1          # ladder entry point the head selected (-1 = not a routed run)
    rungs_used: tuple = ()         # pool indices actually invoked, in order — the cost trail, so a
                                   # validator can price the run without re-executing it
    distribution: tuple = ()       # the head's SOFT output over rungs. Copy-dedup fingerprints on
                                   # this rather than on answers: two heads that agree on argmax may
                                   # be honestly convergent, while near-identical distributions are
                                   # the same weights. Measured: independently-trained honest routers
                                   # reach 0.954 argmax agreement, above the old 0.95 copy threshold.


# THE ATTESTED PAYLOAD'S SHAPE IS VERSIONED, AND EACH VERSION'S FIELD LIST IS FROZEN FOREVER.
#
# `report_data` is a hash of the payload, so when it was built with `asdict(self)` every field added
# to this dataclass silently changed the hash of EVERY PROOF EVER ATTESTED. A genuine 2026 proof then
# fails under 2027 code — and fails as `report_data_mismatch`, which is indistinguishable from
# tampering. A routine software upgrade becomes a false accusation of forgery, and the one claim this
# whole design rests on ("a third party can check this LATER without re-running it") stops holding.
#
# Found by verifying a real recorded Intel-TDX proof from the hardware run: authentic, self-consistent,
# and rejected, because seven fields had been added since it was attested.
#
# To add a field: add a NEW version here. Never edit an existing one.
PROOF_SCHEMA = 2

_SCHEMA_FIELDS: dict[int, dict[str, tuple[str, ...]]] = {
    2: {
        "proof": ("schema", "epoch", "nonce", "hotkey", "source_hash", "weights_hash", "model_id",
                  "results", "total_cost_usd", "n_calls", "call_log_hash", "measurement",
                  "confined", "latency_s", "tokens_in", "tokens_out"),
        "result": ("benchmark", "task_id", "answer", "cost_usd", "chosen_rung", "rungs_used",
                   "distribution"),
    },
}


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
    schema: int = PROOF_SCHEMA
    # The payload as deserialized, for a proof written before payloads were versioned. Only its
    # SHAPE is used (see `_payload`) — verification must reproduce the field set that was actually
    # hashed at attestation time, or old proofs read as forged. Never re-serialized into a new proof.
    legacy_payload: dict | None = field(default=None, compare=False, repr=False)

    def _payload(self) -> dict:
        if self.legacy_payload is not None:
            # Project the CURRENT field values onto the OLD field set. Returning the stored dict
            # verbatim would be catastrophic and briefly was: `report_data` would then ignore the
            # proof's actual contents, so editing a graded answer or zeroing the cost still matched
            # the quote and a tampered proof verified. Caught by scripts/verify_demo.py, whose whole
            # point is that a verifier which only ever accepts has demonstrated nothing.
            old = self.legacy_payload
            rkeys = tuple(old["results"][0]) if old.get("results") else ()
            d = {k: getattr(self, k, old[k]) for k in old if k != "results"}
            if "results" in old:
                d["results"] = [{k: getattr(r, k) for k in rkeys} for r in self.results]
            return d
        f = _SCHEMA_FIELDS.get(self.schema)
        if f is None:
            # schema 1 is "whatever fields existed at the time", which is only reconstructable from
            # the serialized bytes. Without them there is no honest payload to hash, so refuse rather
            # than emit a plausible-looking wrong one.
            raise ValueError(f"proof schema {self.schema} has no field list; a schema-1 proof can "
                             f"only be verified from the JSON it was serialized as")
        f = _SCHEMA_FIELDS[self.schema]
        d = {k: getattr(self, k) for k in f["proof"] if k != "results"}
        d["results"] = [{k: getattr(r, k) for k in f["result"]} for r in self.results]
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
        d = dict(self.legacy_payload) if self.legacy_payload is not None else self._payload()
        d["quote"] = asdict(self.quote) if self.quote else None
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Proof":
        d = json.loads(s)
        q = d.get("quote")
        # A proof with no `schema` key predates payload versioning. Keep its payload verbatim so it
        # still verifies; a fresh proof always declares its schema and needs no such rescue.
        legacy = None if "schema" in d else {k: v for k, v in d.items() if k != "quote"}
        return cls(
            epoch=d["epoch"], nonce=d["nonce"], hotkey=d["hotkey"],
            source_hash=d["source_hash"], weights_hash=d["weights_hash"], model_id=d["model_id"],
            # COERCE THE SEQUENCE FIELDS BACK TO TUPLES. JSON has no tuple, so `rungs_used` and
            # `distribution` return as lists; leaving them that way makes an in-memory proof and a
            # round-tripped one unequal even though both hash the same. This is the same shape as the
            # earlier `sum([]) -> int 0 vs float 0.0` bug that produced `report_data_mismatch` for
            # agents making no pool calls, so it is pinned rather than left to chance.
            results=tuple(BenchmarkResult(**{**r,
                                             "rungs_used": tuple(r.get("rungs_used", ())),
                                             "distribution": tuple(r.get("distribution", ()))})
                          for r in d["results"]),
            total_cost_usd=d["total_cost_usd"], n_calls=d["n_calls"],
            call_log_hash=d["call_log_hash"], measurement=d["measurement"],
            # absent ⇒ False: an old proof that never asserted confinement must not be read as
            # having had it. Fails closed under a validator that gates on it.
            confined=bool(d.get("confined", False)),
            latency_s=float(d.get("latency_s", 0.0)),
            tokens_in=int(d.get("tokens_in", 0)), tokens_out=int(d.get("tokens_out", 0)),
            quote=Quote(**q) if q else None,
            schema=int(d.get("schema", 1)), legacy_payload=legacy,
        )
