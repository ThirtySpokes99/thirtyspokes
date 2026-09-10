"""Chain integration — commit-reveal, metagraph reads, weight setting.

Two implementations behind one interface:
  * MockChain      — in-memory; the offline simulator + tests use this.
  * the live seam lives in `v3/chain.py` (bittensor 11.x); this module keeps the mock.

The interface is deliberately tiny: what a miner needs to commit and what a
validator needs to read commitments + set weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Commitment:
    hotkey: str
    uid: int
    data: str          # the orch1|<repo>|<salted-hash> reveal string
    block: int


def _mock_beacon(epoch: int) -> str:
    """Deterministic mock beacon (PREDICTABLE — real unpredictability comes from the
    live block hash in BittensorChain.beacon)."""
    import hashlib
    return hashlib.sha256(f"koth-beacon|{epoch}".encode()).hexdigest()[:16]


GOV_PREFIX = "kothgov1|"       # owner-published approved-measurement record (H2/H4 governance)


class WeightRateLimited(RuntimeError):
    def __init__(self, retry_blocks: int):
        self.retry_blocks = retry_blocks
        super().__init__(f"weight update is rate-limited for {retry_blocks} more block(s)")


def _decode_raw_commitment(fields) -> str | None:
    """Pull the digest string out of a substrate CommitmentInfo `fields` list (one Raw(bytes) entry).

    The SCALE enum variant is `Raw<N>` where N is the payload's BYTE LENGTH (`Raw70`, `Raw32`, …) —
    never a bare `Raw`. Matching the literal key "Raw" therefore always missed, so every F7 proof
    commit read back as None: with `--commit-window` on, that DQs every miner as
    `commit_out_of_window`. Observed live on testnet 526."""
    for entry in fields or []:
        if hasattr(entry, "get"):
            raw = next((v for k, v in entry.items() if str(k).startswith("Raw")), None)
        else:
            raw = entry
        if isinstance(raw, str):
            return bytes.fromhex(raw[2:]).decode() if raw.startswith("0x") else raw
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode()
    return None


class Chain(Protocol):
    def current_block(self) -> int: ...
    def commit(self, hotkey: str, data: str) -> None: ...
    def revealed_commitments(self) -> list[Commitment]: ...
    def hotkeys(self) -> dict[int, str]: ...           # uid -> hotkey
    # `force` re-submits an UNCHANGED distribution, which the live chain otherwise skips. It is how
    # the daemon keeps `last_update` inside the subnet's activity cutoff during a stable reign.
    def set_weights(self, weights: dict[int, float], *, force: bool = False) -> None: ...
    def beacon(self, epoch: int) -> str: ...           # per-epoch seed for the task nonce
    # owner-governed approved measurements (MRTD/RTMR/runtime/TCB); None if unpublished
    def owner_measurements(self) -> dict | None: ...
    # subnet economics for the public feed (alpha price + emission); None offline
    def market(self) -> dict | None: ...
    # F7 anti-grind + F2 presence: a per-(hotkey, epoch), immediately-visible proof binding (the
    # proof's report_data + its inclusion block). Keyed by epoch so a validator scoring a SETTLED
    # (past-grace) epoch can still read that epoch's commit. Only used when the validator sets
    # commit_window.
    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None: ...
    def proof_commit(self, hotkey: str, epoch: int) -> tuple[str, int] | None: ...   # (digest, block)|None


@dataclass
class MockChain:
    """In-memory chain for offline runs. Registration = assigning a uid+hotkey."""

    _block: int = 0
    _uid_of: dict[str, int] = field(default_factory=dict)
    _commit: dict[str, Commitment] = field(default_factory=dict)   # hotkey -> latest
    _proof_commit: dict[tuple, tuple] = field(default_factory=dict)  # (hotkey, epoch) -> (digest, block)
    _weights: dict[int, float] = field(default_factory=dict)
    _next_uid: int = 0
    _owner_meas: dict | None = None

    def register(self, hotkey: str) -> int:
        if hotkey not in self._uid_of:
            self._uid_of[hotkey] = self._next_uid
            self._next_uid += 1
        return self._uid_of[hotkey]

    def deregister(self, hotkey: str) -> None:
        self._uid_of.pop(hotkey, None)
        self._commit.pop(hotkey, None)
        for k in [k for k in self._proof_commit if k[0] == hotkey]:
            del self._proof_commit[k]

    def advance(self, n: int = 1) -> None:
        self._block += n

    # --- Chain interface ---
    def current_block(self) -> int:
        return self._block

    def commit(self, hotkey: str, data: str) -> None:
        uid = self._uid_of.get(hotkey)
        if uid is None:
            raise ValueError(f"hotkey {hotkey} not registered")
        self._commit[hotkey] = Commitment(hotkey, uid, data, self._block)

    def revealed_commitments(self) -> list[Commitment]:
        return [c for c in self._commit.values() if c.hotkey in self._uid_of]

    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None:
        if hotkey not in self._uid_of:
            raise ValueError(f"hotkey {hotkey} not registered")
        self._proof_commit[(hotkey, epoch)] = (digest, self._block)

    def proof_commit(self, hotkey: str, epoch: int) -> tuple | None:
        return self._proof_commit.get((hotkey, epoch))

    def hotkeys(self) -> dict[int, str]:
        return {uid: hk for hk, uid in self._uid_of.items()}

    def set_weights(self, weights: dict[int, float], *, force: bool = False) -> None:
        del force                       # no activity cutoff offline: a refresh is a plain re-write
        self._weights = dict(weights)

    def beacon(self, epoch: int) -> str:
        return _mock_beacon(epoch)

    def publish_owner_measurements(self, record: dict) -> None:
        self._owner_meas = dict(record)

    def owner_measurements(self) -> dict | None:
        return self._owner_meas

    def market(self) -> dict | None:
        return None                     # no economics offline
