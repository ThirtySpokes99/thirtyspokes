"""The per-epoch POOL REFERENCE — what every pool model scored, on the asks the miners ran.

`verify.router_headroom` scores a router against what was ACHIEVABLE at its own price: the Zero
frontier (randomising over fixed pool models) and the budget-constrained per-ask oracle. Both need
every `(ask, model)` cell for the epoch's slice — and a validator runs NO inference, so it cannot
know what the other models would have answered. Without this the router scalar silently falls back
to the old absolute one.

So the OWNER measures the pool once per epoch and publishes the result, using the same shape as the
governance record (`koth/governance.py`): the body is content-addressed in the owner's public bucket,
and only its sha256 goes on-chain. Validators fetch and verify by hash, so a substituted or tampered
reference is rejected rather than trusted.

WHY A SIGNATURE AND NOT AN ON-CHAIN HASH. The first cut committed the record's sha256 on-chain, the
way `koth/governance.py` does. That cannot work, for a reason the design doc already records about
miners: the chain gives each hotkey exactly ONE commitment slot (`CommitmentOf[(netuid, hotkey)]`),
and `set_commitment` overwrites it. The owner's slot already holds the approved-measurement record
(`kothgov1|…`), so writing a reference digest there each epoch would erase the subnet's governance —
every validator would read `mrtd_gate_unset` and refuse to score at all. A per-epoch on-chain anchor
is therefore not available to the owner.

So the record is SIGNED with the owner's hotkey and served from the owner's bucket at a path both
sides derive from `(epoch, nonce)`. Integrity is still rooted on-chain: validators resolve the owner
hotkey from `SubnetOwnerHotkey` and reject any record that key did not sign. This is strictly cheaper
too — no per-epoch extrinsic — and it leaves the governance slot alone.

WHY THE VALIDATOR MUST CHECK MORE THAN THE SIGNATURE. A signature only proves the owner wrote the
body — not that it describes THIS epoch. A reference from an easier epoch would make every miner look
good (a low Zero frontier is easy to beat) and a reference from a harder one would make everyone look
bad. So the record carries `(epoch, nonce, suite_version, n_per_bench, task_ids)` and `matches`
rejects any mismatch. The nonce is chain-derived and unpredictable, so the owner cannot pre-select a
flattering slice either.

COST. `n_per_bench x |pool|` calls per epoch — 8 x 6 = 48, well under a dollar — and a slice may be
reused across several epochs to amortise it further.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def record_path(epoch: int, nonce: str) -> str:
    """Where this epoch's reference lives. Derived from `(epoch, nonce)` so a validator can address
    it without being told a hash — the nonce is chain-derived, so the path is unforgeable-by-guessing
    and unique per epoch."""
    return f"reference/{int(epoch)}-{str(nonce)[:32]}.json"


def envelope(record: dict, sign) -> dict:
    """Wrap a record with the owner's signature over its canonical bytes."""
    return {"v": 1, "record": record, "sig": sign(canonical(record))}


def open_envelope(raw: bytes, *, owner_ss58: str, verify_sig) -> dict:
    """Unwrap + authenticate. Raises unless the OWNER's key signed exactly these bytes."""
    env = json.loads(raw)
    rec, sig = env.get("record"), env.get("sig")
    if not isinstance(rec, dict) or not sig:
        raise ValueError("malformed pool-reference envelope")
    if not verify_sig(canonical(rec), sig, owner_ss58):
        raise ValueError("pool reference not signed by the subnet owner")
    return rec


def wallet_signer(wallet):
    """Sign with a Bittensor wallet's hotkey — the same identity `SubnetOwnerHotkey` names."""
    return lambda data: wallet.hotkey.sign(data).hex()


def keypair_verifier():
    """Verify an sr25519 signature against an ss58 address. Lazily imported so the offline test
    path (and any validator that never sees a reference) needs no wallet library."""
    def verify_sig(data: bytes, sig: str, ss58: str) -> bool:
        try:
            from bittensor_wallet import Keypair
            return bool(Keypair(ss58_address=ss58).verify(data, bytes.fromhex(sig)))
        except Exception:                   # noqa: BLE001 — unverifiable == not verified
            return False
    return verify_sig


# A chain read that has not returned in this long is treated as wedged. Generous: a slow archive
# node answers in seconds, so anything past this is not slowness.
CHAIN_CALL_TIMEOUT_S = 90.0
# If this many epochs pass with no successful publish, the loop declares itself dead. Validators
# fall back to the absolute scalar without a reference, so a silent publisher degrades scoring from
# "routing" to "~98% pool, ~2% miner" (DESIGN.md (removed with v2, 2026-09-07) §5.0) with nothing anywhere reporting a fault.
MAX_STALLED_EPOCHS = 3


if __name__ == "__main__":
    main()
