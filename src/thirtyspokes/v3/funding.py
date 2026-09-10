"""The miner's own OpenRouter key, sealed to the owner (§4, D18).

THE OWNER'S DECISION, 2026-09-08: a miner pays for their arm's worker calls on THEIR OWN OpenRouter
account, not through a balance the owner holds. The key travels sealed — the mirror of the mailbox
envelope in `access.py`: there the owner seals a credential to the miner's ed25519 hotkey; here the
miner seals an API key to the owner's ed25519 mailbox key (`orchestra-owner key`, the same hex a
miner already passes as `--owner-key`). Only the validator's mailbox seed opens it. The record is
signed by the hotkey, so a key found under a miner's prefix is that miner's and nobody else's, and
it names the registration it belongs to, so a record lifted from one prefix into another opens
nothing.

WHAT THE RECORD CARRIES BESIDE THE KEY. `cap_usd` is the miner's own ceiling on what one window may
spend on their key (§4's "miners set their own allowance", restored). The validator reads it at
arm time, bounds it by what the key itself reports it can still spend, and hands the gateway that
figure as the window's allowance (`OwnerGateway.bind`). A hit on the window's outcome table (§5.2c)
is debited against that figure exactly as a bought call is; only the fills reach the miner's
account, because only the fills reach a provider.

WHAT THIS DOES NOT CLOSE, STATED HERE BECAUSE THE GUIDES REPEAT IT. §11-1: a miner's own account
decides provider routing and bring-your-own-key, so `openai/gpt-5.4` on their key can be served by
a different endpoint than on the owner's. The request body still pins the provider block
(`openrouter.PROVIDER_ROUTING`), which narrows the drift; `Completion.provider` on every fill is
recorded per arm and the reveal publishes which `(model, provider)` pairs a miner-key arm was served
by that the owner's own arms were not. That is evidence, not a gate.

WHERE IT LIVES. One object, `openrouter/key.json`, under the miner's submission prefix, written with
the same prefix-scoped credential the tree is uploaded with — or, once that credential has expired
and the shot is spent, with a KEY-ONLY credential the owner issues for the `openrouter/` sub-prefix
alone (`Mailbox.issue_key`; `orchestra-owner issue --key-only`; `register-key --key-credential`).
`store.fetch_submission` ignores everything under that sub-prefix. Re-uploading the record replaces
the key or the cap; the validator reads it fresh every window.
"""
from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..gateway import signing
from .access import (KEY_PREFIX, AccessError, Registration, encode_ss58, seal, unseal,
                     verify_hotkey)

# Under the submission prefix, inside the sub-prefix a key-only credential is scoped to
# (`access.KEY_PREFIX`, `Mailbox.issue_key`).
KEY_NAME = KEY_PREFIX + "key.json"
KEY_VERSION = 1


@dataclass(frozen=True)
class SealedKey:
    """The record as it sits in the bucket: who, for which registration, how much, and the seal."""

    hotkey: str
    registration_id: str
    cap_usd: float
    ciphertext: bytes
    signature: str = ""

    def payload(self) -> bytes:
        """What the hotkey signs: every field but the signature, canonically encoded."""
        return signing.canonical(self._fields())

    def to_bytes(self) -> bytes:
        return json.dumps({**self._fields(), "sig": self.signature}, sort_keys=True).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> SealedKey:
        try:
            body = json.loads(raw.decode("utf-8"))
            if int(body["v"]) != KEY_VERSION:
                raise AccessError(f"sealed key record is version {body['v']}, not {KEY_VERSION}")
            return cls(hotkey=str(body["hotkey"]), registration_id=str(body["registration_id"]),
                       cap_usd=float(body["cap_usd"]),
                       ciphertext=base64.urlsafe_b64decode(str(body["ciphertext"])),
                       signature=str(body["sig"]))
        except AccessError:
            raise
        except Exception as exc:                 # noqa: BLE001 — any unreadable record is one fact
            raise AccessError(f"sealed key record is unreadable: {exc}") from exc

    def _fields(self) -> dict[str, Any]:
        return {"v": KEY_VERSION, "hotkey": self.hotkey, "registration_id": self.registration_id,
                "cap_usd": self.cap_usd,
                "ciphertext": base64.urlsafe_b64encode(self.ciphertext).decode("ascii")}


def seal_key(api_key: str, *, registration: Registration, cap_usd: float, owner_public_hex: str,
             sign: Callable[[bytes], str]) -> SealedKey:
    """The miner side: seal the key to the owner's mailbox key and sign the record with the hotkey.

    The owner's key is an ed25519 public key published as hex; `access.seal` addresses an SS58
    string because that is what the owner knows of a miner, so the hex is wrapped in the same
    encoding here rather than the seal growing a second entry point.
    """
    if not api_key or api_key != api_key.strip():
        raise AccessError("the API key is empty or carries surrounding whitespace")
    if not cap_usd > 0.0:
        raise AccessError(f"the per-window cap must be positive dollars, not {cap_usd!r}")
    try:
        owner = encode_ss58(bytes.fromhex(owner_public_hex))
    except Exception as exc:                     # noqa: BLE001 — a bad hex is one fact
        raise AccessError(f"the owner key is not a 32-byte hex public key: {exc}") from exc
    unsigned = SealedKey(hotkey=registration.hotkey, registration_id=registration.registration_id,
                         cap_usd=float(cap_usd), ciphertext=seal(api_key.encode("utf-8"), owner))
    return replace(unsigned, signature=sign(unsigned.payload()))


def open_key(record: SealedKey, seed: bytes, *, hotkey: str, registration_id: str) -> str:
    """The owner side: refuse a record that is not this hotkey's for this registration, then open.

    The signature is checked BEFORE the seal: an attacker who cannot sign as the hotkey cannot make
    the validator run an arm on a key of their choosing, and a record copied from another prefix
    names a registration this one is not.
    """
    if record.hotkey != hotkey or record.registration_id != registration_id:
        raise AccessError(f"sealed key record belongs to {record.hotkey}/{record.registration_id}, "
                          f"not to {hotkey}/{registration_id}")
    if not record.cap_usd > 0.0:
        raise AccessError(f"sealed key record caps the window at {record.cap_usd!r}, which buys "
                          f"nothing")
    verify_hotkey(record.hotkey, record.payload(), record.signature)
    try:
        return unseal(record.ciphertext, seed).decode("utf-8")
    except AccessError as exc:
        raise AccessError("sealed key record is not addressed to this validator's mailbox key: "
                          "the miner sealed it to a different owner key") from exc


__all__ = ["KEY_NAME", "SealedKey", "open_key", "seal_key"]
