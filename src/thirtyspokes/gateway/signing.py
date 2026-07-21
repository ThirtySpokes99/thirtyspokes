"""ed25519 signing/verification + canonical hashing for receipts and requests.

Asymmetric on purpose: the gateway (and each miner hotkey) can SIGN; validators —
holding only public keys — can VERIFY. This is what makes a receipt un-forgeable
yet publicly checkable. In production the miner key is the Bittensor hotkey
(sr25519/ed25519 via substrate); here we use ed25519 directly for both, and a
hotkey is simply its public-key hex.
"""

from __future__ import annotations

import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical(obj) -> bytes:
    """Deterministic, hash-stable serialization of a JSON-able object."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_hex(data) -> str:
    b = data if isinstance(data, bytes) else canonical(data)
    return hashlib.sha256(b).hexdigest()


class Signer:
    """A keypair. `public_hex` is the identity (a hotkey, or the gateway id)."""

    def __init__(self, priv: Ed25519PrivateKey | None = None):
        self._priv = priv or Ed25519PrivateKey.generate()

    @property
    def public_hex(self) -> str:
        return self._priv.public_key().public_bytes_raw().hex()

    def sign(self, data: bytes) -> str:
        return self._priv.sign(data).hex()


def verify(public_hex: str, data: bytes, sig_hex: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            bytes.fromhex(sig_hex), data)
        return True
    except Exception:
        return False
