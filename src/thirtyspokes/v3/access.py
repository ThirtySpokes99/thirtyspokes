"""The encrypted mailbox, the scoped credential and the one shot (§7 steps 2-5, the build plan M6).

PORTED FROM `teutonic/access/`, DELIBERATELY, BECAUSE THIS PATTERN IS ALREADY IN PRODUCTION THERE.
`teutonic/access/crypto.py` (an envelope sealed to the hotkey pubkey, owner-signed),
`teutonic/access/repository.py` (one-shot enforced server-side — *"hotkey submission eligibility is
permanently consumed"*), `miner/get_upload_auth.py` (mailbox poll -> scoped credential) and
`teutonic/storage/r2_credentials.py` (the prefix-scoped R2 token) all solve problems v3 has
verbatim. Re-deriving them would be waste, and worse than waste: each carries a decision whose
reason is not visible from the outside, and re-deriving would re-decide it by guess. The fifth
piece of the pattern — `access/contracts.py::ready_signal_payload`, two SHA-256 packed into one
commitment slot — is in `v3/chain.py`, where a commitment is read.

WHAT THIS MODULE HAS TO GET RIGHT, AND WHY EACH ONE IS LOAD-BEARING

1. **The mailbox is public, so the credential must be sealed.** The owner cannot push to a miner —
   there is no address to push to — so the credential is published at a derivable key and the miner
   polls. That credential is a WRITE token for a prefix. Published in the clear it would let anyone
   who fetched the URL upload into that miner's submission prefix, which is a stranger spending a
   hotkey's one shot (§7). So the envelope is sealed to the hotkey's own public key, and signed by
   the owner so a miner cannot be handed a credential pointing at somebody else's bucket.

2. **The one shot is enforced HERE, not in the miner's client.** A rule enforced client-side is a
   suggestion: the client is the miner's. `Mailbox` is the owner's own durable ledger, and
   eligibility is checked inside it before any credential exists.

3. **Eligibility is consumed by the READY SIGNAL, not by the credential request** — and that is the
   most easily-mis-ported decision in the whole pattern. §8b/M6 exit 2 records that at ~72 GB over
   hours an interrupted upload is the COMMON case, so a credential that expires mid-upload is
   ordinary rather than exceptional. If asking for a credential spent the shot, the owner's own
   TTL would destroy submissions, which is the "Kind" property broken by the mechanism that exists
   to protect it. So credentials ROTATE (generation 1, 2, 3 …) against the same server-derived
   prefix — a rotation is more access to the one submission, never a second submission — and the
   shot is spent when the miner signals ready on chain. That is what `revocation_event:
   finalized_ready_signal` in the envelope announces to the miner, and it is teutonic's arrangement
   exactly: `record_parent_token_active` and `request_credential_rotation` both refuse once an
   upload row exists for the hotkey, and `accept_ready_signal` is what creates that row.

4. **The refusal is by HOTKEY, never by registration.** §7: *"a hotkey with any prior entry is
   refused forever."* A `registration_id` includes the registration block, so a hotkey that
   deregisters and re-registers presents a NEW registration id for the same hotkey — keyed on
   registration, the one-shot would be a one-shot-per-registration and the price of a second
   submission would be a registration burn.

WHAT THE SEAL IS, AND WHY IT IS NOT LIBSODIUM'S. teutonic uses PyNaCl's `SealedBox`
(X25519 + XSalsa20-Poly1305). This repository does not depend on PyNaCl, and `cryptography` — which
it does depend on, for `gateway/signing.py` — has no XSalsa20, so a byte-compatible
`crypto_box_seal` cannot be built on it. The construction below is the same shape (an ephemeral
X25519 to the recipient's key, one message per key) with the AEAD swapped for HKDF-SHA256 +
ChaCha20-Poly1305. Both ends are in this repository, so wire compatibility with libsodium buys
nothing; what is kept is the property that only the hotkey's secret key opens the envelope. The
ed25519 -> X25519 conversion is libsodium's own
`crypto_sign_ed25519_pk_to_curve25519` / `..._sk_to_curve25519`, which is why an ed25519 hotkey is
required exactly as teutonic requires one (`crypto_type == 0`).
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..gateway import signing
from ..gateway.signing import Signer
# `Commitment` and its `ReadySignal` live in `chain.py` because that is where a commitment is READ,
# and this module must not grow a second encoding of the same slot. A ready signal spelled two ways
# in one package is a miner committing one string while the validator looks for another, with both
# halves passing their own tests — the exact failure `chain.py`'s own docstring records about
# `r2ready:v1` being copied from teutonic rather than re-invented.
from .chain import Commitment

# Bittensor's SS58 network prefix. One byte, so the two-byte forms substrate also defines are out of
# scope: v3 reads one chain and every hotkey on it encodes at 42.
SS58_NETWORK = 42

# Domain separation for the seal. It goes into the HKDF `info` AND the AEAD's associated data
# alongside both public keys, so a ciphertext cannot be re-presented as an envelope for a different
# recipient: the recipient key is bound into the key derivation and authenticated in the tag.
SEAL_INFO = b"v3-mailbox-v1"

# A CONSTANT nonce, and it is correct rather than an oversight. The key is derived from a FRESH
# ephemeral X25519 keypair on every seal, so no key is ever used twice and the (key, nonce) pair is
# unique by construction — the same argument libsodium's sealed box and HPKE's base mode make. A
# random nonce here would be strictly weaker per byte (it would spend 12 bytes to restate a
# uniqueness the ephemeral key already gives) and it would hide the reasoning.
_SEAL_NONCE = b"\0" * 12

# What the envelope announces about itself, so a miner's client can refuse an envelope that does not
# say these things rather than discovering the difference by uploading into it (`open_credential`).
CREDENTIAL_SCOPE = "object-read-write"
REVOCATION_EVENT = "finalized_ready_signal"
SUBMISSION_POLICY = "one_per_hotkey"
PROTOCOL_VERSION = 1

# R2's ceiling on a temporary credential (7 days) and v3's default. A 72 GB tree over a domestic
# connection is hours, not days, so the default is not the binding constraint — rotation is what
# covers the miner whose upload outlives it, which is the whole reason rotation exists (point 3).
MAX_CREDENTIAL_TTL_SECONDS = 604_800

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIGNATURE = re.compile(r"^[0-9a-fA-F]{128}$")

_BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {char: index for index, char in enumerate(_BASE58)}
_SS58_PREFIX = b"SS58PRE"
_P = 2**255 - 19


class AccessError(Exception):
    """A refusal from the owner's side of the mailbox: a spent hotkey, a malformed address, an
    envelope that is not this miner's. Never a repair — a hotkey has one shot, so a mailbox that
    coerced a near-miss would spend it on something the miner did not ask for."""


# --- SS58 (ported from teutonic/access/crypto.py) ------------------------------------------------


def _base58_encode(value: bytes) -> str:
    number = int.from_bytes(value, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _BASE58[remainder] + encoded
    return "1" * (len(value) - len(value.lstrip(b"\0"))) + encoded


def _base58_decode(value: str) -> bytes:
    number = 0
    for char in value:
        if char not in _BASE58_INDEX:
            raise AccessError(f"invalid base58 character {char!r} in SS58 address")
        number = number * 58 + _BASE58_INDEX[char]
    decoded = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\0" * (len(value) - len(value.lstrip("1"))) + decoded


def encode_ss58(public_key: bytes, *, network: int = SS58_NETWORK) -> str:
    """A 32-byte ed25519 public key as the address a miner reads off their wallet."""
    if len(public_key) != 32:
        raise AccessError("an ed25519 public key must contain 32 bytes")
    payload = bytes([network]) + public_key
    checksum = hashlib.blake2b(_SS58_PREFIX + payload, digest_size=64).digest()[:2]
    return _base58_encode(payload + checksum)


def decode_ss58(address: str, *, network: int = SS58_NETWORK) -> bytes:
    """The public key inside an SS58 address, checksum verified.

    The checksum is not decoration here. This key is the seal's recipient, so a mistyped address
    that decoded anyway would produce an envelope nobody can open — the miner would poll the mailbox
    forever while the owner's ledger recorded a credential as issued.
    """
    decoded = _base58_decode(address)
    if len(decoded) != 35:
        raise AccessError(f"SS58 address {address!r} does not decode to 35 bytes")
    payload, checksum = decoded[:-2], decoded[-2:]
    if payload[0] != network:
        raise AccessError(f"SS58 address {address!r} is not on network {network}")
    if checksum != hashlib.blake2b(_SS58_PREFIX + payload, digest_size=64).digest()[:2]:
        raise AccessError(f"SS58 address {address!r} has an invalid checksum")
    return payload[1:]


def _signature_bytes(signature: str) -> bytes:
    """Hex (`0x…`) or base64, because the two tools that produce these disagree: `bittensor`'s
    `Keypair.sign` hands back bytes an operator usually hexes, while teutonic's manifest writer
    base64s them. Accepting both costs four lines and removes a class of "correct signature,
    rejected" that would spend a shot on an encoding."""
    raw = signature.strip()
    try:
        if raw.startswith("0x"):
            decoded = bytes.fromhex(raw[2:])
        elif _HEX_SIGNATURE.fullmatch(raw):
            decoded = bytes.fromhex(raw)
        else:
            decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise AccessError("signature is neither hexadecimal nor base64") from exc
    if len(decoded) != 64:
        raise AccessError("an ed25519 signature must contain 64 bytes")
    return decoded


def verify_hotkey(hotkey: str, message: bytes, signature: str) -> None:
    """Raise unless `signature` is this hotkey's over `message`. Used on the manifest by
    `store.py`."""
    try:
        Ed25519PublicKey.from_public_bytes(decode_ss58(hotkey)).verify(
            _signature_bytes(signature), message)
    except AccessError:
        raise
    except Exception as exc:
        raise AccessError(f"signature does not verify under hotkey {hotkey}") from exc


# --- the seal ------------------------------------------------------------------------------------


def _x25519_public(ed25519_public: bytes) -> bytes:
    """libsodium's `crypto_sign_ed25519_pk_to_curve25519` — u = (1 + y) / (1 - y), the birational
    map between the two curves.

    The owner only ever learns a miner's SS58 address, which is an ed25519 *signing* key, and X25519
    is what a sealed box needs. This is the standard conversion between the two — the same one
    teutonic calls into PyNaCl for.
    """
    y = int.from_bytes(ed25519_public, "little") & ((1 << 255) - 1)
    if y >= _P:
        raise AccessError("hotkey public key is not a canonical ed25519 point")
    denominator = (1 - y) % _P
    if denominator == 0:
        raise AccessError("hotkey public key is a small-order point and cannot receive a seal")
    return ((1 + y) * pow(denominator, _P - 2, _P) % _P).to_bytes(32, "little")


def _x25519_secret(seed: bytes) -> bytes:
    """libsodium's `crypto_sign_ed25519_sk_to_curve25519`: the clamped first half of SHA-512(seed).

    `seed` is the 32 bytes an ed25519 hotkey is generated from —
    `Ed25519PrivateKey.private_bytes_raw()`, or the `secretSeed` a bittensor hotkey file carries,
    which is what teutonic's `signing_key` reads.
    """
    if len(seed) != 32:
        raise AccessError("an ed25519 hotkey seed must contain 32 bytes")
    scalar = bytearray(hashlib.sha512(seed).digest()[:32])
    scalar[0] &= 248
    scalar[31] &= 127
    scalar[31] |= 64
    return bytes(scalar)


def _seal_key(shared: bytes, ephemeral: bytes, recipient: bytes) -> bytes:
    return HKDF(algorithm=SHA256(), length=32, salt=None,
                info=SEAL_INFO + ephemeral + recipient).derive(shared)


def seal(plaintext: bytes, hotkey: str) -> bytes:
    """Encrypt to a hotkey address. Only that hotkey's secret key opens the result."""
    recipient = _x25519_public(decode_ss58(hotkey))
    ephemeral = X25519PrivateKey.generate()
    public = ephemeral.public_key().public_bytes_raw()
    key = _seal_key(ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient)),
                    public, recipient)
    return public + ChaCha20Poly1305(key).encrypt(_SEAL_NONCE, plaintext, public + recipient)


def unseal(ciphertext: bytes, seed: bytes) -> bytes:
    """Open an envelope with the hotkey's seed. The miner side; the owner never runs this."""
    if len(ciphertext) < 32 + 16:
        raise AccessError("sealed envelope is too short to hold an ephemeral key and a tag")
    ephemeral, body = ciphertext[:32], ciphertext[32:]
    private = X25519PrivateKey.from_private_bytes(_x25519_secret(seed))
    recipient = private.public_key().public_bytes_raw()
    key = _seal_key(private.exchange(X25519PublicKey.from_public_bytes(ephemeral)),
                    ephemeral, recipient)
    try:
        return ChaCha20Poly1305(key).decrypt(_SEAL_NONCE, body, ephemeral + recipient)
    except Exception as exc:
        raise AccessError("sealed envelope is not addressed to this hotkey") from exc


# --- identity derived from finalized chain data --------------------------------------------------


@dataclass(frozen=True)
class Registration:
    """One hotkey's registration, as public finalized chain facts and nothing else.

    Everything downstream — the R2 prefix, the mailbox key, the on-chain ready signal — is derived
    from these four numbers, so BOTH SIDES compute the same identity independently and neither has
    anything to choose. A miner-chosen prefix would let one miner write into another's submission.
    """

    netuid: int
    uid: int
    hotkey: str
    registration_block: int

    def __post_init__(self) -> None:
        if min(self.netuid, self.uid, self.registration_block) < 0:
            raise AccessError("netuid, uid and registration_block must be non-negative")
        decode_ss58(self.hotkey)

    @property
    def registration_id(self) -> str:
        body = json.dumps({"hotkey": self.hotkey, "netuid": self.netuid,
                           "registration_block": self.registration_block, "uid": self.uid},
                          sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"v3-registration-v1\0" + body).hexdigest()

    @property
    def prefix(self) -> str:
        """Slash-terminated, because it scopes a credential: `submissions/<id>` without the slash
        would also match `submissions/<id>evil/`."""
        return f"submissions/{self.registration_id}/"

    @property
    def key_prefix(self) -> str:
        """Where the miner's sealed OpenRouter key lives (`funding.py`, D18): a sub-prefix of the
        submission, so a credential scoped to it can write the key and nothing of the tree."""
        return self.prefix + KEY_PREFIX


# The sub-prefix under a submission that holds the miner's sealed OpenRouter key and nothing the
# manifest names (`funding.KEY_NAME` is `<KEY_PREFIX>key.json`). `store.fetch_submission` ignores
# everything under it; `Mailbox.issue_key` scopes a credential to exactly it, which is what lets a
# miner rotate a key after the submission credential expired and the shot is spent without ever
# regaining write access to the tree.
KEY_PREFIX = "openrouter/"


def mailbox_key(registration_id: str, generation: int) -> str:
    """Where one credential generation is published. Derivable by the miner, hence pollable.

    Zero-padded so the keys sort in generation order in a bucket listing, which is what makes "which
    generations are outstanding and must be revoked" answerable from the store alone.
    """
    if not _HEX_DIGEST.fullmatch(registration_id):
        raise AccessError("registration id must be a lowercase SHA-256 digest")
    if generation < 1:
        raise AccessError("credential generation starts at one")
    return f"mailbox/v1/{registration_id}/{generation:020d}.bin"


def key_mailbox_key(registration_id: str, generation: int) -> str:
    """Where one KEY-ONLY credential generation is published (`Mailbox.issue_key`, D18). Its own
    series beside the submission credentials', so neither numbering moves the other's."""
    if not _HEX_DIGEST.fullmatch(registration_id):
        raise AccessError("registration id must be a lowercase SHA-256 digest")
    if generation < 1:
        raise AccessError("credential generation starts at one")
    return f"mailbox/v1/{registration_id}/key-{generation:020d}.bin"


# --- the prefix-scoped R2 credential (ported from teutonic/storage/r2_credentials.py) ------------


@dataclass(frozen=True)
class Credential:
    """What the miner uploads with. Scoped to one prefix and expiring, never a bucket-wide key."""

    endpoint: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    session_token: str
    expires_at: datetime


def scoped_credential(*, endpoint: str, account_id: str, parent_access_key_id: str,
                      parent_secret_access_key: str, bucket: str, prefix: str,
                      ttl_seconds: int = MAX_CREDENTIAL_TTL_SECONDS,
                      issued_at_unix: int | None = None) -> Credential:
    """Mint a Cloudflare R2 prefix-scoped temporary credential — a signed JWT, no network call.

    Ported from teutonic, whose comment records the one non-obvious thing: Cloudflare rejects the
    otherwise-documented fine-grained `actions` claim on that account, so production credentials use
    the platform's prefix-scoped `object-read-write` capability instead. The port drops the
    `object_path` and `actions` variants, which existed there for capability probes v3 never runs.

    The prefix is the scope, and this function is never called with a caller-supplied one:
    `Mailbox.issue` derives it from `Registration`. That is what makes "one miner cannot write into
    another's submission" a property of the token rather than a promise about the client.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/"):
        raise AccessError("R2 endpoint must be an HTTPS origin without a path")
    if not (account_id and parent_access_key_id and parent_secret_access_key and bucket):
        raise AccessError("account, parent credential and bucket are required")
    if not prefix or not prefix.endswith("/"):
        raise AccessError("a credential prefix must be slash-terminated")
    if not 1 <= ttl_seconds <= MAX_CREDENTIAL_TTL_SECONDS:
        raise AccessError(f"credential TTL must be within 1..{MAX_CREDENTIAL_TTL_SECONDS} seconds")

    issued_at = int(time.time()) if issued_at_unix is None else issued_at_unix
    expires_at = issued_at + ttl_seconds
    claims = {"aud": parsed.netloc, "bucket": bucket, "exp": expires_at, "iat": issued_at,
              "iss": parent_access_key_id,
              "paths": {"objectPaths": [], "prefixPaths": [prefix]},
              "scope": CREDENTIAL_SCOPE, "sub": account_id}
    signing_input = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, sort_keys=True,
                                       separators=(",", ":")).encode()) + "." + _b64url(
        json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    token = signing_input + "." + _b64url(hmac.new(
        parent_secret_access_key.encode(), signing_input.encode(), hashlib.sha256).digest())
    return Credential(
        endpoint=endpoint, bucket=bucket, access_key_id=parent_access_key_id,
        secret_access_key=hashlib.sha256(token.encode()).hexdigest(),
        session_token=base64.b64encode(f"jwt/{token}".encode()).decode(),
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise AccessError("a credential expiry must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --- the mailbox and the one shot ----------------------------------------------------------------


@dataclass(frozen=True)
class Submission:
    """The permanent record of a spent shot. Retained forever (§8b.7) even after the weights go."""

    hotkey: str
    registration_id: str
    manifest_sha256: str
    block: int
    generations: int


@dataclass(frozen=True)
class Outstanding:
    """One hotkey's issuing history, as the operator's revocation list needs to read it.

    `generations` is the HIGHEST generation issued, so the credentials to revoke are `1..generations`
    — they are cumulative rather than replaced, which is why a rotation extends a miner's window
    instead of shortening it, and why revoking only the newest would leave the earlier ones live.
    """

    hotkey: str
    registration_id: str
    prefix: str
    generations: int
    spent: bool


class Mailbox:
    """The owner's durable ledger: who has been issued a credential, and whose shot is spent.

    Durable because the one-shot has to survive a validator restart, and a ledger that lived in
    memory would hand every hotkey a fresh shot on every owner deploy. The file is written through a
    temporary and `replace`d, the pattern `koth/arena_neuron.py` (removed with v2, 2026-09-07) used for the arena's
    history — a torn state file here loses the record that a shot was spent, which is the one piece
    of state in this module that cannot be reconstructed from anywhere else.

    `mint` is how a credential comes into existence: `(prefix) -> Credential`. It is a parameter
    rather than a call to `scoped_credential` because minting is the only step that touches
    Cloudflare's account configuration, and because passing the prefix IN is what stops the prefix
    being an argument a caller could get wrong — `issue` derives it from the registration and hands
    it over, so a credential scoped to somebody else's prefix is not a check that can be forgotten.
    """

    def __init__(self, path: Path | str, signer: Signer,
                 mint: Callable[[str], Credential]) -> None:
        self.path = Path(path)
        self.signer = signer
        self.mint = mint
        self._issued: dict[str, dict[str, Any]] = {}
        self._spent: dict[str, dict[str, Any]] = {}
        self._restore()

    @property
    def owner_public_hex(self) -> str:
        """Published, so a miner can verify the envelope came from the owner, not from a bucket."""
        return self.signer.public_hex

    def spent(self, hotkey: str) -> bool:
        return hotkey in self._spent

    def submission(self, hotkey: str) -> Submission | None:
        record = self._spent.get(hotkey)
        return None if record is None else Submission(**record)

    def outstanding(self) -> tuple[Outstanding, ...]:
        """Every hotkey that holds issued generations, and whether its shot is already spent.

        THIS EXISTS TO DISCHARGE THE OBLIGATION `consume` SAYS THIS MODULE CANNOT. That docstring
        requires the operator to revoke the R2 credential for every generation up to
        `Submission.generations` once a shot is spent, or the miner keeps write access to a prefix
        whose contents the chain now names. Until this accessor there was no way to ask which
        hotkeys those were without reading the ledger file by eye — and an obligation whose input is
        a JSON file an operator parses by hand is an obligation that quietly stops being met.

        Sorted by hotkey so two runs of `thirtyspokes-owner status` are diffable.
        """
        return tuple(sorted(
            (Outstanding(hotkey=hotkey, registration_id=str(record["registration_id"]),
                         prefix=str(record["prefix"]), generations=int(record["generation"]),
                         spent=hotkey in self._spent)
             for hotkey, record in self._issued.items()),
            key=lambda row: row.hotkey))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Re-read the ledger under an exclusive lock, so a mutation cannot erase another writer's.

        THERE ARE TWO WRITERS AND THEY ARE IN DIFFERENT PROCESSES: the owner's tool `issue`s, the
        validator daemon `consume`s, and both persist this one file. Neither re-read before writing,
        and `_persist` writes the WHOLE in-memory view — so an `issue` landing between the daemon's
        startup read and its next `consume` was silently dropped.

        Measured 2026-09-02, before `thirtyspokes-owner` existed, which is exactly what had kept it
        unreachable: with one writer the stale copy is always current. Two harms follow, and the
        second is the serious one. A rotation re-mints **generation 1** over the same mailbox key,
        so a miner polling that key can be handed a different envelope than the one they began with.
        And `Submission.generations` under-counts, which is the number `consume`'s docstring makes
        the operator revoke against — so a live write credential for a prefix already committed on
        chain would never be revoked, leaving open precisely the swap that paragraph exists to close.

        The lock is a SIDECAR file and never `self.path` itself: `_persist` replaces the ledger by
        rename, so a descriptor held on it would guard an unlinked inode while the next writer walked
        straight in.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                self._restore()
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def issue(self, registration: Registration) -> tuple[str, bytes]:
        """Mint, sign and seal the next credential generation. Returns `(mailbox key, ciphertext)`.

        REFUSES A SPENT HOTKEY — server-side, before a credential exists, which is M6 exit 1. The
        refusal is on the HOTKEY and not on the registration id: §7 says a hotkey with any prior
        entry is refused *forever*, and a registration id changes when a hotkey re-registers.

        A repeat call for an UNSPENT hotkey is a rotation, not a second shot (module docstring point
        3): it issues generation N+1 against the identical derived prefix, so what the miner gets is
        more time to finish the one upload they are entitled to. The owner publishes the ciphertext
        at the returned key; that bucket is public and needs no protection, because the envelope is
        sealed to the hotkey and signed by the owner.
        """
        with self._transaction():
            return self._issue(registration)

    def _issue(self, registration: Registration) -> tuple[str, bytes]:
        """`issue`'s body, run with the ledger freshly re-read under the lock (`_transaction`).

        Split out rather than inlined because the spent check and the generation counter both read
        state a concurrent writer may have moved, and a body that could be called without the
        re-read is a body somebody will eventually call that way.
        """
        if self.spent(registration.hotkey):
            raise AccessError(
                f"{registration.hotkey}: hotkey submission eligibility is permanently consumed")
        generation = int(self._issued.get(registration.hotkey, {}).get("generation", 0)) + 1
        ciphertext = self._sealed(registration, registration.prefix, generation)
        key = mailbox_key(registration.registration_id, generation)
        self._issued.setdefault(registration.hotkey, {}).update({
            "registration_id": registration.registration_id, "prefix": registration.prefix,
            "generation": generation})
        self._persist()
        return key, ciphertext

    def issue_key(self, registration: Registration) -> tuple[str, bytes]:
        """Mint, sign and seal a KEY-ONLY credential (D18): write access to `registration.key_prefix`
        and nothing else, so it is issued to a SPENT hotkey too. Returns `(mailbox key, ciphertext)`.

        This is the rotation path the one-shot would otherwise close: once the submission credential
        has expired and the shot is spent, `issue` rightly refuses, but a miner whose OpenRouter key
        died or leaked still has to be able to replace it. The credential's prefix ends in
        `openrouter/`, which `store.fetch_submission` ignores entirely, so nothing this credential
        can write is anything the commitment covers — the revocation obligation `consume` states
        does not extend to it. Its generations are counted in their own series (`key_mailbox_key`).
        """
        with self._transaction():
            generation = int(self._issued.get(registration.hotkey, {})
                             .get("key_generation", 0)) + 1
            ciphertext = self._sealed(registration, registration.key_prefix, generation)
            key = key_mailbox_key(registration.registration_id, generation)
            self._issued.setdefault(registration.hotkey, {}).update({
                "registration_id": registration.registration_id,
                "key_generation": generation})
            self._persist()
            return key, ciphertext

    def _sealed(self, registration: Registration, prefix: str, generation: int) -> bytes:
        """Mint for `prefix`, sign the envelope, seal it to the hotkey. Shared by both issuers."""
        credential = self.mint(prefix)
        if credential.expires_at <= datetime.now(timezone.utc):
            raise AccessError("refusing to publish a credential that has already expired")
        envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "signature_scheme": "ed25519",
            "owner_identity": self.owner_public_hex,
            "netuid": registration.netuid,
            "uid": registration.uid,
            "hotkey": registration.hotkey,
            "registration_id": registration.registration_id,
            "registration_block": registration.registration_block,
            "credential_generation": generation,
            "r2_endpoint": credential.endpoint,
            "bucket": credential.bucket,
            "allowed_prefix": prefix,
            "credential_scope": CREDENTIAL_SCOPE,
            "revocation_event": REVOCATION_EVENT,
            "submission_policy": SUBMISSION_POLICY,
            "access_key_id": credential.access_key_id,
            "secret_access_key": credential.secret_access_key,
            "session_token": credential.session_token,
            "expires_at": _utc_text(credential.expires_at),
        }
        signed = {**envelope, "owner_signature": self.signer.sign(signing.canonical(envelope))}
        return seal(signing.canonical(signed), registration.hotkey)

    def consume(self, commitment: Commitment, registration: Registration) -> Submission:
        """Spend the shot, permanently. Called when the ready signal is seen finalized on chain.

        IDEMPOTENT FOR THE IDENTICAL COMMITMENT, and that is not a nicety: the validator re-reads
        chain commitments every window (§8, step 3), so it will see this exact one again in every
        window the hotkey remains registered. A `consume` that raised on the second sighting would
        stop the validator on its own success. A DIFFERENT signal from a spent hotkey is refused —
        that is the second submission §7 forbids, and it is also the swap this rule exists to catch:
        re-signalling with a new manifest after the first was accepted.

        THE OPERATOR'S OBLIGATION, WHICH THIS MODULE CANNOT DISCHARGE: revoke the R2 credential for
        every generation up to `Submission.generations` (`mailbox_key`) at this point. Without it
        the miner keeps write access to a prefix whose contents are now committed on chain, and
        could replace the tree after the signal. The cryptographic backstop is independent and
        does not depend on revocation succeeding: `store.fetch_submission` re-hashes every file
        against this commitment's `manifest_sha256` at duel time, so a swap is refused even if the
        token outlived it.
        """
        with self._transaction():
            return self._consume(commitment, registration)

    def _consume(self, commitment: Commitment, registration: Registration) -> Submission:
        """`consume`'s body, run with the ledger freshly re-read under the lock (`_transaction`).

        The re-read is what makes `generations` right: it is copied from the issued record, and that
        record is the one an owner tool in another process may have just advanced.
        """
        signal = commitment.ready
        if commitment.hotkey != registration.hotkey:
            raise AccessError("ready signal was committed by a hotkey that does not own it")
        if signal.registration_id != registration.registration_id:
            raise AccessError("ready signal names a different registration than the signalling "
                              "hotkey")
        existing = self._spent.get(commitment.hotkey)
        if existing is not None:
            if (existing["manifest_sha256"] == signal.manifest_sha256
                    and existing["registration_id"] == signal.registration_id
                    and existing["block"] == commitment.block):
                return Submission(**existing)
            raise AccessError(
                f"{commitment.hotkey}: hotkey submission eligibility is permanently consumed")
        record = {"hotkey": commitment.hotkey, "registration_id": signal.registration_id,
                  "manifest_sha256": signal.manifest_sha256, "block": commitment.block,
                  "generations": int(self._issued.get(commitment.hotkey, {}).get("generation", 0))}
        self._spent[commitment.hotkey] = record
        self._persist()
        return Submission(**record)

    def _restore(self) -> None:
        if not self.path.exists():
            return
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self._issued = dict(state.get("issued", {}))
        self._spent = dict(state.get("spent", {}))

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": PROTOCOL_VERSION, "issued": self._issued,
                                         "spent": self._spent}, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


# --- the miner's side ----------------------------------------------------------------------------


def open_credential(ciphertext: bytes, seed: bytes, *, owner_public_hex: str,
                    registration: Registration, generation: int,
                    prefix: str | None = None) -> dict[str, Any]:
    """Decrypt, verify the owner's signature, and check every field the miner is entitled to check.

    Ported from `teutonic/miner/get_upload_auth.py::validate_envelope`, and the checks are the point
    rather than the decryption. A miner about to spend hours and their one shot uploading needs to
    know the envelope is the owner's, is for THIS hotkey and registration, names a prefix derived
    from chain data rather than one somebody chose, and has not already expired. Each of those is a
    way an upload can be lost or misdirected, and every one is cheaper to catch here than after
    72 GB.

    The signature is verified over the envelope WITHOUT `owner_signature`, which is how it was
    signed. Reconstructing it by deletion — rather than re-listing the fields — means a field added
    later is covered automatically instead of silently leaving the signed set.

    `prefix` is the submission prefix unless the caller is opening a KEY-ONLY credential
    (`Mailbox.issue_key`), whose envelope names `registration.key_prefix`; passing the wrong one
    refuses, which is what keeps the two credentials from being mistaken for each other.
    """
    envelope = json.loads(unseal(ciphertext, seed))
    if not isinstance(envelope, dict) or "owner_signature" not in envelope:
        raise AccessError("mailbox envelope is not a signed object")
    unsigned = {field: value for field, value in envelope.items() if field != "owner_signature"}
    if not signing.verify(owner_public_hex, signing.canonical(unsigned),
                          envelope["owner_signature"]):
        raise AccessError("mailbox envelope is not signed by the owner")
    expected = {
        "protocol_version": PROTOCOL_VERSION, "signature_scheme": "ed25519",
        "owner_identity": owner_public_hex, "netuid": registration.netuid,
        "uid": registration.uid, "hotkey": registration.hotkey,
        "registration_id": registration.registration_id,
        "registration_block": registration.registration_block,
        "credential_generation": generation,
        "allowed_prefix": registration.prefix if prefix is None else prefix,
        "credential_scope": CREDENTIAL_SCOPE, "revocation_event": REVOCATION_EVENT,
        "submission_policy": SUBMISSION_POLICY,
    }
    for field, value in expected.items():
        if envelope.get(field) != value:
            raise AccessError(f"mailbox envelope carries an unexpected {field}: "
                              f"{envelope.get(field)!r} != {value!r}")
    for field in ("r2_endpoint", "bucket", "access_key_id", "secret_access_key", "session_token"):
        if not isinstance(envelope.get(field), str) or not envelope[field]:
            raise AccessError(f"mailbox envelope is missing {field}")
    expires_at = datetime.fromisoformat(str(envelope.get("expires_at", "")).replace("Z", "+00:00"))
    if expires_at <= datetime.now(timezone.utc):
        raise AccessError("mailbox credential has already expired; ask for a rotation")
    return envelope


def credential_from_envelope(envelope: Mapping[str, Any]) -> Credential:
    """The five upload fields as a `Credential`, once `open_credential` approved the envelope."""
    return Credential(
        endpoint=str(envelope["r2_endpoint"]), bucket=str(envelope["bucket"]),
        access_key_id=str(envelope["access_key_id"]),
        secret_access_key=str(envelope["secret_access_key"]),
        session_token=str(envelope["session_token"]),
        expires_at=datetime.fromisoformat(str(envelope["expires_at"]).replace("Z", "+00:00")))


__all__ = [
    "AccessError", "CREDENTIAL_SCOPE", "Credential", "KEY_PREFIX", "MAX_CREDENTIAL_TTL_SECONDS",
    "Mailbox", "Outstanding", "REVOCATION_EVENT", "Registration", "SUBMISSION_POLICY", "Submission",
    "credential_from_envelope", "decode_ss58", "encode_ss58", "key_mailbox_key", "mailbox_key",
    "open_credential", "scoped_credential", "seal", "unseal", "verify_hotkey",
]
