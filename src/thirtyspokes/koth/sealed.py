"""Sealed request channel — HPKE to a key that exists only inside an attested enclave.

This is the primitive the confidential serving design rests on (docs/CONFIDENTIAL_SERVING.md §4).
The user seals a request to the enclave's public key; the miner's host relays a ciphertext it
cannot read; the enclave opens it. Encryption alone is not the interesting part — the interesting
part is WHY a user should believe the key belongs to an approved enclave rather than to the miner,
and that is `report_data` binding (§4.2):

    report_data = sha512( "thirtyspokes-enclave-key/1" ‖ pk ‖ image_measurement ‖ epoch )[:64]

The enclave asks the TDX quoting enclave for a quote over exactly those bytes. A user verifies the
quote with the existing `koth.tdx.verify_quote_full` (chain to the pinned Intel root, TCB status,
CRL, QE identity), checks the measurements against the owner's on-chain approved set, and recomputes
this hash from the public key it was handed. Only then does it seal. So the guarantee is not "we
promise to be an enclave" — it is "the hardware says this key was generated inside an image whose
measurements the owner published, and no other image can decrypt what you send".

THREE PROPERTIES WORTH KNOWING BEFORE READING THE CODE.

The key is EPHEMERAL and never touches disk. A reboot mints a new one. That is forward secrecy for
free, and it means there is no persistent key material for a host with physical access to hunt for.
The cost is that every reboot invalidates cached quotes, so clients must re-fetch — which is correct
anyway, since a reboot could be an image change.

The IMAGE MEASUREMENT IS BOUND INTO THE AEAD, not merely checked alongside it. It travels in HPKE's
`info`, so a ciphertext sealed for image A fails to open under image B with an `InvalidTag` rather
than decrypting into something a different image might mishandle. The quote check and the
cryptography therefore agree by construction instead of by convention.

PADDING IS A SECURITY CONTROL HERE, NOT HYGIENE. Validators verify miners by injecting probes that
must be indistinguishable from user traffic (§6.1). A miner that can separate probes from users by
ciphertext length can serve probes well and users badly, and the whole verification loop becomes
voluntary. Bucketing lengths is what denies that signal, so `seal` pads by default and refusing to
is an explicit argument.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from cryptography.hazmat.primitives import hpke
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

BINDING_LABEL = b"thirtyspokes-enclave-key/1"
INFO_LABEL = b"thirtyspokes-sealed-request/1"
REPLY_LABEL = b"thirtyspokes-sealed-response/1"

# Length buckets, in bytes. A sealed payload is padded up to the next bucket, so an observer learns
# only which bucket a request fell into. The ladder is coarse at the bottom (most chat requests are
# small, and a fine ladder there would leak) and doubles thereafter.
BUCKETS: tuple[int, ...] = (1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13,
                            1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18)


@dataclass(frozen=True)
class SuiteSpec:
    """A named HPKE ciphersuite plus how to make a private key for it.

    Named rather than hardcoded because the post-quantum question is a real one for this product:
    a request sealed today is confidential today, but an adversary that records ciphertext can
    revisit it once quantum hardware exists. Measured on this build — X25519 adds 67 bytes of
    overhead, ML-KEM768 adds 1110. Against a chat request of a few kilobytes the PQ tax is small,
    so `pq` is a supported deployment choice rather than a research note. The default stays
    classical for interoperability and size; switch when the threat model warrants it.
    """

    name: str
    kem: object
    kdf: object
    aead: object
    generate: object

    def suite(self) -> hpke.Suite:
        return hpke.Suite(self.kem, self.kdf, self.aead)


def _mlkem768_generate():
    from cryptography.hazmat.primitives.asymmetric import mlkem
    return mlkem.MLKEM768PrivateKey.generate()


SUITES: dict[str, SuiteSpec] = {
    "x25519": SuiteSpec("x25519", hpke.KEM.X25519, hpke.KDF.HKDF_SHA256,
                        hpke.AEAD.CHACHA20_POLY1305, X25519PrivateKey.generate),
    "pq": SuiteSpec("pq", hpke.KEM.MLKEM768, hpke.KDF.HKDF_SHA256,
                    hpke.AEAD.CHACHA20_POLY1305, _mlkem768_generate),
}
DEFAULT_SUITE = "x25519"


def binding(public_bytes: bytes, image_measurement: str, epoch: int) -> bytes:
    """The 64 bytes that go into the TDX quote's `report_data`.

    Covers the key, the image it was generated inside, and the epoch. The image measurement is in
    here so a user cannot be handed a key from an approved-but-different image; the epoch is here
    so an old quote cannot be replayed for a key that has since rotated.
    """
    h = hashlib.sha512()
    h.update(BINDING_LABEL)
    h.update(struct.pack(">I", len(public_bytes)))
    h.update(public_bytes)
    h.update(image_measurement.encode())
    h.update(struct.pack(">Q", int(epoch)))
    return h.digest()[:64]


def info_for(image_measurement: str, epoch: int) -> bytes:
    """HPKE `info` — the context bound into the AEAD (see the module docstring)."""
    return b"|".join([INFO_LABEL, image_measurement.encode(), str(int(epoch)).encode()])


def reply_info(image_measurement: str, epoch: int) -> bytes:
    """Context for the RESPONSE leg. A distinct label from the request leg so a ciphertext from
    one direction can never be replayed as the other."""
    return b"|".join([REPLY_LABEL, image_measurement.encode(), str(int(epoch)).encode()])


def bucket_for(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return ((n + BUCKETS[-1] - 1) // BUCKETS[-1]) * BUCKETS[-1]


def pad(plaintext: bytes) -> bytes:
    """Length-prefix then zero-pad to a bucket. The prefix is INSIDE the sealed envelope, so the
    padding is invisible to anyone who cannot decrypt — which is the entire point."""
    body = struct.pack(">I", len(plaintext)) + plaintext
    return body.ljust(bucket_for(len(body)), b"\x00")


def unpad(padded: bytes) -> bytes:
    if len(padded) < 4:
        raise ValueError("padded payload is too short to carry a length prefix")
    (n,) = struct.unpack(">I", padded[:4])
    if n > len(padded) - 4:
        raise ValueError(f"length prefix {n} exceeds the payload")
    return padded[4:4 + n]


@dataclass
class EnclaveKey:
    """An ephemeral keypair generated inside the enclave. Never serialised to disk.

    `private_bytes` is deliberately not implemented. There is no legitimate reason for this process
    to write the private key anywhere, and the absence of the method is the cheapest way to keep a
    future change from quietly adding one.
    """

    spec: SuiteSpec
    _sk: object
    image_measurement: str
    epoch: int

    @classmethod
    def generate(cls, image_measurement: str, epoch: int,
                 suite: str = DEFAULT_SUITE) -> "EnclaveKey":
        spec = SUITES[suite]
        return cls(spec=spec, _sk=spec.generate(), image_measurement=image_measurement,
                   epoch=int(epoch))

    @property
    def public_bytes(self) -> bytes:
        return self._sk.public_key().public_bytes_raw()

    def report_data(self) -> bytes:
        """Hand this to `koth.tdx.get_quote` so the hardware signs over the key."""
        return binding(self.public_bytes, self.image_measurement, self.epoch)

    def announcement(self) -> dict:
        """What the enclave publishes so a client can verify before sealing. The quote is attached
        by the caller — this carries only what the quote must be checked AGAINST."""
        return {"v": 1, "suite": self.spec.name, "public_key": self.public_bytes.hex(),
                "image_measurement": self.image_measurement, "epoch": self.epoch}

    def open(self, ciphertext: bytes) -> bytes:
        info = info_for(self.image_measurement, self.epoch)
        return unpad(self.spec.suite().decrypt(ciphertext, self._sk, info=info))


def public_key_from(announcement: dict):
    """Rebuild a public key from an announcement. Client-side; never trusts it on its own — the
    caller must first verify the quote over `binding(...)` of these same fields."""
    spec = SUITES[announcement["suite"]]
    raw = bytes.fromhex(announcement["public_key"])
    if spec.name == "x25519":
        return X25519PublicKey.from_public_bytes(raw)
    from cryptography.hazmat.primitives.asymmetric import mlkem
    return mlkem.MLKEM768PublicKey.from_public_bytes(raw)


def seal(announcement: dict, plaintext: bytes, *, padded: bool = True) -> bytes:
    """Seal a request to an enclave. `padded=False` exists for tests and for callers that have
    already bucketed; it should never be used on live traffic (see the module docstring)."""
    spec = SUITES[announcement["suite"]]
    info = info_for(announcement["image_measurement"], announcement["epoch"])
    body = pad(plaintext) if padded else plaintext
    return spec.suite().encrypt(body, public_key_from(announcement), info=info)


@dataclass
class ReplyKey:
    """A CLIENT-side ephemeral keypair that the enclave seals the response to.

    Without this the response leg has nowhere to go. Sealing a reply to the enclave's own public
    key — the obvious mistake, and one that unit tests miss because a test can open it with the
    enclave key that a real client does not possess — produces a response only the enclave can
    read. The client therefore mints a throwaway keypair per request, ships the public half inside
    the sealed request, and keeps the private half locally for exactly one response.

    Ephemeral per request, so a compromise of one reply key exposes one answer.
    """

    spec: SuiteSpec
    _sk: object

    @classmethod
    def generate(cls, suite: str = DEFAULT_SUITE) -> "ReplyKey":
        spec = SUITES[suite]
        return cls(spec=spec, _sk=spec.generate())

    def public(self) -> dict:
        return {"suite": self.spec.name,
                "public_key": self._sk.public_key().public_bytes_raw().hex()}

    def open(self, ciphertext: bytes, image_measurement: str, epoch: int) -> bytes:
        return unpad(self.spec.suite().decrypt(
            ciphertext, self._sk, info=reply_info(image_measurement, epoch)))


def seal_reply(reply_to: dict, plaintext: bytes, image_measurement: str, epoch: int,
               *, padded: bool = True) -> bytes:
    """Enclave-side: seal the response to the client's ephemeral key."""
    spec = SUITES[reply_to["suite"]]
    body = pad(plaintext) if padded else plaintext
    return spec.suite().encrypt(body, public_key_from(reply_to),
                                info=reply_info(image_measurement, epoch))


def verify_announcement(announcement: dict, quote_report_data: bytes) -> bool:
    """Does the quote actually commit to this key, image and epoch?

    The one check a client must not skip. Everything else in this module is confidentiality; this
    is the part that decides whether confidentiality is worth anything.
    """
    expected = binding(bytes.fromhex(announcement["public_key"]),
                       announcement["image_measurement"], int(announcement["epoch"]))
    return len(quote_report_data) >= 64 and quote_report_data[:64] == expected
