"""What a user is allowed to trust — the owner's approved serving measurements.

`serve/client.py` can pin an image, but pinning is only worth something if the pin comes from
somewhere the miner cannot reach. This module is that somewhere: the owner signs a record naming
the measurements of the confidential-router image, and a client verifies the signature before
believing a word of it. The transport is then irrelevant — chain, HTTP, a file, the miner's own
endpoint — because a forged record fails the signature and an altered one fails it too.

    owner ──signs──► record ──any transport, including the miner──► client ──verifies──► pins

WHY THIS IS A SEPARATE RECORD FROM `koth/owner.py` (removed with v2, 2026-09-07). That one governs the benchmark image on
mainnet, and its MRTD is the published runtime. The confidential router is a different image with a
different measurement, and folding it into the same record would mean editing a record the mainnet
validators already consume. Two services, two records, no interaction.

WHAT THE RECORD DOES NOT DO. It says which image is approved, not that the image is good. A user
trusting this record is trusting the owner's build and review, exactly as they trust any vendor.
The attestation stack narrows the question from "is this miner honest?" to "is the owner's image
honest?" — a real reduction, and not the same as eliminating trust. `docs/CONFIDENTIAL_SERVING.md`
§2 states this in the threat model rather than leaving a user to infer it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..gateway import signing

SERVICE = "confidential-router"
RECORD_VERSION = 1


@dataclass
class Approved:
    """The pins a client applies, and the provenance of the pins themselves."""

    mrtd: set[str] = field(default_factory=set)
    rtmr: dict[int, str] = field(default_factory=dict)
    tcb_accept: frozenset[str] = frozenset({"UpToDate"})
    suites: tuple[str, ...] = ()
    min_epoch: int = 0
    record: dict | None = None

    def __bool__(self) -> bool:
        return bool(self.mrtd)


def build_record(*, mrtd: list[str] | set[str], rtmr: dict[int, str] | None = None,
                 tcb_accept: tuple[str, ...] = ("UpToDate",),
                 suites: tuple[str, ...] = ("x25519", "pq"),
                 min_epoch: int = 0, signer=None) -> dict:
    """Owner-side: describe the approved serving image and sign it.

    `mrtd` is a set rather than one value so an image rotation has an overlap window — during a
    rollout both the old and new images are legitimately serving, and a client that accepted only
    one of them would reject half the fleet. Removing the old entry is what ends the window, and
    that is a deliberate owner action rather than a timeout.
    """
    if not mrtd:
        raise ValueError("an approved record with no MRTD would pin nothing")
    body = {
        "v": RECORD_VERSION,
        "service": SERVICE,
        "mrtd": sorted(mrtd),
        "rtmr": {str(k): v for k, v in sorted((rtmr or {}).items())},
        "tcb_accept": list(tcb_accept),
        "suites": list(suites),
        "min_epoch": int(min_epoch),
    }
    if signer is not None:
        body["sig"] = signer.sign(signing.canonical({k: v for k, v in body.items()}))
    return body


def load(record: dict, owner_public_hex: str | None) -> Approved:
    """Client-side: verify then interpret. Raises rather than returning an empty pin.

    A refusal must never look like "no pins configured", because an unpinned client accepts any
    TDX enclave — including the miner's own build. Every failure here is therefore an exception,
    and `Approved()` with an empty `mrtd` is falsy so a caller cannot mistake one for the other.
    """
    if not isinstance(record, dict):
        raise ValueError("approved record is not an object")
    if record.get("service") != SERVICE:
        raise ValueError(f"record is for {record.get('service')!r}, not {SERVICE!r}")
    if int(record.get("v", 0)) != RECORD_VERSION:
        raise ValueError(f"unsupported record version {record.get('v')!r}")

    if owner_public_hex:
        sig = record.get("sig")
        if not sig:
            raise ValueError("approved record is unsigned; refusing to pin from it")
        body = {k: v for k, v in record.items() if k != "sig"}
        if not signing.verify(owner_public_hex, signing.canonical(body), sig):
            raise ValueError("approved record signature does not verify against the owner key")
    elif record.get("sig"):
        # An owner key was not supplied, so the signature cannot be checked. Saying so is better
        # than silently treating a signed record as verified because it happens to carry a sig.
        raise ValueError("record is signed but no owner public key was given to check it against")

    mrtd = {m.lower() for m in record.get("mrtd") or []}
    if not mrtd:
        raise ValueError("approved record names no MRTD")
    return Approved(
        mrtd=mrtd,
        rtmr={int(k): v for k, v in (record.get("rtmr") or {}).items()},
        tcb_accept=frozenset(record.get("tcb_accept") or ("UpToDate",)),
        suites=tuple(record.get("suites") or ()),
        min_epoch=int(record.get("min_epoch") or 0),
        record=record,
    )


def load_json(text: str | bytes, owner_public_hex: str | None) -> Approved:
    return load(json.loads(text), owner_public_hex)


def main() -> None:
    """`thirtyspokes-serving-governance` — sign an approved-image record.

    The MRTD is not discovered for you. Read it from a quote taken on the built image
    (`tdx.self_mrtd()` inside it, or `parse_quote` on a captured quote) and pass it in, so that
    approving an image is an act with a value in front of you rather than a script's guess.
    """
    import argparse
    import sys
    from pathlib import Path

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from ..gateway.signing import Signer

    ap = argparse.ArgumentParser(description="sign the approved confidential-router measurements")
    ap.add_argument("--mrtd", nargs="+", required=True,
                    help="approved MRTD hex values; more than one during a rollout")
    ap.add_argument("--rtmr", nargs="*", default=[], metavar="IDX=HEX")
    ap.add_argument("--tcb-accept", nargs="+", default=["UpToDate"])
    ap.add_argument("--min-epoch", type=int, default=0,
                    help="reject enclave keys below this epoch; the revocation lever")
    ap.add_argument("--owner-key", required=True,
                    help="the owner's ed25519 private seed: 32 raw bytes or 64 hex chars")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    raw = Path(args.owner_key).read_bytes().strip()
    seed = bytes.fromhex(raw.decode()) if len(raw) == 64 else raw
    if len(seed) != 32:
        raise SystemExit(f"owner key must be a 32-byte ed25519 seed, got {len(seed)} bytes")
    signer = Signer(Ed25519PrivateKey.from_private_bytes(seed))

    rtmr = {}
    for item in args.rtmr:
        idx, _, value = item.partition("=")
        rtmr[int(idx)] = value

    record = build_record(mrtd=[m.lower() for m in args.mrtd], rtmr=rtmr,
                          tcb_accept=tuple(args.tcb_accept), min_epoch=args.min_epoch,
                          signer=signer)
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
    print(f"signed by owner key {signer.public_hex}", file=sys.stderr)
