"""The miner's submission tool — `orchestra-miner` (§7 steps 2-5, the build plan M6).

Every piece this drives already existed and none of them were joined: `access` seals and opens the
envelope, `store` hashes and uploads the tree, `chain` writes the ready signal, `admission` says
whether the tree would be accepted. What was missing was the one program that runs them in the right
order — so a miner's only route to a submission was to write it themselves, against five modules,
with one shot and no second try. That is the "Kind" property failing at the last step.

THE ORDER IS THE DESIGN, and each step is placed where it is because of what a failure there costs:

1. **Admission first, before anything touches the network.** It is local, it is the validator's own
   `admit`, and it is the difference between learning that a tree is the wrong architecture now and
   learning it after hours of upload and a spent shot.
2. **The chain is read before the mailbox.** A hotkey that has already committed has spent its shot
   (§7), and the refusal should cost a chain read rather than a 72 GB hash.
3. **The manifest is built, then the tree is uploaded, then the signal is committed.** The upload
   writes `manifest.json` LAST as its own commit marker (`store.upload_tree`), and the on-chain
   signal names that manifest's digest — so committing before the upload drained would point the
   validator at a tree that is not there yet.

WHY THE HOTKEY MUST BE ED25519, stated here because it is the one requirement a miner cannot
discover from the mechanism. The mailbox envelope is sealed to the hotkey's own public key
(`access.seal`), which needs the ed25519 -> X25519 conversion; a Bittensor wallet created with the
default sr25519 scheme cannot open it. `hotkey_seed` refuses such a wallet by verifying that the
seed it extracted re-derives the address it claims, rather than assuming a key layout and producing
an envelope nobody can open.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .access import (
    AccessError,
    Credential,
    Registration,
    credential_from_envelope,
    encode_ss58,
    key_mailbox_key,
    mailbox_key,
    open_credential,
)
from .admission import describe
from .chain import Chain, ReadySignal
from .config import PUBLIC_STORE
from .devkit import check_admission
from .funding import KEY_NAME, seal_key
from .store import S3Bucket, build_manifest, upload_tree

FETCH_TIMEOUT_SECONDS = 60.0


class OpenBucket(Protocol):
    """`store.r2_bucket`, as a seam. Tests pass a fake and never import boto3."""

    def __call__(self, credential: Credential) -> S3Bucket: ...


class MinerError(Exception):
    """A refusal the miner can act on, raised before anything irreversible has happened."""


@dataclass(frozen=True)
class Submitted:
    """What a completed submission put on chain, printed so the miner can check it themselves."""

    hotkey: str
    registration_id: str
    manifest_sha256: str
    files: int
    bytes_uploaded: int
    skipped: int
    key_registered: bool = True

    def __str__(self) -> str:
        funding = ("Keep the OpenRouter account behind your registered key funded"
                   if self.key_registered else
                   "NO OPENROUTER KEY IS REGISTERED under this submission — run `register-key` "
                   "while your credential is valid, or your entry is deferred as unfunded")
        return (f"submitted {self.files} files ({self.bytes_uploaded:,} bytes uploaded, "
                f"{self.skipped} already present)\n"
                f"  registration  {self.registration_id}\n"
                f"  manifest      {self.manifest_sha256}\n"
                f"  committed by  {self.hotkey}\n"
                f"THE SHOT IS NOW SPENT. {funding} (§4, docs/MINER.md §6).")


@dataclass(frozen=True)
class Registered:
    """What `register-key` put under the prefix, printed so the miner can check it themselves."""

    hotkey: str
    registration_id: str
    cap_usd: float
    key_suffix: str

    def __str__(self) -> str:
        return (f"registered an OpenRouter key ending …{self.key_suffix} for {self.hotkey}\n"
                f"  registration  {self.registration_id}\n"
                f"  object        submissions/{self.registration_id}/{KEY_NAME}\n"
                f"  cap           ${self.cap_usd:.2f} per window, bounded by what the key can "
                f"still spend\n"
                f"The key is sealed to the owner's mailbox key and signed by your hotkey; re-run "
                f"this command to rotate the key or move the cap (docs/MINER.md §6).")


def registration_of(chain: Chain, netuid: int, hotkey: str) -> Registration:
    """The miner's own derivation of the identity the owner derived independently (§7).

    Both sides compute this from the same four chain facts and neither chooses anything, which is
    what makes the prefix a token can be scoped to rather than a name somebody picked. A mismatch
    here against the envelope is caught by `open_credential`, which compares every field.
    """
    neuron = chain.metagraph().resolve(hotkey)
    if neuron is None:
        raise MinerError(f"{hotkey} holds no uid on netuid {netuid}. Register the hotkey first; "
                         f"a submission identity is derived from the registration.")
    if neuron.registered_at is None:
        raise MinerError(f"{hotkey} holds uid {neuron.uid} but the chain reports no registration "
                         f"block. Retry once it does — the block is part of the identity.")
    return Registration(netuid=netuid, uid=neuron.uid, hotkey=hotkey,
                        registration_block=neuron.registered_at)


def already_committed(chain: Chain, hotkey: str) -> ReadySignal | None:
    """The shot, as the chain reports it. `None` means unspent.

    Read from the CHAIN rather than from any local note, because the chain is what the validator
    reads: a miner whose local state was lost still cannot submit twice, and a miner who thinks they
    committed can check without asking the owner.
    """
    return next((c.ready for c in chain.commitments() if c.hotkey == hotkey), None)


def hotkey_seed(wallet) -> bytes:
    """The 32-byte ed25519 seed behind a wallet's hotkey, from the KEYFILE, verified before use.

    NOT `wallet.hotkey.private_key`. Measured 2026-09-03 against the pinned `bittensor_wallet`: its
    `Keypair` exposes `sign`, `verify`, `public_key` and `ss58_address` and NO private material at
    all — no `private_key`, no `seed_hex`, no `secret_key`. An earlier version of this function read
    that attribute and would have died with `AttributeError` on the first real submission, which is
    the one moment a miner cannot afford a surprise. The seed lives in the keyfile, which is where
    `Keypair.create_from_seed` originally got it.

    STILL VERIFIED RATHER THAN TRUSTED. Whatever field the keyfile yields is used to re-derive an
    ss58 address and compared against the wallet's own; a mismatch is refused. Guessing wrong here
    would not fail loudly — it would derive a different X25519 key, and the miner would simply be
    unable to open an envelope that was correctly sealed to them, with nothing to point at.

    An encrypted keyfile is named as such rather than reported as a missing field: it is an ordinary
    state with an obvious remedy, and a miner reading "no usable seed" would go looking for a bug.
    """
    import json

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keyfile = wallet.hotkey_file
    if not keyfile.exists_on_device():
        # Named rather than surfaced as a decode error: the keyfile API hands back None for a
        # missing file, and "cannot convert NoneType to bytes" sends a miner hunting a bug
        # when the file simply is not on this machine yet.
        raise MinerError(
            f"no hotkey keyfile at {keyfile.path}. Create it with `btcli wallet new-hotkey "
            f"-w <wallet> -H <hotkey> --crypto-type ed25519`, or copy the hotkey you registered "
            f"to this machine.")
    if keyfile.is_encrypted():
        raise MinerError(
            f"the hotkey keyfile at {keyfile.path} is encrypted. Decrypt it in place first — "
            f"`python -c \"from bittensor_wallet import Wallet; Wallet(name=..., hotkey=...)"
            f".hotkey_file.decrypt()\"` prompts for the password (btcli's `wallet unlock` is for "
            f"the coldkey) — "
            f"the mailbox envelope is sealed to this hotkey and cannot be opened without its seed.")
    try:
        payload = json.loads(bytes(keyfile.data).decode())
    except Exception as exc:                # noqa: BLE001 — any unreadable keyfile is the same fix
        raise MinerError(f"cannot read the hotkey keyfile at {keyfile.path}: {exc}")

    candidates = [payload.get(field) for field in ("secretSeed", "privateKey")]
    for value in [c for c in candidates if isinstance(c, str) and c]:
        raw = bytes.fromhex(value.removeprefix("0x"))
        for candidate in ({raw[:32], raw[-32:]} if len(raw) >= 32 else set()):
            derived = Ed25519PrivateKey.from_private_bytes(candidate).public_key().public_bytes_raw()
            if encode_ss58(derived) == wallet.hotkey.ss58_address:
                return candidate
    raise MinerError(
        f"{wallet.hotkey.ss58_address} is not an ed25519 hotkey, or its keyfile holds no seed this "
        f"can use. The mailbox envelope is sealed to the hotkey's own key and the seal needs "
        f"ed25519, so an sr25519 wallet — btcli's default — cannot open it. Create the hotkey with "
        f"`btcli wallet new-hotkey -w <wallet> -H <hotkey> --crypto-type ed25519` and register "
        f"that one.")


# The public store answers 403 to Python's default `User-Agent` and 200 to any other on the same
# object — on the bucket's r2.dev address AND on `store.thirtyspokes.ai` (measured 2026-09-08:
# `Python-urllib/3.12` refused, `thirtyspokes-miner` served), so it is the zone's bot protection
# and the poll names itself. A miner reading "403 Forbidden" on a public mailbox would go looking
# for a permission the bucket does not have.
USER_AGENT = "thirtyspokes-miner"


def fetch_envelope(mailbox_url: str, key: str) -> bytes:  # pragma: no cover — network
    """GET one sealed envelope from the public mailbox. No credentials, by design.

    The mailbox is public (`access.py`: *"that bucket is public and needs no protection, because the
    envelope is sealed to the hotkey and signed by the owner"*), which is what breaks the otherwise
    circular requirement that a miner hold a credential in order to fetch their first credential.
    """
    url = f"{mailbox_url.rstrip('/')}/{key}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return response.read()
    except Exception as exc:                # noqa: BLE001 — every failure here means "not yet"
        raise MinerError(
            f"no envelope at {url}: {exc}. The owner publishes it after `orchestra-owner issue` "
            f"— send them this hotkey and wait, or ask for a rotation if a previous one expired.")


def preflight(model: Path, reference: Path) -> None:
    """The validator's own admission gate, run locally, refusing in its exact words.

    `check_admission` returns the refusal VERBATIM (its docstring explains why it must), so what a
    miner greps for here is the string the validator would have produced. This is the one check that
    has to happen before the upload rather than after it.
    """
    refusal = check_admission(model, describe(reference))
    if refusal is not None:
        raise MinerError(f"this tree would be REFUSED at admission, and uploading it would spend "
                         f"the shot for nothing:\n  {refusal}")


def submit(chain: Chain, *, netuid: int, hotkey: str, model: Path, reference: Path,
           mailbox_url: str, owner_public_hex: str, seed: bytes, sign: Callable[[bytes], str],
           open_bucket: OpenBucket, generation: int = 1,
           fetch: Callable[[str, str], bytes] = fetch_envelope) -> Submitted:
    """The whole of §7 steps 2-5, in the order the module docstring argues for."""
    registration = registration_of(chain, netuid, hotkey)
    preflight(model, reference)

    spent = already_committed(chain, hotkey)
    if spent is not None:
        raise MinerError(
            f"{hotkey} has already committed manifest {spent.manifest_sha256}. A hotkey gets one "
            f"submission ever (§7); this one is spent and a second commit cannot replace it.")

    envelope = open_credential(fetch(mailbox_url, mailbox_key(registration.registration_id,
                                                             generation)),
                              seed, owner_public_hex=owner_public_hex,
                              registration=registration, generation=generation)
    credential = credential_from_envelope(envelope)

    manifest = build_manifest(model, registration, sign)
    bucket = open_bucket(credential)
    report = upload_tree(model, bucket, registration.prefix, manifest)

    # LAST, and only now: the upload has drained and `manifest.json` is in place, so the digest this
    # commits to names a tree the validator can actually fetch (§8, step 3).
    chain.commit_ready(hotkey, ReadySignal(registration_id=registration.registration_id,
                                           manifest_sha256=manifest.sha256))
    # Not a gate — a key can be registered after the commit while the credential lives — but the
    # one omission that defers an otherwise valid entry, said here rather than in a later reveal.
    registered = registration.prefix + KEY_NAME in bucket.list(registration.prefix)
    return Submitted(hotkey=hotkey, registration_id=registration.registration_id,
                     manifest_sha256=manifest.sha256, files=len(manifest.files),
                     bytes_uploaded=report.bytes_uploaded, skipped=len(report.skipped),
                     key_registered=registered)


def register_key(chain: Chain, *, netuid: int, hotkey: str, api_key: str, cap_usd: float,
                 mailbox_url: str, owner_public_hex: str, seed: bytes,
                 sign: Callable[[bytes], str], open_bucket: OpenBucket, generation: int = 1,
                 key_credential: bool = False,
                 fetch: Callable[[str, str], bytes] = fetch_envelope) -> Registered:
    """§4 (D18): seal the miner's own OpenRouter key to the owner and put it beside the tree.

    Uses the same prefix-scoped credential `submit` opens, so it needs no second grant from the
    owner and can run before or after the commit for as long as that credential is valid. Once it
    has expired and the shot is spent, `key_credential` opens instead a KEY-ONLY credential the
    owner issued with `orchestra-owner issue --key-only` — scoped to the `openrouter/` sub-prefix,
    so it can rotate the key and touch nothing of the tree. Nothing here is irreversible:
    re-running it overwrites the record, which is how a key is rotated or a cap moved. The key
    itself never leaves this machine in the clear — it is sealed to the owner's key before the
    bucket sees it, and the owner's validator is the only thing that opens it.
    """
    registration = registration_of(chain, netuid, hotkey)
    if key_credential:
        envelope = open_credential(
            fetch(mailbox_url, key_mailbox_key(registration.registration_id, generation)), seed,
            owner_public_hex=owner_public_hex, registration=registration, generation=generation,
            prefix=registration.key_prefix)
    else:
        envelope = open_credential(
            fetch(mailbox_url, mailbox_key(registration.registration_id, generation)), seed,
            owner_public_hex=owner_public_hex, registration=registration, generation=generation)
    record = seal_key(api_key, registration=registration, cap_usd=cap_usd,
                      owner_public_hex=owner_public_hex, sign=sign)
    open_bucket(credential_from_envelope(envelope)).put(
        registration.prefix + KEY_NAME, record.to_bytes(), content_type="application/json")
    return Registered(hotkey=hotkey, registration_id=registration.registration_id,
                      cap_usd=float(cap_usd), key_suffix=api_key[-4:])


def read_api_key(path: Path | None) -> str:
    """The key from a file the miner names, else from `OPENROUTER_API_KEY` — never from argv,
    which shell history and process listings keep."""
    import os  # noqa: PLC0415

    if path is not None:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise MinerError(f"cannot read the key file {path}: {exc}") from exc
    else:
        value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise MinerError("no OpenRouter key: pass --key-file, or set OPENROUTER_API_KEY")
    return value


def format_identity(registration: Registration, *, generations: int = 3) -> str:
    """What a miner sends the owner to ask for a credential, and can check an envelope against."""
    keys = "\n".join(f"    generation {n}: {mailbox_key(registration.registration_id, n)}"
                     for n in range(1, generations + 1))
    return (f"hotkey             {registration.hotkey}\n"
            f"uid                {registration.uid}\n"
            f"registration block {registration.registration_block}\n"
            f"registration id    {registration.registration_id}\n"
            f"upload prefix      {registration.prefix}\n"
            f"  mailbox keys to poll:\n{keys}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestra-miner",
        description="Submit a Conductor: check it, upload it, commit it (v3 §7).")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet", required=True, help="your wallet name")
    parser.add_argument("--hotkey", default="default", help="your hotkey name within that wallet")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hotkey", help="verify the local hotkey is ed25519 and readable BEFORE paying "
                                  "to register it; touches nothing but the keyfile")
    sub.add_parser("identity", help="the registration facts to send the owner; touches no network "
                                    "but the chain")

    check = sub.add_parser("check", help="run the validator's admission gate on a local tree")
    check.add_argument("--model", type=Path, required=True, metavar="DIR")
    check.add_argument("--reference", type=Path, required=True, metavar="DIR",
                       help="the pinned reference tree the architecture is checked against")

    send = sub.add_parser("submit", help="check, upload and commit — this SPENDS the shot")
    send.add_argument("--model", type=Path, required=True, metavar="DIR")
    send.add_argument("--reference", type=Path, required=True, metavar="DIR")
    send.add_argument("--mailbox-url", default=PUBLIC_STORE, metavar="URL",
                      help=f"the public store the owner publishes your envelope to "
                           f"(default: {PUBLIC_STORE}; a rehearsal passes its own)")
    send.add_argument("--owner-key", required=True, metavar="HEX",
                      help="the owner's ed25519 public key, hex — an envelope not signed by it is "
                           "refused rather than opened")
    send.add_argument("--generation", type=int, default=1,
                      help="which credential generation to open; raise it after a rotation")

    key = sub.add_parser("register-key",
                         help="seal your own OpenRouter key to the owner and put it beside your "
                              "submission — your arm runs on it (§4); re-run to rotate")
    key.add_argument("--key-file", type=Path, default=None, metavar="FILE",
                     help="a file holding the key; else OPENROUTER_API_KEY from the environment")
    key.add_argument("--cap-usd", type=float, required=True,
                     help="the most one window may spend on this key (§6)")
    key.add_argument("--mailbox-url", default=PUBLIC_STORE, metavar="URL",
                     help=f"the public store (default: {PUBLIC_STORE})")
    key.add_argument("--owner-key", required=True, metavar="HEX",
                     help="the owner's ed25519 public key, hex — the key is sealed to it")
    key.add_argument("--generation", type=int, default=1)
    key.add_argument("--key-credential", action="store_true",
                     help="open a KEY-ONLY credential the owner issued with `issue --key-only` "
                          "(its own generation series) — for rotating a key after the submission "
                          "credential expired and the shot is spent")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            preflight(args.model, args.reference)
            print("admitted: this tree passes the checks the validator runs (§1.1)")
            return

        from bittensor_wallet import Wallet

        if args.command == "hotkey":
            # `submit` refuses a sr25519 wallet, but only AFTER the miner has paid to register
            # it (docs/MINER.md §1a). This is the same check, run before the burn.
            wallet = Wallet(name=args.wallet, hotkey=args.hotkey)
            hotkey_seed(wallet)
            print(f"hotkey             {wallet.hotkey.ss58_address}\n"
                  f"scheme             ed25519 — the mailbox envelope can be sealed to it\n"
                  f"keyfile            readable\n"
                  f"safe to register:  btcli subnets register --netuid {args.netuid} "
                  f"-n {args.network} -w {args.wallet} -H {args.hotkey}")
            return

        from .chain import BittensorChain
        from .store import r2_bucket

        chain = BittensorChain(netuid=args.netuid, wallet_name=args.wallet, network=args.network,
                               hotkey=args.hotkey)
        wallet = Wallet(name=args.wallet, hotkey=args.hotkey)
        address = wallet.hotkey.ss58_address
        if args.command == "identity":
            print(format_identity(registration_of(chain, args.netuid, address)))
            return
        if args.command == "register-key":
            print(register_key(chain, netuid=args.netuid, hotkey=address,
                               api_key=read_api_key(args.key_file), cap_usd=args.cap_usd,
                               mailbox_url=args.mailbox_url, owner_public_hex=args.owner_key,
                               seed=hotkey_seed(wallet),
                               sign=lambda data: wallet.hotkey.sign(data).hex(),
                               open_bucket=r2_bucket, generation=args.generation,
                               key_credential=args.key_credential))
            return
        print(submit(chain, netuid=args.netuid, hotkey=address, model=args.model,
                     reference=args.reference, mailbox_url=args.mailbox_url,
                     owner_public_hex=args.owner_key, seed=hotkey_seed(wallet),
                     sign=lambda data: wallet.hotkey.sign(data).hex(),
                     open_bucket=r2_bucket, generation=args.generation))
    except (MinerError, AccessError) as exc:
        sys.exit(f"orchestra-miner: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
