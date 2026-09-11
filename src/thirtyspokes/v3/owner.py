"""The owner's submission desk — `thirtyspokes-owner` (§7 steps 2-3, the build plan M6).

WHY THIS IS A SEPARATE PROGRAM FROM THE VALIDATOR, which is the only real decision in the file.
`validator.main` wires its mailbox with a minter that always raises, and `never_mint`'s docstring
gives the reason: a daemon that could mint would be able to hand out write access to a submission
prefix from inside the loop that scores it. Issuing therefore had to live somewhere else — and until
this module existed it lived nowhere, so a registered miner had no way to obtain the credential §7
step 3 requires and **could not submit at all**. The mechanism was complete and the door was locked.

THE REGISTRATION IS READ FROM THE CHAIN AND NEVER TYPED BY AN OPERATOR. Everything downstream — the
R2 prefix a credential is scoped to, the mailbox key, the registration id the on-chain signal binds —
is derived from `(netuid, uid, hotkey, registration_block)` (`access.Registration`). An operator who
mistyped a uid would mint a token scoped to somebody else's prefix, which is the one failure the
derivation exists to make impossible; so the numbers come from `metagraph.resolve` and the tool takes
no way to override them.

IT SHARES THE VALIDATOR'S LEDGER FILE, DELIBERATELY AND CAREFULLY. `--state` must be the daemon's
`--state`: two ledgers would be two one-shots, and the "one submission per hotkey, ever" rule would
hold in each file separately while being false overall. That sharing is also what made
`access.Mailbox._transaction` necessary — two processes now write it, and the daemon's whole-view
`_persist` used to erase an `issue` made after it started (`test_access.py`).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..gateway.signing import Signer
from .access import AccessError, Credential, Mailbox, Registration, scoped_credential
from .chain import Chain

CREDENTIAL_TTL_SECONDS = 86_400
"""One day. Long enough for a ~72 GB upload on an ordinary connection, short enough that a leaked
token is not a standing key — and a miner who needs longer asks for a rotation, which `Mailbox.issue`
grants against the identical prefix without spending the shot."""


class Publish(Protocol):
    """`bucket.put`, narrowed to what publishing an envelope needs. A protocol rather than the
    `S3Bucket` type so the tests drive this module with a dict and no boto3."""

    def __call__(self, key: str, body: bytes) -> None: ...


@dataclass(frozen=True)
class Issued:
    """What one `issue` produced, for the operator to read back and for a test to assert on.

    NO EXPIRY FIELD, AND THAT IS NOT AN OMISSION: the envelope is sealed to the miner's hotkey, so
    the owner cannot read back the credential it just published. Reporting an expiry here would mean
    the tool recomputing its own idea of one, which is exactly the number that goes stale the moment
    the configured TTL moves. The operator is told the TTL it passed; the miner is told the truth by
    `open_credential`, which reads the field out of the envelope and refuses an expired one.
    """

    hotkey: str
    registration: Registration
    mailbox_key: str
    generation: int
    key_only: bool = False

    def __str__(self) -> str:
        kind = "key-only generation" if self.key_only else "generation"
        prefix = self.registration.key_prefix if self.key_only else self.registration.prefix
        return (f"issued {kind} {self.generation} to {self.hotkey}\n"
                f"  registration  {self.registration.registration_id}\n"
                f"  prefix        {prefix}\n"
                f"  mailbox key   {self.mailbox_key}")


def format_balances(balances: Mapping[str, float]) -> str:
    if not balances:
        return ("no allowances credited\n"
                "  every arm would run at $0 and score zero on every task, and the window would "
                "publish that as if the corpus carried no spread\n"
                "  credit the owner account for the reference arms, and each miner for theirs: "
                "thirtyspokes-owner --state DIR credit --hotkey SS58 --usd N")
    rows = "\n".join(f"  {hotkey:<52} ${usd:.4f}"
                     for hotkey, usd in sorted(balances.items(), key=lambda kv: -kv[1]))
    return f"{len(balances)} allowance(s)\n{rows}"


def mailbox_signer(state: Path) -> Signer:
    """The owner's mailbox identity, persisted across runs — a per-run key would sign nothing.

    `signing.Signer()` GENERATES A NEW KEYPAIR when constructed with no argument, so a tool that
    built one per invocation would sign every envelope with a different key. The miner verifies the
    envelope against a public key the owner published once (`--owner-key`), so that key has to
    outlive the process that issued the credential. Without this, `open_credential`'s *"not signed by
    the owner"* refusal fires on the owner's own envelopes — and the check that stops a miner being
    pointed at somebody else's bucket degrades into a failure nobody can tell from an attack.

    Created `0600` on first use and never rotated automatically: a silent rotation would invalidate
    every credential a miner is polling for at that moment, and each of those cost a rotation to
    obtain.
    """
    path = state / "mailbox-key.hex"
    if path.exists():
        return Signer(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(
            path.read_text(encoding="utf-8").strip())))
    state.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    # CREATED 0600, NOT CHMODDED AFTERWARDS. `write_text` then `chmod` leaves the key readable at
    # the process umask for the window between them, and an interruption in that window leaves it
    # world-readable permanently. `os.open` with the mode applies it at creation, so the key is
    # never on disk under weaker permissions than it ends with.
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        sink.write(private.private_bytes_raw().hex())
    path.chmod(0o600)      # idempotent, and corrects a file that already existed more permissively
    return Signer(private)


def mailbox_seed(state: Path) -> bytes:
    """The 32-byte seed behind `mailbox_signer`'s key — what opens a miner's sealed OpenRouter key
    (`funding.open_key`). The same file and the same lifetime: miners seal to the public half they
    were handed as `--owner-key`, so rotating this key would strand every key sealed to it."""
    mailbox_signer(state)
    return bytes.fromhex((state / "mailbox-key.hex").read_text(encoding="utf-8").strip())


def registration_of(chain: Chain, netuid: int, hotkey: str) -> Registration:
    """The four chain facts a submission identity is derived from, or a refusal that says which.

    REFUSES A NEURON WITH NO REGISTRATION BLOCK rather than substituting one. `chain.Neuron` keeps
    `registered_at=None` when the chain reported a uid with no `BlockAtRegistration`, and for
    *ownership* questions that is deliberately survivable — dropping such a neuron would burn a live
    pensioner's share over missing metadata. Here it is not survivable: the block is hashed into
    `registration_id`, so a guessed one derives a prefix the miner's own independent derivation will
    not agree with, and the mismatch surfaces only after a 72 GB upload lands in the wrong place.
    """
    neuron = chain.metagraph().resolve(hotkey)
    if neuron is None:
        raise AccessError(f"{hotkey} holds no uid on netuid {netuid}: it must register before a "
                          f"credential can be scoped to it")
    if neuron.registered_at is None:
        raise AccessError(
            f"{hotkey} holds uid {neuron.uid} but the chain reports no registration block for it. "
            f"The block is hashed into the registration id, so issuing against a guessed one would "
            f"scope the credential to a prefix the miner does not derive. Retry once the chain "
            f"reports BlockAtRegistration.")
    return Registration(netuid=netuid, uid=neuron.uid, hotkey=hotkey,
                        registration_block=neuron.registered_at)


def owner_mint(*, endpoint: str, bucket: str, account_id: str, access_key_id: str,
               secret_access_key: str,
               ttl_seconds: int = CREDENTIAL_TTL_SECONDS) -> Callable[[str], Credential]:
    """The `mint` seam `Mailbox` takes: a prefix in, a prefix-scoped R2 token out.

    The prefix is the ONLY argument, and it arrives from `Mailbox.issue`'s own derivation rather
    than from anything here — which is what makes "one miner cannot write into another's submission"
    a property of the token instead of a promise about the caller.
    """
    def mint(prefix: str) -> Credential:
        return scoped_credential(endpoint=endpoint, bucket=bucket, account_id=account_id,
                                 parent_access_key_id=access_key_id,
                                 parent_secret_access_key=secret_access_key,
                                 prefix=prefix, ttl_seconds=ttl_seconds)
    return mint


def issue(chain: Chain, mailbox: Mailbox, publish: Publish, *, netuid: int, hotkey: str,
          key_only: bool = False) -> Issued:
    """Mint, seal and publish one credential generation for a registered hotkey.

    ORDER: the ledger is written before the object is published, and that is the safe direction. A
    generation recorded but never published costs the miner a rotation; a generation published but
    never recorded is a live write credential the operator's revocation list does not know about
    (`Mailbox.consume`'s obligation) — the same hole `_transaction` was added to close, re-opened by
    ordering. `Mailbox.issue` has already refused a spent hotkey by the time anything is uploaded.
    """
    registration = registration_of(chain, netuid, hotkey)
    key, ciphertext = (mailbox.issue_key if key_only else mailbox.issue)(registration)
    publish(key, ciphertext)
    return Issued(hotkey=hotkey, registration=registration, mailbox_key=key,
                  generation=int(key.rsplit("/", 1)[-1].removesuffix(".bin").removeprefix("key-")),
                  key_only=key_only)


def format_status(mailbox: Mailbox) -> str:
    """The revocation list, which is the operator obligation `access.py` says it cannot discharge.

    `Mailbox.consume` requires the owner to revoke every generation up to `Submission.generations`
    once a shot is spent, because the miner otherwise keeps write access to a prefix whose contents
    are now named on chain. Nothing could answer "which generations" from outside the module, so the
    answer is printed here rather than left to an operator reading a JSON file by eye.
    """
    rows = mailbox.outstanding()
    if not rows:
        return "no credentials issued"
    lines = [f"{'hotkey':<50} {'gens':>4}  {'shot':<7} prefix"]
    for row in rows:
        state = "SPENT" if row.spent else "open"
        lines.append(f"{row.hotkey:<50} {row.generations:>4}  {state:<7} {row.prefix}")
    revoke = [row for row in rows if row.spent and row.generations]
    if revoke:
        lines.append("")
        lines.append("REVOKE NOW — these shots are spent and their write credentials outlive them:")
        lines.extend(f"  {row.hotkey}: generations 1..{row.generations} on {row.prefix}"
                     for row in revoke)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thirtyspokes-owner",
        description="Issue submission credentials and read the one-shot ledger (v3 §7).")
    parser.add_argument("--state", type=Path, required=True, metavar="DIR",
                        help="the validator's OWN --state directory: the one-shot ledger lives at "
                             "<state>/mailbox.json and a second ledger would be a second shot")
    sub = parser.add_subparsers(dest="command", required=True)

    issue_cmd = sub.add_parser("issue", help="mint and publish a credential for one hotkey")
    issue_cmd.add_argument("--hotkey", required=True, metavar="SS58",
                           help="the MINER's hotkey; its uid and registration block are read from "
                                "the chain and cannot be supplied by hand")
    issue_cmd.add_argument("--netuid", type=int, required=True)
    issue_cmd.add_argument("--network", default="finney")
    issue_cmd.add_argument("--wallet", required=True, help="the owner's wallet name")
    issue_cmd.add_argument("--owner-hotkey", default="default",
                           help="the owner's own hotkey, used to read the chain")
    issue_cmd.add_argument("--r2-endpoint", required=True)
    issue_cmd.add_argument("--r2-bucket", required=True)
    issue_cmd.add_argument("--ttl-seconds", type=int, default=CREDENTIAL_TTL_SECONDS)
    issue_cmd.add_argument("--key-only", action="store_true",
                           help="a credential for the miner's sealed OpenRouter key alone "
                                "(`openrouter/` under their prefix), issued to a SPENT hotkey too "
                                "— the rotation path (D18); it can write nothing of the tree")

    # THE MONEY PATH. `OwnerGateway` implements the whole allowance machinery — balances,
    # exhaustion, per-duel ledgers, signed receipts — and until this command existed nothing could
    # put a dollar into it. `fund()` had no caller outside the tests, so the shipped daemon ran
    # every arm at a zero balance, every episode tripped the exhaustion check at its first step, and
    # a window of all-zero scores was published as if the corpus carried no spread.
    #
    # `--ref` is free text this program never interprets. The gateway's promise is that HOW the
    # money arrives is out of its module, so the rail — an on-chain transfer, an invoice, staked
    # alpha — stays the owner's to choose and is merely recorded beside the credit.
    credit = sub.add_parser("credit",
                            help="credit a hotkey's spend allowance, durably (§4)")
    credit.add_argument("--hotkey", required=True, metavar="SS58",
                        help="the miner being credited, or the owner's own account for the "
                             "reference arms")
    credit.add_argument("--usd", type=float, required=True,
                        help="dollars to ADD; the command prints the resulting balance")
    credit.add_argument("--ref", default="",
                        help="what was settled, in your own words — never parsed, only recorded")

    sub.add_parser("balances", help="every allowance the gateway would run a window against")

    sub.add_parser("status", help="who holds credentials, and which must now be revoked")
    sub.add_parser("key", help="the owner public key miners pass as --owner-key")

    sched = sub.add_parser("commit-schedule",
                           help="publish the chained-manifest root for the WHOLE schedule (§6.3)")
    sched.add_argument("--netuid", type=int, required=True)
    sched.add_argument("--network", default="finney")
    sched.add_argument("--wallet", required=True, help="the owner's wallet name")
    sched.add_argument("--owner-hotkey", default="default")
    sched.add_argument("--world", required=True, metavar="module:attr",
                       help="the same --world the validator runs, so the root committed is the one "
                            "it will compute")
    sched.add_argument("--windows", type=int, required=True)
    sched.add_argument("--per-benchmark", type=int, required=True)
    sched.add_argument("--minimum", type=int, required=True)
    sched.add_argument("--replace-governance-record", action="store_true",
                       help="overwrite a retired mechanism's `kothgov1|…` record in the owner's slot "
                            "(the cutover's one deliberate overwrite; refused otherwise)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    signer = mailbox_signer(args.state)
    if args.command == "key":
        print(signer.public_hex)
        return
    if args.command == "status":
        print(format_status(Mailbox(args.state / "mailbox.json", signer, _no_mint)))
        return
    if args.command in ("credit", "balances"):
        from .gateway import OwnerGateway, allowance_journal

        # No provider: crediting and reading balances never calls a model, and passing None here
        # means a mistyped command cannot reach one.
        gateway = OwnerGateway(None, journal=allowance_journal(args.state))  # type: ignore[arg-type]
        if args.command == "balances":
            print(format_balances(gateway.balances()))
            return
        balance = gateway.fund(args.hotkey, args.usd, ref=args.ref)
        print(f"credited {args.hotkey} ${args.usd:.4f}\n"
              f"  balance now ${balance:.4f}\n"
              f"  recorded in {allowance_journal(args.state)} — the validator replays this file on "
              f"start, so the credit survives a restart")
        return

    from .chain import BittensorChain
    from .store import r2_bucket

    if args.command == "commit-schedule":
        from ..koth import holdout_feed
        from .devkit import _load
        from .window import schedule

        # `_load` ALREADY calls the attribute (devkit.py: `getattr(...)()`), so a second
        # `()` here calls `Pins(...)` and dies. The validator spells it `_load(args.world)`;
        # this is the same seam and must spell it the same way.
        pins = _load(args.world)
        entries = schedule(range(1, args.windows + 1), tasks=pins.world.tasks,
                           per_benchmark=args.per_benchmark, minimum=args.minimum)
        root = holdout_feed.manifest(entries)["root"]
        chain = BittensorChain(netuid=args.netuid, wallet_name=args.wallet,
                               network=args.network, hotkey=args.owner_hotkey)
        chain.commit_schedule(root, replace_governance=args.replace_governance_record)
        print(f"committed schedule root {root}\n"
              f"  under hotkey {args.owner_hotkey} — the validator reads it back under ITS\n"
              f"  --hotkey, so the two must name the same key or the gate sees nothing\n"
              f"  {args.windows} windows, {args.per_benchmark} tasks per benchmark\n"
              f"  The validator refuses to start unless the root it computes matches this one, so "
              f"re-commit deliberately if the world or the window count changes.")
        return

    mint = owner_mint(endpoint=args.r2_endpoint, bucket=args.r2_bucket,
                      account_id=os.environ["R2_ACCOUNT_ID"],
                      access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                      ttl_seconds=args.ttl_seconds)
    parent = Credential(
        endpoint=args.r2_endpoint, bucket=args.r2_bucket,
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        session_token=os.environ.get("R2_SESSION_TOKEN", ""),
        expires_at=datetime.now(timezone.utc))
    bucket = r2_bucket(parent)
    chain = BittensorChain(netuid=args.netuid, wallet_name=args.wallet, network=args.network,
                           hotkey=args.owner_hotkey)
    try:
        print(issue(chain, Mailbox(args.state / "mailbox.json", signer, mint),
                    bucket.put, netuid=args.netuid, hotkey=args.hotkey, key_only=args.key_only))
    except AccessError as exc:
        sys.exit(f"thirtyspokes-owner: {exc}")


def _no_mint(prefix: str) -> Credential:
    """`status` reads and never issues, so it is wired with a minter that cannot."""
    raise AccessError(f"status does not issue credentials (asked for {prefix})")


if __name__ == "__main__":  # pragma: no cover
    main()
