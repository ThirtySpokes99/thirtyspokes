"""The ~72 GB upload: hash it, sign it, put `manifest.json` LAST (§1, §8b.7; the build plan M6).

PORTED FROM `teutonic/miner/upload_model.py` + `teutonic/access/contracts.py` +
`teutonic/storage/artifacts.py`, which run this at production scale today. Four of their decisions
are load-bearing and none of them is obvious from the outside:

1. **`manifest.json` IS THE COMMIT MARKER, AND IT GOES LAST.** M6 exit 2, and the reason is a
   measurement about size rather than a preference about ordering: at ~72 GB over hours an
   interrupted upload is the COMMON case, not the edge case. Every partial tree therefore has to be
   unambiguously *not a submission*, and "the last object written is the one that says the tree is
   complete" is what makes that true with no extra state, no partial-upload table, and no timeout to
   tune. A tree with no `manifest.json` is a work in progress; a tree with one is a submission and
   the miner's shot is spent on it. `upload_tree` therefore writes the manifest only after every
   `executor.map` future has drained — an exception in any worker propagates before the marker
   exists.

2. **PER-FILE DIGESTS, NEVER A DIGEST OF THE WHOLE.** The reference tree is 1,045 tensors over 26
   shards and 71.9 GB (`config.REFERENCE_TENSOR_COUNT`, `REFERENCE_WEIGHTS_BYTES`). A single
   tree-wide hash would say "something differs" and nothing else — one swapped shard would be
   averaged into a wall of bytes
   and the operator would be told to re-upload 72 GB to find out which. `tree_digest` is a hash OF
   THE INVENTORY (path, size, per-file digest), so the commitment is one value while the failure is
   localised to the file that broke it, and `fetch_submission` refuses at that shard rather than at
   the end.

3. **64 MiB parts, 4 streams per file, 2 files at a time.** The part size is teutonic's shipped
   number and the arithmetic still holds: a 50 GB shard is ~800 parts, an order below S3's
   10,000-part ceiling, and 64 MiB x 10,000 puts the ceiling on a single object at 640 GB. Sizing
   parts smaller would multiply request count against a per-request round trip; larger would make a
   retried part cost more. THE CONCURRENCIES ARE NOT teutonic's 16 x 16, AND THE REASON IS MEASURED:
   on 2026-09-08 two 68 GB submissions ran 32 part streams over a ~35 MB/s WAN and every socket
   showed thousands of TCP retransmits, a third of what was sent landed as completed parts, and
   one run died on `SSL … EOF occurred in violation of protocol` after boto3's ten attempts. The
   same link at 4 streams moved a 1 GiB slice at 22 MB/s with no retransmits, and 8 streams did
   no better — the per-file read (a shard over sshfs) is the ceiling, so more streams only fight
   each other. Two files at a time keeps the total at 8 streams, which a fast link still fills.

4. **RESUME BY PER-FILE DIGEST, NOT BY PRESENCE.** A resumed upload must produce a byte-identical
   tree (M6 exit 5). Presence alone is not enough: a miner who edits one shard between attempts
   would keep the old bytes in the bucket under a manifest naming the new ones, and the mismatch
   would only surface at duel time — with the shot already spent. So a file is skipped only when the
   stored object's recorded `sha256` metadata AND size match the manifest. (Presence is a sound
   *first* filter because completing a multipart upload is atomic in S3: an interrupted upload
   leaves no key at all rather than a short object.)

WHAT THIS MODULE DELIBERATELY DOES NOT CHECK. Not the architecture — `admission.py` owns §1.1, and a
second copy of "safetensors only" here would be a second pin to drift from the first. The order at
duel time is: `fetch_submission` (are these the bytes the chain committed to?) and then
`admission.admit` (is that tree the pinned architecture?). Bytes first, because there is no point
asking what a tree is until it is known to be the tree that was submitted.

THE STORE SEAM. `S3Bucket` wraps a boto3-style S3 client rather than constructing one, so the class
that actually talks to R2 is the class under test and only the client differs — the tests drive it
through an in-memory bucket that speaks the same seven calls. `r2_bucket` is the one function here
that cannot be exercised offline; it is pure configuration and carries no logic beyond the pinned
transfer numbers.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import sys
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import access
from .access import AccessError, Registration
from .access import KEY_PREFIX
from .config import UNCLAIMED_GRACE_SECONDS
from .chain import Commitment

# The transfer profile (docstring point 3), and §11-8's "multipart upload is mandatory". The two
# concurrencies multiply — 2 files in flight x 4 part streams each is 8 sockets — and the boto3
# client below sizes `max_pool_connections` to the product so the transfer is never serialised
# behind connection-pool contention. Measured 2026-09-08: the 16 x 16 this shipped with collapsed
# a 68 GB upload on a ~35 MB/s WAN (docstring point 3).
PART_SIZE = 64 * 1024 * 1024
MULTIPART_THRESHOLD = 32 * 1024 * 1024
PART_STREAMS = 4
FILE_WORKERS = 2
# A transport failure during a multi-hour upload is retried AT THE FILE, above boto3's own
# per-request attempts: a TLS reset that outlives those ten attempts otherwise ends the run, and
# although `upload_tree` resumes, a rerun re-hashes the whole tree first (measured 2026-09-08:
# ~45 minutes over sshfs). Six attempts with a growing pause cover an outage of a few minutes.
UPLOAD_ATTEMPTS = 6
UPLOAD_BACKOFF_SECONDS = 20.0

# The commit marker (docstring point 1). Reserved: a tree may not contain a file of this name,
# because that file would be overwritten by the marker and the manifest would then describe a file
# whose bytes it does not name.
MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
"""What a miner may call their model: 3–40 chars, lowercase, no spaces.

An identifier rather than a title, because it is published beside a king and read back by tooling.
Lowercase and a fixed alphabet so two names cannot differ by something a reader cannot see, and the
`thirtyspokes-` prefix is REFUSED below: the subnet names the genesis king `thirtyspokes-genesis`
(`config.KING_ZERO_NAME`), and a submission that could take that name could claim to be it.
"""

RESERVED_NAME_PREFIX = "thirtyspokes-"

MANIFEST_NAME = "manifest.json"
# Written BESIDE a fetched tree's directory (`<dest>.verified`) once every file has been hashed
# against the committed manifest (`fetch_submission`); holds that manifest's digest. Beside, not
# inside: `admission.admit` walks the tree and a file it does not expect is a refusal.
VERIFIED_MARKER = ".verified"
# Where a winning model is published in the public models bucket: content-addressed by the manifest
# digest the chain committed to, so a published king is immutable by name and a repeated promotion
# lands on the same keys.
PUBLIC_MODEL_ROOT = "models/sha256/"

# §8b.7's "published grace window" for a losing submission, DERIVED rather than chosen. The binding
# floor is the queue: `MAX_QUEUE_DEPTH = 2 x MAX_DUELS_PER_WINDOW` means the back of the queue is
# reached within two windows, and §8b.1 asks the owner to cover three (one to wait for the next
# window to open, two to drain). Deleting a tree still waiting in that queue would spend a shot on
# nothing — the exact outcome §8b.1's cap exists to prevent — so the floor is three windows, and 14
# days sits an order of magnitude above it at a 24-hour window while keeping storage bounded
# (§8b.7: 70 TB at a thousand entrants). RE-DERIVE IT IF A WINDOW EVER EXCEEDS ~4.5 DAYS.
GRACE_SECONDS = 14 * 86_400

_HEX_DIGEST_LENGTH = 64


class StoreError(Exception):
    """A tree that is not a submission, or not the submission that was committed to.

    Never repaired and never partially accepted: a hotkey has one shot (§7), so a store that
    accepted a tree it could not fully account for would score a model other than the one the miner
    uploaded and spend their whole entry on it.
    """


# --- the manifest --------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ManifestFile:
    """One file's identity. `size` alongside `sha256` because the size is what a bucket listing
    gives for free, so resume can filter on it before spending a HEAD per object."""

    path: str
    size: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> ManifestFile:
        if not isinstance(value, Mapping) or set(value) != {"path", "size", "sha256"}:
            raise StoreError("a manifest file entry is exactly path, size and sha256")
        path, size, digest = value["path"], value["size"], value["sha256"]
        if not isinstance(path, str) or not path or path == MANIFEST_NAME:
            raise StoreError(f"manifest file path is invalid: {path!r}")
        # An absolute path or a `..` component in a manifest is a write outside the destination
        # directory when the validator materialises the tree — a miner's file landing in the
        # validator's filesystem, on the single machine that holds the subnet's scoring authority.
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise StoreError(f"unsafe manifest path: {path!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StoreError(f"manifest size is invalid for {path!r}: {size!r}")
        if not isinstance(digest, str) or not _is_digest(digest):
            raise StoreError(f"manifest sha256 is invalid for {path!r}: {digest!r}")
        return cls(path=path, size=size, sha256=digest)

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class Manifest:
    """The signed inventory. Its own SHA-256 is what goes on chain (`access.ready_signal`).

    Signed BY THE HOTKEY, not by the owner: the manifest is the miner's statement about what they
    uploaded, and the on-chain signal is the same hotkey committing to it. Without the signature the
    tree would be authenticated only by who held a write credential, and a credential that leaked —
    or an owner-side bug that scoped one too widely — would let a third party publish a submission
    in a miner's name and spend their shot.
    """

    registration_id: str
    hotkey: str
    files: tuple[ManifestFile, ...]
    tree_digest: str
    signature: str
    protocol_version: int = 1
    signature_scheme: str = "ed25519"
    # What the miner calls this model (protocol 2). None for a protocol 1 manifest, whose signing
    # payload must stay byte-identical or every submission made before names existed stops verifying.
    model_name: str | None = None

    def signing_payload(self) -> bytes:
        """The bytes the hotkey signs: everything except the signature, canonically ordered.

        THE NAME IS SIGNED, and therefore covered by the digest the chain commits to. A name outside
        the signature would be a label anyone holding the prefix could rewrite after the commit —
        the one field of a submission its own miner did not vouch for.
        """
        payload: dict[str, Any] = {
            "files": [item.as_dict() for item in self.files], "hotkey": self.hotkey,
            "protocol_version": self.protocol_version, "registration_id": self.registration_id,
            "signature_scheme": self.signature_scheme, "tree_digest": self.tree_digest,
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def as_bytes(self) -> bytes:
        return json.dumps({**json.loads(self.signing_payload()), "signature": self.signature},
                          sort_keys=True, separators=(",", ":")).encode()

    @property
    def sha256(self) -> str:
        """What the chain commits to. Over `as_bytes`, so it covers the signature too — a commitment
        that named an unsigned manifest would be satisfied by an unsigned one."""
        return hashlib.sha256(self.as_bytes()).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    @classmethod
    def from_bytes(cls, raw: bytes) -> Manifest:
        """Parse and CHECK — every field, plus the inventory against its own digest.

        Re-deriving `tree_digest` from the file list here rather than trusting the field is what
        makes the on-chain commitment reach the individual shards: the chain names the manifest, the
        manifest names the digest, and the digest is a pure function of the inventory. Trusting the
        stored value would leave a manifest whose digest and file list disagree.
        """
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise StoreError(f"manifest is not valid JSON: {exc}") from exc
        required = {"protocol_version", "signature_scheme", "registration_id", "hotkey", "files",
                    "tree_digest", "signature"}
        version = value.get("protocol_version") if isinstance(value, dict) else None
        # Protocol 2 adds ONE field, and only the field it adds: a manifest that carries a name must
        # say it is a 2, and one that says it is a 1 must not carry one. Otherwise the same bytes
        # could be read as either version, and "which payload did the hotkey sign" stops having an
        # answer — the signature covers the version too.
        if version == 2:
            required = required | {"model_name"}
        if not isinstance(value, dict) or set(value) != required:
            seen = value if isinstance(value, dict) else {}
            missing = sorted(required.symmetric_difference(seen))
            raise StoreError(f"manifest fields differ from the v{version if version in (1, 2) else 1} "
                             f"contract: {missing}")
        if version not in (1, 2) or value["signature_scheme"] != "ed25519":
            raise StoreError("manifest requires protocol version 1 or 2 with ed25519")
        name = value.get("model_name")
        if version == 2:
            check_model_name(name)
        registration = value["registration_id"]
        if not isinstance(registration, str) or not _is_digest(registration):
            raise StoreError("manifest registration_id is not a SHA-256 digest")
        if not isinstance(value["hotkey"], str) or not value["hotkey"]:
            raise StoreError("manifest hotkey is required")
        if not isinstance(value["files"], list):
            raise StoreError("manifest files must be an array")
        files = tuple(sorted(ManifestFile.from_mapping(item) for item in value["files"]))
        if not files or len({item.path for item in files}) != len(files):
            raise StoreError("manifest must carry a non-empty inventory of unique paths")
        if not isinstance(value["signature"], str) or not value["signature"]:
            raise StoreError("manifest signature is required")
        manifest = cls(registration_id=value["registration_id"], hotkey=value["hotkey"],
                       files=files, tree_digest=value["tree_digest"],
                       signature=value["signature"], protocol_version=version,
                       model_name=None if version == 1 else str(name))
        if tree_digest(files) != manifest.tree_digest:
            raise StoreError("manifest tree_digest does not match its own file inventory")
        return manifest


def tree_digest(files: Iterable[ManifestFile]) -> str:
    """One digest over the whole inventory — path, size and per-file digest, in path order.

    Each field is fixed-length or delimited (`\\0` after the path and after the size) so that no two
    different inventories can produce the same byte stream: without the delimiters, moving a
    character from the end of one path to the start of the next size would hash identically.
    """
    entries = sorted(files)
    if not entries:
        raise StoreError("a model tree with no files cannot be a submission")
    digest = hashlib.sha256()
    seen: set[str] = set()
    for item in entries:
        if item.path in seen:
            raise StoreError(f"duplicate inventory path: {item.path}")
        seen.add(item.path)
        digest.update(item.path.encode())
        digest.update(b"\0")
        digest.update(str(item.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(item.sha256))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """8 MiB at a time: a 2.8 GB shard read whole would be 2.8 GB resident, and 16 of them at once
    on the validator is the machine."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[ManifestFile, ...]:
    """Hash every file in a tree, 16 at a time. Refuses symlinks and the reserved manifest name.

    Symlinks are refused rather than followed because a symlink is a file whose CONTENT is decided
    by whoever resolves it: hashed on the miner's machine it names their bytes, materialised on the
    validator it names the validator's. Hashing is `hashlib`, which releases the GIL, so the thread
    pool is real parallelism on a 72 GB tree rather than decoration.
    """
    if not root.is_dir():
        raise StoreError(f"model directory does not exist: {root}")
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise StoreError(f"model directory contains a symlink: {relative}")
        if not path.is_file():
            continue
        if relative == MANIFEST_NAME:
            raise StoreError(f"model directory contains the reserved file {MANIFEST_NAME}; it is "
                             f"written by the upload as the commit marker")
        paths.append(path)
    if not paths:
        raise StoreError(f"model directory contains no files: {root}")

    def inspect(path: Path) -> ManifestFile:
        return ManifestFile(path=path.relative_to(root).as_posix(), size=path.stat().st_size,
                            sha256=sha256_file(path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
        return tuple(sorted(executor.map(inspect, paths)))


def check_model_name(name: object) -> str:
    """A miner's name for their model, or a refusal saying exactly what is wrong with it."""
    if not isinstance(name, str) or not MODEL_NAME.fullmatch(name):
        raise StoreError(f"model name {name!r} must be 3-40 characters of lowercase letters, "
                         f"digits, dot, dash or underscore, starting with a letter or digit")
    if name.startswith(RESERVED_NAME_PREFIX):
        raise StoreError(f"model name {name!r} is reserved: {RESERVED_NAME_PREFIX}* names belong to "
                         f"the subnet itself, and one of them is the genesis king")
    return name


def build_manifest(root: Path, registration: Registration, sign: Callable[[bytes], str],
                   *, model_name: str | None = None) -> Manifest:
    """Hash and sign a local tree. `sign` is the miner's hotkey signer, `(bytes) -> signature`."""
    files = inventory(root)
    unsigned = Manifest(registration_id=registration.registration_id, hotkey=registration.hotkey,
                        files=files, tree_digest=tree_digest(files), signature="unsigned",
                        protocol_version=1 if model_name is None else 2,
                        model_name=None if model_name is None else check_model_name(model_name))
    return replace(unsigned, signature=sign(unsigned.signing_payload()))


# --- the bucket ----------------------------------------------------------------------------------


class S3Bucket:
    """One bucket, through a boto3-style S3 client. Seven calls, and no state of its own.

    `list` FOLLOWS THE CONTINUATION TOKEN, which is the one thing about this API that fails
    silently: `list_objects_v2` truncates at 1000 keys and reports it only in `IsTruncated`. A
    single submission is ~30 objects so no duel would ever notice, but `usage` lists the whole
    submissions prefix, and at a hundred entrants (§8b.7's own figure) an unpaginated listing would
    under-report storage by however much sat past the first page — an accounting bug that grows with
    the thing it is accounting for.
    """

    def __init__(self, client: Any, bucket: str, transfer: Any = None) -> None:
        self.client = client
        self.bucket = bucket
        # boto3's managed-multipart settings travel WITH the client rather than being rebuilt per
        # call, because they are a property of the connection this bucket was opened on: `r2_bucket`
        # sizes the client's connection pool to the same two concurrencies, and a transfer config
        # that disagreed with the pool would serialise the upload behind it.
        self.transfer = transfer

    def put(self, key: str, body: bytes, *, digest: str | None = None,
            content_type: str | None = None) -> None:
        extra: dict[str, Any] = {}
        if digest is not None:
            extra["Metadata"] = {"sha256": digest}
        if content_type is not None:
            extra["ContentType"] = content_type
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, **extra)

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def head(self, key: str) -> dict[str, Any]:
        """Only ever called on a key a listing just returned, so a raised exception here is a real
        failure rather than a 404 — which is why there is no `None` branch to mistake for one."""
        return self.client.head_object(Bucket=self.bucket, Key=key)

    def list(self, prefix: str) -> dict[str, int]:
        found: dict[str, int] = {}
        token: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                request["ContinuationToken"] = token
            page = self.client.list_objects_v2(**request)
            for item in page.get("Contents", ()):
                found[item["Key"]] = int(item["Size"])
            if not page.get("IsTruncated"):
                return found
            token = page["NextContinuationToken"]

    def listing(self, prefix: str) -> dict[str, tuple[int, float]]:
        """Every object under `prefix`, with its size and when it was last written.

        `list` answers resume's question — is this object here, at this size — and callers that only
        need that keep using it. Retention asks a different one: how long has nobody touched this,
        which is the only thing that tells an abandoned upload from one still arriving.
        """
        found: dict[str, tuple[int, float]] = {}
        token: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                request["ContinuationToken"] = token
            page = self.client.list_objects_v2(**request)
            for item in page.get("Contents", ()):
                modified = item.get("LastModified")
                found[item["Key"]] = (int(item["Size"]),
                                      modified.timestamp() if hasattr(modified, "timestamp")
                                      else float(modified or 0.0))
            if not page.get("IsTruncated"):
                return found
            token = page["NextContinuationToken"]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def upload_file(self, path: Path, key: str, *, digest: str) -> None:
        self.client.upload_file(str(path), self.bucket, key,
                                ExtraArgs={"Metadata": {"sha256": digest}}, Config=self.transfer)

    def download_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path), Config=self.transfer)


def r2_bucket(credential: access.Credential) -> S3Bucket:
    """The real R2 bucket from a mailbox credential. THE ONE FUNCTION HERE WITH NO OFFLINE TEST.

    It carries no logic — every decision in it is one of the constants above — and is deliberately
    kept that way, because the alternative to "untested configuration" is "untested configuration
    plus untested behaviour". `region_name="auto"` is R2's requirement; the connection pool is sized
    to the two concurrencies (2 files x 4 part streams) so the transfer is not serialised behind
    it; and boto3's default 8 MiB chunk is replaced by the pinned 64 MiB, which over a 71.9 GB tree
    is ~1,100 part requests instead of ~8,600; and the body is sent unsigned, because a signed body
    is read twice (once to hash, once to send) and the read is the ceiling.

    The three imports are function-local: `boto3` is only in the optional `store` extra, and a
    module-scope import would make `v3.store` unimportable without it — including for `admission`,
    the dev kit and every offline test, none of which ever reach R2.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    client = boto3.client(
        "s3", endpoint_url=credential.endpoint, aws_access_key_id=credential.access_key_id,
        aws_secret_access_key=credential.secret_access_key,
        aws_session_token=credential.session_token, region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"},
                      max_pool_connections=FILE_WORKERS * PART_STREAMS,
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required",
                      # Without a checksum header botocore hashes every part body for the SigV4
                      # signature, i.e. reads each part TWICE — measured 2026-09-08 as 32 MB/s of
                      # sshfs reads for 14 MB/s of completed parts. TLS carries integrity on the
                      # wire, the per-file digest travels in the metadata, and the validator
                      # re-hashes every shard at fetch, so the body is sent unsigned.
                      s3={"payload_signing_enabled": False}))
    return S3Bucket(client, credential.bucket, TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD, multipart_chunksize=PART_SIZE,
        max_concurrency=PART_STREAMS, num_download_attempts=10, use_threads=True))


# --- upload ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadReport:
    """What an upload did — including what it did NOT do, the interesting half on a resume."""

    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]
    bytes_uploaded: int


def _transient(exc: BaseException) -> bool:
    """Whether an upload failure is worth another attempt.

    Transport errors of every layer are: botocore's `ConnectionError` family (which is where the
    TLS EOF surfaces), urllib3's protocol errors, and the `OSError`s under them — except the
    local-file ones, which describe the tree rather than the link. An S3 error is transient only
    when the service says so (throttling, a timeout, a 5xx). A refused credential — 403, expired —
    is none of these and propagates at once: waiting would not make it valid. Classified by class
    NAME so this module stays importable without boto3 (the offline suite, the dev kit).
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "ClientError" in names:
        response = getattr(exc, "response", None) or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0) or 0)
        return status >= 500 or code in {"SlowDown", "RequestTimeout", "InternalError",
                                          "ServiceUnavailable", "Throttling"}
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError)):
        return False
    return isinstance(exc, OSError) or bool(names & {
        "ConnectionError", "SSLError", "ReadTimeoutError", "ConnectTimeoutError",
        "EndpointConnectionError", "ConnectionClosedError", "HTTPClientError", "ProtocolError"})


def upload_tree(root: Path, bucket: S3Bucket, prefix: str, manifest: Manifest) -> UploadReport:
    """Upload the tree, then `manifest.json` LAST. Resumable, and idempotent on a complete tree.

    Refuses a prefix that already carries a manifest. That refusal is a courtesy rather than a
    defence — the miner's client is the miner's, and the server-side stop is credential revocation
    at the ready signal plus the duel-time re-hash (`fetch_submission`) — but it is the courtesy
    that matters most here: a second `upload_tree` into a committed prefix would replace the bytes
    the chain names, and a miner who did it by accident would fail their own submission.
    """
    listing = bucket.list(prefix)
    if prefix + MANIFEST_NAME in listing:
        raise StoreError(f"{prefix} already carries a {MANIFEST_NAME}: this submission is "
                         f"committed and its bytes are what the chain names")

    def stored_digest(key: str) -> str | None:
        return (bucket.head(key).get("Metadata") or {}).get("sha256")

    pending, skipped = [], []
    for item in manifest.files:
        key = prefix + item.path
        # Presence and size come free with the listing; the digest costs one HEAD and is what makes
        # the resume byte-exact (docstring point 4). Checked in that order so the HEAD is spent only
        # on files that could plausibly be skipped.
        if listing.get(key) == item.size and stored_digest(key) == item.sha256:
            skipped.append(item.path)
        else:
            pending.append(item)

    def send(item: ManifestFile) -> None:
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                bucket.upload_file(root / item.path, prefix + item.path, digest=item.sha256)
                return
            except Exception as exc:        # noqa: BLE001 — classified by `_transient`
                if attempt == UPLOAD_ATTEMPTS or not _transient(exc):
                    raise
                delay = UPLOAD_BACKOFF_SECONDS * attempt
                print(f"upload of {item.path} failed (attempt {attempt}/{UPLOAD_ATTEMPTS}): "
                      f"{type(exc).__name__}: {str(exc)[:160]} — retrying in {delay:.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(delay)

    with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
        # `list()` re-raises the first worker exception, and the `with` block then waits for the
        # rest — so a failed shard means the marker below is never written and the tree stays
        # unambiguously not-a-submission (docstring point 1).
        list(executor.map(send, pending))

    bucket.put(prefix + MANIFEST_NAME, manifest.as_bytes(), digest=manifest.sha256,
               content_type="application/json")
    return UploadReport(uploaded=tuple(item.path for item in pending), skipped=tuple(skipped),
                        bytes_uploaded=sum(item.size for item in pending))


# --- download and verification, at duel time -----------------------------------------------------


def fetch_manifest(bucket: S3Bucket, prefix: str) -> Manifest:
    """The commit marker, parsed. A prefix without one is NOT A SUBMISSION (M6 exit 2).

    The absence is read off a LISTING rather than off a failed GET, deliberately. A GET that threw
    would make "the object is not there" and "the owner's credentials expired" the same exception,
    and treating the second as the first would refuse a valid submission for an owner-side outage —
    spending a hotkey's one shot on the owner's own downtime.
    """
    key = prefix + MANIFEST_NAME
    if key not in bucket.list(prefix):
        raise StoreError(f"{prefix} has no {MANIFEST_NAME}: an interrupted upload is not a "
                         f"submission")
    return Manifest.from_bytes(bucket.get(key))


def fetch_submission(bucket: S3Bucket, dest: Path, *, commitment: Commitment,
                     registration: Registration) -> Manifest:
    """Pull the tree the chain committed to, and refuse anything else. M6 exit 4.

    ORDER IS THE DESIGN. The manifest is fetched and checked against the on-chain commitment, the
    hotkey's signature and the registration FIRST, and only then are 72 GB pulled: a tree whose
    manifest does not match the commitment is refused for the price of one small object rather than
    an hour of transfer. Then every file is hashed AS IT LANDS, so a swapped shard is refused at
    that shard.

    Three refusals, each closing a different substitution:

    * the manifest's own SHA-256 must equal the one in the ready signal — otherwise the miner
      committed to one tree and served another;
    * the manifest must be signed by the signalling hotkey and name its registration — otherwise a
      manifest lifted from another miner's prefix would satisfy the digest check;
    * every object under the prefix must appear in the manifest, and every manifest file must
      appear under the prefix. An UNLISTED extra is refused rather than ignored, because "ignored"
      means the validator materialises a file the commitment does not cover. The one named
      exception is the `access.KEY_PREFIX` sub-prefix, which holds the miner's sealed OpenRouter
      key, is read by the validator's funding step and never materialised.
    """
    signal = commitment.ready
    manifest = fetch_manifest(bucket, registration.prefix)
    if manifest.sha256 != signal.manifest_sha256:
        raise StoreError(f"manifest digest {manifest.sha256} is not the committed "
                         f"{signal.manifest_sha256}")
    if manifest.hotkey != commitment.hotkey or manifest.registration_id != signal.registration_id:
        raise StoreError(f"manifest belongs to {manifest.hotkey}/{manifest.registration_id}, the "
                         f"commitment to {commitment.hotkey}/{signal.registration_id}")
    try:
        access.verify_hotkey(manifest.hotkey, manifest.signing_payload(), manifest.signature)
    except AccessError as exc:
        raise StoreError(f"manifest is not signed by {manifest.hotkey}: {exc}") from exc

    prefix = registration.prefix
    listed = {key[len(prefix):] for key in bucket.list(prefix) if key.startswith(prefix)}
    # `KEY_PREFIX` is the ONE place beside `manifest.json` the manifest need not name: the miner's
    # sealed OpenRouter key lives there (`funding.py`), is not part of the tree, is never
    # materialised here, and is meant to be replaced after the commit — so it cannot be in a digest
    # the commitment pins, and a key-only credential scoped to it can touch nothing this checks.
    # `openrouter.json` is where the record lived on 2026-09-08 (the testnet rehearsal registered
    # three); tolerated so a daemon on this code can still admit those submissions, read nowhere.
    extra = sorted(name for name in listed - {item.path for item in manifest.files}
                   if name not in (MANIFEST_NAME, "openrouter.json")
                   and not name.startswith(KEY_PREFIX))
    if extra:
        raise StoreError(f"{prefix} carries files the manifest does not name: {extra}")

    dest.mkdir(parents=True, exist_ok=True)
    marker = dest.parent / f"{dest.name}{VERIFIED_MARKER}"    # beside the tree, never inside it
    if _verified(marker, manifest, dest):
        return manifest
    for item in manifest.files:
        target = dest / item.path
        # A file already here at its committed size and digest is kept, not re-pulled: the
        # rehearsal of 2026-09-08 found this function re-downloading every ~70 GB tree on EVERY
        # window — a 3-hour window could not finish its challenger phase — where the design assumes
        # a tree is pulled once and resident (§8b.1, §11-7).
        if not (target.is_file() and target.stat().st_size == item.size
                and sha256_file(target) == item.sha256):
            bucket.download_file(prefix + item.path, target)
            observed = sha256_file(target)
            if observed != item.sha256:
                raise StoreError(f"{item.path} hashes to {observed}, the committed manifest "
                                 f"says {item.sha256}")
            if target.stat().st_size != item.size:
                raise StoreError(f"{item.path} is {target.stat().st_size} bytes, the committed "
                                 f"manifest says {item.size}")
    marker.write_text(manifest.sha256, encoding="utf-8")
    return manifest


def _verified(marker: Path, manifest: Manifest, dest: Path) -> bool:
    """A tree this validator already fetched and hashed against THIS manifest, still whole.

    The marker names the manifest digest the tree was verified against and every listed file is
    still present at its committed size; then the hash pass is not repeated. The commitment is one
    shot, so the digest cannot move under the marker, and the disk is the validator's own (§8b.7):
    what this trusts is the validator's earlier verification, not the miner. A tree with no marker,
    a marker for another digest, or a missing or resized file takes the full path above.
    """
    try:
        if marker.read_text(encoding="utf-8").strip() != manifest.sha256:
            return False
    except OSError:
        return False
    return all((dest / item.path).is_file() and (dest / item.path).stat().st_size == item.size
               for item in manifest.files)


# --- promotion: a winner's tree, published (D14) -------------------------------------------------


def public_model_prefix(manifest_sha256: str) -> str:
    """Where a promoted model lives in the public models bucket."""
    if not _is_digest(manifest_sha256):
        raise StoreError(f"{manifest_sha256!r} is not a SHA-256 digest")
    return f"{PUBLIC_MODEL_ROOT}{manifest_sha256}/"


def promote_submission(tree: Path, public: S3Bucket, *, manifest: Manifest) -> str:
    """Publish a winner's tree to the public models bucket, prove it arrived, return its prefix.

    FROM THE VALIDATOR'S OWN DISK, NEVER FROM THE PRIVATE BUCKET. `tree` is the directory
    `fetch_submission` hashed every byte of on arrival — the copy the king is served from — so what
    is published is what was judged, whatever the private bucket holds by now. A server-side copy
    could not promise that on R2: `UploadPartCopy`, which every multi-GB shard needs, does not
    honour `x-amz-copy-source-if-match`, so the copy could not be tied to the judged version.
    Teutonic's promotion is host-routed too, and refuses a server-side copy outright.

    Three refusals, and none of them repairs anything:

    * `tree` must carry this validator's verification marker for THIS manifest, every file at its
      committed size (`_verified` — the same trust the resident-tree path already places in this
      disk);
    * the destination may hold nothing the manifest does not name, and nothing at a size or digest
      the manifest does not give — a content-addressed prefix holding other bytes is a collision;
    * after `upload_tree`, which resumes past files already in place and writes `manifest.json`
      LAST, the prefix must hold exactly the committed tree, and its `manifest.json` must be the
      committed manifest byte for byte, so it hashes to the digest the chain names.
    """
    marker = tree.parent / f"{tree.name}{VERIFIED_MARKER}"
    if not _verified(marker, manifest, tree):
        raise StoreError(f"{tree} is not a tree this validator verified against manifest "
                         f"{manifest.sha256}; only verified bytes are published")
    destination = public_model_prefix(manifest.sha256)
    expected = {item.path: item for item in manifest.files}

    def in_place(listing: Mapping[str, int]) -> set[str]:
        """The committed paths already at the destination; raises on anything else there."""
        present = set()
        for key, size in listing.items():
            path = key[len(destination):]
            if path == MANIFEST_NAME:
                continue
            item = expected.get(path)
            if item is None:
                raise StoreError(f"{destination} holds {path}, which manifest {manifest.sha256} "
                                 f"does not name")
            digest = (public.head(key).get("Metadata") or {}).get("sha256")
            if size != item.size or digest != item.sha256:
                raise StoreError(f"{destination}{path} is {size} bytes at digest {digest}; the "
                                 f"manifest says {item.size} at {item.sha256}")
            present.add(path)
        return present

    listing = public.list(destination)
    in_place(listing)
    if destination + MANIFEST_NAME not in listing:
        upload_tree(tree, public, destination, manifest)
    arrived = public.list(destination)
    if in_place(arrived) != set(expected) or destination + MANIFEST_NAME not in arrived:
        raise StoreError(f"{destination} does not hold the whole committed tree after promotion")
    if public.get(destination + MANIFEST_NAME) != manifest.as_bytes():
        raise StoreError(f"{destination}{MANIFEST_NAME} is not the committed manifest")
    return destination


# --- storage accounting and retention (§8b.7) ----------------------------------------------------


def usage(bucket: S3Bucket, prefix: str = "submissions/") -> dict[str, int]:
    """Bytes held per submission prefix — M6 exit 6, and the input to retention.

    §8b.7's arithmetic is the reason this exists at all: ~72 GB per hotkey is 7 TB at a hundred
    entrants and 70 TB at a thousand, and "unbounded retention is not a plan". A number nobody
    measures is a number nobody acts on.
    """
    totals: dict[str, int] = {}
    for key, size in bucket.list(prefix).items():
        rest = key[len(prefix):]
        if "/" not in rest:
            continue
        submission = prefix + rest.split("/", 1)[0] + "/"
        totals[submission] = totals.get(submission, 0) + size
    return totals


@dataclass(frozen=True)
class Retention:
    """Who is kept, who is inside the grace window, and whose weights go."""

    kept: tuple[str, ...]
    within_grace: tuple[str, ...]
    expired: tuple[str, ...]


def retention_plan(submissions: Mapping[str, float], *, protected: Collection[str], now: float,
                   grace_seconds: float = GRACE_SECONDS) -> Retention:
    """Decide what §8b.7 keeps: `{prefix: uploaded_at}` in, three disjoint lists out.

    `protected` is the reigning king's prefix and the five pensioners' — kept INDEFINITELY, because
    D14 makes the king's weights public and §5.7 pays the pension for having been the platform
    others build on. A pensioner whose weights were deleted would be a public artifact that is not
    there, and the derivative culture D14 chose depends on it being there.

    Everything else is kept for the grace window and then deleted — WEIGHTS ONLY. `apply_retention`
    leaves `manifest.json` behind, so the record of every submission ever made is permanent even
    when its bytes are not, which is the second half of §8b.7 and the half that keeps history
    auditable at ~1 KB per entrant instead of 72 GB.

    A pure function returning a plan rather than issuing deletes, because deleting 72 GB is
    irreversible and the owner should be able to read what is about to happen first.
    """
    kept, grace, expired = [], [], []
    for prefix, uploaded_at in sorted(submissions.items()):
        if prefix in protected:
            kept.append(prefix)
        elif now - uploaded_at < grace_seconds:
            grace.append(prefix)
        else:
            expired.append(prefix)
    return Retention(kept=tuple(kept), within_grace=tuple(grace), expired=tuple(expired))


def apply_retention(bucket: S3Bucket, plan: Retention) -> int:
    """Delete the expired weights, keep every `manifest.json`. Returns the bytes freed."""
    freed = 0
    for prefix in plan.expired:
        for key, size in sorted(bucket.list(prefix).items()):
            if key == prefix + MANIFEST_NAME:
                continue
            bucket.delete(key)
            freed += size
    return freed


def unclaimed_plan(bucket: S3Bucket, *, committed: Collection[str], now: float,
                   grace_seconds: float = UNCLAIMED_GRACE_SECONDS,
                   prefix: str = "submissions/") -> tuple[str, ...]:
    """Prefixes holding bytes no ready signal ever named, untouched for the grace window.

    KEYED OFF THE OBJECTS, NOT THE CREDENTIAL LEDGER. An upload still arriving keeps moving its own
    newest timestamp, so a 70 GB transfer spread over two days is never swept mid-flight, while a
    prefix abandoned after one shard is. A ledger-based rule would have to guess at both.

    `committed` is the registration ids the chain names. A prefix among them is out of scope here
    whatever its age: it is a submission, and §8b.7's grace applies to it once it has been judged.
    """
    newest: dict[str, float] = {}
    for key, (_size, modified) in bucket.listing(prefix).items():
        rest = key[len(prefix):]
        if "/" not in rest:
            continue
        registration = rest.split("/", 1)[0]
        newest[registration] = max(newest.get(registration, 0.0), modified)
    return tuple(sorted(f"{prefix}{registration}/"
                        for registration, seen in newest.items()
                        if registration not in committed and now - seen >= grace_seconds))


def delete_prefix(bucket: S3Bucket, prefix: str) -> int:
    """Delete everything under one prefix. Returns the bytes freed.

    Unlike `apply_retention` this keeps no `manifest.json`: that record exists so a JUDGED
    submission stays auditable forever, and nothing here was ever judged — there is no verdict for a
    manifest to be the evidence of.
    """
    freed = 0
    for key, size in sorted(bucket.list(prefix).items()):
        bucket.delete(key)
        freed += size
    return freed


def _is_digest(value: str) -> bool:
    return (len(value) == _HEX_DIGEST_LENGTH
            and all(char in "0123456789abcdef" for char in value))


__all__ = [
    "FILE_WORKERS", "GRACE_SECONDS", "MANIFEST_NAME", "MULTIPART_THRESHOLD", "Manifest",
    "UPLOAD_ATTEMPTS", "UPLOAD_BACKOFF_SECONDS",
    "ManifestFile", "PART_SIZE", "PART_STREAMS", "Retention", "S3Bucket", "StoreError",
    "UploadReport", "apply_retention", "build_manifest", "fetch_manifest", "fetch_submission",
    "PUBLIC_MODEL_ROOT", "promote_submission", "public_model_prefix",
    "MODEL_NAME", "RESERVED_NAME_PREFIX", "check_model_name",
    "UNCLAIMED_GRACE_SECONDS", "delete_prefix", "unclaimed_plan",
    "inventory", "r2_bucket", "retention_plan", "sha256_file", "tree_digest", "upload_tree",
    "usage",
]
