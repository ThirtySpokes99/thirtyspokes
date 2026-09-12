"""The 72 GB upload's properties (docs/WHITEPAPER.md §1, §8b.7; the build plan M6).

NO REAL R2, AND NONE IS REACHABLE FROM HERE. `FakeS3` is an in-memory bucket that speaks the seven
boto3 calls `S3Bucket` makes, so the class under test is the production one and only the client
differs. It is hand-rolled rather than `moto` because neither `moto` nor `boto3` is installed in
this repository's offline test environment, and a test that skips when a dependency is missing is
a test that stops running exactly where it is needed most — and this is the module that spends a
miner's one shot.

`FakeS3` deliberately reproduces three real behaviours that a naive stub would smooth over, because
each one has a bug behind it:

* **`list_objects_v2` truncates**, here at three keys instead of a thousand, so an implementation
  that ignores `NextContinuationToken` under-reports rather than passing;
* **completing a multipart upload is atomic** — `fail_after` interrupts an upload by leaving the
  remaining keys ABSENT, which is what an interrupted 72 GB transfer really looks like, rather than
  by leaving a short object;
* **object metadata round-trips**, since resume reads the digest it wrote (M6 exit 5).
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from thirtyspokes.v3.access import Registration, encode_ss58
from thirtyspokes.v3.chain import Commitment, ReadySignal
import thirtyspokes.v3.store as store_module
from thirtyspokes.v3.store import (
    FILE_WORKERS,
    MANIFEST_NAME,
    MULTIPART_THRESHOLD,
    PART_SIZE,
    PART_STREAMS,
    PUBLIC_MODEL_ROOT,
    Manifest,
    ManifestFile,
    S3Bucket,
    StoreError,
    apply_retention,
    build_manifest,
    fetch_manifest,
    fetch_submission,
    inventory,
    promote_submission,
    public_model_prefix,
    retention_plan,
    tree_digest,
    upload_tree,
    usage,
)

MINER_SEED = bytes.fromhex("11" * 32)
OTHER_SEED = bytes.fromhex("22" * 32)
BUCKET = "v3-submissions"
PUBLIC_BUCKET = "v3-public-models"


def address(seed: bytes) -> str:
    return encode_ss58(Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw())


MINER, OTHER = address(MINER_SEED), address(OTHER_SEED)
REGISTRATION = Registration(netuid=99, uid=7, hotkey=MINER, registration_block=5_000_000)
OTHER_REGISTRATION = Registration(netuid=99, uid=8, hotkey=OTHER, registration_block=5_000_001)


def sign_with(seed: bytes):
    key = Ed25519PrivateKey.from_private_bytes(seed)
    return lambda payload: key.sign(payload).hex()


def commitment(manifest_sha256: str, *, registration: Registration = REGISTRATION) -> Commitment:
    """The ready signal as `chain.read_commitments` hands it over — what a duel starts from."""
    return Commitment(hotkey=registration.hotkey, block=5_000_100,
                      ready=ReadySignal(registration_id=registration.registration_id,
                                        manifest_sha256=manifest_sha256))


class FakeS3:
    """An in-memory bucket speaking boto3's S3 surface. The module docstring says what it copies."""

    PAGE = 3

    def __init__(self, name: str = BUCKET) -> None:
        self.name = name
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.fail_after: int | None = None
        # `fail_next` transient failures before the next transfer succeeds — the retry's case.
        self.fail_next = 0
        self.fail_with: BaseException = ConnectionError("connection reset by peer")
        self.transfers = 0
        # `upload_tree` runs FILE_WORKERS threads, so an unlocked check-then-increment would let
        # more than `fail_after` transfers through and the interruption count would drift.
        self.lock = threading.Lock()

    # --- the seven calls S3Bucket makes ---
    def put_object(self, *, Bucket, Key, Body, Metadata=None, ContentType=None) -> None:
        assert Bucket == self.name
        self.objects[Key] = (bytes(Body), dict(Metadata or {}))

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key][0]),
                "Metadata": dict(self.objects[Key][1])}

    def head_object(self, *, Bucket, Key):
        body, metadata = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": dict(metadata)}

    def list_objects_v2(self, *, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = keys.index(ContinuationToken) if ContinuationToken else 0
        page = keys[start:start + self.PAGE]
        response = {"Contents": [{"Key": key, "Size": len(self.objects[key][0])} for key in page]}
        if start + self.PAGE < len(keys):
            response["IsTruncated"] = True
            response["NextContinuationToken"] = keys[start + self.PAGE]
        return response

    def delete_object(self, *, Bucket, Key) -> None:
        self.objects.pop(Key, None)

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, Config=None) -> None:
        # An interrupted transfer leaves NO key: completing a multipart upload is atomic, so a
        # partial tree is missing objects rather than holding short ones.
        with self.lock:
            if self.fail_next:
                self.fail_next -= 1
                raise self.fail_with
            if self.fail_after is not None and self.transfers >= self.fail_after:
                raise ConnectionError("connection reset by peer")
            self.transfers += 1
            self.objects[Key] = (Path(Filename).read_bytes(),
                                 dict((ExtraArgs or {}).get("Metadata", {})))

    def download_file(self, Bucket, Key, Filename, Config=None) -> None:
        Path(Filename).write_bytes(self.objects[Key][0])


@pytest.fixture(autouse=True)
def no_upload_backoff(monkeypatch):
    """The file-level retry pauses for real between attempts; the suite must not."""
    monkeypatch.setattr(store_module, "UPLOAD_BACKOFF_SECONDS", 0.0)


@pytest.fixture
def bucket() -> S3Bucket:
    return S3Bucket(FakeS3(), BUCKET)


def tree(root: Path, shards: int = 4) -> Path:
    """A miniature of the real artifact's shape: sharded weights plus config and tokenizer."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    (root / "tokenizer.json").write_text('{"vocab": {}}')
    for index in range(shards):
        (root / f"model-{index:05d}-of-{shards:05d}.safetensors").write_bytes(
            bytes([index]) * (1024 + index))
    return root


def submission(tmp_path: Path, bucket: S3Bucket, *, registration=REGISTRATION,
               seed: bytes = MINER_SEED) -> tuple[Path, Manifest]:
    root = tree(tmp_path / f"model-{registration.uid}")
    manifest = build_manifest(root, registration, sign_with(seed))
    upload_tree(root, bucket, registration.prefix, manifest)
    return root, manifest


def test_a_partial_upload_is_not_a_submission_because_the_manifest_lands_last(tmp_path, bucket):
    """M6 exit 2. At ~72 GB over hours an interrupted upload is the COMMON case, so every partial
    tree has to be unambiguously not-a-submission — and the last object written saying so is what
    makes that true with no partial-upload table and no timeout to tune."""
    root = tree(tmp_path / "model")
    manifest = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))
    bucket.client.fail_after = 3

    with pytest.raises(ConnectionError):
        upload_tree(root, bucket, REGISTRATION.prefix, manifest)

    assert bucket.list(REGISTRATION.prefix)
    assert REGISTRATION.prefix + MANIFEST_NAME not in bucket.list(REGISTRATION.prefix)
    with pytest.raises(StoreError, match="not a submission"):
        fetch_manifest(bucket, REGISTRATION.prefix)


def test_a_resumed_upload_sends_only_what_is_missing_and_makes_the_same_tree(tmp_path, bucket):
    """M6 exit 5. Hours per miner, so re-sending 72 GB to finish the last shard is not a resume."""
    root = tree(tmp_path / "model")
    manifest = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))
    bucket.client.fail_after = 3
    with pytest.raises(ConnectionError):
        upload_tree(root, bucket, REGISTRATION.prefix, manifest)
    partial = dict(bucket.client.objects)

    bucket.client.fail_after = None
    report = upload_tree(root, bucket, REGISTRATION.prefix, manifest)

    assert len(report.skipped) == 3
    assert set(report.uploaded) | set(report.skipped) == {item.path for item in manifest.files}
    assert report.bytes_uploaded < manifest.total_bytes
    for key, (body, _metadata) in partial.items():
        assert bucket.client.objects[key][0] == body
    for item in manifest.files:
        assert bucket.client.objects[REGISTRATION.prefix + item.path][0] == (
            root / item.path).read_bytes()


def test_a_shard_edited_between_attempts_is_reuploaded_rather_than_skipped(tmp_path, bucket):
    """Resume compares the stored object's DIGEST, not its presence. Skipping on presence would
    leave the old bytes under a manifest naming the new ones — a mismatch that surfaces at duel
    time, with the shot already spent."""
    root = tree(tmp_path / "model")
    upload_tree(root, bucket, REGISTRATION.prefix,
                build_manifest(root, REGISTRATION, sign_with(MINER_SEED)))
    edited = root / "model-00001-of-00004.safetensors"
    edited.write_bytes(b"\xff" * edited.stat().st_size)

    for key in list(bucket.client.objects):
        if key.endswith(MANIFEST_NAME):
            del bucket.client.objects[key]
    manifest = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))
    report = upload_tree(root, bucket, REGISTRATION.prefix, manifest)

    assert edited.name in report.uploaded
    assert bucket.client.objects[REGISTRATION.prefix + edited.name][0] == edited.read_bytes()


def test_uploading_over_a_committed_submission_is_refused(tmp_path, bucket):
    """The chain names those bytes once the ready signal lands, so replacing them fails the miner's
    own submission. The server-side stop is credential revocation plus the duel-time re-hash; this
    refusal is what stops an honest miner doing it by accident."""
    root, manifest = submission(tmp_path, bucket)
    with pytest.raises(StoreError, match="already carries"):
        upload_tree(root, bucket, REGISTRATION.prefix, manifest)


def test_one_swapped_shard_is_caught_at_that_shard_rather_than_averaged_away(tmp_path, bucket):
    """M6 exit 4. The reference tree is 26 shards; a single tree-wide hash would say only that
    something differed, and the operator would re-upload 72 GB to find out which."""
    _root, manifest = submission(tmp_path, bucket)
    signal = commitment(manifest.sha256)
    swapped = REGISTRATION.prefix + "model-00002-of-00004.safetensors"
    body, metadata = bucket.client.objects[swapped]
    bucket.client.objects[swapped] = (bytes(len(body)), metadata)

    with pytest.raises(StoreError, match="model-00002-of-00004.safetensors hashes to"):
        fetch_submission(bucket, tmp_path / "pulled", commitment=signal, registration=REGISTRATION)


def test_the_committed_manifest_is_checked_before_seventy_gigabytes_are_pulled(tmp_path, bucket):
    """A tree whose manifest does not match the on-chain signal is refused for the price of one
    small object rather than an hour of transfer — so nothing is downloaded at all."""
    _root, manifest = submission(tmp_path, bucket)
    stale = commitment("ab" * 32)
    before = bucket.client.transfers

    with pytest.raises(StoreError, match="is not the committed"):
        fetch_submission(bucket, tmp_path / "pulled", commitment=stale, registration=REGISTRATION)
    assert bucket.client.transfers == before
    assert not (tmp_path / "pulled").exists()


def test_a_manifest_lifted_from_another_miner_does_not_pass_as_this_one(tmp_path, bucket):
    """The digest check alone is satisfied by any manifest whose bytes hash correctly — including
    one copied out of another miner's prefix — so the manifest must also be SIGNED by the signalling
    hotkey and name its registration."""
    _root, theirs = submission(tmp_path, bucket, registration=OTHER_REGISTRATION, seed=OTHER_SEED)
    bucket.put(REGISTRATION.prefix + MANIFEST_NAME, theirs.as_bytes())
    signal = commitment(theirs.sha256)

    with pytest.raises(StoreError, match="manifest belongs to"):
        fetch_submission(bucket, tmp_path / "pulled", commitment=signal, registration=REGISTRATION)


def test_a_manifest_the_hotkey_did_not_sign_is_refused(tmp_path, bucket):
    root = tree(tmp_path / "model")
    forged = build_manifest(root, REGISTRATION, sign_with(OTHER_SEED))
    upload_tree(root, bucket, REGISTRATION.prefix, forged)
    signal = commitment(forged.sha256)

    with pytest.raises(StoreError, match="is not signed by"):
        fetch_submission(bucket, tmp_path / "pulled", commitment=signal, registration=REGISTRATION)


def test_a_file_the_manifest_does_not_name_is_refused_rather_than_ignored(tmp_path, bucket):
    """Ignoring it would materialise a file on the validator that the commitment does not cover —
    on the single machine holding the subnet's scoring authority."""
    _root, manifest = submission(tmp_path, bucket)
    bucket.put(REGISTRATION.prefix + "modeling_qwen.py", b"import os")
    signal = commitment(manifest.sha256)

    with pytest.raises(StoreError, match="files the manifest does not name"):
        fetch_submission(bucket, tmp_path / "pulled", commitment=signal, registration=REGISTRATION)


def test_a_committed_tree_round_trips_to_the_validators_disk_byte_for_byte(tmp_path, bucket):
    root, manifest = submission(tmp_path, bucket)
    signal = commitment(manifest.sha256)

    pulled = fetch_submission(bucket, tmp_path / "pulled", commitment=signal,
                              registration=REGISTRATION)

    assert pulled == manifest
    assert inventory(tmp_path / "pulled") == manifest.files
    for item in manifest.files:
        assert (tmp_path / "pulled" / item.path).read_bytes() == (root / item.path).read_bytes()


def test_a_manifest_whose_digest_disagrees_with_its_own_inventory_is_refused():
    """The chain names the manifest and the manifest names the digest, so a digest taken on trust
    would break the only link that reaches the individual shards."""
    files = (ManifestFile(path="a.safetensors", size=3, sha256="ab" * 32),)
    honest = Manifest(registration_id="cd" * 32, hotkey=MINER, files=files,
                      tree_digest=tree_digest(files), signature="sig")
    tampered = json.loads(honest.as_bytes())
    tampered["files"][0]["sha256"] = "ef" * 32

    Manifest.from_bytes(honest.as_bytes())
    with pytest.raises(StoreError, match="tree_digest does not match"):
        Manifest.from_bytes(json.dumps(tampered).encode())


def test_a_manifest_path_that_escapes_the_destination_directory_is_refused():
    """A `..` component is a miner's file landing outside the tree when the validator materialises
    it — arbitrary write on the machine that holds the scoring authority."""
    for path in ("../escape.safetensors", "/etc/passwd", MANIFEST_NAME, "a/./b"):
        with pytest.raises(StoreError, match="unsafe manifest path|path is invalid"):
            ManifestFile.from_mapping({"path": path, "size": 1, "sha256": "ab" * 32})


def test_a_symlink_is_refused_because_its_content_is_decided_by_whoever_resolves_it(tmp_path):
    root = tree(tmp_path / "model")
    (root / "link.safetensors").symlink_to(root / "config.json")
    with pytest.raises(StoreError, match="symlink"):
        inventory(root)


def test_the_reserved_manifest_name_cannot_be_smuggled_in_as_a_tree_file(tmp_path):
    """It is written by the upload as the commit marker, so a tree file of that name would be
    overwritten and the manifest would describe a file whose bytes it does not name."""
    root = tree(tmp_path / "model")
    (root / MANIFEST_NAME).write_text("{}")
    with pytest.raises(StoreError, match="reserved file"):
        inventory(root)


def test_the_transfer_profile_is_sized_for_a_seventy_gigabyte_tree():
    """The part size is teutonic's (§11-8) and is what bounds a single object: 64 MiB x S3's
    10,000-part ceiling is 640 GB, an order above the whole tree, while the largest shard is ~50 GB
    — ~800 parts. The concurrencies are NOT teutonic's 16 x 16: measured 2026-09-08, 32 streams on
    a ~35 MB/s WAN retransmitted themselves into a TLS reset that ended a 68 GB upload, and 4
    streams per file moved the same bytes at the link's ceiling with clean sockets."""
    assert (PART_SIZE, MULTIPART_THRESHOLD, PART_STREAMS, FILE_WORKERS) == (
        64 * 1024 * 1024, 32 * 1024 * 1024, 4, 2)
    assert PART_SIZE * 10_000 > 72_000_000_000


def test_storage_accounting_counts_every_page_of_a_bucket_listing(tmp_path, bucket):
    """M6 exit 6. `list_objects_v2` truncates at 1000 keys and says so only in `IsTruncated`, so an
    unpaginated listing under-reports by however much sits past the first page — an accounting bug
    that grows with the thing being accounted for (§8b.7: 70 TB at a thousand entrants)."""
    _mine, mine = submission(tmp_path, bucket)
    _theirs, theirs = submission(tmp_path, bucket, registration=OTHER_REGISTRATION, seed=OTHER_SEED)
    assert len(mine.files) > bucket.client.PAGE, "the listing must span more than one page"

    held = usage(bucket)

    assert set(held) == {REGISTRATION.prefix, OTHER_REGISTRATION.prefix}
    assert held[REGISTRATION.prefix] == mine.total_bytes + len(mine.as_bytes())
    assert sum(held.values()) == sum(
        len(body) for key, (body, _metadata) in bucket.client.objects.items()
        if key.startswith("submissions/"))
    assert theirs.total_bytes > 0


def test_the_king_and_the_five_pensioners_are_kept_and_the_rest_expire_after_the_grace_window():
    """§8b.7. D14 makes the king's weights public and §5.7 pays the pension for having been the
    platform others build on — a pensioner whose weights were deleted would be a public artifact
    that is not there."""
    now = 1_000_000.0
    prefixes = {f"submissions/{index:064x}/": now - index * 86_400 for index in range(8)}
    protected = {f"submissions/{index:064x}/" for index in range(6)}

    plan = retention_plan(prefixes, protected=protected, now=now, grace_seconds=6.5 * 86_400)

    assert set(plan.kept) == protected
    assert plan.within_grace == ("submissions/" + f"{6:064x}" + "/",)
    assert plan.expired == ("submissions/" + f"{7:064x}" + "/",)


def test_expiring_a_submission_deletes_the_weights_and_keeps_the_record_forever(tmp_path, bucket):
    """§8b.7's second half: the manifest and the history entry are retained forever regardless, so
    the RECORD of every submission is permanent at ~1 KB per entrant even when its 72 GB is not."""
    _root, manifest = submission(tmp_path, bucket)
    plan = retention_plan({REGISTRATION.prefix: 0.0}, protected=(), now=1e9)

    freed = apply_retention(bucket, plan)

    remaining = bucket.list(REGISTRATION.prefix)
    assert set(remaining) == {REGISTRATION.prefix + MANIFEST_NAME}
    assert freed == manifest.total_bytes
    assert fetch_manifest(bucket, REGISTRATION.prefix) == manifest


def test_a_transport_failure_that_outlives_boto3_costs_one_part_not_the_run(tmp_path, bucket, capsys):
    """Measured 2026-09-08: a 68 GB submission died fifty minutes in on a TLS EOF that outlived
    boto3's ten attempts, and although `upload_tree` resumes, the rerun re-hashed the whole tree
    first. Two transient failures on one file are now two pauses and a complete tree."""
    root = tree(tmp_path / "model")
    manifest = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))
    bucket.client.fail_next = 2

    report = upload_tree(root, bucket, REGISTRATION.prefix, manifest)

    assert set(report.uploaded) == {item.path for item in manifest.files}
    assert REGISTRATION.prefix + MANIFEST_NAME in bucket.list(REGISTRATION.prefix)
    assert capsys.readouterr().err.count("retrying in") == 2


def test_a_refused_credential_is_not_retried(tmp_path, bucket):
    """Waiting does not make an expired credential valid, and six pauses on a 403 would hide the
    real message under minutes of silence."""
    class ClientError(Exception):          # botocore's, by name — the classifier reads the MRO
        response = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}

    root = tree(tmp_path / "model")
    manifest = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))
    bucket.client.fail_next = 1
    bucket.client.fail_with = ClientError("AccessDenied")

    with pytest.raises(ClientError):
        upload_tree(root, bucket, REGISTRATION.prefix, manifest)
    assert bucket.client.fail_next == 0 and REGISTRATION.prefix + MANIFEST_NAME not in bucket.list(REGISTRATION.prefix)


# --- promotion: a winner's tree, published ------------------------------------------------------


@pytest.fixture
def public() -> S3Bucket:
    return S3Bucket(FakeS3(PUBLIC_BUCKET), PUBLIC_BUCKET)


def judged_tree(tmp_path: Path, bucket: S3Bucket) -> tuple[Path, Manifest]:
    """A submission as the validator holds it after a duel: fetched, hashed, and marked verified."""
    _, manifest = submission(tmp_path, bucket)
    dest = tmp_path / "trees" / REGISTRATION.registration_id
    fetch_submission(bucket, dest, commitment=commitment(manifest.sha256),
                     registration=REGISTRATION)
    return dest, manifest


def test_a_promoted_king_is_the_verified_tree_under_the_digest_the_chain_committed(tmp_path, bucket,
                                                                                   public):
    """A name nobody chose — the committed manifest digest — holding exactly the committed tree,
    and a `manifest.json` that hashes to what the chain names, so a downloaded king is checkable."""
    dest, manifest = judged_tree(tmp_path, bucket)

    prefix = promote_submission(dest, public, manifest=manifest)

    assert prefix == public_model_prefix(manifest.sha256)
    assert prefix == f"{PUBLIC_MODEL_ROOT}{manifest.sha256}/"
    assert set(public.list(prefix)) == ({prefix + MANIFEST_NAME}
                                        | {prefix + item.path for item in manifest.files})
    for item in manifest.files:
        assert public.get(prefix + item.path) == (dest / item.path).read_bytes()
        assert public.head(prefix + item.path)["Metadata"]["sha256"] == item.sha256
    assert hashlib.sha256(public.get(prefix + MANIFEST_NAME)).hexdigest() == manifest.sha256


def test_bytes_replaced_in_the_private_bucket_after_judgment_are_never_published(tmp_path, bucket,
                                                                                 public):
    """Teutonic revokes the upload token so a judged submission cannot change. Promotion here does
    not lean on that: it reads the tree from the validator's verified copy, so a same-size shard
    swapped into the private bucket after the duel changes nothing that is published."""
    dest, manifest = judged_tree(tmp_path, bucket)
    shard = next(item for item in manifest.files if item.path.endswith(".safetensors"))
    judged, metadata = bucket.client.objects[REGISTRATION.prefix + shard.path]
    bucket.client.objects[REGISTRATION.prefix + shard.path] = (b"\xff" * len(judged), metadata)

    prefix = promote_submission(dest, public, manifest=manifest)

    assert public.get(prefix + shard.path) == judged


def test_a_tree_this_validator_never_verified_is_not_published(tmp_path, bucket, public):
    """The miner's directory has the right bytes and no marker: nothing unhashed is published."""
    root, manifest = submission(tmp_path, bucket)

    with pytest.raises(StoreError, match="not a tree this validator verified"):
        promote_submission(root, public, manifest=manifest)
    assert public.client.objects == {}


def test_a_verified_tree_that_has_lost_a_file_since_is_not_published(tmp_path, bucket, public):
    dest, manifest = judged_tree(tmp_path, bucket)
    (dest / manifest.files[0].path).unlink()

    with pytest.raises(StoreError, match="not a tree this validator verified"):
        promote_submission(dest, public, manifest=manifest)
    assert public.client.objects == {}


def test_a_public_prefix_holding_other_bytes_is_a_collision_and_no_king_lands(tmp_path, bucket,
                                                                            public):
    """Content addressing is only worth something if the name cannot hold anything else."""
    dest, manifest = judged_tree(tmp_path, bucket)
    prefix = public_model_prefix(manifest.sha256)

    public.put(prefix + "stray.bin", b"not this model")
    with pytest.raises(StoreError, match="does not name"):
        promote_submission(dest, public, manifest=manifest)

    del public.client.objects[prefix + "stray.bin"]
    item = manifest.files[0]
    public.put(prefix + item.path, b"z" * item.size, digest="00" * 32)
    with pytest.raises(StoreError, match="the manifest says"):
        promote_submission(dest, public, manifest=manifest)
    assert prefix + MANIFEST_NAME not in public.client.objects


def test_an_interrupted_promotion_resumes_and_sends_each_file_once(tmp_path, bucket, public):
    """A 70 GB king is an hour of upload; a retry that re-sent it all would never finish on a flaky
    link. And until `manifest.json` lands last, the prefix is not a king anyone should download."""
    dest, manifest = judged_tree(tmp_path, bucket)
    prefix = public_model_prefix(manifest.sha256)
    public.client.fail_after = 2
    with pytest.raises(ConnectionError):
        promote_submission(dest, public, manifest=manifest)
    assert prefix + MANIFEST_NAME not in public.client.objects

    public.client.fail_after = None
    assert promote_submission(dest, public, manifest=manifest) == prefix
    assert public.client.transfers == len(manifest.files)
    assert promote_submission(dest, public, manifest=manifest) == prefix     # idempotent once whole
    assert public.client.transfers == len(manifest.files)


# --- protocol 2: the name a miner gives their model -----------------------------------------------


def test_a_named_model_is_a_protocol_2_manifest_and_the_name_is_signed(tmp_path):
    """Outside the signature the name would be the one field of a submission its miner did not vouch
    for — rewritable by anyone holding the prefix, after the chain had committed to the digest."""
    root = tree(tmp_path / "model")
    named = build_manifest(root, REGISTRATION, sign_with(MINER_SEED), model_name="my-router")

    assert named.protocol_version == 2 and named.model_name == "my-router"
    assert b'"model_name":"my-router"' in named.signing_payload()
    assert Manifest.from_bytes(named.as_bytes()) == named


def test_a_manifest_with_no_name_signs_exactly_what_protocol_1_signed(tmp_path):
    """Names must not invalidate a submission made before they existed."""
    root = tree(tmp_path / "model")
    plain = build_manifest(root, REGISTRATION, sign_with(MINER_SEED))

    assert plain.protocol_version == 1 and plain.model_name is None
    assert b"model_name" not in plain.signing_payload()
    assert Manifest.from_bytes(plain.as_bytes()) == plain


@pytest.mark.parametrize("name", ["ab", "A-Router", "my router", "x" * 41, "-leading",
                                  "thirtyspokes-genesis", "thirtyspokes-anything"])
def test_a_name_that_could_be_misread_or_impersonate_the_subnet_is_refused(tmp_path, name):
    root = tree(tmp_path / "model")
    with pytest.raises(StoreError):
        build_manifest(root, REGISTRATION, sign_with(MINER_SEED), model_name=name)


def test_a_manifest_that_carries_a_name_while_claiming_version_1_is_refused(tmp_path):
    """The version is signed too, so the same bytes must not be readable as either protocol."""
    root = tree(tmp_path / "model")
    named = build_manifest(root, REGISTRATION, sign_with(MINER_SEED), model_name="my-router")
    tampered = {**json.loads(named.as_bytes()), "protocol_version": 1}

    with pytest.raises(StoreError, match="contract"):
        Manifest.from_bytes(json.dumps(tampered).encode())
