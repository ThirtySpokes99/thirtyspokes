"""Challenger trees pulled, and kings published, on the serving host (docs/WHITEPAPER.md §8b.9).

TWO FAR SIDES, BOTH OFFLINE. The real-bash tests run the GENERATED host script with a local
`bash -s` — exactly what `serve.ssh_runner` feeds to ssh — against a temp directory standing in for
the host's trees, and a 127.0.0.1 HTTP server standing in for R2's presigned GET and UploadPart.
They skip cleanly when curl or sha256sum is missing. `SimulatedHost` is a pure-Python runner that
answers the script the way a host would, or the way a broken one would, for the failures a real
shell cannot be made to produce on demand (a dropped line, a lying host, a timeout).

WHAT IS UNDER TEST IS THE CONTROLLER'S JUDGMENT. Every assertion that matters is about which
exception a failure becomes: a plain `StoreError` spends a miner's one shot, so it must mean
"these bytes are not the committed tree" and nothing else, while a host that failed, timed out or
said something unreadable must surface as the owner's error.

No miner, uid or model here is real; names and keys are synthetic.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import thirtyspokes.v3.hosttrees as hosttrees_module
from thirtyspokes.v3.access import Registration, encode_ss58
from thirtyspokes.v3.chain import Commitment, ReadySignal
from thirtyspokes.v3.hosttrees import (FETCH_ATTEMPTS, HOST_TRANSFER_PARALLELISM,
                                       PRESIGNED_TRANSFER_SECONDS, TREE_TRANSFER_TIMEOUT_SECONDS,
                                       HostTransferError, HostTrees, host_trees)
from thirtyspokes.v3.serve import ServeError
from thirtyspokes.v3.store import (MANIFEST_NAME, PARTIAL_MARKER, VERIFIED_MARKER, Manifest,
                                   ManifestFile, S3Bucket, StoreError, build_manifest,
                                   fetch_submission, promote_submission, public_model_prefix,
                                   tree_digest, upload_tree)

SECRET = "presign-secret-7f3a9c"          # what a presigned URL's signature stands in for
SEED = hashlib.sha256(b"synthetic-miner").digest()
HOTKEY = encode_ss58(Ed25519PrivateKey.from_private_bytes(SEED).public_key().public_bytes_raw())
REGISTRATION = Registration(netuid=99, uid=3, hotkey=HOTKEY, registration_block=1_000)
PRIVATE, PUBLIC = "synthetic-private", "synthetic-public"

needs_shell = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("sha256sum") is None
    or shutil.which("bash") is None, reason="the host script needs bash, curl and sha256sum")


def sign(payload: bytes) -> str:
    return Ed25519PrivateKey.from_private_bytes(SEED).sign(payload).hex()


def commitment(manifest: Manifest) -> Commitment:
    return Commitment(hotkey=HOTKEY, block=1_100,
                      ready=ReadySignal(registration_id=REGISTRATION.registration_id,
                                        manifest_sha256=manifest.sha256))


# --- the bucket and the HTTP far side --------------------------------------------------------------


class FakeS3:
    """An in-memory bucket speaking boto3's calls, multipart and presigning included.

    A presigned URL here points at `base` (the local server, or an unroutable name for the
    simulated host), names the operation, and carries `SECRET` the way a real one carries its
    signature — so "the URL never leaked" is testable as "the secret never appears"."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.base = "https://r2.invalid"
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.writes: list[str] = []
        self.uploads: dict[str, dict] = {}
        self.aborted: list[str] = []
        self.presigned = 0
        self.transfers = 0
        self.lock = threading.Lock()

    def put_object(self, *, Bucket, Key, Body, Metadata=None, ContentType=None) -> None:
        self.objects[Key] = (bytes(Body), dict(Metadata or {}))
        self.writes.append(Key)

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def head_object(self, *, Bucket, Key):
        body, metadata = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": dict(metadata)}

    def list_objects_v2(self, *, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        return {"Contents": [{"Key": key, "Size": len(self.objects[key][0])} for key in keys]}

    def delete_object(self, *, Bucket, Key) -> None:
        self.objects.pop(Key, None)

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, Config=None) -> None:
        self.transfers += 1
        self.objects[Key] = (Path(Filename).read_bytes(), dict((ExtraArgs or {}).get("Metadata", {})))
        self.writes.append(Key)

    def download_file(self, Bucket, Key, Filename, Config=None) -> None:
        self.transfers += 1
        Path(Filename).write_bytes(self.objects[Key][0])

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        assert ExpiresIn == PRESIGNED_TRANSFER_SECONDS
        self.presigned += 1
        url = (f"{self.base}/{operation}/{Params['Bucket']}/{quote(Params['Key'], safe='')}"
               f"?token={SECRET}")
        if operation == "upload_part":
            url += f"&uploadId={Params['UploadId']}&partNumber={Params['PartNumber']}"
        return url

    def create_multipart_upload(self, *, Bucket, Key, Metadata):
        with self.lock:
            upload_id = f"upload-{len(self.uploads) + 1}"
            self.uploads[upload_id] = {"key": Key, "metadata": dict(Metadata), "parts": {},
                                       "open": True}
        return {"UploadId": upload_id}

    def upload_part(self, upload_id: str, number: int, body: bytes) -> str:
        with self.lock:
            upload = self.uploads[upload_id]
            assert upload["open"], "a part was sent to an upload that is no longer open"
            upload["parts"][number] = body
        return hashlib.md5(body).hexdigest()

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
        upload = self.uploads[UploadId]
        assert upload["open"] and upload["key"] == Key
        body = b""
        for part in MultipartUpload["Parts"]:
            stored = upload["parts"][part["PartNumber"]]
            if part["ETag"].strip('"') != hashlib.md5(stored).hexdigest():     # S3's InvalidPart
                raise RuntimeError("InvalidPart")
            body += stored
        upload["open"] = False
        self.objects[Key] = (body, dict(upload["metadata"]))
        self.writes.append(Key)

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.uploads[UploadId]["open"] = False
        self.aborted.append(UploadId)

    def list_multipart_uploads(self, *, Bucket, Prefix, **markers):
        return {"Uploads": [{"Key": u["key"], "UploadId": upload_id}
                            for upload_id, u in self.uploads.items()
                            if u["open"] and u["key"].startswith(Prefix)]}

    def open_uploads(self) -> list[str]:
        return [upload_id for upload_id, upload in self.uploads.items() if upload["open"]]

    # R2 cannot pin a server-side copy to the judged version; promotion must never ask for one.
    def copy_object(self, **kwargs):
        raise AssertionError("promotion used CopyObject")

    def upload_part_copy(self, **kwargs):
        raise AssertionError("promotion used UploadPartCopy")


class R2Http:
    """The presigned-URL endpoint: GET an object, PUT a part. Refuses a request without the
    secret, and can be told to misbehave per key."""

    def __init__(self, *buckets: FakeS3) -> None:
        self.buckets = {bucket.name: bucket for bucket in buckets}
        self.gets: dict[str, int] = {}
        self.forbidden: set[str] = set()
        self.corrupt: dict[str, object] = {}       # key -> bytes, or a callable giving fresh bytes
        self.part_etag: object = None              # None = honest; a str replaces it; "" drops it
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # the suite's output stays quiet
                pass

            def _target(self):
                parts = urlsplit(self.path)
                _, operation, bucket, key = parts.path.split("/", 3)
                return operation, outer.buckets[bucket], unquote(key), parse_qs(parts.query)

            def do_GET(self):
                operation, bucket, key, query = self._target()
                outer.requests.append(self.path)
                if query.get("token") != [SECRET] or operation != "get_object" or key in outer.forbidden:
                    self.send_response(403)
                    self.end_headers()
                    return
                outer.gets[key] = outer.gets.get(key, 0) + 1
                body = bucket.objects[key][0]
                if key in outer.corrupt:
                    spoiled = outer.corrupt[key]
                    body = spoiled() if callable(spoiled) else spoiled
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):
                operation, bucket, key, query = self._target()
                outer.requests.append(self.path)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if query.get("token") != [SECRET] or operation != "upload_part" or key in outer.forbidden:
                    self.send_response(403)
                    self.end_headers()
                    return
                md5 = bucket.upload_part(query["uploadId"][0], int(query["partNumber"][0]), body)
                self.send_response(200)
                etag = f'"{md5}"' if outer.part_etag is None else outer.part_etag
                if etag:
                    # Lowercase, as HTTP/2 delivers it: the parser must not depend on the case.
                    self.send_header("etag", etag)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        for bucket in buckets:
            bucket.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# --- runners ---------------------------------------------------------------------------------------


def local_runner(tmp_path: Path, *, timeout: float = 60.0):
    """`serve.ssh_runner` without the ssh: the script on `bash -s`'s stdin, ServeError on a non-zero
    exit. `curl` on its PATH is a shim that records its argv before running the real one, so what a
    process listing on the host would show is checkable."""
    shims = tmp_path / "shims"
    shims.mkdir(exist_ok=True)
    argv_log = tmp_path / "curl-argv.log"
    shim = shims / "curl"
    shim.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> {argv_log}\nexec {shutil.which("curl")} "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{shims}:{os.environ.get('PATH', '')}"}
    scripts: list[str] = []

    def run(script: str) -> str:
        scripts.append(script)
        done = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True,
                              timeout=timeout, check=False, env=env)
        if done.returncode != 0:
            raise ServeError(f"host: exit {done.returncode}: {done.stderr.strip()[-300:]}")
        return done.stdout

    run.argv_log = argv_log
    run.scripts = scripts
    return run


def script_data(script: str) -> tuple[str, list[list[str]]]:
    """The host-side trees path and the heredoc's data lines, as a host's bash would read them."""
    base = re.search(r"^base=(.*)$", script, re.M).group(1)
    delimiter = re.search(r"<<'(THIRTYSPOKES_DATA_[0-9a-f]+)'", script).group(1)
    body = script.split(f"<<'{delimiter}'\n", 1)[1].split(f"{delimiter}\n", 1)[0]
    import shlex
    return shlex.split(base)[0], [line.split("\t") for line in body.split("\n") if line]


class SimulatedHost:
    """A pure-Python host: reads the fetch script's data, places honest bytes, and reports — or
    reports what a broken host would. `bucket` is where it 'downloads' from."""

    def __init__(self, bucket: FakeS3) -> None:
        self.bucket = bucket
        self.calls = 0
        self.mutate = lambda lines: lines
        self.raises: BaseException | None = None
        self.write = True

    def __call__(self, script: str) -> str:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        base, rows = script_data(script)
        lines = []
        for index, (size, digest, url, path) in enumerate(rows):
            key = unquote(urlsplit(url).path.split("/", 3)[3])
            body = self.bucket.objects[key][0]
            observed = hashlib.sha256(body).hexdigest()
            if self.write:
                target = Path(base) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            status = "fetched" if (observed, str(len(body))) == (digest, size) else "mismatch"
            lines.append(f"THIRTYSPOKES-FETCH\t{index}\t{status}\t{len(body)}\t{observed}\t0\t{path}")
        lines.append(f"THIRTYSPOKES-FETCH-DONE\t{len(rows)}")
        return "".join(line + "\n" for line in self.mutate(lines))


# --- fixtures ------------------------------------------------------------------------------------


def write_tree(root: Path) -> Path:
    """Sharded weights, a nested file, an empty file, and names a careless script would break on:
    a space, a backslash, non-ASCII, and the renderer's own token spelling."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "synthetic"}')
    for index in range(3):
        (root / f"model-{index:05d}-of-00003.safetensors").write_bytes(bytes([index + 1]) * (3000 + index))
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "naïve name \\ with @@DATA@@.json").write_text('{"vocab": {}}')
    (root / "empty.txt").write_bytes(b"")
    return root


@pytest.fixture
def world(tmp_path):
    """A committed submission in the private bucket, and the host-side trees directory, which in
    these tests IS the controller's `<state>/trees` (the mount, collapsed to one disk)."""
    private = FakeS3(PRIVATE)
    bucket = S3Bucket(private, PRIVATE)
    source = write_tree(tmp_path / "miner")
    manifest = build_manifest(source, REGISTRATION, sign)
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    private.transfers = 0                  # the miner's upload is not the validator's transfer
    trees = tmp_path / "trees"
    trees.mkdir()
    return {"private": private, "bucket": bucket, "manifest": manifest, "source": source,
            "trees": trees, "dest": trees / REGISTRATION.registration_id}


def shard(manifest: Manifest) -> ManifestFile:
    return next(item for item in manifest.files if item.path.endswith("00001-of-00003.safetensors"))


def host_for(run, trees: Path, **overrides) -> HostTrees:
    options = dict(schemes=("http",), pause_seconds=0, settle_attempts=2, settle_seconds=0.0)
    options.update(overrides)
    return HostTrees(run, str(trees), **options)


@pytest.fixture
def http(world):
    server = R2Http(world["private"])
    yield server
    server.close()


def fetch(world, host):
    return fetch_submission(world["bucket"], world["dest"], commitment=commitment(world["manifest"]),
                            registration=REGISTRATION, host=host)


def leftovers(root: Path) -> list[str]:
    return sorted(str(path) for path in root.rglob(f"*{PARTIAL_MARKER}*"))


# --- fetch, through a real shell ---------------------------------------------------------------------


@needs_shell
def test_a_tree_fetched_on_the_host_is_the_committed_tree_byte_for_byte_and_carries_the_same_verified_marker(
        tmp_path, world, http):
    run = local_runner(tmp_path)

    manifest = fetch(world, host_for(run, world["trees"]))

    assert manifest == world["manifest"]
    held = {path.relative_to(world["dest"]).as_posix() for path in world["dest"].rglob("*")
            if path.is_file()}
    assert held == {item.path for item in manifest.files}
    for item in manifest.files:
        assert (world["dest"] / item.path).read_bytes() == (world["source"] / item.path).read_bytes()
    marker = world["trees"] / f"{REGISTRATION.registration_id}{VERIFIED_MARKER}"
    assert marker.read_text(encoding="utf-8") == manifest.sha256
    assert world["private"].transfers == 0, "no byte of the tree moved through this process"
    assert leftovers(world["trees"]) == []


@needs_shell
def test_a_shard_that_hashes_wrong_on_the_host_is_a_store_error_naming_that_shard_and_nothing_lands_under_its_name(
        tmp_path, world, http):
    """Same size, different bytes, served identically twice: that is the bucket's content, and the
    refusal carries the same message a local fetch gives."""
    item = shard(world["manifest"])
    http.corrupt[REGISTRATION.prefix + item.path] = b"\x00" * item.size

    with pytest.raises(StoreError, match=rf"{re.escape(item.path)} hashes to [0-9a-f]{{64}}, the "
                                         rf"committed manifest says {item.sha256}") as caught:
        fetch(world, host_for(local_runner(tmp_path), world["trees"]))

    assert type(caught.value) is StoreError
    assert not (world["dest"] / item.path).exists()
    assert leftovers(world["trees"]) == []
    assert not (world["trees"] / f"{REGISTRATION.registration_id}{VERIFIED_MARKER}").exists()
    assert http.gets[REGISTRATION.prefix + item.path] == 2, "one bad read is not evidence"


@needs_shell
def test_files_already_on_the_host_at_their_committed_digest_are_kept_not_pulled_again(
        tmp_path, world, http):
    run = local_runner(tmp_path)
    fetch(world, host_for(run, world["trees"]))
    (world["trees"] / f"{REGISTRATION.registration_id}{VERIFIED_MARKER}").unlink()
    edited = shard(world["manifest"])
    (world["dest"] / edited.path).write_bytes(b"\xff" * edited.size)       # one file gone bad
    http.gets.clear()

    fetch(world, host_for(run, world["trees"]))

    assert http.gets == {REGISTRATION.prefix + edited.path: 1}
    assert (world["dest"] / edited.path).read_bytes() == (world["source"] / edited.path).read_bytes()


@needs_shell
def test_an_interrupted_fetch_resumes_without_a_partial_file_under_a_final_name(tmp_path, world,
                                                                                   http):
    """A killed run leaves partials; an older downloader left its own temp names; nothing the
    manifest does not name may survive into the tree admission walks, and no symlink may redirect a
    write out of it."""
    dest = world["dest"]
    first = next(item for item in world["manifest"].files
                 if item.path == "model-00000-of-00003.safetensors")
    (dest / "tokenizer").mkdir(parents=True)
    (dest / first.path).write_bytes((world["source"] / first.path).read_bytes())
    (dest / f"model-00002-of-00003.safetensors{PARTIAL_MARKER}-4242").write_bytes(b"half")
    (dest / "model-00000-of-00003.safetensors.5f1e2d3c").write_bytes(b"an older temp name")
    outside = tmp_path / "outside.txt"
    outside.write_text("must not change")
    (dest / "config.json").unlink(missing_ok=True)
    (dest / "config.json").symlink_to(outside)

    fetch(world, host_for(local_runner(tmp_path), world["trees"]))

    held = {path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()}
    assert held == {item.path for item in world["manifest"].files}
    assert not (dest / "config.json").is_symlink()
    assert outside.read_text() == "must not change"
    assert http.gets.get(REGISTRATION.prefix + first.path) is None, "the present file was kept"


@needs_shell
def test_a_transfer_that_fails_on_the_host_defers_rather_than_refusing(tmp_path, world, http):
    item = shard(world["manifest"])
    http.forbidden.add(REGISTRATION.prefix + item.path)

    with pytest.raises(HostTransferError, match="host transfer failed") as caught:
        fetch(world, host_for(local_runner(tmp_path), world["trees"]))

    assert not isinstance(caught.value, StoreError) and isinstance(caught.value, ServeError)
    assert SECRET not in str(caught.value)


@needs_shell
def test_a_host_that_receives_a_different_size_than_the_bucket_lists_is_a_transfer_fault(
        tmp_path, world, http):
    """The listing was checked against the manifest before the host was asked, so a short body is
    the path's fault — a proxy, a truncation — and not the miner's tree."""
    item = shard(world["manifest"])
    http.corrupt[REGISTRATION.prefix + item.path] = b"\x01" * (item.size - 1)

    with pytest.raises(HostTransferError, match="transfer fault"):
        fetch(world, host_for(local_runner(tmp_path), world["trees"]))


@needs_shell
def test_two_downloads_that_disagree_are_never_read_as_evidence(tmp_path, world, http):
    item = shard(world["manifest"])
    counter = iter(range(100, 256))          # never the shard's real fill byte
    http.corrupt[REGISTRATION.prefix + item.path] = lambda: bytes([next(counter)]) * item.size

    with pytest.raises(HostTransferError, match="disagreed"):
        fetch(world, host_for(local_runner(tmp_path), world["trees"]))


@needs_shell
def test_presigned_urls_never_appear_in_argv_logs_or_exception_text(tmp_path, world, http, capsys):
    run = local_runner(tmp_path)
    fetch(world, host_for(run, world["trees"]))
    (world["trees"] / f"{REGISTRATION.registration_id}{VERIFIED_MARKER}").unlink()
    shutil.rmtree(world["dest"])
    item = shard(world["manifest"])
    http.forbidden.add(REGISTRATION.prefix + item.path)
    with pytest.raises(HostTransferError) as failed:
        fetch(world, host_for(run, world["trees"]))
    http.forbidden.clear()
    http.corrupt[REGISTRATION.prefix + item.path] = b"\x00" * item.size
    with pytest.raises(StoreError) as refused:
        fetch(world, host_for(run, world["trees"]))

    assert run.argv_log.read_text(), "the shim saw curl run"
    assert SECRET not in run.argv_log.read_text()
    assert "http://" not in run.argv_log.read_text()
    for exc in (failed.value, refused.value):
        assert SECRET not in str(exc) and SECRET not in repr(exc)
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    # And the only place a URL occurs in the script is the data section on stdin.
    for script in run.scripts:
        head, _, body = script.partition("<<'THIRTYSPOKES_DATA_")
        assert SECRET not in head and SECRET in body
        commands = script.replace("\\\n", " ").splitlines()
        curls = [line for line in commands if re.match(r"^\s*(\|\s*)?curl ", line)]
        assert curls and all("-K <(printf" in line for line in curls)


@needs_shell
def test_a_serve_trees_that_does_not_match_the_mount_defers_after_the_settle_retries(tmp_path,
                                                                                       world, http):
    elsewhere = tmp_path / "host-trees-somewhere-else"
    elsewhere.mkdir()
    pauses: list[float] = []
    host = host_for(local_runner(tmp_path), elsewhere, settle_attempts=3, settle_seconds=7.0,
                    sleep=pauses.append)

    with pytest.raises(HostTransferError, match="--serve-trees does not name"):
        fetch(world, host)

    assert pauses == [7.0, 7.0]
    assert not (world["trees"] / f"{REGISTRATION.registration_id}{VERIFIED_MARKER}").exists()


# --- fetch, against a simulated host -----------------------------------------------------------------


def simulated(world) -> tuple[SimulatedHost, HostTrees]:
    runner = SimulatedHost(world["private"])
    return runner, host_for(runner, world["trees"], schemes=("https",))


def test_the_manifest_is_checked_before_the_host_is_asked_for_anything(world):
    runner, host = simulated(world)
    stale = Commitment(hotkey=HOTKEY, block=1_100,
                       ready=ReadySignal(registration_id=REGISTRATION.registration_id,
                                         manifest_sha256="ab" * 32))

    with pytest.raises(StoreError, match="is not the committed"):
        fetch_submission(world["bucket"], world["dest"], commitment=stale,
                         registration=REGISTRATION, host=host)
    assert runner.calls == 0 and world["private"].presigned == 0


def test_a_committed_file_missing_from_the_bucket_is_refused_before_the_host_is_asked(world):
    runner, host = simulated(world)
    item = shard(world["manifest"])
    del world["private"].objects[REGISTRATION.prefix + item.path]

    with pytest.raises(StoreError, match="does not hold"):
        fetch(world, host)
    body, metadata = world["private"].objects[REGISTRATION.prefix + world["manifest"].files[0].path]
    assert runner.calls == 0


def test_a_committed_file_stored_at_the_wrong_size_is_refused_before_the_host_is_asked(world):
    runner, host = simulated(world)
    item = shard(world["manifest"])
    world["private"].objects[REGISTRATION.prefix + item.path] = (b"short", {})

    with pytest.raises(StoreError, match=rf"{re.escape(item.path)} is 5 bytes"):
        fetch(world, host)
    assert runner.calls == 0


def test_a_corrupted_file_reported_by_the_host_is_a_store_error_with_the_local_message(world):
    runner, host = simulated(world)
    item = shard(world["manifest"])
    world["private"].objects[REGISTRATION.prefix + item.path] = (b"\x00" * item.size, {})

    with pytest.raises(StoreError, match=rf"{re.escape(item.path)} hashes to [0-9a-f]{{64}}, "
                                         rf"the committed manifest says {item.sha256}"):
        fetch(world, host)


def test_a_missing_result_line_is_an_owner_side_failure_not_a_refusal(world):
    runner, host = simulated(world)
    runner.mutate = lambda lines: [line for line in lines if "\t1\t" not in line[:24]]

    with pytest.raises(HostTransferError, match="reported nothing") as caught:
        fetch(world, host)
    assert not isinstance(caught.value, StoreError)


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda lines: lines[:-1], id="no-completion-line"),
    pytest.param(lambda lines: lines[:-1] + ["THIRTYSPOKES-FETCH-DONE\t999"], id="wrong-count"),
    pytest.param(lambda lines: [lines[0].replace("\tfetched\t", "\tfetched\tnot-a-size\t", 1)]
                 + lines[1:], id="extra-field"),
    pytest.param(lambda lines: [re.sub(r"\t[0-9a-f]{64}\t", "\tZZZZ\t", lines[0])] + lines[1:],
                 id="unreadable-digest"),
    pytest.param(lambda lines: [lines[0], lines[0]] + lines[1:], id="duplicate-line"),
    pytest.param(lambda lines: [lines[0].rsplit("\t", 1)[0] + "\tsomething-else"] + lines[1:],
                 id="wrong-path"),
    pytest.param(lambda lines: ["THIRTYSPOKES-UPLOAD-DONE\t1\t1"] + lines, id="foreign-record"),
])
def test_garbled_or_truncated_host_output_is_never_read_as_a_verdict(world, mutation):
    runner, host = simulated(world)
    item = shard(world["manifest"])
    # Even with a genuinely corrupt shard in the run, output that cannot be read is not evidence.
    world["private"].objects[REGISTRATION.prefix + item.path] = (b"\x00" * item.size, {})
    runner.mutate = mutation

    with pytest.raises(HostTransferError):
        fetch(world, host)


def test_chatter_that_is_not_a_record_is_ignored(world):
    runner, host = simulated(world)
    runner.mutate = lambda lines: ["Welcome to the serving host", ""] + lines + ["bye"]

    assert fetch(world, host) == world["manifest"]


def test_a_host_that_claims_a_wrong_digest_was_kept_is_an_owner_side_failure(world):
    runner, host = simulated(world)
    runner.mutate = lambda lines: [re.sub(r"\tfetched\t(\d+)\t[0-9a-f]{64}\t",
                                          lambda m: f"\tkept\t{m.group(1)}\t{'0' * 64}\t", line)
                                   for line in lines]

    with pytest.raises(HostTransferError, match="does not give") as caught:
        fetch(world, host)
    assert not isinstance(caught.value, StoreError)


def test_a_runner_timeout_propagates_as_itself(world):
    runner, host = simulated(world)
    runner.raises = subprocess.TimeoutExpired(["ssh", "gpu", "bash", "-s"], TREE_TRANSFER_TIMEOUT_SECONDS)

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        fetch(world, host)
    assert SECRET not in str(caught.value)


def test_a_runner_failure_propagates_as_a_serve_error(world):
    runner, host = simulated(world)
    runner.raises = ServeError("ssh gpu: exit 255: connection refused")

    with pytest.raises(ServeError) as caught:
        fetch(world, host)
    assert type(caught.value) is ServeError


def test_output_that_cannot_be_decoded_is_the_hosts_failure(world):
    runner, host = simulated(world)
    runner.raises = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(HostTransferError, match="decoded"):
        fetch(world, host)


def test_a_resident_verified_tree_needs_no_host_call(world):
    """A marker written by the in-process fetch — the format already on the live disk — is honoured
    by the host path without a single call."""
    fetch(world, None)
    runner, host = simulated(world)

    assert fetch(world, host) == world["manifest"]
    assert runner.calls == 0 and world["private"].presigned == 0


def test_the_mount_check_reads_the_tree_the_way_admission_does(world):
    """A stat of each committed path would pass while a stray file sat in the listing admission
    walks; the check compares the whole listing, so an extra file is caught too."""
    runner, host = simulated(world)
    original = runner.__call__

    def with_stray(script: str) -> str:
        output = original(script)
        (world["dest"] / "stray.py").write_text("print('not committed')")
        return output

    host.run = with_stray
    with pytest.raises(HostTransferError, match="does not show the tree"):
        fetch(world, host)


def forged(world, path: str) -> Manifest:
    """A signed manifest naming `path`, bypassing the parser, with the object in the bucket."""
    body = b"synthetic"
    files = tuple(sorted(world["manifest"].files + (ManifestFile(path=path, size=len(body),
                                                                 sha256=hashlib.sha256(body).hexdigest()),)))
    unsigned = replace(world["manifest"], files=files, tree_digest=tree_digest(files), signature="-")
    manifest = replace(unsigned, signature=sign(unsigned.signing_payload()))
    world["private"].objects[REGISTRATION.prefix + path] = (body, {})
    world["private"].objects[REGISTRATION.prefix + MANIFEST_NAME] = (manifest.as_bytes(), {})
    return manifest


@pytest.mark.parametrize("path", ["tokenizer/a\nb.json", "a\tb.json", "a\x00b.json", "a\x7fb.json",
                                  "a\rb.json"])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "host"])
def test_a_path_with_a_control_character_is_refused_in_every_mode_before_anything_moves(
        world, path, remote):
    manifest = forged(world, path)
    runner, host = simulated(world)

    with pytest.raises(StoreError, match="control character"):
        fetch_submission(world["bucket"], world["dest"], commitment=commitment(manifest),
                         registration=REGISTRATION, host=host if remote else None)
    assert runner.calls == 0 and world["private"].transfers == 0


@pytest.mark.parametrize("path", [f"model.safetensors{PARTIAL_MARKER}-1", "config.json/inner.json"])
def test_a_path_that_collides_with_the_download_or_another_file_is_refused_in_every_mode(world,
                                                                                         path):
    manifest = forged(world, path)
    for host in (None, simulated(world)[1]):
        with pytest.raises(StoreError, match="reserved|as a file and as the directory"):
            fetch_submission(world["bucket"], world["dest"], commitment=commitment(manifest),
                             registration=REGISTRATION, host=host)


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape.json", "a/../../b", "a\nb", "a\x00b",
                                  "a\x1bb"])
def test_unsafe_paths_never_reach_a_shell(world, path):
    """The manifest parser and `fetch_submission` both refuse these first; the host module refuses
    them again on its own, so no caller can hand one to a script by skipping those."""
    runner, host = simulated(world)
    item = ManifestFile.__new__(ManifestFile)
    object.__setattr__(item, "path", path)
    object.__setattr__(item, "size", 1)
    object.__setattr__(item, "sha256", "ab" * 32)
    manifest = replace(world["manifest"], files=(item,))

    with pytest.raises(StoreError, match="cannot be passed to the host"):
        host.fetch(world["bucket"], REGISTRATION.prefix, manifest, world["dest"])
    assert runner.calls == 0 and world["private"].presigned == 0


def test_a_presigned_url_that_could_break_its_quoting_is_refused_without_being_quoted(world):
    runner, host = simulated(world)
    world["private"].generate_presigned_url = (
        lambda operation, Params, ExpiresIn: f'https://r2.invalid/x?token={SECRET}"; rm -rf /')

    with pytest.raises(HostTransferError) as caught:
        fetch(world, host)
    assert SECRET not in str(caught.value) and runner.calls == 0


@pytest.mark.parametrize("trees", ["relative/trees", "/var/v3/../etc", "/var/v3/trees; rm -rf /",
                                   "/var/v3/$HOME"])
def test_a_serve_trees_that_is_not_a_plain_absolute_path_is_refused_at_launch(trees):
    with pytest.raises(ServeError):
        HostTrees(lambda script: "", trees)


# --- the transfer profile and the wiring --------------------------------------------------------------


def test_the_presign_ttl_outlives_the_transfer_timeout():
    assert PRESIGNED_TRANSFER_SECONDS > TREE_TRANSFER_TIMEOUT_SECONDS
    assert PRESIGNED_TRANSFER_SECONDS <= 7 * 86_400, "SigV4's ceiling on a presigned URL"


def test_the_fetch_runner_timeout_is_hours_and_not_the_serving_runners(monkeypatch):
    seen: dict = {}

    def fake_ssh_runner(host, *, timeout=180.0):
        seen.update(host=host, timeout=timeout)
        return lambda script: ""

    monkeypatch.setattr(hosttrees_module, "ssh_runner", fake_ssh_runner)
    trees = host_trees("gpu-host", "/var/v3/trees")

    assert seen == {"host": "gpu-host", "timeout": TREE_TRANSFER_TIMEOUT_SECONDS}
    assert TREE_TRANSFER_TIMEOUT_SECONDS >= 3_600 and TREE_TRANSFER_TIMEOUT_SECONDS != 180.0
    assert trees.trees == "/var/v3/trees" and trees.schemes == ("https",)
    assert (trees.parallelism, trees.fetch_attempts) == (HOST_TRANSFER_PARALLELISM, FETCH_ATTEMPTS)


def test_the_bucket_presigns_real_sigv4_urls_offline_for_one_object_and_one_part():
    """boto3's presigning is a local computation; this checks the two calls the host relies on
    against the real library, with a fake credential and no request made."""
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url="https://account.r2.example", region_name="auto",
                          aws_access_key_id="AKIDSYNTHETIC",
                          aws_secret_access_key="SecretKeyNeverInAUrl",
                          config=Config(signature_version="s3v4",
                                        request_checksum_calculation="when_required"))
    bucket = S3Bucket(client, PRIVATE)

    get = bucket.presign_get("submissions/abc/model 1.safetensors", expires=PRESIGNED_TRANSFER_SECONDS)
    part = bucket.presign_upload_part("models/sha256/x/model.safetensors", "UPLOAD123", 7,
                                      expires=PRESIGNED_TRANSFER_SECONDS)

    for url in (get, part):
        query = parse_qs(urlsplit(url).query)
        assert query["X-Amz-Expires"] == [str(PRESIGNED_TRANSFER_SECONDS)]
        assert "SecretKeyNeverInAUrl" not in url                       # the secret is never in it
        assert hosttrees_module._safe_url(url, ("https",)) == url
    assert "model%201.safetensors" in get
    assert parse_qs(urlsplit(part).query)["uploadId"] == ["UPLOAD123"]
    assert parse_qs(urlsplit(part).query)["partNumber"] == ["7"]
    assert not any(name.startswith("x-amz-checksum") for name in parse_qs(urlsplit(part).query))


# --- promotion -------------------------------------------------------------------------------------


@needs_shell
def test_the_etag_parser_reads_the_last_header_block_whatever_the_transport():
    function = re.search(r"^etag_of\(\) \{.*?^\}$", hosttrees_module._UPLOAD_SCRIPT,
                         re.S | re.M).group(0)
    cases = {
        "HTTP/1.1 200 OK\r\nETag: \"0123456789abcdef0123456789abcdef\"\r\nContent-Length: 0\r\n\r\n":
            '"0123456789abcdef0123456789abcdef"',
        "HTTP/2 200\r\netag: \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\r\n\r\n": '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
        ("HTTP/1.1 100 Continue\r\nEtag: \"stale\"\r\n\r\n"
         "HTTP/1.1 200 OK\r\nETAG:   \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"  \r\n\r\n"):
            '"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"',
        "HTTP/1.1 100 Continue\r\netag: \"stale\"\r\n\r\nHTTP/1.1 200 OK\r\n\r\n": "",
        "HTTP/2 200\nx-etag-like: \"cccccccccccccccccccccccccccccccc\"\n": "",
    }
    for headers, expected in cases.items():
        done = subprocess.run(["bash", "-c", f'{function}\netag_of /dev/stdin'], input=headers,
                              capture_output=True, text=True, check=True)
        assert done.stdout == expected, headers


@pytest.fixture
def judged(tmp_path, world, http, monkeypatch):
    """A tree the host fetched and the controller verified, a public bucket, small parts."""
    monkeypatch.setattr(hosttrees_module, "PART_SIZE", 1024)      # every shard is several parts
    run = local_runner(tmp_path)
    public = FakeS3(PUBLIC)
    http.buckets[PUBLIC] = public
    public.base = world["private"].base
    fetch(world, host_for(run, world["trees"]))
    return {**world, "run": run, "public": public, "public_bucket": S3Bucket(public, PUBLIC)}


def promote(judged, host):
    return promote_submission(judged["dest"], judged["public_bucket"], manifest=judged["manifest"],
                              host=host)


@needs_shell
def test_a_king_promoted_through_the_host_lands_whole_with_sha256_metadata_and_manifest_last(judged):
    host = host_for(judged["run"], judged["trees"])
    manifest, public = judged["manifest"], judged["public"]

    prefix = promote(judged, host)

    assert prefix == public_model_prefix(manifest.sha256)
    for item in manifest.files:
        body, metadata = public.objects[prefix + item.path]
        assert body == (judged["source"] / item.path).read_bytes()
        assert metadata == {"sha256": item.sha256}
    assert public.writes[-1] == prefix + MANIFEST_NAME
    assert hashlib.sha256(public.objects[prefix + MANIFEST_NAME][0]).hexdigest() == manifest.sha256
    assert public.transfers == 0 and public.open_uploads() == []
    assert max(len(u["parts"]) for u in public.uploads.values()) >= 3, "multipart, really"
    assert SECRET not in judged["run"].argv_log.read_text()


@needs_shell
def test_a_tree_whose_bytes_changed_on_the_host_is_never_completed_and_its_uploads_are_aborted(
        judged):
    item = shard(judged["manifest"])
    (judged["dest"] / item.path).write_bytes(b"\x07" * item.size)      # same size, not the judged bytes

    with pytest.raises(StoreError, match="only verified bytes are published"):
        promote(judged, host_for(judged["run"], judged["trees"]))

    public = judged["public"]
    assert public.open_uploads() == []
    assert not any(key.endswith(".safetensors") for key in public.objects)
    assert public_model_prefix(judged["manifest"].sha256) + MANIFEST_NAME not in public.objects


@needs_shell
@pytest.mark.parametrize("etag", ["", '"ffffffffffffffffffffffffffffffff"', "garbage"],
                         ids=["no-etag", "etag-not-the-parts-md5", "unreadable-etag"])
def test_a_part_whose_etag_does_not_prove_its_bytes_aborts_every_open_upload_and_is_not_a_store_error(
        judged, http, etag):
    http.part_etag = etag

    with pytest.raises(HostTransferError) as caught:
        promote(judged, host_for(judged["run"], judged["trees"]))

    assert not isinstance(caught.value, StoreError)
    assert judged["public"].open_uploads() == []
    assert public_model_prefix(judged["manifest"].sha256) + MANIFEST_NAME not in judged["public"].objects


@needs_shell
def test_an_interrupted_host_promotion_resumes_past_files_already_in_place(judged, http):
    manifest = judged["manifest"]
    prefix = public_model_prefix(manifest.sha256)
    item = shard(manifest)
    http.forbidden.add(prefix + item.path)
    # A controller killed mid-promotion earlier left an upload open under the prefix.
    judged["public"].create_multipart_upload(Bucket=PUBLIC, Key=prefix + "orphan", Metadata={})

    with pytest.raises(HostTransferError, match="was not sent"):
        promote(judged, host_for(judged["run"], judged["trees"]))
    assert judged["public"].open_uploads() == []
    landed = {key for key in judged["public"].objects if key != prefix + MANIFEST_NAME}
    assert prefix + item.path not in landed and len(landed) == len(manifest.files) - 1
    assert prefix + MANIFEST_NAME not in judged["public"].objects

    http.forbidden.clear()
    http.requests.clear()
    assert promote(judged, host_for(judged["run"], judged["trees"])) == prefix
    sent = {unquote(urlsplit(path).path.split("/", 3)[3]) for path in http.requests}
    assert sent == {prefix + item.path}, "only the file that had not landed is sent again"


@needs_shell
def test_promotion_from_the_host_never_writes_inside_the_tree(judged):
    before = {path: path.stat().st_mtime_ns for path in judged["dest"].rglob("*")}

    promote(judged, host_for(judged["run"], judged["trees"]))

    assert {path: path.stat().st_mtime_ns for path in judged["dest"].rglob("*")} == before
