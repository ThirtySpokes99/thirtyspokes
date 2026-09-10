"""The only code `scripts/live_smoke.py` adds to the daemon — its bucket, and its wait.

The smoke itself is not under test here and could not be: it is `Validator.run_window` over live
seams, and every property of that window already has a test in `test_validator.py` against far
sides that do not cost money. What IS new is the two pieces the script had to write to reach those
seams, and both can fail in a way that reads as a daemon bug during a paid run:

* `DirectoryClient` stands where R2 stands. It hard-links instead of copying, so the question that
  matters is whether `fetch_submission`'s digest checks still bite through a link — if they did not,
  the smoke would certify a store path that never verified anything.
* `wait_for` is what keeps an empty port from being published as a challenger who could not serve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from thirtyspokes.v3.access import Registration, encode_ss58
from thirtyspokes.v3.chain import Commitment, ReadySignal
from thirtyspokes.v3.store import (S3Bucket, StoreError, build_manifest, fetch_submission,
                                    upload_tree)

SEED = bytes.fromhex("33" * 32)
KEY = Ed25519PrivateKey.from_private_bytes(SEED)
HOTKEY = encode_ss58(KEY.public_key().public_bytes_raw())
REGISTRATION = Registration(netuid=99, uid=11, hotkey=HOTKEY, registration_block=5_000_000)
BUCKET = "v3-smoke"


def _load_script():
    """Import the script by path — `scripts/` is not a package, and the smoke is a program."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_smoke.py"
    spec = importlib.util.spec_from_file_location("live_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "qwen3_next"}', encoding="utf-8")
    (root / "model-00001.safetensors").write_bytes(b"shard-one" * 64)
    (root / "model-00002.safetensors").write_bytes(b"shard-two" * 64)
    return root


def bucket_over(root: Path) -> S3Bucket:
    return S3Bucket(smoke.DirectoryClient(root), BUCKET)


def commitment_for(manifest) -> Commitment:
    return Commitment(hotkey=HOTKEY, block=5_000_100,
                      ready=ReadySignal(registration_id=REGISTRATION.registration_id,
                                        manifest_sha256=manifest.sha256))


# --- the bucket ----------------------------------------------------------------------------------


def test_the_production_bucket_round_trips_over_a_directory(tmp_path):
    """`S3Bucket` is the class under test; only the client differs, exactly as in `test_store`."""
    bucket = bucket_over(tmp_path / "bucket")
    bucket.put("v3/window/1.json", b"{}", digest="ab" * 32, content_type="application/json")

    assert bucket.get("v3/window/1.json") == b"{}"
    assert bucket.head("v3/window/1.json")["Metadata"]["sha256"] == "ab" * 32
    assert bucket.list("v3/") == {"v3/window/1.json": 2}
    bucket.delete("v3/window/1.json")
    assert bucket.list("v3/") == {}


def test_a_listing_names_only_its_own_prefix(tmp_path):
    """`fetch_submission` refuses a prefix carrying a file the manifest does not name, so a listing
    that leaked a neighbouring prefix would refuse an honest submission."""
    bucket = bucket_over(tmp_path / "bucket")
    bucket.put("submissions/a/one", b"1")
    bucket.put("submissions/ab/two", b"2")

    assert set(bucket.list("submissions/a/")) == {"submissions/a/one"}


def test_a_linked_tree_still_arrives_byte_for_byte(tmp_path):
    """The whole point of linking is that it is free; the point of this test is that it is also the
    same bytes, checked by the production `fetch_submission` rather than by comparing files here."""
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)

    dest = tmp_path / "duel"
    fetched = fetch_submission(bucket, dest, commitment=commitment_for(manifest),
                               registration=REGISTRATION)

    assert fetched.sha256 == manifest.sha256
    for item in manifest.files:
        assert (dest / item.path).read_bytes() == (source / item.path).read_bytes()


def test_a_shard_swapped_in_the_bucket_is_refused_through_the_link(tmp_path):
    """The failure a linking client could hide: if `download_file` handed back the file the caller
    already had rather than the one the bucket holds, a swapped shard would pass every check."""
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    swapped = tmp_path / "bucket" / (REGISTRATION.prefix + "model-00002.safetensors")
    swapped.unlink()
    swapped.write_bytes(b"not-the-committed-shard" * 25)

    with pytest.raises(StoreError, match="model-00002.safetensors hashes to"):
        fetch_submission(bucket, tmp_path / "duel", commitment=commitment_for(manifest),
                         registration=REGISTRATION)


def test_a_resumed_upload_writes_over_the_key_it_is_unsure_of(tmp_path):
    """M6 exit 5's resume re-sends any file whose stored digest it cannot confirm — onto a key that
    already exists. `os.link` onto an existing path raises `FileExistsError`, so a client that did
    not clear the target first would turn every resume into a crash."""
    source = tree(tmp_path / "miner")
    client = smoke.DirectoryClient(tmp_path / "bucket")
    bucket = S3Bucket(client, BUCKET)
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    # A transfer that died before the marker, leaving one shard on the bucket with no digest to
    # confirm it by — which is exactly the file the resume must send again.
    bucket.delete(REGISTRATION.prefix + "manifest.json")
    client.meta.pop(REGISTRATION.prefix + "model-00002.safetensors")

    report = upload_tree(source, bucket, REGISTRATION.prefix, manifest)

    assert report.uploaded == ("model-00002.safetensors",) and len(report.skipped) == 2
    assert fetch_submission(bucket, tmp_path / "duel", commitment=commitment_for(manifest),
                            registration=REGISTRATION).sha256 == manifest.sha256


def test_a_copying_filesystem_is_the_same_bucket(tmp_path, monkeypatch):
    """`os.link` fails across filesystems and the client falls back to copying. That fallback is the
    path a real deployment takes whenever the bucket and the tree are not on one mount, so it cannot
    be the untested branch."""
    monkeypatch.setattr(smoke.os, "link",
                        lambda src, dest: (_ for _ in ()).throw(OSError("cross-device link")))
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)

    dest = tmp_path / "duel"
    assert fetch_submission(bucket, dest, commitment=commitment_for(manifest),
                            registration=REGISTRATION).sha256 == manifest.sha256
    assert (dest / "model-00001.safetensors").read_bytes() == \
        (source / "model-00001.safetensors").read_bytes()


# --- the wait ------------------------------------------------------------------------------------


class FakeModels:
    """`/v1/models` as an OpenAI-compatible server answers it, one scripted reply per call."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.replies
        if isinstance(reply, Exception):
            raise reply
        return _Body(reply)


class _Body:
    def __init__(self, ids) -> None:
        self.ids = ids

    def json(self):
        return {"data": [{"id": name} for name in self.ids]}


def _client(server, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: server)
    monkeypatch.setattr(smoke.time, "sleep", lambda seconds: None)


def test_the_wait_returns_as_soon_as_the_artifact_is_reported(monkeypatch):
    server = FakeModels([], ["v3-hot@digest"])
    _client(server, monkeypatch)

    smoke.wait_for("http://x/v1", "v3-hot@digest", seconds=60.0)

    assert server.calls == 2                       # it really did wait out the empty first answer


def test_a_port_that_never_serves_the_artifact_says_what_it_served_instead(monkeypatch):
    """The message is the point: `local_serving` would raise a REFUSAL the daemon records against
    the challenger, so an operator who launched the wrong name must be told that here instead of
    reading it as a lost duel."""
    server = FakeModels(*[["v3-someone-else@digest"]] * 50)
    _client(server, monkeypatch)
    monkeypatch.setattr(smoke.time, "monotonic", iter([0.0, 1.0, 999.0]).__next__)

    with pytest.raises(SystemExit, match="v3-someone-else@digest"):
        smoke.wait_for("http://x/v1", "v3-hot@digest", seconds=60.0)


def test_a_port_that_is_not_up_at_all_is_not_reported_as_a_wrong_name(monkeypatch):
    """A connection error must not leave the previous answer in the message — "it reported nothing"
    and "it reported the wrong model" send an operator to different places."""
    import httpx
    server = FakeModels(["v3-stale@digest"], httpx.ConnectError("refused"))
    _client(server, monkeypatch)
    monkeypatch.setattr(smoke.time, "monotonic", iter([0.0, 1.0, 2.0, 999.0]).__next__)

    with pytest.raises(SystemExit, match="it reported nothing"):
        smoke.wait_for("http://x/v1", "v3-hot@digest", seconds=60.0)



# --- the pass condition -------------------------------------------------------------------------------


from types import SimpleNamespace as _NS


def _reveal(outcomes, arms=()):
    return _NS(report=_NS(outcomes=outcomes, crowned=None), arms=list(arms))


def _outcome(status, arm="scored"):
    return _NS(hotkey="5Cabcdefghijklmnop", status=status, detail="why", arm=arm)


def test_a_deferred_only_window_is_not_exit_eleven():
    """The first live run: power gate refused at 8 tasks, challenger DEFERRED, king never ran, the
    served conductor got no request — and the smoke printed the exit-11 line. Never again."""
    problem = smoke.verdict_of(_reveal([_outcome("deferred", arm=None)]))
    assert problem and "no challenger was duelled" in problem and "raise --tasks" in problem


def test_a_duelled_challenger_with_a_scored_arm_is_exit_eleven():
    assert smoke.verdict_of(_reveal([_outcome("deferred", arm=None), _outcome("duelled")])) is None


def test_a_duelled_challenger_without_an_arm_is_still_not_exit_eleven():
    assert "no scored arm" in smoke.verdict_of(_reveal([_outcome("duelled", arm=None)]))


def test_an_unreconciled_arm_fails_before_anything_else_is_read():
    bad = _NS(arm="chal-x", scoreable=False, audit=_NS(metered_usd=0.0))
    assert "did not reconcile" in smoke.verdict_of(_reveal([_outcome("duelled")], arms=[bad]))



# --- a restart is not a second submission -------------------------------------------------------------


def test_a_restart_resumes_the_committed_submission_instead_of_uploading_twice(tmp_path):
    """Run 2b died in `upload_tree` on a prefix that already carried a manifest — the store's own
    one-submission rule, working. The smoke must recognise its own committed tree and skip."""
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())

    assert smoke.already_committed(bucket, REGISTRATION.prefix, manifest) is None
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    assert smoke.already_committed(bucket, REGISTRATION.prefix, manifest) == manifest.sha256


def test_a_different_tree_under_the_same_hotkey_is_refused_as_a_second_submission(tmp_path):
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    (source / "model-00002.safetensors").write_bytes(b"retrained" * 64)
    other = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())

    with pytest.raises(SystemExit, match="DIFFERENT submission.*One hotkey, one submission"):
        smoke.already_committed(bucket, REGISTRATION.prefix, other)


def test_a_verified_tree_is_resident_and_not_pulled_again_on_the_next_window(tmp_path):
    """The rehearsal of 2026-09-08 found `fetch_submission` re-pulling every ~70 GB tree on every
    window, where the design has a tree pulled once and resident (§8b.1, §11-7). Once every file
    has been hashed against the committed manifest, a marker beside the tree names that digest and
    the next window skips both the pull and the hash pass; a file that vanished is pulled again,
    alone; a marker for another digest buys nothing."""
    source = tree(tmp_path / "miner")
    bucket = bucket_over(tmp_path / "bucket")
    manifest = build_manifest(source, REGISTRATION, lambda body: KEY.sign(body).hex())
    upload_tree(source, bucket, REGISTRATION.prefix, manifest)
    pulls: list[str] = []
    pull = bucket.download_file
    bucket.download_file = lambda key, path: (pulls.append(key), pull(key, path))[1]
    dest, marker = tmp_path / "duel", tmp_path / "duel.verified"

    def fetch():
        return fetch_submission(bucket, dest, commitment=commitment_for(manifest),
                                registration=REGISTRATION)

    assert fetch().sha256 == manifest.sha256
    assert len(pulls) == len(manifest.files)
    assert marker.read_text() == manifest.sha256
    assert not (dest / ".verified").exists(), "the marker must never sit inside the tree"

    fetch()                                                   # the next window
    assert len(pulls) == len(manifest.files), "a resident, verified tree was pulled again"

    (dest / "model-00002.safetensors").unlink()
    fetch()
    assert len(pulls) == len(manifest.files) + 1 and pulls[-1].endswith("model-00002.safetensors")

    marker.write_text("00" * 32)
    fetch()
    assert len(pulls) == len(manifest.files) + 1, "intact files were re-pulled under a stale marker"
    assert marker.read_text() == manifest.sha256
