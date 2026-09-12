"""The submission path end to end, offline: owner issues, miner uploads, chain records the shot.

THIS IS THE PATH THAT DID NOT EXIST. Every piece of it shipped — `access` seals the envelope, `store`
uploads the tree, `chain` writes the ready signal, `admission` decides what is acceptable — and
nothing joined them, so a registered miner had no route to a submission and the owner had no way to
hand one out (`validator.main` wires a minter that always raises). What is tested here is the join:
`thirtyspokes-owner` and `thirtyspokes-miner`, driven with a mock chain, an in-memory bucket and a
local reference tree.

NO NETWORK, NO WALLET, NO BOTO3. The two seams that would reach a service — fetching the envelope
from the public mailbox and opening an R2 bucket — are parameters, so what runs here is the
production code path with only its edges replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_admission import _reference_tree
from test_store import FakeS3

from thirtyspokes.gateway.signing import Signer
from thirtyspokes.v3 import miner as miner_tool
from thirtyspokes.v3 import owner as owner_tool
from thirtyspokes.v3.access import AccessError, Mailbox, encode_ss58, scoped_credential
from thirtyspokes.v3.chain import MockChain
from thirtyspokes.v3.store import MANIFEST_NAME, S3Bucket

MINER_SEED = bytes.fromhex("11" * 32)
STRANGER_SEED = bytes.fromhex("22" * 32)
ENDPOINT = "https://acct.r2.cloudflarestorage.com"
BUCKET = "v3-submissions"


def address(seed: bytes) -> str:
    return encode_ss58(Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw())


MINER = address(MINER_SEED)
STRANGER = address(STRANGER_SEED)


def sign_with(seed: bytes):
    key = Ed25519PrivateKey.from_private_bytes(seed)
    return lambda payload: key.sign(payload).hex()


@pytest.fixture
def chain() -> MockChain:
    chain = MockChain()
    chain.advance(5_000_000)
    chain.register(MINER)
    return chain


@pytest.fixture
def bucket() -> S3Bucket:
    return S3Bucket(FakeS3(), BUCKET)


@pytest.fixture
def mailbox(tmp_path: Path) -> Mailbox:
    return Mailbox(tmp_path / "state" / "mailbox.json", Signer(),
                   owner_tool.owner_mint(endpoint=ENDPOINT, bucket=BUCKET, account_id="acct",
                                         access_key_id="parent-key",
                                         secret_access_key="parent-secret"))


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    return _reference_tree(tmp_path / "reference")


@pytest.fixture
def model(tmp_path: Path) -> Path:
    """A miner's own weights on the pinned architecture: the reference shape, different bytes.

    Different bytes matter — a tree byte-identical to the reference would pass admission for the
    wrong reason, and this file's happy path would then prove nothing about a real submission.
    """
    tree = _reference_tree(tmp_path / "model")
    shard = tree / "model-00001-of-00002.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] = (raw[-1] + 1) % 256
    shard.write_bytes(bytes(raw))
    return tree


def deliver(bucket: S3Bucket):
    """The owner publishes to the bucket; the miner fetches by key. The public mailbox, in memory."""
    def fetch(mailbox_url: str, key: str) -> bytes:
        assert mailbox_url == "https://mailbox.example"
        return bucket.get(key)
    return fetch


def submit(chain, mailbox, bucket, model, reference, **overrides):
    kwargs = dict(netuid=0, hotkey=MINER, model=model, reference=reference,
                  mailbox_url="https://mailbox.example",
                  owner_public_hex=mailbox.owner_public_hex, seed=MINER_SEED,
                  sign=sign_with(MINER_SEED), open_bucket=lambda credential: bucket,
                  fetch=deliver(bucket))
    return miner_tool.submit(chain, **{**kwargs, **overrides})


# --- the join, end to end -------------------------------------------------------------------


def test_a_miner_can_submit_once_the_owner_has_issued(chain, mailbox, bucket, model, reference):
    """The path that did not exist: issue -> poll -> upload -> commit, with nothing hand-wired."""
    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    assert issued.generation == 1
    assert issued.registration.prefix.startswith("submissions/")

    done = submit(chain, mailbox, bucket, model, reference)

    # The tree is in the bucket under the DERIVED prefix, with the commit marker in place.
    prefix = issued.registration.prefix
    assert prefix + MANIFEST_NAME in bucket.list(prefix)
    assert done.files == len(bucket.list(prefix)) - 1

    # And the chain names exactly that manifest.
    committed = miner_tool.already_committed(chain, MINER)
    assert committed is not None
    assert committed.manifest_sha256 == done.manifest_sha256
    assert committed.registration_id == issued.registration.registration_id
    # The manifest the validator will fetch is the one the commitment names.
    stored = json.loads(bucket.get(prefix + MANIFEST_NAME))
    assert stored["hotkey"] == MINER


def test_both_sides_derive_the_same_identity_without_agreeing_on_it(chain):
    """§7's whole reason for deriving from chain facts: neither side chooses the prefix."""
    assert (owner_tool.registration_of(chain, 0, MINER)
            == miner_tool.registration_of(chain, 0, MINER))


# --- what the miner is protected from -------------------------------------------------------


def test_a_tree_that_would_fail_admission_is_refused_before_anything_is_uploaded(
        chain, mailbox, bucket, model, reference):
    """THE POINT OF THE TOOL. A refusal after the upload would have spent the shot for nothing."""
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    (model / "config.json").write_text(json.dumps({"model_type": "not-the-reference"}))
    before = dict(bucket.client.objects)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the mailbox was polled before the tree was checked")

    with pytest.raises(miner_tool.MinerError, match="REFUSED at admission"):
        submit(chain, mailbox, bucket, model, reference, fetch=refuse)

    assert bucket.client.objects == before, "a refused submission uploaded bytes"
    assert miner_tool.already_committed(chain, MINER) is None


def test_a_second_submission_is_refused_against_the_chain_not_a_local_note(
        chain, mailbox, bucket, model, reference):
    """The shot is spent on chain, so a miner who lost local state still cannot submit twice."""
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    submit(chain, mailbox, bucket, model, reference)

    with pytest.raises(miner_tool.MinerError, match="already committed"):
        submit(chain, mailbox, bucket, model, reference)


def test_an_unregistered_hotkey_is_told_to_register_rather_than_handed_a_prefix(chain):
    with pytest.raises(miner_tool.MinerError, match="holds no uid"):
        miner_tool.registration_of(chain, 0, STRANGER)
    with pytest.raises(AccessError, match="holds no uid"):
        owner_tool.registration_of(chain, 0, STRANGER)


def test_the_commit_happens_only_after_the_upload_has_drained(
        chain, mailbox, bucket, model, reference):
    """`manifest.json` is the commit marker and the signal names its digest, so a signal written
    first would point the validator at a tree that is not there yet (§8, step 3)."""
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    seen: list[str] = []

    class Watched:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def put(self, key, body, **kwargs):
            seen.append("upload-complete" if key.endswith(MANIFEST_NAME) else "file")
            return self._inner.put(key, body, **kwargs)

    watched = Watched(bucket)
    original = chain.commit_ready
    chain.commit_ready = lambda *args, **kw: (seen.append("commit"), original(*args, **kw))[1]
    submit(chain, mailbox, watched, model, reference, open_bucket=lambda credential: watched)

    # Exactly one commit, and the marker before it. `seen[-2:]` alone would pass a submission that
    # committed first and committed AGAIN at the end, which is a real shape: an early commit that
    # the later one appears to repair still named a tree the validator could have fetched empty.
    assert seen.count("commit") == 1, f"the shot must be committed once, saw {seen}"
    assert seen[-1] == "commit"
    assert seen[-2] == "upload-complete"


# --- the owner's side ------------------------------------------------------------------------


def test_a_spent_hotkey_is_refused_a_further_credential(chain, mailbox, bucket, model, reference):
    """M6 exit 1, through the tool: the refusal is server-side, before a credential exists."""
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    submit(chain, mailbox, bucket, model, reference)
    mailbox.consume(chain.commitments()[0], owner_tool.registration_of(chain, 0, MINER))

    with pytest.raises(AccessError, match="permanently consumed"):
        owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)


def test_a_rotation_publishes_a_new_generation_at_a_new_key(chain, mailbox, bucket):
    """Rotation is more time for the ONE upload, never a second shot (§7, access.py point 3)."""
    first = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    second = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)

    assert (first.generation, second.generation) == (1, 2)
    assert first.registration.prefix == second.registration.prefix
    assert {first.mailbox_key, second.mailbox_key} <= set(bucket.client.objects)


def test_status_names_the_generations_the_operator_must_now_revoke(
        chain, mailbox, bucket, model, reference):
    """`Mailbox.consume` makes revocation the operator's obligation and could not say which
    credentials; unrevoked, the miner keeps write access to a prefix the chain already names."""
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    assert "REVOKE NOW" not in owner_tool.format_status(mailbox)

    submit(chain, mailbox, bucket, model, reference)
    mailbox.consume(chain.commitments()[0], owner_tool.registration_of(chain, 0, MINER))

    report = owner_tool.format_status(mailbox)
    assert "REVOKE NOW" in report
    assert "generations 1..2" in report


def test_a_credential_is_scoped_to_the_prefix_the_registration_derives(chain, mailbox, bucket):
    """A token scoped to somebody else's prefix is a stranger spending a hotkey's one shot."""
    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    envelope = miner_tool.open_credential(
        bucket.get(issued.mailbox_key), MINER_SEED, owner_public_hex=mailbox.owner_public_hex,
        registration=issued.registration, generation=1)
    assert envelope["allowed_prefix"] == issued.registration.prefix


def test_a_stranger_cannot_open_an_envelope_sealed_to_another_hotkey(chain, mailbox, bucket):
    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    with pytest.raises(Exception):
        miner_tool.open_credential(bucket.get(issued.mailbox_key), STRANGER_SEED,
                                   owner_public_hex=mailbox.owner_public_hex,
                                   registration=issued.registration, generation=1)


def test_an_envelope_not_signed_by_the_owner_is_refused(chain, mailbox, bucket, tmp_path):
    """Signed as well as sealed, so a miner cannot be pointed at somebody else's bucket."""
    impostor = Mailbox(tmp_path / "impostor.json", Signer(),
                       owner_tool.owner_mint(endpoint=ENDPOINT, bucket="not-ours",
                                             account_id="acct", access_key_id="k",
                                             secret_access_key="s"))
    issued = owner_tool.issue(chain, impostor, bucket.put, netuid=0, hotkey=MINER)
    with pytest.raises(AccessError, match="not signed by the owner"):
        miner_tool.open_credential(bucket.get(issued.mailbox_key), MINER_SEED,
                                   owner_public_hex=mailbox.owner_public_hex,
                                   registration=issued.registration, generation=1)


def test_a_neuron_with_no_registration_block_is_refused_rather_than_guessed(chain):
    """The block is hashed into the identity, so guessing derives a prefix the miner disagrees with.

    `chain.Neuron` deliberately survives a missing `BlockAtRegistration` for ownership questions;
    this asserts that the two identity derivations do not inherit that tolerance.
    """
    chain._registered_at.pop(MINER)
    with pytest.raises(AccessError, match="no registration block"):
        owner_tool.registration_of(chain, 0, MINER)
    with pytest.raises(miner_tool.MinerError, match="no registration block"):
        miner_tool.registration_of(chain, 0, MINER)


def test_the_scoped_credential_the_owner_mints_is_the_real_one():
    """`owner_mint` is wiring, so what it must not do is change the token. Same claims, same shape."""
    minted = owner_tool.owner_mint(endpoint=ENDPOINT, bucket=BUCKET, account_id="acct",
                                   access_key_id="parent-key",
                                   secret_access_key="parent-secret", ttl_seconds=600)("p/")
    direct = scoped_credential(endpoint=ENDPOINT, bucket=BUCKET, account_id="acct",
                               parent_access_key_id="parent-key",
                               parent_secret_access_key="parent-secret", prefix="p/",
                               ttl_seconds=600, issued_at_unix=int(minted.expires_at.timestamp())
                               - 600)
    assert minted == direct


def test_the_owner_signing_key_survives_the_process_that_issued(tmp_path, chain, bucket):
    """`Signer()` MINTS A NEW KEYPAIR when built with no argument, so a per-run identity would sign
    every envelope with a different key — and the miner verifies against one the owner published
    once. The failure would surface as `open_credential` refusing the owner's own envelopes.
    """
    state = tmp_path / "state"
    first = owner_tool.mailbox_signer(state)
    issued = owner_tool.issue(chain, Mailbox(state / "mailbox.json", first,
                                             owner_tool.owner_mint(
                                                 endpoint=ENDPOINT, bucket=BUCKET,
                                                 account_id="acct", access_key_id="k",
                                                 secret_access_key="s")),
                              bucket.put, netuid=0, hotkey=MINER)

    # A LATER, SEPARATE invocation of the tool: the miner opens against the published key.
    later = owner_tool.mailbox_signer(state)
    assert later.public_hex == first.public_hex
    miner_tool.open_credential(bucket.get(issued.mailbox_key), MINER_SEED,
                               owner_public_hex=later.public_hex,
                               registration=issued.registration, generation=1)
    assert (state / "mailbox-key.hex").stat().st_mode & 0o777 == 0o600


def test_a_miner_names_their_model_and_the_chain_commits_to_the_name(chain, mailbox, bucket, model,
                                                                     reference):
    """The name rides inside the signed manifest, so the digest on chain covers it."""
    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)

    done = submit(chain, mailbox, bucket, model, reference, model_name="my-router")

    written = json.loads(bucket.get(issued.registration.prefix + MANIFEST_NAME))
    assert written["model_name"] == "my-router" and written["protocol_version"] == 2
    assert done.manifest_sha256 == chain.commitments()[0].ready.manifest_sha256


def test_a_name_the_validator_would_refuse_is_refused_before_a_byte_moves(chain, mailbox, bucket,
                                                                          model, reference):
    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)

    with pytest.raises(Exception, match="reserved|lowercase"):
        submit(chain, mailbox, bucket, model, reference, model_name="thirtyspokes-genesis")

    assert not chain.commitments(), "the shot must still be there"
