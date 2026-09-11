"""The mailbox's properties (docs/WHITEPAPER.md §7, the build plan M6).

Every test here is one of two things: an M6 exit criterion, or a property the ported teutonic
pattern carries that is invisible from outside and would be lost by re-derivation — the one-shot
keyed on the hotkey rather than the registration, rotation before the shot is spent, the ledger
surviving a restart, the envelope sealed to one hotkey and not another.

NOTHING HERE TOUCHES R2, CLOUDFLARE OR A CHAIN. The credential minter is pure (a signed JWT, no
network call), the mailbox is a local file, and the one place a real service would be reached — the
publishing of a ciphertext to a public bucket — is a two-line concern of the caller's, not of this
module's.

WHAT IS NOT TESTED HERE, DELIBERATELY: the `r2ready:v1` commitment ENCODING. It belongs to
`v3/chain.py`, where a commitment is read, and `test_chain.py` already owns its round trip, its
128-byte fit and its refusals. A second copy of those assertions here would be a second pin on the
one string a miner writes and the validator looks for, and two pins drift. What this file tests is
what the mailbox adds on top of it: whether a commitment spends the right hotkey's shot.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from thirtyspokes.gateway.signing import Signer
from thirtyspokes.v3.access import (
    CREDENTIAL_SCOPE,
    REVOCATION_EVENT,
    AccessError,
    Credential,
    Mailbox,
    Registration,
    credential_from_envelope,
    decode_ss58,
    encode_ss58,
    mailbox_key,
    open_credential,
    scoped_credential,
    seal,
    unseal,
    verify_hotkey,
)
from thirtyspokes.v3.chain import Commitment, ReadySignal

MINER_SEED = bytes.fromhex("11" * 32)
OTHER_SEED = bytes.fromhex("22" * 32)
ENDPOINT = "https://accountid.r2.cloudflarestorage.com"
BUCKET = "v3-submissions"


def address(seed: bytes) -> str:
    return encode_ss58(Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw())


MINER = address(MINER_SEED)
OTHER = address(OTHER_SEED)


def registration(hotkey: str = MINER, *, uid: int = 7, block: int = 5_000_000) -> Registration:
    return Registration(netuid=99, uid=uid, hotkey=hotkey, registration_block=block)


def mint(prefix: str, *, ttl: int = 86_400) -> Credential:
    """Stands in for the owner's Cloudflare call, but mints the REAL prefix-scoped token: the minter
    is pure, so the thing under test is the production one and only the caller is a test."""
    return scoped_credential(endpoint=ENDPOINT, account_id="acct", bucket=BUCKET, prefix=prefix,
                             parent_access_key_id="parent-key", ttl_seconds=ttl,
                             parent_secret_access_key="parent-secret")


def mailbox(tmp_path: Path, signer: Signer | None = None) -> Mailbox:
    return Mailbox(tmp_path / "mailbox.json", signer or Signer(), mint)


def commitment(reg: Registration, manifest: str, *, hotkey: str | None = None,
               block: int = 5_000_100) -> Commitment:
    """A ready signal as the validator reads it off the chain (`chain.read_commitments`)."""
    return Commitment(hotkey=hotkey or reg.hotkey, block=block,
                      ready=ReadySignal(registration_id=reg.registration_id,
                                        manifest_sha256=manifest))


def test_ss58_round_trips_and_matches_the_real_substrate_encoding():
    """The address is the seal's recipient, so an encoding that is merely self-consistent would
    produce envelopes only this repository can address. Cross-checked against `bittensor_wallet`
    when it is installed — it is in the `chain` extra, not `dev`, hence the skip."""
    public = Ed25519PrivateKey.from_private_bytes(MINER_SEED).public_key().public_bytes_raw()
    assert decode_ss58(encode_ss58(public)) == public

    wallet = pytest.importorskip("bittensor_wallet")
    real = wallet.Keypair.create_from_seed(MINER_SEED.hex(), crypto_type=0)
    assert encode_ss58(public) == real.ss58_address
    assert decode_ss58(real.ss58_address) == bytes(real.public_key)


def test_a_mistyped_address_is_refused_rather_than_decoded_into_an_unopenable_envelope():
    swapped = MINER[:10] + ("X" if MINER[10] != "X" else "Y") + MINER[11:]
    with pytest.raises(AccessError, match="checksum|base58"):
        decode_ss58(swapped)


def test_only_the_named_hotkeys_secret_key_opens_a_sealed_envelope():
    """The mailbox is public: a credential readable by its fetcher is a write token to a stranger's
    submission prefix, which is somebody else spending a hotkey's one shot (§7)."""
    ciphertext = seal(b"credential", MINER)
    assert unseal(ciphertext, MINER_SEED) == b"credential"
    with pytest.raises(AccessError, match="not addressed to this hotkey"):
        unseal(ciphertext, OTHER_SEED)


def test_a_tampered_envelope_is_refused_rather_than_partially_read():
    """Both public keys are bound into the key derivation AND authenticated as the AEAD's associated
    data, so an edited ciphertext fails as a whole — there is no prefix of it that decrypts."""
    ciphertext = bytearray(seal(b"credential", MINER))
    ciphertext[-1] ^= 0x01
    with pytest.raises(AccessError):
        unseal(bytes(ciphertext), MINER_SEED)
    with pytest.raises(AccessError, match="too short"):
        unseal(b"short", MINER_SEED)


def test_a_second_credential_request_is_refused_by_the_owners_ledger_not_by_the_client(tmp_path):
    """M6 exit 1. The refusal happens inside the owner's `Mailbox` before a credential exists — a
    check in the miner's client would be a check in the miner's code."""
    box = mailbox(tmp_path)
    reg = registration()
    box.issue(reg)
    box.consume(commitment(reg, "ab" * 32), reg)

    with pytest.raises(AccessError, match="permanently consumed"):
        box.issue(reg)


def test_the_one_shot_is_keyed_on_the_hotkey_so_re_registering_buys_nothing(tmp_path):
    """§7: *a hotkey with any prior entry is refused forever*. A `registration_id` carries the
    registration block, so keyed on registration the one-shot would price a second submission at one
    registration burn."""
    box = mailbox(tmp_path)
    first = registration(block=5_000_000)
    box.issue(first)
    box.consume(commitment(first, "ab" * 32), first)

    reregistered = registration(uid=41, block=6_000_000)
    assert reregistered.registration_id != first.registration_id
    with pytest.raises(AccessError, match="permanently consumed"):
        box.issue(reregistered)


def test_a_credential_may_rotate_until_the_shot_is_spent(tmp_path):
    """At ~72 GB over hours an interrupted upload is the common case (M6 exit 2), so a credential
    that expires mid-upload is ordinary. If asking for one spent the shot, the owner's own TTL would
    destroy submissions — so rotation issues generation N+1 against the IDENTICAL derived prefix,
    which is more access to the one submission rather than a second one."""
    box = mailbox(tmp_path)
    reg = registration()
    first_key, first = box.issue(reg)
    second_key, second = box.issue(reg)

    assert first_key == mailbox_key(reg.registration_id, 1)
    assert second_key == mailbox_key(reg.registration_id, 2)
    one = open_credential(first, MINER_SEED, owner_public_hex=box.owner_public_hex,
                          registration=reg, generation=1)
    two = open_credential(second, MINER_SEED, owner_public_hex=box.owner_public_hex,
                          registration=reg, generation=2)
    assert one["allowed_prefix"] == two["allowed_prefix"] == reg.prefix


def test_the_ledger_survives_a_restart_because_a_forgotten_shot_is_a_free_one(tmp_path):
    """A one-shot held in memory hands every hotkey a fresh shot on every owner deploy."""
    signer = Signer()
    box = mailbox(tmp_path, signer)
    reg = registration()
    box.issue(reg)
    box.consume(commitment(reg, "cd" * 32), reg)

    restarted = Mailbox(tmp_path / "mailbox.json", signer, mint)
    assert restarted.spent(MINER)
    assert restarted.submission(MINER).manifest_sha256 == "cd" * 32
    with pytest.raises(AccessError, match="permanently consumed"):
        restarted.issue(reg)


def test_re_reading_the_same_ready_signal_is_idempotent_but_a_new_one_is_refused(tmp_path):
    """The validator re-reads chain commitments every window (§8 step 3), so it sees this exact
    signal again in every window the hotkey stays registered — a `consume` that raised on the second
    sighting would stop the validator on its own success. A DIFFERENT manifest from a spent hotkey
    is the second submission §7 forbids, and also the post-signal swap."""
    box = mailbox(tmp_path)
    reg = registration()
    box.issue(reg)
    signal = commitment(reg, "ab" * 32)

    assert box.consume(signal, reg) == box.consume(signal, reg)
    swap = commitment(reg, "ef" * 32, block=5_000_200)
    with pytest.raises(AccessError, match="permanently consumed"):
        box.consume(swap, reg)


def test_a_ready_signal_committed_by_a_stranger_does_not_spend_the_owners_registration(tmp_path):
    box = mailbox(tmp_path)
    reg = registration()
    stranger = commitment(reg, "ab" * 32, hotkey=OTHER)
    with pytest.raises(AccessError, match="does not own it"):
        box.consume(stranger, reg)
    assert not box.spent(MINER)


def test_a_commitment_naming_another_registration_does_not_spend_this_hotkeys_shot(tmp_path):
    """M6 exit 3's other half, and the half that lives here rather than in `chain.py`: the encoding
    binds `(registration, manifest)`, and the ledger is what makes the registration MEAN something —
    the commitment must name the registration whose prefix this hotkey was issued a credential for,
    or a hotkey could commit to a tree sitting under somebody else's prefix."""
    box = mailbox(tmp_path)
    reg = registration()
    elsewhere = Commitment(hotkey=MINER, block=5_000_100,
                           ready=ReadySignal(registration_id=registration(uid=41).registration_id,
                                             manifest_sha256="ab" * 32))

    with pytest.raises(AccessError, match="different registration"):
        box.consume(elsewhere, reg)
    assert not box.spent(MINER)


def test_the_credential_is_scoped_to_the_prefix_the_registration_derives(tmp_path):
    """The prefix is never a caller's argument: `issue` derives it from finalized chain data and
    hands it to the minter, so 'one miner cannot write into another's submission' is a property of
    the token rather than a promise about the client."""
    seen: list[str] = []

    def recording_mint(prefix: str) -> Credential:
        seen.append(prefix)
        return mint(prefix)

    reg = registration()
    box = Mailbox(tmp_path / "mailbox.json", Signer(), recording_mint)
    _key, ciphertext = box.issue(reg)
    envelope = open_credential(ciphertext, MINER_SEED, owner_public_hex=box.owner_public_hex,
                               registration=reg, generation=1)

    assert seen == [reg.prefix] == [f"submissions/{reg.registration_id}/"]
    claims = _jwt_claims(credential_from_envelope(envelope))
    assert claims["paths"] == {"objectPaths": [], "prefixPaths": [reg.prefix]}
    assert claims["scope"] == CREDENTIAL_SCOPE and claims["bucket"] == BUCKET


def test_a_prefix_that_does_not_end_in_a_slash_is_refused():
    """`submissions/<id>` without the terminating slash also matches `submissions/<id>evil/`."""
    with pytest.raises(AccessError, match="slash-terminated"):
        scoped_credential(endpoint=ENDPOINT, account_id="a", parent_access_key_id="k",
                          parent_secret_access_key="s", bucket=BUCKET, prefix="submissions/abc")


def test_a_miner_refuses_an_envelope_that_is_not_the_owners_or_not_theirs(tmp_path):
    """Ported from `get_upload_auth.validate_envelope`: each check is a way an upload can be lost or
    misdirected, and every one is cheaper here than after 72 GB."""
    box = mailbox(tmp_path)
    reg = registration()
    _key, ciphertext = box.issue(reg)

    with pytest.raises(AccessError, match="not signed by the owner"):
        open_credential(ciphertext, MINER_SEED, owner_public_hex=Signer().public_hex,
                        registration=reg, generation=1)
    with pytest.raises(AccessError, match="credential_generation"):
        open_credential(ciphertext, MINER_SEED, owner_public_hex=box.owner_public_hex,
                        registration=reg, generation=2)
    with pytest.raises(AccessError, match="uid"):
        open_credential(ciphertext, MINER_SEED, owner_public_hex=box.owner_public_hex,
                        registration=registration(uid=8), generation=1)


def test_the_envelope_tells_the_miner_when_the_shot_is_spent(tmp_path):
    """`revocation_event: finalized_ready_signal` is the mechanism's promise written where the miner
    reads it — the one-shot is consumed by the on-chain signal, not by the upload or the credential,
    so a miner knows exactly which action is irreversible before taking it."""
    box = mailbox(tmp_path)
    reg = registration()
    _key, ciphertext = box.issue(reg)
    envelope = open_credential(ciphertext, MINER_SEED, owner_public_hex=box.owner_public_hex,
                               registration=reg, generation=1)

    assert envelope["revocation_event"] == REVOCATION_EVENT == "finalized_ready_signal"
    assert envelope["submission_policy"] == "one_per_hotkey"


def test_an_expired_credential_is_never_published_and_never_accepted(tmp_path):
    box = Mailbox(tmp_path / "mailbox.json", Signer(),
                  lambda prefix: Credential(endpoint=ENDPOINT, bucket=BUCKET, access_key_id="k",
                                            secret_access_key="s", session_token="t",
                                            expires_at=datetime.now(timezone.utc) - timedelta(1)))
    with pytest.raises(AccessError, match="already expired"):
        box.issue(registration())


def test_a_hotkey_signature_verifies_from_either_encoding_its_tools_produce():
    """`bittensor`'s `Keypair.sign` is usually hexed by an operator and teutonic's manifest writer
    base64s it; a correct signature refused for its encoding would spend a shot on a format."""
    import base64

    message = b"v3 manifest"
    signature = Ed25519PrivateKey.from_private_bytes(MINER_SEED).sign(message)
    for encoded in (signature.hex(), "0x" + signature.hex(),
                    base64.b64encode(signature).decode()):
        verify_hotkey(MINER, message, encoded)
    with pytest.raises(AccessError):
        verify_hotkey(OTHER, message, signature.hex())


def _jwt_claims(credential: Credential) -> dict:
    import base64

    token = base64.b64decode(credential.session_token).decode().removeprefix("jwt/")
    claims = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))


def test_a_concurrent_writer_does_not_erase_an_issued_generation(tmp_path):
    """THE OWNER TOOL AND THE VALIDATOR DAEMON BOTH WRITE THIS FILE, from different processes.

    Measured 2026-09-02, and reachable only once `thirtyspokes-owner` gave the ledger a second
    writer: the daemon loads it at startup and `_persist` writes its whole in-memory view, so an
    `issue` landing before the daemon's next `consume` used to vanish. The rotation then re-minted
    generation 1 over a mailbox key a miner may already be polling, and `Submission.generations`
    under-counted — which is the number the operator revokes against, so a live write credential for
    an already-committed prefix would have been left un-revoked.
    """
    path = tmp_path / "mailbox.json"
    daemon = Mailbox(path, Signer(), mint)          # loaded once, as `run_forever` loads it
    owner = Mailbox(path, Signer(), mint)           # the separate `thirtyspokes-owner` process

    other = registration(OTHER, uid=9)
    first, _ = owner.issue(other)
    daemon.consume(commitment(registration(), "ab" * 32), registration())

    rotated, _ = Mailbox(path, Signer(), mint).issue(other)
    assert first != rotated, "a rotation must advance the generation, not overwrite the same key"
    assert mailbox_key(other.registration_id, 2) == rotated


def test_generations_counts_an_issue_made_after_the_consuming_process_started(tmp_path):
    """`Submission.generations` is the operator's revocation list, so it is read under the lock."""
    path = tmp_path / "mailbox.json"
    daemon = Mailbox(path, Signer(), mint)
    owner = Mailbox(path, Signer(), mint)
    owner.issue(registration())
    owner.issue(registration())

    spent = daemon.consume(commitment(registration(), "cd" * 32), registration())
    assert spent.generations == 2
