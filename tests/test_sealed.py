"""The sealed request channel (`koth/sealed.py`).

The properties under test are the ones the privacy claim actually rests on, in order of how much
damage their absence would do:

  1. a ciphertext sealed for one image does NOT open under another — the image binding is
     cryptographic, not a convention two components agree to follow;
  2. the quote binding commits to the key, so a miner cannot substitute its own key and relay
     plaintext it can read;
  3. padding hides length, so a validator's probe is indistinguishable from a user request and the
     verification loop cannot be gamed by serving probes well and users badly.

Everything here is offline: no TEE, no network, no API.
"""

from __future__ import annotations

import pytest

from thirtyspokes.koth import sealed as S

IMAGE = "mrtd:aabbcc"
OTHER_IMAGE = "mrtd:ddeeff"
EPOCH = 4207


def enclave(suite=S.DEFAULT_SUITE, image=IMAGE, epoch=EPOCH):
    return S.EnclaveKey.generate(image, epoch, suite=suite)


# --- the basic contract -------------------------------------------------------------------------

@pytest.mark.parametrize("suite", ["x25519", "pq"])
def test_round_trip(suite):
    k = enclave(suite)
    msg = b'{"messages":[{"role":"user","content":"my private question"}]}'
    assert k.open(S.seal(k.announcement(), msg)) == msg


@pytest.mark.parametrize("suite", ["x25519", "pq"])
def test_the_relay_cannot_read_it(suite):
    """What the miner's host holds is the ciphertext. It must not contain the plaintext."""
    k = enclave(suite)
    msg = b"the user's actual words"
    ct = S.seal(k.announcement(), msg)
    assert msg not in ct
    assert b"user" not in ct


# --- binding: the properties that make it worth anything ----------------------------------------

def test_a_request_sealed_for_one_image_does_not_open_under_another():
    """THE CORE PROPERTY. A different approved image is still the wrong image: the measurement is
    inside the AEAD, so this fails closed rather than decrypting into the wrong context."""
    user_sees = enclave(image=IMAGE)
    ct = S.seal(user_sees.announcement(), b"secret")

    impostor = S.EnclaveKey(spec=user_sees.spec, _sk=user_sees._sk,
                            image_measurement=OTHER_IMAGE, epoch=EPOCH)
    with pytest.raises(Exception):
        impostor.open(ct)


def test_a_request_sealed_for_one_epoch_does_not_open_under_another():
    k = enclave(epoch=EPOCH)
    ct = S.seal(k.announcement(), b"secret")
    rotated = S.EnclaveKey(spec=k.spec, _sk=k._sk, image_measurement=IMAGE, epoch=EPOCH + 1)
    with pytest.raises(Exception):
        rotated.open(ct)


def test_another_enclave_key_cannot_open_it():
    a, b = enclave(), enclave()
    with pytest.raises(Exception):
        b.open(S.seal(a.announcement(), b"secret"))


def test_quote_binding_commits_to_the_key():
    k = enclave()
    assert S.verify_announcement(k.announcement(), k.report_data())


def test_a_swapped_key_fails_the_binding():
    """The attack this stops: a miner announces its OWN key alongside a genuine enclave quote, so
    it can decrypt everything while the client believes it verified an enclave."""
    real, attacker = enclave(), enclave()
    forged = dict(real.announcement())
    forged["public_key"] = attacker.public_bytes.hex()

    assert not S.verify_announcement(forged, real.report_data())


def test_a_swapped_image_or_epoch_fails_the_binding():
    k = enclave()
    for field, value in (("image_measurement", OTHER_IMAGE), ("epoch", EPOCH + 1)):
        tampered = dict(k.announcement())
        tampered[field] = value
        assert not S.verify_announcement(tampered, k.report_data()), field


def test_report_data_is_exactly_64_bytes():
    """`tdx.get_quote` takes 64 bytes; a longer digest would be silently truncated somewhere."""
    assert len(enclave().report_data()) == 64


# --- padding: what keeps probes indistinguishable -----------------------------------------------

def test_padding_hides_length_within_a_bucket():
    k = enclave()
    a = S.seal(k.announcement(), b"x" * 20)
    b = S.seal(k.announcement(), b"y" * 400)
    assert len(a) == len(b), "two small requests must be the same size on the wire"
    assert k.open(a) == b"x" * 20 and k.open(b) == b"y" * 400


def test_a_long_request_lands_in_a_larger_bucket():
    k = enclave()
    small = S.seal(k.announcement(), b"x" * 20)
    large = S.seal(k.announcement(), b"x" * 5000)
    assert len(large) > len(small), "bucketing is coarse, not constant-size"


def test_bucket_ladder_is_monotone_and_covers_oversize():
    assert S.bucket_for(1) == S.BUCKETS[0]
    assert S.bucket_for(S.BUCKETS[0]) == S.BUCKETS[0]
    assert S.bucket_for(S.BUCKETS[0] + 1) == S.BUCKETS[1]
    assert S.bucket_for(S.BUCKETS[-1] * 3 - 5) == S.BUCKETS[-1] * 3


@pytest.mark.parametrize("n", [0, 1, 511, 512, 513, 5000, 300000])
def test_pad_unpad_round_trip(n):
    assert S.unpad(S.pad(b"z" * n)) == b"z" * n


def test_unpad_rejects_a_lying_length_prefix():
    import struct
    assert S.unpad(S.pad(b"ok")) == b"ok"
    with pytest.raises(ValueError, match="exceeds"):
        S.unpad(struct.pack(">I", 9999) + b"short")
    with pytest.raises(ValueError, match="too short"):
        S.unpad(b"ab")


# --- the key must not be extractable ------------------------------------------------------------

def test_the_private_key_has_no_serialisation_path():
    """Not a cryptographic guarantee — a deliberate absence. The private key exists only in enclave
    memory, and there is no legitimate reason for this class to be able to write it out."""
    k = enclave()
    assert not hasattr(k, "private_bytes")
    assert "private" not in k.announcement()
    assert k.public_bytes.hex() == k.announcement()["public_key"]


def test_pq_suite_costs_more_bytes_and_is_worth_naming():
    """Documents the measured trade rather than asserting a preference."""
    msg = b"q" * 100
    classical = len(S.seal(enclave("x25519").announcement(), msg))
    pq = len(S.seal(enclave("pq").announcement(), msg))
    assert pq > classical
    assert pq - classical < 2000, "the PQ tax should be ~1KB, not unbounded"


# --- the response leg ---------------------------------------------------------------------------

def test_the_reply_leg_round_trips():
    rk = S.ReplyKey.generate()
    ct = S.seal_reply(rk.public(), b"the answer", IMAGE, EPOCH)
    assert rk.open(ct, IMAGE, EPOCH) == b"the answer"


def test_a_reply_is_padded_to_a_bucket_like_a_request():
    """Response lengths leak as much as request lengths — an observer watching a cascade could
    otherwise read answer size off the wire."""
    rk = S.ReplyKey.generate()
    sizes = {len(S.seal_reply(rk.public(), b"x" * n, IMAGE, EPOCH)) for n in (10, 100, 400)}
    assert len(sizes) == 1


def test_a_request_ciphertext_cannot_be_replayed_as_a_response():
    """The two legs use distinct labels, so direction is part of the AEAD context."""
    rk = S.ReplyKey.generate()
    ct = S.seal_reply(rk.public(), b"answer", IMAGE, EPOCH)
    with pytest.raises(Exception):
        rk.spec.suite().decrypt(ct, rk._sk, info=S.info_for(IMAGE, EPOCH))


def test_a_reply_does_not_open_under_a_different_image_or_epoch():
    rk = S.ReplyKey.generate()
    ct = S.seal_reply(rk.public(), b"answer", IMAGE, EPOCH)
    for image, epoch in [("other-image", EPOCH), (IMAGE, EPOCH + 1)]:
        with pytest.raises(Exception):
            rk.open(ct, image, epoch)


def test_a_reply_key_is_fresh_every_time():
    """Ephemeral per request: one compromised reply key must expose exactly one answer."""
    assert S.ReplyKey.generate().public() != S.ReplyKey.generate().public()
