"""The client's trust decision (`serve/client.py`) and the enclave's HTTP surface.

The client is where the privacy claim is decided, so most of these are attacks. The one that
matters most is `test_a_real_quote_with_a_swapped_key_is_refused`: a miner can hold a genuine,
fully-verifiable quote from a genuine approved enclave and still hand out its own public key. Every
other check in the stack passes in that scenario. Only the binding catches it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thirtyspokes.koth import sealed
from thirtyspokes.koth import tdx as _tdx_module
from thirtyspokes.serve import client as C
from thirtyspokes.serve.confidential import ConfidentialServer

# MRTD and the image identity are DIFFERENT THINGS, and keeping them apart here is the point.
# MRTD on a cloud TD is the virtual firmware, shared by every guest on the platform; the image is
# identified by the boot RTMRs as well. A fixture that used one value for both would pass whether
# or not the client checked the RTMRs at all.
#
# Realistic values also matter for a second reason: an earlier version used "mrtd:approved" with a
# hand-rolled fake verdict exposing `.mrtd`, while the real TDXVerdict field is `.mr_td`. The client
# read an attribute that did not exist, silently got None, and the fake agreed with it. These tests
# build the REAL TDXVerdict, so a field rename breaks them instead of disabling a check.
MRTD = "a3" * 48
RTMRS = ("00" * 48, "11" * 48, "22" * 48, "33" * 48)
IMAGE = _tdx_module.measurement_id(MRTD, RTMRS)
EPOCH = 7
LADDER = ("cheap", "strong")
SECRET = "my private question about a medical result"


class Gate:
    ok = True

    def require(self):
        return self

    def check(self, **_):
        return self

    limit_remaining = 20.0

    def receipt_fields(self):
        return {"key_ok": True, "key_creator": "user_Owner"}


class Backend:
    def complete(self, model, messages, params):
        return f"[{model}] private answer", 5, 7, 0.0002


def encode(prompts):
    return np.zeros((len(prompts), 8), dtype=float)


def server(key=None) -> ConfidentialServer:
    return ConfidentialServer(enclave_key=key or sealed.EnclaveKey.generate(IMAGE, EPOCH),
                              gate=Gate(), policy=lambda s: 0, backend=Backend(),
                              ladder=LADDER, encode=encode)


def verifier_for(report_data: bytes, ok: bool = True, reason: str = "ok", mr_td: str = MRTD,
                 rtmrs: tuple = RTMRS):
    """Stands in for tdx.verify_quote_full, returning the REAL TDXVerdict.

    Only the hardware/chain/TCB portion is simulated — everything the client then does with the
    verdict runs against the genuine type, so a change to its fields breaks these tests instead of
    silently disabling a check.
    """
    from thirtyspokes.koth.tdx import TDXVerdict

    def fake(raw, *, expect_report_data, **kw):
        if expect_report_data != report_data:
            return TDXVerdict(False, "report_data mismatch", mr_td, rtmrs, report_data, "UpToDate")
        return TDXVerdict(ok, reason, mr_td, rtmrs, report_data, "UpToDate")
    return fake


@pytest.fixture
def patched(monkeypatch):
    def apply(report_data, ok=True, reason="ok", mr_td=MRTD, rtmrs=RTMRS):
        from thirtyspokes.koth import tdx
        f = verifier_for(report_data, ok, reason, mr_td, rtmrs)
        monkeypatch.setattr(tdx, "verify_quote_full", f)
        monkeypatch.setattr(tdx, "verify_quote", f)
    return apply


# --- THE attack -----------------------------------------------------------------------------

def test_a_real_quote_with_a_swapped_key_is_refused(patched):
    """A miner runs a genuine approved enclave and serves its own key alongside the real quote.

    Hardware signature: valid. Intel chain: valid. TCB: UpToDate. Measurements: approved. If the
    client stopped there it would seal every request to the miner. This is the whole reason the
    binding exists, and the reason `verify()` has no flag to skip it.
    """
    real = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    patched(real.report_data())

    miner_key = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    forged = dict(real.announcement(), public_key=miner_key.public_bytes.hex())

    v = C.verify(forged, b"a genuine quote")
    assert not v.ok and "report_data mismatch" in v.reason


def test_the_honest_announcement_passes(patched):
    key = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    patched(key.report_data())
    v = C.verify(key.announcement(), b"quote")
    assert v.ok and v.tcb_status == "UpToDate"
    assert v.mrtd == MRTD, "the verdict must carry the real MRTD, not None"
    assert v.rtmrs == dict(enumerate(RTMRS))


def test_an_image_the_hardware_did_not_measure_is_refused(patched):
    """The second key-swap, one level up. An enclave can put any string in `image_measurement` and
    the binding still verifies, because the string is bound into report_data by the enclave itself.
    Only comparing it against the hardware's own measurement makes the field mean anything."""
    lying = sealed.EnclaveKey.generate("00" * 64, EPOCH)      # claims an image it is not running
    patched(lying.report_data())                              # hardware says otherwise

    v = C.verify(lying.announcement(), b"quote")
    assert not v.ok and "measured identity" in v.reason


def test_the_same_firmware_running_a_different_image_is_refused(patched):
    """THE reason identity is not MRTD. On a cloud TD, MRTD is the virtual firmware and is identical
    for every guest on the platform — so a miner running a completely different image reports the
    same MRTD. Only the boot RTMRs separate them, and a client that pinned MRTD alone would accept
    this."""
    other_boot = ("00" * 48, "de" * 48, "22" * 48, "33" * 48)     # different kernel/UKI -> RTMR1
    key = sealed.EnclaveKey.generate(IMAGE, EPOCH)                # announces the APPROVED identity
    patched(key.report_data(), mr_td=MRTD, rtmrs=other_boot)      # same MRTD, different image

    v = C.verify(key.announcement(), b"quote")
    assert not v.ok and "measured identity" in v.reason


def test_a_rejected_quote_is_a_rejected_enclave(patched):
    key = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    patched(key.report_data(), ok=False, reason="mrtd not approved")
    v = C.verify(key.announcement(), b"quote")
    assert not v.ok and "mrtd not approved" in v.reason


def test_a_claimed_image_the_key_was_not_bound_to_is_refused(patched):
    """Announcing a different image than the one in the quote's binding must fail — otherwise a
    miner could present an approved image name over a key from somewhere else."""
    key = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    patched(key.report_data())
    assert not C.verify(dict(key.announcement(), image_measurement="mrtd:other"), b"q").ok
    assert not C.verify(dict(key.announcement(), epoch=EPOCH + 1), b"q").ok


def test_a_malformed_announcement_is_refused_not_crashed():
    for bad in ({}, {"public_key": "zz", "image_measurement": "i", "epoch": 1}):
        assert not C.verify(bad, b"q").ok


def test_a_verification_error_is_a_refusal(monkeypatch):
    """An exception from the verifier must not escape as a crash, nor be read as success."""
    from thirtyspokes.koth import tdx

    def boom(raw, **kw):
        raise RuntimeError("collateral unreachable")

    monkeypatch.setattr(tdx, "verify_quote_full", boom)
    key = sealed.EnclaveKey.generate(IMAGE, EPOCH)
    v = C.verify(key.announcement(), b"q")
    assert not v.ok and "collateral unreachable" in v.reason


# --- the client object ------------------------------------------------------------------------

def payload_for(s: ConfidentialServer, quote=b"quote") -> dict:
    return {"announcement": s.enclave_key.announcement(), "quote": quote.hex()}


def test_an_enclave_with_no_quote_is_refused():
    """A plain HTTP process claiming to be an enclave. The absence of evidence is a refusal."""
    s = server()
    c = C.ConfidentialClient("http://miner")
    v = c.refresh(fetch=lambda url: {"announcement": s.enclave_key.announcement(), "quote": None})
    assert not v.ok and "not an enclave" in v.reason


def test_sending_without_a_trusted_announcement_raises(patched):
    s = server()
    patched(b"\x00" * 64)                        # a quote that will not match
    c = C.ConfidentialClient("http://miner")
    c.refresh(fetch=lambda url: payload_for(s))
    with pytest.raises(PermissionError, match="refusing to send"):
        c.complete([{"role": "user", "content": "hi"}], post=lambda u, b: b"")


def test_a_full_round_trip_through_the_client(patched):
    s = server()
    patched(s.enclave_key.report_data())
    c = C.ConfidentialClient("http://miner")
    assert c.refresh(fetch=lambda url: payload_for(s)).ok

    sent = {}

    def post(url, ciphertext):
        sent["ciphertext"] = ciphertext
        return s.handle(ciphertext).sealed_response

    body = c.complete([{"role": "user", "content": SECRET}], post=post)
    assert body["choices"][0]["message"]["content"] == "[cheap] private answer"
    assert body["model"] == "thirtyspokes"
    assert SECRET.encode() not in sent["ciphertext"]


def test_each_call_uses_a_fresh_reply_key(patched):
    """One recorded ciphertext must expose one answer, so no reply key may be reused."""
    s = server()
    patched(s.enclave_key.report_data())
    c = C.ConfidentialClient("http://miner")
    c.refresh(fetch=lambda url: payload_for(s))

    seen = []

    def post(url, ciphertext):
        seen.append(json.loads(s.enclave_key.open(ciphertext))["reply_to"]["public_key"])
        return s.handle(ciphertext).sealed_response

    for _ in range(3):
        c.complete([{"role": "user", "content": "hi"}], post=post)
    assert len(set(seen)) == 3


def test_insecure_skip_verify_refuses_to_discard_a_real_quote():
    """The dev bypass must not become a way to ignore attestation that is actually present."""
    s = server()
    c = C.ConfidentialClient("http://miner", insecure_skip_verify=True)
    with pytest.raises(ValueError, match="refusing to discard"):
        c.refresh(fetch=lambda url: payload_for(s))

    v = c.refresh(fetch=lambda url: {"announcement": s.enclave_key.announcement(), "quote": None})
    assert v.ok and "UNVERIFIED" in v.reason


# --- the enclave HTTP surface -------------------------------------------------------------------

@pytest.fixture
def client_pair():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thirtyspokes.serve.enclave_app import create_enclave_app
    s = server()
    app = create_enclave_app(s, quote_provider=lambda rd: b"QUOTE" + rd)
    return s, TestClient(app)


def test_the_enclave_endpoint_publishes_what_a_client_needs(client_pair):
    s, http = client_pair
    body = http.get("/v1/enclave").json()
    assert body["announcement"] == s.enclave_key.announcement()
    assert bytes.fromhex(body["quote"]).endswith(s.enclave_key.report_data())


def test_the_sealed_endpoint_round_trips(client_pair):
    s, http = client_pair
    reply_key = sealed.ReplyKey.generate()
    plaintext = json.dumps({"reply_to": reply_key.public(),
                            "messages": [{"role": "user", "content": SECRET}]}).encode()
    r = http.post("/v1/sealed", content=sealed.seal(s.enclave_key.announcement(), plaintext))

    assert r.status_code == 200
    body = json.loads(reply_key.open(r.content, IMAGE, EPOCH))
    assert body["served_model"] == "cheap"


def test_there_is_no_plaintext_endpoint(client_pair):
    """A clear-JSON path would be more convenient and would void the guarantee."""
    _, http = client_pair
    for path in ("/v1/chat/completions", "/v1/completions"):
        assert http.post(path, json={"messages": []}).status_code == 404


def test_a_garbage_ciphertext_gets_an_indistinguishable_error(client_pair):
    """A caller probing with crafted bytes must not learn where the failure happened."""
    _, http = client_pair
    a = http.post("/v1/sealed", content=b"not a ciphertext at all")
    b = http.post("/v1/sealed", content=b"\x00" * 560)
    assert a.status_code == b.status_code == 400
    assert a.json()["error"] == b.json()["error"]


def test_the_receipt_feed_carries_decisions_but_no_content(client_pair):
    s, http = client_pair
    reply_key = sealed.ReplyKey.generate()
    plaintext = json.dumps({"reply_to": reply_key.public(),
                            "messages": [{"role": "user", "content": SECRET}]}).encode()
    http.post("/v1/sealed", content=sealed.seal(s.enclave_key.announcement(), plaintext))

    feed = http.get("/v1/receipts").json()["receipts"]
    assert len(feed) == 1 and feed[0]["served_model"] == "cheap"
    assert SECRET not in json.dumps(feed) and "medical" not in json.dumps(feed)

    m = http.get("/metrics").json()
    assert m["requests"] == 1 and m["attested"] is True and m["epoch"] == EPOCH


def test_an_unattested_deployment_is_visible_rather_than_silent():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thirtyspokes.serve.enclave_app import create_enclave_app
    http = TestClient(create_enclave_app(server(), quote_provider=None))
    assert http.get("/v1/enclave").json()["quote"] is None
    assert http.get("/metrics").json()["attested"] is False
