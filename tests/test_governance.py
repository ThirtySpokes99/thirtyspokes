"""The owner's approved-measurements record (`serve/governance.py`).

A pin is only worth what its source is worth. These tests are about the ways a record could arrive
looking authoritative when it is not — forged, altered, unsigned, checked against nothing, or for a
different service — and about the one failure mode that would be worst of all: a refusal that
degrades into "no pins", since an unpinned client accepts any TDX enclave including the miner's own.
"""

from __future__ import annotations

import json

import pytest

from thirtyspokes.gateway.signing import Signer
from thirtyspokes.serve import governance as G

MRTD = "a1" * 48
OTHER = "c9" * 48
RTMRS = ("00" * 48, "11" * 48, "22" * 48, "33" * 48)


@pytest.fixture
def owner():
    return Signer()


def record(owner, **over):
    kw = dict(mrtd=[MRTD], rtmr={1: "b" * 96}, min_epoch=0, signer=owner)
    kw.update(over)
    return G.build_record(**kw)


# --- what a client must not be talked into ------------------------------------------------------

def test_a_forged_record_is_refused(owner):
    """The miner writes its own record naming its own image. This is the whole attack."""
    miner = Signer()
    forged = G.build_record(mrtd=[OTHER], signer=miner)
    with pytest.raises(ValueError, match="signature does not verify"):
        G.load(forged, owner.public_hex)


def test_altering_any_field_breaks_the_signature(owner):
    rec = record(owner)
    for field, value in [("mrtd", [OTHER]), ("min_epoch", 99), ("tcb_accept", ["OutOfDate"]),
                         ("rtmr", {"1": "0" * 96})]:
        with pytest.raises(ValueError, match="signature does not verify"):
            G.load(dict(rec, **{field: value}), owner.public_hex)


def test_an_unsigned_record_is_refused(owner):
    rec = G.build_record(mrtd=[MRTD])            # no signer
    with pytest.raises(ValueError, match="unsigned"):
        G.load(rec, owner.public_hex)


def test_a_signed_record_with_no_key_to_check_it_is_refused(owner):
    """Silently accepting a signature nobody verified is the quiet version of no security."""
    with pytest.raises(ValueError, match="no owner public key"):
        G.load(record(owner), None)


def test_a_record_for_another_service_is_refused(owner):
    """koth/owner.py governs the benchmark image. Its record must not pin the router."""
    rec = record(owner)
    with pytest.raises(ValueError, match="not 'confidential-router'"):
        G.load(dict(rec, service="koth-benchmark"), owner.public_hex)


def test_an_unknown_version_is_refused(owner):
    with pytest.raises(ValueError, match="unsupported record version"):
        G.load(dict(record(owner), v=99), owner.public_hex)


def test_a_record_pinning_nothing_is_refused():
    """Refused at both ends: the owner cannot publish one, and a client will not load one."""
    with pytest.raises(ValueError, match="would pin nothing"):
        G.build_record(mrtd=[])
    with pytest.raises(ValueError, match="no MRTD"):
        G.load({"v": G.RECORD_VERSION, "service": G.SERVICE, "mrtd": []}, None)


def test_every_failure_raises_rather_than_returning_empty_pins(owner):
    """The failure mode that matters most: an empty Approved would mean 'accept any enclave'."""
    assert not G.Approved(), "an empty Approved must be falsy so it cannot pass for a real pin"
    for bad in ("not a dict", 42, None):
        with pytest.raises(ValueError):
            G.load(bad, owner.public_hex)


# --- what it does when honest --------------------------------------------------------------------

def test_an_owner_record_round_trips(owner):
    a = G.load_json(json.dumps(record(owner, min_epoch=4)), owner.public_hex)
    assert a and a.mrtd == {MRTD} and a.rtmr == {1: "b" * 96}
    assert a.min_epoch == 4 and "x25519" in a.suites


def test_two_images_can_be_approved_at_once(owner):
    """A rollout has both images serving; accepting only one would reject half the fleet."""
    a = G.load(record(owner, mrtd=[MRTD, OTHER]), owner.public_hex)
    assert a.mrtd == {MRTD, OTHER}


def test_mrtd_comparison_is_case_insensitive(owner):
    a = G.load(record(owner, mrtd=[MRTD.upper()]), owner.public_hex)
    assert MRTD in a.mrtd, "a hex case difference must not silently unpin an approved image"


# --- the client applying them ---------------------------------------------------------------------

def test_the_client_pins_from_the_record(owner, monkeypatch):
    from thirtyspokes.koth import sealed, tdx
    from thirtyspokes.serve import client as C

    key = sealed.EnclaveKey.generate(tdx.measurement_id(MRTD, RTMRS), 7)
    seen = {}

    def fake(raw, *, expect_report_data, approved_mrtd=None, approved_rtmr=None, **kw):
        seen["mrtd"], seen["rtmr"] = approved_mrtd, approved_rtmr
        return tdx.TDXVerdict(True, "ok", MRTD, RTMRS, expect_report_data, "UpToDate")

    monkeypatch.setattr(tdx, "verify_quote_full", fake)

    c = C.ConfidentialClient("http://miner", approved=G.load(record(owner), owner.public_hex))
    v = c.refresh(fetch=lambda url: {"announcement": key.announcement(), "quote": "00"})

    assert v.ok
    assert seen["mrtd"] == {MRTD}, "the client must pass the owner's pins to the verifier"
    assert seen["rtmr"] == {1: "b" * 96}


def test_an_epoch_below_the_owners_floor_is_refused(owner, monkeypatch):
    """The revocation lever: a compromised enclave key is retired by raising min_epoch, with no
    change to the image or its measurements."""
    from thirtyspokes.koth import sealed, tdx
    from thirtyspokes.serve import client as C

    key = sealed.EnclaveKey.generate(tdx.measurement_id(MRTD, RTMRS), 3)
    monkeypatch.setattr(tdx, "verify_quote_full",
                        lambda raw, *, expect_report_data, **kw:
                        tdx.TDXVerdict(True, "ok", MRTD, RTMRS, expect_report_data, "UpToDate"))

    c = C.ConfidentialClient("http://miner",
                             approved=G.load(record(owner, min_epoch=5), owner.public_hex))
    v = c.refresh(fetch=lambda url: {"announcement": key.announcement(), "quote": "00"})
    assert not v.ok and "below the owner's floor" in v.reason
