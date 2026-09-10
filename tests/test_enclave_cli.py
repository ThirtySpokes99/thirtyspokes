"""Enclave boot (`serve/enclave_cli.py`).

Boot is where a confidential service quietly stops being confidential: a missing quote, a miner's
own provider key, or a dev flag left on in production all yield a process that serves perfectly
well. So these tests are about what boot REFUSES to do, and about the order it refuses in.
"""

from __future__ import annotations

import argparse

import pytest

from thirtyspokes.serve import enclave_cli

IMAGE = "b7" * 32          # the composite identity the hardware reports, not a bare MRTD
OWNER = "user_Owner"


def args(**over) -> argparse.Namespace:
    base = dict(host="127.0.0.1", port=8080, epoch=3, suite="x25519", owner_account=OWNER,
                weights=None, ladder=["cheap", "mid"], max_tokens=256, timeout=30.0,
                key_recheck=900.0, insecure_no_tdx=False)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def env(monkeypatch):
    """Stand up the collaborators boot touches, so the test controls TDX presence and key verdict."""
    from thirtyspokes.eval import config as cfgmod
    from thirtyspokes.gateway import gateway
    from thirtyspokes.koth import harness, orkey, tdx

    state = {"tdx": True, "key": dict(creator_user_id=OWNER, limit=25.0, limit_remaining=20.0,
                                      is_management_key=False, is_provisioning_key=False,
                                      is_free_tier=False, label="sk-or-v1-x", expires_at=None,
                                      limit_reset="monthly"),
             "quoted": []}

    monkeypatch.setattr(tdx, "tdx_available", lambda: state["tdx"])
    monkeypatch.setattr(tdx, "self_measurement", lambda: IMAGE)
    monkeypatch.setattr(tdx, "get_quote", lambda rd: state["quoted"].append(rd) or b"QUOTE" + rd)
    class Cfg:
        api_key = "sk-or-v1-test"
        base_url = "https://example.invalid/api/v1"

        def require_key(self):
            return True

    monkeypatch.setattr(cfgmod, "LiveConfig", Cfg)
    monkeypatch.setattr(gateway, "OpenRouterBackend", lambda *a, **k: object())
    monkeypatch.setattr(harness, "encode", lambda prompts: [[0.0] * 384 for _ in prompts])
    monkeypatch.setattr(orkey.KeyGate, "check",
                        lambda self, **kw: _verdict(state, owner_account_id=self.owner_account_id))
    return state


def _verdict(state, *, owner_account_id=OWNER, **_):
    """The REAL verify_key against a controllable key info — the gate's own logic is under test
    elsewhere (test_orkey.py); what matters here is that boot honours its verdict."""
    from thirtyspokes.koth.orkey import verify_key
    return verify_key(state["key"], owner_account_id=owner_account_id)


# --- what boot refuses --------------------------------------------------------------------------

def test_no_tdx_and_no_flag_refuses_to_start(env):
    """The default on ordinary hardware is a dead process, not a quiet plaintext service."""
    env["tdx"] = False
    with pytest.raises(SystemExit, match="no TDX available"):
        enclave_cli.build(args())


def test_the_dev_flag_is_refused_on_real_tdx(env):
    """A leftover --insecure-no-tdx in a production unit file must fail loudly rather than
    discard attestation the hardware was perfectly willing to provide."""
    with pytest.raises(SystemExit, match="refusing to discard real attestation"):
        enclave_cli.build(args(insecure_no_tdx=True))


def test_a_miners_own_provider_key_stops_boot(env):
    """And it stops it BEFORE an enclave key exists, so there is never a public key to seal to."""
    env["key"]["creator_user_id"] = "user_TheMiner"
    with pytest.raises(SystemExit, match="provider key refused"):
        enclave_cli.build(args())
    assert env["quoted"] == [], "no quote should have been taken for a key that was never minted"


def test_an_unlimited_provider_key_stops_boot(env):
    env["key"]["limit"] = None
    env["key"]["limit_remaining"] = None
    with pytest.raises(SystemExit, match="provider key refused"):
        enclave_cli.build(args())


def test_an_unconfigured_owner_account_stops_boot(env):
    """Empty owner id would make the creator check vacuous — the gate would pass everything."""
    with pytest.raises(SystemExit, match="provider key refused"):
        enclave_cli.build(args(owner_account=""))


# --- what boot produces -------------------------------------------------------------------------

def test_the_image_comes_from_the_hardware_not_the_arguments(env):
    """There is no flag to set the image measurement, and that is the point: a modified image
    cannot claim an approved MRTD because the value is read from the measurement register."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    http = TestClient(enclave_cli.build(args()))
    body = http.get("/v1/enclave").json()

    assert body["announcement"]["image_measurement"] == IMAGE
    assert body["announcement"]["epoch"] == 3
    assert bytes.fromhex(body["quote"]).startswith(b"QUOTE")
    assert "image" not in vars(args()), "no CLI flag may set the image measurement"


def test_the_quote_covers_the_key_that_was_published(env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thirtyspokes.koth import sealed

    http = TestClient(enclave_cli.build(args()))
    body = http.get("/v1/enclave").json()
    expected = sealed.binding(bytes.fromhex(body["announcement"]["public_key"]), IMAGE, 3)
    assert env["quoted"] == [expected]


def test_the_dev_endpoint_publishes_no_quote_so_clients_refuse_it(env):
    """The dev mode must be visibly unattested, not merely undocumented."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thirtyspokes.serve import client as C

    env["tdx"] = False
    http = TestClient(enclave_cli.build(args(insecure_no_tdx=True)))
    payload = http.get("/v1/enclave").json()
    assert payload["quote"] is None

    c = C.ConfidentialClient("http://dev")
    assert not c.refresh(fetch=lambda url: payload).ok


# --- the policy the enclave serves ---------------------------------------------------------------

def test_without_weights_the_enclave_serves_the_cheapest_rung(env):
    """The measured baseline is a legitimate thing to serve, so a missing model is not an error."""
    assert enclave_cli._load_policy(None, 3)([0.0] * 400) == 0


def test_a_pick_one_head_is_fed_only_the_encoder_features(tmp_path):
    """A KOTH head expects d features; the cascade state that follows them must not reach it."""
    np = pytest.importorskip("numpy")
    seen = {}

    class Head:
        def distribution(self, theta, features):
            seen["shape"] = features.shape
            return np.array([[0.1, 0.9]])

    import thirtyspokes.koth.harness as H
    orig = H.load_head
    H.load_head = lambda w, *, k, d=384, **kw: (Head(), None)
    try:
        w = tmp_path / "head.npz"
        w.write_bytes(b"x")
        policy = enclave_cli._load_policy(str(w), 2, d=8)
        state = list(range(8)) + [1.0, 0.0, 0.0, 0.0, 0.5]     # features ++ tried/failed/spend
        assert policy(state) == 1
        assert seen["shape"] == (1, 8), "the head must not see the cascade state"
    finally:
        H.load_head = orig
