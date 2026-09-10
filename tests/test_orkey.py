"""The OpenRouter key provenance gate (`koth/orkey.py`).

This gate is the whole of the enforceable privacy claim. If it passes a key the owner did not
issue, the miner reads every user request while the enclave, the quote and the sealed channel all
keep working perfectly — nothing else in the system would notice. So the tests are written from the
attacker's side: each one is a way a miner might try to get its own key past the gate, or to escape
supervision entirely.

The response shapes below are the real ones, taken from a live `GET /api/v1/key`.
"""

from __future__ import annotations

import pytest

from thirtyspokes.koth import orkey

# A placeholder, not the real account. The gate compares against a value baked into
# the measured image at build time; a fixture never needs the production identifier,
# and this repo is public.
OWNER = "user_OwnerAccountPlaceholder00"
ATTACKER = "user_MinerOwnAccount0000000000"


def key_info(**over) -> dict:
    """A well-formed owner-issued inference key, as the live API returns it."""
    info = {"label": "sk-or-v1-xxx...xxx", "is_management_key": False,
            "is_provisioning_key": False, "limit": 25.0, "limit_reset": "monthly",
            "limit_remaining": 18.4, "usage": 6.6, "is_free_tier": False,
            "expires_at": None, "creator_user_id": OWNER}
    info.update(over)
    return info


def fetch_returning(info: dict):
    return lambda url: {"data": info}


def fetch_raising(exc: Exception):
    def f(url):
        raise exc
    return f


# --- the check that carries the privacy claim ---------------------------------------------------

def test_an_owner_issued_key_passes():
    v = orkey.verify_key(key_info(), owner_account_id=OWNER)
    assert v.ok and v.reason == "ok"
    assert v.creator_user_id == OWNER and v.limit_remaining == 18.4


def test_a_miner_substituting_its_own_key_is_refused():
    """THE ATTACK THIS EXISTS FOR. A key from the miner's own account would let the miner enable
    prompt logging and read every request; nothing else in the system detects it."""
    v = orkey.verify_key(key_info(creator_user_id=ATTACKER), owner_account_id=OWNER)
    assert not v.ok and "not the owner account" in v.reason


def test_a_key_with_no_creator_is_refused():
    v = orkey.verify_key(key_info(creator_user_id=None), owner_account_id=OWNER)
    assert not v.ok and "<unknown>" in v.reason


def test_an_unconfigured_owner_account_refuses_rather_than_passing_everything():
    """An empty expected account would make the creator comparison vacuous and silently disable the
    gate — the worst possible failure, because everything would look healthy."""
    v = orkey.verify_key(key_info(), owner_account_id="")
    assert not v.ok and "no owner account configured" in v.reason


# --- the other three conditions -----------------------------------------------------------------

def test_a_management_key_is_refused():
    for field in ("is_management_key", "is_provisioning_key"):
        v = orkey.verify_key(key_info(**{field: True}), owner_account_id=OWNER)
        assert not v.ok and "inference key is required" in v.reason, field


def test_an_unlimited_key_is_refused():
    """An owner-issued key with no cap is a blank cheque on the owner's credit balance."""
    v = orkey.verify_key(key_info(limit=None, limit_remaining=None), owner_account_id=OWNER)
    assert not v.ok and "blank cheque" in v.reason


def test_an_exhausted_budget_is_refused():
    for remaining in (0.0, -1.0, None):
        v = orkey.verify_key(key_info(limit_remaining=remaining), owner_account_id=OWNER)
        assert not v.ok and "exhausted" in v.reason, remaining


# --- fail-closed behaviour ----------------------------------------------------------------------

def test_blocking_introspection_does_not_escape_the_gate():
    """The miner controls the enclave's network. Dropping the introspection host must not be a way
    to disable supervision — an unreachable API is a refusal, never a pass."""
    gate = orkey.KeyGate("sk-x", OWNER, fetch=fetch_raising(OSError("network unreachable")))
    v = gate.check()
    assert not v.ok and "introspection failed" in v.reason
    with pytest.raises(PermissionError, match="refusing to serve"):
        gate.require()


def test_a_malformed_response_is_refused():
    gate = orkey.KeyGate("sk-x", OWNER, fetch=lambda url: {"unexpected": True})
    assert not gate.check().ok


def test_require_raises_only_when_the_key_is_bad():
    good = orkey.KeyGate("sk-x", OWNER, fetch=fetch_returning(key_info()))
    assert good.require().ok
    bad = orkey.KeyGate("sk-x", OWNER, fetch=fetch_returning(key_info(creator_user_id=ATTACKER)))
    with pytest.raises(PermissionError):
        bad.require()


# --- re-verification ----------------------------------------------------------------------------

def test_the_verdict_is_cached_then_rechecked():
    """Provenance is not a one-time property: the owner can revoke, and budgets run out. A gate
    that only checked at boot would let a revoked key serve until the next restart."""
    calls = []
    state = {"info": key_info()}

    def fetch(url):
        calls.append(url)
        return {"data": state["info"]}

    clock = {"t": 1000.0}
    gate = orkey.KeyGate("sk-x", OWNER, recheck_seconds=900.0, fetch=fetch)

    assert gate.check(now=lambda: clock["t"]).ok
    gate.check(now=lambda: clock["t"] + 10)          # inside the window: cached
    assert len(calls) == 1

    state["info"] = key_info(limit_remaining=0.0)     # owner revokes / budget exhausts
    clock["t"] += 1000                                 # past the window
    v = gate.check(now=lambda: clock["t"])
    assert len(calls) == 2 and not v.ok and "exhausted" in v.reason


def test_force_bypasses_the_cache():
    calls = []

    def fetch(url):
        calls.append(url)
        return {"data": key_info()}

    gate = orkey.KeyGate("sk-x", OWNER, fetch=fetch)
    gate.check()
    gate.check(force=True)
    assert len(calls) == 2


# --- what reaches the public receipt -------------------------------------------------------------

def test_receipt_fields_carry_the_verdict_but_never_the_credential():
    """Receipts are published. A key label is a partial credential and must not travel."""
    v = orkey.verify_key(key_info(), owner_account_id=OWNER)
    fields = v.receipt_fields()

    assert fields["key_ok"] is True and fields["key_creator"] == OWNER
    blob = repr(fields)
    assert "sk-or" not in blob and "label" not in fields


def test_a_refusal_is_publicly_legible():
    """A miner attempting a key swap should be visible in the receipt feed, not merely blocked."""
    v = orkey.verify_key(key_info(creator_user_id=ATTACKER), owner_account_id=OWNER)
    assert v.receipt_fields()["key_ok"] is False
    assert "not the owner account" in v.receipt_fields()["key_reason"]


# --- provisioning -------------------------------------------------------------------------------

def test_provision_requires_a_positive_limit():
    with pytest.raises(ValueError, match="refuses unlimited"):
        orkey.provision("sk-mgmt", label="miner-1", limit_usd=0)


def test_provision_sends_the_limit_and_reset():
    seen = {}

    def post(url, body):
        seen.update({"url": url, "body": body})
        return {"data": {"key": "sk-or-v1-new"}}

    out = orkey.provision("sk-mgmt", label="miner-1", limit_usd=25, limit_reset="monthly",
                          post=post)
    assert seen["url"].endswith("/keys")
    assert seen["body"] == {"name": "miner-1", "limit": 25.0, "limit_reset": "monthly"}
    assert out["data"]["key"].startswith("sk-or")
