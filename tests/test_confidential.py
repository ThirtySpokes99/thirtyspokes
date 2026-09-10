"""In-enclave confidential serving (`serve/confidential.py`).

The test that matters most in this file is `test_the_miner_model_never_receives_the_prompt`. Every
other property here — routing, escalation, receipts — is ordinary engineering that would be caught
by ordinary bugs. That one is different: if it regresses, nothing observably breaks. The service
keeps serving, the quotes keep verifying, the receipts keep signing, and a miner quietly reads
every user request. So it is written as a trap rather than an assertion: the policy records
everything it is handed, and the test hunts for the plaintext in it.

Offline throughout — no TEE, no network, no API.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thirtyspokes.gateway.signing import Signer
from thirtyspokes.koth import orkey, sealed
from thirtyspokes.serve.confidential import ConfidentialServer, verify_receipt

IMAGE, EPOCH, OWNER = "mrtd:test", 7, "user_owner"
LADDER = ("cheap", "mid", "dear")
SECRET = "my psychiatric history and current medication"
DIM = 16


def encode(prompts):
    """Stand-in for the pinned encoder: deterministic, and deliberately NOT invertible to text."""
    out = np.zeros((len(prompts), DIM))
    for i, p in enumerate(prompts):
        out[i, 0] = len(p) % 7
        out[i, 1] = 1.0 if "urgent" in p else -1.0
    return out


class Recorder:
    """A miner policy that keeps everything it is ever given, so a test can search it."""

    def __init__(self, action_for=None):
        self.seen: list[np.ndarray] = []
        self.action_for = action_for or (lambda state: 0)

    def __call__(self, state):
        self.seen.append(np.array(state, copy=True))
        return self.action_for(state)


class Backend:
    def __init__(self, empty_for=(), raise_for=(), cost=0.001, reply=None):
        self.empty_for, self.raise_for, self.cost = set(empty_for), set(raise_for), cost
        self.calls: list[str] = []
        self.reply = reply or (lambda m: f"answer from {m}")

    def complete(self, model, messages, params):
        self.calls.append(model)
        if model in self.raise_for:
            raise RuntimeError("provider down")
        if model in self.empty_for:
            return "  ", 10, 0, self.cost
        return self.reply(model), 10, 20, self.cost


def gate_ok(remaining=25.0):
    info = {"creator_user_id": OWNER, "is_provisioning_key": False, "is_management_key": False,
            "limit": 50.0, "limit_remaining": remaining, "usage": 1.0}
    return orkey.KeyGate("sk-x", OWNER, fetch=lambda url: {"data": info})


def server(policy=None, backend=None, gate=None, signer=None) -> ConfidentialServer:
    return ConfidentialServer(
        enclave_key=sealed.EnclaveKey.generate(IMAGE, EPOCH),
        gate=gate or gate_ok(), policy=policy or Recorder(), backend=backend or Backend(),
        ladder=LADDER, encode=encode, signer=signer)


class Client:
    """A genuinely EXTERNAL client — it holds the enclave's announcement and its own ephemeral
    reply key, and it has no access to the enclave's private key.

    That last part is why this class exists. An earlier version of these tests opened the response
    with `s.enclave_key`, which no real client possesses, and so happily passed while the server
    sealed responses to itself — a service that returns a ciphertext the user cannot read. Modelling
    the client's actual key material is what turns the round-trip assertions into real ones.
    """

    def __init__(self, s: ConfidentialServer, suite: str = "x25519"):
        self.announcement = s.enclave_key.announcement()
        self.reply_key = sealed.ReplyKey.generate(suite)

    def seal(self, content=SECRET, **extra) -> bytes:
        body = json.dumps({"model": "thirtyspokes", "reply_to": self.reply_key.public(),
                           "messages": [{"role": "user", "content": content}], **extra}).encode()
        self.last_plaintext = body      # a real user keeps their own request; disputes need it
        return sealed.seal(self.announcement, body)

    def open(self, sealed_response: bytes) -> dict:
        return json.loads(self.reply_key.open(
            sealed_response, self.announcement["image_measurement"], self.announcement["epoch"]))


def sealed_request(s: ConfidentialServer, content=SECRET, **extra) -> bytes:
    return Client(s).seal(content, **extra)


# --- THE property ---------------------------------------------------------------------------

def test_the_miner_model_never_receives_the_prompt():
    """If this regresses nothing else fails — the service keeps working while the miner reads
    everything. Hence a trap rather than an assertion."""
    rec = Recorder()
    s = server(policy=rec)
    s.handle(sealed_request(s))

    assert rec.seen, "the policy must have been consulted at all"
    for state in rec.seen:
        assert state.dtype.kind == "f", "the miner boundary must carry numbers, not objects"
        blob = state.tobytes()
        for fragment in (SECRET, "psychiatric", "medication"):
            assert fragment.encode() not in blob
    # and the only thing derived from the prompt is the encoder's output
    assert len(rec.seen[0]) == DIM + 2 * len(LADDER) + 1


def test_the_relay_never_sees_plaintext_in_either_direction():
    s = server(backend=Backend(reply=lambda m: "the diagnosis is X"))
    client = Client(s)
    ct = client.seal()
    out = s.handle(ct)

    assert SECRET.encode() not in ct
    assert b"diagnosis" not in out.sealed_response
    body = client.open(out.sealed_response)          # the CLIENT reads it, using only its own key
    assert body["choices"][0]["message"]["content"] == "the diagnosis is X"


def test_the_enclave_cannot_read_back_its_own_response():
    """The response belongs to the client. Confirming the enclave is locked out of it is how we
    know the reply leg is really keyed to the caller and not sealed to ourselves."""
    s = server()
    client = Client(s)
    out = s.handle(client.seal())

    with pytest.raises(Exception):
        s.enclave_key.open(out.sealed_response)
    assert client.open(out.sealed_response)["served_model"]


def test_a_second_client_cannot_open_another_clients_response():
    s = server()
    alice, mallory = Client(s), Client(s)
    out = s.handle(alice.seal("alice's private question"))

    with pytest.raises(Exception):
        mallory.open(out.sealed_response)


def test_a_request_with_no_reply_key_is_refused():
    """Undeliverable rather than silently answered into the void."""
    s = server()
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    with pytest.raises(ValueError, match="reply_to"):
        s.handle(sealed.seal(s.enclave_key.announcement(), body))


def test_the_post_quantum_suite_carries_a_full_round_trip():
    """`pq` is offered as a deployment choice, so it must work on both legs, not just the request."""
    s = server()
    client = Client(s, suite="pq")
    out = s.handle(client.seal())
    assert client.open(out.sealed_response)["model"] == "thirtyspokes"


def test_the_receipt_carries_no_content():
    s = server(signer=Signer())
    r = s.handle(sealed_request(s)).receipt
    blob = json.dumps(r)
    for fragment in (SECRET, "psychiatric", "answer from"):
        assert fragment not in blob
    assert len(r["request_hash"]) == 64 and len(r["response_hash"]) == 64


# --- the key gate fails closed before anything is decrypted -----------------------------------

def test_a_bad_key_refuses_before_decryption():
    bad = orkey.KeyGate("sk-x", OWNER, fetch=lambda url: {"data": {
        "creator_user_id": "user_attacker", "is_provisioning_key": False,
        "limit": 5.0, "limit_remaining": 5.0}})
    s = server(gate=bad)
    with pytest.raises(PermissionError, match="refusing to serve"):
        s.handle(sealed_request(s))
    assert s.backend.calls == [], "no provider call may happen behind a failed gate"


def test_an_exhausted_budget_refuses():
    s = server(gate=gate_ok(remaining=0.0))
    with pytest.raises(PermissionError):
        s.handle(sealed_request(s))


# --- routing and escalation --------------------------------------------------------------------

def test_the_policy_chooses_the_rung():
    for want in range(len(LADDER)):
        s = server(policy=Recorder(action_for=lambda st, w=want: w))
        s.handle(sealed_request(s))
        assert s.backend.calls[0] == LADDER[want]


def test_escalation_on_a_failed_check():
    b = Backend(empty_for={"cheap"})
    s = server(policy=Recorder(action_for=lambda st: int(np.argmin(st[DIM:DIM + len(LADDER)]))),
               backend=b)
    out = s.handle(sealed_request(s))
    assert b.calls[:2] == ["cheap", "mid"]
    assert out.receipt["escalated"] and out.receipt["served_model"] == "mid"


def test_a_provider_error_is_not_a_dead_request():
    b = Backend(raise_for={"cheap"})
    s = server(policy=Recorder(action_for=lambda st: int(np.argmin(st[DIM:DIM + len(LADDER)]))),
               backend=b)
    out = s.handle(sealed_request(s))
    assert out.receipt["served_model"] == "mid"
    assert out.receipt["attempts"][0]["ok"] is False


def test_a_request_may_ask_for_a_stronger_check():
    """Free-form traffic only gets well-formedness; a caller that CAN specify a checker gets it."""
    b = Backend(reply=lambda m: "not json" if m == "cheap" else '{"ok":true}')
    s = server(policy=Recorder(action_for=lambda st: int(np.argmin(st[DIM:DIM + len(LADDER)]))),
               backend=b)
    out = s.handle(sealed_request(s, verify="json"))
    assert out.receipt["served_model"] == "mid" and b.calls == ["cheap", "mid"]


def test_a_broken_miner_model_cannot_take_the_request_down():
    def explode(state):
        raise ZeroDivisionError("miner bug")

    s = server(policy=explode)
    with pytest.raises(RuntimeError, match="policy stopped before any attempt"):
        s.handle(sealed_request(s))
    assert s.backend.calls == []


def test_an_out_of_range_action_is_treated_as_stop():
    """The policy is untrusted code; an index from it is a claim, not a fact."""
    for bogus in (99, -5):
        s = server(policy=Recorder(action_for=lambda st, b=bogus: b))
        with pytest.raises(RuntimeError):
            s.handle(sealed_request(s))
        assert s.backend.calls == []


def test_a_repeated_action_does_not_loop():
    s = server(policy=Recorder(action_for=lambda st: 0), backend=Backend(empty_for=set(LADDER)))
    out = s.handle(sealed_request(s))
    assert s.backend.calls == ["cheap"], "a repeat is STOP, not an infinite ladder"
    assert out.receipt["served_model"] == "cheap"


# --- receipts -----------------------------------------------------------------------------------

def test_cost_sums_every_attempt_including_failures():
    b = Backend(empty_for={"cheap"}, cost=0.002)
    s = server(policy=Recorder(action_for=lambda st: int(np.argmin(st[DIM:DIM + len(LADDER)]))),
               backend=b)
    r = s.handle(sealed_request(s)).receipt
    assert r["cost_usd"] == pytest.approx(0.004), "the failed rung is billed too"
    assert r["decision_path"] == ["cheap", "mid"]


def test_a_user_can_dispute_their_own_request_but_the_miner_cannot():
    """Selective disclosure: the hash is public, the preimage is the user's alone."""
    s = server()
    client = Client(s)
    ct = client.seal()
    r = s.handle(ct).receipt
    from thirtyspokes.gateway import signing
    assert r["request_hash"] == signing.sha256_hex(client.last_plaintext)   # user holds the preimage
    assert r["request_hash"] != signing.sha256_hex(ct)                 # the miner holds only this


def test_receipts_are_signed_and_verifiable():
    signer = Signer()
    s = server(signer=signer)
    r = s.handle(sealed_request(s)).receipt
    assert verify_receipt(r, signer.public_hex)

    tampered = dict(r)
    tampered["cost_usd"] = 0.0
    assert not verify_receipt(tampered, signer.public_hex)


def test_the_receipt_records_the_key_gate_verdict():
    """A key-swap attempt should be publicly legible, not merely blocked."""
    s = server()
    rec = s.handle(sealed_request(s)).receipt
    assert rec["key_ok"] is True and rec["key_creator"] == OWNER
    assert "sk-" not in json.dumps(rec)
