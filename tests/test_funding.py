"""The miner's own OpenRouter key (§4, D18): sealed to the owner, bound to the arm, audited.

The owner's decision of 2026-09-08 reverses §7 step 6's "no key is registered": a miner pays for
their arm on their own OpenRouter account. What these tests hold the line on is everything that
decision must not cost — the seal opens only for the owner's mailbox key and only for the hotkey
and registration it names; the gateway still builds the one pinned request body and still meters
every call, hit or miss, against a window allowance; a refused key is the METER's refusal and never
a model's outcome (§5.2c); an unusable key defers the entry with its shot intact; and the residual
§11-1 leaves open — the miner's account routing differently from the owner's — is published as
evidence rather than silently absorbed.

NOTHING HERE TOUCHES A PROVIDER. The key-side probe runs the real `OpenRouterClient` against
`httpx.MockTransport`; every arm runs against scripted pools behind the `Provider` seam.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_gateway import CATALOG, DELEGATE_THEN_STOP, SOLVED, FakeProvider, Policy, grade, tasks
from test_openrouter import Wire
from test_store import FakeS3
from test_validator import NETUID, Pool, address, _key, harness, outcome, router
from test_submission import (
    BUCKET,
    ENDPOINT,
    MINER,
    MINER_SEED,
    deliver,
    sign_with,
)

from thirtyspokes.gateway.signing import Signer
from thirtyspokes.v3 import miner as miner_tool
from thirtyspokes.v3 import owner as owner_tool
from thirtyspokes.v3.access import AccessError, Mailbox, Registration, open_credential
from thirtyspokes.v3.chain import MockChain
from thirtyspokes.v3.funding import KEY_NAME, SealedKey, open_key, seal_key
from thirtyspokes.v3.gateway import GatewayError, GatewayWorker, OwnerGateway
from thirtyspokes.v3.openrouter import Funding
from thirtyspokes.v3.scaffold import OutcomeTable, Scaffold
from thirtyspokes.v3.simulate import DEFERRED, DUELLED
from thirtyspokes.v3.store import S3Bucket, fetch_submission
from thirtyspokes.v3.validator import reveal_path
from thirtyspokes.v3.worker import NotAnOutcome, WorkerError

OWNER_SEED = bytes.fromhex("33" * 32)
OWNER_PUBLIC_HEX = Ed25519PrivateKey.from_private_bytes(OWNER_SEED).public_key() \
    .public_bytes_raw().hex()
STRANGER_SEED = bytes.fromhex("44" * 32)
API_KEY = "sk-or-v1-0123456789abcdef"


def registration(name: str, *, uid: int = 3, block: int = 500) -> Registration:
    return Registration(netuid=NETUID, uid=uid, hotkey=address(name), registration_block=block)


def sealed(name: str, *, cap_usd: float = 50.0, api_key: str = API_KEY,
           owner_hex: str = OWNER_PUBLIC_HEX, reg: Registration | None = None) -> SealedKey:
    reg = reg or registration(name)
    return seal_key(api_key, registration=reg, cap_usd=cap_usd, owner_public_hex=owner_hex,
                    sign=lambda payload: _key(name).sign(payload).hex())


# --- the seal -------------------------------------------------------------------------------------

def test_only_the_owners_mailbox_seed_opens_a_sealed_key():
    """The mirror of the mailbox envelope: sealed to the owner's ed25519 key, signed by the
    hotkey, and a round trip through the bytes that sit in the bucket."""
    reg = registration("alice")
    record = SealedKey.from_bytes(sealed("alice", reg=reg).to_bytes())

    assert open_key(record, OWNER_SEED, hotkey=reg.hotkey,
                    registration_id=reg.registration_id) == API_KEY
    with pytest.raises(AccessError, match="not addressed to this validator"):
        open_key(record, STRANGER_SEED, hotkey=reg.hotkey, registration_id=reg.registration_id)


def test_a_record_for_another_registration_a_forged_one_or_a_tampered_one_is_refused():
    """A record copied between prefixes, one signed by somebody else, and one whose bytes moved
    after signing — three ways for the validator to run an arm on a key the miner did not
    register, all refused before the seal is touched."""
    reg = registration("alice")
    record = sealed("alice", reg=reg)

    with pytest.raises(AccessError, match="belongs to"):
        open_key(record, OWNER_SEED, hotkey=reg.hotkey, registration_id="somebody-else")
    forged = seal_key(API_KEY, registration=reg, cap_usd=1.0, owner_public_hex=OWNER_PUBLIC_HEX,
                      sign=lambda payload: _key("mallory").sign(payload).hex())
    with pytest.raises(AccessError, match="does not verify"):
        open_key(forged, OWNER_SEED, hotkey=reg.hotkey, registration_id=reg.registration_id)
    body = json.loads(record.to_bytes())
    body["cap_usd"] = 1_000_000.0
    with pytest.raises(AccessError, match="does not verify"):
        open_key(SealedKey.from_bytes(json.dumps(body).encode()), OWNER_SEED,
                 hotkey=reg.hotkey, registration_id=reg.registration_id)


def test_the_seal_refuses_what_cannot_fund_an_arm():
    reg = registration("alice")
    with pytest.raises(AccessError, match="positive dollars"):
        sealed("alice", reg=reg, cap_usd=0.0)
    with pytest.raises(AccessError, match="whitespace"):
        sealed("alice", reg=reg, api_key=" sk-or-v1-x\n")
    with pytest.raises(AccessError, match="hex public key"):
        sealed("alice", reg=reg, owner_hex="not-hex")


# --- the gateway: whose provider, whose money --------------------------------------------------

def test_a_bound_hotkeys_calls_go_to_its_own_provider_with_the_owners_pinned_body():
    """D18 at the meter: the bytes are the owner's (`chat_body` over the pin, identical to the
    unbound arm's), the destination is the miner's, and the debit is metered against the cap
    exactly as it was against a credited allowance."""
    owner_pool = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    miner_pool = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(owner_pool)
    gateway.fund("hk_owner_paid", 10.0)
    assert gateway.bind("hk_own_key", miner_pool, 10.0, window=1) == 10.0

    for hotkey in ("hk_owner_paid", "hk_own_key"):
        token = gateway.open_duel(hotkey, f"w1-{hotkey}")
        Scaffold(CATALOG, Policy(DELEGATE_THEN_STOP), GatewayWorker(gateway, token),
                 grade).run_window(tasks(3), nonce="w1", budget_usd=10.0)
        gateway.close(token)

    assert len(owner_pool.seen) == 3 and len(miner_pool.seen) == 3
    assert owner_pool.seen == miner_pool.seen, "the miner's key changed the bytes"
    assert gateway.balance("hk_own_key") == pytest.approx(10.0 - 0.03)
    assert gateway.own_key("hk_own_key") and not gateway.own_key("hk_owner_paid")
    assert gateway.ledger("w1-hk_own_key").served[0]["provider"] == "Fake"


def test_a_cap_is_per_window_and_a_restart_keeps_the_debits_made_under_it(tmp_path):
    """The journal's one new movement. A cap SETS the balance — last window's unspent cap is not
    money anyone holds — and a restart inside the window replays the cap and the debits after it,
    so re-binding the same window cannot refund what the arm already spent."""
    journal = tmp_path / "allowances.jsonl"
    pool = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.25})
    gateway = OwnerGateway(pool, journal=journal)
    gateway.bind("hk", pool, 1.0, window=1)
    token = gateway.open_duel("hk", "w1-hk")
    gateway.call(token, "cheap/model", "fix it")
    assert gateway.balance("hk") == pytest.approx(0.75)

    restarted = OwnerGateway(pool, journal=journal)
    assert restarted.balance("hk") == pytest.approx(0.75)
    assert restarted.bind("hk", pool, 1.0, window=1) == pytest.approx(0.75), \
        "re-binding the same window refunded the debit"
    assert restarted.bind("hk", pool, 1.0, window=2) == 1.0
    assert OwnerGateway(pool, journal=journal).balance("hk") == 1.0


@dataclass
class Refusing:
    """A provider whose account is empty (402) or whose key is dead (401) — the PAYER refused."""

    status: int
    calls: int = 0

    def chat(self, model_id, task_text, params):
        self.calls += 1
        raise WorkerError(f"POST /chat/completions refused with HTTP {self.status}",
                          status=self.status)


def test_a_key_refusal_is_the_meters_refusal_and_never_a_row_in_the_outcome_table():
    """§5.2c's line between an outcome and a refusal, drawn one layer down. A 402 says nothing
    about the model, so it is `GatewayError` (a `NotAnOutcome`), counted in `unfunded_calls`, the
    hold returned — and the window's table stores no row for it, or every later arm would face
    a dead rung that was really one miner's empty wallet. A moderation 403 is the model's answer
    to that task and stays a `WorkerError`, which the table does store."""
    gateway = OwnerGateway(FakeProvider())
    gateway.bind("hk", Refusing(402), 5.0, window=1)
    token = gateway.open_duel("hk", "w1-hk")

    with pytest.raises(GatewayError, match="HTTP 402") as refused:
        gateway.call(token, "cheap/model", "fix it")
    assert isinstance(refused.value, NotAnOutcome)
    assert gateway.balance("hk") == 5.0, "a refused call kept its hold"
    assert gateway.ledger("w1-hk").unfunded_calls == 1

    table = OutcomeTable()
    Scaffold(CATALOG, Policy(DELEGATE_THEN_STOP), GatewayWorker(gateway, token), grade,
             table=table).run_window(tasks(2), nonce="w1", budget_usd=5.0)
    assert table.stats()["rows"] == 0

    gateway.bind("hk_moderated", Refusing(403), 5.0, window=1)
    moderated = gateway.open_duel("hk_moderated", "w1-mod")
    with pytest.raises(WorkerError):
        gateway.call(moderated, "cheap/model", "fix it")
    assert gateway.ledger("w1-mod").unfunded_calls == 0


# --- the validator: the arm runs on the key, and the reveal says so ------------------------------

@dataclass
class KeyedPool(Pool):
    """A miner's own provider: `Pool` plus the funding probe, served from its own endpoint."""

    funding: object = None
    endpoint: str = "miner-endpoint"

    def chat(self, model_id, task_text, params):
        completion = super().chat(model_id, task_text, params)
        return completion.__class__(**{**completion.__dict__, "provider": self.endpoint})


def enrol_with_key(h, name: str, *, cap_usd: float = 50.0, pool: KeyedPool) -> tuple[str, str]:
    """Enrol a challenger with NO hand credit and a sealed key under its prefix; wire the
    validator to open it and to run that key on `pool`."""
    hotkey = h.enrol(name, block=10, conductor=router(h), budget_usd=0.0)
    uid = h.chain.metagraph().resolve(hotkey).uid
    reg = Registration(netuid=NETUID, uid=uid, hotkey=hotkey, registration_block=500)
    # Beside the tree in the PRIVATE models bucket — where the miner's credential writes, and
    # the only bucket the daemon's funding step reads a key record from.
    h.private.put(reg.prefix + KEY_NAME, sealed(name, cap_usd=cap_usd, reg=reg).to_bytes())
    h.validator.key_seed = OWNER_SEED
    providers = getattr(h.validator, "_test_providers", {})
    providers[API_KEY] = pool
    h.validator._test_providers = providers
    h.validator.provider_for = providers.__getitem__
    return hotkey, reg.registration_id


def test_a_challenger_with_a_registered_key_runs_on_it_and_the_reveal_publishes_the_evidence(
        tmp_path):
    """The whole rail through the daemon: no credit from the owner, the arm bought on the miner's
    own provider, metered and reconciled like any other, and the reveal carrying `own_key`, the
    endpoints that served it, and the §11-1 drift — here every model, because the owner's arms
    were served by `mock` and the miner's by `miner-endpoint`."""
    h = harness(tmp_path, owner_budget=1_000.0)
    pool = KeyedPool(answers=h.pool.answers, costs=h.pool.costs,
                     funding=lambda: Funding(limit_remaining=None, credits_remaining=None))
    hotkey, _ = enrol_with_key(h, "keyed", pool=pool)

    reveal = h.run(1)

    assert outcome(reveal, hotkey).status == DUELLED
    assert pool.calls > 0, "the arm never reached the miner's provider"
    arm = next(m for m in reveal.arms if m.arm == hotkey)
    assert arm.own_key and arm.scoreable and arm.audit.charged_usd > 0.0
    assert set(arm.endpoints.values()) == {("miner-endpoint",)}
    assert arm.drift and all(pair.endswith("@miner-endpoint") for pair in arm.drift)
    assert not any(m.own_key for m in reveal.arms if m.arm != hotkey)

    published = json.loads(h.store.get(reveal_path(1)))["record"]["metering"]
    row = next(m for m in published if m["arm"] == hotkey)
    assert row["own_key"] is True and row["drift"] == list(arm.drift)
    assert next(m for m in published if m["arm"] == "ref-cascade")["own_key"] is False


def test_a_key_the_provider_refuses_defers_the_entry_with_its_shot_intact(tmp_path):
    """The probe runs before any arm is opened: a dead key costs one GET, the entry is DEFERRED
    with the reason, nothing is bought, and the hotkey is not judged."""
    h = harness(tmp_path, owner_budget=1_000.0)

    def dead():
        raise WorkerError("GET /auth/key refused with HTTP 401", status=401)
    pool = KeyedPool(answers=h.pool.answers, costs=h.pool.costs, funding=dead)
    hotkey, _ = enrol_with_key(h, "dead-key", pool=pool)

    reveal = h.run(1)

    row = outcome(reveal, hotkey)
    assert row.status == DEFERRED and "refused the registered key (HTTP 401)" in row.detail
    assert pool.calls == 0
    assert hotkey not in h.validator.history.judged
    assert reveal.report.crowned is None


def test_an_empty_account_behind_the_key_is_deferred_not_run_to_zero(tmp_path):
    """The account reports nothing left: deferred as unfunded (§4's "Kind" property) rather than
    run as an arm whose every call the provider refuses."""
    h = harness(tmp_path, owner_budget=1_000.0)
    pool = KeyedPool(answers=h.pool.answers, costs=h.pool.costs,
                     funding=lambda: Funding(limit_remaining=None, credits_remaining=0.0))
    hotkey, _ = enrol_with_key(h, "broke", pool=pool)

    reveal = h.run(1)

    row = outcome(reveal, hotkey)
    assert row.status == DEFERRED and "nothing to spend this window" in row.detail
    assert pool.calls == 0 and hotkey not in h.validator.history.judged


def test_the_window_allowance_is_the_miners_cap_bounded_by_what_the_key_can_still_spend(
        tmp_path):
    """§4's allowance under D18: the cap is the miner's, and the key's own remaining limit or the
    account's remaining credits bound it, whichever is smaller."""
    h = harness(tmp_path, owner_budget=1_000.0)
    pool = KeyedPool(answers=h.pool.answers, costs=h.pool.costs,
                     funding=lambda: Funding(limit_remaining=0.30, credits_remaining=12.0))
    hotkey, registration_id = enrol_with_key(h, "capped", cap_usd=50.0, pool=pool)

    assert h.validator._fund(hotkey, registration_id, 1) is None
    assert h.gateway.balance(hotkey) == pytest.approx(0.30)
    assert h.gateway.own_key(hotkey)


def test_a_hotkey_without_a_registered_key_stays_on_the_owners_credited_allowance(tmp_path):
    """Nothing changes for an entry the owner credited by hand: no record, no bind, the ordinary
    deferral text — and it now names the key as the way to fund oneself."""
    h = harness(tmp_path, owner_budget=1_000.0)
    h.validator.key_seed = OWNER_SEED
    credited = h.enrol("credited", block=10, conductor=router(h), budget_usd=100.0)
    penniless = h.enrol("penniless", block=11, conductor=router(h), budget_usd=0.0)

    reveal = h.run(1)

    assert outcome(reveal, credited).status == DUELLED
    assert not next(m for m in reveal.arms if m.arm == credited).own_key
    row = outcome(reveal, penniless)
    assert row.status == DEFERRED and "Register an OpenRouter key" in row.detail


# --- the miner's command, and the store's one exception --------------------------------------

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
def mailbox(tmp_path) -> Mailbox:
    """The owner's mailbox, on a KNOWN key, so the test can open what the miner sealed to it."""
    return Mailbox(tmp_path / "state" / "mailbox.json",
                   Signer(Ed25519PrivateKey.from_private_bytes(OWNER_SEED)),
                   owner_tool.owner_mint(endpoint=ENDPOINT, bucket=BUCKET, account_id="acct",
                                         access_key_id="parent-key",
                                         secret_access_key="parent-secret"))


def test_register_key_puts_a_record_the_validator_opens_and_submit_tolerates(
        chain, mailbox, bucket, tmp_path):
    """`register-key` on the same credential `submit` opens: the record lands under the derived
    prefix, opens under the owner's mailbox seed to the key that went in, `submit` reports it,
    and `fetch_submission` — which refuses any object the manifest does not name — treats it as
    the one named exception."""
    from test_admission import _reference_tree
    from test_submission import submit

    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    common = dict(netuid=0, hotkey=MINER, mailbox_url="https://mailbox.example",
                  owner_public_hex=mailbox.owner_public_hex, seed=MINER_SEED,
                  sign=sign_with(MINER_SEED), open_bucket=lambda credential: bucket,
                  fetch=deliver(bucket))
    done = miner_tool.register_key(chain, api_key=API_KEY, cap_usd=25.0, **common)
    assert done.registration_id == issued.registration.registration_id
    assert "…cdef" in str(done)

    prefix = issued.registration.prefix
    record = SealedKey.from_bytes(bucket.get(prefix + KEY_NAME))
    assert open_key(record, OWNER_SEED,
                    hotkey=MINER, registration_id=done.registration_id) == API_KEY
    assert record.cap_usd == 25.0

    reference = _reference_tree(tmp_path / "reference")
    model = _reference_tree(tmp_path / "model")
    shard = model / "model-00001-of-00002.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] = (raw[-1] + 1) % 256
    shard.write_bytes(bytes(raw))
    submitted = submit(chain, mailbox, bucket, model, reference)
    assert submitted.key_registered and "Keep the OpenRouter account" in str(submitted)

    commitment = next(c for c in chain.commitments() if c.hotkey == MINER)
    fetch_submission(bucket, tmp_path / "fetched", commitment=commitment,
                     registration=miner_tool.registration_of(chain, 0, MINER))
    assert not (tmp_path / "fetched" / KEY_NAME).exists(), "the key was materialised as a file"

    # Rotation is a re-run: the newer record replaces the older one.
    miner_tool.register_key(chain, api_key="sk-or-v1-rotated", cap_usd=5.0, **common)
    rotated = SealedKey.from_bytes(bucket.get(prefix + KEY_NAME))
    assert open_key(rotated, OWNER_SEED, hotkey=MINER,
                    registration_id=done.registration_id) == "sk-or-v1-rotated"


def test_submit_says_plainly_when_no_key_is_registered():
    plain = miner_tool.Submitted(hotkey=MINER, registration_id="r", manifest_sha256="m", files=1,
                                 bytes_uploaded=1, skipped=0, key_registered=False)
    assert "NO OPENROUTER KEY IS REGISTERED" in str(plain)


def test_the_key_comes_from_a_file_or_the_environment_never_from_argv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(miner_tool.MinerError, match="no OpenRouter key"):
        miner_tool.read_api_key(None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-env  ")
    assert miner_tool.read_api_key(None) == "sk-env"
    (tmp_path / "key").write_text("sk-file\n")
    assert miner_tool.read_api_key(tmp_path / "key") == "sk-file"


def test_mailbox_seed_is_the_key_orchestra_owner_key_publishes(tmp_path):
    """Miners seal to `thirtyspokes-owner key`; the daemon opens with `mailbox_seed`. Same file."""
    state = tmp_path / "state"
    seed = owner_tool.mailbox_seed(state)
    assert Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex() \
        == owner_tool.mailbox_signer(state).public_hex


# --- the live key's probe ---------------------------------------------------------------------

def test_funding_reads_the_keys_remaining_limit_and_the_accounts_remaining_credits():
    wire = Wire(httpx.Response(200, json={"data": {"label": "sk-or-v1-…", "usage": 2.5,
                                                    "limit": 10.0, "limit_remaining": 7.5}}),
                httpx.Response(200, json={"data": {"total_credits": 10.0, "total_usage": 4.0}}))
    funding = wire.client().funding()
    assert funding == Funding(limit_remaining=7.5, credits_remaining=6.0)
    assert funding.bound(50.0) == 6.0 and funding.bound(1.0) == 1.0
    assert [request.url.path for request in wire.sent] == ["/api/v1/auth/key", "/api/v1/credits"]


def test_a_key_the_provider_does_not_know_is_refused_with_its_status():
    wire = Wire(httpx.Response(401, json={"error": {"message": "No auth credentials found"}}))
    with pytest.raises(WorkerError) as refused:
        wire.client().funding()
    assert refused.value.status == 401


def test_an_unlimited_key_on_an_account_that_hides_its_credits_is_bounded_by_the_cap_alone():
    wire = Wire(httpx.Response(200, json={"data": {"limit": None, "limit_remaining": None}}),
                httpx.Response(403, json={"error": {"message": "forbidden"}}))
    assert wire.client().funding().bound(3.0) == 3.0


def test_a_key_only_credential_rotates_the_key_after_the_shot_is_spent_and_touches_no_tree(
        chain, mailbox, bucket, tmp_path):
    """The rotation path the one-shot closes. Once the submission credential has expired and the
    shot is spent, `issue` rightly refuses — but a dead or leaked OpenRouter key still has to be
    replaceable. `issue --key-only` mints for the `openrouter/` sub-prefix alone, to a spent hotkey
    too, in its own generation series; the envelope opens only as what it is; and everything under
    that sub-prefix is invisible to the tree's verification, so the credential can write nothing
    the commitment covers."""
    from test_admission import _reference_tree
    from test_submission import submit

    owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)
    reference = _reference_tree(tmp_path / "reference")
    model = _reference_tree(tmp_path / "model")
    shard = model / "model-00001-of-00002.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] = (raw[-1] + 1) % 256
    shard.write_bytes(bytes(raw))
    submit(chain, mailbox, bucket, model, reference)
    commitment = next(c for c in chain.commitments() if c.hotkey == MINER)
    registration = miner_tool.registration_of(chain, 0, MINER)
    mailbox.consume(commitment, registration)                 # the daemon's step: the shot is spent
    with pytest.raises(AccessError, match="permanently consumed"):
        owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER)

    issued = owner_tool.issue(chain, mailbox, bucket.put, netuid=0, hotkey=MINER, key_only=True)
    assert issued.key_only and issued.generation == 1
    assert issued.mailbox_key.endswith("/key-00000000000000000001.bin")
    assert "key-only generation 1" in str(issued) and registration.key_prefix in str(issued)
    assert mailbox.outstanding()[0].generations == 1, "the key series moved the submission series"

    sealed_envelope = bucket.get(issued.mailbox_key)
    with pytest.raises(AccessError, match="unexpected allowed_prefix"):
        open_credential(sealed_envelope, MINER_SEED, owner_public_hex=mailbox.owner_public_hex,
                        registration=registration, generation=1)
    envelope = open_credential(sealed_envelope, MINER_SEED,
                               owner_public_hex=mailbox.owner_public_hex,
                               registration=registration, generation=1,
                               prefix=registration.key_prefix)
    assert envelope["allowed_prefix"] == registration.prefix + "openrouter/"

    common = dict(netuid=0, hotkey=MINER, mailbox_url="https://mailbox.example",
                  owner_public_hex=mailbox.owner_public_hex, seed=MINER_SEED,
                  sign=sign_with(MINER_SEED), open_bucket=lambda credential: bucket,
                  fetch=deliver(bucket))
    done = miner_tool.register_key(chain, api_key="sk-or-v1-rotated", cap_usd=5.0,
                                   generation=1, key_credential=True, **common)
    record = SealedKey.from_bytes(bucket.get(registration.prefix + KEY_NAME))
    assert open_key(record, OWNER_SEED, hotkey=MINER,
                    registration_id=done.registration_id) == "sk-or-v1-rotated"

    # Anything under the sub-prefix is invisible to the tree's verification; anything else is not.
    bucket.put(registration.prefix + "openrouter/stale.json", b"{}")
    bucket.put(registration.prefix + "openrouter.json", b"{}")     # the record's first location
    fetch_submission(bucket, tmp_path / "fetched", commitment=commitment,
                     registration=registration)
    bucket.put(registration.prefix + "stray.bin", b"x")
    with pytest.raises(Exception, match="files the manifest does not name"):
        fetch_submission(bucket, tmp_path / "fetched-again", commitment=commitment,
                         registration=registration)
