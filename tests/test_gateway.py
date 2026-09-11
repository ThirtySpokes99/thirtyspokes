"""The owner-run gateway's properties (docs/WHITEPAPER.md §11-1, the build plan M5).

M5 closes what the spec calls the largest remaining hole: with an open catalog (D4) and the miner
supplying the key (§4), BYOK and per-key provider preferences mean `openai/gpt-5.4` need not be the
same object across two arms, so a paired duel measures the two miners' provider settings alongside
their routing. Every test here is one of that milestone's four exit criteria or one of the two
accounting rules the spec attaches to them.

WHAT IS *NOT* TESTED HERE, DELIBERATELY: the request body and the transport. Both belong to
`v3/openrouter.py` and are covered in `test_openrouter.py` — the pinned `PROVIDER_ROUTING`, the
one-user-message body, a 200 that is really a failure, a missing `usage.cost`. This module builds no
request of its own, so a second copy of those assertions here would be a second pin drifting from
the first, which is the very failure §11-1 is about. What is tested here is what the gateway adds:
whose key pays, whose duel it is, and whether the arm's spend can be believed.

NOT ONE OF THESE TESTS TOUCHES A PROVIDER. The offline ones drive a scripted `FakeProvider` behind
the `Provider` seam; the end-to-end one runs the real `OpenRouterClient` against an
`httpx.MockTransport`, so the client that would spend money is the one under test and only the
destination of the bytes differs.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest import mock
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from thirtyspokes.gateway.receipts import Receipt
from thirtyspokes.gateway.signing import Signer, sha256_hex
from thirtyspokes.v3.conductor import MockConductor
from thirtyspokes.v3.config import PRICE_PROBE_WAIT_SECONDS
from thirtyspokes.v3.gateway import (
    Audit,
    DuelLedger,
    GatewayError,
    GatewayWorker,
    OwnerGateway,
    audit,
    request_hash,
)
from thirtyspokes.v3.openrouter import (
    PROVIDER_ROUTING,
    Completion,
    OpenRouterClient,
    chat_body,
)
from thirtyspokes.v3.scaffold import Scaffold, task_order
from thirtyspokes.v3.types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from thirtyspokes.v3.worker import WORKER_PARAMS, MockWorker

CATALOG = Catalog(entries=(
    CatalogEntry("cheap/model", 0.05, 0.20, 128_000),
    CatalogEntry("strong/model", 2.50, 10.00, 400_000),
))

TASK = TaskSpec(task_id="tb-0007", benchmark="terminal-bench",
                prompt="The test suite fails on a fresh checkout of the repo. Make it pass.",
                tools=("bash", "editor"))

SOLVED = "SOLVED"
KING, CHALLENGER = "hk_king", "hk_challenger"


def grade(task: TaskSpec, answer: str) -> float:
    return 1.0 if answer == SOLVED else 0.0


def tasks(n: int) -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(task_id=f"tb-{i:04d}", benchmark="terminal-bench",
                          prompt=f"fix bug {i}", tools=()) for i in range(n))


@dataclass
class FakeProvider:
    """A scripted pool behind the `Provider` seam, and the witness for what the gateway sent.

    `seen` records the request body as `openrouter.chat_body` built it — the same role
    `MockWorker.seen` plays for §2.1 in `test_scaffold.py`, one layer further out. A log the
    gateway writes cannot prove what the gateway sent.
    """

    answers: Mapping[str, str] = field(default_factory=dict)
    prices: Mapping[str, float] = field(default_factory=dict)   # dollars per call
    seen: list[dict] = field(default_factory=list)

    def chat(self, model_id, task_text, params):
        self.seen.append(chat_body(model_id, task_text, params))
        return Completion(text=self.answers.get(model_id, ""),
                          cost_usd=float(self.prices.get(model_id, 0.0)), tokens_in=100,
                          tokens_out=50, finish_reason="stop", served_model=model_id,
                          provider="Fake")


@dataclass(frozen=True)
class Policy:
    """A Conductor that is a function of the rendered state, so it repeats per episode."""

    turns: tuple[str, ...]

    def act(self, prompt: str) -> str:
        return self.turns[min(prompt.count("\nstep "), len(self.turns) - 1)]


DELEGATE_THEN_STOP = ("DELEGATE cheap/model", "STOP")


def arm(gateway: OwnerGateway, hotkey: str, duel_id: str, *, budget_usd: float = 100.0,
        conductor=None, slice_=None) -> tuple[tuple[EpisodeResult, ...], DuelLedger]:
    """One arm through the gateway, as the validator runs it: open, run, close for the bill."""
    token = gateway.open_duel(hotkey, duel_id)
    scaffold = Scaffold(CATALOG, conductor or Policy(DELEGATE_THEN_STOP),
                        GatewayWorker(gateway, token), grade)
    episodes = scaffold.run_window(slice_ or tasks(3), nonce="w7", budget_usd=budget_usd)
    return tuple(e.result for e in episodes), gateway.close(token)


# --- M5 exit 2: the owner pins provider and model identity ---------------------------------------
def test_two_miners_arms_send_the_same_bytes_because_the_owner_builds_the_request():
    """M5 exit 2, and the whole of §11-1. Under §4's BYOK the king's key and the challenger's key
    each carry their own provider preferences, so the two arms can be routed to different upstream
    deployments of the same catalog slug — different quantisation, different fallback chain,
    different price — and §5.2's paired verdict is then computed on that difference rather than on
    routing.

    That the body itself is right is `test_openrouter.py`'s property. What is asserted here is
    the gateway's: the two arms' bodies are IDENTICAL, and a miner has nowhere to make them differ.
    `fund` takes dollars and `open_duel` takes an identifier; neither has a slot for a key or a
    provider preference, so the substitution channel is not guarded but absent."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(KING, 10.0)
    gateway.fund(CHALLENGER, 10.0)

    arm(gateway, KING, "w7-king")
    king_bodies = list(provider.seen)
    provider.seen.clear()
    arm(gateway, CHALLENGER, "w7-challenger")

    assert king_bodies == provider.seen, (
        "the two arms did not send the same bytes, so the duel is not paired (§11-1)")
    assert all(body["provider"] == dict(PROVIDER_ROUTING) for body in provider.seen)
    assert all(body["temperature"] == WORKER_PARAMS["temperature"] for body in provider.seen)


def test_the_worker_seam_refuses_params_that_are_not_the_pinned_constant():
    """§2.1's tripwire. The gateway rebuilds the body from the pin regardless, which is exactly why
    the check has to be here and has to be loud: a caller that started passing per-call settings
    would otherwise run an arm on parameters nobody chose while the scaffold recorded that it had
    sent the pinned ones."""
    gateway = OwnerGateway(FakeProvider())
    gateway.fund(KING, 10.0)
    worker = GatewayWorker(gateway, gateway.open_duel(KING, "w7-king"))

    with pytest.raises(GatewayError, match="not the pinned"):
        worker.complete("cheap/model", TASK.prompt, {"temperature": 0.7})
    with pytest.raises(GatewayError, match="not the pinned"):
        worker.complete("cheap/model", TASK.prompt, {**WORKER_PARAMS, "system": "you are helpful"})
    assert gateway.ledger("w7-king").receipts == [], "a refused call must not reach a provider"


# --- M5 exit 3: the miner never sees the task prompts ---------------------------------------------
def test_a_miner_can_reconcile_its_bill_without_ever_seeing_a_task_prompt():
    """M5 exit 3, which is a defence of D12 rather than of privacy.

    Under §4's BYOK every worker call appears prompt-by-prompt in the MINER's own provider log. The
    slice is drawn after challengers commit (D12) precisely so nobody knows which tasks will be
    scored — and a miner reading their own request log knows all of it, live, every window they are
    in. The sitting king is in every window (§5.2a), so the incumbent gets M7a's ~$60,000 profiling
    table for free and forever.

    What the miner gets instead is the receipt, and this test asserts both halves of what that has
    to be: no task text anywhere in the record they can audit, and enough to check the bill — the
    model called, the tokens, the real cost, and a signature they can verify."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(CHALLENGER, 10.0)
    results, ledger = arm(gateway, CHALLENGER, "w7-challenger")

    billed = json.dumps([receipt.__dict__ for receipt in ledger.receipts])
    for task in tasks(3):
        assert task.prompt not in billed, "the bill disclosed the task text it was billed for"
    assert all(receipt.verify(gateway.public_hex) for receipt in ledger.receipts)
    assert ledger.spend_usd == pytest.approx(sum(r.spend_usd for r in results))
    assert [r.prompt_hash for r in ledger.receipts] == [
        request_hash("cheap/model", task.prompt) for task in task_order(tasks(3), "w7")], (
        "a bill has to be reconcilable against the window's own nonce-derived order (§4)")


def test_a_receipts_hash_carries_the_2_1_audit_past_the_seam():
    """`Episode.requests` proves what the validator MEANT to send; the receipt proves what the
    provider was sent. Anyone holding the public benchmark can recompute the hash, so §2.1 becomes
    checkable by a third party from the published record — without that record disclosing a prompt
    to the miner being billed for it.

    It covers the whole body rather than the text alone, which is the point of hashing it here
    rather than reusing a prompt digest: the two fields this module pins are the model and the
    provider block, and a hash that left them out would be evidence about everything except what
    §11-1 is worried about."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(KING, 10.0)
    token = gateway.open_duel(KING, "w7-king")
    _, receipt = gateway.call(token, "cheap/model", TASK.prompt)

    assert receipt.prompt_hash == request_hash("cheap/model", TASK.prompt)
    assert receipt.prompt_hash != request_hash("cheap/model", TASK.prompt + " ")
    assert receipt.prompt_hash != request_hash("strong/model", TASK.prompt), (
        "a hash that does not cover the model is not evidence about substitution")
    assert provider.seen[0] == chat_body("cheap/model", TASK.prompt, WORKER_PARAMS)


# --- M5 exit 4: per-duel token scope --------------------------------------------------------------
def test_a_token_cannot_be_replayed_against_another_duel():
    """M5 exit 4. The token IS the duel — `call` has no duel argument to disagree with — so charging
    one arm's work to another's allowance is not a check that can be forgotten but a call that
    cannot be written. Closing the arm is the other half: a finished duel's token must not quietly
    fund a later one.

    The second half is what makes the first matter. Even if a token escaped before it was closed,
    the receipts it produces are stamped with ITS duel, so they are not evidence for the other arm —
    which `audit` refuses rather than silently accepting as spend."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(KING, 10.0)
    gateway.fund(CHALLENGER, 10.0)

    first = gateway.open_duel(KING, "w7-king")
    gateway.close(first)
    with pytest.raises(GatewayError, match="closed"):
        gateway.call(first, "cheap/model", TASK.prompt)

    leaked = gateway.open_duel(KING, "w8-king")
    gateway.open_duel(CHALLENGER, "w8-challenger")
    gateway.call(leaked, "cheap/model", TASK.prompt)
    assert gateway.ledger("w8-challenger").receipts == []
    assert gateway.balance(CHALLENGER) == 10.0, "another duel's token spent this miner's allowance"

    stolen = gateway.ledger("w8-king").receipts
    verdict = audit((), stolen, duel_id="w8-challenger", hotkey=CHALLENGER,
                    gateway_public_hex=gateway.public_hex)
    assert verdict.scoreable is False and verdict.reason == "receipt_from_another_duel"


def test_a_duel_id_is_opened_once_and_an_unknown_token_buys_nothing():
    """Re-opening a duel would give two arms one ledger and one receipt scope, so the second arm's
    spend could be presented as the first's. And a token is unguessable because it authorises
    spending a miner's real money, so anything that can reach the gateway must not be able to make
    one."""
    gateway = OwnerGateway(FakeProvider())
    gateway.fund(KING, 10.0)
    token = gateway.open_duel(KING, "w7-king")

    with pytest.raises(GatewayError, match="already been opened"):
        gateway.open_duel(KING, "w7-king")
    with pytest.raises(GatewayError, match="unknown duel token"):
        gateway.call("w7-king", "cheap/model", TASK.prompt)
    with pytest.raises(GatewayError, match="contains '/'"):
        gateway.open_duel(KING, "w7/king")
    assert token not in ("w7-king", "") and len(token) >= 32


# --- M5 exit 1: a call without a receipt is unscoreable -------------------------------------------
def test_an_arm_that_worked_off_gateway_is_unscoreable():
    """M5 exit 1. An arm run against a provider directly has spend with nothing behind it: no
    receipt, no metered dollar, no way to know the calls went where the catalog said. Scoring it
    would price `final_b` (§5.1b) on a number the owner never observed, so the arm is refused
    instead — and the reveal is told by how much the two figures disagreed rather than only that
    they did."""
    off_gateway = MockWorker(answers={"cheap/model": SOLVED}, costs={"cheap/model": 0.05})
    episodes = Scaffold(CATALOG, Policy(DELEGATE_THEN_STOP), off_gateway, grade).run_window(
        tasks(3), nonce="w7", budget_usd=100.0)
    results = tuple(e.result for e in episodes)

    verdict = audit(results, (), duel_id="w7-challenger", hotkey=CHALLENGER,
                    gateway_public_hex=Signer().public_hex)
    assert verdict.scoreable is False and verdict.reason == "spend_not_metered"
    assert verdict.recorded_usd == pytest.approx(0.15) and verdict.metered_usd == 0.0


def test_the_gateway_meters_exactly_the_arm_the_scaffold_scores():
    """The other half: an arm run through the gateway reconciles to the cent, which is M1 exit 6's
    property carried across the seam that now holds the money. `spend_b` is half of `final_b`, so
    the arm's recorded spend and the gateway's metered spend agreeing is what makes the score priced
    rather than asserted."""
    provider = FakeProvider(answers={"cheap/model": "no", "strong/model": SOLVED},
                            prices={"cheap/model": 0.02, "strong/model": 0.30})
    gateway = OwnerGateway(provider)
    gateway.fund(CHALLENGER, 10.0)
    results, ledger = arm(gateway, CHALLENGER, "w7-challenger",
                          conductor=Policy(("DELEGATE cheap/model", "RETRY strong/model", "STOP")))

    verdict = audit(results, ledger.receipts, duel_id="w7-challenger", hotkey=CHALLENGER,
                    gateway_public_hex=gateway.public_hex)
    assert verdict == Audit(scoreable=True, reason="ok", metered_usd=pytest.approx(0.96),
                            recorded_usd=pytest.approx(0.96), calls=6,
                            charged_usd=pytest.approx(0.96), provider_usd=pytest.approx(0.96))
    assert [r.graded_score for r in results] == [1.0] * 3
    assert gateway.balance(CHALLENGER) == pytest.approx(10.0 - 0.96)


def test_a_receipt_the_gateway_did_not_sign_is_not_evidence():
    """The forged-receipt case from `gateway/verify.py`, in the v3 shape: a receipt minted
    elsewhere claiming the arm cost almost nothing. `final_b` subtracts priced spend, so a
    fabricated near-zero cost is a free score bonus."""
    impostor = Signer()
    gateway = OwnerGateway(FakeProvider())
    forged = Receipt("w7-challenger/0", CHALLENGER, "strong/model",
                     request_hash("strong/model", TASK.prompt), sha256_hex(SOLVED), 100, 50,
                     0.00001, 1).signed_by(impostor)
    results = (EpisodeResult(TASK.task_id, TASK.benchmark, (), 1.0, 0.00001, "stop"),)

    verdict = audit(results, (forged,), duel_id="w7-challenger", hotkey=CHALLENGER,
                    gateway_public_hex=gateway.public_hex)
    assert verdict.scoreable is False and verdict.reason == "bad_gateway_signature"


def test_one_arms_receipts_do_not_pay_for_another_miners():
    """Receipts are bound to the hotkey whose allowance paid for them. In `gateway/` that binding
    stopped a miner citing someone else's work as their own; here it stops the king's metered spend
    being presented as the challenger's — which under a paired verdict moves a crown."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(KING, 10.0)
    king_results, ledger = arm(gateway, KING, "w7-king")

    verdict = audit(king_results, ledger.receipts, duel_id="w7-king", hotkey=CHALLENGER,
                    gateway_public_hex=gateway.public_hex)
    assert verdict.scoreable is False and verdict.reason == "receipt_hotkey_mismatch"


# --- §4: the allowance, enforced where the real cost is observed ----------------------------------
def test_the_allowance_is_enforced_on_the_real_cost_not_on_the_reported_one():
    """§4, and the reason M5 owns allowance accounting at all: `Scaffold.run_window` subtracts the
    figure the worker seam HANDS BACK, and under BYOK that figure comes from the same place the
    money does. Here the snapshot budget is generous (or stale) and the funded allowance is not, so
    the two limits disagree — and the one that binds must be the one holding the dollars.

    The miner overruns its funding by at most one call, which is the same bound `run_window`
    documents and for the same reason: a cost is not knowable until it is spent."""
    provider = FakeProvider(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.40})
    gateway = OwnerGateway(provider)
    gateway.fund(CHALLENGER, 1.00)
    results, ledger = arm(gateway, CHALLENGER, "w7-challenger", budget_usd=1_000.0,
                          slice_=tasks(5))

    assert len(ledger.receipts) == 3, "the funded allowance bound the arm, not the snapshot budget"
    assert ledger.spend_usd == pytest.approx(1.20)
    assert gateway.balance(CHALLENGER) == pytest.approx(-0.20), "at most one call of overrun"
    assert ledger.unfunded_calls > 0, (
        "an arm stopped by funding must be distinguishable from one stopped by a flaky pool (§5.8)")
    assert [r.graded_score for r in results[3:]] == [0.0, 0.0]
    assert audit(results, ledger.receipts, duel_id="w7-challenger", hotkey=CHALLENGER,
                 gateway_public_hex=gateway.public_hex).scoreable


def test_a_negative_cost_cannot_refill_a_miners_allowance():
    """`scaffold._priced`'s measured attack, one layer down and against real money: `balance -=
    dollars` with a minus sign REFILLS the allowance, and a NaN disables the exhaustion check
    permanently because it compares false against every threshold. `read_completion` refuses a cost
    that is absent or unreadable, which is a different question from whether a readable one is
    dollars — and this is the layer holding the dollars."""
    gateway = OwnerGateway(FakeProvider(prices={"cheap/model": -60.0}))
    gateway.fund(KING, 1.00)
    token = gateway.open_duel(KING, "w7-king")

    with pytest.raises(GatewayError, match="not dollars"):
        gateway.call(token, "cheap/model", TASK.prompt)
    assert gateway.balance(KING) == 1.00 and gateway.ledger("w7-king").receipts == []

    with pytest.raises(GatewayError, match="an allowance is dollars"):
        gateway.fund(KING, float("nan"))
    assert gateway.balance(KING) == 1.00


def test_a_provider_failure_stays_a_dead_rung_rather_than_a_dead_window():
    """§8b.3 survives the new seam. A gateway refusal reaches the scaffold through
    `Worker.complete`, whose caller catches broadly — so an unfunded or failed call is recorded as a
    failed step and the episode continues, exactly as a provider timeout is. The arm still
    reconciles: a call that was never priced contributes no receipt AND no spend."""
    class DownProvider:
        def chat(self, model_id, task_text, params):
            raise TimeoutError("the provider hung up")

    gateway = OwnerGateway(DownProvider())
    gateway.fund(KING, 10.0)
    token = gateway.open_duel(KING, "w7-king")
    episode = Scaffold(CATALOG, MockConductor(("DELEGATE cheap/model", "STOP")),
                       GatewayWorker(gateway, token), grade).run_episode(TASK,
                                                                        budget_remaining=10.0)

    assert episode.result.stopped_reason == "stop" and episode.result.spend_usd == 0.0
    assert episode.result.steps[0].success is False
    assert gateway.ledger("w7-king").receipts == []
    assert audit((episode.result,), (), duel_id="w7-king", hotkey=KING,
                 gateway_public_hex=gateway.public_hex).scoreable, (
        "a rung nobody was charged for is not unmetered spend")


# --- end to end, on the real client against a fake transport (no network, no spend) ---------------
CHAT_RESPONSE = {
    "id": "gen-1", "model": "cheap/model", "provider": "Together",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": SOLVED}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0123},
}


def test_a_duel_runs_on_the_owners_key_and_meters_what_came_back_off_the_wire():
    """The seam the validator wires: scaffold -> `GatewayWorker` -> `OwnerGateway` ->
    `OpenRouterClient` -> a fake socket. Nothing between the episode loop and the provider is
    mocked, so the metered spend the audit reconciles is the figure the client parsed off the wire,
    and the credential on the wire is the OWNER's — the miner has no key in the loop at all
    (§11-1), which is M5 in one assertion."""
    httpx = pytest.importorskip("httpx")
    sent: list[dict] = []

    def handler(request):
        sent.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = OpenRouterClient("sk-owner-key", transport=httpx.MockTransport(handler))
    gateway = OwnerGateway(client)
    gateway.fund(CHALLENGER, 10.0)
    results, ledger = arm(gateway, CHALLENGER, "w7-challenger")

    assert [call["headers"]["authorization"] for call in sent] == ["Bearer sk-owner-key"] * 3
    assert all(call["body"] == chat_body("cheap/model", task.prompt, WORKER_PARAMS)
               for call, task in zip(sent, task_order(tasks(3), "w7")))
    assert ledger.spend_usd == pytest.approx(3 * 0.0123)
    assert [r.prompt_hash for r in ledger.receipts] == [
        request_hash("cheap/model", task.prompt) for task in task_order(tasks(3), "w7")]
    assert audit(results, ledger.receipts, duel_id="w7-challenger", hotkey=CHALLENGER,
                 gateway_public_hex=gateway.public_hex).scoreable


def test_concurrent_calls_in_one_duel_mint_distinct_receipts_and_lose_no_money():
    """`Validator._arm` now drives this gateway from EPISODE_CONCURRENCY episodes at once, and it
    was written for a serial caller. Three of its updates are read-modify-writes that each decide a
    number somebody is paid or charged: the balance, the call counter behind `Receipt.ts`, and the
    receipt id. Two callers reading the same `len(receipts)` mint the SAME id, and an arm's spend is
    then reconciled against a receipt set with duplicate identities.

    A BARRIER PLUS AN AGGRESSIVE SWITCH INTERVAL, because neither alone is enough. Left to chance
    the threads do not overlap — the fake provider returns too fast — and the test passes against
    the broken code, which is how it was written the first time.

    WHAT THIS DOES AND DOES NOT BIND, stated because a concurrency test that overclaims is worse
    than none. Deriving `call_id` from `len(receipts)` instead of a reserved index makes it FAIL —
    that property is genuinely pinned. Removing the lock around the balance decrement does NOT make
    it fail: losing a read-modify-write needs an interleave inside a few bytecodes that cannot be
    forced from a test. The lock is justified by inspection there, not by this test.
    """
    import concurrent.futures
    import sys
    import threading

    # The critical section is a handful of bytecodes, so at the default switch interval the GIL
    # almost never interleaves it and the test passes against the broken code. Forcing a switch
    # after essentially every bytecode is what makes the race reachable at all.
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    workers = 32
    gate = threading.Barrier(workers)
    armed = False

    class Synchronised(FakeProvider):
        # `chat`, NOT `complete`. `OwnerGateway.call` reaches the provider through `chat`, so an
        # override named `complete` is never called and the barrier below never runs — the
        # interleave this test documents did not happen, and it passed anyway. Instrumented: zero
        # hits before the rename.
        def chat(self, model_id, task_text, params):
            if armed:                        # the warm-up call must not wait for 31 peers
                gate.wait()                  # every worker enters the money path together
            return super().chat(model_id, task_text, params)

    provider = Synchronised(answers={"cheap/model": SOLVED}, prices={"cheap/model": 0.01})
    gateway = OwnerGateway(provider)
    gateway.fund(KING, 100.0)
    token = gateway.open_duel(KING, "w1-arm")

    # WARMED FIRST, ON ITS OWN DUEL, AND THE BARRIER IS WHY THIS IS NOW NECESSARY. A cold gateway has no price for
    # this model, so the first caller reserves the whole balance and every other caller waits for it
    # to settle — which is the correct behaviour and makes 32 simultaneous cold calls impossible by
    # construction. With the barrier on the seam the gateway actually uses, that showed up as a
    # deadlock: the probe waiting on 31 peers who are waiting on the probe. Production has the same
    # shape and the same answer — `Validator._arm` runs its first batch as one episode, precisely
    # because there is no price to reason with before it. The warm-up runs on a separate duel so
    # the storm's ledger holds exactly the storm's receipts; the learned price is per MODEL, so a
    # different duel teaches it just as well.
    warm = gateway.open_duel(KING, "w1-warm")
    gateway.call(warm, "cheap/model", TASK.prompt)
    gateway.close(warm)
    before_storm = gateway.balance(KING)     # the warm-up spent from the same allowance
    armed = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: gateway.call(token, "cheap/model", TASK.prompt), range(workers)))

    sys.setswitchinterval(previous)
    ledger = gateway.close(token)
    ids = [r.call_id for r in ledger.receipts]
    assert len(ids) == workers, "every call must produce a receipt"
    assert len(set(ids)) == workers, f"receipt ids collided: {workers - len(set(ids))} duplicates"
    assert len({r.ts for r in ledger.receipts}) == workers, "two receipts share a timestamp"
    spent = sum(r.cost_usd for r in ledger.receipts)
    assert abs(gateway.balance(KING) - (before_storm - spent)) < 1e-9, (
        "a balance decrement was lost")


# --- the money path (§4) --------------------------------------------------------------------------
#
# The allowance machinery above was complete and had no way in: `fund()` had no caller outside these
# tests, and the balances were process memory. So the shipped daemon ran every arm at $0, every
# episode tripped the exhaustion check at its first step, and a window of all-zero scores was
# published as though the corpus carried no spread. These tests pin the two halves that were absent
# — a durable record, and the refusal that makes an empty wallet say so.

def test_a_credited_allowance_survives_the_process_that_credited_it(tmp_path):
    """The balances used to be a plain dict. A deploy, an OOM or a reboot zeroed every miner's
    allowance while the checkpoint's `<arm>.budget` still named the money, so §4's two enforcement
    layers disagreed: one said $X, the other $0, and every call was refused as unfunded."""
    journal = tmp_path / "allowances.jsonl"
    first = OwnerGateway(FakeProvider(), journal=journal)
    first.fund("hk_miner", 25.0, ref="extrinsic 0xabc")
    first.fund("hk_owner", 10.0, ref="invoice 7")
    first.fund("hk_miner", 5.0, ref="top-up")

    restarted = OwnerGateway(FakeProvider(), journal=journal)

    assert restarted.balance("hk_miner") == 30.0
    assert restarted.balance("hk_owner") == 10.0


def test_a_spend_survives_the_restart_as_well_as_the_credit(tmp_path):
    """Replaying credits alone would hand every arm its money back after a crash."""
    journal = tmp_path / "allowances.jsonl"
    provider = FakeProvider(answers={"cheap/model": "ok"}, prices={"cheap/model": 0.25})
    gateway = OwnerGateway(provider, journal=journal)
    gateway.fund("hk_miner", 10.0, ref="")
    token = gateway.open_duel("hk_miner", "w1-hk_miner")
    gateway.call(token, "cheap/model", "a task")
    spent = 10.0 - gateway.balance("hk_miner")
    assert spent > 0.0, "the fixture provider must cost something for this test to mean anything"

    restarted = OwnerGateway(FakeProvider(), journal=journal)

    assert restarted.balance("hk_miner") == pytest.approx(10.0 - spent)


def test_a_torn_final_row_is_the_process_that_died_not_a_bad_ledger(tmp_path):
    """Same rule as `Checkpoint.resume`, one layer down and against real dollars: a half-written
    row costs at most the movement that was in flight, where refusing to load would cost the
    balance of every miner on the subnet."""
    journal = tmp_path / "allowances.jsonl"
    gateway = OwnerGateway(FakeProvider(), journal=journal)
    gateway.fund("hk_miner", 12.0, ref="")
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"hotkey": "hk_miner", "usd": 999')

    assert OwnerGateway(FakeProvider(), journal=journal).balance("hk_miner") == 12.0


def test_the_reference_is_recorded_and_never_interpreted(tmp_path):
    """How the money arrived is the owner's business — an extrinsic, an invoice, staked alpha. The
    gateway records the words and reads none of them, which is what keeps the settlement rail out
    of the meter."""
    journal = tmp_path / "allowances.jsonl"
    gateway = OwnerGateway(FakeProvider(), journal=journal)
    gateway.fund("hk_miner", 1.0, ref="paid in bottle caps")

    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert row["ref"] == "paid in bottle caps"
    assert OwnerGateway(FakeProvider(), journal=journal).balance("hk_miner") == 1.0


def test_a_gateway_with_no_journal_still_works_and_writes_nothing(tmp_path):
    """The simulation and every test above construct one without a path."""
    gateway = OwnerGateway(FakeProvider())
    gateway.fund("hk_miner", 3.0)

    assert gateway.balance("hk_miner") == 3.0
    assert list(tmp_path.iterdir()) == []


def test_every_balance_is_readable_before_a_window_rather_than_after_it(tmp_path):
    gateway = OwnerGateway(FakeProvider(), journal=tmp_path / "allowances.jsonl")
    gateway.fund("hk_a", 2.0)
    gateway.fund("hk_b", 4.0)

    assert dict(gateway.balances()) == {"hk_a": 2.0, "hk_b": 4.0}


def test_every_refused_call_on_an_unfunded_arm_is_counted(tmp_path):
    """The counter that separates an unfunded arm from a dead pool, under the concurrency an arm
    actually runs at.

    WHAT THIS DOES NOT PIN, stated because the obvious reading is wrong: it does not prove the
    increment is under `_money`. Moving it back outside the lock leaves this test passing, because
    on a GIL build the three bytecodes of `obj.attr += 1` do not interleave — measured at a 1e-9
    switch interval, 0 of 48,000 increments lost. The lock is there for the free-threaded build
    where they do, and for the day this field grows a property; neither is reachable from a test on
    this interpreter. What this pins is the property that survives either way — sixteen refusals are
    sixteen, not fifteen — and the honest note is that the lock rests on review, not on this test.
    """
    gateway = OwnerGateway(FakeProvider())
    token = gateway.open_duel("hk_broke", "w1-hk_broke")

    threads = [threading.Thread(target=lambda: _refused(gateway, token)) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert gateway._ledger(token).unfunded_calls == 16


def _refused(gateway, token):
    with pytest.raises(GatewayError):
        gateway.call(token, "cheap/model", "a task")


# --- the allowance under concurrency --------------------------------------------------------------
#
# `Validator._arm` drives EPISODE_CONCURRENCY episodes through one gateway. The allowance used to be
# READ here and debited after the provider returned — a check-then-act straddling a network round
# trip — so every in-flight call saw the same balance and none had paid yet.

@dataclass
class SlowProvider(FakeProvider):
    """A pool that takes long enough for the calls to overlap, which is what a real one does."""

    delay: float = 0.05

    def chat(self, model_id, task_text, params):
        time.sleep(self.delay)
        return super().chat(model_id, task_text, params)


def _storm(gateway, token, threads: int = 16) -> int:
    done = []

    def call():
        try:
            gateway.call(token, "m", "a task")
            done.append(1)
        except GatewayError:
            pass

    workers = [threading.Thread(target=call) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return len(done)


def test_a_starved_allowance_overruns_by_at_most_one_call_even_under_concurrency():
    """The bound this module documents and `test_the_allowance...` asserts serially, now true when
    it is contended. Measured before the fix: $6.40 metered against a $0.50 allowance over sixteen
    concurrent calls, where one call costs $0.40."""
    gateway = OwnerGateway(SlowProvider(answers={"m": "ok"}, prices={"m": 0.40}))
    gateway.fund("hk", 0.50)
    token = gateway.open_duel("hk", "w1-arm")

    _storm(gateway, token)

    ledger = gateway.ledger("w1-arm")
    assert ledger.spend_usd <= 0.50 + 0.40 + 1e-9, "overran by more than one call"
    assert gateway.balance("hk") == pytest.approx(0.50 - ledger.spend_usd)


def test_a_funded_allowance_is_not_mistaken_for_an_empty_one_under_concurrency():
    """The failure the reservation could easily introduce and must not: the first call of a
    gateway's life has no price to reserve against, so it reserves the whole balance. Refusing every
    concurrent caller meanwhile would report a funded miner as unfunded — a worse lie than the
    overrun, and one that lands in `unfunded_calls`, the counter §5.8 relies on."""
    gateway = OwnerGateway(SlowProvider(answers={"m": "ok"}, prices={"m": 0.40}))
    gateway.fund("hk", 100.0)
    token = gateway.open_duel("hk", "w1-arm")

    started = time.monotonic()
    assert _storm(gateway, token) == 16
    elapsed = time.monotonic() - started

    assert gateway.ledger("w1-arm").unfunded_calls == 0
    # TIMED, BECAUSE THE OUTCOME ALONE IS NOT ENOUGH. Break `_release` — never clear `_probing`,
    # never notify — and the fifteen waiters still succeed: they simply block until
    # `PRICE_PROBE_WAIT_SECONDS` expires and then proceed. The assertions above pass and the suite
    # takes a minute longer, which is how a broken release path stayed invisible. The wait is the
    # thing being tested, so the wait is what must be asserted.
    assert elapsed < PRICE_PROBE_WAIT_SECONDS / 2, (
        f"the storm took {elapsed:.1f}s — the waiters timed out rather than being notified")
    assert gateway.balance("hk") == pytest.approx(100.0 - gateway.ledger("w1-arm").spend_usd)


def test_a_call_that_was_never_priced_gives_its_reservation_back():
    """A transport failure is data about a rung (§8b.3) and was never charged, so holding money for
    it would let a flaky pool drain an allowance nobody was billed for."""

    @dataclass
    class Broken(FakeProvider):
        def chat(self, model_id, task_text, params):
            raise RuntimeError("the pool hung up")

    gateway = OwnerGateway(Broken())
    gateway.fund("hk", 5.0)
    token = gateway.open_duel("hk", "w1-arm")

    with pytest.raises(RuntimeError):
        gateway.call(token, "m", "a task")

    assert gateway.balance("hk") == 5.0


def test_a_cheap_call_does_not_teach_the_gateway_that_expensive_calls_are_cheap():
    """The mixed-price hole in the first reservation, which restored the exact overrun it removed.

    The reservation was sized by a single running maximum over calls SETTLED so far — which is a
    lower bound on what is in flight, not an upper one. A gateway that had settled one $0.001 call
    then let sixteen concurrent $0.40 calls each reserve $0.001: measured, $6.40 metered against a
    $0.50 allowance, the same figure the fix was written to remove.

    Per model, a price this gateway has never settled reserves the whole balance, so the first call
    to a new model runs alone and the fifteen behind it are refused rather than funded on a cheaper
    model's price.
    """
    gateway = OwnerGateway(SlowProvider(answers={"cheap": "ok", "strong": "ok"},
                                        prices={"cheap": 0.001, "strong": 0.40}))
    gateway.fund("hk", 0.50)
    token = gateway.open_duel("hk", "w1-arm")
    gateway.call(token, "cheap", "a task")          # teaches it a CHEAP price

    def expensive():
        try:
            gateway.call(token, "strong", "a task")
        except GatewayError:
            pass

    workers = [threading.Thread(target=expensive) for _ in range(16)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    ledger = gateway.ledger("w1-arm")
    assert ledger.spend_usd <= 0.50 + 0.40 + 1e-9, (
        f"overran by more than one call: metered ${ledger.spend_usd:.2f} on a $0.50 allowance")
    assert gateway.balance("hk") == pytest.approx(0.50 - ledger.spend_usd)


def test_a_funded_allowance_still_runs_expensive_calls_concurrently():
    """The per-model rule must serialise only the FIRST call to a model, not every call to it."""
    gateway = OwnerGateway(SlowProvider(answers={"cheap": "ok", "strong": "ok"},
                                        prices={"cheap": 0.001, "strong": 0.40}))
    gateway.fund("hk", 100.0)
    token = gateway.open_duel("hk", "w1-arm")
    gateway.call(token, "cheap", "a task")

    done = []

    def expensive():
        gateway.call(token, "strong", "a task")
        done.append(1)

    workers = [threading.Thread(target=expensive) for _ in range(16)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert len(done) == 16
    assert gateway.ledger("w1-arm").unfunded_calls == 0


def test_a_credit_made_while_the_daemon_runs_reaches_it(tmp_path):
    """The instruction and the behaviour disagreed.

    `thirtyspokes-owner credit` builds its own gateway, appends a row and exits. The daemon
    replayed the journal once at construction and never again — so an owner topping up a miner
    mid-run, which is exactly what the miner guide says to do and what the validator's own refusal
    message asks for, changed nothing until somebody restarted the validator.
    """
    journal = tmp_path / "allowances.jsonl"
    daemon = OwnerGateway(FakeProvider(), journal=journal)
    assert daemon.balance("hk_miner") == 0.0

    OwnerGateway(FakeProvider(), journal=journal).fund("hk_miner", 12.5, ref="invoice 7")

    assert daemon.balance("hk_miner") == 0.0, "nothing should change until it looks"
    daemon.refresh()
    assert daemon.balance("hk_miner") == 12.5


def test_refresh_never_double_counts_this_gateways_own_spending(tmp_path):
    """The reason reading resumes at a byte offset rather than replaying the file.

    The daemon's own debits go into the same journal. A refresh that re-read from the start would
    apply them a second time and starve a miner who had spent nothing extra.
    """
    journal = tmp_path / "allowances.jsonl"
    provider = FakeProvider(answers={"m": "ok"}, prices={"m": 0.25})
    gateway = OwnerGateway(provider, journal=journal)
    gateway.fund("hk", 10.0)
    token = gateway.open_duel("hk", "w1-arm")
    gateway.call(token, "m", "a task")
    after_spending = gateway.balance("hk")

    for _ in range(3):
        gateway.refresh()

    assert gateway.balance("hk") == pytest.approx(after_spending)


def test_a_torn_row_is_picked_up_once_the_writer_finishes(tmp_path):
    journal = tmp_path / "allowances.jsonl"
    daemon = OwnerGateway(FakeProvider(), journal=journal)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"hotkey": "hk", "usd": 5.0')          # a half-written row

    daemon.refresh()
    assert daemon.balance("hk") == 0.0

    with journal.open("a", encoding="utf-8") as handle:
        handle.write('}\n')                                   # the writer finishes
    daemon.refresh()

    assert daemon.balance("hk") == 5.0


def test_a_journal_that_cannot_be_written_does_not_charge_the_miner(tmp_path):
    """The debit and its receipt must not come apart.

    The journal write is the only disk I/O on the money path, and it sits between the balance moving
    and the receipt being minted. A full disk therefore left a miner charged for a call that no
    receipt covers and no audit can reconcile — the money gone with no evidence it was spent.
    """
    journal = tmp_path / "allowances.jsonl"
    gateway = OwnerGateway(FakeProvider(answers={"m": "ok"}, prices={"m": 0.25}), journal=journal)
    gateway.fund("hk", 10.0)
    token = gateway.open_duel("hk", "w1-arm")
    gateway.call(token, "m", "warm the price")
    before = gateway.balance("hk")
    receipts_before = len(gateway.ledger("w1-arm").receipts)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with mock.patch.object(Path, "open", full_disk):
        with pytest.raises(GatewayError, match="allowance journal"):
            gateway.call(token, "m", "a task")

    assert gateway.balance("hk") == pytest.approx(before), "the miner was charged for a lost call"
    assert len(gateway.ledger("w1-arm").receipts) == receipts_before


def test_a_reservation_is_an_upper_bound_on_the_call_it_covers(tmp_path):
    """Every rule about the reservation's SIZE survived mutation before this test existed — dropping
    the `min(..., balance)` cap, learning the smallest call instead of the largest, learning once and
    never updating, never learning at all. All five mutants passed the whole file.

    The property they all break is one sentence: a call must never settle for more than was held for
    it, because the hold is the only thing standing between concurrency and an overrun.
    """
    provider = FakeProvider(answers={"m": "ok"}, prices={"m": 0.10})
    gateway = OwnerGateway(provider)
    gateway.fund("hk", 100.0)
    token = gateway.open_duel("hk", "w1-arm")

    gateway.call(token, "m", "cheap first")             # learns $0.10
    provider.prices = {"m": 0.40}                       # the same model gets dearer
    gateway.call(token, "m", "then dearer")             # overruns by 0.30, and LEARNS 0.40

    before = gateway.balance("hk")
    gateway.call(token, "m", "now covered")
    spent = before - gateway.balance("hk")

    assert gateway._ceiling["m"] == pytest.approx(0.40), (
        "the ceiling must learn the LARGEST call settled, not the smallest or the first")
    assert spent <= gateway._ceiling["m"] + 1e-9, "a call settled for more than was held for it"


def test_the_reservation_never_exceeds_what_is_actually_there():
    """The cap that reproduces the serial bound: taking only what exists is what drives the balance
    to zero and refuses the rest, instead of promising money the miner does not have."""
    gateway = OwnerGateway(FakeProvider(answers={"m": "ok"}, prices={"m": 0.10}))
    gateway.fund("hk", 100.0)
    token = gateway.open_duel("hk", "w1-arm")
    gateway.call(token, "m", "learn a price")

    # Observed WHILE the call is in flight, which is the only moment the cap is doing anything: a
    # reservation larger than the balance drives it negative and promises money the miner does not
    # have, and by the time the call settles the hold has come back and the evidence is gone.
    seen = []

    class Watching(FakeProvider):
        def chat(self, model_id, task_text, params):
            seen.append(gateway.balance("hk"))
            return super().chat(model_id, task_text, params)

    gateway.provider = Watching(answers={"m": "ok"}, prices={"m": 0.10})
    gateway._ceiling["m"] = 1_000.0                     # a ceiling far above the balance
    gateway.call(token, "m", "still fine")

    assert seen and min(seen) >= 0.0, (
        f"the reservation exceeded the balance: it went to {min(seen)} mid-flight")


def test_a_corrupt_row_mid_journal_is_refused_rather_than_silently_dropped(tmp_path):
    """A torn tail and a corrupt ledger are not the same thing.

    The first version read them the same way — stop at the first unreadable row — so a journal
    holding a $10 credit, one damaged line and a $5 credit read back as $10, with the later credit
    gone and no error anywhere. These balances are what every miner is paid and charged against, so a
    ledger that cannot be read whole must say so.
    """
    journal = tmp_path / "allowances.jsonl"
    OwnerGateway(FakeProvider(), journal=journal).fund("hk", 10.0)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"hotkey": "hk", "usd": not-a-number}\n')      # damage, with rows to follow
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"hotkey": "hk", "op": "credit", "usd": 5.0}\n')

    with pytest.raises(GatewayError, match="unreadable"):
        OwnerGateway(FakeProvider(), journal=journal)


def test_a_torn_final_row_is_still_just_a_torn_final_row(tmp_path):
    """The tail case must keep working: a half-written last line is a process that died mid-write,
    not damage, and the movement it describes never completed."""
    journal = tmp_path / "allowances.jsonl"
    OwnerGateway(FakeProvider(), journal=journal).fund("hk", 7.0)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"hotkey": "hk", "usd": 3')                    # no newline: still being written

    assert OwnerGateway(FakeProvider(), journal=journal).balance("hk") == 7.0


def test_an_unreadable_cost_returns_the_hold_and_frees_the_probe(tmp_path):
    """`cost_usd` crosses a seam this module does not control, so `float()` on it can raise — and
    outside the recovery block it raised past every recovery path at once.

    Measured before this: one unparseable cost took a $10.00 allowance to $0.00, left $10.00
    stranded in `_held` forever, and kept `_probing` set so every later caller waited out the full
    probe timeout and was then refused as unfunded.
    """
    class Unreadable(FakeProvider):
        def chat(self, model_id, task_text, params):
            return Completion(text="x", served_model=model_id, provider="mock",
                              finish_reason="stop", tokens_in=1, tokens_out=1,
                              cost_usd="not-a-number")

    gateway = OwnerGateway(Unreadable())
    gateway.fund("hk", 10.0)
    token = gateway.open_duel("hk", "w1-arm")

    with pytest.raises(GatewayError, match="cannot read"):
        gateway.call(token, "m", "a task")

    assert gateway.balance("hk") == 10.0, "the hold leaked"
    assert gateway._held.get("hk", 0.0) == 0.0, "the reservation was never released"
    assert gateway._probing is False, "the price probe was stranded for every later caller"


def test_a_credit_is_on_the_disk_and_not_just_in_the_kernel(tmp_path):
    """`flush` moves a row out of Python's buffer into the kernel's; a power loss between there and
    the platter loses it. The claim made to an owner is that a credit survives a restart, and only
    fsync makes that true of a reboot."""
    journal = tmp_path / "allowances.jsonl"
    synced = []
    real = os.fsync

    def watched(fd):
        synced.append(fd)
        return real(fd)

    with mock.patch.object(os, "fsync", watched):
        OwnerGateway(FakeProvider(), journal=journal).fund("hk", 5.0)

    assert synced, "the credit was written but never synced"
