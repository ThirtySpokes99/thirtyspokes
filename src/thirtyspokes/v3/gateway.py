"""The owner-run gateway — where a model ID is made to mean one thing (§11-1, the build plan M5).

§11-1 calls this the largest remaining hole, and it is a hole in the PAIRING rather than in the
plumbing. §4 has every miner register their own OpenRouter key and the validator spend against it.
OpenRouter supports BYOK and per-key provider preferences, so `openai/gpt-5.4` on the king's key and
`openai/gpt-5.4` on the challenger's key need not be the same object: a different upstream
deployment, a different quantisation, a different fallback chain, a different price. §6.2 already
freezes the catalog so both arms face the same *action space*; without this module they still do not
face the same *actions*, and §5.2 computes a crown from the difference. Under D4's open catalog that
is not one model but hundreds of them.

Three leaks close together once the OWNER holds the credentials, and each is worth naming because
only the first is the one §11-1 describes:

1. **Substitution.** One key, and one request body — `openrouter.chat_body` with its pinned
   `PROVIDER_ROUTING` — for every arm of every duel. A miner has no key to configure, so there is
   nothing left to configure differently.

2. **The slice leaks through the miner's own billing log.** Under §4's BYOK every worker call the
   validator makes appears, prompt by prompt, in the MINER's OpenRouter request log. D12 draws the
   slice *after* challengers commit precisely so nobody can know which tasks will be scored — and a
   miner reading their own log knows the entire scored slice, live, for every window they are in.
   The sitting king is in all of them (§5.2a), so the incumbent gets the `(task -> best model)`
   profiling table M7a prices at ~$60,000 for free and permanently, and ossification stops being a
   risk and becomes a subscription. Owner-held credentials put that log on the owner's account; what
   a miner gets is a `Receipt`, which carries a hash of the request and never the request.

3. **Cost is trusted where it is returned but must be enforced where it is spent.**
   `Scaffold.run_window` subtracts the number the worker seam hands back (§4), and under BYOK that
   number arrives from the same place the money does. Here the gateway debits the REAL cost against
   a pre-funded balance, so §4's allowance is enforced twice: once on the figure the arm reported,
   once on the figure the provider charged. `final_b` prices spend (§5.1b), so those two disagreeing
   is a mispriced crown rather than an accounting nuisance.

WHAT IS REUSED, AND WHAT IS DELIBERATELY NOT.

* From `gateway/`, the un-forgeable-cost machinery, taken whole: `signing` (ed25519 + canonical
  hashing) and `Receipt` (gateway-signed, hotkey-bound, real cost from the provider). What is NOT
  reused is `MeteringGateway`, `CallRequest` and `gateway_call`, because their authentication model
  is INVERTED here. There the miner was the caller and signed every request to prove it was theirs,
  and that gateway's documented weakness is that a miner-run gateway cannot be trusted on metering.
  In v3 the owner is the caller and the miner is only the PAYER: it holds no key, makes no request
  and signs nothing, so a `CallRequest.miner_sig` would be the owner signing on the miner's behalf —
  a signature over a claim nobody disputes. The per-duel token below replaces it, and it authorises
  spending rather than authenticating a caller.
* From `v3/openrouter.py`, the entire provider seam: `chat_body` (which is where §2.1 becomes bytes
  and where `PROVIDER_ROUTING` is pinned) and `OpenRouterClient`. THIS MODULE MUST NOT BUILD ITS OWN
  REQUEST, and that is not a preference about duplication. A second body builder here would be a
  second pin, and two pins in one package drift — at which point the gateway and the worker seam
  send different requests and the arms this module exists to make comparable stop being comparable
  by way of the fix.
"""

from __future__ import annotations

import json
import os
import threading

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from typing import Protocol

from ..gateway import signing
from ..gateway.receipts import Receipt
from ..gateway.signing import Signer
from .config import PRICE_PROBE_WAIT_SECONDS
from .openrouter import Completion, chat_body
from .types import EpisodeResult
from .worker import WORKER_PARAMS, NotAnOutcome, WorkerError

# Float association order, nothing more: the arm sums per episode and the ledger sums per call, so
# the same dollars added in a different order differ in the last bits. Nine decimal places is seven
# orders of magnitude below the cent M1 exit 6 asks the accounting to reconcile to, so a real
# discrepancy — a call made off-gateway, or a cost reported rather than metered — cannot hide
# underneath it.
SPEND_TOLERANCE_USD = 1e-9

# The PAYER's key refused, not the model (D18): 401 is a key OpenRouter does not recognise, 402 an
# account with nothing left. Both are meter refusals — `unfunded_calls`, `GatewayError`, no row in
# the window's outcome table (§5.2c) — because a provider that never ran said nothing about the rung.
# 403 (moderation) is deliberately not here: the provider looked at the task and refused it, which
# is an outcome of that model on that task and every arm should see the same one.
KEY_REFUSALS = frozenset({401, 402})


class GatewayError(NotAnOutcome):
    """The gateway refused: no such token, a closed one, an exhausted allowance, a cost that is not
    dollars.

    Every one of these reaches the scaffold through `Worker.complete`, whose caller catches broadly
    and records a dead rung (§8b.3, `scaffold._call_worker`). That is the right outcome — an arm
    that cannot be metered must not be scored for the work — but it is also SILENT, which is why
    `DuelLedger.unfunded_calls` exists: a miner who ran out of money and a provider that was down
    look identical in an episode trace, and §5.8 requires the first to be visible in the reveal.
    """


class Provider(Protocol):
    """The far side of the meter: `openrouter.OpenRouterClient`, on the owner's key.

    `chat` rather than `complete` (the `worker.Worker` contract) because a receipt needs what the
    narrower seam deliberately drops — the token counts, and the served model and provider that are
    §11-1's own audit evidence. `complete` gives the scaffold exactly what §2.1 permits it; the
    meter sits underneath and sees the whole response.
    """

    def chat(self, model_id: str, task_text: str,
             params: Mapping[str, object]) -> Completion:
        """One worker call, with the whole response — cost, tokens, and who served it."""


def request_hash(model_id: str, task_text: str) -> str:
    """What a receipt carries INSTEAD of the request — and the §2.1 audit, extended past the seam.

    Over the WHOLE body rather than the prompt alone, which is why it fills `Receipt.prompt_hash`
    while not being named for it: §2.1's claim is about everything that reached a worker model, so a
    hash that covered the text but not the pinned provider block or the model would leave the two
    fields this module exists to pin outside the evidence.

    `scaffold.Episode.requests` proves what the validator *intended* to send; this proves what went
    to the provider. Anyone holding the public benchmark can recompute it and compare — so the
    invariant becomes checkable by a third party from the published record, while the record itself
    still discloses no task text to the miner being billed for it (leak 2 above).
    """
    return signing.sha256_hex(chat_body(model_id, task_text, WORKER_PARAMS))


@dataclass
class DuelLedger:
    """One arm's bill: what the gateway spent on it, and what it refused to spend.

    `unfunded_calls` is not bookkeeping. When the real cost outruns the figure the arm reported,
    §4's allowance is exhausted at the gateway rather than in `run_window` — so the remaining tasks
    fill with dead rungs and score zero, and not one of them carries `budget_exhausted`. That is a
    verdict decided by funding wearing the face of a flaky pool, which is precisely the confusion
    §5.8 requires the reveal to avoid, and this counter is the only place the difference exists.
    """

    duel_id: str
    hotkey: str
    receipts: list[Receipt] = field(default_factory=list)
    unfunded_calls: int = 0
    closed: bool = False
    # The `call_id`s among `receipts` that `replay` minted (§5.2c): real debits for delegates the
    # window had already bought, so they are in the arm's bill and NOT in the provider's.
    replayed: list[str] = field(default_factory=list)
    # Which endpoint answered each FILL — `Completion.provider` and `served_model` per `call_id` —
    # the §11-1 evidence an arm run on a miner's own key (D18) is audited against. A replay has no
    # entry: it was served to whoever filled it.
    served: list[dict] = field(default_factory=list)

    @property
    def spend_usd(self) -> float:
        """The metered total — the one spend figure in the system nobody had to be trusted for.
        What the arm was CHARGED, hits included; `provider_usd` is what the provider was paid."""
        return sum(receipt.cost_usd for receipt in self.receipts)

    @property
    def provider_usd(self) -> float:
        """The owner's real outlay on this arm: every receipt that is not a replay (§5.2c)."""
        replayed = set(self.replayed)
        return sum(receipt.cost_usd for receipt in self.receipts
                   if receipt.call_id not in replayed)


def allowance_journal(state: Path) -> Path:
    """Where the gateway's allowances live, derived from `--state` so nobody has to pass it twice.

    The owner credits with `thirtyspokes-owner credit`; the validator replays the same file when it
    builds its gateway. Deriving both from `--state` is what stops the two naming different files
    and the daemon starting with a wallet the owner believes it filled.
    """
    return state / "allowances.jsonl"


class OwnerGateway:
    """The owner's meter: owner credentials, miner money, one request body for every arm.

    It holds three things and they are deliberately separate: a balance is a miner's pre-funded
    allowance and belongs to a HOTKEY; a `DuelLedger` is one arm's spend and belongs to a DUEL; a
    token authorises the second to draw on the first. Keeping them apart is what makes "this arm
    cost that much" and "this miner paid that much" two verifiable statements rather than one
    aggregate.
    """

    def __init__(self, provider: Provider, signer: Signer | None = None,
                 journal: Path | str | None = None) -> None:
        self.provider = provider
        self._gw = signer or Signer()
        # WITHOUT THIS FILE THE ALLOWANCES ARE PROCESS MEMORY, and that is not a small thing: a
        # deploy, an OOM or a reboot would zero every miner's balance while `Checkpoint`'s
        # `<arm>.budget` still names the money, so the two layers §4 deliberately doubles would
        # disagree — one says $X, the other $0 — and every call would be refused as unfunded.
        # Append-only and replayed on construction, the same shape and the same torn-tail rule as
        # `Checkpoint`, because it is the same problem one layer down and against real dollars.
        #
        # `ref` IS NEVER PARSED. It is free text naming whatever the owner settled — an extrinsic
        # hash, an invoice number, a stake receipt — which is how this stays true to the promise
        # `fund` makes below: how the money arrives is out of this module. Committing to a
        # settlement rail here would put the owner's payment plumbing inside the meter.
        self._journal = Path(journal) if journal is not None else None
        # How far into the journal these balances already account for, so `refresh` can apply what
        # another process appended without re-applying this one's own debits.
        self._consumed = 0
        self._balances: dict[str, float] = {}
        self._ledgers: dict[str, DuelLedger] = {}
        self._tokens: dict[str, str] = {}
        # D18: a hotkey that runs on ITS OWN provider (`bind`), and the window its allowance was
        # last capped for. Everyone else runs on `self.provider`, the owner's.
        self._providers: dict[str, Provider] = {}
        self._caps: dict[str, int] = {}
        self._calls = 0
        # ONE LOCK OVER THE MONEY. This class was written for a serial caller and is now driven by
        # `EPISODE_CONCURRENCY` episodes of one arm at once (`Validator._arm`). Four of its updates
        # are read-modify-writes and every one of them decides a number somebody is paid or charged:
        # the balance decrement below, `self._calls` (which stamps `Receipt.ts`), and the `call_id`
        # taken from `len(ledger.receipts)` — two concurrent calls read the same length and mint the
        # SAME receipt id, so an arm's spend can be reconciled against a receipt set with duplicate
        # identities and a lost decrement leaves a miner charged less than they spent.
        self._money = threading.Lock()
        # Receipt indices are RESERVED, not derived from `len(receipts)`. Computing the id under the
        # lock and appending after it still collides: two callers both read the same length before
        # either appends. A counter handed out under the same lock gives the identical 0,1,2...
        # sequence the serial version produced, and gives it exactly once.
        self._issued: dict[str, int] = {}
        # THE RESERVATION'S SIZE, learned rather than configured: the largest call this gateway has
        # actually settled. None until the first one, which is what makes the first call reserve the
        # whole balance and so run alone — the same reasoning `Validator._arm` gives for its first
        # batch being one episode, because before it there is no price to reason with.
        # PER MODEL, because one number across every model is a lower bound on what is in flight,
        # not an upper one. The first version learned a single running max over SETTLED calls: a
        # gateway that had settled one $0.001 call then let sixteen concurrent $0.40 calls each
        # reserve $0.001, and metered $6.40 against a $0.50 allowance — the exact figure the fix was
        # written to remove. A model whose price this gateway has never settled reserves the whole
        # balance, which serialises the first call to it and nothing else.
        self._ceiling: dict[str, float] = {}
        # Dollars currently RESERVED by calls in flight, per hotkey. A balance of zero means two
        # completely different things — the miner has spent their allowance, or their allowance is
        # entirely held by calls that have not settled yet — and `unfunded_calls` exists (§5.8) to
        # say the first. Without this the second was reported as the first.
        self._held: dict[str, float] = {}
        # The first call of a gateway's life has no price to reserve against, so it reserves the
        # whole balance and every concurrent caller would find nothing left. Refusing them would
        # report a funded miner as unfunded; letting them through would be the check-then-act this
        # reservation exists to remove. So they WAIT for the first call to settle, which is the only
        # thing that can teach them a price. Bounded, so a hung provider degrades to the ordinary
        # refusal rather than parking an arm forever.
        self._priced = threading.Condition(self._money)
        self._probing = False
        self._replay()

    def _release(self, probing: bool) -> None:
        """End this call's turn as the price probe. Called with `_money` held, on every exit."""
        if probing:
            self._probing = False
            self._priced.notify_all()

    def refresh(self) -> None:
        """Apply movements another process appended since this one last looked.

        WITHOUT THIS A CREDIT IS INVISIBLE UNTIL A RESTART. `thirtyspokes-owner credit` builds its
        own gateway, appends a row and exits; the daemon replayed the file once at construction and
        never again. So an owner topping up a miner mid-run — which is exactly what the miner guide
        tells them to do, and what the validator's own refusal message asks for — changed nothing
        until somebody restarted the validator. The instruction and the behaviour disagreed.

        Reading resumes at the byte this process has already accounted for, which is what makes it
        safe to call repeatedly: this gateway's own debits advance that mark as they are written, so
        a refresh applies exactly the rows somebody else wrote and never double-counts its own. A
        torn final line does not advance it, so the row is re-read once whoever was writing finishes.
        """
        with self._money:
            self._replay()

    def _replay(self) -> None:
        """Apply journal rows from `_consumed` onward. Call with `_money` held."""
        if self._journal is None or not self._journal.exists():
            return
        raw = self._journal.read_bytes()[self._consumed:]
        chunks = raw.split(b"\n")
        consumed = self._consumed
        for index, chunk in enumerate(chunks):
            line = chunk.decode("utf-8", "replace")
            if not line.strip():
                consumed += len(chunk) + 1
                continue
            try:
                row = json.loads(line)
                hotkey, usd = str(row["hotkey"]), float(row["usd"])
            except (ValueError, KeyError, TypeError) as exc:
                # A TORN TAIL AND A CORRUPT LEDGER ARE NOT THE SAME THING, and reading them the same
                # way lost money in silence. An unreadable row that is the LAST thing in the file
                # and has no newline after it is a process that died mid-write: not a completed
                # movement, so it is left unconsumed and re-read when the writer finishes.
                #
                # An unreadable row with rows AFTER it is damage, and the first version simply
                # stopped there — so a journal holding a $10 credit, one corrupt line, and a $5
                # credit read back as $10, with the later credit gone and no error. The balances are
                # what a miner is paid and charged against; a ledger that cannot be read whole must
                # say so rather than be quietly reinterpreted.
                if index == len(chunks) - 1:
                    break
                raise GatewayError(
                    f"the allowance journal {self._journal} is unreadable at byte {consumed} "
                    f"({exc}); it is the record of what every miner has been credited and charged, "
                    f"so it is refused rather than partially applied") from exc
            if row.get("op") == "cap":
                # D18: a window's allowance on the miner's own key SETS the balance rather than
                # adding to it — last window's unspent cap is not money anyone holds — and the
                # debits journaled after it are what a restart subtracts (`bind`).
                self._balances[hotkey] = usd
                self._caps[hotkey] = int(row.get("window", 0))
            else:
                self._balances[hotkey] = self._balances.get(hotkey, 0.0) + usd
            consumed += len(chunk) + 1
        self._consumed = min(consumed, self._journal.stat().st_size)

    def _record(self, hotkey: str, usd: float, **fields: object) -> None:
        """Append one movement. Called with `_money` held wherever the caller already holds it."""
        if self._journal is None:
            return
        self._journal.parent.mkdir(parents=True, exist_ok=True)
        with self._journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"hotkey": hotkey, "usd": usd, **fields},
                                    sort_keys=True) + "\n")
            handle.flush()
            # FLUSHED IS NOT DURABLE. `flush` moves the row out of Python's buffer and into the
            # kernel's; a power loss between there and the disk loses it, and this file is the
            # record of what every miner has been credited and charged. The claim made to an owner
            # is that a credit survives a restart, and only fsync makes that true of a reboot. One
            # sync per model call is a bounded cost against money that cannot be reconstructed.
            os.fsync(handle.fileno())
        # This row is already in `_balances`, so mark it consumed: a later `refresh` must not add it
        # a second time.
        self._consumed = self._journal.stat().st_size

    @property
    def public_hex(self) -> str:
        """The key every receipt is verified against — published, so `audit` needs no secret."""
        return self._gw.public_hex

    def fund(self, hotkey: str, usd: float, *, ref: str = "") -> float:
        """Credit a miner's pre-funded allowance. Returns the new balance.

        HOW the money arrives is out of this module: the owner credits a hotkey against whatever
        payment it settled. Since D18 this is the owner's GRANT rather than the ordinary rail — a
        miner's arm runs on the key they registered (`bind`) — and a credit on top of a bound
        hotkey simply widens that window's allowance.

        Written `not >= 0` so the one comparison also refuses NaN: a NaN balance compares false
        against every threshold, so the exhaustion check in `call` would pass forever and the
        allowance would be unbounded (the shape of `scaffold._priced`, one layer down and against
        real money rather than a score).
        """
        amount = float(usd)
        if not amount >= 0.0:
            raise GatewayError(f"cannot fund {hotkey} with {amount}: an allowance is dollars")
        # Under the money lock like every other balance write. `fund` was the one read-modify-write
        # on `_balances` outside it — harmless while nothing called it, and a lost credit the moment
        # anything does while an arm is running.
        with self._money:
            self._balances[hotkey] = self._balances.get(hotkey, 0.0) + amount
            self._record(hotkey, amount, op="credit", ref=str(ref))
            return self._balances[hotkey]

    def bind(self, hotkey: str, provider: Provider, cap_usd: float, *, window: int) -> float:
        """Run this hotkey's calls on ITS OWN provider, against `cap_usd` for `window` (D18).

        Returns the balance the window's arm starts against. The cap is journaled as a movement
        that SETS the balance (`op: cap`), so a restart mid-window replays it and then the debits
        made since — the same figure `Checkpoint.snapshot` threads forward, from the other side.
        Binding the same window twice re-points the provider and leaves the money alone: that is
        the restart, and re-capping it would refund what the arm already spent.

        The provider is per HOTKEY and not per duel because the king is the same miner in every
        window it reigns: the crown's arm and the challenger's arm that won it run on one key.
        """
        cap = float(cap_usd)
        if not cap >= 0.0:
            raise GatewayError(f"cannot bind {hotkey} to a cap of {cap}: an allowance is dollars")
        with self._money:
            self._replay()
            self._providers[hotkey] = provider
            if self._caps.get(hotkey) == int(window):
                return self.balance(hotkey)
            self._balances[hotkey] = cap
            self._caps[hotkey] = int(window)
            self._record(hotkey, cap, op="cap", window=int(window))
            return cap

    def own_key(self, hotkey: str) -> bool:
        """Does this hotkey run on a provider of its own (D18)? Published per arm in the reveal."""
        return hotkey in self._providers

    def balance(self, hotkey: str) -> float:
        return self._balances.get(hotkey, 0.0)

    def balances(self) -> Mapping[str, float]:
        """Every allowance, for the owner to read before a window rather than after it."""
        return dict(self._balances)

    def open_duel(self, hotkey: str, duel_id: str) -> str:
        """Mint the token this one arm spends through. It is scoped to the duel, SERVER-SIDE.

        The token is an unguessable handle (`secrets`) and the gateway keeps its scope in its own
        map, rather than the token carrying a signed scope that its holder presents. Two
        consequences, both wanted:

        * **A token cannot be replayed against another duel, because it does not name one** — it IS
          the duel. There is no duel argument on `call` to disagree with, so charging duel B's arm
          to duel A's allowance is not a check that can be forgotten; it is a call that cannot be
          written. A bearer token carrying its own scope would put that check back.
        * **A token can be revoked.** `close` ends the arm, and every later call on it is refused —
          which is what stops a finished arm's token from quietly funding a later one.

        Unguessable rather than derived (`f"tok-{duel_id}"` would do everything above): a token
        authorises spending a miner's real money, so anything that can reach the gateway must not be
        able to construct one.

        A `duel_id` is opened ONCE. Re-opening would give two arms one ledger and one identity, and
        their receipts — which are scoped by `call_id` — would become indistinguishable, so the
        spend of the second could be presented as the spend of the first.
        """
        if "/" in duel_id:
            raise GatewayError(
                f"duel id {duel_id!r} contains '/', which scopes a receipt's call_id")
        if duel_id in self._ledgers:
            raise GatewayError(f"duel {duel_id!r} has already been opened; one arm, one ledger")
        token = token_hex(16)
        self._tokens[token] = duel_id
        self._ledgers[duel_id] = DuelLedger(duel_id=duel_id, hotkey=hotkey)
        return token

    def close(self, token: str) -> DuelLedger:
        """End the arm and return its bill. Every later call on this token is refused."""
        ledger = self._ledger(token)
        ledger.closed = True
        return ledger

    def ledger(self, duel_id: str) -> DuelLedger:
        """One arm's bill, by duel — what the owner reconciles and publishes."""
        if duel_id not in self._ledgers:
            raise GatewayError(f"no duel {duel_id!r} was ever opened")
        return self._ledgers[duel_id]

    def call(self, token: str, model_id: str, task_text: str) -> tuple[str, Receipt]:
        """One worker call, metered. THE ONLY WAY A PROVIDER IS REACHED IN A DUEL.

        There is no params argument, and that is M5's exit 2 in one line: provider and model
        identity come from the owner's pinned body (`openrouter.chat_body` over `WORKER_PARAMS`),
        never from a caller's settings. The two arguments left are the two `worker.Worker.complete`
        takes and for the same reason (§2.1) — a validated model ID and the benchmark's own text —
        with the third dropped because the owner pins it rather than accepting it.

        The allowance is checked BEFORE the call and debited with the provider's real figure after,
        so an arm overruns its funding by at most one call — the same bound `Scaffold.run_window`
        documents for §4, and for the same reason: a cost is not knowable until it is spent. The
        hard stop underneath both is the owner's own provider credit.

        A transport failure raises `WorkerError` out of the provider seam and is left to travel: it
        is data about that rung (§8b.3), the call was never priced, and a ledger row for it would be
        a receipt for work nobody was charged for.
        """
        ledger = self._ledger(token)
        if ledger.closed:
            raise GatewayError(f"duel {ledger.duel_id!r} is closed; its token buys nothing")
        # RESERVE, DO NOT MERELY CHECK. The allowance used to be read here and debited after the
        # provider returned, with a network round trip in between — a check-then-act straddling the
        # slowest thing in the system. `Validator._arm` drives `EPISODE_CONCURRENCY` episodes
        # through one gateway, so sixteen calls read the same balance before any of them had paid:
        # measured, $6.40 metered against a $0.50 allowance, where this module's own docstring and
        # `test_the_allowance_stops_an_arm...` both say the overrun is at most ONE call. The
        # validator's batch sizing bounded it in practice, which is a different layer's heuristic
        # standing in for this layer's rule about somebody's money.
        #
        # So the money moves BEFORE the call, under the lock, and settles to the true cost after.
        # The reservation is the largest call settled so far, capped by what is actually there —
        # capped, because taking only what exists reproduces the serial bound exactly: the balance
        # reaches zero, every other in-flight call is refused, and the arm overruns by at most the
        # one call that was already committed.
        with self._money:
            while model_id not in self._ceiling and self._probing:
                if not self._priced.wait(timeout=PRICE_PROBE_WAIT_SECONDS):
                    break
            balance = self.balance(ledger.hotkey)
            if balance <= 0.0:
                outstanding = self._held.get(ledger.hotkey, 0.0)
                if outstanding > 0.0:
                    # NOT UNFUNDED — FULLY RESERVED. The miner's money is held by calls in flight
                    # and will mostly come back when they settle, so counting this in
                    # `unfunded_calls` would report a funded miner as broke in the one field §5.8
                    # relies on to tell those apart. It is congestion, and it says so.
                    raise GatewayError(
                        f"{ledger.hotkey} has ${outstanding:.4f} reserved by calls still in "
                        f"flight and nothing free right now; this is congestion, not an exhausted "
                        f"allowance — retry once they settle")
                # The fourth read-modify-write on this contended object, and the one the lock's own
                # comment above enumerates three of. Latent under CPython's GIL and a lost update
                # the moment this runs free-threaded — on the counter that is the only evidence
                # distinguishing an unfunded arm from a dead pool.
                ledger.unfunded_calls += 1
                raise GatewayError(
                    f"{ledger.hotkey} has no allowance left: {ledger.duel_id} has metered "
                    f"${ledger.spend_usd:.4f} (§4)")
            known = self._ceiling.get(model_id)
            held = balance if known is None else min(known, balance)
            probing = known is None
            self._probing = self._probing or probing
            self._balances[ledger.hotkey] = balance - held
            self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) + held

        # D18: a bound hotkey's calls go to ITS provider; the request body is still the owner's
        # pinned one (`chat_body` over `WORKER_PARAMS` and `PROVIDER_ROUTING`), which is what keeps
        # the two arms' bytes identical even when the keys are not.
        provider = self._providers.get(ledger.hotkey, self.provider)
        try:
            completion = provider.chat(model_id, task_text, WORKER_PARAMS)
            # PARSED INSIDE THE TRY, and that is not tidiness. `cost_usd` crosses a seam this module
            # does not control, so `float()` on it can raise — and outside this block it raised past
            # every recovery path at once: the hold was never returned, `_held` kept it forever, and
            # `_probing` stayed set so every later caller waited out the full probe timeout and was
            # then refused. Measured: one unparseable cost took a $10.00 allowance to $0.00 and
            # stranded the probe.
            dollars = float(completion.cost_usd)
        except (TypeError, ValueError) as exc:
            with self._money:
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + held
                self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) - held
                self._release(probing)
            raise GatewayError(
                f"{model_id} reported a cost this gateway cannot read ({completion.cost_usd!r}); "
                f"the call is refused and {ledger.hotkey} is not charged for it") from exc
        except WorkerError as exc:
            # Never priced either way, so the hold comes back. Then the one distinction D18 adds:
            # a 401/402 is the PAYER refused, and it is counted and raised as the meter's refusal
            # (`GatewayError`, not an outcome) — an empty account must not be memoised by the
            # window's table as `model_id` failing for every later arm (§5.2c).
            with self._money:
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + held
                self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) - held
                self._release(probing)
                if exc.status in KEY_REFUSALS:
                    ledger.unfunded_calls += 1
            if exc.status in KEY_REFUSALS:
                raise GatewayError(
                    f"{ledger.hotkey}'s key was refused by the provider (HTTP {exc.status}) on "
                    f"{ledger.duel_id}: the account is unfunded or the key is dead, and the rung "
                    f"is not an outcome of {model_id}") from exc
            raise
        except BaseException:
            # A transport failure was never priced (§8b.3), so the hold must come back or a flaky
            # pool would drain an allowance nobody was charged for.
            with self._money:
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + held
                self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) - held
                self._release(probing)
            raise
        if not dollars >= 0.0:
            # `balance -= dollars` with a minus sign REFILLS the allowance, and a NaN disables the
            # check above permanently by comparing false against every threshold. Measured in the
            # scaffold before it was closed there (`_priced`): three episodes at -$60 against a
            # $1.00 window all ran to completion having spent nothing the budget could see.
            # `read_completion` refuses a cost that is absent or unreadable, which is a different
            # question from whether a readable one is dollars; this is the layer holding the money,
            # so it asks the second.
            # Refused, and the hold comes back with it: like a transport failure this call was
            # never priced, so keeping the reservation would charge a miner for a refusal.
            with self._money:
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + held
                self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) - held
                self._release(probing)
            raise GatewayError(f"{model_id} reported a cost of {dollars}, which is not dollars")

        with self._money:
            # Settle: give back what was held and take what it cost. The journal records the true
            # cost, never the hold — a reservation is this process's bookkeeping, while the journal
            # is what a restart replays and what the owner reconciles against receipts.
            self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + held - dollars
            self._held[ledger.hotkey] = self._held.get(ledger.hotkey, 0.0) - held
            self._ceiling[model_id] = max(self._ceiling.get(model_id, 0.0), dollars)
            self._release(probing)
            try:
                self._record(ledger.hotkey, -dollars, op="debit", duel=ledger.duel_id,
                             model=model_id)
            except OSError as exc:
                # THE DEBIT AND ITS RECEIPT MUST NOT COME APART. This write is the only disk I/O on
                # the money path, and it sits between the balance moving and the receipt being
                # minted — so a full disk left a miner charged for a call that no receipt covers and
                # no audit can reconcile. Put the money back and refuse the call instead: the arm
                # records a dead rung, which is honest data about that rung (§8b.3), and the owner
                # gets an error naming the actual fault rather than a mysterious shortfall.
                # Undo the DEBIT only. The settle above already returned the hold, and returning
                # it twice would credit a miner for money that was never taken.
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + dollars
                raise GatewayError(
                    f"could not record the debit for {ledger.duel_id} in the allowance journal "
                    f"({exc}); the call is refused and {ledger.hotkey} is not charged for it") \
                    from exc
            self._calls += 1
            index = self._issued.get(ledger.duel_id, 0)
            self._issued[ledger.duel_id] = index + 1
            call_id = f"{ledger.duel_id}/{index}"
            stamp = self._calls
        # `call_id` is scoped to the duel because `Receipt` has no duel field and a receipt that
        # does not name its arm can be presented as evidence for a different one — a cheap arm's
        # receipts standing in for an expensive arm's spend, which is `final_b` mispriced by
        # whichever arm had the smaller bill. Scoping the identity the gateway already mints costs
        # one f-string; a second signature over the same bytes would cost a second key to verify.
        # (`completion.provider` — which endpoint actually served — has no field here either, and
        # belongs in the published trace where D15 puts the rest of the step record.)
        receipt = Receipt(
            call_id=call_id, hotkey=ledger.hotkey,
            model=model_id, prompt_hash=request_hash(model_id, task_text),
            response_hash=signing.sha256_hex(completion.text),
            tokens_in=completion.tokens_in, tokens_out=completion.tokens_out,
            cost_usd=dollars, ts=stamp).signed_by(self._gw)
        with self._money:
            ledger.receipts.append(receipt)
            ledger.served.append({"call_id": call_id, "model": model_id,
                                  "provider": completion.provider,
                                  "served_model": completion.served_model})
        return completion.text, receipt

    def replay(self, token: str, model_id: str, task_text: str, cost_usd: float, *,
               tokens_in: int = 0, tokens_out: int = 0, response_hash: str = "") -> Receipt:
        """Debit a REPLAYED delegate (§5.2c): the same money as `call`, and no provider.

        The row was bought once, by whichever arm drew the key first this window; this arm made the
        same choice and is charged the row's priced cost exactly as if it had bought it — so §4's
        allowance, D5, exhaustion, `remaining -= spend` and `final_b` do not know the table exists,
        and the miner guide's "you pay for the calls your Conductor makes" stays true. Only the
        owner's provider bill is untouched. The refusals are `call`'s: a closed token, an allowance
        at zero (counted in `unfunded_calls`, §5.8), a cost that is not dollars, a journal that
        cannot take the row. No reservation, because the price is known before the money moves.

        The receipt is a real receipt for a real debit — scoped to the duel, signed by the gateway,
        summed by `audit` like every other — and its id is listed in `DuelLedger.replayed`, which
        is the only thing that lets the reveal split what the arm was charged from what the provider
        was paid. `prompt_hash` is over the delegate's last request, the same body `call` would have
        sent; `response_hash` is the stored reply's.
        """
        ledger = self._ledger(token)
        if ledger.closed:
            raise GatewayError(f"duel {ledger.duel_id!r} is closed; its token buys nothing")
        dollars = float(cost_usd)
        if not dollars >= 0.0:
            raise GatewayError(f"a replayed row priced {dollars}, which is not dollars")
        with self._money:
            balance = self.balance(ledger.hotkey)
            if balance <= 0.0:
                ledger.unfunded_calls += 1
                raise GatewayError(
                    f"{ledger.hotkey} has no allowance left: {ledger.duel_id} has metered "
                    f"${ledger.spend_usd:.4f} (§4)")
            self._balances[ledger.hotkey] = balance - dollars
            try:
                self._record(ledger.hotkey, -dollars, op="debit", duel=ledger.duel_id,
                             model=model_id, replay=True)
            except OSError as exc:
                self._balances[ledger.hotkey] = self.balance(ledger.hotkey) + dollars
                raise GatewayError(
                    f"could not record the replayed debit for {ledger.duel_id} in the allowance "
                    f"journal ({exc}); the delegate is refused and {ledger.hotkey} is not charged "
                    f"for it") from exc
            self._calls += 1
            index = self._issued.get(ledger.duel_id, 0)
            self._issued[ledger.duel_id] = index + 1
            call_id = f"{ledger.duel_id}/{index}"
            stamp = self._calls
        receipt = Receipt(
            call_id=call_id, hotkey=ledger.hotkey, model=model_id,
            prompt_hash=request_hash(model_id, task_text), response_hash=response_hash,
            tokens_in=int(tokens_in), tokens_out=int(tokens_out), cost_usd=dollars,
            ts=stamp).signed_by(self._gw)
        with self._money:
            ledger.receipts.append(receipt)
            ledger.replayed.append(call_id)
        return receipt

    def _ledger(self, token: str) -> DuelLedger:
        if token not in self._tokens:
            raise GatewayError("unknown duel token")
        return self._ledgers[self._tokens[token]]


@dataclass(frozen=True)
class GatewayWorker:
    """The scaffold's `Worker`, pointed at the owner's gateway. One arm, one token.

    The params are checked and then discarded rather than forwarded: `call` rebuilds the body from
    the pin. The check is not redundant with that — it is the §2.1 tripwire. A caller that starts
    passing per-call settings is a caller that has grown a channel, and a gateway that sent the pin
    regardless would hide it: the arm would run on parameters nobody chose while the scaffold
    recorded that it had sent the ones it did.
    """

    gateway: OwnerGateway
    token: str
    # The receipts each THREAD minted since it last asked (`minted`). A delegate runs on one thread
    # start to finish (`Scaffold.run_episode`), so what a thread minted during one delegate is that
    # delegate's receipts, and the row the scaffold fills names them (§5.2c). Thread-local because
    # one `GatewayWorker` serves `EPISODE_CONCURRENCY` episodes at once, whose receipts interleave.
    _minted: threading.local = field(default_factory=threading.local, compare=False, repr=False)

    def complete(self, model_id: str, task_text: str,
                 params: Mapping[str, object]) -> tuple[str, float]:
        if dict(params) != dict(WORKER_PARAMS):
            raise GatewayError(
                f"worker params {dict(params)!r} are not the pinned {dict(WORKER_PARAMS)!r}; the "
                "owner pins the request (§2.1), so a per-call setting is a channel, not an option")
        text, receipt = self.gateway.call(self.token, model_id, task_text)
        self._remember(receipt)
        return text, receipt.cost_usd

    def charge(self, model_id: str, task_text: str, cost_usd: float, *, tokens_in: int = 0,
               tokens_out: int = 0, response_text: str | None = None) -> None:
        """A replayed delegate's debit (§5.2c): `OwnerGateway.replay`, on this arm's token. The
        reply is hashed here exactly as `call` hashes a live one, so a replay's receipt and the
        fill's receipt pin the same bytes."""
        self._remember(self.gateway.replay(
            self.token, model_id, task_text, cost_usd, tokens_in=tokens_in, tokens_out=tokens_out,
            response_hash="" if response_text is None else signing.sha256_hex(response_text)))

    def minted(self) -> tuple[Receipt, ...]:
        """The receipts this thread minted since it last asked, and forget them."""
        rows = tuple(getattr(self._minted, "rows", ()))
        self._minted.rows = []
        return rows

    def _remember(self, receipt: Receipt) -> None:
        rows = getattr(self._minted, "rows", None)
        if rows is None:
            rows = self._minted.rows = []
        rows.append(receipt)


@dataclass(frozen=True)
class Audit:
    """Whether an arm can be scored at all, and the two spend figures that had to agree.

    `metered_usd` is what the gateway CHARGED the arm — every receipt, replays included — and is
    the figure that must equal `recorded_usd`. `charged_usd` is the same number under the name the
    reveal splits it by, and `provider_usd` is the part the provider was actually paid, which is
    at most `charged_usd` and, under §5.2c, usually less.
    """

    scoreable: bool
    reason: str
    metered_usd: float
    recorded_usd: float
    calls: int
    charged_usd: float = 0.0
    provider_usd: float = 0.0


def audit(results: Sequence[EpisodeResult], receipts: Sequence[Receipt], *, duel_id: str,
          hotkey: str, gateway_public_hex: str,
          replayed: Collection[str] = frozenset()) -> Audit:
    """M5 exit 1: a call without a gateway receipt is UNSCOREABLE — stated as arithmetic.

    "Every worker call carries a receipt" cannot be checked call-for-call against the episode trace,
    and trying to would get §8b.3 backwards: a rung the provider never answered is logged as a
    request and has no receipt, because no call was ever priced. What IS exact is the money.
    `spend_usd` is what `final_b` prices (§5.1b), so the property that matters is that every dollar
    the arm was charged with came out of a gateway receipt:

        sum(arm's recorded spend)  ==  sum(verified receipts for this duel)

    An arm that worked off-gateway has spend with nothing behind it; an arm whose seam under-reports
    has receipts with nothing in front of them. Both fail the same comparison, and neither can be
    scored — refusing is the only safe verdict, because the alternative is a crown priced on a
    number nobody metered.

    Receipts are checked before they are summed, in the vocabulary `gateway/verify.py` already uses:
    a receipt the gateway did not sign is not evidence, one from another duel is not this arm's, and
    one billed to another hotkey is not this miner's. A third party can run this: the gateway key is
    public and the receipts are what the reveal publishes.
    """
    for receipt in receipts:
        if not receipt.verify(gateway_public_hex):
            return _unscoreable("bad_gateway_signature", results, receipts, replayed)
        if not receipt.call_id.startswith(f"{duel_id}/"):
            return _unscoreable("receipt_from_another_duel", results, receipts, replayed)
        if receipt.hotkey != hotkey:
            return _unscoreable("receipt_hotkey_mismatch", results, receipts, replayed)

    metered = sum(receipt.cost_usd for receipt in receipts)
    recorded = sum(result.spend_usd for result in results)
    if abs(metered - recorded) > SPEND_TOLERANCE_USD:
        return _unscoreable("spend_not_metered", results, receipts, replayed)
    return Audit(scoreable=True, reason="ok", metered_usd=metered, recorded_usd=recorded,
                 calls=len(receipts), charged_usd=metered,
                 provider_usd=_provider_usd(receipts, replayed))


def _provider_usd(receipts: Sequence[Receipt], replayed: Collection[str]) -> float:
    """What the provider was paid: every receipt that is not a replay (§5.2c)."""
    skip = set(replayed)
    return sum(receipt.cost_usd for receipt in receipts if receipt.call_id not in skip)


def _unscoreable(reason: str, results: Sequence[EpisodeResult],
                 receipts: Sequence[Receipt], replayed: Collection[str] = frozenset()) -> Audit:
    """The two figures are reported even when the arm is refused: the reveal has to say by how much
    they disagreed, or "unscoreable" is an accusation with no evidence attached."""
    metered = sum(receipt.cost_usd for receipt in receipts)
    return Audit(scoreable=False, reason=reason, metered_usd=metered,
                 recorded_usd=sum(result.spend_usd for result in results), calls=len(receipts),
                 charged_usd=metered, provider_usd=_provider_usd(receipts, replayed))
