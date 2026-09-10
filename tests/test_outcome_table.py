"""The window's shared outcome table (docs/WHITEPAPER.md §5.2c, D17) — the properties that make
replaying real draws exact rather than approximate.

The claim the table rests on is a sufficiency claim: a Conductor never changes what a worker
answers, only which `(task, model, attempt)` keys get drawn and in what order, so a table of real
draws scores any Conductor on the slice exactly as buying every draw again would — minus the noise
of buying it again. Every test here is one face of that: identical policies are byte-identical, a
hit reaches no provider and no grader, a RETRY is a new draw, a grader failure is never a row, a
provider failure is one, the money is unchanged for the miner and smaller for the owner, and the
§2.1 audit sentence still holds over an arm that was replayed.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pytest

from thirtyspokes.v3 import tools
from thirtyspokes.v3.gateway import GatewayWorker, OwnerGateway, audit
from thirtyspokes.v3.openrouter import Completion
from thirtyspokes.v3.scaffold import OutcomeTable, Scaffold
from thirtyspokes.v3.types import Catalog, CatalogEntry, TaskSpec
from thirtyspokes.v3.worker import MockWorker, NotAnOutcome, WorkerError

CATALOG = Catalog(entries=(
    CatalogEntry("cheap/model", 0.05, 0.20, 128_000),
    CatalogEntry("flaky/model", 0.30, 1.20, 200_000),
    CatalogEntry("strong/model", 2.50, 10.00, 400_000),
))
SOLVED = "SOLVED"


def grade(task: TaskSpec, answer: str) -> float:
    return 1.0 if answer == SOLVED else 0.0


def tasks(n: int) -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(task_id=f"tb-{i:04d}", benchmark="terminal-bench",
                          prompt=f"fix bug {i}", tools=()) for i in range(n))


@dataclass(frozen=True)
class Policy:
    """A Conductor that is a function of the rendered state, so it repeats per episode."""

    turns: tuple[str, ...]

    def act(self, prompt: str) -> str:
        return self.turns[min(prompt.count("\nstep "), len(self.turns) - 1)]


ESCALATE = ("DELEGATE cheap/model", "RETRY strong/model", "STOP")
POOL = {"cheap/model": "no", "strong/model": SOLVED}
PRICES = {"cheap/model": 0.02, "strong/model": 0.30}


def arm(worker, table, policy=Policy(ESCALATE), grader=grade, n: int = 3):
    scaffold = Scaffold(CATALOG, policy, worker, grader, table=table)
    return scaffold.run_window(tasks(n), nonce="w7", budget_usd=100.0)


# --- exactness ------------------------------------------------------------------------------------


def test_two_identical_conductors_replayed_are_byte_identical_and_the_second_reaches_no_provider():
    """THE HEADLINE PROPERTY, and a correctness gain before it is a saving: M3a measured
    always-cheapest duelled against a second draw of ITSELF clearing three of four verdict
    conditions. Against one table the two arms are the same rows, so they tie exactly."""
    table = OutcomeTable()
    first, second = MockWorker(answers=POOL, costs=PRICES), MockWorker(answers=POOL, costs=PRICES)

    one = arm(first, table)
    two = arm(second, table)

    assert [e.result for e in one] == [e.result for e in two]
    assert second.seen == [], "every delegate of the second arm was a hit"
    assert [(e.hits, e.fills) for e in one] == [(0, 2)] * 3
    assert [(e.hits, e.fills) for e in two] == [(2, 0)] * 3
    assert table.stats()["fills"] == 6 and table.stats()["hits"] == 6
    # The second arm was still CHARGED every delegate (§4 is unchanged); only the provider was
    # not asked twice.
    assert second.charged == pytest.approx(first.charged) == pytest.approx(3 * 0.32)


def test_a_hit_calls_neither_the_worker_nor_the_grader():
    """The grade is on the row, so grading runs once per key — the sandbox is the expensive half
    of a delegate on a code benchmark, and it is the half a replay skips entirely."""
    table = OutcomeTable()
    graded = []

    def counting_grade(task, answer):
        graded.append((task.task_id, answer))
        return grade(task, answer)

    arm(MockWorker(answers=POOL, costs=PRICES), table, grader=counting_grade)
    assert len(graded) == 6
    worker = MockWorker(answers=POOL, costs=PRICES)
    arm(worker, table, grader=counting_grade)
    assert len(graded) == 6 and worker.seen == []


def test_a_retry_of_the_same_model_on_the_same_task_is_attempt_two_and_fills_live():
    """`attempt` is what keeps the table a memo of DRAWS: a Conductor that asks the same model
    again is asking for a new sample, and handing back the first one would make retrying free and
    useless at once. The second arm then replays both attempts."""
    table = OutcomeTable()
    worker = MockWorker(answers={"cheap/model": "no"}, costs={"cheap/model": 0.02})

    arm(worker, table, policy=Policy(("DELEGATE cheap/model", "RETRY cheap/model", "STOP")), n=1)

    assert len(worker.seen) == 2
    assert sorted(row.attempt for row in table.rows()) == [1, 2]
    again = MockWorker(answers={"cheap/model": "no"}, costs={"cheap/model": 0.02})
    two = arm(again, table, policy=Policy(("DELEGATE cheap/model", "RETRY cheap/model", "STOP")), n=1)
    assert again.seen == [] and (two[0].hits, two[0].fills) == (2, 0)


def test_a_grader_failure_is_never_a_row_and_still_propagates():
    """§8b.3 and §6.3c through the table: the row is written only AFTER the benchmark's own grader
    returned, so a dead sandbox cannot leave a zero behind for every later arm to replay as the
    miner's miss. The failure still reaches the caller, which drops the task from both arms."""
    table = OutcomeTable()

    def broken(task, answer):
        raise RuntimeError("the sandbox container died")

    with pytest.raises(RuntimeError, match="sandbox"):
        arm(MockWorker(answers=POOL, costs=PRICES), table, grader=broken, n=1)
    assert len(table) == 0

    worker = MockWorker(answers=POOL, costs=PRICES)
    arm(worker, table, n=1)
    assert len(worker.seen) == 2 and len(table) == 2, "bought live once the grader is back"


def test_a_provider_failure_is_a_row_and_every_arm_sees_the_same_dead_rung():
    """§8b.3: a worker failing is an OUTCOME about that rung. Stored, so the invariant every arm
    faces the same draws beats a retry — a second arm whose provider would have answered still
    sees the failure the first arm saw, and the reveal publishes how often that happened."""
    table = OutcomeTable()
    down = MockWorker(costs={"flaky/model": 0.30}, failures=frozenset({"flaky/model"}))
    policy = Policy(("DELEGATE flaky/model", "STOP"))

    one = arm(down, table, policy=policy, n=2)
    up = MockWorker(answers={"flaky/model": SOLVED}, costs={"flaky/model": 0.30})
    two = arm(up, table, policy=policy, n=2)

    assert [e.result for e in one] == [e.result for e in two]
    assert all(e.result.graded_score == 0.0 for e in two) and up.seen == []
    stats = table.stats()
    assert stats["failed_rows"] == 2 and stats["failure_rate"] == 1.0
    # The reason travels with the row: the arm that replayed the failure says why the fill died.
    assert two[0].result.steps[0].failure == "WorkerError: flaky/model is down"
    assert all(row.failure == "WorkerError: flaky/model is down" for row in table.rows())


def test_a_metering_refusal_is_a_dead_step_for_that_arm_but_never_a_row():
    """`worker.NotAnOutcome`: the meter refusing says nothing about the model, so it must not be
    replayed as the model failing for every later arm. The refused arm records a dead step; the
    next arm buys the key live."""
    table = OutcomeTable()

    class Broke:
        def complete(self, model_id, task_text, params):
            raise NotAnOutcome("no allowance left")

    one = arm(Broke(), table, n=1)
    assert one[0].result.graded_score == 0.0 and len(table) == 0
    assert one[0].result.steps[0].failure == "refused: no allowance left"

    worker = MockWorker(answers=POOL, costs=PRICES)
    two = arm(worker, table, n=1)
    assert two[0].result.graded_score == 1.0 and len(worker.seen) == 2


def test_the_audit_sentence_holds_over_a_replayed_arm():
    """§2.1's recomputation, `text == tools.compose(task.prompt, observations)`, over every request
    row of an arm that reached no provider: the rows are logged on a hit exactly as on a miss, one
    per turn of the delegate the row records, so a reader of `Episode.requests` cannot tell a
    replayed arm from a bought one and the invariant is checkable on both."""
    table = OutcomeTable()
    arm(MockWorker(answers=POOL, costs=PRICES), table)
    replayed = arm(MockWorker(answers=POOL, costs=PRICES), table)
    by_id = {t.task_id: t for t in tasks(3)}

    assert sum(len(e.requests) for e in replayed) == 6
    for episode in replayed:
        for request in episode.requests:
            assert request.text == tools.compose(by_id[request.task_id].prompt,
                                                 request.observations)


def test_the_table_persists_every_fill_and_a_reopened_one_re_buys_nothing(tmp_path):
    """§8b.5 for the draws: a restart must not re-spend what was spent, so every fill is on disk
    before `put` returns and a table opened on the file starts from its rows."""
    path = tmp_path / "window-1" / "outcomes.jsonl"
    arm(MockWorker(answers=POOL, costs=PRICES), OutcomeTable(path))

    reopened = OutcomeTable(path)
    assert len(reopened) == 6 and {row.key for row in reopened.rows()} == {
        (t.task_id, m, 1) for t in tasks(3) for m in ("cheap/model", "strong/model")}
    worker = MockWorker(answers=POOL, costs=PRICES)
    two = arm(worker, reopened)
    assert worker.seen == [] and all(e.hits == 2 for e in two)


# --- the money, through the owner's gateway ---------------------------------------------------------


@dataclass
class FakeProvider:
    answers: dict
    prices: dict
    calls: int = 0
    seen: list = field(default_factory=list)

    def chat(self, model_id, task_text, params):
        self.calls += 1
        self.seen.append(model_id)
        if model_id == "flaky/model":
            raise WorkerError("down")
        return Completion(text=self.answers.get(model_id, ""),
                          cost_usd=float(self.prices.get(model_id, 0.0)), tokens_in=100,
                          tokens_out=50, finish_reason="stop", served_model=model_id,
                          provider="Fake")


def metered_arm(gateway, hotkey, duel_id, table, policy=Policy(ESCALATE)):
    token = gateway.open_duel(hotkey, duel_id)
    scaffold = Scaffold(CATALOG, policy, GatewayWorker(gateway, token), grade, table=table)
    episodes = scaffold.run_window(tasks(3), nonce="w7", budget_usd=100.0)
    return [e.result for e in episodes], gateway.close(token)


def test_a_replayed_delegate_is_charged_like_a_live_one_and_the_provider_is_paid_once():
    """§4 and D5 unchanged for the miner, the owner's outlay cut to the fills: `charged_usd` equals
    `recorded_usd` for both arms, `provider_usd` is the whole bill for the arm that bought and zero
    for the arm that replayed, and the provider saw one arm's worth of calls."""
    provider = FakeProvider(POOL, PRICES)
    gateway = OwnerGateway(provider)
    gateway.fund("king", 10.0)
    gateway.fund("challenger", 10.0)
    table = OutcomeTable()

    king, king_ledger = metered_arm(gateway, "king", "w7-king", table)
    challenger, chal_ledger = metered_arm(gateway, "challenger", "w7-challenger", table)

    assert provider.calls == 6 and king == challenger
    for results, ledger, hotkey in ((king, king_ledger, "king"),
                                    (challenger, chal_ledger, "challenger")):
        verdict = audit(results, ledger.receipts, duel_id=ledger.duel_id, hotkey=hotkey,
                        gateway_public_hex=gateway.public_hex,
                        replayed=frozenset(ledger.replayed))
        assert verdict.scoreable
        assert verdict.charged_usd == pytest.approx(verdict.recorded_usd) == pytest.approx(0.96)
        assert verdict.provider_usd <= verdict.charged_usd
    assert king_ledger.provider_usd == pytest.approx(0.96) and not king_ledger.replayed
    assert chal_ledger.provider_usd == 0.0 and len(chal_ledger.replayed) == 6
    assert gateway.balance("challenger") == pytest.approx(10.0 - 0.96)
    # A replay's receipt pins the same reply bytes the fill's receipt pinned.
    fills = {r.prompt_hash: r.response_hash for r in king_ledger.receipts}
    assert all(fills[r.prompt_hash] == r.response_hash for r in chal_ledger.receipts)


def test_a_replay_against_an_exhausted_allowance_is_refused_and_counted_not_given_away():
    """A hit a miner cannot pay for is not a model's answer they got for free: the meter refuses
    it as it refuses a live call, the arm records a dead step, `unfunded_calls` says why (§5.8),
    and the table is untouched."""
    provider = FakeProvider(POOL, PRICES)
    gateway = OwnerGateway(provider)
    gateway.fund("king", 10.0)
    table = OutcomeTable()
    metered_arm(gateway, "king", "w7-king", table)

    broke, ledger = metered_arm(gateway, "broke", "w7-broke", table)

    assert provider.calls == 6 and all(r.spend_usd == 0.0 for r in broke)
    assert all(r.graded_score == 0.0 for r in broke)
    assert ledger.unfunded_calls == 6 and not ledger.receipts   # two refused delegates per task
    assert table.stats()["fills"] == 6


def test_the_persisted_row_round_trips_with_its_receipts_and_tokens(tmp_path):
    """What a row holds (§5.2c): the reply, the transcript, the price, the token sums and the ids
    of the receipts the fill minted — enough to point every replay at the receipts that paid."""
    provider = FakeProvider(POOL, PRICES)
    gateway = OwnerGateway(provider)
    gateway.fund("king", 10.0)
    path = tmp_path / "outcomes.jsonl"
    metered_arm(gateway, "king", "w7-king", OutcomeTable(path))

    rows = {row.key: row for row in OutcomeTable(path).rows()}
    row = rows[(tasks(3)[0].task_id, "strong/model", 1)]
    assert row.text == SOLVED and row.grade == 1.0 and row.cost_usd == pytest.approx(0.30)
    assert (row.tokens_in, row.tokens_out) == (100, 50)
    assert len(row.receipt_ids) == 1 and row.receipt_ids[0].startswith("w7-king/")
    assert row.filled_at > 0 and not row.failed


def test_the_failure_reason_survives_the_checkpoint_and_the_published_trace():
    """`StepRecord.failure` is what the reveal of 2026-09-08 lacked: it must round-trip through the
    daemon's checkpoint and D15's trace JSON, and be absent from a record written before it."""
    from thirtyspokes.v3.validator import episode_from_json, episode_json

    down = MockWorker(costs={"flaky/model": 0.30}, failures=frozenset({"flaky/model"}))
    episode = arm(down, OutcomeTable(), policy=Policy(("DELEGATE flaky/model", "STOP")), n=1)[0]
    body = episode_json(episode.result)
    assert body["steps"][0]["failure"] == "WorkerError: flaky/model is down"
    assert episode_from_json(body) == episode.result
    del body["steps"][0]["failure"]
    assert episode_from_json(body).steps[0].failure is None
