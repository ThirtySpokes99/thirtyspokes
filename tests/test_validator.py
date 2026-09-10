"""The real validator daemon (the build plan M9) — the phase order once the seams carry money.

`test_simulate.py` holds the mechanism's properties against mock I/O. This file holds the ones
that only EXIST once there is a chain that remembers, a store that can hand back the wrong bytes, a
meter that can refuse a call, and a process that can die halfway through an arm. Each is a property
of the daemon's ORDER or of its durability, so none of them can be tested where the module lives.

NOTHING HERE IS A MOCK OF THE THING UNDER TEST. The chain is `v3.chain.MockChain` (which recycles
UIDs and honours the rate limit, because a mock that did neither would certify nothing), the store is
the production `store.S3Bucket` over an in-memory client, the meter is the production
`gateway.OwnerGateway` over a scripted provider, and the one-shot ledger is the production
`access.Mailbox` on a temp file. Only the far side of each seam is replaced — no network, no key, no
GPU, no money.

The world is `test_simulate.py`'s: prices are deliberately NOT a capability ranking, so
`mock/premium` is the dearest model and only the second strongest and the price-ordered cascade never
tries `mock/mid`, which is strong enough for everything at half the price. Without that, a window in
which the cheapest model dominates would make every test here pass for the wrong reason
(ROUTING_MEASUREMENTS: five times, the blocker was the pool).
"""

from __future__ import annotations

import hashlib
import io
import json
import socket
import threading
import time
from unittest import mock
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from thirtyspokes.gateway import signing
from thirtyspokes.v3 import simulate, validator as daemon
from thirtyspokes.v3.access import Mailbox, Registration, encode_ss58
from thirtyspokes.v3.admission import describe
from thirtyspokes.v3.archetypes import Copier
from thirtyspokes.v3.benchmarks.mock import MockBenchmark, worker_answers
from thirtyspokes.v3.chain import Commitment, MockChain, ReadySignal
import thirtyspokes.v3.validator as validator_module
from thirtyspokes.v3.conductor import Conductor, MockConductor
from thirtyspokes.v3.config import (DUEL_WALL_CLOCK_REASON, DUEL_WALL_CLOCK_SECONDS,
                                     EPISODE_CONCURRENCY,
                                     EXCLUDED_REASON,
                                     MAX_DUELS_PER_WINDOW, MAX_QUEUE_DEPTH, FUTILE_REASON)
from thirtyspokes.v3.devkit import suite_grade
from thirtyspokes.v3.emissions import KING0
from thirtyspokes.v3.gateway import OwnerGateway
from thirtyspokes.v3.openrouter import Completion
from thirtyspokes.v3.scaffold import Scaffold
from thirtyspokes.v3.simulate import (DEFERRED, DUELLED, REFUSED, SKIPPED, UNQUEUED,
                                      _GraderGuard, learned_router)
from thirtyspokes.v3.store import S3Bucket, build_manifest, upload_tree
from thirtyspokes.v3.tools import PROTOCOL
from thirtyspokes.v3.types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from thirtyspokes.v3.window import task_order
from thirtyspokes.v3.validator import (Cadence, Checkpoint, Crown, History, Owner, Validator,
                                        ValidatorError, WindowUnavailable, chain_beacon,
                                        check_launch, format_reveal, reveal_path, served_name)
from thirtyspokes.v3.window import build as build_window
from thirtyspokes.v3.window import window_path
from thirtyspokes.v3.worker import MockWorker, WorkerError

CATALOG = Catalog(entries=(
    CatalogEntry("mock/cheap", 0.20, 0.80, 128_000),
    CatalogEntry("mock/mid", 0.40, 1.60, 200_000),
    CatalogEntry("mock/heavy", 1.00, 4.00, 300_000),
    CatalogEntry("mock/premium", 1.20, 4.80, 400_000),
))
STRENGTH = {"mock/cheap": 1, "mock/mid": 3, "mock/heavy": 3, "mock/premium": 2}
COST = {"mock/cheap": 0.04, "mock/mid": 0.08, "mock/heavy": 0.16, "mock/premium": 0.20}

LIVE = (MockBenchmark("frontier-swe", STRENGTH, n_tasks=4, difficulty=(3, 3)),
        MockBenchmark("nl2repo", STRENGTH, n_tasks=4, difficulty=(2, 2)),
        MockBenchmark("programbench", STRENGTH, n_tasks=4, difficulty=(2, 2)))
ROUTABLE = ("nl2repo", "programbench")

# A window the gate must refuse: every model is one subgoal in twenty-five apart, so the spread is
# real, perfectly separable from zero, and below `POWER_SPREAD_FLOOR` (ROUTING_MEASUREMENTS §15).
DEAD = (MockBenchmark("flat-a", STRENGTH, n_tasks=4, difficulty=(2, 2), subgoals=25),
        MockBenchmark("flat-b", STRENGTH, n_tasks=4, difficulty=(2, 2), subgoals=25))

BUCKET = "v3-submissions"
NETUID = 99
# `window_blocks` must exceed MockChain's 100-block rate limit (§8b.6) and `immunity_blocks` must
# cover three windows (§8b.1) — the two gates `check_launch` refuses a launch on.
CADENCE = Cadence(genesis_block=1_000, window_blocks=200, windows=(1, 2, 3), immunity_blocks=600)


# --- the far side of each seam ---------------------------------------------------------------------


class FakeS3:
    """An in-memory bucket speaking boto3's S3 surface, so `store.S3Bucket` itself is under test."""

    PAGE = 100

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.readonly = False

    def put_object(self, *, Bucket, Key, Body, Metadata=None, ContentType=None) -> None:
        if self.readonly:
            raise ConnectionError("bucket is unreachable")
        self.objects[Key] = (bytes(Body), dict(Metadata or {}))

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key][0]), "Metadata": dict(self.objects[Key][1])}

    def head_object(self, *, Bucket, Key):
        body, metadata = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": dict(metadata)}

    def list_objects_v2(self, *, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = keys.index(ContinuationToken) if ContinuationToken else 0
        page = keys[start:start + self.PAGE]
        response = {"Contents": [{"Key": k, "Size": len(self.objects[k][0])} for k in page]}
        if start + self.PAGE < len(keys):
            response["IsTruncated"] = True
            response["NextContinuationToken"] = keys[start + self.PAGE]
        return response

    def delete_object(self, *, Bucket, Key) -> None:
        self.objects.pop(Key, None)

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, Config=None) -> None:
        self.objects[Key] = (Path(Filename).read_bytes(),
                             dict((ExtraArgs or {}).get("Metadata", {})))

    def download_file(self, Bucket, Key, Filename, Config=None) -> None:
        Path(Filename).write_bytes(self.objects[Key][0])


@dataclass
class Pool:
    """The provider behind the owner's gateway: `Provider.chat`, scripted per model."""

    answers: Mapping[str, str]
    costs: Mapping[str, float]
    failures: frozenset[str] = frozenset()
    calls: int = 0

    def chat(self, model_id: str, task_text: str, params: Mapping[str, object]) -> Completion:
        self.calls += 1
        if model_id in self.failures:
            raise WorkerError(f"{model_id} is down")
        return Completion(text=self.answers.get(model_id, ""), served_model=model_id,
                          provider="mock", finish_reason="stop", tokens_in=1, tokens_out=1,
                          cost_usd=float(self.costs.get(model_id, 0.0)))


@dataclass
class Spy:
    """A Conductor that counts the turns it was asked for — how "once per window" is measured."""

    inner: Conductor
    calls: int = 0
    on_act: object = None

    def act(self, prompt: str) -> str:
        self.calls += 1
        if self.on_act is not None:
            self.on_act()
        return self.inner.act(prompt)


@dataclass
class Clock:
    """The wall-clock seam (§8b.2), driven by hand so a two-hour bound is testable in microseconds."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def overrun(self) -> None:
        self.now = DUEL_WALL_CLOCK_SECONDS * 10


@dataclass
class Sandbox:
    """A grader that dies on one task, and only once `armed` — flaky infra, not a policy (§8b.3)."""

    grade: object
    armed: bool = False
    dead: str | None = None

    def __call__(self, task, answer) -> float:
        if self.armed and self.dead in (None, task.task_id):
            self.dead = task.task_id
            raise RuntimeError("sandbox died")
        return self.grade(task, answer)


@dataclass
class Inviting:
    """An adapter that appends the read protocol and declares no verb — the disagreement no
    signature and no hash can see, because `tools.PROTOCOL` lives INSIDE the prompt they
    authenticate (`r2egym.r2e_prompt` appends it, which is what keeps §2.1's audit byte-true)."""

    inner: MockBenchmark

    @property
    def name(self) -> str:
        return self.inner.name

    def load(self):
        return tuple(replace(t, prompt=t.prompt + PROTOCOL) for t in self.inner.load())

    def tools(self):
        return self.inner.tools()

    def environment(self, task):
        return self.inner.environment(task)

    def grade(self, submission, task):
        return self.inner.grade(submission, task)


def address(name: str) -> str:
    return encode_ss58(_key(name).public_key().public_bytes_raw())


def _key(name: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(name.encode()).digest())


# --- the harness ------------------------------------------------------------------------------------


@dataclass
class Harness:
    """One owner-run validator with every seam wired to an offline far side."""

    validator: Validator
    chain: MockChain
    client: FakeS3
    store: S3Bucket
    pool: Pool
    gateway: OwnerGateway
    owner_signer: object
    root: Path
    tree: Path
    registry: dict = field(default_factory=dict)
    nonces: dict = field(default_factory=dict)

    # --- the owner's side ---------------------------------------------------------------
    def open_window(self, window: int, *, nonce: str | None = None,
                    catalog: Catalog = CATALOG) -> dict:
        """Build, sign and publish the window file, exactly as the owner does at window open."""
        record = build_window(self.validator._entries[window],
                              tasks=self.validator.pins.world.tasks, catalog=catalog,
                              nonce=nonce or self.nonces[window],
                              revealed=sorted(self.validator.history.revealed))
        envelope = {"record": record,
                    "sig": self.owner_signer.sign(signing.canonical(record))}
        self.store.put(window_path(window), signing.canonical(envelope))
        return record

    # --- a miner's side -----------------------------------------------------------------
    def enrol(self, name: str, *, block: int, conductor: Conductor, pickled: bool = False,
              budget_usd: float = 100.0, register_at: int = 500) -> str:
        """Register, upload a tree, commit the ready signal, fund the allowance. Returns the hotkey.

        Every step is the production path: `store.build_manifest` + `upload_tree` put the bytes in
        the bucket with `manifest.json` last, and `chain.commit_ready` writes the `r2ready:v1` slot
        the validator reads the queue from.
        """
        hotkey = address(name)
        self.chain.block = register_at
        uid = self.chain.register(hotkey)
        registration = Registration(netuid=NETUID, uid=uid, hotkey=hotkey,
                                    registration_block=register_at)
        tree = simulate._write_tree(self.root / "miners" / name, pickled=pickled)
        manifest = build_manifest(tree, registration, lambda body: _key(name).sign(body).hex())
        upload_tree(tree, self.store, registration.prefix, manifest)
        self.chain.block = block
        self.chain.commit_ready(hotkey, ReadySignal(registration_id=registration.registration_id,
                                                    manifest_sha256=manifest.sha256))
        self.registry[registration.registration_id] = conductor
        self.gateway.fund(hotkey, budget_usd)
        return hotkey

    def crown(self, name: str, conductor: Conductor, *, budget_usd: float = 100.0) -> str:
        """Put a miner on the throne without duelling for it — the ordinary mid-life window.

        The reigning king is JUDGED: it won its crown in an earlier window, and §7 gives a hotkey one
        submission ever. Leaving it out of `judged` would put the king back in its own challenger
        queue, which is the state the daemon's queue filter exists to prevent.
        """
        hotkey = self.enrol(name, block=1, conductor=conductor, budget_usd=budget_usd)
        registration_id = next(rid for rid, c in self.registry.items() if c is conductor)
        self.validator.history.crown_to(Crown(hotkey=hotkey,
                                              model_dir=str(self.root / "king" / registration_id),
                                              served_model=f"v3-{hotkey}"))
        self.validator.history.coronations.append(hotkey)
        self.validator.history.judged.add(hotkey)
        self.validator.history.save()
        return hotkey

    def run(self, window: int, *, at_block: int | None = None):
        self.chain.block = at_block if at_block is not None else CADENCE.opens_at(window)
        self.open_window(window)
        return self.validator.run_window(window)


def harness(root: Path, benchmarks=LIVE, *, per_benchmark: int = 3, minimum: int = 2,
            grade=None, failures: frozenset[str] = frozenset(), owner_budget: float = 1_000.0,
            clock=None) -> Harness:
    worker = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
    tasks = tuple(task for benchmark in benchmarks for task in benchmark.load())
    # λ_b is pinned per corpus (§5.1b), so the sweep runs against the HONEST world; a flaky grader or
    # a dead model injected before it would measure the outage rather than the pool's slope.
    pins = simulate.pin_corpus(CATALOG, tasks, suite_grade(benchmarks), worker)
    if grade is not None:
        pins = replace(pins, world=replace(pins.world, grade=grade(pins.world.grade)))

    client = FakeS3()
    store = S3Bucket(client, BUCKET)
    chain = MockChain(immunity=CADENCE.immunity_blocks)        # §8b.1: preflight reads it from here
    chain.register("burn")                                    # uid 0, the burn address (§5.7)
    pool = Pool(answers=worker_answers(STRENGTH), costs=COST, failures=failures)
    gateway = OwnerGateway(pool)
    gateway.fund("owner", owner_budget)
    owner_signer = signing.Signer()
    registry: dict = {}
    nonces = {window: f"beacon-{window}" for window in CADENCE.windows}

    def serve(model_dir: Path, served_model: str) -> Conductor:
        """§8b.2's load check: raise if this tree cannot be served."""
        conductor = registry.get(Path(model_dir).name)
        if conductor is None:
            raise RuntimeError(f"no server is holding {served_model}")
        return conductor

    validator = Validator(
        pins=pins, reference=describe(simulate._write_tree(root / "reference")), chain=chain,
        store=store, mailbox=Mailbox(root / "mailbox.json", signing.Signer(), _never_mint),
        gateway=gateway,
        owner=Owner(ss58=owner_signer.public_hex, sign=owner_signer.sign,
                    verify_sig=lambda data, sig, who: signing.verify(who, data, sig)),
        cadence=CADENCE, serve=serve, beacon=nonces.__getitem__, netuid=NETUID, root=root / "state",
        per_benchmark=per_benchmark, minimum=minimum,
        **({} if clock is None else {"clock": clock}))
    # The owner commits the schedule root before any window opens (§6.3) — `preflight` refuses
    # otherwise, and a harness that skipped it would test a validator no owner could run.
    chain.commit_schedule(validator.manifest["root"])
    return Harness(validator=validator, chain=chain, client=client, store=store, pool=pool,
                   gateway=gateway, owner_signer=owner_signer, root=root, tree=root / "reference",
                   registry=registry, nonces=nonces)


def _never_mint(prefix: str):
    raise AssertionError("the daemon must never issue an upload credential")


def router(h: Harness, learned=ROUTABLE, rung: str = "mock/mid") -> Conductor:
    return learned_router(h.validator.pins.king_zero, learned, rung)


def outcome(reveal, hotkey: str):
    return next(o for o in reveal.report.outcomes if o.hotkey == hotkey)


def meter(reveal, arm: str):
    return next(m for m in reveal.arms if m.arm == arm)


# --- the whole window over the real seams -------------------------------------------------------


def test_a_full_window_runs_end_to_end_over_the_real_seams_and_crowns_the_better_conductor(
        tmp_path):
    """M9 exit 1, with I/O: the chain supplies the queue, R2 supplies the bytes, the gateway meters
    every call, and the crown is decided by the same three conditions the simulation decides it by.

    The copier is here for D14: a DIFFERENT artifact that behaves identically to King₀ ties, and a
    tie keeps the crown where it is — anti-copy priced rather than detected (measurement 11 found
    copier agreement and honest convergence to be the same signal).
    """
    h = harness(tmp_path)
    weak = h.enrol("router-a", block=10, conductor=router(h, rung="mock/heavy"))
    strong = h.enrol("router-b", block=11, conductor=router(h))
    copy = h.enrol("copycat", block=12, conductor=Copier(h.validator.pins.king_zero))

    reveal = h.run(1)

    assert reveal.report.power.separates
    assert outcome(reveal, weak).won and outcome(reveal, strong).won
    assert not outcome(reveal, copy).won
    assert "does not clear eps" in outcome(reveal, copy).detail
    assert reveal.report.crowned == strong
    # The traces D15 publishes as the entry ramp: the king's arm and both reference arms.
    assert len(reveal.report.reference) == 2
    assert any(result.steps for result in reveal.report.king_results)
    # Every arm reconciled, and the pool really was reached through the meter.
    assert all(arm.scoreable for arm in reveal.arms)
    assert h.pool.calls > 0


def test_every_arm_reaches_the_read_channel_through_the_guard_and_never_the_raw_adapter(
        tmp_path, monkeypatch):
    """§8b.3 AT THE OTHER CONTAINER, and it is a WIRING fact, so it is asserted on the wiring.

    `benchmarks.base.inspector` catches nothing on purpose: a container that could not run must reach
    the caller. `_arm` is the caller, and it caught nothing either — so an unguarded read seam let a
    `SandboxError` out of `run_episode`, out of `_arm`, and into `Validator.run`'s "one bad window
    must not end the reign", abandoning the whole window and every dollar already spent in it, with
    the challenger's shot unspent so it could repeat forever. `guard.inspecting` is what turns that
    into §6.3c's symmetric one-task exclusion, exactly as `guard.__call__` does for grading.

    Rewiring it back to `self.pins.world.inspect` is a mutant that survived every other test here,
    because the fixture's benchmarks declare no tools and so never enter the loop at all."""
    seen = []
    built = daemon.Scaffold

    def capture(*args, **kwargs):
        seen.append(kwargs.get("inspect"))
        return built(*args, **kwargs)

    monkeypatch.setattr(daemon, "Scaffold", capture)
    h = harness(tmp_path)
    h.enrol("router-a", block=10, conductor=router(h))

    h.run(1)

    assert len(seen) >= 3, "the reference arms and the challenger arm all build a scaffold"
    assert all(getattr(f, "__func__", None) is _GraderGuard.inspecting for f in seen), (
        "an arm holding the adapter's own inspector is an arm one dead container can abandon")


def test_the_reveal_is_published_signed_and_carries_the_owner_paid_arms_full_traces(tmp_path):
    """D15: without published traces only miners who can afford a five-figure data-collection bill
    can enter, so the traces are a by-product of running the subnet rather than a favour.

    They are the KING's and the REFERENCE arms' — the ones the owner paid for — never a losing
    challenger's, whose arm is theirs. Signed by the owner because the reveal is what a third party
    re-derives a crown from, and bytes found in a bucket authenticate nothing on their own.
    """
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))

    reveal = h.run(1)

    envelope = json.loads(h.store.get(reveal_path(1)))
    record = envelope["record"]
    assert signing.verify(h.owner_signer.public_hex, signing.canonical(record), envelope["sig"])
    assert set(record["traces"]) == {"king", *(arm.name for arm in reveal.report.reference)}
    assert hopeful not in record["traces"]
    step = next(s for trace in record["traces"]["king"] for s in trace["steps"])
    # A trace row is `(state, action, outcome, cost)` — what a miner needs to train on (§6.4).
    assert {"rendered_state_digest", "raw_output", "action", "cost_usd"} <= set(step)
    assert record["metering"] and record["weights"] and record["outcomes"]
    assert "METERING" in format_reveal(reveal)


# --- phase 1: fail closed --------------------------------------------------------------------------


def test_an_unverifiable_window_file_spends_no_shot_and_leaves_the_slate_untouched(tmp_path):
    """M7 exit 3. Scoring against a window nobody committed to would spend a miner's single shot on
    a measurement that was never pinned — the "Kind" property broken at its most expensive point.

    The refusal is fail-closed and total: no duels, no arms, no consumed eligibility, and the
    schedule on chain is exactly what it was. The daemon publishes an ABSENT window file itself
    (it is the owner, D1), so the case that remains is a file it cannot verify — and that one it
    refuses without touching it.
    """
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.client.objects[window_path(1)] = (b"not a window anybody signed", {})

    with pytest.raises(WindowUnavailable):
        h.validator.run_window(1)

    assert h.validator.history.judged == set()
    assert h.gateway.balance(hopeful) == pytest.approx(100.0)
    assert h.pool.calls == 0
    assert h.chain.weights == {}
    # The loop turns that into a refresh of the EXISTING slate, never a decay (§8b.6).
    assert h.validator.step() is None
    assert h.chain.weights == {0: pytest.approx(1.0)}        # King₀ reigns, so 0.85 burns


def test_a_window_whose_nonce_is_not_the_chains_is_refused_because_the_nonce_picks_the_slice(
        tmp_path):
    """§6.3's residual, closed here and nowhere else.

    `window.verify` re-derives the slice from the nonce IN THE RECORD, which proves the published
    file is self-consistent and says nothing about whether the owner chose it. Whoever picks the
    nonce picks the slice, so it must come from the chain — and an owner free to pick it could draw
    as many candidate slices as it liked after challengers had committed, with every signature and
    every hash still verifying.
    """
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1, nonce="a-nonce-the-owner-preferred")

    with pytest.raises(WindowUnavailable, match="the chain says"):
        h.validator.run_window(1)
    assert h.validator.history.judged == set()


def test_a_window_inviting_a_read_its_tasks_do_not_declare_is_refused_before_it_spends(tmp_path):
    """The read channel's third fact, and the only one the window's signature cannot cover.

    Three things must agree per benchmark: `tools()` declares a runnable verb, `environment()` is
    not None, and the prompt ends in `tools.PROTOCOL`. The first two are refused at assembly
    (`base.check_channel`); the third is inside the prompt, so a corpus that gets it wrong builds,
    signs, verifies and runs a whole arm. This is the expensive direction: `Scaffold._read` refuses
    a verb the task does not declare, so a worker that took the invitation has its `READ <path>` line
    graded as its patch — a scored answer nobody wrote, on the miner's own dollar.

    Refused after the record is authenticated and before the gateway is opened, which is what the
    three witnesses below are: no meter, no shot, no weight.
    """
    h = harness(tmp_path, benchmarks=tuple(Inviting(b) for b in LIVE))
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)

    with pytest.raises(ValueError, match="read protocol"):
        h.validator.run_window(1)

    assert h.pool.calls == 0
    assert h.validator.history.judged == set()
    assert h.gateway.balance(hopeful) == pytest.approx(100.0)
    assert h.chain.weights == {}


# --- phase 2/3/4: the order that decides who pays ---------------------------------------------------


def test_the_kings_arm_is_computed_once_per_window_and_shared_by_every_duel(tmp_path):
    """§5.2a / D13, measured in turns: N challengers must cost the king ONE arm.

    Re-running per duel would compare challenger A against king-run-1 and B against king-run-2, so
    run-to-run variance could crown the worse of the two — and §5.2's batch coronation ranks
    challengers against each other, which is meaningful only against one baseline.
    """
    def window_with(n: int, root: Path):
        h = harness(root)
        spy = Spy(router(h))
        h.crown("incumbent", spy)
        for i in range(n):
            h.enrol(f"ch-{i}", block=10 + i, conductor=router(h))
        return h.run(1), spy

    one, one_spy = window_with(1, tmp_path / "one")
    two, two_spy = window_with(2, tmp_path / "two")

    assert one_spy.calls > 0                       # the counter would have seen a second arm
    assert two_spy.calls == one_spy.calls          # ...and there was not one
    assert one_spy.calls == sum(len(r.steps) for r in one.report.king_results)
    assert len([o for o in two.report.outcomes if o.verdict is not None]) == 2
    assert two.report.king == one.report.king
    assert {o.verdict.delta for o in two.report.outcomes} == \
        {next(o for o in one.report.outcomes if o.verdict).verdict.delta}


def test_admission_runs_before_the_kings_arm_so_a_broken_challenger_costs_the_king_nothing(
        tmp_path):
    """§8b.2: an invalid submission must not be able to spend the king's allowance.

    Here the refusal comes out of the real `admission.admit` over bytes really pulled from the
    bucket, and the shot is spent — it was judged, and the answer was no.
    """
    h = harness(tmp_path)
    spy = Spy(router(h))
    king = h.crown("incumbent", spy)
    dud = h.enrol("dud", block=5, conductor=router(h), pickled=True)

    refused = h.run(1)

    assert spy.calls == 0
    assert refused.report.king is None and refused.report.king_results == ()
    assert outcome(refused, dud).status == REFUSED and outcome(refused, dud).shot_spent
    assert "pickle archive refused" in outcome(refused, dud).detail
    assert dud in h.validator.history.judged
    assert h.gateway.balance(king) == pytest.approx(100.0)

    # The control: a valid challenger DOES buy the king an arm, so the assertion above is about
    # admission and not about a spy that never fires.
    h.enrol("ok", block=6, conductor=router(h))
    assert h.run(2, at_block=CADENCE.opens_at(2)).report.king is not None
    assert spy.calls > 0


def test_a_window_that_cannot_separate_two_fixed_policies_runs_no_duels_and_spends_no_shot(
        tmp_path):
    """The measurement-15 rule: a window nobody can be ranked on costs a delay, not a shot.

    Our inability to measure is never charged to a miner, so the challenger rolls over with its one
    submission intact — and because the gate runs two FIXED policies, a near-optimal king can never
    stall the arena by being good.
    """
    h = harness(tmp_path, DEAD)
    spy = Spy(h.validator.pins.king_zero)
    incumbent = h.crown("incumbent", spy)
    hopeful = h.enrol("hopeful", block=5, conductor=router(h, ("flat-a",)))

    reveal = h.run(1)

    assert not reveal.report.power.separates
    assert outcome(reveal, hopeful).status == DEFERRED
    assert not outcome(reveal, hopeful).shot_spent
    assert h.validator.history.judged == {incumbent}     # nothing NEW was spent
    assert reveal.report.king is None and spy.calls == 0
    # Refused on the FLOOR, not on the lower bound: real, separable from zero, and still too small
    # to host a coronation (§5.2b).
    assert reveal.report.power.lcb > 0.0
    assert reveal.report.power.spread < reveal.report.power.floor
    # A refused window is not owner downtime (§8b.6): the schedule stands and the king keeps earning.
    assert reveal.weights_written


def test_every_verdict_waits_for_the_last_arm_so_one_denominator_serves_the_whole_window(tmp_path):
    """§6.3c by the back door, which is the reason phases 5 and 6 are separate.

    The sandbox here dies during the LAST challenger's arm. If duel 1 were judged before duel 2 ran,
    the first challenger would be scored on a denominator that still contained the task the second
    one lost — an asymmetric exclusion arriving as an ordering bug rather than as a rule anybody
    broke. Both challengers must lose the same task from the same denominator.
    """
    sandboxes: list[Sandbox] = []

    def flaky(grade):
        sandboxes.append(Sandbox(grade))
        return sandboxes[-1]

    h = harness(tmp_path, grade=flaky)
    h.enrol("early", block=10, conductor=router(h))
    # The second challenger arms the sandbox on its very first turn, so the king's arm and the first
    # challenger's arm are already complete and graded when the grader starts failing.
    late_router = router(h)
    h.enrol("late", block=11,
            conductor=Spy(late_router, on_act=lambda: setattr(sandboxes[0], "armed", True)))

    reveal = h.run(1)

    assert reveal.report.graders_failed == (sandboxes[0].dead,)
    verdicts = [o.verdict for o in reveal.report.outcomes if o.verdict is not None]
    assert len(verdicts) == 2
    admitted = {b for b, rate in h.validator.pins.world.exchange.items() if not rate.flags}
    full = len(admitted) * h.validator.per_benchmark
    assert {v.n_tasks for v in verdicts} == {full - 1}
    assert sum(row.n_tasks for row in reveal.report.king.per_benchmark) == full - 1


# --- §8b.2: a broken KING arm settles the whole window ----------------------------------------------


def test_a_king_arm_that_hits_the_per_duel_clock_settles_no_verdict_and_spends_no_shot(tmp_path):
    """§8b.2's last paragraph — the rule that only exists because of §5.2a.

    One king arm is shared by every duel in the window, so a king arm truncated by the OWNER's clock
    would hand every challenger in the window the same zeroed tail and decide all of them at once.
    That is our failure, not theirs, so the window is treated exactly as the power gate treats a dead
    one. Note what is NOT rolled back: the refusals from phase 3 stand, because those were judged on
    their own artifact and the king's clock has nothing to do with them.
    """
    clock = Clock()
    h = harness(tmp_path, clock=clock)
    # The king's own first turn moves the clock past the arm's deadline, so the reference arms and
    # the start of the king's arm run normally and the rest of its slice is never reached.
    h.crown("incumbent", Spy(router(h), on_act=clock.overrun))
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))

    reveal = h.run(1)

    assert any(r.stopped_reason == DUEL_WALL_CLOCK_REASON for r in reveal.report.king_results)
    assert outcome(reveal, hopeful).status == DEFERRED
    assert "per-duel wall clock" in outcome(reveal, hopeful).detail
    assert not outcome(reveal, hopeful).shot_spent
    assert hopeful not in h.validator.history.judged
    assert reveal.report.crowned is None


# --- §8b.1: the queue is a chain read ---------------------------------------------------------------


def test_a_judged_hotkey_leaves_the_queue_before_the_depth_cap_rather_than_holding_a_slot(tmp_path):
    """A commitment survives forever on chain, and that is the trap the daemon has that the sim does
    not: every miner ever judged is still committed and still visible, so counting them against
    `MAX_QUEUE_DEPTH` would turn away LIVE challengers on behalf of miners settled months ago —
    §8b.1's deregistration trap sprung by the mechanism built to close it.
    """
    h = harness(tmp_path)
    old = [h.enrol(f"old-{i:02d}", block=i, conductor=router(h)) for i in range(MAX_QUEUE_DEPTH)]
    h.validator.history.judged.update(old)
    h.validator.history.save()
    fresh = h.enrol("fresh", block=900, conductor=router(h))

    reveal = h.run(1)

    assert outcome(reveal, fresh).status == DUELLED
    # And they are not re-judged, not re-charged, and not reported: their record is permanent
    # elsewhere, and a reveal that listed every settled miner would grow without bound.
    assert not any(o.hotkey in set(old) for o in reveal.report.outcomes)
    assert not any(m.arm in set(old) for m in reveal.arms)


def test_a_commit_beyond_the_queue_depth_is_refused_at_the_door_and_keeps_its_shot(tmp_path):
    """§8b.1 / M9 exit 10: taking a registration burn for a slot that will expire before it is judged
    is the outcome the cap exists to prevent, and taking the money first makes it worse rather than
    kinder. Refused at the door costs nothing — no download, no arm, and eligibility is NOT consumed,
    so the miner can still rotate a credential and finish an upload.
    """
    h = harness(tmp_path)
    hotkeys = [h.enrol(f"ch-{i:02d}", block=10 + i, conductor=router(h), pickled=True)
               for i in range(MAX_QUEUE_DEPTH + 1)]
    last = hotkeys[-1]

    reveal = h.run(1)

    assert len([o for o in reveal.report.outcomes if o.status == REFUSED]) == MAX_QUEUE_DEPTH
    turned_away = outcome(reveal, last)
    assert turned_away.status == UNQUEUED and not turned_away.shot_spent
    assert f"MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH}" in turned_away.detail
    assert last not in h.validator.history.judged
    assert not h.validator.mailbox.spent(last)          # eligibility untouched by a refusal


def test_a_full_window_defers_the_overflow_with_the_wait_the_immunity_period_must_cover(tmp_path):
    """§8b.1's other cap, and the number the owner has to publish beside it.

    Challenger arms are serial and each is bounded by the two-hour per-duel clock, so a window holds
    a fixed number. The remainder roll over with their shot unspent and are told how long they will
    wait — which is the figure `check_launch` refuses a launch over.
    """
    h = harness(tmp_path)
    hotkeys = [h.enrol(f"ch-{i:02d}", block=10 + i, conductor=router(h))
               for i in range(MAX_DUELS_PER_WINDOW + 1)]

    reveal = h.run(1)

    judged = [o for o in reveal.report.outcomes if o.status == DUELLED]
    rolled = [o for o in reveal.report.outcomes if o.status == DEFERRED]
    assert len(judged) == MAX_DUELS_PER_WINDOW
    assert [o.hotkey for o in rolled] == hotkeys[MAX_DUELS_PER_WINDOW:]
    assert "expected wait 1 window(s)" in rolled[0].detail
    assert not any(o.shot_spent for o in rolled)
    assert h.validator.history.judged == {o.hotkey for o in judged}


def test_a_second_ready_signal_from_a_spent_hotkey_never_reaches_a_duel(tmp_path):
    """§7, enforced by the owner's own ledger rather than by the miner's client.

    Re-signalling with a NEW manifest after the first was accepted is the swap the one-shot exists to
    catch: the shot is spent on the tree the hotkey committed to, and a second submission is refused
    forever. It is refused server-side, before an arm exists, so it costs nothing to say no.
    """
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    first = h.run(1)
    assert outcome(first, hopeful).status == DUELLED

    # A different tree, committed under the same hotkey, after the shot was spent.
    h.validator.history.judged.discard(hopeful)          # pretend only the mailbox remembers
    h.chain.block = 20
    h.chain.commit_ready(hopeful, ReadySignal(registration_id="ab" * 32, manifest_sha256="cd" * 32))
    h.gateway.fund(hopeful, 100.0)

    second = h.run(2, at_block=CADENCE.opens_at(2))
    assert outcome(second, hopeful).status == SKIPPED
    assert "different registration" in outcome(second, hopeful).detail
    assert not any(m.arm == hopeful for m in second.arms)


def test_a_swapped_shard_is_caught_at_duel_time_and_spends_the_shot(tmp_path):
    """M6 exit 4 reaching the daemon: the commitment names a manifest, the manifest names every
    file, and `fetch_submission` re-hashes each one AS IT LANDS — so a tree replaced after the signal
    is refused at that shard rather than averaged into a wall of bytes. The operator is asked to
    revoke the write credential at `consume`; this is the backstop that does not depend on it.
    """
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    key = next(k for k in h.client.objects if k.endswith("model.safetensors"))
    h.client.objects[key] = (b"different bytes entirely", h.client.objects[key][1])

    reveal = h.run(1)

    assert outcome(reveal, hopeful).status == REFUSED
    assert "hashes to" in outcome(reveal, hopeful).detail
    assert hopeful in h.validator.history.judged
    assert reveal.report.king is None                # and the king was never charged for it


# --- M5: the meter ------------------------------------------------------------------------------------


def test_every_dollar_a_scored_arm_reports_came_out_of_a_gateway_receipt(tmp_path):
    """M5 exit 1 as arithmetic. `final_b` prices spend (§5.1b), so an arm whose dollars cannot be
    traced to gateway receipts would price a crown on a number nobody metered."""
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))

    reveal = h.run(1)

    arm = meter(reveal, hopeful)
    assert arm.scoreable and arm.audit.calls > 0
    assert arm.audit.metered_usd > 0.0
    assert arm.audit.metered_usd == pytest.approx(arm.audit.recorded_usd)
    # The miner's allowance really moved, against a pre-funded balance on the OWNER's key — so the
    # miner never held a key to configure, never saw a task prompt, and never billed the substitution
    # channel §11-1 is about.
    assert h.gateway.balance(hopeful) == pytest.approx(100.0 - arm.audit.metered_usd)
    assert not any("mock" in str(k) for k in h.client.objects if "prompt" in str(k))


def test_an_arm_whose_spend_cannot_be_reconciled_is_not_scored_and_its_shot_is_not_spent(tmp_path):
    """A challenger that ran but cannot be metered is OUR failure, so it rolls over intact.

    Simulated by dropping a receipt from the ledger, which is what an off-gateway call would look
    like from the audit's side: spend with nothing behind it. Refusing is the only safe verdict —
    the alternative is a crown priced on a number nobody metered.
    """
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    real_close = h.gateway.close

    def close(token):
        ledger = real_close(token)
        if ledger.hotkey == hopeful and ledger.receipts:
            ledger.receipts.pop()
        return ledger

    h.gateway.close = close
    reveal = h.run(1)

    assert not meter(reveal, hopeful).scoreable
    assert meter(reveal, hopeful).audit.reason == "spend_not_metered"
    assert outcome(reveal, hopeful).status == DEFERRED
    assert "could not be reconciled" in outcome(reveal, hopeful).detail
    assert hopeful not in h.validator.history.judged
    assert reveal.report.crowned is None


def test_a_verdict_decided_by_funding_is_visible_in_the_reveal_rather_than_read_as_skill(tmp_path):
    """§5.8, and it needs BOTH of the metering's counters to be legible separately.

    A starved challenger against a funded king is legitimate under D5 — allowances are the miner's
    and unequal — but the leaderboard would otherwise report skill for a result that measured
    capital. So the tail an arm never reached is published per arm, and it is kept apart from
    `unfunded_calls`, which counts calls the GATEWAY refused: those produce dead rungs mid-episode
    that read exactly like a flaky provider, and the counter is the only place that difference
    exists.
    """
    h = harness(tmp_path)
    h.crown("incumbent", router(h))                                 # funded at 100.0
    hopeful = h.enrol("hopeful", block=10, conductor=router(h), budget_usd=0.1)

    reveal = h.run(1)

    starved, king = meter(reveal, hopeful), meter(reveal, "king")
    assert starved.exhausted > 0 and king.exhausted == 0
    assert outcome(reveal, hopeful).exhausted == starved.exhausted
    assert "tasks the allowance never reached" in format_reveal(reveal)
    assert "unfunded calls" in format_reveal(reveal)
    # Both counters are separate fields of the published record, not one aggregate.
    row = next(r for r in json.loads(h.store.get(reveal_path(1)))["record"]["metering"]
               if r["arm"] == hopeful)
    assert {"exhausted", "unfunded_calls"} <= set(row)


# --- §8b.5: restart -----------------------------------------------------------------------------------


def test_the_daemons_arm_matches_the_scaffolds_own_run_window_on_a_cold_start(tmp_path):
    """THE ANTI-DRIFT TEST, and the reason `_arm` is allowed to exist at all.

    `_arm` replaces `Scaffold.run_window` only because §8b.5 needs per-task control, so on a cold
    start it must be indistinguishable from it: same nonce-derived order, same budget threading,
    same clock check ahead of the allowance check, same rows. A divergence here would score the
    daemon on a different arm than the one the dev kit shows a miner (M6b's whole promise).
    """
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    policy = h.validator.pins.king_zero

    scaffold = Scaffold(opened.catalog, policy, MockWorker(answers=worker_answers(STRENGTH),
                                                           costs=COST),
                        h.validator.pins.world.grade)
    expected = tuple(episode.result for episode in
                     scaffold.run_window(opened.tasks, nonce=opened.nonce, budget_usd=100.0))

    h.gateway.fund("solo", 100.0)
    got, _ = h.validator._arm(opened, arm="solo", conductor=policy, payer="solo", budget_usd=100.0,
                              guard=simulate._GraderGuard(h.validator.pins.world.grade),
                              tasks=opened.tasks)
    assert got == expected


def test_a_restart_resumes_from_the_last_checkpointed_task_instead_of_re_spending_it(tmp_path):
    """§8b.5 / M9 exit 9. A restart must not re-spend money already spent, and under the one-shot
    rule a partially-recorded duel must never be re-judged from scratch.

    The resumed arm is byte-identical to the uninterrupted one, the provider is called only for the
    tasks that were still outstanding, and the allowance is threaded forward from the checkpointed
    spend rather than restarted — that money is gone.
    """
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    policy = h.validator.pins.king_zero
    guard = simulate._GraderGuard(h.validator.pins.world.grade)

    h.gateway.fund("solo", 100.0)
    whole, _ = h.validator._arm(opened, arm="solo", conductor=policy, payer="solo",
                                budget_usd=100.0, guard=guard, tasks=opened.tasks)
    calls_for_the_whole_arm = h.pool.calls

    # Now the same arm, but the process died after the first three tasks: keep their checkpoint
    # rows, throw the rest away, and start again behind a FRESH gateway — the meter is in-process,
    # so a restart is exactly what losing its ledger looks like.
    path = h.validator.checkpoint.path(1, "solo")
    kept = path.read_text().splitlines()[:3]
    path.write_text("\n".join(kept) + "\n")
    h.pool.calls = 0
    h.validator.gateway = OwnerGateway(h.pool)
    h.validator.gateway.fund("solo", 100.0)
    resumed, audit = h.validator._arm(opened, arm="solo", conductor=policy, payer="solo",
                                      budget_usd=100.0, guard=guard, tasks=opened.tasks)

    assert resumed == whole
    assert audit.resumed == 3
    assert 0 < h.pool.calls < calls_for_the_whole_arm
    # Reconciled over what THIS process metered: a resumed prefix's receipts belong to the process
    # that spent them, and including it would refuse an arm for having survived a restart.
    assert audit.scoreable


def test_a_resumed_arm_threads_the_allowance_forward_because_that_money_is_already_gone(tmp_path):
    """The half of §8b.5 a byte-comparison on a rich arm cannot see.

    A restart must NOT hand the arm its allowance back. §4 zeroes every remaining task when the
    allowance runs out and both arms are cut on the same tail, so an arm that restarted its budget
    would run further than its funding allowed and be cut at a different task than the king's — the
    duel silently stops being paired, and a challenger buys the difference with a crash.
    """
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    policy = h.validator.pins.king_zero
    guard = simulate._GraderGuard(h.validator.pins.world.grade)
    starved = 0.1                                     # ~2.5 tasks' worth at this world's prices

    h.gateway.fund("solo", starved)
    whole, _ = h.validator._arm(opened, arm="solo", conductor=policy, payer="solo",
                                budget_usd=starved, guard=guard, tasks=opened.tasks)
    exhausted = [r.task_id for r in whole if r.stopped_reason == "budget_exhausted"]
    assert exhausted, "this budget has to bind, or the test proves nothing"

    path = h.validator.checkpoint.path(1, "solo")
    path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
    h.validator.gateway = OwnerGateway(h.pool)
    h.validator.gateway.fund("solo", starved)
    resumed, _ = h.validator._arm(opened, arm="solo", conductor=policy, payer="solo",
                                  budget_usd=starved, guard=guard, tasks=opened.tasks)

    assert [r.task_id for r in resumed if r.stopped_reason == "budget_exhausted"] == exhausted


def test_a_window_that_dies_halfway_is_retried_without_re_spending_what_it_already_spent(tmp_path):
    """The failure `run_forever` is built to survive, and it is not the same as a process restart.

    A window that raises after some of its arms have run is NOT settled, so the next poll runs it
    again — and everything the first attempt paid for is in the checkpoint. The only thing the retry
    cannot reuse is the gateway's ledger identity: `open_duel` refuses a duel id twice, so without an
    attempt number the retry would fail on the FIRST arm forever, with the first attempt's money
    already spent and no path that could ever read it back.
    """
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    real_duel = daemon.duel

    def explode(*args, **kwargs):
        raise RuntimeError("the verdict blew up")

    daemon.duel = explode
    try:
        with pytest.raises(RuntimeError):
            h.validator.run_window(1)
    finally:
        daemon.duel = real_duel
    spent_by_the_first_attempt = h.pool.calls
    assert not h.validator.history.completed(1)

    h.pool.calls = 0
    reveal = h.validator.run_window(1)

    assert reveal.report.power.separates and h.validator.history.completed(1)
    assert h.pool.calls == 0              # every task came back from the checkpoint
    assert spent_by_the_first_attempt > 0
    assert all(arm.resumed > 0 for arm in reveal.arms)
    assert all(arm.duel_id.endswith("-2") for arm in reveal.arms)


def test_the_windows_allowance_is_snapshotted_once_and_a_restart_cannot_re_read_it(tmp_path):
    """§4: the allowance is read at ONE definite instant and frozen for that window's duels.

    Re-reading the balance after a crash is wrong in both directions, and which direction depends on
    whether the meter's balance is durable — a property no caller should have to know. Against a
    durable gateway the balance is already debited, so subtracting the checkpointed spend again would
    starve the arm; against a volatile one it is not, so not subtracting would hand the arm its money
    back and let a challenger buy the difference with a crash.
    """
    checkpoint = Checkpoint(tmp_path / "cp")
    assert checkpoint.snapshot(3, "king", 12.5) == 12.5
    assert checkpoint.snapshot(3, "king", 0.0) == 12.5          # the restart's re-read is ignored
    assert checkpoint.snapshot(3, "5Cother", 4.0) == 4.0        # ...and it is per arm, not per window


def test_a_torn_checkpoint_line_costs_the_task_that_was_in_flight_and_nothing_else(tmp_path):
    """The bound the append-only checkpoint buys: a process killed mid-write leaves a partial final
    line, which is dropped — so at most one task is re-run. That is the same bound the allowance
    already overruns by (`Scaffold.run_window`) and the one `OwnerGateway.call` documents."""
    checkpoint = Checkpoint(tmp_path / "cp")
    rows = [simulate.EpisodeResult(task_id=f"t-{i}", benchmark="b", steps=(), graded_score=1.0,
                                  spend_usd=0.5, stopped_reason="stop") for i in range(3)]
    for row in rows:
        checkpoint.append(7, "king", row)
    path = checkpoint.path(7, "king")
    path.write_text(path.read_text() + '{"task_id": "t-3", "bench')

    assert checkpoint.resume(7, "king") == tuple(rows)


# --- §5.5 / §5.7 / §8b.6: the slate ------------------------------------------------------------------


def test_the_crown_reverts_to_king_zero_when_the_king_leaves_the_metagraph(tmp_path):
    """§5.5: a king that has left the metagraph does not keep the throne — or defend it.

    The payout bug this closes is invisible in the burn total, which is why it survived: §5.7
    excludes the REIGNING hotkey from its own lineage, so a king still recorded as reigning after it
    has left is excluded from a lineage it belongs at the top of, and every surviving pensioner is
    paid one rank too high while the sixth drops off the end for nothing.
    """
    h = harness(tmp_path)
    deposed = Spy(router(h))
    gone = h.crown("gone", deposed)
    h.chain.deregister(gone)
    hopeful = h.enrol("hopeful", block=5, conductor=router(h))

    reveal = h.run(1)

    assert reveal.report.king_hotkey == KING0
    assert deposed.calls == 0                                   # it did not defend
    assert h.validator.history.crown.hotkey == hopeful          # the throne was vacant; §8b.8
    assert reveal.report.crowned == hopeful
    # The deposed king takes the rank-1 slot it was dethroned into, and its share burns because the
    # hotkey resolves to no uid — the ranks behind it do NOT move up (§5.7).
    assert reveal.report.pensioners == (gone,)
    by_hotkey = {reveal.report.metagraph.get(uid, "burn"): share
                 for uid, share in reveal.report.weights.items()}
    assert by_hotkey[hopeful] == pytest.approx(0.85)
    assert gone not in by_hotkey
    assert sum(reveal.report.weights.values()) == 1.0


def test_a_stable_reign_is_re_submitted_between_windows_so_it_does_not_age_out(tmp_path):
    """§8b.6's ceiling, which the spec states only as a floor.

    `koth/neuron.py` records netuid 99 scoring 20+ consecutive epochs correctly while `last_update`
    aged 2234 blocks against an `activity_cutoff` of 5000: Yuma stops counting a validator for being
    too stable. A 24-hour window is 7200 blocks, so ONE write per window ages out by design — and a
    stable reign is precisely an unchanged slate, which both the chain and the mock short-circuit
    unless the write is forced.
    """
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    h.run(1)
    first_write = h.chain.weights_set_at

    h.chain.block = first_write + daemon.WEIGHT_REFRESH_BLOCKS
    assert h.validator.step() is None                       # mid-window: nothing to score
    assert h.chain.weights_set_at == h.chain.block          # ...and the slate was re-submitted

    # Too soon is a no-op rather than a rejected extrinsic: the refresh is a cadence, not a retry.
    h.chain.block += 1
    h.validator.step()
    assert h.chain.weights_set_at == h.chain.block - 1


def test_a_launch_the_chain_would_silently_defeat_is_refused_before_anything_is_scored(tmp_path):
    """`check_launch`, whose two gates both fail SILENTLY if they are not checked.

    A window no longer than `weights_rate_limit` has its update rejected and the PREVIOUS schedule
    quietly keeps paying; an immunity period shorter than the queue's drain time deregisters
    challengers before they are ever evaluated, having paid a registration burn for nothing. Neither
    shows up in any number the reveal publishes.
    """
    check_launch(window_blocks=200, immunity_blocks=600, rate_limit=100)     # the shipped cadence

    with pytest.raises(Exception, match="weights_rate_limit"):
        check_launch(window_blocks=100, immunity_blocks=600, rate_limit=100)
    with pytest.raises(ValidatorError, match="immunity_period"):
        check_launch(window_blocks=200, immunity_blocks=599, rate_limit=100)
    # The drain-time arithmetic is derived from the caps, so it stays correct when the window moves.
    windows = -(-MAX_QUEUE_DEPTH // MAX_DUELS_PER_WINDOW) + 1
    check_launch(window_blocks=200, immunity_blocks=windows * 200, rate_limit=100)

    h = harness(tmp_path)
    h.chain.rate_limit = 500                        # longer than the window: §8b.6 refuses it
    h.validator._preflighted = False
    with pytest.raises(Exception, match="weights_rate_limit"):
        h.validator.preflight()


# --- §8 step 5: persist, weights, publish ------------------------------------------------------------


def test_history_is_persisted_before_the_weights_and_the_reveal_is_published_last(tmp_path):
    """§8 step 5's order, and the reason it is an order: re-judging is NOT idempotent under the
    one-shot rule, so a crash must be able to repeat the cheap half and never the half that spent a
    miner's submission. The witness is the chain itself — by the time the weight write happens, the
    verdicts are already on disk."""
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    seen: list[set[str]] = []
    real_set_weights = h.chain.set_weights

    def set_weights(weights, *, force=False):
        seen.append(set(json.loads(h.validator.history.path.read_text())["judged"]))
        return real_set_weights(weights, force=force)

    h.chain.set_weights = set_weights
    reveal = h.run(1)

    assert seen == [{hopeful}]                                   # persisted BEFORE the write
    assert reveal.weights_written
    assert h.validator.history.published(1)
    assert reveal_path(1) in h.client.objects


def test_a_crash_between_the_weights_and_the_publish_re_does_only_the_publish(tmp_path):
    """M9 exit 2. The window is settled — its verdicts stand and no shot is spent twice — so the
    restart repeats the cheap half. Without the local copy written during the persist step, a store
    outage at exactly this moment would lose that window's traces (D15) for good."""
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    h.client.readonly = True                       # the store dies at the moment of publishing

    reveal = h.validator.run_window(1)
    assert reveal.weights_written
    assert not h.validator.history.published(1)
    assert reveal_path(1) not in h.client.objects

    h.client.readonly = False
    h.chain.block = CADENCE.opens_at(1) + 10
    assert h.validator.step() is None               # the window is settled; only the publish repeats
    assert h.validator.history.published(1)
    assert reveal_path(1) in h.client.objects
    assert h.validator.history.judged == {address("hopeful")}     # and nothing was re-judged


def test_a_settled_window_is_never_re_judged_even_if_the_daemon_restarts_into_it(tmp_path):
    """§7 through a crash: the history file is the only thing that survives, and it is enough."""
    h = harness(tmp_path)
    hopeful = h.enrol("hopeful", block=10, conductor=router(h))
    h.run(1)
    calls = h.pool.calls

    reborn = harness(tmp_path)                     # a fresh process over the same state directory
    reborn.chain.block = CADENCE.opens_at(1) + 5
    assert reborn.validator.history.judged == {hopeful}
    assert reborn.validator.history.completed(1)
    assert reborn.validator.step() is None
    assert h.pool.calls == calls


# --- the loop ------------------------------------------------------------------------------------------


def test_the_loop_runs_each_committed_window_once_and_stops_scoring_past_the_schedule(tmp_path):
    """The daemon's outer shape. A window is hours long and the poll is a minute, so the ordinary
    return is None — and once the committed schedule is exhausted there is no entry to score
    against, so the loop keeps the slate warm and scores nobody."""
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    for window in CADENCE.windows:
        h.chain.block = CADENCE.opens_at(window)
        h.open_window(window)

    blocks = [CADENCE.opens_at(1), CADENCE.opens_at(1) + 1, CADENCE.opens_at(2),
              CADENCE.opens_at(3) + CADENCE.window_blocks]        # past the schedule
    revealed = []
    for block in blocks:
        h.chain.block = block
        reveal = h.validator.step()
        if reveal is not None:
            revealed.append(reveal.window)

    assert revealed == [1, 2]                       # window 1 once, then 2; nothing past the end
    assert h.validator.cadence.window_at(blocks[-1]) is None


def test_the_loop_survives_a_window_that_raises_and_leaves_the_reign_earning(tmp_path):
    """One bad window must not end the reign (D1's owner-liveness dependency, taken seriously).

    The right failure is the one §8b.6 names: weights are left as they are and the king keeps
    earning, rather than the daemon dying and the schedule decaying toward nothing.
    """
    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    h.validator.run_window = lambda window: (_ for _ in ()).throw(RuntimeError("disk on fire"))

    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 2

    h.validator.run_forever(poll_seconds=0.0, should_stop=stop)
    assert not h.validator.history.completed(1)
    assert h.validator.history.judged == set()


# --- the seams that only exist in production ------------------------------------------------------------


def test_the_window_nonce_is_the_hash_of_the_block_the_window_opens_at(tmp_path):
    """`chain_beacon`, against a stub substrate. Whoever picks the nonce picks the slice (§6.3), so
    the value has to be fixed before the window exists and chosen by nobody — and re-derivable by any
    reader from `(genesis_block, window_blocks, window)` plus one RPC."""
    class Substrate:
        def __init__(self, hashes):
            self.hashes, self.asked = hashes, []

        def get_block_hash(self, block):
            self.asked.append(block)
            return self.hashes.get(block)

    substrate = Substrate({1_000: "0xaaa", 1_200: "0xbbb"})
    beacon = chain_beacon(substrate, CADENCE)

    assert beacon(1) == "0xaaa" and beacon(2) == "0xbbb"
    assert substrate.asked == [1_000, 1_200]
    with pytest.raises(ValidatorError, match="refusing rather than choosing"):
        beacon(3)


def test_the_served_name_is_the_only_channel_the_committed_artifact_has_onto_the_wire():
    """`serve.reference_served_name`'s rule, applied to a challenger: an OpenAI-compatible server
    reports the name the operator gave it and nothing about the weights it loaded, so the artifact's
    digest reaches the endpoint only inside `--served-model-name`. It catches operator error — a
    stale server on the port, the previous challenger still resident — and is not, and must not be
    read as, an adversarial check: real weight identity is `store.fetch_submission` over the bytes
    and `admission.admit` over the tree."""
    commitment = Commitment(hotkey="5Cxyz", block=1,
                            ready=ReadySignal(registration_id="ab" * 32,
                                              manifest_sha256="cd" * 32))
    assert served_name(commitment) == "v3-5Cxyz@" + "cd" * 32


def test_the_whole_daemon_runs_with_no_network_and_no_credentials(tmp_path, monkeypatch):
    """Every seam is offline: the chain is in memory, the bucket is a dict, the meter's provider is
    scripted, and the Conductor is a fixed policy. If any of them reached out, this fails."""
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "R2_ACCESS_KEY_ID",
                 "AWS_ACCESS_KEY_ID", "BT_WALLET_NAME"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("the daemon opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    h = harness(tmp_path)
    h.enrol("hopeful", block=10, conductor=router(h))
    assert h.run(1).report.power.separates


def test_history_survives_a_restart_and_carries_the_reign_with_it(tmp_path):
    """The durable state is small, append-only and complete: without it every owner deploy would
    hand every hotkey a fresh shot and forget who reigns."""
    path = tmp_path / "history.json"
    history = History(path)
    history.crown_to(Crown(hotkey="5Cking", model_dir="/srv/king", served_model="v3-5Cking@abc"))
    history.settle(4, crowned="5Cking", judged=["5Ca", "5Cb"], revealed=["t-1"])
    history.mark_published(4)

    restored = History(path)
    assert restored.crown == Crown("5Cking", "/srv/king", "v3-5Cking@abc")
    assert restored.judged == {"5Ca", "5Cb"} and restored.revealed == {"t-1"}
    assert restored.completed(4) and restored.published(4)
    assert restored.lineage().coronations == ("5Cking",)


def test_a_starved_allowance_still_binds_when_episodes_run_concurrently(tmp_path):
    """THE REGRESSION CONCURRENCY INTRODUCED, and the reason the batch is sized by the allowance.

    Workers that check `remaining` and then all dispatch read the same figure — none has reported
    yet — so a batch wider than what the money buys runs every task anyway. §4 says exhaustion
    zeroes the remainder and §5.8 says the cut must be visible; an arm that quietly spent several
    times a miner's allowance on the miner's own key breaks both.
    """
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    guard = simulate._GraderGuard(h.validator.pins.world.grade)
    starved = 0.1                                     # ~2.5 tasks' worth at this world's prices

    h.gateway.fund("solo", starved)
    results, meter = h.validator._arm(opened, arm="solo", conductor=h.validator.pins.king_zero,
                                      payer="solo", budget_usd=starved, guard=guard,
                                      tasks=opened.tasks)

    exhausted = [r for r in results if r.stopped_reason == "budget_exhausted"]
    assert exhausted, "a starved allowance must still cut the arm short"
    assert meter.exhausted == len(exhausted)
    # and the overrun is bounded by a batch rather than by the whole slice
    assert sum(r.spend_usd for r in results) < starved * 3


def test_a_validator_whose_schedule_is_not_on_chain_refuses_to_start(tmp_path):
    """§6.3: the root is what removes the owner's choice of slice, and it only removes it if a third
    party can read it. Computed and kept locally it proves nothing, so an uncommitted schedule is a
    launch-time refusal rather than a window that scores and cannot be checked."""
    h = harness(tmp_path)
    h.chain._schedule = None
    with pytest.raises(ValidatorError, match="no schedule root is committed"):
        h.validator.preflight()


def test_a_committed_root_that_is_not_this_schedule_refuses_to_start(tmp_path):
    """A root for a different world or window count means the slices scored are not the ones the
    chain was told about — which is exactly the discretion the commitment exists to remove."""
    h = harness(tmp_path)
    h.chain.commit_schedule("ff" * 32)
    with pytest.raises(ValidatorError, match="does not match this validator's schedule"):
        h.validator.preflight()


def test_a_daemon_whose_owner_account_is_empty_refuses_to_start(tmp_path):
    """§4's allowance, checked before a window rather than discovered inside one.

    The failure this replaces is silent and expensive. `fund()` had no caller in shipped code, so
    the daemon started against a $0 gateway; both reference arms tripped the exhaustion check at
    their first step, scored zero, and the power gate reported that the slice CANNOT separate two
    policies known to differ. The window was then published blaming the corpus for the owner's empty
    wallet — every window, burning emissions, with `--check` green throughout.
    """
    h = harness(tmp_path, owner_budget=0.0)

    with pytest.raises(ValidatorError) as caught:
        h.validator.preflight()

    message = str(caught.value)
    assert "no allowance" in message
    # The refusal has to say what to DO: this is the one failure whose symptom points at the corpus.
    assert "orchestra-owner" in message and "credit" in message


def test_a_funded_owner_account_passes_the_same_gate(tmp_path):
    """The gate must not be the reason a correctly configured daemon cannot start."""
    harness(tmp_path, owner_budget=1.0).validator.preflight()


def test_an_unfunded_challenger_is_not_crowned_for_doing_nothing(tmp_path):
    """The exploit this closes was measured, not imagined.

    §4 pays an arm from the miner's own allowance, so an unfunded challenger reaches no worker,
    scores 0 quality and spends $0. That is not a weak router but an absent one — and `final_b`
    prices spend, so `final = quality − λ_b·spend_b/C_b` puts the do-nothing arm at exactly 0.0
    against a king whose quality does not cover its own bill. Before this guard the null arm was
    crowned, took 0.85 of emissions, and the audit saw nothing wrong: zero metered equals zero
    recorded, so the arm reconciled perfectly.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("penniless", block=10, conductor=router(h), budget_usd=0.0)

    report = h.run(1).report

    assert report.crowned is None
    row = next(o for o in report.outcomes if o.hotkey == hotkey)
    assert row.status == DEFERRED
    assert "allowance" in row.detail
    # The "Kind" property: a registration burn and a ~72 GB upload must not be spent on an omission
    # the miner can fix by sending money.
    assert hotkey not in h.validator.history.judged


def test_a_funded_arm_that_never_delegates_is_refused_rather_than_crowned(tmp_path):
    """The same exploit with the money paid — and the cheaper way to run it.

    A miner funds an allowance, uploads a model that answers STOP on every task, and spends nothing.
    The admission-time allowance check does not see this one: the balance is real. But the arm still
    reaches no worker, still scores 0 quality at $0 spend, and 0 still beats a king whose priced
    spend exceeds its quality. An arm that never exercised the action space has no routing in it to
    compare, so it is refused at the point of scoring — which also covers an allowance that dies on
    its first call and a pool that refuses every request.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("quitter", block=10, conductor=MockConductor(("STOP",)), budget_usd=100.0)

    report = h.run(1).report

    assert report.crowned is None
    row = next(o for o in report.outcomes if o.hotkey == hotkey)
    assert row.status == DEFERRED
    assert "no worker model" in row.detail
    assert hotkey not in h.validator.history.judged


def test_a_funded_challenger_still_wins_normally(tmp_path):
    """The guard must refuse absence, not competence."""
    h = harness(tmp_path, owner_budget=1_000.0)
    h.enrol("solvent", block=10, conductor=router(h), budget_usd=100.0)

    assert h.run(1).report.crowned is not None


def test_a_hopeless_challenger_stops_paying_before_the_slice_ends(tmp_path):
    """M9 exit 4. A challenger that cannot win should not go on buying episodes to prove it.

    The bar is known before any challenger runs, because §5.2a already measured the king over this
    whole slice; `best_possible_final` bounds what the challenger can still reach. When even the
    ceiling cannot clear `king + eps`, every further episode is the miner's money spent on a settled
    question.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    # A king that is actually ahead. Against King zero in this fixture the bar is NEGATIVE — its
    # priced spend exceeds its quality — so an arm scoring nothing is not hopeless there, it is
    # winning, which is the case the null-arm guard covers rather than this one.
    h.crown("incumbent", router(h))
    # Answers nothing on any task: scores zero, and its ceiling collapses as the slice runs.
    hotkey = h.enrol("hopeless", block=10,
                     conductor=MockConductor(("DELEGATE mock/dud",)), budget_usd=100.0)

    reveal = h.run(1)

    assert meter(reveal, hotkey).futile > 0, "a provably beaten arm ran the whole slice anyway"
    rows = h.validator.checkpoint.resume(1, hotkey)
    futile = [r for r in rows if r.stopped_reason == FUTILE_REASON]
    # §5.8: the reason a task was abandoned is published, and is its own finding — not the clock's,
    # not the allowance's.
    assert all(r.graded_score == 0.0 for r in futile)
    # The saving is the point: the abandoned tail was never bought.
    assert len(futile) < len(rows)
    assert reveal.report.crowned is None


def test_a_winning_challenger_is_never_abandoned(tmp_path):
    """The property the bound exists to guarantee, at the level that matters.

    §7 gives a hotkey ONE submission ever, so an abort that fires on a challenger who would have
    won spends it forever. That is why the ceiling is an exact bound rather than the reference
    implementation's observed-quantile heuristic, which says of itself that it is not one.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("winner", block=10, conductor=router(h), budget_usd=100.0)

    reveal = h.run(1)

    assert meter(reveal, hotkey).futile == 0
    assert reveal.report.crowned == hotkey
    # HOW TIGHTLY THIS PINS THE BAR, measured rather than assumed: widening the abort's margin from
    # `king + eps` to `king + 20·eps` is caught (by the funding-visibility test, not this one), and
    # anything narrower than that slips through. This challenger clears the king by far more than
    # eps, so it cannot see a small error. Pinning the margin finely needs a near-miss fixture — a
    # king whose final sits just under the challenger's reachable ceiling — which a nine-task corpus
    # cannot express. Recorded so the next person does not mistake this for a tight bound.


def test_an_exclusion_made_before_a_crash_is_still_an_exclusion_after_one(tmp_path):
    """§6.3c across §8b.5's restart, which is where the two rules meet and used to disagree.

    `_GraderGuard` hands a failed grade back as 0.0 and its docstring says that zero "is never
    scored" — true only while the guard remembers why. The set lived in the object, the daemon
    rebuilds the object every window, so a mid-window restart forgot every exclusion made before it.
    The checkpointed rows still carried the placeholder zero, nothing remembered it was a placeholder,
    and the owner's dead sandbox was scored as the miner's miss in whichever arm had already run.
    """
    h = harness(tmp_path)
    checkpoint = h.validator.checkpoint

    checkpoint.exclude(7, "alpha/task-3")
    checkpoint.exclude(7, "beta/task-9")

    assert checkpoint.exclusions(7) == {"alpha/task-3", "beta/task-9"}
    # Per WINDOW: a later window starts clean, because a slice is drawn fresh (§6.3b).
    assert checkpoint.exclusions(8) == set()


def test_a_torn_exclusion_record_is_the_process_that_died(tmp_path):
    h = harness(tmp_path)
    checkpoint = h.validator.checkpoint
    checkpoint.exclude(7, "alpha/task-3")
    path = checkpoint.root / "window-7" / "excluded.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "beta/tas')

    assert checkpoint.exclusions(7) == {"alpha/task-3"}


def test_the_window_writes_every_exclusion_through_to_disk(tmp_path):
    """The seam has to be WIRED, not merely available: the defect was a guard nobody persisted.

    Same dying sandbox as the window-wide-denominator test above, asked the one further question —
    is the exclusion it produced still there for the process that comes after this one.
    """
    sandboxes: list[Sandbox] = []

    def flaky(grade):
        sandboxes.append(Sandbox(grade))
        return sandboxes[-1]

    h = harness(tmp_path, grade=flaky)
    late = router(h)
    h.enrol("late", block=10,
            conductor=Spy(late, on_act=lambda: setattr(sandboxes[0], "armed", True)))

    reveal = h.run(1)

    assert reveal.report.graders_failed == (sandboxes[0].dead,)
    assert h.validator.checkpoint.exclusions(1) == {sandboxes[0].dead}


def test_a_restarted_window_still_drops_the_task_a_dead_grader_took(tmp_path):
    """The defect this closes, from the side that matters: not "is it written" but "is it READ".

    A window excludes a task, the daemon dies, and the daemon that replaces it must run the rest of
    the window on the same denominator. Before this, the replacement built a fresh guard with an
    empty `failed` set: the king's checkpointed rows still carried the placeholder zero
    `_GraderGuard` hands back, nothing remembered that it was a placeholder, and §6.3c's symmetric
    exclusion silently became the owner's dead sandbox scored as the miner's miss.
    """
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    dead = h.validator._open(1).tasks[0].task_id
    h.validator.checkpoint.exclude(1, dead)
    h.enrol("router-a", block=10, conductor=router(h))   # the king's arm runs only against someone

    # The daemon that comes back: same state directory, no memory of its own.
    reveal = h.run(1)

    assert dead in reveal.report.graders_failed
    admitted = {b for b, rate in h.validator.pins.world.exchange.items() if not rate.flags}
    full = len(admitted) * h.validator.per_benchmark
    assert sum(row.n_tasks for row in reveal.report.king.per_benchmark) == full - 1


def test_a_declared_immunity_that_the_subnet_does_not_have_stops_the_daemon(tmp_path):
    """§8b.1's obligation, checked against the chain rather than against the owner's own flag.

    `--immunity-blocks` states what the owner BELIEVES they configured. The queue cap is derived
    from it, so a gate that reads the belief certifies a claim: an owner who types three windows and
    sets the subnet to one passes launch, and challengers who paid a registration burn and uploaded
    ~70 GB are deregistered before they are ever judged — the outcome this very check exists to
    prevent, arriving through the check itself.
    """
    h = harness(tmp_path)
    h.chain.immunity = CADENCE.immunity_blocks // 2          # the subnet, not the flag

    with pytest.raises(ValidatorError) as caught:
        h.validator.preflight()

    assert "immunity_period" in str(caught.value)


def test_an_immunity_that_matches_but_is_too_short_is_still_refused(tmp_path):
    """Agreement is not sufficiency: the arithmetic §8b.1 derives still has to hold."""
    short = Cadence(genesis_block=1_000, window_blocks=200, windows=(1, 2, 3), immunity_blocks=200)
    h = harness(tmp_path)
    h.chain.immunity = short.immunity_blocks
    h.validator.cadence = short

    with pytest.raises(ValidatorError) as caught:
        h.validator.preflight()

    assert "does not cover" in str(caught.value)


def test_a_model_that_will_not_load_is_refused_before_the_king_spends_anything(tmp_path):
    """M9 exit 6's other half: the LOAD check, not the bytes check.

    §8b.2 says a load timeout or OOM makes a submission invalid, its shot spent, and the king not
    charged — and the king is not charged only because the check runs in PHASE 3, before PHASE 4
    touches the king's arm at all. The bytes half of that was pinned; this half was not, because the
    harness only ever fails `serve` for a tree it was never given, and `enrol` always registers one.
    So a regression that served the challenger lazily inside PHASE 5 — after the king had already
    paid for a whole slice — would have kept every test green.
    """
    h = harness(tmp_path)
    hotkey = h.enrol("unservable", block=10, conductor=router(h))
    # The bytes are perfect and the tree admits; the server refuses to hold it. That is §8b.2's
    # case, and it is the one that costs money if it is discovered late.
    h.registry.clear()

    reveal = h.run(1)

    row = outcome(reveal, hotkey)
    assert row.status == REFUSED
    assert hotkey in h.validator.history.judged, "an invalid artifact was judged: the shot is spent"
    # The king never ran, so the king never paid. (The reference arms DID run: they are the power
    # gate, they precede admission, and the owner pays for them — that is PHASE 2, not PHASE 4.)
    assert reveal.report.king is None
    armed = {meter.arm for meter in reveal.arms}
    assert "king" not in armed and hotkey not in armed


def test_a_servable_model_is_the_control_for_that_refusal(tmp_path):
    """The same window with nothing cleared, so the assertions above are about the load and not
    about a window that never runs anything."""
    h = harness(tmp_path)
    h.enrol("servable", block=10, conductor=router(h))

    reveal = h.run(1)

    assert reveal.report.king is not None
    assert "king" in {meter.arm for meter in reveal.arms}


class ParallelClock:
    """Fake wall time, advanced by the calls that consume it — `d` seconds shared by a batch.

    Two models were tried and are wrong, and both are worth naming because either would have passed
    while proving nothing. Advancing `d` per CALL — the shape `test_scaffold` uses, correctly,
    for a serial question — costs 250 episodes 250·d whatever the concurrency. Advancing on a clock
    READ that follows work degenerates to the same thing, because `Scaffold` checks the per-episode
    deadline on this very clock, so every episode reads it once anyway.

    So the batch is made to overlap for real: `Concurrent.chat` holds each caller briefly so its
    peers arrive, and each advances the clock by `d` divided by how many are actually in flight. A
    batch of `k` costs `d` once rather than `k` times, which is what running them together means.
    """

    def __init__(self, per_episode: float) -> None:
        self.per_episode, self.now = per_episode, 0.0
        self.in_flight = 0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        return self.now


@dataclass
class Concurrent:
    """A pool whose calls take real (tiny) time, so a batch genuinely overlaps.

    It implements `chat`, which is the seam `OwnerGateway.call` actually uses. A fixture overriding
    `complete` instead — the shape one test in `test_gateway.py` shipped with — is never called
    at all, and a throughput test built on one would measure a clock that never moves.
    """

    clock: ParallelClock
    answers: Mapping[str, str] = field(default_factory=dict)
    overlap: float = 0.01
    calls: int = 0

    def chat(self, model_id: str, task_text: str, params: Mapping[str, object]) -> Completion:
        with self.clock.lock:
            self.clock.in_flight += 1
            self.calls += 1
        time.sleep(self.overlap)               # long enough for this batch's peers to arrive
        with self.clock.lock:
            self.clock.now += self.clock.per_episode / max(1, self.clock.in_flight)
            self.clock.in_flight -= 1
        return Completion(text=self.answers.get(model_id, ""), served_model=model_id,
                          provider="mock", finish_reason="stop", tokens_in=1, tokens_out=1,
                          cost_usd=0.01)


def _throughput_arm(tmp_path, concurrency: int, *, tasks: int = 250, per_episode: float = 100.0):
    """One arm over a production-sized slice, at a chosen concurrency, on a clock that models it."""
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    benchmark = opened.tasks[0].benchmark
    slice_ = tuple(TaskSpec(task_id=f"{benchmark}/t{index}", benchmark=benchmark,
                            prompt=f"task {index}", tools=())
                   for index in range(tasks))

    clock = ParallelClock(per_episode)
    h.validator.clock = clock
    h.validator.gateway = OwnerGateway(Concurrent(clock, answers=worker_answers(STRENGTH)))
    h.validator.gateway.fund("solo", 10_000.0)
    with mock.patch.object(validator_module, "EPISODE_CONCURRENCY", concurrency):
        results, _ = h.validator._arm(opened, arm="solo", conductor=h.validator.pins.king_zero,
                                      payer="solo", budget_usd=10_000.0,
                                      guard=simulate._GraderGuard(h.validator.pins.world.grade),
                                      tasks=slice_)
    return results, clock


def test_a_parallel_arm_reaches_a_production_sized_slice_inside_the_duel_clock(tmp_path):
    """M9 exit 3, which nothing established: the suite passed with parallelism removed entirely,
    because every test pins invariance under concurrency and none pins throughput.

    §5.4 sizes an arm at ~250 tasks and says wall clock is not the constraint — but only because the
    arm runs in parallel. At 100 s an episode, serially that is 25,000 s against a 9,000 s per-duel
    clock, so the claim is false at concurrency 1 and true at 16. The companion test below runs the
    same arm serially and watches it fail, which is what stops this one passing by arrangement.

    WHAT THIS DOES NOT SAY: that the SHIPPED deployment fits. `config.py`'s note above
    `DUEL_WALL_CLOCK_SECONDS` puts an arm at ~10,383 s under arrangement (B) — a shared sshfs grade
    dir costing ~1,650 s of filesystem round trips per arm — against the same 9,000 s clock, and
    says so in terms. This test pins the mechanism: that CONCURRENCY is what makes 250 tasks
    reachable at a given per-episode latency. Whether today's per-episode latency is low enough is a
    deployment question, it is answered "not under (B)", and the fix named there is to move the
    grade dir off shared storage rather than to raise the clock.
    """
    results, clock = _throughput_arm(tmp_path, EPISODE_CONCURRENCY)

    assert len(results) == 250
    assert not [r for r in results if r.stopped_reason == DUEL_WALL_CLOCK_REASON]
    assert clock.now < DUEL_WALL_CLOCK_SECONDS


def test_the_same_arm_run_serially_does_not_reach_it(tmp_path):
    """The control. Without it the test above says only that 250 x something fits in 9,000."""
    results, clock = _throughput_arm(tmp_path, 1)

    assert [r for r in results if r.stopped_reason == DUEL_WALL_CLOCK_REASON], (
        "a serial arm finished a production-sized slice: the fake clock is not modelling wall time")
    assert clock.now >= DUEL_WALL_CLOCK_SECONDS


def test_a_nanodollar_allowance_cannot_buy_the_crown(tmp_path):
    """The exploit the zero-spend guard missed, and the reason it is the CROWN that is bounded.

    `OwnerGateway.call` tests `balance <= 0` before the call and debits after, so ANY positive
    allowance buys one metered call. A hotkey funded with a nanodollar therefore clears the
    admission check and the spend check, answers almost none of the slice, and scores ~0 quality at
    ~0 spend — which `final = quality − λ·spend/C` puts above any king whose priced spend exceeds
    its quality. Measured before this guard: crowned, with 0.85 of emissions.

    It is still scored and still published, because §4 zeroes a starved tail and §5.8 requires the
    cut to be legible rather than hidden. What it may not do is win.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("nanodollar", block=10, conductor=router(h), budget_usd=1e-9)

    reveal = h.run(1)

    assert reveal.report.crowned is None
    row = outcome(reveal, hotkey)
    assert row.status == DUELLED, "§5.8: a funding-decided result is published, not hidden"
    assert "crown is withheld" in row.detail
    assert row.verdict.challenger_wins, "it did win on the numbers — that is the whole problem"


def test_a_resumed_arm_is_still_judged_after_a_restart(tmp_path):
    """§8b.5, and the regression the first version of the null-arm guard introduced.

    `ArmAudit` is built over FRESH results only, deliberately, so a resumed prefix does not fail
    reconciliation. A guard that read `metered_usd` therefore saw $0 for a fully-resumed arm and
    deferred it — breaking the restart path on exactly the arms it exists to protect. The spend that
    matters is what the whole slice recorded, not what this process happened to meter.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("resumed", block=10, conductor=router(h), budget_usd=100.0)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    # Run the arm to completion first, so the scored window resumes every task from the checkpoint
    # and meters nothing.
    h.validator._arm(opened, arm=hotkey, conductor=router(h), payer=hotkey, budget_usd=100.0,
                     guard=simulate._GraderGuard(h.validator.pins.world.grade), tasks=opened.tasks)

    reveal = h.run(1)

    assert outcome(reveal, hotkey).status == DUELLED
    assert reveal.report.crowned == hotkey


def test_a_drained_owner_account_refuses_the_next_window_not_just_the_first(tmp_path):
    """The failure the launch check exists to prevent, arriving the way it actually arrives.

    `preflight` returns early once it has run, so a balance check living only there covers the
    never-funded-at-startup case and nothing else. The interesting case is the other one, and it is
    guaranteed rather than hypothetical: a prepaid allowance always reaches zero. Measured before
    this fix — an owner funded for one window drains to -$0.02, and every window after publishes a
    CANNOT-separate reveal, defers every challenger and burns every emission, with the launch check
    still green and the headline blaming the corpus.
    """
    h = harness(tmp_path, owner_budget=0.30)          # enough for window 1, not for window 2
    h.validator.preflight()
    h.enrol("hopeful", block=10, conductor=router(h))
    h.run(1)
    assert h.gateway.balance("owner") <= 0.0, "the fixture must actually drain the owner"

    with pytest.raises(ValidatorError) as caught:
        h.run(2)

    assert "no allowance" in str(caught.value)
    assert "orchestra-owner" in str(caught.value)


def test_a_king_that_stops_paying_loses_the_throne(tmp_path):
    """§4 in full: "The king must stay funded ... an unfunded king scores zeroes and loses."

    Only the first half was enforced. An arm that never reaches the slice scores 0 on every task,
    which puts its `final` at exactly 0.0 — and 0.0 beats any challenger whose priced spend exceeds
    its quality. So a reigning miner could stop funding and keep 85% of emissions for as long as
    nobody cleared that bar: free money, against a rule the spec states outright.

    The reign ends rather than the window being voided, because voiding lets the freeloader keep the
    throne — the opposite of what §4 asks for.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    h.crown("freeloader", router(h), budget_usd=0.0)
    assert not h.validator.history.crown.is_king_zero
    h.enrol("challenger", block=10, conductor=router(h), budget_usd=1e-9)

    h.run(1)

    assert h.validator.history.crown.is_king_zero, "an unfunded king kept the throne"


def test_a_funded_king_keeps_defending_normally(tmp_path):
    """The rule must end a freeloader's reign, not every reign."""
    h = harness(tmp_path, owner_budget=1_000.0)
    king = h.crown("incumbent", router(h), budget_usd=100.0)
    h.enrol("weak", block=10, conductor=MockConductor(("STOP",)), budget_usd=100.0)

    h.run(1)

    assert h.validator.history.crown.hotkey == king


def test_unfunded_entries_cannot_squat_the_queue(tmp_path):
    """The denial of service the kind deferral opened.

    §4 defers a challenger with no allowance without spending its shot — right, because the omission
    is fixable by sending money. But a deferred hotkey is never judged, so it never leaves the
    pending list, so `MAX_QUEUE_DEPTH` unfunded commits made early and never funded would fill the
    queue forever and every funded challenger after them would be refused at the door.

    The cap exists to bound how long a queued miner waits on US (§8b.1's deregistration trap). A
    miner waiting on their own wallet is not in that queue and must not consume its capacity.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    for index in range(MAX_QUEUE_DEPTH):
        h.enrol(f"squatter-{index}", block=10 + index, conductor=router(h), budget_usd=0.0)
    funded = h.enrol("funded", block=100, conductor=router(h), budget_usd=100.0)

    reveal = h.run(1)

    row = outcome(reveal, funded)
    assert row.status == DUELLED, "a funded challenger was blocked by hotkeys that never paid"
    assert reveal.report.crowned == funded


def test_the_futility_abort_keeps_episodes_an_earlier_attempt_already_paid_for(tmp_path):
    """§8b.5 against §5.4, and the one writer in `_arm` that forgot the checkpoint.

    The abort fills the tail with `futile` zeros — correct for a task never reached, wrong for one an
    earlier attempt already ran. It was the only place in `_arm` that wrote `placed[index]` without
    checking `done` first, so a retried window overwrote completed episodes with zeros AND appended
    those zeros to the checkpoint, making the loss survive the restart that was supposed to recover
    it. The miner had already paid for that work.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    h.crown("incumbent", router(h))
    hotkey = h.enrol("hopeless", block=10,
                     conductor=MockConductor(("DELEGATE mock/dud",)), budget_usd=100.0)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)

    # Seed the checkpoint with one finished, paid-for episode for the LAST task in nonce order —
    # the one the abort would otherwise reach.
    last = list(task_order(opened.tasks, opened.nonce))[-1]
    paid = EpisodeResult(task_id=last.task_id, benchmark=last.benchmark, steps=(),
                         graded_score=1.0, spend_usd=0.5, stopped_reason="stop")
    h.validator.checkpoint.append(1, hotkey, paid)

    # The abort only fires against a real bar, so the king's arm has to have run.
    guard = simulate._GraderGuard(h.validator.pins.world.grade)
    king_results, _ = h.validator._arm(
        opened, arm="king", conductor=h.validator._king_conductor(h.validator.history.crown),
        payer="owner", budget_usd=1_000.0, guard=guard, tasks=opened.tasks)

    results, _ = h.validator._arm(
        opened, arm=hotkey, conductor=MockConductor(("DELEGATE mock/dud",)), payer=hotkey,
        budget_usd=100.0, guard=guard, tasks=opened.tasks, king_arm_results=king_results)

    assert any(r.stopped_reason == FUTILE_REASON for r in results), (
        "the abort never fired, so this test would pass whatever the abort loop does")
    kept = next(r for r in results if r.task_id == last.task_id)
    assert kept.graded_score == 1.0 and kept.stopped_reason == "stop", (
        "the abort overwrote an episode that was already run and already paid for")


def test_a_subnet_more_generous_than_declared_is_not_refused(tmp_path):
    """Only one direction is dangerous.

    Exact equality rejected a subnet configured MORE generously than `--immunity-blocks` claims — a
    strictly safer setting — with a message implying the owner had got it wrong. §8b.1 needs
    challengers to get at LEAST the drain time the queue cap is derived from; a longer immunity
    gives them more.
    """
    h = harness(tmp_path)
    h.chain.immunity = CADENCE.immunity_blocks * 10

    h.validator.preflight()          # must not raise


def test_a_subnet_less_generous_than_declared_names_the_flag_that_is_wrong(tmp_path):
    """The direction that deregisters challengers the daemon believes are safe.

    Asserted on text only the CROSS-CHECK produces. `check_launch` refuses the same configuration
    with its own message, which also contains "immunity_period" — so a test matching that word
    passes with the cross-check deleted, and the first version of this test did exactly that. What
    the cross-check adds over `check_launch` is not safety but diagnosis: it names the flag the
    owner set and the number the subnet actually has, instead of reporting arithmetic about windows.
    """
    h = harness(tmp_path)
    h.chain.immunity = CADENCE.immunity_blocks // 3

    with pytest.raises(ValidatorError) as caught:
        h.validator.preflight()

    assert "--immunity-blocks says" in str(caught.value)


def test_a_damaged_exclusion_record_is_refused_rather_than_half_read(tmp_path):
    """Stopping at a damaged row would silently drop every later exclusion — restoring, quietly, the
    exact bug the durable record exists to prevent: a dead grader's task scored as a miner's miss."""
    h = harness(tmp_path)
    checkpoint = h.validator.checkpoint
    checkpoint.exclude(1, "alpha/task-1")
    path = checkpoint.root / "window-1" / "excluded.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    checkpoint.exclude(1, "alpha/task-2")

    with pytest.raises(ValidatorError, match="unreadable"):
        checkpoint.exclusions(1)


def test_a_torn_exclusion_row_is_still_just_a_torn_row(tmp_path):
    h = harness(tmp_path)
    checkpoint = h.validator.checkpoint
    checkpoint.exclude(1, "alpha/task-1")
    path = checkpoint.root / "window-1" / "excluded.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "alpha/tas')

    assert checkpoint.exclusions(1) == {"alpha/task-1"}


def test_a_task_the_window_already_dropped_is_not_bought_again(tmp_path):
    """§6.3c removes a task whose grader died from every arm and from the denominator, and the
    exclusion set is window-wide and durable. So by the time a later arm reaches such a task, the
    window has already decided it will not count — and running it anyway spends the miner's
    allowance on a result nothing will score. Measured before this: $0.08 of a miner's money on a
    task the window had dropped before that arm even started.
    """
    h = harness(tmp_path, owner_budget=1_000.0)
    hotkey = h.enrol("miner", block=10, conductor=router(h), budget_usd=100.0)
    h.chain.block = CADENCE.opens_at(1)
    h.open_window(1)
    opened = h.validator._open(1)
    dead = opened.tasks[0].task_id
    h.validator.checkpoint.exclude(1, dead)

    guard = simulate._GraderGuard(h.validator.pins.world.grade,
                                  failed=h.validator.checkpoint.exclusions(1))
    results, _ = h.validator._arm(opened, arm=hotkey, conductor=router(h), payer=hotkey,
                                  budget_usd=100.0, guard=guard, tasks=opened.tasks)

    row = next(r for r in results if r.task_id == dead)
    assert row.spend_usd == 0.0, "the miner paid for a task the window had already dropped"
    assert row.stopped_reason == EXCLUDED_REASON, (
        "§5.8: an excluded task is its own finding, not the clock's and not the wallet's")


def test_the_endings_that_are_not_routing_stay_distinguishable(tmp_path):
    """§5.8, claimed in three docstrings and pinned by nothing until now.

    Four endings mean four different findings — the model stalled, the validator ran out of clock,
    the miner ran out of money, and the arm could no longer win — plus a fifth for a task the window
    itself dropped. Collapse any two and a verdict decided by one is published as the other, which is
    the confusion §5.8 exists to forbid. Mutating `FUTILE_REASON` to `"budget_exhausted"` left the
    whole suite green: `_exhausted` then counted futile rows as starvation and the reveal said a
    miner had run out of money when the mechanism had abandoned them.
    """
    endings = {"stop", "max_steps", "parse_failures", "wall_clock",
               DUEL_WALL_CLOCK_REASON, "budget_exhausted", FUTILE_REASON, EXCLUDED_REASON}

    assert len(endings) == 8, f"two endings collapsed into one: {sorted(endings)}"


def test_an_abandoned_arm_is_not_reported_as_a_starved_one(tmp_path):
    """The pair §5.8 most needs kept apart, because both fill a tail with zeros and only one is
    about the miner's wallet."""
    h = harness(tmp_path, owner_budget=1_000.0)
    h.crown("incumbent", router(h))
    hopeless = h.enrol("hopeless", block=10,
                       conductor=MockConductor(("DELEGATE mock/dud",)), budget_usd=100.0)

    reveal = h.run(1)

    arm = meter(reveal, hopeless)
    assert arm.futile > 0, "the arm was abandoned, so the reveal must say so"
    assert arm.exhausted == 0, (
        "an abandoned arm was published as a starved one — §5.8's exact confusion")


def test_the_daemon_publishes_the_window_file_it_then_verifies_and_never_replaces_one(tmp_path):
    """The owner's half of window open (§6.2) is the daemon's, because the daemon is the owner
    (D1) and holds the signer the reveal already uses. Measured 2026-09-08 on testnet 526: the
    first real window refused itself with NoSuchKey — the only thing that had ever written this
    file was the smoke script, inline. Publishing is for an ABSENT file only: what is there stays,
    byte for byte, whether it is the daemon's own earlier file or something unverifiable, which
    `_open` then refuses in `load`'s words rather than papering over."""
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    assert window_path(1) not in h.client.objects

    opened = h.validator._open(1)

    assert opened.nonce == h.nonces[1]
    published = h.client.objects[window_path(1)]
    h.validator._open(1)
    assert h.client.objects[window_path(1)] == published, "a second open must re-read, not re-write"

    h.client.objects[window_path(2)] = (b"not a window", {})
    h.chain.block = CADENCE.opens_at(2)
    with pytest.raises(WindowUnavailable, match="missing or unauthentic"):
        h.validator._open(2)
    assert h.client.objects[window_path(2)] == (b"not a window", {})


def test_local_serving_asks_every_endpoint_and_uses_the_one_that_lists_the_name(monkeypatch):
    """`serve.launch_command`'s arrangement is one server per card, and a committed name is served by
    exactly one of them. Measured 2026-09-08 on testnet 526: a third challenger with different
    weights needed its own server, and a single --serve-url could not name it."""
    import thirtyspokes.v3.validator as validator_module
    from thirtyspokes.v3.serve import ServeError

    class Fake:
        def __init__(self, base_url, served_model, timeout):
            self.base_url, self.served_model = base_url, served_model

        def check_ready(self):
            if not (self.base_url.startswith("http://b/") and self.served_model == "v3-hk@sha"):
                raise ServeError(f"{self.base_url} serves ('other',), not {self.served_model!r}")

    monkeypatch.setattr(validator_module, "ServedConductor", Fake)
    serve = validator_module.local_serving("http://a/v1, http://b/v1")
    assert serve(Path("x"), "v3-hk@sha").base_url == "http://b/v1"

    with pytest.raises(ServeError, match="http://a/v1.*http://b/v1") as refused:
        validator_module.local_serving("http://a/v1,http://b/v1")(Path("x"), "nobody")
    assert "no endpoint serves" in str(refused.value)
    with pytest.raises(ServeError, match="^http://a/v1 serves"):   # one URL: the refusal untouched
        validator_module.local_serving("http://a/v1")(Path("x"), "nobody")


# --- §5.2c: the window's shared outcome table, through the daemon -----------------------------------


def test_every_arm_faces_the_same_draws_and_the_reveal_splits_charged_from_provider_spend(tmp_path):
    """D17 through the real seams. The copier makes the king's choices, so under one table every one
    of its delegates is a hit: it is charged the full price (§4 unchanged), the provider is paid
    nothing for it, and it ties to the last bit. The reveal says all of that per arm, and reconciles:
    `charged_usd` is what `recorded_usd` was priced from, `provider_usd` never exceeds it."""
    from thirtyspokes.v3.archetypes import Copier
    h = harness(tmp_path)
    strong = h.enrol("router-b", block=11, conductor=router(h))
    copy = h.enrol("copycat", block=12, conductor=Copier(h.validator.pins.king_zero))

    reveal = h.run(1)

    record = json.loads(h.store.get(reveal_path(1)))["record"]
    rows = {m["arm"]: m for m in record["metering"]}
    for m in rows.values():
        assert m["charged_usd"] == pytest.approx(m["recorded_usd"])
        assert m["provider_usd"] <= m["charged_usd"] + 1e-9
    assert rows[copy]["hits"] > 0 and rows[copy]["fills"] == 0
    assert rows[copy]["provider_usd"] == 0.0 and rows[copy]["charged_usd"] > 0.0
    assert outcome(reveal, copy).verdict.delta == 0.0 and not outcome(reveal, copy).won
    assert outcome(reveal, strong).won and reveal.report.crowned == strong
    table = record["outcome_table"]
    assert table["fills"] > 0 and table["hits"] >= rows[copy]["hits"]
    assert table["rows"] == table["fills"] and table["failed_rows"] == 0
    # The table's rows are not published — D15 publishes the arms' traces, as before.
    assert "outcome_rows" not in record and set(record["traces"]) >= {"king"}
    # And the drift audit ran on the keys that were replayed, on the owner's allowance.
    assert record["retest"]["keys"] > 0 and record["retest"]["usd"] > 0.0
    assert record["retest"]["grade_disagreements"] == 0     # a deterministic mock pool cannot drift
    assert "charged" in format_reveal(reveal) and "test-retest" in format_reveal(reveal)


def test_the_table_survives_a_restart_and_a_resumed_window_re_buys_nothing(tmp_path):
    """§8b.5 for the draws: the table is on disk beside the checkpoints, so a process that starts
    over on the same window replays every key an earlier process bought and the provider is not
    asked again."""
    from thirtyspokes.v3.scaffold import OutcomeTable
    h = harness(tmp_path)
    h.chain.block = CADENCE.opens_at(1)
    opened = h.validator._open(1)
    guard = simulate._GraderGuard(h.validator.pins.world.grade)
    h.gateway.fund("first", 100.0)
    h.gateway.fund("second", 100.0)

    _, first = h.validator._arm(opened, arm="first", conductor=h.validator.pins.king_zero,
                                payer="first", budget_usd=100.0, guard=guard, tasks=opened.tasks,
                                table=OutcomeTable(h.validator.checkpoint.outcomes(1)))
    bought = h.pool.calls
    reopened = OutcomeTable(h.validator.checkpoint.outcomes(1))     # a new process, the same file
    _, second = h.validator._arm(opened, arm="second", conductor=h.validator.pins.king_zero,
                                 payer="second", budget_usd=100.0, guard=guard, tasks=opened.tasks,
                                 table=reopened)

    assert first.fills > 0 and first.hits == 0
    assert h.pool.calls == bought and second.fills == 0 and second.hits == first.fills
    assert second.audit.provider_usd == 0.0
    assert second.audit.charged_usd == pytest.approx(first.audit.charged_usd)
    assert second.scoreable and first.scoreable


def test_seen_and_unseen_scores_partition_every_arm_and_mark_what_an_earlier_reveal_showed(tmp_path):
    """The memorisation meter (§5.2c): at window 1 nothing has been shown, so every row is unseen;
    at window 2 the tasks window 1's traces published are seen (D12: they recur by design), and each
    arm's two halves add up to the arm — a diagnostic the verdict never reads."""
    h = harness(tmp_path)
    h.enrol("router-a", block=10, conductor=router(h))
    one = h.run(1)
    first = json.loads(h.store.get(reveal_path(1)))["record"]["seen_unseen"]
    assert first and all(split["seen"]["n_tasks"] == 0 for split in first.values())
    assert first["king"]["unseen"]["n_tasks"] == len(one.report.king_results) - len(one.report.graders_failed)

    h.enrol("router-b", block=CADENCE.opens_at(2) - 1, conductor=router(h))
    two = h.run(2)
    split = json.loads(h.store.get(reveal_path(2)))["record"]["seen_unseen"]["king"]
    scored = len(two.report.king_results) - len(two.report.graders_failed)
    assert split["seen"]["n_tasks"] + split["unseen"]["n_tasks"] == scored
    assert split["seen"]["n_tasks"] > 0, "a 12-task pool drawn 9 at a time repeats itself"
    assert split["seen"]["quality"] is not None and split["seen"]["final"] is not None
