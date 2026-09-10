"""The chain seam (docs/WHITEPAPER.md §5.5, §5.7, §8b.1, §8b.6; the build plan M8b/M9).

Most of this file is about ONE failure, because it is the only one here that is both silent and
permanent: **Bittensor recycles UIDs.** If ex-king `hk_A` held UID 5 and is deregistered, UID 5 goes
to an unrelated `hk_B`, and a schedule that remembered `{5: 0.04}` pays a stranger a pension they
never earned — every window, forever, with the slate still summing to 1.0 and nothing in any total
looking wrong. `test_a_stale_metagraph_pays_the_stranger` runs that mistake to completion so the
correct read has something to be correct *against*; the tests around it pin each thing that keeps it
from happening: the read is `uid -> hotkey`, it happens at weight-setting time, and a `Commitment`
has no uid field for a stale one to travel inside.

The mock is only worth testing against because it RECYCLES — a mock that hands every registrant a
fresh number would make every test here vacuously green — and for the same reason it enforces the
weight rate limit and skips an unchanged slate. Each of those three is pinned by its own test before
anything is built on top of it.

The live read paths are exercised against a stub substrate: no chain, no wallet, no network, no
spend. What they buy is coverage of the parts that have actually broken on a live chain before —
the `Raw<N>` commitment field decode, and reading two storage maps at one block.
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3.chain import (COMMITMENT_MAX_BYTES, ChainError, Commitment, Metagraph,
                                    MockChain, Neuron, ReadySignal, WeightRateLimited,
                                    check_weight_cadence, read_commitments, read_metagraph,
                                    read_weights_rate_limit)
from thirtyspokes.v3.emissions import Lineage, emission_weights

BURN_UID = 0
NETUID = 99

REGISTRATION = "a" * 64
MANIFEST = "b" * 64


def arena() -> MockChain:
    """A chain where the burn address is UID 0 and `hk_ex_king` holds the UID that gets recycled.

    The fillers are there so the ex-king sits at UID 5 rather than at 1: a recycling bug that only
    ever hit UID 1 would be indistinguishable from an off-by-one, and §5.7's example is UID 5.
    """
    chain = MockChain()
    for hotkey in ("hk_burn", "hk_f1", "hk_f2", "hk_f3", "hk_f4"):
        chain.register(hotkey)
    assert chain.register("hk_ex_king") == 5
    chain.register("hk_king")
    return chain


# --- the rule this module exists for: UID recycling (§5.7) ---------------------------------------

def test_the_mock_recycles_a_deregistered_uid_because_one_that_cannot_certifies_nothing():
    """Every other test in this file is only meaningful if this one holds."""
    chain = arena()

    chain.deregister("hk_ex_king")

    assert chain.register("hk_stranger") == 5
    assert chain.metagraph().hotkey_of[5] == "hk_stranger"


def test_a_recycled_uid_pays_the_stranger_nothing_and_burns_the_ex_kings_slot():
    """The pension resolved against a LIVE metagraph: `hk_stranger` earned nothing, so it gets 0.

    `hk_ex_king` keeps its position in the lineage — the ranks behind it do not move up (§5.7) — its
    0.05 burns, and UID 5 appears nowhere in the slate even though somebody holds it.
    """
    chain = arena()
    lineage = Lineage.from_coronations(("hk_ex_king", "hk_king"))
    chain.deregister("hk_ex_king")
    chain.register("hk_stranger")

    weights = emission_weights(lineage, "hk_king", chain.metagraph().hotkey_of)

    assert 5 not in weights
    assert weights[6] == pytest.approx(0.85)
    assert weights[BURN_UID] == pytest.approx(0.15)


def test_a_stale_metagraph_pays_the_stranger():
    """The bug, run to completion, so the fix above has something to be correct against.

    A snapshot taken at window open is up to a whole window stale, which is long enough for a
    deregistration and a re-registration to happen inside it. Resolved against that snapshot the
    schedule hands `hk_stranger` the ex-king's 0.05 at UID 5, and nothing about the slate — it still
    sums to 1.0 — says anything is wrong. This is why the metagraph read is a method and never a
    field (§5.7: resolved at weight-setting time).
    """
    chain = arena()
    lineage = Lineage.from_coronations(("hk_ex_king", "hk_king"))
    at_window_open = chain.metagraph().hotkey_of

    chain.deregister("hk_ex_king")
    chain.register("hk_stranger")
    at_weight_setting = chain.metagraph().hotkey_of

    assert at_window_open[5] == "hk_ex_king" and at_weight_setting[5] == "hk_stranger"
    assert emission_weights(lineage, "hk_king", at_window_open)[5] == pytest.approx(0.05)
    assert sum(emission_weights(lineage, "hk_king", at_window_open).values()) == 1.0
    assert 5 not in emission_weights(lineage, "hk_king", at_weight_setting)


def test_resolving_a_hotkey_that_lost_its_uid_returns_none_rather_than_its_old_number():
    """§5.5's crown reversion and §5.7's burning slot are both this one answer."""
    chain = arena()

    chain.deregister("hk_ex_king")
    chain.register("hk_stranger")
    metagraph = chain.metagraph()

    assert metagraph.resolve("hk_ex_king") is None
    assert metagraph.resolve("hk_stranger") == Neuron(uid=5, hotkey="hk_stranger",
                                                      registered_at=chain.current_block())


def test_the_metagraph_is_read_in_the_direction_that_verifies():
    """`uid -> hotkey`: resolution and verification are the same step, so neither can be skipped.

    Handed the inverse, `emission_weights` would resolve a hotkey to a UID that now belongs to
    somebody else and pay it — the check would have to be a separate line somebody remembered to
    write. Here there is nothing to remember.
    """
    chain = arena()

    hotkey_of = chain.metagraph().hotkey_of

    assert hotkey_of[BURN_UID] == "hk_burn"
    assert set(hotkey_of) == {0, 1, 2, 3, 4, 5, 6}


def test_a_commitment_carries_no_uid_so_a_remembered_one_cannot_reach_the_queue():
    """The queue is `(block, hotkey)` (§8b.1); a uid inside it would be a stale one with a window to
    go stale in. `Metagraph.resolve` is the only producer of a uid, and only from a live read."""
    assert "uid" not in Commitment.__dataclass_fields__
    assert set(Commitment.__dataclass_fields__) == {"hotkey", "block", "ready"}


# --- the ready signal (§7 step 5) ----------------------------------------------------------------

def test_a_ready_signal_round_trips_through_one_commitment_slot():
    signal = ReadySignal(registration_id=REGISTRATION, manifest_sha256=MANIFEST)

    assert ReadySignal.parse(signal.encode()) == signal


def test_the_packed_signal_fits_the_commitment_slot_and_the_readable_form_does_not():
    """Why the digests are packed rather than written out: `Raw<N>` caps at 128 bytes.

    The readable `r2ready:v1|<64 hex>|<64 hex>` is 140 bytes, so it could never have been committed —
    which is why `parse` refuses it instead of accepting it for symmetry with teutonic's reader.
    """
    packed = ReadySignal(REGISTRATION, MANIFEST).encode()
    readable = f"r2ready:v1|{REGISTRATION}|{MANIFEST}"

    assert len(packed.encode()) <= COMMITMENT_MAX_BYTES
    assert len(readable.encode()) > COMMITMENT_MAX_BYTES
    with pytest.raises(ValueError):
        ReadySignal.parse(readable)


@pytest.mark.parametrize("payload", [
    "",
    "kothgov1|deadbeef",                                    # the owner's own record, same map
    "r2ready:v2:" + "A" * 86,                               # a version this reader does not know
    "r2ready:v1:" + "A" * 85,                               # a character short of two digests
    "r2ready:v1:" + "A" * 86 + "=",                         # padding the encoder strips
])
def test_a_commitment_that_is_not_a_ready_signal_is_refused_with_a_reason(payload: str):
    """Refused, not returned as None: the caller must tell "not a v3 submission" (the owner's
    governance record shares this storage map) from "this miner committed something broken"."""
    with pytest.raises(ValueError):
        ReadySignal.parse(payload)


@pytest.mark.parametrize("registration,manifest", [
    ("A" * 64, MANIFEST),          # uppercase hex is a different string for a digest comparison
    ("a" * 63, MANIFEST),
    (REGISTRATION, "not a digest"),
])
def test_a_digest_that_is_not_a_lowercase_sha256_is_refused_at_construction(registration, manifest):
    with pytest.raises(ValueError):
        ReadySignal(registration_id=registration, manifest_sha256=manifest)


# --- the queue (§8b.1) ---------------------------------------------------------------------------

def test_commitments_come_back_in_the_order_the_queue_evaluates_them():
    """`(commit_block, hotkey)` — a property of the read, so no consumer has to remember to sort.

    Both the UID order and the commit order are set to DISAGREE with the queue order here: the
    earliest commit holds the lowest UID's successor, and the two same-block hotkeys are registered
    in reverse alphabetical order. A read that returned the metagraph's own order would look correct
    on any fixture where those three orders happen to coincide.
    """
    chain = arena()
    signal = ReadySignal(REGISTRATION, MANIFEST)
    chain.register("hk_zebra")                   # uid 7
    chain.register("hk_alpha")                   # uid 8
    chain.advance(10)
    chain.commit_ready("hk_ex_king", signal)     # uid 5, and first in the queue
    chain.advance(5)
    chain.commit_ready("hk_zebra", signal)
    chain.commit_ready("hk_alpha", signal)       # same block: ties break on hotkey

    queued = chain.commitments()

    assert [(c.hotkey, c.block) for c in queued] == [("hk_ex_king", 10), ("hk_alpha", 15),
                                                     ("hk_zebra", 15)]


def test_a_deregistered_hotkeys_commitment_is_not_in_the_queue_but_is_not_destroyed_either():
    """Faithful to the chain in both halves: `CommitmentOf` survives deregistration, and the reason
    a ghost is not queued is that the read walks the METAGRAPH rather than the commitment map."""
    chain = arena()
    chain.commit_ready("hk_ex_king", ReadySignal(REGISTRATION, MANIFEST))
    assert [c.hotkey for c in chain.commitments()] == ["hk_ex_king"]

    chain.deregister("hk_ex_king")
    assert chain.commitments() == ()

    chain.register("hk_ex_king")
    assert [c.hotkey for c in chain.commitments()] == ["hk_ex_king"]


def test_one_unreadable_slot_does_not_hide_the_rest_of_the_queue():
    """The mock stores exactly what a slot holds; this is about what the READ does with junk."""
    chain = arena()
    chain.commit_ready("hk_f1", ReadySignal(REGISTRATION, MANIFEST))
    chain._commit["hk_f2"] = ("kothgov1|not-a-submission", chain.current_block())

    assert [c.hotkey for c in chain.commitments()] == ["hk_f1"]


def test_committing_under_a_hotkey_that_holds_no_uid_is_refused():
    chain = arena()

    with pytest.raises(ValueError):
        chain.commit_ready("hk_never_registered", ReadySignal(REGISTRATION, MANIFEST))


# --- weights: the cadence floor (§8b.6) and owner downtime (D1) ----------------------------------

@pytest.mark.parametrize("window_blocks", [1, 99, 100])
def test_a_window_no_longer_than_the_weights_rate_limit_is_refused(window_blocks: int):
    """Equality is refused too, because the write path refuses it: `subnet/chain.py::set_weights`
    rate-limits at `blocks_since_last_update <= limit`. A gate looser than the writer's own guard
    would certify a window whose every second update is rejected."""
    with pytest.raises(ChainError):
        check_weight_cadence(window_blocks, rate_limit=100)


def test_a_window_longer_than_the_rate_limit_is_accepted():
    check_weight_cadence(101, rate_limit=100)


def test_a_slate_written_inside_the_rate_limit_is_rejected_and_the_old_one_stays_in_force():
    """§8b.6's failure, reproduced: the crown keeps paying whoever held it and nothing says so.

    This is what `check_weight_cadence` exists to catch at launch rather than in the ledger.
    """
    chain = MockChain(rate_limit=100)
    chain.register("hk_burn")
    chain.register("hk_king")
    chain.set_weights({0: 0.15, 1: 0.85})
    chain.advance(50)                                   # a 50-block window

    with pytest.raises(WeightRateLimited):
        chain.set_weights({0: 1.0})

    assert chain.weights == {0: 0.15, 1: 0.85}
    with pytest.raises(ChainError):
        check_weight_cadence(50, chain.weights_rate_limit())


def test_a_window_the_validator_missed_leaves_the_previous_slate_untouched():
    """D1/§8b.6: the previous schedule persists and the king keeps earning. There is deliberately no
    decay, no heartbeat and no default schedule for an outage to fall back to — decaying weights
    toward nothing would punish a king for the OWNER's outage."""
    chain = MockChain(rate_limit=100)
    chain.register("hk_burn")
    chain.register("hk_king")
    slate = {0: 0.15, 1: 0.85}
    chain.set_weights(slate)
    written_at = chain.weights_set_at

    chain.advance(7200 * 2)                             # two windows, one of them missed entirely

    assert chain.weights == slate
    assert chain.weights_set_at == written_at


def test_an_unchanged_slate_is_not_rewritten_unless_forced():
    """A stable reign IS an unchanged slate, and a validator that never re-submits ages out of
    Yuma's `activity_cutoff` while scoring every window correctly — measured on netuid 99, where
    `last_update` reached 2234 blocks against a cutoff of 5000 (`koth/neuron.py`). So `force` has to
    mean here what it means on the live chain, or a refresh cannot be tested offline at all."""
    chain = MockChain(rate_limit=100)
    chain.register("hk_burn")
    chain.register("hk_king")
    slate = {0: 0.15, 1: 0.85}
    chain.set_weights(slate)
    written_at = chain.weights_set_at
    chain.advance(200)

    chain.set_weights(dict(slate))
    assert chain.weights_set_at == written_at, "an unchanged slate must not count as a submission"

    chain.set_weights(dict(slate), force=True)
    assert chain.weights_set_at == written_at + 200


def test_the_slate_the_chain_records_is_the_one_emissions_built():
    """The seam's whole job, in one line: nothing between `emission_weights` and the chain."""
    chain = arena()
    lineage = Lineage.from_coronations(("hk_ex_king", "hk_king"))
    weights = emission_weights(lineage, "hk_king", chain.metagraph().hotkey_of)

    chain.set_weights(weights)

    assert chain.weights == weights
    assert chain.weights == pytest.approx({6: 0.85, 5: 0.05, BURN_UID: 0.10})


# --- the live reads, against a stub substrate (no chain, no wallet, no network) -------------------

class _Scale:
    """What the SDK hands back: a `ScaleType` whose payload is under `.value`."""

    def __init__(self, value):
        self.value = value


class _StubSubstrate:
    """The three substrate calls the live reads make, and a record of the block each was pinned to."""

    def __init__(self, *, keys, registered, slots=None):
        self.keys, self.registered, self.slots = keys, registered, slots or {}
        self.pinned: list[tuple[str, str | None]] = []

    def get_block_hash(self, block: int) -> str:
        return f"0xblock{block}"

    def query_map(self, module, storage_function, params, block_hash=None):
        assert module == "SubtensorModule" and params == [NETUID]
        self.pinned.append((storage_function, block_hash))
        rows = {"Keys": self.keys, "BlockAtRegistration": self.registered}[storage_function]
        return [(_Scale(key), _Scale(value)) for key, value in rows.items()]

    def query(self, module, storage_function, params, block_hash=None):
        assert (module, storage_function) == ("Commitments", "CommitmentOf")
        return _Scale(self.slots.get(params[1]))


def _slot(payload: str, block: int) -> dict:
    """A `CommitmentOf` record as the node serializes it: the SCALE variant is `Raw<byte length>`,
    never a bare `Raw`, and the payload arrives hex-encoded. Matching the literal key "Raw" read
    every commitment back as None on testnet 526 — `subnet/chain.py` owns that fix and this seam
    reuses it rather than re-deriving it."""
    return {"info": {"fields": [{f"Raw{len(payload)}": "0x" + payload.encode().hex()}]},
            "block": block}


def test_ownership_and_registration_are_read_at_one_block():
    """Two RPCs answering one question. Unpinned, a deregistration between them yields a uid whose
    holder comes from one chain state and whose registration block comes from another — and the
    registration block is what §8b.1's queue wait is computed from."""
    substrate = _StubSubstrate(keys={0: "hk_burn", 5: "hk_A"}, registered={0: 10, 5: 99})

    metagraph = read_metagraph(substrate, NETUID, block=1234)

    assert {block_hash for _, block_hash in substrate.pinned} == {"0xblock1234"}
    assert metagraph.block == 1234
    assert metagraph.neurons == (Neuron(0, "hk_burn", 10), Neuron(5, "hk_A", 99))


def test_a_uid_with_no_readable_registration_block_still_owns_it():
    """Ownership comes from `Keys` alone. Dropping the neuron would burn a live pensioner's share
    over a missing metadata field — a strictly worse error than an unknown queue wait."""
    substrate = _StubSubstrate(keys={5: "hk_A"}, registered={})

    metagraph = read_metagraph(substrate, NETUID, block=7)

    assert metagraph.resolve("hk_A") == Neuron(5, "hk_A", None)
    assert metagraph.hotkey_of == {5: "hk_A"}


def test_the_live_read_decodes_the_raw_field_and_returns_the_queue_in_order():
    """UID order and queue order are made to disagree, for the reason the mock's ordering test
    gives: a read that returned the metagraph's own order passes any fixture where they coincide."""
    signal = ReadySignal(REGISTRATION, MANIFEST)
    substrate = _StubSubstrate(keys={5: "hk_zebra", 7: "hk_alpha"}, registered={5: 1, 7: 1},
                               slots={"hk_zebra": _slot(signal.encode(), block=42),
                                      "hk_alpha": _slot(signal.encode(), block=42)})
    metagraph = read_metagraph(substrate, NETUID, block=50)

    queued = read_commitments(substrate, NETUID, metagraph)

    assert queued == (Commitment(hotkey="hk_alpha", block=42, ready=signal),
                      Commitment(hotkey="hk_zebra", block=42, ready=signal))


def test_the_owner_governance_record_in_the_same_map_is_not_read_as_a_submission():
    """`CommitmentOf` is one slot per hotkey and the owner's holds `kothgov1|…`. One unreadable slot
    must not blind the validator to the rest of the queue."""
    signal = ReadySignal(REGISTRATION, MANIFEST)
    substrate = _StubSubstrate(
        keys={0: "hk_owner", 5: "hk_A", 6: "hk_silent"}, registered={0: 1, 5: 1, 6: 1},
        slots={"hk_owner": _slot("kothgov1|" + "d" * 64, block=3),
               "hk_A": _slot(signal.encode(), block=42)})
    metagraph = read_metagraph(substrate, NETUID, block=50)

    queued = read_commitments(substrate, NETUID, metagraph)

    assert [commitment.hotkey for commitment in queued] == ["hk_A"]


def test_a_slot_belonging_to_a_hotkey_that_holds_no_uid_is_not_queued():
    """`CommitmentOf` survives deregistration on the real chain, so a read that walked the
    COMMITMENT map instead of the metagraph would queue ghosts — hotkeys with no uid to resolve, no
    immunity, and no share to pay. Walking the metagraph makes that structural."""
    signal = ReadySignal(REGISTRATION, MANIFEST)
    substrate = _StubSubstrate(keys={5: "hk_A"}, registered={5: 1},
                               slots={"hk_A": _slot(signal.encode(), block=42),
                                      "hk_ghost": _slot(signal.encode(), block=8)})
    metagraph = read_metagraph(substrate, NETUID, block=50)

    queued = read_commitments(substrate, NETUID, metagraph)

    assert [commitment.hotkey for commitment in queued] == ["hk_A"]


def test_a_weights_rate_limit_the_chain_cannot_report_fails_closed():
    """Read as 0 it would certify every window, including the ones §8b.6 refuses."""
    class _Subtensor:
        def __init__(self, limit):
            self.limit = limit

        def weights_rate_limit(self, netuid):
            assert netuid == NETUID
            return self.limit

    assert read_weights_rate_limit(_Subtensor(100), NETUID) == 100
    with pytest.raises(ChainError):
        read_weights_rate_limit(_Subtensor(None), NETUID)


def test_an_empty_metagraph_is_a_metagraph_and_not_a_crash():
    """Window 1 before anyone registers. The slate then burns everything, which is §8b.8's cold
    start, not an error condition."""
    metagraph = Metagraph(block=0)

    assert metagraph.hotkey_of == {}
    assert metagraph.resolve("hk_anybody") is None


def test_a_schedule_root_round_trips_through_a_commitment_slot():
    """§6.3's owner-side commitment: one root for the whole schedule, readable by anyone."""
    chain = MockChain()
    assert chain.schedule_root() is None, "nothing committed is not an error, it is the cold start"
    root = "ab" * 32
    chain.commit_schedule(root)
    assert chain.schedule_root() == root


def test_a_schedule_commitment_fits_the_slot_and_refuses_a_non_digest():
    """The 128-byte `Raw<N>` cap is what forced the ready signal to pack two digests; one written
    out fits, and anything that is not a digest is refused rather than committed."""
    from thirtyspokes.v3.chain import SCHEDULE_PREFIX, encode_schedule, parse_schedule

    payload = encode_schedule("cd" * 32)
    assert len(payload.encode()) <= 128 and payload.startswith(SCHEDULE_PREFIX)
    assert parse_schedule(payload) == "cd" * 32
    for bad in ("", "schedule:v1:nothex", "AB" * 32, SCHEDULE_PREFIX + "ab" * 31):
        with pytest.raises(ValueError):
            parse_schedule(bad) if bad.startswith(SCHEDULE_PREFIX) else encode_schedule(bad)


def test_committing_a_schedule_root_refuses_to_clobber_another_mechanism_s_record():
    """§6.3 says the owner's slot "already holds the governance record (`kothgov1|...`)", and
    `set_commitment` OVERWRITES. That record is what `subnet/chain.py::commit_governance` publishes;
    its own comment records the cost of losing it — validators fall back to the previous approved
    measurement, every miner on the current image is rejected as `unapproved_runtime`, "and the
    fault looks like it belongs to the miners". Netuid 99 runs that mechanism on mainnet today with
    earning miners, so an unguarded write here is a command that breaks a live subnet and blames its
    participants.
    """
    chain = MockChain()
    chain._schedule = "kothgov1|" + "de" * 16          # the live subnet's governance record
    with pytest.raises(ChainError, match="not a schedule root"):
        chain.commit_schedule("ab" * 32)
    assert chain._schedule.startswith("kothgov1|"), "the foreign record must survive the refusal"


def test_replacing_the_retired_governance_record_takes_an_explicit_flag():
    """The cutover's one deliberate overwrite (M10): with the other mechanism retired, its record is
    dead and the owner replaces it by saying so. The default stays the refusal above — the flag is a
    statement about the other mechanism that only the owner can make, and the guard must not stop
    applying just because a date passed."""
    chain = MockChain()
    chain._schedule = "kothgov1|" + "de" * 16
    chain.commit_schedule("ab" * 32, replace_governance=True)
    assert chain.schedule_root() == "ab" * 32
    # and it is not a standing permission: a second foreign record is refused again by default
    chain._schedule = "kothgov1|" + "ef" * 16
    with pytest.raises(ChainError, match="not a schedule root"):
        chain.commit_schedule("ab" * 32)


def test_re_committing_a_schedule_root_over_an_older_one_is_allowed():
    """A re-commit is ordinary — the world or the window count changed — and the validator checks
    the root against what it computes anyway, so the guard must not block the owner's own updates."""
    chain = MockChain()
    chain.commit_schedule("ab" * 32)
    chain.commit_schedule("cd" * 32)
    assert chain.schedule_root() == "cd" * 32


def test_the_mock_chains_raw_slot_reads_back_what_the_guard_compares_against():
    """The clobber guard in `commit_schedule` reads `_raw_commitment`, so the mock every offline test
    drives has to answer it. It shipped as a verbatim paste of the live body — dereferencing
    `_inner`/`_substrate`, which `MockChain` does not have — and so raised `AttributeError` on the
    one object the guard is exercised against. Dead code today only because `commit_schedule` reads
    the attribute directly; a mock whose method raises models nothing."""
    from thirtyspokes.v3.chain import SCHEDULE_PREFIX

    chain = MockChain()
    assert chain._raw_commitment() is None

    chain.commit_schedule("b" * 64)
    raw = chain._raw_commitment()
    assert raw is not None and raw.startswith(SCHEDULE_PREFIX)
    assert chain.schedule_root() == "b" * 64


def test_the_seam_declares_the_raw_slot_without_implementing_it():
    """`Chain` is a Protocol: every member states what an implementation must provide. A real body
    here — one implementation's substrate plumbing — is inherited by anything that structurally
    satisfies the seam and contradicts the mock that does."""
    import inspect

    from thirtyspokes.v3.chain import Chain

    body = inspect.getsource(Chain._raw_commitment)
    assert "_substrate" not in body and "_inner" not in body


def test_an_immunity_period_the_chain_cannot_report_fails_closed():
    """The mirror of the rate-limit read, and the only code in this module that runs against a live
    chain. Read as 0 it would certify every queue depth — which is verbatim the failure its own
    docstring says it prevents, and which nothing executed until now."""
    from thirtyspokes.v3.chain import read_immunity_period

    class _Subtensor:
        def __init__(self, period):
            self.period = period

        def immunity_period(self, netuid):
            assert netuid == NETUID, "the read must be scoped to the subnet it is asked about"
            return self.period

    assert read_immunity_period(_Subtensor(100), NETUID) == 100
    assert read_immunity_period(_Subtensor("7200"), NETUID) == 7200, "the chain may report a string"
    with pytest.raises(ChainError):
        read_immunity_period(_Subtensor(None), NETUID)


# --- the bittensor 11 seam: the three helpers the live class is built from --------------------------


class _FakeSubtensor11:
    """Bittensor 11's read surface as the seam uses it: storage `Item`s, reads pinned by block
    NUMBER, and `block_info(n).hash`. Records what it was asked so the test can see the pin."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str, list, int | None]] = []

    def block_info(self, block: int):
        return type("Info", (), {"hash": f"0xhash{block}"})()

    def query(self, item, params, *, block=None):
        self.asked.append((item.container, item.name, list(params), block))
        return {"ImmunityPeriod": 5000, "WeightsSetRateLimit": 100}.get(item.name)

    def query_map(self, item, params, *, block=None):
        self.asked.append((item.container, item.name, list(params), block))
        return [(0, "hk0")]


def test_storage_adapter_pins_reads_to_the_block_behind_the_hash_it_issued():
    """`read_metagraph` reads two maps at one `get_block_hash(block)`; the 11 SDK pins by block
    number. The adapter must translate the hash it handed out back to that number, or the two reads
    silently come from the head — the exact drift the pin exists to prevent."""
    from thirtyspokes.v3.chain import _Storage
    fake = _FakeSubtensor11(); storage = _Storage(fake)
    digest = storage.get_block_hash(7)
    assert digest == "0xhash7"
    storage.query_map("SubtensorModule", "Keys", [526], block_hash=digest)
    storage.query("Commitments", "CommitmentOf", [526, "hk0"])
    assert fake.asked == [("SubtensorModule", "Keys", [526], 7),
                          ("Commitments", "CommitmentOf", [526, "hk0"], None)]
    assert storage.immunity_period(526) == 5000 and storage.weights_rate_limit(526) == 100


def test_u16_weights_scale_the_largest_share_to_65535_and_drop_zeros():
    from thirtyspokes.v3.chain import _u16_weights
    uids, u16 = _u16_weights({0: 0.0, 3: 0.85, 7: 0.05, 9: 0.10})
    assert uids == [3, 7, 9] and u16[0] == 65535
    assert u16 == [65535, round(0.05 / 0.85 * 65535), round(0.10 / 0.85 * 65535)]
    with pytest.raises(ChainError, match="positive"):
        _u16_weights({0: 0.0})


def test_a_rejected_extrinsic_raises_with_the_chain_message():
    """A discarded result makes a rejected write indistinguishable from a landed one — the failure
    the old seam paid for once already."""
    from thirtyspokes.v3.chain import _accepted
    class Result:
        success = False; message = "Priority is too low"; error = None
    with pytest.raises(ChainError, match="ready signal REJECTED.*Priority is too low"):
        _accepted(Result(), "ready signal")
    _accepted(type("Ok", (), {"success": True})(), "ready signal")     # no raise
