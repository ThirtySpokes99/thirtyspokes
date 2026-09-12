"""Chain integration — who owns a UID *now*, who is queued, and what the schedule pays
(docs/WHITEPAPER.md §5.5, §5.7, §8b.1, §8b.6; the build plan M8b/M9).

THE LIVE SEAM IS WRITTEN AGAINST BITTENSOR 11.x (ported 2026-09-07 from the 10.5 seam it used to
hold). The 11 SDK dropped `Subtensor.substrate`; what it offers instead — `Subtensor.query` and
`query_map` over generated storage items pinned by block NUMBER, `block_info(...).hash`, and
`submit_call` over generated call builders — is wrapped once in `_Storage`, so `read_metagraph`,
`read_commitments` and `chain_beacon` keep the call shape they were tested against. Every finding
the old seam carried is kept, because each was measured on a live chain rather than reasoned about:
the `Raw<N>` commitment decode (`subnet/chain.py::_decode_raw_commitment` — the SCALE variant is
`Raw70`, never a bare `Raw`, so matching the literal key silently read every commitment back as
None), the checked extrinsic result (a commit once reported success, never appeared, and the
failure was visible only by re-submitting), and the weight write's rate-limit guard plus its
idempotent recovery.

*** THE RULE THIS MODULE EXISTS FOR: BITTENSOR RECYCLES UIDS (§5.7, risk register #22). ***

If ex-king `hk_A` held UID 5 and is deregistered, UID 5 is handed to an unrelated `hk_B`. A schedule
that remembered `{5: 0.04}` would then pay a stranger a pension they never earned — silently, every
window, forever, and invisibly in the totals, because the slate still sums to 1.0. So:

* the metagraph is read in the direction that VERIFIES, `uid -> hotkey`, which is the direction the
  chain stores it (`SubtensorModule.Keys`) and the direction `emissions.emission_weights` consumes.
  A hotkey that no longer holds its old UID is simply not in the map, and its share burns — there is
  no separate ownership check to forget, because resolution *is* the check;
* `Commitment` carries NO uid. A queued challenger is a hotkey and a block, and nothing else. A uid
  travelling from window open to weight-setting inside a queue entry is the same bug wearing a
  different hat;
* the read is a method, never a cached field. §5.7 requires resolution *at weight-setting time*, and
  a snapshot taken at window open is up to a whole window stale — which is long enough for a
  deregistration and a re-registration to happen inside it.

`MockChain` therefore RECYCLES UIDS. A mock that hands every registrant a fresh number cannot
reproduce the one failure this module exists to prevent, so it would certify nothing; the same
argument makes its weight write honour the rate limit (§8b.6) and short-circuit an unchanged slate.

WHAT IS DELIBERATELY ABSENT: any decay, heartbeat or default schedule. If the validator misses a
window, weights are left untouched — the previous slate persists and the king keeps earning (§8b.6).
That is the documented cost of the owner-liveness dependency (D1) and it is the right failure:
decaying weights toward nothing would punish a king for the owner's outage. The way to express "we
do nothing" in code is to have nothing here that could do something, which is why the only write is
one the caller must ask for.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

# Private on purpose in `subnet.chain`, imported anyway: `_decode_raw_commitment` encodes a live
# testnet finding (see this module's header). Copying it would let the mock and the chain drift
# apart on the one decode whose disagreement is undetectable from either side.
from ..subnet.chain import WeightRateLimited, _decode_raw_commitment

__all__ = ["Chain", "ChainError", "Commitment", "MockChain", "Metagraph", "Neuron", "ReadySignal",
           "SCHEDULE_PREFIX", "WeightRateLimited", "check_weight_cadence", "encode_schedule",
           "parse_schedule", "read_commitments", "read_metagraph",
           "read_immunity_period", "read_weights_rate_limit"]


class ChainError(RuntimeError):
    """A chain fact v3 cannot proceed without, or a window the chain would refuse to weight."""


# --- the challenger's on-chain ready signal (§7 step 5) -------------------------------------------

READY_PREFIX = "r2ready:v1:"

# A plain commitment's `Raw<N>` field caps at 128 bytes — measured in this repo as
# "Value 'Raw657' not present in type_mapping" when a ~657-byte governance record was pushed at it
# (`subnet/chain.py::publish_owner_measurements`). That cap is the whole reason the two digests are
# PACKED rather than written out: the readable form `r2ready:v1|<64 hex>|<64 hex>` is 140 bytes and
# could never have been committed at all, so it is refused here rather than accepted for symmetry.
COMMITMENT_MAX_BYTES = 128

# THE OWNER'S SIDE OF THE SAME MAP (§6.3). The whole schedule is committed ONCE, as one chained
# manifest root, and the reason is discretion rather than convenience: the draw is a pure function of
# a schedule entry (§6.3b, D12), so N entries are N slices, and an owner free to publish them
# epoch-by-epoch could pick a slice AFTER seeing who committed. A root fixed before any challenger
# commits removes that choice, and it is only removed if somebody other than the owner can read it —
# which is what putting it on chain is for. Computed and used locally it proves nothing to anyone.
#
# `schedule:v1:` + 64 hex is 76 bytes, inside `COMMITMENT_MAX_BYTES`, so the root is written out
# rather than packed: unlike the ready signal's two digests there is only one, and a form a person
# can read off a block explorer is worth more than the 12 bytes packing would save.
SCHEDULE_PREFIX = "schedule:v1:"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_READY = re.compile(rf"^{re.escape(READY_PREFIX)}(?P<identity>[A-Za-z0-9_-]{{86}})$")
_SCHEDULE = re.compile(rf"^{re.escape(SCHEDULE_PREFIX)}(?P<root>[0-9a-f]{{64}})$")


def encode_schedule(root: str) -> str:
    """The owner's commitment text for a chained-manifest root."""
    if not _HEX64.fullmatch(root):
        raise ValueError(f"a schedule root must be a lowercase SHA-256 hex digest, got {root!r}")
    payload = SCHEDULE_PREFIX + root
    if len(payload.encode()) > COMMITMENT_MAX_BYTES:
        raise ValueError(f"schedule commitment is {len(payload.encode())} bytes, over the "
                         f"{COMMITMENT_MAX_BYTES}-byte cap")
    return payload


def parse_schedule(payload: str) -> str:
    """One commitment slot -> the schedule root, or raise.

    Raises rather than returning None for the same reason `ReadySignal.parse` does: a caller reading
    a whole subnet's slots must tell "this is not a schedule commitment" from "the owner committed
    something broken", and only a reason distinguishes them.
    """
    match = _SCHEDULE.fullmatch(payload or "")
    if match is None:
        raise ValueError(f"not a {SCHEDULE_PREFIX} commitment: {payload!r}")
    return match.group("root")


@dataclass(frozen=True)
class ReadySignal:
    """Two SHA-256 in one commitment slot: teutonic's `r2ready:v1` encoding, unchanged.

    WHY IT BINDS BOTH, and not just the tree. `manifest_sha256` names the uploaded model tree;
    `registration_id` names the one-shot registration that produced the upload prefix, which the
    access controller has already tied to this hotkey and permanently consumed. A commitment naming
    only a manifest would let a hotkey claim a tree it never uploaded — and under D14 the king's
    weights are public, so the most attractive tree to claim is one that already exists and already
    wins. Binding the registration is what makes the commitment say *this hotkey's own shot produced
    this tree*.

    The encoding is copied from `teutonic/access/contracts.py::ready_signal_payload` rather than
    re-invented, because the miner tooling that writes it lives there and a second encoding would
    mean a miner's commit and the validator's read could disagree while both looked correct.
    """

    registration_id: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (("registration_id", self.registration_id),
                             ("manifest_sha256", self.manifest_sha256)):
            if not _HEX64.fullmatch(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest, got {digest!r}")

    def encode(self) -> str:
        """The commitment text. Padding is stripped: 64 bytes is 86 base64url characters + `==`."""
        packed = bytes.fromhex(self.registration_id) + bytes.fromhex(self.manifest_sha256)
        return READY_PREFIX + base64.urlsafe_b64encode(packed).rstrip(b"=").decode("ascii")

    @classmethod
    def parse(cls, payload: str) -> ReadySignal:
        """Decode one commitment slot, or raise.

        Raises rather than returning None because the caller reading a whole subnet's slots
        (`read_commitments`) must distinguish "this is not a v3 submission" — the owner's own
        governance record shares this storage map — from "this miner committed something broken",
        and it can only do that if the failure carries a reason.
        """
        match = _READY.fullmatch(payload or "")
        if match is None:
            raise ValueError(f"not a {READY_PREFIX} commitment: {payload!r}")
        packed = base64.urlsafe_b64decode(match.group("identity") + "==")
        return cls(registration_id=packed[:32].hex(), manifest_sha256=packed[32:].hex())


@dataclass(frozen=True)
class Commitment:
    """One challenger's submission as the chain holds it: a hotkey, a block, and two digests.

    NO UID FIELD, DELIBERATELY. The queue is ordered by `(block, hotkey)` (§8b.1) and the schedule is
    resolved from hotkeys at weight-setting time (§5.7); a uid recorded here would be a remembered
    uid with a window's worth of time to go stale in, which is the recycling failure this module
    exists to prevent. `Metagraph.resolve` is the only place a uid is ever produced, and it produces
    one only from a live read.
    """

    hotkey: str
    block: int
    ready: ReadySignal


# --- the metagraph (§5.5, §5.7, §8b.1) ------------------------------------------------------------

@dataclass(frozen=True)
class Neuron:
    """A UID and the hotkey that holds it *at the block this was read*.

    `registered_at` is `None` when the chain reported a uid in `Keys` with no `BlockAtRegistration`.
    Ownership comes from `Keys` alone, so such a neuron still owns its uid: dropping it would burn a
    live pensioner's share over a missing metadata field, which is a strictly worse error than
    publishing an unknown queue wait.
    """

    uid: int
    hotkey: str
    registered_at: int | None = None


@dataclass(frozen=True)
class Metagraph:
    """One consistent read of who holds what, at one block.

    `block` is not decoration: `hotkey_of` and every `registered_at` are read at that block's hash
    (`read_metagraph`), so ownership and registration cannot come from two different chain states.
    Two unpinned reads can straddle a deregistration and report a uid's old holder beside its new
    occupant's registration block — the same class of mistake as remembering a uid, in miniature.
    """

    block: int
    neurons: tuple[Neuron, ...] = ()

    @property
    def hotkey_of(self) -> dict[int, str]:
        """`uid -> hotkey` — exactly what `emissions.emission_weights` takes, in that direction.

        The direction is the verification. `emission_weights` inverts this map, so a pensioner
        appears in the slate only at a uid the chain currently says is its own, and a hotkey that
        lost its uid resolves to nothing and burns. Handing it `hotkey -> uid` instead would make
        resolution succeed for a hotkey whose uid now belongs to somebody else.
        """
        return {neuron.uid: neuron.hotkey for neuron in self.neurons}

    def resolve(self, hotkey: str) -> Neuron | None:
        """The neuron this hotkey holds right now, or None — the verified `hotkey -> uid` step.

        None means "holds no uid on this subnet at this block", which is what §5.5 reverts a crown
        on and what §5.7 burns a pension slot for. It is never an error: a king deregistering is an
        ordinary event that the schedule is required to survive.
        """
        return next((neuron for neuron in self.neurons if neuron.hotkey == hotkey), None)


# --- the seam ------------------------------------------------------------------------------------

class Chain(Protocol):
    """What v3 asks of a chain. Deliberately six methods.

    `commit_ready` takes the hotkey explicitly even though the live chain can only ever write under
    its own wallet, matching `subnet.chain.Chain.commit`: the mock serves many miners in one process
    and the live implementation checks the argument against its wallet rather than ignoring it, so a
    test cannot pass by asking for something the real chain would not do.
    """

    def current_block(self) -> int: ...
    def metagraph(self) -> Metagraph: ...
    def commitments(self) -> tuple[Commitment, ...]: ...
    def commit_ready(self, hotkey: str, signal: ReadySignal) -> None: ...
    # The owner's side of §6.3: one root for the whole schedule, written before any challenger
    # commits and readable by anyone. `schedule_root()` returns None when nothing is committed,
    # which is a launch-time refusal rather than an error — see `Validator.preflight`.
    # `replace_governance` is the cutover's one deliberate overwrite: once the other mechanism is
    # retired its `kothgov1|…` record is dead, and the owner replaces it by saying so, not by the
    # guard below silently ceasing to apply.
    def commit_schedule(self, root: str, *, replace_governance: bool = False) -> None: ...
    # The raw slot behind `schedule_root`, shared by the read and the clobber guard so both see
    # the same bytes. On the seam because both implementations need it; a body here would put one
    # implementation's private plumbing into the interface every other member states abstractly.
    def _raw_commitment(self) -> str | None: ...

    def schedule_root(self) -> str | None: ...
    # `force` re-submits an UNCHANGED slate, which both implementations otherwise skip. A stable
    # reign is precisely an unchanged slate, and a validator that never re-submits ages out of Yuma's
    # `activity_cutoff` while scoring every window correctly — measured on netuid 99, where
    # `last_update` reached 2234 blocks against a cutoff of 5000 (`koth/neuron.py`).
    def set_weights(self, weights: Mapping[int, float], *, force: bool = False) -> None: ...
    def weights_rate_limit(self) -> int: ...
    # How stale the slate this hotkey has ON CHAIN is, or None if it has never set one. Read
    # after a write so the reveal can say the slate landed rather than that it was attempted.
    def blocks_since_weight_update(self) -> int | None: ...
    # §8b.1's chain-side obligation, READ rather than taken on trust. See `read_immunity_period`.
    def immunity_period(self) -> int: ...


def check_weight_cadence(window_blocks: int, rate_limit: int) -> None:
    """§8b.6: refuse a window the chain would not accept a weight update for.

    A window shorter than `weights_rate_limit` means the validator sets weights faster than the chain
    accepts, the write is rejected, and **an older schedule silently stays in force** — the crown
    keeps paying whoever held it, and nothing in the reveal says the slate on chain is not the one
    that was computed. The check is a launch-time gate on the window length, not a per-write guard;
    the guard is in the write path, where it raises `WeightRateLimited`.

    STRICTLY GREATER, not "no shorter than" as §8b.6 words it, because the write path this seam
    delegates to refuses at equality: `subnet/chain.py::set_weights` rate-limits when
    `blocks_since_last_update <= limit`. A check looser than the writer's own guard would certify a
    window whose every second update is rejected, which is the failure it exists to prevent.
    """
    if window_blocks <= rate_limit:
        raise ChainError(
            f"a {window_blocks}-block window is not longer than the chain's weights_rate_limit of "
            f"{rate_limit} blocks (§8b.6): the validator would set weights faster than the chain "
            f"accepts, the update would be rejected, and the PREVIOUS schedule would silently stay "
            f"in force. Lengthen the window past {rate_limit} blocks.")


# --- live reads, as functions over the SDK objects that own them ----------------------------------
# They are module-level and duck-typed rather than methods so that the decode paths — the part that
# has actually broken before — are exercised against a stub in tests, with no chain, no wallet and no
# network. `BittensorChain` below is then thin enough to read as wiring.

def _value(entry):
    """Unwrap a scalecodec `ScaleType`, or pass a plain value through (SDKs return both)."""
    return getattr(entry, "value", entry)


def read_metagraph(substrate, netuid: int, block: int) -> Metagraph:
    """`Keys` + `BlockAtRegistration`, both pinned to one block hash.

    `Keys` (uid -> hotkey) is the ownership truth and is the map `subnet/chain.py` already proved
    against a live testnet: `subtensor.metagraph()` also fetches stakes, neuron metadata and
    MetagraphInfo, and the sole public endpoint repeatedly timed out on that unrelated runtime API
    and blocked every validator. The selective `get_metagraph_info` would return both fields in one
    runtime-API call, and it is deliberately not used for the same reason.

    Both maps are read at `get_block_hash(block)` because they are two RPCs answering one question.
    Unpinned, a deregistration between them yields a uid whose holder comes from one chain state and
    whose registration block comes from another — and the registration block is what §8b.1's queue
    wait is computed from, so the miner told they are inside their immunity period would be the
    wrong miner.
    """
    block_hash = substrate.get_block_hash(block)
    registered = {
        int(_value(uid)): int(_value(at))
        for uid, at in substrate.query_map(module="SubtensorModule",
                                           storage_function="BlockAtRegistration",
                                           params=[netuid], block_hash=block_hash)}
    neurons = sorted(
        (Neuron(uid=int(_value(uid)), hotkey=str(_value(hotkey)),
                registered_at=registered.get(int(_value(uid))))
         for uid, hotkey in substrate.query_map(module="SubtensorModule", storage_function="Keys",
                                                params=[netuid], block_hash=block_hash)),
        key=lambda neuron: neuron.uid)
    return Metagraph(block=block, neurons=tuple(neurons))


def read_commitments(substrate, netuid: int, metagraph: Metagraph) -> tuple[Commitment, ...]:
    """Every readable `r2ready:v1` commitment, IN QUEUE ORDER (§8b.1).

    Only hotkeys currently on the metagraph are looked up, which makes "a deregistered hotkey is not
    in the queue" structural rather than a filter somebody has to remember: the commitment survives
    deregistration in `CommitmentOf` on the real chain, so a loop over the storage map instead of
    over the metagraph would queue ghosts.

    A slot that does not decode is SKIPPED, not raised on, for the reason `decode_revealed` gives:
    the owner's own governance record lives in this same map, and one malformed payload must not
    blind the validator to every other miner's submission. Returned sorted by `(block, hotkey)` so
    the queue order is a property of the read rather than of whoever consumes it.
    """
    found: list[Commitment] = []
    for neuron in metagraph.neurons:
        try:
            record = _value(substrate.query("Commitments", "CommitmentOf", [netuid, neuron.hotkey]))
            raw = _decode_raw_commitment(record["info"]["fields"]) if record else None
            if raw is None:
                continue
            found.append(Commitment(hotkey=neuron.hotkey, block=int(record["block"]),
                                    ready=ReadySignal.parse(raw)))
        except Exception:  # noqa: BLE001 — one unreadable slot must not hide the rest of the queue
            continue
    return tuple(sorted(found, key=lambda commitment: (commitment.block, commitment.hotkey)))


def read_immunity_period(subtensor, netuid: int) -> int:
    """The chain's `immunity_period`, or refuse to guess one.

    §8b.1 makes this the owner's one chain-side obligation, and the daemon used to check the number
    the owner TYPED at `--immunity-blocks` rather than the one the subnet actually has. That gate
    certifies a claim, not a fact: an owner who sets the flag to three windows and the subnet to one
    passes launch and still deregisters challengers who have paid a registration burn and uploaded
    ~70 GB — the exact outcome the "Kind" property forbids, arriving through the check meant to
    prevent it.

    Fails closed for the same reason `read_weights_rate_limit` does: a missing period read as 0
    would certify every queue depth.
    """
    period = subtensor.immunity_period(netuid)
    if period is None:
        raise ChainError(f"the chain did not report an immunity_period for netuid {netuid}; "
                         f"§8b.1's queue-drain check cannot be made against an unknown period")
    return int(period)


def read_weights_rate_limit(subtensor, netuid: int) -> int:
    """The chain's `weights_rate_limit`, or refuse to guess one.

    Fails closed because the number's only consumer is §8b.6's window-length gate: a missing limit
    read as 0 would certify every window, including the ones whose second weight update is rejected
    and whose old schedule then quietly keeps paying.
    """
    limit = subtensor.weights_rate_limit(netuid)
    if limit is None:
        raise ChainError(f"the chain did not report a weights_rate_limit for netuid {netuid}; "
                         f"§8b.6's window-length check cannot be made against an unknown limit")
    return int(limit)


# --- offline -------------------------------------------------------------------------------------

@dataclass
class MockChain:
    """In-memory chain for offline runs — faithful in the three places that decide money.

    1. **It recycles UIDs.** `deregister` frees the number and the next registrant takes the lowest
       free one, exactly as the pension's failure mode requires. A mock that always handed out a
       fresh uid would make every recycling test vacuously green.
    2. **It honours the rate limit**, so §8b.6's failure — the update is rejected and the OLD slate
       stays in force — is reproducible offline instead of being a paragraph.
    3. **It skips an unchanged slate unless forced**, using the live chain's own comparison, so the
       `force` flag means here what it means there.

    A commitment SURVIVES deregistration, as `CommitmentOf` does on the real chain. It becomes
    unreadable simply because `commitments()` walks the metagraph, which is the same reason it is
    unreadable live.
    """

    # 100 blocks is the value `koth/neuron.py` records for the live subnet; it is a field rather than
    # a constant so a test can set a window shorter than it and watch §8b.6 happen.
    rate_limit: int = 100
    # §8b.1's obligation, as a field for the same reason: a test sets it short and watches the
    # launch gate refuse. Generous by default so it is the tests that opt into the failure.
    immunity: int = 100_000
    block: int = 0
    weights: dict[int, float] = field(default_factory=dict)
    weights_set_at: int | None = None

    _uid_of: dict[str, int] = field(default_factory=dict)
    _registered_at: dict[str, int] = field(default_factory=dict)
    _commit: dict[str, tuple[str, int]] = field(default_factory=dict)   # hotkey -> (payload, block)
    _free: list[int] = field(default_factory=list)
    _next_uid: int = 0
    _schedule: str | None = None

    # --- what a test drives it with ---------------------------------------------------------
    def register(self, hotkey: str) -> int:
        if hotkey not in self._uid_of:
            self._uid_of[hotkey] = self._free.pop(0) if self._free else self._claim_uid()
            self._registered_at[hotkey] = self.block
        return self._uid_of[hotkey]

    def _claim_uid(self) -> int:
        uid, self._next_uid = self._next_uid, self._next_uid + 1
        return uid

    def deregister(self, hotkey: str) -> None:
        """Free the uid FOR REUSE. The commitment stays, as it does on chain."""
        uid = self._uid_of.pop(hotkey, None)
        self._registered_at.pop(hotkey, None)
        if uid is not None:
            self._free = sorted([*self._free, uid])

    def advance(self, n: int = 1) -> None:
        self.block += n

    # --- Chain ---------------------------------------------------------------------------------
    def current_block(self) -> int:
        return self.block

    def metagraph(self) -> Metagraph:
        return Metagraph(block=self.block, neurons=tuple(sorted(
            (Neuron(uid=uid, hotkey=hotkey, registered_at=self._registered_at.get(hotkey))
             for hotkey, uid in self._uid_of.items()), key=lambda neuron: neuron.uid)))

    def commitments(self) -> tuple[Commitment, ...]:
        found: list[Commitment] = []
        for neuron in self.metagraph().neurons:
            payload, block = self._commit.get(neuron.hotkey, (None, 0))
            try:
                found.append(Commitment(neuron.hotkey, block, ReadySignal.parse(payload or "")))
            except ValueError:
                continue
        return tuple(sorted(found, key=lambda commitment: (commitment.block, commitment.hotkey)))

    def commit_schedule(self, root: str, *, replace_governance: bool = False) -> None:
        # Models the live guard: the owner's slot is shared with whatever else publishes there, and
        # clobbering a foreign record breaks a running subnet (see `BittensorChain.commit_schedule`).
        # Through `_raw_commitment`, as `BittensorChain.commit_schedule` does, so the mock every
        # offline test drives exercises the same read the live guard depends on rather than a
        # parallel one that can drift from it.
        held = self._raw_commitment()
        if held is not None and not held.startswith(SCHEDULE_PREFIX) and not replace_governance:
            raise ChainError(
                f"the owner's commitment slot already holds {held[:24]!r}, which is not a "
                f"schedule root; overwriting it would destroy another mechanism's governance record "
                f"(pass replace_governance=True only once that mechanism is retired)")
        self._schedule = encode_schedule(root)

    def _raw_commitment(self) -> str | None:
        """The owner slot's raw contents — here just the string `commit_schedule` guards against.

        The live chain decodes this out of a substrate record; the mock IS the record, so the two
        agree on what the method means rather than on how it is fetched.
        """
        return self._schedule

    def schedule_root(self) -> str | None:
        return None if self._schedule is None else parse_schedule(self._schedule)

    def commit_ready(self, hotkey: str, signal: ReadySignal) -> None:
        if hotkey not in self._uid_of:
            raise ValueError(f"hotkey {hotkey} holds no uid; nothing to commit against")
        self._commit[hotkey] = (signal.encode(), self.block)

    def set_weights(self, weights: Mapping[int, float], *, force: bool = False) -> None:
        if not force and same_weight_distribution(dict(weights),
                                                                list(self.weights.items())):
            return
        elapsed = None if self.weights_set_at is None else self.block - self.weights_set_at
        if elapsed is not None and elapsed <= self.rate_limit:
            raise WeightRateLimited(self.rate_limit - elapsed + 1)
        self.weights = dict(weights)
        self.weights_set_at = self.block

    def weights_rate_limit(self) -> int:
        return self.rate_limit

    def blocks_since_weight_update(self) -> int | None:
        return None if self.weights_set_at is None else self.block - self.weights_set_at

    def immunity_period(self) -> int:
        return self.immunity


class _Storage:
    """Bittensor 11's `Subtensor.query`/`query_map` behind the call shape the readers were written
    against: `(module, storage_function, params, block_hash=)`.

    The 11 SDK pins a read by block NUMBER and names storage by generated `Item`; the readers pin by
    the hash `get_block_hash` handed them, because two reads answering one question must see one
    state (`read_metagraph`). So this remembers which number each hash it issued stands for.
    """

    def __init__(self, subtensor) -> None:
        self._subtensor = subtensor
        self._numbers: dict[str, int] = {}

    @staticmethod
    def _item(module: str, storage_function: str):
        from bittensor._generated import storage  # noqa: PLC0415
        return getattr(getattr(storage, module), storage_function)

    def get_block_hash(self, block: int) -> str:
        digest = str(self._subtensor.block_info(int(block)).hash)
        self._numbers[digest] = int(block)
        return digest

    def query(self, module: str, storage_function: str, params, block_hash: str | None = None):
        return self._subtensor.query(self._item(module, storage_function), list(params),
                                     block=self._numbers.get(block_hash) if block_hash else None)

    def query_map(self, module: str, storage_function: str, params, block_hash: str | None = None):
        return self._subtensor.query_map(self._item(module, storage_function), list(params),
                                         block=self._numbers.get(block_hash) if block_hash else None)

    # The two hyperparameters `read_immunity_period` / `read_weights_rate_limit` ask for.
    def immunity_period(self, netuid: int):
        return self.query("SubtensorModule", "ImmunityPeriod", [netuid])

    def weights_rate_limit(self, netuid: int):
        return self.query("SubtensorModule", "WeightsSetRateLimit", [netuid])


def same_weight_distribution(requested: Mapping[int, float], current) -> bool:
    """What "the slate did not change" MEANS on chain: equal normalised distributions, ignoring zero
    entries and u16 quantisation (`2 / 65535`). The mock and the live seam share this one definition."""
    want = {int(uid): float(weight) for uid, weight in requested.items() if weight > 0}
    got = {int(uid): float(weight) for uid, weight in (current or []) if weight > 0}
    want_total, got_total = sum(want.values()), sum(got.values())
    if not want_total or not got_total or set(want) != set(got):
        return False
    return all(abs(want[uid] / want_total - got[uid] / got_total) <= 2 / 65535 for uid in want)


def _u16_weights(slate: Mapping[int, float]) -> tuple[list[int], list[int]]:
    """The chain stores weights as u16 per uid; the largest share becomes 65535 and zeros are dropped
    (the 10.x SDK's own conversion, which the chain then normalises)."""
    positive = {int(uid): float(share) for uid, share in slate.items() if share > 0}
    if not positive:
        raise ChainError("a weight slate must carry at least one positive share")
    top = max(positive.values())
    uids = sorted(positive)
    return uids, [max(1, round(positive[uid] / top * 65535)) for uid in uids]


def _accepted(result, what: str) -> None:
    """Raise unless the chain accepted the extrinsic. A discarded result makes a REJECTED write
    indistinguishable from a successful one, and that has cost hours before (this module's header)."""
    if not getattr(result, "success", False):
        raise ChainError(f"{what} REJECTED by the chain: "
                         f"{getattr(result, 'message', '') or getattr(result, 'error', None) or result!r}")


class BittensorChain:  # pragma: no cover — needs a live chain + wallet
    """The live seam, on bittensor 11.x — exactly the `Chain` protocol and nothing else.

    Reads go through `_Storage` (generated storage items, pinned by block); the two writes —
    a commitment and a weight update — go through `Subtensor.submit_call` with the generated call
    builders, signed by the HOTKEY, and every result is checked (`_accepted`) rather than trusted.
    Verified against testnet 526 on 2026-09-07: reads live, both extrinsics composed and prepared
    as unsigned extrinsics without submission.
    """

    def __init__(self, netuid: int, wallet_name: str, network: str = "finney", *,
                 hotkey: str = "default"):
        import bittensor as bt  # noqa: PLC0415 — the SDK is an extra; the mock needs none of it
        self.netuid = netuid
        self.network = network
        self.subtensor = bt.Subtensor(network=network)
        self.wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        self._storage = _Storage(self.subtensor)

    @property
    def _substrate(self) -> "_Storage":
        """What `chain_beacon` and the two readers drive: block hashes and pinned storage reads."""
        return self._storage

    def current_block(self) -> int:
        return int(self.subtensor.block)

    def metagraph(self) -> Metagraph:
        return read_metagraph(self._substrate, self.netuid, self.current_block())

    def commitments(self) -> tuple[Commitment, ...]:
        return read_commitments(self._substrate, self.netuid, self.metagraph())

    def commit_ready(self, hotkey: str, signal: ReadySignal) -> None:
        """Write this wallet's ready signal, and RAISE if the chain did not accept it.

        The plain commitment map, not the timelocked reveal one: the validator reads this while
        building the queue, so it must be visible immediately, and `(block, hotkey)` is the queue
        order (§8b.1) — a reveal delayed ~360 blocks would order challengers by when the owner got
        around to reading them. Nothing here needs the timelock's anti-front-run property: the slice
        is drawn after commits from a chain nonce nobody controls (§6.3).
        """
        mine = self.wallet.hotkey.ss58_address
        if hotkey != mine:
            raise ChainError(f"cannot commit under {hotkey}: this wallet is {mine}. The chain writes "
                             f"a commitment under the signing hotkey and nothing else, so a mismatch "
                             f"here would put the signal in the wrong slot rather than in none.")
        self._commit(signal.encode(), "ready signal")

    def commit_schedule(self, root: str, *, replace_governance: bool = False) -> None:
        """Write the owner's chained-manifest root, and RAISE if the chain did not accept it.

        The same plain commitment map the ready signals live in — the owner's slot rather than a
        miner's, and `set_commitment` OVERWRITES, which is exactly why §6.3 commits ONE root for the
        whole schedule instead of a hash per window: a per-window write would erase the previous one
        and there would be nothing for a third party to check an earlier window against.

        IT READS BEFORE IT WRITES, AND REFUSES A SLOT THAT BELONGS TO SOMETHING ELSE. §6.3 says in
        as many words that the owner's slot "already holds the governance record (`kothgov1|…`)",
        and that record is not decoration: `subnet/chain.py::commit_governance` publishes the
        approved-measurement digest there, and its own comment records what losing it costs —
        validators fall back to the PREVIOUS approved measurement, every miner running the new image
        is rejected as `unapproved_runtime`, "and the fault looks like it belongs to the miners".
        Netuid 99 is live on mainnet with earning miners today, and v3 does not replace that
        mechanism until the cutover, so an unguarded `set_commitment` here is a command that breaks a
        running subnet and blames its participants. Overwriting a previous SCHEDULE root is fine —
        that is a re-commit, and the validator checks the root against what it computes anyway.

        `replace_governance` is the cutover's deliberate overwrite (M10, 2026-09-07): the KOTH
        mechanism the record belonged to is retired, its validators are stopped, and the slot on
        every netuid it ran on still holds `kothgov1|…`. The owner passes the flag ONCE, having
        read what the slot holds, and from then on the slot holds a schedule root and the guard
        is ordinary again. The default stays refusing, because the flag is a statement about the
        other mechanism that only the owner can make.
        """
        held = self._raw_commitment()
        if held and not held.startswith(SCHEDULE_PREFIX) and not replace_governance:
            raise ChainError(
                f"the owner's commitment slot on netuid {self.netuid} already holds {held[:24]!r}, "
                f"which is not a schedule root. `set_commitment` OVERWRITES, so writing here would "
                f"destroy it — and if that is the KOTH governance record (§6.3), every validator "
                f"falls back to the previous approved measurement and rejects every miner on the "
                f"current image as `unapproved_runtime`, which reads as the miners' fault. Use a "
                f"netuid or an owner hotkey that v3 owns outright, or retire the other mechanism "
                f"first and re-run with --replace-governance-record.")
        if held and replace_governance:
            print(f"[v3-chain] replacing the retired record {held[:24]!r}… in the owner's slot on "
                  f"netuid {self.netuid} with a schedule root")
        self._commit(encode_schedule(root), "schedule root")

    def _raw_commitment(self) -> str | None:
        """This wallet's own commitment slot, decoded, or None. One decode path for read and guard."""
        mine = self.wallet.hotkey.ss58_address
        record = _value(self._substrate.query("Commitments", "CommitmentOf", [self.netuid, mine]))
        return _decode_raw_commitment(record["info"]["fields"]) if record else None

    def schedule_root(self) -> str | None:
        """The owner's committed root, or None if the slot holds nothing or holds something else.

        None rather than an exception for a slot holding a NON-schedule payload, because that is the
        ordinary state of every other hotkey's slot and of the owner's own before it commits. A
        malformed schedule commitment is the case `parse_schedule` raises on, and it reaches the
        caller.
        """
        raw = self._raw_commitment()
        if raw is None:
            return None
        try:
            return parse_schedule(raw)
        except ValueError:
            return None

    def _commit(self, text: str, what: str) -> None:
        """One `Commitments.set_commitment` under this wallet's hotkey, checked.

        The payload is the SCALE `Data::Raw<N>` variant where N is the BYTE length — the same shape
        `_decode_raw_commitment` reads back — which bounds a commitment at 128 bytes; both texts
        this seam writes are under that (`COMMITMENT_MAX_BYTES`).
        """
        import bittensor as bt  # noqa: PLC0415
        data = text.encode()
        if len(data) > COMMITMENT_MAX_BYTES:
            raise ChainError(f"{what} is {len(data)} bytes; a commitment holds {COMMITMENT_MAX_BYTES}")
        call = bt.calls.Commitments.set_commitment(self.netuid, {"fields": [{f"Raw{len(data)}": data}]})
        _accepted(self.subtensor.submit_call(call, self.wallet, signer="hotkey",
                                             wait_for_inclusion=True, wait_for_finalization=False),
                  what)

    def _uid(self) -> int | None:
        uid = self._storage.query("SubtensorModule", "Uids", [self.netuid, self.wallet.hotkey.ss58_address])
        return None if uid is None else int(uid)

    def _blocks_since_update(self, uid: int) -> int | None:
        """`LastUpdate[netuid][uid]` against the head — the staleness Yuma's activity cutoff acts on."""
        updates = self._storage.query("SubtensorModule", "LastUpdate", [self.netuid]) or []
        if uid >= len(updates):
            return None
        return int(self.subtensor.block) - int(updates[uid])

    def _weights_are_current(self, requested: Mapping[int, float]) -> bool:
        """True when this hotkey already has the requested distribution on chain."""
        try:
            uid = self._uid()
            if uid is None:
                return False
            rows = self._storage.query("SubtensorModule", "Weights", [self.netuid, uid]) or []
            return same_weight_distribution(requested, rows)
        except Exception:  # noqa: BLE001 — a failed preflight must not suppress a real submission
            return False

    def set_weights(self, weights: Mapping[int, float], *, force: bool = False) -> None:
        """Set weights, rate-limit-guarded and idempotent — the 10.x seam's rules, on the 11 SDK.

        `force` SKIPS THE ALREADY-CURRENT SHORT-CIRCUIT, and without it a periodic refresh is
        impossible: the guard compares distributions only, so re-submitting an unchanged slate
        returned immediately and `last_update` never moved. The rate limit is still honoured — a
        refresh that arrives too soon raises `WeightRateLimited` exactly like a real change. And
        THE FALLBACK CANNOT VOUCH FOR A REFRESH: on a forced refresh the distribution is current by
        definition, so a rejected write is a failure, not a landed one.
        """
        slate = {int(uid): float(share) for uid, share in weights.items()}
        if not force and self._weights_are_current(slate):
            return
        uid = self._uid()
        if uid is None:
            raise ChainError(f"{self.wallet.hotkey.ss58_address} holds no uid on netuid {self.netuid}; "
                             f"it cannot set weights")
        limit = self.weights_rate_limit()
        since = self._blocks_since_update(uid)
        if since is not None and since <= limit:
            raise WeightRateLimited(int(limit - since + 1))
        import bittensor as bt  # noqa: PLC0415
        version_key = int(self._storage.query("SubtensorModule", "WeightsVersionKey", [self.netuid]) or 0)
        # The SDK's own intent rather than a raw `set_weights` call: it conforms the slate to the
        # subnet's hyperparameters and picks the path the subnet runs — plaintext when commit-reveal
        # is off, a drand-timelocked commit the chain reveals itself when it is on. Both testnet 526
        # and mainnet 99 run commit-reveal (measured 2026-09-08: the raw call was refused with
        # 'Attempting to call set_weights when commit/reveal is enabled'), so the raw call could
        # never have landed on either. The u16 quantisation `_u16_weights` pins is the intent's too.
        uids = sorted(slate)
        result = self.subtensor.execute(
            bt.SetWeights(netuid=self.netuid, uids=uids, weights=[slate[u] for u in uids],
                          version_key=version_key),
            self.wallet, wait_for_inclusion=True, wait_for_finalization=False)
        if not getattr(result, "success", False) and (force or not self._weights_are_current(slate)):
            raise ChainError(f"set_weights was not included: "
                             f"{getattr(result, 'message', '') or getattr(result, 'error', None) or result!r}")

    def weights_rate_limit(self) -> int:
        return read_weights_rate_limit(self._storage, self.netuid)

    def blocks_since_weight_update(self) -> int | None:
        uid = self._uid()
        return None if uid is None else self._blocks_since_update(uid)

    def immunity_period(self) -> int:
        return read_immunity_period(self._storage, self.netuid)

    def close(self) -> None:
        self.subtensor.close()
