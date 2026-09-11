"""The owner-run validator daemon — the mechanism with its seams attached (§8, §8b; M9).

`simulate.py` is this mechanism with mock I/O: it encodes the phase order and every rule the
mechanism depends on, and its green run is what proves them. This module is the same engine with the
real seams — the chain (`v3/chain.py`), the store and the one-shot mailbox (`v3/store.py`,
`v3/access.py`), the owner's meter (`v3/gateway.py`) and a served Conductor (`v3/serve.py`) — plus
the three failure modes that only exist once there is real money, a real chain and a real crash.

**THE MECHANISM IS IMPORTED, NOT RE-DERIVED.** Every helper whose divergence would move a crown
comes from `simulate`: `priced` (§5.1b's adapter, which *calls* `score_arm` rather than re-typing the
formula), `_pairs` (the benchmarks §5.1b admits), `_GraderGuard` (§8b.3), `_why` (the sentence a
refused miner reads), the five status words and `WindowReport`/`format_window` (the reveal). Two
copies of any of those would let the offline simulation and the live daemon disagree about what a
window means, with each half green against its own tests — the drift this repository has already paid
for twice (`koth/reference.py`, the `r2ready:v1` encoding). What is written here is I/O, durability,
and the rules below.

THE THREE RULES THAT ONLY EXIST ONCE THE SEAMS ARE REAL

1. **§8b.5, mid-duel restart.** `Scaffold.run_window` owns the arm loop offline, which is why
   `simulate`'s docstring records that the per-duel clock and restart "belong to the daemon that
   replaces this loop's `_arm`". `_arm` below is that replacement: it drives `Scaffold.run_episode`
   task by task and checkpoints each result **before** the next task starts, so a crash re-spends at
   most the task that was in flight. Resumption is exact because `task_order` is a hash of
   `(nonce, task_id)` — a subset keeps its relative order — and because the allowance is threaded
   forward from the checkpointed spend rather than restarted.

2. **§8b.2's last paragraph: a broken KING arm decides every duel at once.** Under §5.2a one king arm
   is shared by every duel in the window, so if it hit the per-duel wall clock — or if its spend
   cannot be reconciled against gateway receipts — the zeroed or unpriceable tail is shared by all of
   them, and the owner's own slowness would settle every verdict. Such a window is treated exactly as
   the power gate treats a dead one: no verdicts, no shots spent, everyone rolls over.

3. **The queue is a chain read, not a list somebody handed us.** `CommitmentOf` survives forever, so
   every miner ever judged is still committed and still visible. Judged hotkeys are therefore
   filtered out **before** `MAX_QUEUE_DEPTH`, or the cap would turn away live challengers on behalf
   of miners settled months ago — §8b.1's deregistration trap sprung by the mechanism built to
   prevent it.

WHICH MACHINE THIS PROCESS RUNS ON IS PART OF THE MECHANISM, NOT AN OPS DETAIL (§8b.9). It holds the
chain hotkey, the R2 credentials and the gateway key, AND it executes programs a miner's model wrote
— through the graders and, since the read channel, through a container per read. Those two must not
be the same machine, and only one of them can be moved by configuration: `--sandbox-host` moves the
DAEMON that starts the containers, never this process. So the deployed arrangement is this process on
the Orchestrator with the credentials, its containers on the sandbox host, and a payload directory
both machines see at the identical absolute path — the constraint the sandbox record §1.3 works
through, and the one thing `main` refuses to start without.

WHAT IS DELIBERATELY ABSENT: any decay, heartbeat or default slate. If the validator is down for a
window, weights are left untouched and the previous schedule persists (§8b.6) — the documented cost
of the owner-liveness dependency (D1) and the right failure, since decaying weights toward nothing
would punish a king for the owner's outage. The only way to express "we do nothing" in code is to
have nothing here that could do something, so the only write is `_write_weights`, and nothing calls
it on a schedule of its own.

**THIS MODULE IS NEVER RUN AGAINST A LIVE NETWORK BY ITS TESTS.** `thirtyspokes-validator` builds
real seams from real credentials; the tests drive the identical `Validator` over `chain.MockChain`, a
`store.S3Bucket` on an in-memory client, and an `OwnerGateway` over a scripted provider — the
production classes with only the far side replaced.
"""

from __future__ import annotations

import math

import concurrent.futures
import threading

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..gateway import signing
from ..koth import holdout_feed
from .access import AccessError, Mailbox, Registration
from .admission import Reference, admit
from .benchmarks.base import check_protocol
from .chain import Chain, Commitment, Metagraph, WeightRateLimited, check_weight_cadence
from .conductor import Conductor
from .config import (
    RETEST_COST_TOLERANCE,
    RETEST_MAX_KEYS,
    RETEST_MAX_USD,
    DUEL_WALL_CLOCK_REASON, DUEL_WALL_CLOCK_SECONDS, EPISODE_CONCURRENCY, EPS,
                     EXCLUDED_REASON, FUTILE_REASON, MAX_DUELS_PER_WINDOW, MAX_QUEUE_DEPTH, MIN_SLICE_REACHED)
from .duel import Contender, champion, duel
from .emissions import KING0, Lineage, emission_weights
from .funding import KEY_NAME, SealedKey, open_key
from .gateway import (KEY_REFUSALS, Audit, GatewayWorker, OwnerGateway, Provider, allowance_journal,
                      audit)
from .reference import ReferenceArm, power_gate, subsample
# `_unreached` is imported rather than re-typed for `chain.py`'s reason for importing
# `_decode_raw_commitment`: it is the exact row a task the arm never reached produces, and §5.8 reads
# its `stopped_reason` to tell a starved allowance from an overrunning validator. A second
# constructor here would be a second definition of that row, and the two would drift silently.
from .scaffold import OutcomeTable, Scaffold, _unreached, task_order
from .score import ArmScore, best_possible_final, score_arm
from .serve import DEFAULT_PREAMBLE, ServedConductor, ServeError, remote_serving
from .simulate import (DEFERRED, DUELLED, REFUSED, SKIPPED, UNQUEUED, Outcome, Pins, WindowReport,
                       _exhausted, _GraderGuard, _pairs, _why, format_window, priced)
from .store import (GRACE_SECONDS, Retention, S3Bucket, apply_retention, fetch_manifest,
                    fetch_submission, promote_submission, retention_plan)
from .types import Action, EpisodeResult, Observation, StepRecord, ToolCall
from .window import Window, WindowError, exclude, schedule, window_path
from .worker import WorkerError
from .window import build as build_window
from .window import load as load_window

__all__ = ["Cadence", "Checkpoint", "Crown", "check_chain_launch", "History", "Owner", "Reveal", "Validator",
           "ValidatorError", "WindowUnavailable", "chain_beacon", "check_launch", "format_reveal",
           "local_serving", "main", "reveal_path", "served_name"]

# Yuma stops counting a validator for being too stable: `koth/neuron.py` records netuid 99 scoring
# 20+ consecutive epochs correctly while `last_update` aged 2234 blocks against an `activity_cutoff`
# of 5000. A 24-hour window is 7200 blocks, so ONE weight write per window ages out of consensus by
# design. The fix is KOTH's: re-submit the unchanged slate on a cadence the cutoff cannot outrun.
# It must exceed the chain's `weights_rate_limit` or every second refresh is rejected (`_preflight`).
WEIGHT_REFRESH_BLOCKS = 180

# How often the daemon looks at the chain. Windows are hours long, so this only bounds how late a
# window opens; polling faster costs RPC and buys nothing.
POLL_SECONDS = 60.0

STATE_VERSION = 1

# A winner whose copy to the public models bucket will not verify is retried on later polls rather
# than crowned or dropped at the first failure — the budget teutonic's promotion worker uses.
# Exhausted, the crown is FORFEITED: a king whose weights are not public breaks D14's promise that
# the crown is derivable, and holding the throne open indefinitely would stall every later duel.
PROMOTION_MAX_ATTEMPTS = 8
# ...and spaced as teutonic spaces them: 30 s after the first failure, doubling per attempt (the
# exponent capped at 8), so the eight attempts span a little over an hour. Retrying on every poll
# instead would spend the whole budget inside a few minutes of an R2 blip.
PROMOTION_RETRY_BASE_SECONDS = 30.0

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class ValidatorError(Exception):
    """The daemon cannot proceed. Never a repair: a hotkey has one shot (§7)."""


class WindowUnavailable(ValidatorError):
    """M7 exit 3, fail closed. The window file is missing, unauthentic, or not the one committed.

    Raised rather than returned because there is exactly one correct response and it is the same for
    every cause: skip the window. No duels, no shot spent, challengers roll over with their one
    submission intact. Scoring against a stale or substituted window would spend a miner's single
    shot on a measurement nobody committed to.
    """


# --- seams ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Owner:
    """The owner's identity on both sides of a published record.

    `verify_sig(data, sig, ss58) -> bool` authenticates the window file (`koth/reference.py`'s
    `open_envelope`), `sign(data) -> hex` signs the reveal. They are separate callables because in
    production they are separate keys' worth of trust: verification needs only the public address,
    while signing needs the wallet — and a validator that could not verify without being able to sign
    would be unable to check its own owner.
    """

    ss58: str
    sign: Callable[[bytes], str]
    verify_sig: Callable[[bytes, str, str], bool]


@dataclass(frozen=True)
class Cadence:
    """Where a window sits in block time, and the chain-side obligation §8b.1 puts on the owner.

    `immunity_blocks` is what the owner DECLARES at `--immunity-blocks`, and `preflight` refuses
    unless the subnet's own `immunity_period` agrees with it. Declaring it keeps the owner stating
    the intent the queue cap is derived from; checking it against the chain is what stops that
    statement being the only thing checked. §8b.1: the queue drains from the back in `MAX_QUEUE_DEPTH /
    MAX_DUELS_PER_WINDOW` windows, plus one to wait for the next window to open, so immunity must
    cover three windows or a challenger is deregistered before it is ever evaluated — having paid a
    registration burn for nothing.
    """

    genesis_block: int
    window_blocks: int
    windows: tuple[int, ...]
    immunity_blocks: int

    def window_at(self, block: int) -> int | None:
        """Which committed window this block falls in, or None once the schedule is exhausted.

        None rather than an unbounded index: the schedule is what the chained manifest pins (§6.3),
        so a window past its end has no committed entry and running one would score against
        parameters nobody committed to.
        """
        if block < self.genesis_block:
            return None
        window = (block - self.genesis_block) // self.window_blocks + 1
        return window if window in set(self.windows) else None

    def opens_at(self, window: int) -> int:
        """The block window N opens at — where its nonce comes from (`chain_beacon`)."""
        return self.genesis_block + (int(window) - 1) * self.window_blocks


def check_launch(*, window_blocks: int, immunity_blocks: int, rate_limit: int) -> None:
    """The two chain-side gates, as arithmetic, so they can be checked before anything is wired.

    Both fail CLOSED and both are launch-time rather than per-write, because their failure mode is
    silence:

    * a window no longer than `weights_rate_limit` means the update is rejected and the PREVIOUS
      schedule quietly keeps paying (§8b.6) — nothing in the reveal says the slate on chain is not
      the one that was computed. `check_weight_cadence` owns that comparison, including the
      off-by-one against the writer's own guard;
    * `WEIGHT_REFRESH_BLOCKS` below the same limit would have every second forced refresh rejected,
      and the refresh is what stops a stable reign ageing out of Yuma's `activity_cutoff`;
    * an `immunity_period` shorter than the queue's drain time deregisters challengers before they
      are ever evaluated, having paid a registration burn for nothing (§8b.1). The requirement is
      derived from the caps rather than written down as a headcount, so it stays correct when the
      window length moves: `MAX_QUEUE_DEPTH / MAX_DUELS_PER_WINDOW` windows to drain, plus one to
      wait for the next window to open.
    """
    check_weight_cadence(window_blocks, rate_limit)
    if WEIGHT_REFRESH_BLOCKS <= rate_limit:
        raise ValidatorError(
            f"WEIGHT_REFRESH_BLOCKS={WEIGHT_REFRESH_BLOCKS} is not above the chain's "
            f"weights_rate_limit of {rate_limit}: every second forced refresh would be rejected, "
            f"and the refresh exists to stop a stable reign ageing out of activity_cutoff")
    windows_to_drain = -(-MAX_QUEUE_DEPTH // MAX_DUELS_PER_WINDOW)          # ceiling division
    required = (windows_to_drain + 1) * window_blocks
    if immunity_blocks < required:
        raise ValidatorError(
            f"immunity_period={immunity_blocks} blocks does not cover {windows_to_drain + 1} "
            f"windows ({required} blocks): a challenger at the back of a "
            f"MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH} queue would be deregistered before it is ever "
            f"evaluated, having paid a registration burn for nothing (§8b.1)")


def check_chain_launch(chain: Chain, cadence: Cadence) -> None:
    """`check_launch`, against what the SUBNET reports rather than what the owner typed.

    ONE FUNCTION BECAUSE THERE ARE TWO CALLERS AND THEY MUST NOT DIVERGE. `--check` is documented as
    the pre-launch gate and is the one mode safe to point at a live chain, so an owner runs it,
    reads "launch gates pass", and starts the daemon. When the cross-check lived only in `preflight`
    that pair disagreed: `--check` certified a subnet whose `immunity_period` was a third of what the
    flag claimed, and the daemon then refused to start on the same configuration.

    `--immunity-blocks` states what the owner BELIEVES they configured; the subnet states what they
    actually did. A gate that reads the belief certifies a claim rather than a fact, and the one
    mistake it most needs to catch — typing three windows and setting one — is exactly the one it
    cannot see.
    """
    on_chain = chain.immunity_period()
    # LESS than declared is the dangerous direction and the only one refused. Exact equality also
    # rejected a subnet configured MORE generously than the flag claims — a strictly safer setting,
    # refused with a message implying the owner had got it wrong. What §8b.1 needs is that
    # challengers actually get at least the drain time the cap is derived from; a longer immunity
    # gives them more, and `check_launch` below is asked about the real number either way.
    if on_chain < cadence.immunity_blocks:
        raise ValidatorError(
            f"--immunity-blocks says {cadence.immunity_blocks} but netuid's immunity_period is only "
            f"{on_chain}. The queue cap is derived from this number (§8b.1), so a subnet less "
            f"generous than the flag claims deregisters challengers this daemon believes are safe — "
            f"raise the subnet's immunity_period, or correct the flag to match it.")
    check_launch(window_blocks=cadence.window_blocks, immunity_blocks=on_chain,
                 rate_limit=chain.weights_rate_limit())


def chain_beacon(substrate, cadence: Cadence) -> Callable[[int], str]:
    """The window nonce: the hash of the block the window opens at.

    WHOEVER PICKS THE NONCE PICKS THE SLICE (§6.3, and `window.verify` says so in as many words), so
    it has to come from the chain rather than from the owner. The opening block's hash is the
    cheapest value with that property: it is fixed before the window exists, nobody chooses it, and
    any reader can re-derive it from `(genesis_block, window_blocks, window)` and one RPC — which is
    what makes the published slice checkable by a third party rather than merely signed.

    Duck-typed over the substrate object for `chain.read_metagraph`'s reason: the decode path is the
    part that breaks on a live chain, so it is exercised against a stub with no chain, no wallet and
    no network.
    """
    def beacon(window: int) -> str:
        block_hash = substrate.get_block_hash(cadence.opens_at(window))
        if not block_hash:
            raise ValidatorError(
                f"the chain returned no block hash for {cadence.opens_at(window)}, so window "
                f"{window}'s nonce cannot be derived; refusing rather than choosing one")
        return str(block_hash)
    return beacon


# `serve(model_dir, served_model) -> Conductor` — the serving stack, and its CONTRACT IS THE §8b.2
# LOAD CHECK: it must raise if this tree cannot be served, because the daemon calls it for the whole
# queue before the king's arm so that a broken challenger never costs the king money. An
# implementation that defers the real load to the first `act()` satisfies the type and breaks the
# rule.
Serve = Callable[[Path, str], Conductor]


def served_name(commitment: Commitment) -> str:
    """The name the artifact must be served under: hotkey and the digest the chain committed to.

    `serve.reference_served_name` records why this exists at all — an OpenAI-compatible server
    reports the name the operator gave it and nothing about the weights it loaded, so
    `--served-model-name` is the only channel the artifact's identity has onto the wire, and
    `ServedConductor` re-checks it on every call. It catches operator error (a stale server on the
    port, the previous challenger still resident); adversarial identity is `admission.admit` over the
    tree on disk and `store.fetch_submission` over its bytes, never this.
    """
    return f"v3-{commitment.hotkey}@{commitment.ready.manifest_sha256}"


def local_serving(base_url: str, *, timeout: float = 180.0) -> Serve:
    """A `Serve` over servers the operator has already brought up (`serve.launch_command`).

    `model_dir` is unused here ON PURPOSE and that is the honest shape of the seam: this daemon does
    not own the cards. The operator launches the model with the helper — which is the only path that
    puts the pinned revision into `--served-model-name` — and this adapter refuses unless an
    endpoint says it is serving exactly that artifact. `check_ready` therefore IS the load check
    §8b.2 asks for, provided the operator's launcher blocks until the engine has loaded, which an
    OpenAI-compatible server does by not routing `/v1/models` until it has.

    SEVERAL SERVERS, COMMA-SEPARATED. `serve.launch_command`'s own arrangement is two cards, two
    servers — the king on one, a challenger on the other — and a name is served by exactly one of
    them, so the first endpoint that lists it is the one used. With one URL every failure is
    passed through untouched (the daemon records it as the refusal it is); with several, the
    refusal names every endpoint asked. Measured 2026-09-08 on testnet 526: a third challenger
    with different weights needed its own server, and one URL could not name it.
    """
    urls = [url.strip() for url in base_url.split(",") if url.strip()]

    def serve(model_dir: Path, served_model: str) -> Conductor:
        refusals: list[BaseException] = []
        for url in urls:
            conductor = ServedConductor(base_url=url, served_model=served_model, timeout=timeout)
            try:
                conductor.check_ready()
                return conductor
            except Exception as exc:          # noqa: BLE001 — every endpoint is asked before refusing
                refusals.append(exc)
        if len(refusals) == 1:
            raise refusals[0]
        raise ServeError(f"no endpoint serves {served_model!r}: "
                         + "; ".join(str(exc) for exc in refusals))
    return serve


# --- what one window published (D15) ---------------------------------------------------------------


def _absent(exc: BaseException) -> bool:
    """Whether a store read failed because the key is not there — as opposed to a link, a
    credential or a decode failure, none of which licenses writing over what may be there."""
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    response = getattr(exc, "response", None) or {}
    return str((response.get("Error") or {}).get("Code", "")) in {"NoSuchKey", "NotFound", "404"}


def reveal_path(window: int) -> str:
    """Where window N's reveal lives — a pure function of the window, like `window.window_path`, so
    a reader addresses it without being told anything and authenticity comes from the owner's
    signature rather than from where the bytes were found."""
    return f"v3/reveal/{int(window)}.json"


LATEST_PATH = "v3/latest.json"
"""The newest settled window, so a reader finds the reveals without guessing.

`reveal_path` is addressable but not DISCOVERABLE: a dashboard holding only the bucket has no way
to learn which windows exist short of probing `1.json`, `2.json`, … until one 404s, which is N
requests to render one page and races a window that settles mid-probe. The bucket cannot be listed
(public read is object-scoped), so the pointer has to be published rather than derived.

Deliberately NOT signed and deliberately carrying no verdict: it is a hint about where to look, and
every claim a reader acts on still comes from the signed reveal it points at. That keeps this file
outside the trust boundary — the worst a tampered pointer can do is name a window whose reveal then
fails its own signature check.
"""


@dataclass(frozen=True)
class ArmAudit:
    """One arm's metering, reconciled (M5 exit 1) — and the two numbers §5.8 will not let hide.

    `unfunded_calls` and `exhausted` are different facts and both are published. A task the arm never
    reached carries `budget_exhausted` and is visible in the trace; a call the gateway REFUSED
    because the balance ran out mid-episode is a dead rung that looks exactly like a flaky provider,
    and this counter is the only place that difference exists (`gateway.DuelLedger`).

    `resumed` is how many of this arm's tasks came back from a checkpoint rather than being run
    (§8b.5). It is published because the audit below covers only what THIS process metered: the
    receipts for a resumed prefix belong to the process that spent them.
    """

    arm: str
    duel_id: str
    payer: str
    audit: Audit
    unfunded_calls: int
    exhausted: int
    resumed: int
    # §5.4's futility abort, counted beside the allowance's. Both fill unreached tasks with zero,
    # and §5.8 requires the reveal to say which — "ran out of money" and "could no longer win" are
    # different findings about a miner, and one number for both would publish them as one.
    futile: int = 0
    # §5.2c: how many of this arm's delegates the window's outcome table answered (charged, not
    # bought) and how many it bought live. With `audit.charged_usd` and `audit.provider_usd` these
    # are what let a reader see that the miner paid for every delegate while the provider was paid
    # for the fills alone.
    hits: int = 0
    fills: int = 0
    # D18: whether the arm ran on the miner's own registered key, which endpoint served each model
    # it bought live (`provider`, or `provider:served_model` when the two names differ), and the
    # §11-1 evidence — the `model@provider` pairs this arm was served by that no arm on the OWNER's
    # key was served by this window (`_endpoint_audit`). Published; gates nothing.
    own_key: bool = False
    endpoints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    drift: tuple[str, ...] = ()

    @property
    def scoreable(self) -> bool:
        return self.audit.scoreable


@dataclass(frozen=True)
class Reveal:
    """One window's published record: the sim's own report, plus what the seams metered.

    `report` is `simulate.WindowReport` unchanged, so the reveal the daemon publishes is byte-for-
    byte the reveal the offline simulation prints. The metering lives beside it rather than inside it
    because it is a property of the seams and not of the mechanism — `simulate` has no gateway to
    reconcile — and folding it in would have meant a second report type that the simulation could not
    produce and therefore could not test.
    """

    report: WindowReport
    arms: tuple[ArmAudit, ...]
    weights_written: bool
    # §5.2c's two diagnostics, published beside the metering and gating nothing in v1: the
    # test-retest sample (a few hit keys re-bought live, and how often the fresh draw disagreed)
    # and every arm's quality/final on the slice's seen and unseen rows.
    retest: dict | None = None
    splits: dict | None = None

    @property
    def window(self) -> int:
        return self.report.window


def format_reveal(reveal: Reveal) -> str:
    """The reveal as text: the mechanism's own report, then what the money did."""
    return "\n".join([format_window(reveal.report), "", *_metering_section(reveal)])


def _metering_section(reveal: Reveal) -> list[str]:
    lines = ["# METERING — every dollar in `final` came out of a gateway receipt (M5 exit 1)",
             f"  gateway weights written to chain: {reveal.weights_written}"]
    for arm in reveal.arms:
        lines.append(
            f"  {arm.arm:<20} {arm.audit.reason:<24} charged ${arm.audit.charged_usd:.6f} "
            f"recorded ${arm.audit.recorded_usd:.6f} provider ${arm.audit.provider_usd:.6f} "
            f"calls {arm.audit.calls} (hits {arm.hits}, fills {arm.fills}; §5.2c)")
        lines.append(
            f"    unfunded calls {arm.unfunded_calls} (gateway refused, §5.8)  "
            f"tasks the allowance never reached {arm.exhausted}  "
            f"resumed from a checkpoint {arm.resumed} (§8b.5)")
        if arm.own_key:
            lines.append(f"    on the miner's own key (D18); endpoints not seen on the owner's key: "
                         f"{', '.join(arm.drift) if arm.drift else 'none'}")
    if reveal.retest is not None:
        r = reveal.retest
        lines.append(f"  test-retest (§5.2c): {r['keys']} hit keys re-bought for ${r['usd']:.6f}; "
                     f"grade disagreed on {r['grade_disagreements']}, cost on "
                     f"{r['cost_disagreements']} (diagnostic)")
    return lines


# --- durable state ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Crown:
    """Who reigns, and what has to be served to make it defend (§5.2a, D6).

    `model_dir` and `served_model` are stored because the king is re-evaluated EVERY window and the
    daemon must be able to bring its artifact back up after a restart without re-deriving where it
    came from. §8b.7 keeps the reigning king's weights indefinitely for exactly this reason.
    `KING0` carries neither: it is a policy in the owner's scaffold, not a model (§8b.8).
    """

    hotkey: str = KING0
    model_dir: str = ""
    served_model: str = ""
    # Where D14 makes this king's weights public: `models/sha256/<manifest>/` in the public models
    # bucket, set only once that copy has verified. Empty for King0, which has no weights.
    public_prefix: str = ""

    @property
    def is_king_zero(self) -> bool:
        return self.hotkey == KING0


class History:
    """Everything that must survive a restart, and nothing else.

    All of it is append-only: which hotkeys have been judged (§7's one shot), who has ever been
    crowned (the pension's only input), who reigns, which tasks have been scored before (D12's
    published staleness), and which windows are settled. Everything else about a window is derived
    from these, which is what lets §8 step 5 persist the cheap half and re-derive the rest.

    Written through a temporary and `replace`d — `koth/arena_neuron.py`'s (removed with v2, 2026-09-07) pattern — because a torn
    state file here loses the record that a shot was spent, and re-judging is not idempotent under
    the one-shot rule.

    `windows[n]["published"]` is the flag that makes §8's order survive a crash *between* its steps:
    history is persisted first, weights second, the reveal last, so a crash after the weights leaves
    a settled window whose reveal was never published — and D15's entry ramp would lose a window's
    traces for good. The reveal is written locally as part of persisting, so the restart re-does only
    the publish.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.judged: set[str] = set()
        self.coronations: list[str] = []
        self.crown = Crown()
        self.revealed: set[str] = set()
        self.windows: dict[int, dict] = {}
        # A verdict that won but whose public copy has not verified yet (`Validator._promote`).
        self.pending_crown: dict | None = None
        # hotkey -> {registration_id, at}: when each submission was judged, which is where §8b.7's
        # grace window starts, and which prefix its weights live under.
        self.judged_at: dict[str, dict] = {}
        # registration ids whose weights retention has already deleted.
        self.purged: set[str] = set()
        self._restore()

    # --- reads ------------------------------------------------------------------------------
    def completed(self, window: int) -> bool:
        """Settled: this window's verdicts are recorded and must never be recomputed."""
        return window in self.windows

    def published(self, window: int) -> bool:
        return bool(self.windows.get(window, {}).get("published"))

    def lineage(self) -> Lineage:
        return Lineage.from_coronations(self.coronations)

    # --- writes -----------------------------------------------------------------------------
    def settle(self, window: int, *, crowned: str | None, judged: Iterable[str],
               revealed: Iterable[str], registrations: Mapping[str, str] | None = None,
               at: float | None = None) -> None:
        """§8 step 5's first half — the half a crash must never repeat.

        `registrations` maps a judged hotkey to the registration its submission lives under and `at`
        is the wall-clock moment of judgment; together they are what §8b.7's retention counts from.
        A hotkey keeps its FIRST judgment time.
        """
        judged = list(judged)
        for hotkey in judged:
            if registrations and at is not None and hotkey in registrations:
                self.judged_at.setdefault(hotkey, {"registration_id": registrations[hotkey],
                                                   "at": float(at)})
        self.judged.update(judged)
        self.revealed.update(revealed)
        if crowned is not None:
            self.coronations.append(crowned)
        self.windows[window] = {"crowned": crowned, "published": False}
        self.save()

    def mark_published(self, window: int) -> None:
        self.windows.setdefault(window, {"crowned": None})["published"] = True
        self.save()

    def crown_to(self, crown: Crown) -> None:
        self.crown = crown
        self.save()

    def defer_crown(self, pending: Mapping) -> None:
        """A verdict won but its public copy has not verified: remember the winner, do not crown."""
        self.pending_crown = dict(pending)
        self.save()

    def crown_promoted(self, crown: Crown, *, window: int) -> None:
        """Finish a deferred coronation once its copy verified on a later poll."""
        self.crown = crown
        self.coronations.append(crown.hotkey)
        self.windows.setdefault(window, {"crowned": None})["crowned"] = crown.hotkey
        self.pending_crown = None
        self.save()

    def forfeit_crown(self) -> None:
        self.pending_crown = None
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "version": STATE_VERSION,
            "judged": sorted(self.judged),
            "coronations": list(self.coronations),
            "crown": {"hotkey": self.crown.hotkey, "model_dir": self.crown.model_dir,
                      "served_model": self.crown.served_model,
                      "public_prefix": self.crown.public_prefix},
            "revealed": sorted(self.revealed),
            "windows": {str(w): body for w, body in sorted(self.windows.items())},
            "pending_crown": self.pending_crown,
            "judged_at": {hotkey: dict(entry) for hotkey, entry in sorted(self.judged_at.items())},
            "purged": sorted(self.purged),
        }, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def _restore(self) -> None:
        if not self.path.exists():
            return
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self.judged = set(state.get("judged", ()))
        self.coronations = list(state.get("coronations", ()))
        self.crown = Crown(**state.get("crown", {}))
        self.revealed = set(state.get("revealed", ()))
        self.windows = {int(w): dict(body) for w, body in state.get("windows", {}).items()}
        self.pending_crown = state.get("pending_crown")
        self.judged_at = {str(k): dict(v) for k, v in state.get("judged_at", {}).items()}
        self.purged = set(state.get("purged", ()))


class Checkpoint:
    """§8b.5: per-task results, written as they land, so a restart never re-spends what was spent.

    One append-only JSONL per `(window, arm)`. Append rather than an atomic rewrite because the file
    grows to ~250 rows over two hours and rewriting it per task would make the checkpoint cost grow
    with the arm; a torn final line is dropped on read, which costs at most the one task that was in
    flight — the same bound the allowance already overruns by (`Scaffold.run_window`), and the same
    bound `OwnerGateway.call` documents.

    It holds RESULTS, not episodes: `Episode.requests` is the §2.1 audit log and belongs to the arm
    that made the calls, and a resumed arm cannot honestly claim to have sent requests it did not.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, window: int, arm: str) -> Path:
        return self.root / f"window-{int(window)}" / f"{_UNSAFE.sub('_', arm)}.jsonl"

    def resume(self, window: int, arm: str) -> tuple[EpisodeResult, ...]:
        path = self.path(window, arm)
        if not path.exists():
            return ()
        results = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                results.append(episode_from_json(json.loads(line)))
            except (ValueError, KeyError, TypeError):
                break                      # a torn tail is the process that died, not a bad record
        return tuple(results)

    def append(self, window: int, arm: str, result: EpisodeResult) -> None:
        path = self.path(window, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_json(result), sort_keys=True) + "\n")
            handle.flush()

    def outcomes(self, window: int) -> Path:
        """Where the window's outcome table lives (§5.2c) — one JSONL beside the arms' files, so a
        restart re-buys no row (§8b.5) and one window's state is one directory."""
        return self.root / f"window-{int(window)}" / "outcomes.jsonl"

    def exclusions(self, window: int) -> set[str]:
        """§6.3c's excluded tasks, recovered after a restart.

        Per WINDOW, not per arm, because that is the scope of the rule: a grader that dies during
        the last arm still drops its task from the first duel's denominator. Kept beside the arms'
        own files so one window's state is one directory.
        """
        path = self.root / f"window-{int(window)}" / "excluded.jsonl"
        if not path.exists():
            return set()
        raw = path.read_bytes()
        chunks = raw.split(b"\n")
        excluded = set()
        for index, chunk in enumerate(chunks):
            line = chunk.decode("utf-8", "replace")
            if not line.strip():
                continue
            try:
                excluded.add(str(json.loads(line)["task_id"]))
            except (ValueError, KeyError, TypeError) as exc:
                # A torn LAST row is the process that died mid-write. A damaged row with rows after
                # it is not, and stopping at it would silently drop every later exclusion —
                # restoring, quietly, the exact bug this file exists to prevent: a dead grader's task
                # scored as a miner's miss. §6.3c is not a best-effort rule.
                if index == len(chunks) - 1:
                    break
                raise ValidatorError(
                    f"the exclusion record {path} is unreadable at line {index + 1} ({exc}); it "
                    f"names the tasks §6.3c drops from every arm, so a partial read would score a "
                    f"dead grader as a miner's miss") from exc
        return excluded

    def exclude(self, window: int, task_id: str) -> None:
        path = self.root / f"window-{int(window)}" / "excluded.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id}) + "\n")
            handle.flush()

    def snapshot(self, window: int, arm: str, budget_usd: float) -> float:
        """§4's allowance, read at ONE definite instant and REMEMBERED across a restart.

        "Snapshotted at window open and frozen for that window's duels" has to survive the restart or
        it is not a snapshot. Re-reading the balance after a crash would be wrong in both directions
        and the direction depends on a property of the meter nobody should have to know: against a
        durable gateway the balance is already debited, so subtracting the checkpointed spend a
        second time would starve the arm; against a volatile one it is not, so not subtracting would
        hand the arm its money back. Remembering the number removes the question.
        """
        path = self.path(window, arm).with_suffix(".budget")
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(float(budget_usd)), encoding="utf-8")
        return float(budget_usd)


# --- traces (D15) and the checkpoint share one encoding ---------------------------------------------
# One encoding, because they carry the same thing for two reasons: the checkpoint has to reconstruct
# an arm exactly, and D15 publishes the arm so miners can train on it. Two encodings would mean the
# traces a miner learns from and the results a restart resumes could describe different runs.


def episode_json(result: EpisodeResult) -> dict:
    return {"task_id": result.task_id, "benchmark": result.benchmark,
            "graded_score": result.graded_score, "spend_usd": result.spend_usd,
            "stopped_reason": result.stopped_reason,
            "steps": [{"step_index": s.step_index,
                       "rendered_state_digest": s.rendered_state_digest,
                       "raw_output": s.raw_output,
                       "action": None if s.action is None else
                                 {"kind": s.action.kind, "model_id": s.action.model_id},
                       "model_id": s.model_id, "success": s.success, "cost_usd": s.cost_usd,
                       "parse_failed": s.parse_failed, "failure": s.failure,
                       # §2.1's audit input. With it the published record recomputes the exact bytes
                       # every worker was sent — `tools.compose(task.prompt, observations[:k])` for
                       # the k-th turn of the delegate — and therefore the `prompt_hash` of every
                       # gateway receipt the step paid for. Without it a delegate that read is an
                       # opaque row and the sentence becomes an assertion (`types.StepRecord`).
                       "observations": [{"call": {"name": o.call.name, "arg": o.call.arg,
                                                  "start": o.call.start},
                                         "response_sha256": o.response_sha256,
                                         "output": o.output, "truncated": o.truncated}
                                        for o in s.observations]}
                      for s in result.steps]}


def episode_from_json(body: Mapping) -> EpisodeResult:
    return EpisodeResult(
        task_id=body["task_id"], benchmark=body["benchmark"],
        graded_score=float(body["graded_score"]), spend_usd=float(body["spend_usd"]),
        stopped_reason=body["stopped_reason"],
        steps=tuple(StepRecord(
            step_index=int(s["step_index"]), rendered_state_digest=s["rendered_state_digest"],
            raw_output=s["raw_output"],
            action=None if s["action"] is None else Action(s["action"]["kind"],
                                                           s["action"]["model_id"]),
            model_id=s["model_id"], success=bool(s["success"]), cost_usd=float(s["cost_usd"]),
            parse_failed=bool(s["parse_failed"]), failure=s.get("failure"),
            # Absent on a record written before the read channel existed, which is every checkpoint
            # of every benchmark that declares no tools — and `()` is what such a step held anyway.
            observations=tuple(
                Observation(call=ToolCall(name=o["call"]["name"], arg=o["call"]["arg"],
                                          start=int(o["call"]["start"])),
                            response_sha256=o["response_sha256"], output=o["output"],
                            truncated=bool(o["truncated"]))
                for o in s.get("observations", ()))) for s in body["steps"]))


# --- the queue ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Queued:
    """One challenger, resolved: what the chain says and where its bytes are.

    The `Registration` is derived from a LIVE metagraph read rather than carried in the commitment
    (`chain.Commitment` has no uid, deliberately), so the R2 prefix this challenger's tree is fetched
    from is a function of chain facts both sides compute independently — which is what stops one
    miner writing into another's submission.
    """

    commitment: Commitment
    registration: Registration

    @property
    def hotkey(self) -> str:
        return self.commitment.hotkey

    @property
    def block(self) -> int:
        return self.commitment.block


# --- the daemon --------------------------------------------------------------------------------------


@dataclass
class Validator:
    """The single owner-run validator (D1). One object, real seams, `simulate`'s phase order.

    `pins` is M3a's published λ/C table plus King₀ and its contrast — measured once, published, and
    never re-derived per window (§5.1b), because it is the exchange rate miners train against.
    `pins.world.worker` is ignored: every worker call in a duel goes through the owner's gateway
    (§11-1), which is the whole point of M5.
    """

    pins: Pins
    reference: Reference
    chain: Chain
    store: S3Bucket
    # Teutonic's layout. `store` stays the PUBLIC store for window files, reveals and envelopes;
    # `private_models` holds every submission and is readable by nobody but the miner's scoped
    # credential and this daemon; `public_models` receives a winner's tree, content-addressed, once
    # it has won — D14's public king and nothing else. Three distinct buckets, checked at preflight.
    private_models: S3Bucket
    public_models: S3Bucket
    mailbox: Mailbox
    gateway: OwnerGateway
    owner: Owner
    cadence: Cadence
    serve: Serve
    beacon: Callable[[int], str]
    netuid: int
    root: Path
    per_benchmark: int
    minimum: int
    # King₀ is a policy in the owner's scaffold rather than a miner's model, so the OWNER pays for
    # its arm and for both reference arms every window (§5.6, §8b.8). A named account rather than an
    # implicit exemption: the owner's spend is metered by the same gateway, so "what did this window
    # cost the owner" is a number rather than an estimate.
    owner_account: str = "owner"
    # D18: the seed of the owner's mailbox key, which opens a miner's sealed OpenRouter key. None
    # leaves every hotkey on the owner's provider and the hand-credited ledger (the harness, and an
    # owner without a mailbox key yet). `provider_for` builds the client an arm runs on from the
    # opened key — the live client by default, a fake in tests.
    key_seed: bytes | None = None
    provider_for: Callable[[str], Provider] | None = None
    burn_uid: int = 0
    # §5.5's default is that King₀'s 0.85 BURNS, and `None` keeps it. An owner who wants the genesis
    # king to appear as a reigning king rather than an absence sets this to the UID it reigns at;
    # the reveal then names that UID's hotkey as `king_hotkey` and pays the crown there. Setting it
    # to `burn_uid` leaves the on-chain slate bit-identical, because that UID was already collecting
    # the same share as burn — it changes what the published record CLAIMS, not what is paid.
    king_zero_uid: int | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    # Wall-clock seconds for anything PERSISTED (judgment times retention counts from). Not `clock`,
    # which is monotonic and means nothing across a restart.
    now: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.history = History(self.root / "history.json")
        self.checkpoint = Checkpoint(self.root / "checkpoints")
        # The WHOLE schedule is committed once (§6.3): the chain gives each hotkey one commitment
        # slot and `set_commitment` overwrites it, so a per-window write would erase the owner's
        # governance record. One chained-manifest root covers every window and each slice is
        # re-derived and compared rather than trusted.
        entries = schedule(self.cadence.windows, tasks=self.pins.world.tasks,
                           per_benchmark=self.per_benchmark, minimum=self.minimum)
        self._entries = {entry["epoch"]: entry for entry in entries}
        self.manifest = holdout_feed.manifest(entries)
        self._last_weight_block: int | None = None
        self._preflighted = False
        # `(window, arm) -> attempts made in THIS process`. Only the gateway's ledger identity needs
        # it; the money it protects is on disk in the checkpoint, which is what makes a retry cheap.
        self._attempts: dict[tuple[int, str], int] = {}

    # --- launch-time gates ------------------------------------------------------------------

    def preflight(self) -> None:
        """`check_launch` against this chain's own reported limit, once. Idempotent."""
        if self._preflighted:
            return
        self._require_private_submissions()
        check_chain_launch(self.chain, self.cadence)
        # §6.3: the schedule root must be ON CHAIN and must be the one this validator computed.
        # The draw is a pure function of a schedule entry (D12), so N entries are N slices; a root
        # fixed before any challenger commits is what removes the owner's choice among them, and it
        # only removes it if somebody else can read it. Checked at launch rather than per window
        # because it covers the whole schedule and a mismatch invalidates every window equally.
        committed = self.chain.schedule_root()
        if committed is None:
            raise ValidatorError(
                "no schedule root is committed on chain under this validator's own hotkey (§6.3). "
                "Publish it with `thirtyspokes-owner commit-schedule` before opening a window, and "
                "pass it the SAME --owner-hotkey as this daemon's --hotkey: the root is read back "
                "from the signing key's own commitment slot, so a root committed under a different "
                "hotkey is indistinguishable here from no root at all. Uncommitted, the owner could "
                "choose a slice after seeing who committed, and no third party can check any window "
                "against anything.")
        if committed != self.manifest["root"]:
            raise ValidatorError(
                f"the committed schedule root does not match this validator's schedule: on chain "
                f"{committed}, computed {self.manifest['root']}. The world, the window count or the "
                f"stratification changed since the commit, so the slices this would score are not "
                f"the ones the chain was told about — re-derive the schedule or re-commit it "
                f"deliberately.")
        # §4's allowance, checked before a window rather than discovered inside one. An owner
        # account at $0 does not fail visibly: every reference episode trips the exhaustion check at
        # its first step, both fixed policies score zero, the power gate reports that the slice
        # "CANNOT separate policies" — and the window is published blaming the corpus for the
        # owner's empty wallet. That is exactly the confusion §5.8 forbids, and it is worse here
        # than in the reveal because it repeats every window, burning emissions, until somebody
        # reads the code.
        self._require_owner_funds()
        self._preflighted = True

    def _require_private_submissions(self) -> None:
        """Refuse a deployment where a submission could be public (D14; WHITEPAPER's exposure line).

        The three buckets are three visibilities, and the mistake that matters is the silent one:
        pointing `private_models` at a public bucket uploads every challenger's weights where anyone
        can download them, and nothing about a window would look wrong.
        """
        names = (self.store.bucket, self.private_models.bucket, self.public_models.bucket)
        if len(set(names)) != len(names):
            raise ValidatorError(
                f"the store, private models and public models buckets must be distinct, got "
                f"{names}: a submission in a public bucket is downloadable by anyone before it has "
                f"won anything")

    def _require_owner_funds(self) -> None:
        """§4's allowance, checked before every window rather than once at launch.

        CHECKED PER WINDOW BECAUSE A PREPAID BALANCE ALWAYS REACHES ZERO. `preflight` returns early
        once it has run, so a check living only there covers the never-funded-at-startup case and
        nothing else — and the interesting case is the other one, which is guaranteed rather than
        hypothetical. Measured: an owner funded for one window drains to -$0.02, and from the next
        window on the daemon publishes CANNOT-separate reveals, defers every challenger and burns
        every emission, with the launch check still green and the reveal blaming the corpus.

        Raised from inside a window so `run_forever` logs it and keeps polling: a drained owner is
        fixed by crediting the account, and a daemon that had to be restarted to notice would turn a
        recoverable outage into an operator's problem.
        """
        # Pick up anything `thirtyspokes-owner credit` appended while this daemon was running,
        # before deciding whether anyone is funded.
        self.gateway.refresh()
        if self.gateway.balance(self.owner_account) <= 0.0:
            raise ValidatorError(
                f"the owner account {self.owner_account!r} has no allowance (§4). The reference "
                f"arms and, while King zero reigns, the king's arm are paid by the owner, so every "
                f"one of them would spend nothing, score zero on every task, and publish a window "
                f"that reads as a corpus with no spread. Credit it with `thirtyspokes-owner "
                f"--state DIR credit --hotkey {self.owner_account} --usd N`, and check the rest "
                f"with `thirtyspokes-owner --state DIR balances`.")

    # --- the loop ---------------------------------------------------------------------------

    def run_forever(self, *, poll_seconds: float = POLL_SECONDS,
                    should_stop: Callable[[], bool] | None = None,
                    on_reveal: Callable[[Reveal], None] | None = None) -> None:
        """Poll the chain, run each window once, keep the slate alive between them.

        Every failure inside `step` is caught and the loop continues, with ONE exception: a
        `ValidatorError` from `preflight` is a misconfiguration that would make every window wrong,
        so it stops the daemon rather than being retried sixty seconds later forever.
        """
        self.preflight()
        while not (should_stop is not None and should_stop()):
            try:
                reveal = self.step()
            except Exception as exc:      # noqa: BLE001 — one bad window must not end the reign
                self._log(f"window step failed, weights untouched: {exc}")
            else:
                if reveal is not None and on_reveal is not None:
                    on_reveal(reveal)
            self.sleep(poll_seconds)

    def step(self) -> Reveal | None:
        """One turn of the loop: run this block's window if it is due, else keep the slate warm.

        Returns None when nothing was scored, which is the ordinary case — a window is hours long and
        the poll is a minute. The three None paths are different and all of them leave the ledger
        alone: the schedule is exhausted, the window is already settled, or the window file could not
        be verified (fail closed, §8 step 2 — challengers roll over with their shot intact).
        """
        self.preflight()
        self._settle_pending_crown()
        self._apply_retention()
        block = self.chain.current_block()
        window = self.cadence.window_at(block)
        if window is None or self.history.completed(window):
            if window is not None and not self.history.published(window):
                # A crash between §8's second and third steps. Re-doing the CHEAP half is the whole
                # reason that order is fixed: the verdicts stand, and only the publish repeats.
                self._republish(window)
            self._refresh_weights(block)
            return None
        try:
            return self.run_window(window)
        except WindowUnavailable as exc:
            self._log(f"window {window}: {exc}")
            self._refresh_weights(block, force=True)
            return None

    # --- one window -------------------------------------------------------------------------

    def run_window(self, window: int) -> Reveal:
        """The eight phases. Every boundary is load-bearing; `simulate`'s docstring says why.

        What is added here, and only here: the queue is read from the chain rather than handed in
        (so judged hotkeys leave it before the depth cap), every arm is metered and reconciled, every
        arm is resumable, and a king arm that cannot be used settles the window as a dead one.
        """
        # Before anything is drawn or spent. A window opened on an empty owner account produces the
        # one failure that reads as somebody else's fault (§5.8), so it is refused rather than run.
        self._require_owner_funds()
        self.preflight()
        # Seeded from disk and written through, so a mid-window restart does not forget which tasks
        # §6.3c already dropped. Without this the guard's own "the 0.0 is never scored" stops being
        # true across a crash: the king's checkpointed rows still carry the placeholder, nothing
        # remembers why, and the owner's infrastructure failure is scored as the miner's miss.
        guard = _GraderGuard(self.pins.world.grade, self.pins.world.inspect,
                             failed=self.checkpoint.exclusions(window),
                             on_exclude=lambda task_id: self.checkpoint.exclude(window, task_id))
        # ONE OUTCOME TABLE FOR THE WINDOW (§5.2c, D17), durable beside the checkpoints: every arm
        # below — the two references, the king, every challenger, in that order — is scored
        # against the same sampled draws, and a restart re-buys none of them (§8b.5).
        table = OutcomeTable(self.checkpoint.outcomes(window))
        metagraph = self.chain.metagraph()

        # PHASE 1 — open and verify the window, or refuse it outright (§6.2, §6.3, M7 exit 3).
        opened = self._open(window)
        # And refuse it for the read channel's third fact too, which no signature or hash covers: a
        # task inviting a read it does not declare has its `READ` line graded as its patch, and one
        # declaring a verb without the protocol never tells the worker it may read. Free, and here
        # because this is the last moment before the slice costs anyone money (`benchmarks.base`).
        check_protocol(opened.tasks)

        # The queue is read here, before anything is spent, because a refused window still has to
        # report who rolled over. §5.5's reversion is resolved on the same metagraph read.
        notes: dict[str, str] = {}              # D18: why a registered key could not fund an arm
        queue, turned_away = self._queue(metagraph, window, notes)
        # Every queued hotkey's registration, for §8b.7's retention: taken from the whole queue
        # here because two of `_publish`'s callers pass it no `queued` mapping at all.
        registrations = {entry.hotkey: entry.registration.registration_id for entry in queue}
        self._resolve_reign(metagraph)
        crown = self.history.crown
        king_hotkey = crown.hotkey            # the START-OF-WINDOW king; every duel faces this one

        # PHASE 2 — the power gate (§5.6). Two FIXED policies over a subsample, on the owner's
        # allowance: a window that cannot separate them cannot rank two competitors either, and a
        # refused window must cost a delay rather than a miner's one shot.
        arms, ref_audits = [], []
        splits: dict[str, dict] = {}
        for policy in (self.pins.king_zero, self.pins.contrast):
            results, meter = self._arm(opened, arm=f"ref-{policy.name}", conductor=policy,
                                       payer=self.owner_account,
                                       budget_usd=self.gateway.balance(self.owner_account),
                                       guard=guard, tasks=subsample(opened.tasks,
                                                                    nonce=opened.nonce),
                                       table=table)
            arms.append(ReferenceArm(policy.name, results))
            ref_audits.append(meter)
            splits[f"ref-{policy.name}"] = _seen_split(results, opened.seen, guard.failed,
                                                       self.pins.world.exchange)
        power = power_gate(arms[0], arms[1], nonce=opened.nonce, failed=guard.failed)

        if not power.separates:
            return self._publish(opened, power, tuple(arms), king_hotkey, None, (), guard,
                                 tuple([Outcome(q.hotkey, q.block, DEFERRED, power.reason)
                                        for q in queue] + turned_away),
                                 None, tuple(ref_audits), {}, table=table, splits=splits,
                                 registrations=registrations)

        # PHASE 3 — admission for the WHOLE queue, before the king's arm is touched (§8b.2). A
        # window whose entire queue is invalid therefore spends the king nothing at all.
        outcomes: list[Outcome] = list(turned_away)
        admitted: list[tuple[Queued, Conductor]] = []
        judged: list[str] = []
        deferred = 0
        for queued in queue:
            if len(admitted) == MAX_DUELS_PER_WINDOW:
                # §8b.1's DUEL cap. Deferred BEFORE `_admit`, so a full window costs the overflow the
                # wait and not a ~72 GB pull, and counted against ADMITTED challengers — deferring a
                # live one because somebody else's upload was broken would spend the clock on
                # nothing. `_wait` is §8b.1's published queue wait.
                outcomes.append(Outcome(
                    queued.hotkey, queued.block, DEFERRED,
                    f"the window is full at MAX_DUELS_PER_WINDOW={MAX_DUELS_PER_WINDOW}; rolls over "
                    f"with its shot unspent, expected wait {self._wait(deferred)} window(s) "
                    f"(§8b.1)"))
                deferred += 1
                continue
            if self.gateway.balance(queued.hotkey) <= 0.0:
                # §4: the arm is paid from the miner's own allowance, so an unfunded challenger
                # makes NO worker calls, scores zero on every task, and spends nothing. That is not
                # a weak router, it is an absent one — and it is not harmless, because `final_b`
                # prices spend: an arm that scores 0 quality at $0 beats a king whose quality does
                # not cover its own bill, so a challenger who funds nothing takes the crown for
                # doing nothing. Measured on the harness, and it was crowned.
                #
                # DEFERRED, not REFUSED: the miner paid a registration burn and uploaded ~72 GB, and
                # the omission is fixable by sending money. Spending the one shot over an empty
                # balance is the "Kind" property broken at its cheapest point. Checked BEFORE
                # `_admit` so it costs no download either, and the queue cap bounds how long an
                # unfunded entry can sit in front of a funded one.
                outcomes.append(Outcome(
                    queued.hotkey, queued.block, DEFERRED, notes.get(queued.hotkey) or (
                        f"no spend allowance (§4): this arm would make no worker calls and score "
                        f"zero on every task. Register an OpenRouter key under this submission "
                        f"(docs/MINER.md §6) or ask the owner to credit {queued.hotkey}; the shot "
                        f"is not spent and the entry is judged in the first window after it is "
                        f"funded")))
                continue
            try:
                admitted.append((queued, self._admit(queued)))
            except Exception as exc:      # noqa: BLE001 — see `_admit`: OOM and load timeouts too
                # An invalid artifact SPENDS THE SHOT: it was judged, and the answer was no (§8b.2).
                # That is not the same as `DEFERRED`, which our own failures produce, and the
                # difference is the whole "Kind" property — so `judged` grows here and not only
                # after a duel.
                judged.append(queued.hotkey)
                outcomes.append(Outcome(queued.hotkey, queued.block, REFUSED, str(exc)))

        # PHASE 4 — the king's arm, ONCE, and only if something is left to duel it (§5.2a, D13).
        king_results: tuple[EpisodeResult, ...] = ()
        king_meter: ArmAudit | None = None
        if admitted and not crown.is_king_zero:
            # D18: a miner king defends on the key it registered, read fresh like a challenger's.
            note = self._fund(crown.hotkey, Path(crown.model_dir).name, opened.epoch)
            if note is not None:
                self._log(f"window {opened.epoch}: the king {crown.hotkey}: {note}")
        if admitted:
            king_results, king_meter = self._arm(
                opened, arm="king", conductor=self._king_conductor(crown),
                payer=self._payer(crown), budget_usd=self.gateway.balance(self._payer(crown)),
                guard=guard, tasks=opened.tasks, table=table)
            splits["king"] = _seen_split(king_results, opened.seen, guard.failed,
                                         self.pins.world.exchange)

        meters = tuple(ref_audits) + ((king_meter,) if king_meter is not None else ())
        broken = self._king_arm_unusable(king_results, king_meter)
        if admitted and broken is not None:
            # §8b.2's last paragraph. Under §5.2a the king's arm is shared by every duel, so a
            # truncated or unreconcilable one decides all of them at once — the owner's own failure
            # settling every verdict in the window. Treated as the power gate treats a dead window.
            outcomes.extend(Outcome(q.hotkey, q.block, DEFERRED, broken) for q, _ in admitted)
            return self._publish(opened, power, tuple(arms), king_hotkey, None, king_results, guard,
                                 tuple(outcomes), None, meters, {}, judged=judged, table=table,
                                 registrations=registrations,
                                 splits=splits)

        # §4: "The king must stay funded ... an unfunded king scores zeroes and loses." The code
        # enforced only the first half. A king whose arm never reaches the slice scores 0 on every
        # task, which puts its `final` at exactly 0.0 — and 0.0 beats any challenger whose priced
        # spend exceeds its quality, so a reigning miner could stop paying and keep 85% of emissions
        # for as long as nobody cleared that bar. Free money, and it contradicts a rule the spec
        # states outright.
        #
        # So the reign ENDS rather than the window being voided. Voiding would let the freeloader
        # keep the throne, which is the opposite of §4. This is §5.5's reversion, for the same reason
        # §5.5 gives: it is not a coronation, nothing is appended to the lineage, and King zero's
        # share burns until a challenger clears the verdict against it. The duels that follow stand —
        # they were run against the arm that actually ran, and a challenger that beats a zeroed king
        # has beaten what defended the throne.
        if admitted and not crown.is_king_zero:
            king_reached = sum(1 for result in king_results
                               if result.stopped_reason != "budget_exhausted")
            if king_reached < MIN_SLICE_REACHED * len(king_results):
                self._log(f"window {opened.epoch}: the king's arm reached {king_reached} of "
                          f"{len(king_results)} tasks before its allowance ran out; §4 ends the "
                          f"reign rather than letting an unfunded king defend for nothing")
                self.history.crown_to(Crown())
                crown = self.history.crown

        # PHASE 5 — every challenger's arm. Splitting arms from verdicts is what makes the exclusion
        # set window-wide: a grader that dies during the LAST arm still drops its task from the FIRST
        # duel's denominator (§6.3c).
        # §5.4: a challenger is abandoned when its CEILING cannot clear `king + eps`. Both sides of
        # that comparison are recomputed inside `_arm` against the live exclusion set, because §6.3c
        # moves the king's score too — see the check there.
        ran: list[tuple[Queued, tuple[EpisodeResult, ...], ArmAudit]] = []
        for queued, conductor in admitted:
            results, meter = self._arm(opened, arm=queued.hotkey, conductor=conductor,
                                       payer=queued.hotkey,
                                       budget_usd=self.gateway.balance(queued.hotkey), guard=guard,
                                       tasks=opened.tasks, king_arm_results=king_results,
                                       table=table)
            ran.append((queued, results, meter))
            splits[queued.hotkey] = _seen_split(results, opened.seen, guard.failed,
                                                self.pins.world.exchange)
        meters = _endpoint_audit(meters + tuple(meter for _, _, meter in ran))
        # §5.2c's drift audit, after every arm has drawn and before any verdict: a few of the keys
        # that were REPLAYED this window, bought once more live on the owner's allowance.
        retest = self._retest(opened, table, guard)

        # PHASE 6 — every verdict, only now.
        final_b = priced(self.pins.world.exchange)
        contenders: list[Contender] = []
        crowned_dirs: dict[str, Queued] = {}
        king_arm: ArmScore | None = None
        for queued, results, meter in ran:
            if not meter.scoreable:
                # An arm whose dollars cannot be traced to gateway receipts cannot be scored at all
                # (M5 exit 1): `final_b` prices spend, so scoring it would price a crown on a number
                # nobody metered. Refusing is OUR failure, so the shot stays unspent.
                outcomes.append(Outcome(queued.hotkey, queued.block, DEFERRED,
                                        f"this arm's spend could not be reconciled against gateway "
                                        f"receipts ({meter.audit.reason}); it is not scoreable and "
                                        f"the shot is not spent"))
                continue
            # RECORDED spend over the whole slice, not what THIS process metered. `ArmAudit` is
            # built over fresh results only — deliberately, so a resumed prefix does not fail
            # reconciliation — so a fully-resumed arm meters $0 while having spent real money. The
            # first version of this guard read the meter and therefore deferred every challenger a
            # restart had resumed, breaking §8b.5 on the one path it exists to protect.
            spent = sum(result.spend_usd for result in results)
            reached = sum(1 for result in results
                          if result.stopped_reason != "budget_exhausted")
            if not spent > 0.0:
                # No worker was reached on any task, so the action space was never exercised — a
                # model that answers STOP everywhere, or a pool that refused every request. There is
                # no routing in it to compare.
                outcomes.append(Outcome(queued.hotkey, queued.block, DEFERRED,
                                        f"this arm reached no worker model on any of "
                                        f"{len(results)} tasks and spent $0, so there is no routing "
                                        f"to score; the shot is not spent"))
                continue
            # Funding does not decide the crown. §4 zeroes a starved arm's tail and §5.8 requires
            # that cut to be published rather than hidden — so the arm below is scored and revealed
            # exactly as any other. What it may not do is WIN: an allowance of a nanodollar buys one
            # metered call, answers almost none of the slice, and scores ~0 at ~0 cost, which beats
            # any king whose priced spend exceeds its quality. A challenger may take the throne by
            # buying the same quality for less; it may not take it by declining to buy anything.
            eligible = reached >= MIN_SLICE_REACHED * len(results)
            paired = exclude(king_results, results, failed=sorted(guard.failed))
            verdict = duel(_pairs(paired, self.pins.world.exchange, opened.tasks), final_b,
                           nonce=opened.nonce)
            # Scored from INSIDE the loop and identical every time: all arms ran the same task list
            # and the exclusion set is now final, so every duel drops exactly the same rows and the
            # published king arm is the one the duels were computed on.
            king_arm = score_arm(paired.king, self.pins.world.exchange)
            judged.append(queued.hotkey)
            why = _why(verdict)
            if verdict.challenger_wins and not eligible:
                # §5.8: the reader must be able to see that funding, not routing, settled this.
                why = (f"{why} — but the crown is withheld: this arm reached {reached} of "
                       f"{len(results)} tasks before its allowance ran out, and a challenger may "
                       f"not take the throne by declining to buy anything")
            outcomes.append(Outcome(queued.hotkey, queued.block, DUELLED, why,
                                    verdict=verdict,
                                    arm=score_arm(paired.challenger, self.pins.world.exchange),
                                    exhausted=_exhausted(results)))
            if eligible:
                contenders.append(Contender(queued.hotkey, queued.block, verdict))
                crowned_dirs[queued.hotkey] = queued

        # PHASE 7 — batch coronation, every delta measured against the same king arm (§5.2a).
        return self._publish(opened, power, tuple(arms), king_hotkey, king_arm, king_results, guard,
                             tuple(outcomes), champion(contenders), meters, crowned_dirs,
                             judged=judged, table=table, retest=retest, splits=splits,
                             registrations=registrations)

    # --- the steps ---------------------------------------------------------------------------

    def _open(self, window: int) -> Window:
        """Fetch, authenticate, verify — and check the nonce against the chain (§6.3).

        `window.load` closes every check it can, and states the one residual it cannot: **whoever
        picks the nonce picks the slice**, so it re-derives from the nonce IN THE RECORD and leaves
        the comparison against the chain-derived value to the caller. This is that caller. Without
        this line the owner could publish any nonce and choose among as many slices as it liked,
        after challengers had committed — and every signature and every hash would still verify.
        """
        self._publish_window(window)
        try:
            opened = load_window(window, fetch_bytes=self.store.get, owner_ss58=self.owner.ss58,
                                 verify_sig=self.owner.verify_sig, man=self.manifest,
                                 tasks=self.pins.world.tasks)
        except WindowError as exc:
            raise WindowUnavailable(str(exc)) from exc
        expected = self.beacon(window)
        if opened.nonce != expected:
            raise WindowUnavailable(
                f"window {window} carries nonce {opened.nonce!r}, the chain says {expected!r}: "
                f"whoever picks the nonce picks the slice (§6.3)")
        return opened

    def _publish_window(self, window: int) -> None:
        """The owner's half of window open (§6.2), done by the daemon because the daemon IS the owner
        (D1) and already holds the signer the reveal uses: build the self-contained window file from
        the committed schedule entry, the chain beacon and today's catalog, sign it, and put it at
        `window_path`. ONLY WHEN THE STORE HAS NONE. An existing file is never replaced — a restart
        re-reads what it published, and no second build can hand a later arm a different catalog
        snapshot than an earlier arm faced — and an unreadable one is left for `load` to refuse in
        its own words. Measured 2026-09-08 on testnet 526: the first real window refused itself with
        `NoSuchKey`, because the only thing that had ever written this file was the smoke script,
        inline, and §8 step 1 says "fetch the committed window file" as if someone else had.
        """
        path = window_path(window)
        try:
            self.store.get(path)
            return
        except Exception as exc:                 # noqa: BLE001 — classified by `_absent`
            if not _absent(exc):
                return
        record = build_window(self._entries[window], tasks=self.pins.world.tasks,
                              catalog=self.pins.world.catalog, nonce=self.beacon(window),
                              revealed=sorted(self.history.revealed))
        envelope = {"record": record, "sig": self.owner.sign(signing.canonical(record))}
        self.store.put(path, signing.canonical(envelope), content_type="application/json")
        self._log(f"window {window}: published the window file at {path}")

    def _fund(self, hotkey: str, registration_id: str, window: int) -> str | None:
        """D18: bind this hotkey's arm to the key it registered, capped for this window.

        Returns None when the arm is funded — on its own key, or on a hand-credited allowance when
        no key is registered — and otherwise the reason it is not, which becomes the deferral row.
        Every refusal here is DEFERRED rather than REFUSED: a dead key, an empty account and a zero
        cap are all fixable by the miner without a new submission, so the shot stays intact (§4's
        "Kind" property, the same reason an unfunded allowance defers).

        The record is read fresh every window, so a miner rotates a key or moves a cap by
        re-uploading it. The key is probed (`funding`) before any arm is opened on it, so a key the
        provider refuses is deferred here for the price of one GET rather than discovered one
        refused call at a time — and the window's allowance is the miner's cap bounded by what the
        key reports it can still spend.
        """
        if self.key_seed is None:
            return None
        path = f"submissions/{registration_id}/{KEY_NAME}"
        try:
            raw = self.private_models.get(path)
        except Exception as exc:                 # noqa: BLE001 — absent, or the store is down
            if not _absent(exc):
                self._log(f"window {window}: could not read {path} ({exc}); {hotkey} stays on "
                          f"its credited allowance")
            return None
        try:
            record = SealedKey.from_bytes(raw)
            api_key = open_key(record, self.key_seed, hotkey=hotkey,
                               registration_id=registration_id)
        except AccessError as exc:
            return f"the registered OpenRouter key cannot be used: {exc} (D18)"
        if self.provider_for is None:
            from .openrouter import OpenRouterClient  # noqa: PLC0415 — the live client, like `main`

            provider: Provider = OpenRouterClient(api_key)
        else:
            provider = self.provider_for(api_key)
        cap = record.cap_usd
        probe = getattr(provider, "funding", None)
        if probe is not None:
            try:
                cap = probe().bound(cap)
            except WorkerError as exc:
                if exc.status in KEY_REFUSALS:
                    return (f"OpenRouter refused the registered key (HTTP {exc.status}); register "
                            f"a live key, or fund the account behind this one (D18)")
                self._log(f"window {window}: could not probe {hotkey}'s key ({exc}); the arm "
                          f"runs against the cap of ${record.cap_usd:.2f}")
        balance = self.gateway.bind(hotkey, provider, max(cap, 0.0), window=window)
        if balance <= 0.0:
            return (f"the registered OpenRouter key has nothing to spend this window: the cap is "
                    f"${record.cap_usd:.2f} and the account reports ${cap:.2f} left; fund the "
                    f"OpenRouter account or raise the cap (D18)")
        return None

    def _queue(self, metagraph: Metagraph, window: int,
               notes: dict[str, str]) -> tuple[list[Queued], list[Outcome]]:
        """The challenger queue, from the chain, in `(commit_block, hotkey)` order (§8b.1).

        THREE FILTERS, IN THIS ORDER, AND THE ORDER IS THE POINT.

        1. **Judged hotkeys leave first.** `CommitmentOf` survives forever, so every miner ever
           judged is still committed and still visible; counting them against `MAX_QUEUE_DEPTH`
           would turn away live challengers on behalf of settled ones and spring the deregistration
           trap the cap exists to close. They are not reported either — the reveal would grow
           without bound, and their permanent record is the mailbox's and the history's.
        2. **The depth cap, at the door.** A commit beyond it is REFUSED rather than accepted
           (§8b.1): taking a registration burn for a slot that will expire before it is judged is the
           outcome the cap exists to prevent, and taking the money first makes it worse rather than
           kinder. Refused costs nothing — no download, no arm, no consumed eligibility — and the one
           shot is still there once the queue has drained.
        3. **The mailbox consumes eligibility, after the cap.** Consuming is what §7 spends, so a
           challenger we turned away must not be consumed: it is still entitled to rotate a
           credential and finish its upload.
        """
        pending = [c for c in self.chain.commitments() if c.hotkey not in self.history.judged]
        # 1b. UNFUNDED ENTRIES DO NOT HOLD A SLOT. §4 defers a challenger with no allowance without
        #     spending its shot, which is the kind thing to do and — before this line — a permanent
        #     squat: a deferred hotkey is never judged, so it never leaves `pending`, so twelve of
        #     them committed early and never funded would fill `MAX_QUEUE_DEPTH` forever and every
        #     funded challenger after them would be refused at the door. The cap exists to bound how
        #     long a QUEUED miner waits on US (§8b.1's deregistration trap); a miner waiting on their
        #     own wallet is not in that queue and must not consume its capacity. They are still
        #     listed and still reported — just not counted.
        # 1c. THE MINER'S OWN KEY FUNDS THE ARM (D18), so it is bound BEFORE the funded split —
        #     which needs the registration for the prefix, so that is resolved here once. A key
        #     that cannot be used leaves its reason in `notes` for the deferral row below.
        resolved: dict[str, Registration] = {}
        for commitment in pending:
            neuron = metagraph.resolve(commitment.hotkey)
            if neuron is None or neuron.registered_at is None:
                continue
            resolved[commitment.hotkey] = Registration(
                netuid=self.netuid, uid=neuron.uid, hotkey=commitment.hotkey,
                registration_block=neuron.registered_at)
            note = self._fund(commitment.hotkey, resolved[commitment.hotkey].registration_id,
                              window)
            if note is not None:
                notes[commitment.hotkey] = note
        funded = [c for c in pending if self.gateway.balance(c.hotkey) > 0.0]
        unfunded = [c for c in pending if self.gateway.balance(c.hotkey) <= 0.0]
        queue, turned_away = [], []
        for commitment in funded[:MAX_QUEUE_DEPTH] + unfunded:
            registration = resolved.get(commitment.hotkey)
            if registration is None:
                # Ownership comes from `Keys`, but the R2 prefix is derived from the registration
                # block, so without it there is no prefix to fetch from. Refused entry, nothing
                # spent — it costs the miner a window, not a submission.
                turned_away.append(Outcome(
                    commitment.hotkey, commitment.block, UNQUEUED,
                    "the chain reports no registration block for this hotkey, so its submission "
                    "prefix cannot be derived"))
                continue
            try:
                self.mailbox.consume(commitment, registration)
            except AccessError as exc:
                turned_away.append(Outcome(commitment.hotkey, commitment.block, SKIPPED, str(exc)))
                continue
            queue.append(Queued(commitment=commitment, registration=registration))
        turned_away.extend(
            Outcome(c.hotkey, c.block, UNQUEUED,
                    f"the queue was already at MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH}; a commit beyond "
                    f"the cap is refused, not queued (§8b.1)")
            for c in funded[MAX_QUEUE_DEPTH:])
        return queue, turned_away

    def _wait(self, rank: int) -> int:
        """§8b.1: **publish the current wait** — the number a deferred challenger needs and the one
        the owner's `immunity_period` has to cover (`check_launch`).

        `rank` is its 0-based place among this window's DEFERRALS, not its place in the queue: the
        overflow is exactly the queue that will be at the front next time, so the first one deferred
        waits one window, and the ninth waits two. At `MAX_QUEUE_DEPTH` the worst wait is
        `MAX_QUEUE_DEPTH / MAX_DUELS_PER_WINDOW` windows, which is where `check_launch`'s immunity
        arithmetic comes from — the two numbers must move together or the cap stops being safe.
        """
        return rank // MAX_DUELS_PER_WINDOW + 1

    def _admit(self, queued: Queued) -> Conductor:
        """§8b.2's pre-flight, run before the king's arm: bytes, then architecture, then the load.

        ORDER IS THE DESIGN, and it is `store.py`'s: the manifest is checked against the on-chain
        commitment first, so a tree that is not the one committed to is refused for the price of one
        small object rather than an hour of transfer; then every file is hashed as it lands, so a
        swapped shard is refused at that shard; then `admit` reads headers only (§1.1); then the
        serving stack is asked to load it, which is where an OOM or a load timeout appears.

        Every one of those is `REFUSED` at the call site: the submission is invalid, the shot is
        spent, and the king is not charged (§8b.2). The `except` there is broad because these three
        failures are of three different kinds and §8b.2 gives them one consequence — narrowing it
        would let an OOM crash the window instead, which spends every OTHER queued miner's shot on
        our failure.
        """
        dest = self.root / "trees" / queued.registration.registration_id
        fetch_submission(self.private_models, dest, commitment=queued.commitment,
                         registration=queued.registration)
        admit(dest, self.reference)
        return self.serve(dest, served_name(queued.commitment))

    def _king_conductor(self, crown: Crown) -> Conductor:
        if crown.is_king_zero:
            return self.pins.king_zero
        return self.serve(Path(crown.model_dir), crown.served_model)

    def _payer(self, crown: Crown) -> str:
        """Who the king's arm is billed to. King₀ is a policy in the owner's scaffold, so the owner
        pays (§8b.8); a miner king pays from its own allowance and must stay funded (D6)."""
        return self.owner_account if crown.is_king_zero else crown.hotkey

    def _arm(self, opened: Window, *, arm: str, conductor: Conductor, payer: str,
             budget_usd: float, guard: _GraderGuard, tasks: Sequence,
             king_arm_results: Sequence[EpisodeResult] = (),
             table: OutcomeTable | None = None) -> tuple[tuple[EpisodeResult, ...], ArmAudit]:
        """One arm over the window's slice — metered, resumable, and bounded by the per-duel clock.

        THIS REPLACES `Scaffold.run_window`, AND ONLY BECAUSE §8b.5 REQUIRES IT. `simulate`'s
        docstring records that the per-duel clock and mid-duel restart "need per-task control of the
        arm, which `Scaffold.run_window` owns" and "belong to the daemon that replaces this loop's
        `_arm`". This is that replacement, and it is written to be indistinguishable from
        `run_window` on a cold start — same `task_order`, same budget threading, same clock check
        ahead of the allowance check, same `_unreached` row — which is asserted by test rather than
        left to inspection, because a divergence here would score the daemon on a different arm than
        the one the dev kit shows a miner (M6b's whole promise).

        THE CATALOG COMES FROM THE VERIFIED RECORD, never from a live fetch: that is what freezing it
        is for, since king and challenger evaluated minutes apart would otherwise face different
        action spaces and the duel would not be paired (§6.2).

        A RESTART GRANTS THE REMAINING TASKS A FRESH PER-DUEL CLOCK, deliberately. Persisting the
        deadline would abandon an arm because the validator was down, which is the owner's outage
        charged to the miner — the failure §8b.6 refuses on the weight path, arriving on the spend
        path instead. The allowance is NOT refreshed: it is threaded forward from the checkpointed
        spend, because that money is gone.

        ONE ARM, ONE LEDGER — AND A RETRY IS A NEW ARM'S WORTH OF CALLS. `open_duel` refuses a
        `duel_id` twice, which is right (two arms sharing one identity make their receipts
        indistinguishable), so the attempt number is part of the id. Without it, a window that raised
        after its first arm would fail on `open_duel` on every later poll and never recover, with the
        first attempt's money already spent — the checkpoint would be there and nothing could read
        it. The receipts of attempt 2 cover exactly the calls attempt 2 made, which is what `audit`
        compares them against.
        """
        self._attempts[(opened.epoch, arm)] = self._attempts.get((opened.epoch, arm), 0) + 1
        token = self.gateway.open_duel(
            payer, f"w{opened.epoch}-{_UNSAFE.sub('_', arm)}-{self._attempts[(opened.epoch, arm)]}")
        scaffold = Scaffold(opened.catalog, conductor, GatewayWorker(self.gateway, token),
                            guard, clock=self.clock, inspect=guard.inspecting, table=table)
        drawn = {"hits": 0, "fills": 0}        # §5.2c, per arm — summed under `writing`
        done = {result.task_id: result for result in self.checkpoint.resume(opened.epoch, arm)}
        allowance = self.checkpoint.snapshot(opened.epoch, arm, budget_usd)
        remaining = allowance - sum(result.spend_usd for result in done.values())
        deadline = self.clock() + DUEL_WALL_CLOCK_SECONDS

        # RUN CONCURRENTLY, ACCOUNT IN NONCE ORDER. Measured 2026-09-03, an episode takes 45 s to
        # 1.4 min, so a serial 250-task arm is 3-6 h against `DUEL_WALL_CLOCK_SECONDS` and every
        # window would end on the clock with most of the slice unreached — §5.4 sizes the arm
        # assuming parallelism. Concurrency changes only how many episodes are in flight: each is
        # independent (same catalog, same history, same grader), so nothing a miner trains against
        # moves, and `EPISODE_CONCURRENCY = 1` reproduces the serial arm exactly.
        #
        # THE ALLOWANCE SIZES THE BATCH, AND WITHOUT THAT THE BUDGET STOPS BINDING. Workers that only
        # check `remaining` before dispatching all read the same figure — none has reported yet — so
        # `k` episodes start against a budget that covers one, and a miner with a starved allowance
        # has every task run and overspends several-fold on their own key. That is not a widened
        # bound; it is the loss of §4's rule that exhaustion zeroes the remainder and of §5.8's
        # requirement that the cut be visible. So each batch is sized to what the money left can
        # actually buy, priced at the mean of what episodes have cost THIS arm. The first batch is
        # one episode, because before it there is no price to reason with.
        #
        # Batches rather than a worker pool with a condition variable, deliberately: the pool wants
        # workers to block until the allowance frees up, and a blocked worker that never re-checks
        # whether the slice is finished deadlocks the arm. A batch boundary is a place where all of
        # that is simply already true.
        #
        # Two bounds still widen against a serial arm, both recorded rather than smoothed over: the
        # overrun is one batch's spend rather than exactly one DELEGATE, and a crash can lose the
        # batch in flight rather than nothing. The second is why a result is checkpointed the moment
        # it lands rather than with its peers — §8b.5 wants a finished result durable before its
        # money stops being visible, and `resume` keys by task id, so the checkpoint's ORDER never
        # mattered.
        # §5.4's second multiplier: stop paying for an arm that provably cannot win. `beat` is the
        # figure this arm has to exceed — the king's `final` plus `eps` — and it is known before the
        # arm starts because §5.2a already ran the king's whole slice. `best_possible_final` bounds
        # what this arm can still reach; when even that cannot clear the bar, every further episode
        # is money spent on a settled question, on the miner's own allowance.
        ordered = list(task_order(tasks, opened.nonce))
        placed: list[EpisodeResult | None] = [None] * len(ordered)
        fresh_at: list[int] = []
        runs = spent = 0.0
        writing = threading.Lock()
        index = 0
        while index < len(ordered):
            # An UNBOUNDED allowance imposes no bound, so it sizes nothing: `inf // mean` is not a
            # batch width, and `int()` of it raises rather than returning something large. The
            # archetype sweep runs with `budget_usd=inf` deliberately, so this is a live path.
            mean = (spent / runs) if runs else None
            if not math.isfinite(remaining):
                width = EPISODE_CONCURRENCY
            elif mean is None or mean <= 0.0:
                width = 1
            else:
                width = max(1, min(EPISODE_CONCURRENCY, int(remaining // mean)))
            batch: list[int] = []
            while index < len(ordered) and len(batch) < width:
                task = ordered[index]
                if task.task_id in done:
                    placed[index] = done[task.task_id]
                elif task.task_id in guard.failed:
                    # ALREADY DROPPED, SO DO NOT BUY IT. §6.3c removes a task whose grader died from
                    # every arm and from the denominator, and the exclusion set is window-wide and
                    # durable — so by the time a later arm reaches such a task the window has already
                    # decided it will not count. Running it anyway spent the miner's allowance on a
                    # result nothing would score. Measured: $0.08 of a miner's money on a task the
                    # window had dropped before the arm started.
                    placed[index] = _unreached(task, EXCLUDED_REASON).result
                else:
                    if self.clock() >= deadline:
                        placed[index] = _unreached(task, DUEL_WALL_CLOCK_REASON).result
                    elif remaining <= 0.0:
                        placed[index] = _unreached(task, "budget_exhausted").result
                    if placed[index] is None:
                        batch.append(index)
                    else:
                        self.checkpoint.append(opened.epoch, arm, placed[index])
                        fresh_at.append(index)
                index += 1
            if not batch:
                continue

            def run(position: int, allowance: float = remaining) -> None:
                episode = scaffold.run_episode(ordered[position], budget_remaining=allowance)
                result = episode.result
                with writing:
                    self.checkpoint.append(opened.epoch, arm, result)
                    placed[position] = result
                    drawn["hits"] += episode.hits
                    drawn["fills"] += episode.fills

            if len(batch) == 1:
                run(batch[0])
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    list(pool.map(run, batch))
            for position in batch:
                remaining -= placed[position].spend_usd
                spent += placed[position].spend_usd
                runs += 1
                fresh_at.append(position)

            # Checked at a batch boundary, where every dispatched episode has reported and the
            # ceiling is computed over a settled prefix. The bound only falls as an arm runs, so a
            # check that does not fire here cannot have fired earlier.
            if king_arm_results and index < len(ordered):
                # THE BAR IS RECOMPUTED WITH THE LIVE EXCLUSION SET, not fixed before the arm ran.
                # §6.3c drops a dead grader's task from BOTH arms, so an exclusion moves the king's
                # score as well as the challenger's; comparing a ceiling that knows about the
                # exclusions against a bar that does not is a false abort waiting to happen.
                scored = [result for result in placed if result is not None]
                failed = frozenset(guard.failed)
                king_now = [r for r in king_arm_results if r.task_id not in failed]
                if scored and king_now and best_possible_final(
                        scored, tasks=tasks, exchange=self.pins.world.exchange,
                        excluded=failed) <= score_arm(king_now, self.pins.world.exchange).final + EPS:
                    while index < len(ordered):
                        # A RESUMED TASK IS NOT AN UNREACHED ONE. This was the only writer of
                        # `placed[index]` in `_arm` that did not check `done` first, so abandoning a
                        # tail overwrote episodes an earlier attempt had already run and already
                        # PAID FOR with futile zeros — and appended those zeros to the checkpoint, so
                        # the loss survived the restart that was meant to recover them. The batch
                        # builder above has always checked; this loop simply did not.
                        task = ordered[index]
                        if task.task_id in done:
                            placed[index] = done[task.task_id]
                        else:
                            placed[index] = _unreached(task, FUTILE_REASON).result
                            self.checkpoint.append(opened.epoch, arm, placed[index])
                            fresh_at.append(index)
                        index += 1

        results = [result for result in placed if result is not None]
        fresh = [placed[position] for position in sorted(fresh_at)]

        ledger = self.gateway.close(token)
        # Reconciled over what THIS process metered. A resumed prefix's receipts belong to the
        # process that spent them, so including it would fail the comparison on every restart and
        # refuse an arm for having survived one — `ArmAudit.resumed` publishes the difference.
        return tuple(results), ArmAudit(
            arm=arm, duel_id=ledger.duel_id, payer=payer,
            audit=audit(fresh, ledger.receipts, duel_id=ledger.duel_id, hotkey=payer,
                        gateway_public_hex=self.gateway.public_hex,
                        replayed=frozenset(ledger.replayed)),
            unfunded_calls=ledger.unfunded_calls, exhausted=_exhausted(results),
            futile=sum(1 for r in results if r.stopped_reason == FUTILE_REASON),
            resumed=len(done), hits=drawn["hits"], fills=drawn["fills"],
            own_key=self.gateway.own_key(payer), endpoints=_endpoints(ledger.served))

    def _retest(self, opened: Window, table: OutcomeTable, guard: _GraderGuard) -> dict | None:
        """§5.2c's drift audit: re-buy a few of the keys the window REPLAYED, live, and say how
        often the fresh draw disagreed with the stored one. Diagnostic — it gates nothing in v1 —
        and it is the baseline any cross-window reuse of rows would have to clear first.

        Owner money, bounded by `RETEST_MAX_KEYS` and `RETEST_MAX_USD` (the stored price is the
        estimate). Chosen from the nonce, so the sample is reproducible from the reveal. Bought
        through a scaffold WITHOUT a table, on its own owner duel, and graded through the guard —
        a grader failure here drops the key from the sample, never the task from the window's
        arms, which have already run. None when nothing was replayed.
        """
        hit = [row for row in table.rows() if table.hit_count(row.key) > 0]
        if not hit:
            return None
        hit.sort(key=lambda row: hashlib.sha256(f"{opened.nonce}|{row.key}".encode()).hexdigest())
        chosen, budget = [], RETEST_MAX_USD
        for row in hit:
            if len(chosen) == RETEST_MAX_KEYS or row.cost_usd > budget:
                break
            chosen.append(row)
            budget -= row.cost_usd
        if not chosen or self.gateway.balance(self.owner_account) <= 0.0:
            return None
        self._attempts[(opened.epoch, "retest")] = self._attempts.get((opened.epoch, "retest"), 0) + 1
        token = self.gateway.open_duel(
            self.owner_account, f"w{opened.epoch}-retest-{self._attempts[(opened.epoch, 'retest')]}")
        scaffold = Scaffold(opened.catalog, self.pins.king_zero, GatewayWorker(self.gateway, token),
                            guard, clock=self.clock, inspect=guard.inspecting)
        by_id = {task.task_id: task for task in opened.tasks}
        sampled = grade_disagreed = cost_disagreed = 0
        for row in chosen:
            task = by_id.get(row.task_id)
            if task is None:
                continue
            fresh = scaffold._call_worker(task, row.model_id, [])
            if not fresh.outcome:
                break                              # the owner's allowance ran out: stop sampling
            excluded_before = row.task_id in guard.failed
            grade = None if fresh.text is None else guard(task, fresh.text)
            if fresh.text is not None and row.task_id in guard.failed and not excluded_before:
                continue                           # the grader died on the re-buy: not a sample
            sampled += 1
            if (grade is None) != (row.grade is None) or (
                    grade is not None and abs(grade - row.grade) > 1e-9):
                grade_disagreed += 1
            scale = max(row.cost_usd, fresh.cost_usd, 1e-9)
            if abs(fresh.cost_usd - row.cost_usd) / scale > RETEST_COST_TOLERANCE:
                cost_disagreed += 1
        ledger = self.gateway.close(token)
        return {"keys": sampled, "usd": ledger.spend_usd,
                "grade_disagreements": grade_disagreed, "cost_disagreements": cost_disagreed,
                "grade_disagreement_rate": (grade_disagreed / sampled) if sampled else None,
                "cost_disagreement_rate": (cost_disagreed / sampled) if sampled else None,
                "cost_tolerance": RETEST_COST_TOLERANCE}

    def _king_arm_unusable(self, results: Sequence[EpisodeResult],
                           meter: ArmAudit | None) -> str | None:
        """§8b.2: the two ways one king arm settles a whole window by accident. None means usable."""
        if meter is None:
            return None
        if any(result.stopped_reason == DUEL_WALL_CLOCK_REASON for result in results):
            return ("the king's arm hit the per-duel wall clock, so under §5.2a its zeroed tail "
                    "would be shared by every duel in this window; no duel is scored and no shot is "
                    "spent (§8b.2)")
        if not meter.scoreable:
            return (f"the king's arm could not be reconciled against gateway receipts "
                    f"({meter.audit.reason}); every duel in this window would be priced against an "
                    f"unmetered baseline, so none is scored and no shot is spent")
        return None

    def _resolve_reign(self, metagraph: Metagraph) -> None:
        """§5.5: the crown reverts to King₀ the moment the reigning hotkey stops resolving.

        The bug this closes is invisible in the burn total, which is why it survived: a deregistered
        king is paid nothing either way, so the slate still sums to 1.0 and still burns the same
        0.85. What moves is the PENSION — §5.7 excludes the reigning hotkey from its own lineage, so
        a king still recorded as reigning after leaving the metagraph is excluded from a lineage it
        belongs at the top of, and every surviving pensioner is paid one rank too high.

        A reversion is NOT a coronation: nothing is appended to the lineage, the deposed king keeps
        the rank-1 slot it was dethroned into — a slot whose share then burns — and the throne stays
        vacant until a challenger clears the verdict against King₀, exactly as at cold start.
        """
        crown = self.history.crown
        if not crown.is_king_zero and metagraph.resolve(crown.hotkey) is None:
            self.history.crown_to(Crown())

    # --- §8 step 5: persist, set weights, publish ---------------------------------------------

    def _publish(self, opened: Window, power, arms: tuple[ReferenceArm, ...], king_hotkey: str,
                 king_arm: ArmScore | None, king_results: tuple[EpisodeResult, ...],
                 guard: _GraderGuard, outcomes: tuple[Outcome, ...], crowned: Contender | None,
                 meters: tuple[ArmAudit, ...], queued: Mapping[str, Queued],
                 judged: Sequence[str] = (), table: OutcomeTable | None = None,
                 retest: dict | None = None, splits: Mapping[str, dict] | None = None,
                 registrations: Mapping[str, str] | None = None) -> Reveal:
        """§8 step 5, in its required order: persist history, set weights, publish the reveal.

        Persisting first is not tidiness. Re-judging is NOT idempotent under the one-shot rule, so a
        crash must be able to repeat the cheap half — weights and the reveal — and never the half
        that spent a miner's submission. The reveal is written locally inside the persist step for
        the same reason: a crash between the weights and the publish would otherwise lose a window's
        traces permanently, and D15's entry ramp is what makes this subnet enterable at all.

        Weights are set even when the window scored nobody: a refused window is not owner downtime
        (§8b.6), the schedule simply does not change, and the king keeps earning while the corpus is
        re-measured next window.
        """
        # Back into queue order for the reveal (§8b.1). The phases append in PHASE order — refusals
        # before duels — and a reveal that printed that would claim a queue order it did not use.
        # Sorted here rather than at each return, because there are three of them and the one that
        # forgets is the one nobody reads.
        outcomes = tuple(sorted(outcomes, key=lambda o: (o.commit_block, o.hotkey)))
        # CROWN AFTER VERIFY. A winner's weights must be public before the crown is theirs (D14), so
        # the coronation waits on `_promote`; a copy that fails leaves the verdict standing and the
        # shot spent, and the crown pending for `step` to retry.
        promotion: dict | None = None
        crowned_hotkey: str | None = None
        if crowned is not None:
            entry = queued[crowned.hotkey]
            pending = {"window": opened.epoch, "hotkey": crowned.hotkey,
                       "registration_id": entry.registration.registration_id,
                       "manifest_sha256": entry.commitment.ready.manifest_sha256,
                       "served_model": served_name(entry.commitment), "attempts": 1}
            public_prefix, error = self._promote(pending)
            if public_prefix is not None:
                self._supersede_pending(crowned.hotkey)
                self.history.crown_to(self._crown_for(pending, public_prefix))
                crowned_hotkey = crowned.hotkey
                promotion = {"hotkey": crowned.hotkey, "state": "promoted", "prefix": public_prefix}
            else:
                self._supersede_pending(crowned.hotkey)
                self.history.defer_crown({**pending, "error": error,
                                          "next_attempt_at": self._next_attempt_at(1)})
                promotion = {"hotkey": crowned.hotkey, "state": "pending", "attempts": 1,
                             "error": error}
        self.history.settle(opened.epoch, crowned=crowned_hotkey,
                            judged=judged,
                            revealed=[task.task_id for task in opened.tasks]
                            if power.separates else (),
                            registrations=registrations, at=self.now())

        metagraph = self.chain.metagraph()
        lineage = self.history.lineage()
        weights = emission_weights(lineage, self.history.crown.hotkey, metagraph.hotkey_of,
                                   burn_uid=self.burn_uid, king_zero_uid=self.king_zero_uid)
        # §5.5, captured HERE and not from `self.history.crown`: `settle` has already run by this
        # point and may have crowned a challenger, so reading the live crown would describe the
        # END-of-window king while `king_hotkey` beside it documents the START-of-window one — the
        # two fields would disagree in exactly the window where a coronation happened. At entry
        # `king_hotkey` is still `crown.hotkey` from before the duels, and KING0 is the empty
        # string, so this is that king and no other. Read before the mapping below overwrites it.
        king_is_genesis = not king_hotkey
        # DISPLAY ONLY, AND DELIBERATELY LATE. When King₀ reigns its hotkey is the empty sentinel,
        # which publishes as an absence — a reader (and the dashboard) cannot tell "the genesis king
        # holds the throne" from "this field was forgotten". With `king_zero_uid` set the reveal
        # names that UID's hotkey instead, and `power.king_zero` beside it says which fixed policy
        # that is. Computed here rather than at the call sites because every branch of `step` funnels
        # through `_publish`, and because nothing downstream may read it back: `crown.is_king_zero`
        # still decides whether an arm is served or paid (§8b.2, §8b.8), and it is unchanged.
        if not king_hotkey and self.king_zero_uid is not None:
            king_hotkey = metagraph.hotkey_of.get(self.king_zero_uid, king_hotkey)
        report = WindowReport(
            window=opened.epoch, nonce=opened.nonce, benchmarks=opened.benchmarks,
            n_tasks=len(opened.tasks), freshness=opened.record["freshness"], power=power,
            reference=arms, king_hotkey=king_hotkey, king=king_arm, king_results=king_results,
            king_is_genesis=king_is_genesis,
            graders_failed=tuple(sorted(guard.failed)), outcomes=outcomes,
            crowned=crowned_hotkey,
            pensioners=lineage.pensioners(self.history.crown.hotkey), weights=weights,
            metagraph=metagraph.hotkey_of, table=None if table is None else table.stats())

        body = _record(report, meters, retest=retest, splits=splits, promotion=promotion,
                       crown_model=self._crown_model())
        self._local_reveal(opened.epoch).parent.mkdir(parents=True, exist_ok=True)
        self._local_reveal(opened.epoch).write_text(json.dumps(body, sort_keys=True),
                                                    encoding="utf-8")
        written = self._write_weights(weights, metagraph.block)
        published = self._put_reveal(opened.epoch, body)
        if published:
            self.history.mark_published(opened.epoch)
        return Reveal(report=report, arms=meters, weights_written=written, retest=retest,
                      splits=None if splits is None else dict(splits))

    # --- private submissions, public kings --------------------------------------------------

    def _crown_for(self, pending: Mapping, public_prefix: str) -> Crown:
        return Crown(hotkey=pending["hotkey"],
                     model_dir=str(self.root / "trees" / pending["registration_id"]),
                     served_model=pending["served_model"], public_prefix=public_prefix)

    def _crown_model(self) -> dict | None:
        """Where the reigning king's weights can be downloaded — None for King0."""
        crown = self.history.crown
        if crown.is_king_zero or not crown.public_prefix:
            return None
        return {"bucket": self.public_models.bucket, "prefix": crown.public_prefix,
                "manifest_sha256": crown.public_prefix.rstrip("/").rsplit("/", 1)[-1]}

    def _promote(self, pending: Mapping) -> tuple[str | None, str | None]:
        """Publish a winner's tree to the public models bucket: (prefix, None), or (None, error).

        The only thing read from the private bucket is the manifest, which must still be the one
        the chain names. The bytes come from this validator's verified copy of the tree, never from
        the bucket — `store.promote_submission` says why.
        """
        registration_id = str(pending["registration_id"])
        prefix = f"submissions/{registration_id}/"
        try:
            manifest = fetch_manifest(self.private_models, prefix)
            if manifest.sha256 != pending["manifest_sha256"]:
                raise ValidatorError(f"{prefix} now holds manifest {manifest.sha256}, the chain "
                                     f"committed {pending['manifest_sha256']}")
            public_prefix = promote_submission(self.root / "trees" / registration_id,
                                               self.public_models, manifest=manifest)
        except Exception as exc:          # noqa: BLE001 — every failure defers; none crowns
            # Logged in full, published by type only (teutonic's `error_code`): a store exception
            # can carry the account endpoint and this host's paths, and the reveal is public.
            self._log(f"window {pending['window']}: {pending['hotkey']} won, but its model is not "
                      f"public yet ({type(exc).__name__}: {exc}); the crown waits")
            return None, type(exc).__name__
        self._log(f"window {pending['window']}: {pending['hotkey']}'s model is public at "
                  f"{self.public_models.bucket}/{public_prefix}")
        return public_prefix, None

    def _next_attempt_at(self, attempts: int) -> float:
        return self.now() + PROMOTION_RETRY_BASE_SECONDS * 2 ** max(0, min(attempts - 1, 8))

    def _supersede_pending(self, hotkey: str) -> None:
        """A newer winner replaces an older crown still waiting on its copy.

        The older winner never duelled the newer one, and its claim was conditional on a public copy
        that has not verified; crowning it afterwards would put a king on the throne behind a
        verdict that has already been overtaken.
        """
        pending = self.history.pending_crown
        if pending and pending["hotkey"] != hotkey:
            self._log(f"window {pending['window']}: {pending['hotkey']}'s deferred crown is "
                      f"superseded by {hotkey}, a newer winner")
            self.history.forfeit_crown()

    def _settle_pending_crown(self) -> None:
        """Retry a deferred coronation; crown on success, forfeit once the budget is spent."""
        pending = self.history.pending_crown
        if not pending:
            return
        if self.now() < float(pending.get("next_attempt_at", 0.0)):
            return
        if self.chain.metagraph().resolve(pending["hotkey"]) is None:
            # Crowning it would append a coronation §5.5 reverts on the next refresh, and spend a
            # pension slot on a hotkey that is gone.
            self._log(f"window {pending['window']}: {pending['hotkey']} deregistered before its "
                      f"model was public; the crown is forfeited")
            self.history.forfeit_crown()
            return
        public_prefix, error = self._promote(pending)
        if public_prefix is not None:
            self.history.crown_promoted(self._crown_for(pending, public_prefix),
                                        window=int(pending["window"]))
            self._refresh_weights(self.chain.current_block(), force=True)
            return
        attempts = int(pending.get("attempts", 1)) + 1
        if attempts >= PROMOTION_MAX_ATTEMPTS:
            self._log(f"window {pending['window']}: {pending['hotkey']}'s crown is FORFEITED after "
                      f"{attempts} failed promotions ({error}); a king whose weights nobody can "
                      f"download would break D14")
            self.history.forfeit_crown()
            return
        self.history.defer_crown({**pending, "attempts": attempts, "error": error,
                                  "next_attempt_at": self._next_attempt_at(attempts)})

    def _apply_retention(self) -> None:
        """§8b.7: delete a loser's weights 14 days after it was judged, and keep its manifest.

        Only JUDGED hotkeys are candidates, so a submission still waiting in the queue is never
        touched, and nothing ever crowned — or waiting on its copy — is: the winner's private prefix
        also holds the sealed OpenRouter key a reigning king defends on.
        """
        protected = set(self.history.coronations) | {self.history.crown.hotkey}
        if self.history.pending_crown:
            protected.add(self.history.pending_crown["hotkey"])
        candidates = {f"submissions/{entry['registration_id']}/": float(entry["at"])
                      for hotkey, entry in self.history.judged_at.items()
                      if hotkey not in protected
                      and entry["registration_id"] not in self.history.purged}
        if not candidates:
            return
        plan = retention_plan(candidates, protected=(), now=self.now(), grace_seconds=GRACE_SECONDS)
        if not plan.expired:
            return
        for prefix in plan.expired:
            try:
                freed = apply_retention(self.private_models,
                                        Retention(kept=(), within_grace=(), expired=(prefix,)))
            except Exception as exc:      # noqa: BLE001 — retried on the next poll
                self._log(f"retention: could not delete {prefix} ({exc}); will retry")
                continue
            self.history.purged.add(prefix.split("/")[1])
            self._log(f"retention: deleted {freed} bytes of weights under {prefix}; its manifest "
                      f"is kept")
        self.history.save()

    def _local_reveal(self, window: int) -> Path:
        return self.root / "reveals" / f"{int(window)}.json"

    def _put_reveal(self, window: int, body: Mapping) -> bool:
        """Sign and publish. A store outage does NOT undo the window: the verdicts are already
        persisted and the local copy is already written, so `step` re-publishes on the next turn."""
        envelope = {"record": body, "sig": self.owner.sign(signing.canonical(body))}
        try:
            self.store.put(reveal_path(window), signing.canonical(envelope),
                           content_type="application/json")
        except Exception as exc:          # noqa: BLE001 — the reveal is the only thing lost
            self._log(f"window {window}: reveal not published ({exc}); will retry")
            return False
        # After the reveal and never before it: the pointer must not name a window whose bytes are
        # not there yet, or a reader that believes it fetches a 404 and reports an outage that does
        # not exist. A failure to move the pointer leaves it on the PREVIOUS window, which shows a
        # stale-but-real reveal — strictly better than pointing at nothing — so it is logged and
        # swallowed rather than failing the publish that already succeeded.
        try:
            self.store.put(LATEST_PATH,
                           json.dumps({"window": int(window)}, sort_keys=True).encode(),
                           content_type="application/json")
        except Exception as exc:          # noqa: BLE001 — a stale pointer is not a lost window
            self._log(f"window {window}: latest pointer not moved ({exc}); reveal is published")
        return True

    def _republish(self, window: int) -> None:
        """Re-do the cheap half after a crash between the weights and the publish (M9 exit 2)."""
        path = self._local_reveal(window)
        if not path.exists():
            return
        if self._put_reveal(window, json.loads(path.read_text(encoding="utf-8"))):
            self.history.mark_published(window)

    def _write_weights(self, weights: Mapping[int, float], block: int) -> bool:
        """The ONLY write. `force` because a stable reign is an unchanged slate, and both
        implementations short-circuit one — so without it the validator physically cannot re-submit
        and ages out of Yuma's `activity_cutoff` while scoring every window correctly."""
        try:
            self.chain.set_weights(weights, force=True)
        except WeightRateLimited as exc:
            self._log(f"weights rejected by the rate limit, the previous slate stands: {exc}")
            return False
        self._last_weight_block = block
        return True

    def _refresh_weights(self, block: int, *, force: bool = False) -> bool:
        """Re-submit the CURRENT slate between windows (see `WEIGHT_REFRESH_BLOCKS`).

        It recomputes rather than replays: a pensioner that deregistered since the last write must
        have its share burn, and §5.5's reversion must happen even in a window that never opened, or
        the pension ranks behind a departed king are all paid one rank too high.
        """
        due = (self._last_weight_block is None
               or block - self._last_weight_block >= WEIGHT_REFRESH_BLOCKS)
        if not (force or due):
            return False
        metagraph = self.chain.metagraph()
        self._resolve_reign(metagraph)
        return self._write_weights(
            emission_weights(self.history.lineage(), self.history.crown.hotkey,
                             metagraph.hotkey_of, burn_uid=self.burn_uid), block)

    def _log(self, message: str) -> None:
        print(f"[v3-validator] {message}", flush=True)


def _seen_split(results: Sequence[EpisodeResult], seen: Collection[str], failed: Collection[str],
                exchange: Mapping) -> dict:
    """One arm's `quality` and `final` on the slice's seen rows and on its unseen rows (§5.2c's
    memorisation meter), scored by the one definition (`score_arm`) and excluding what §6.3c
    dropped. Diagnostic: the verdict never reads it. A side with no rows reports None."""
    def part(rows: list[EpisodeResult]) -> dict:
        rows = [r for r in rows if r.task_id not in failed]
        if not rows:
            return {"n_tasks": 0, "quality": None, "final": None}
        scored = score_arm(rows, exchange)
        return {"n_tasks": len(rows), "quality": scored.quality, "final": scored.final}
    seen = set(seen)
    return {"seen": part([r for r in results if r.task_id in seen]),
            "unseen": part([r for r in results if r.task_id not in seen])}


def _record(report: WindowReport, meters: Sequence[ArmAudit], *, retest: dict | None = None,
            splits: Mapping[str, dict] | None = None, promotion: dict | None = None,
            crown_model: dict | None = None) -> dict:
    """The reveal as JSON — including D15's full decision traces for the arms the OWNER paid for.

    The king's arm and the two reference arms, never a losing challenger's: D15 publishes the traces
    the owner generated as a by-product of running the subnet, so that entering does not require
    first buying a five-figure training set. A miner's own arm is theirs, and §5.8's exhaustion
    numbers are what the reveal says about it instead.
    """
    return {
        "v": 1,
        "window": report.window,
        "nonce": report.nonce,
        "benchmarks": list(report.benchmarks),
        "n_tasks": report.n_tasks,
        "freshness": report.freshness,
        "power": {"king_zero": report.power.king_zero, "contrast": report.power.contrast,
                  "higher": report.power.higher, "spread": report.power.spread,
                  "lcb": report.power.lcb, "floor": report.power.floor,
                  "n_tasks": report.power.n_tasks, "separates": report.power.separates,
                  "by_benchmark": [list(row) for row in report.power.by_benchmark]},
        "graders_failed": list(report.graders_failed),
        "king_hotkey": report.king_hotkey,
        # §5.5, stated rather than left to be inferred from `king_hotkey` being empty — see
        # `WindowReport.king_is_genesis`. `power.king_zero` beside it names WHICH fixed policy.
        "king_is_genesis": report.king_is_genesis,
        "king": _arm_json(report.king),
        "outcomes": [_outcome_json(outcome) for outcome in report.outcomes],
        "crowned": report.crowned,
        # Where the reigning king's weights can be downloaded (D14): the public models bucket, at
        # the prefix named by the manifest digest the chain committed to. None while King0 reigns.
        "crown_model": crown_model,
        # This window's coronation and its public copy: promoted, or pending a copy that has not
        # verified yet (the verdict stands; the crown waits). None when nobody won.
        "promotion": promotion,
        "pensioners": list(report.pensioners),
        "weights": {str(uid): share for uid, share in sorted(report.weights.items())},
        "metagraph": {str(uid): hotkey for uid, hotkey in sorted(report.metagraph.items())},
        "metering": [{"arm": m.arm, "duel_id": m.duel_id, "payer": m.payer,
                      "scoreable": m.audit.scoreable, "reason": m.audit.reason,
                      "metered_usd": m.audit.metered_usd, "recorded_usd": m.audit.recorded_usd,
                      # §5.2c: what the arm was debited (equal to `recorded_usd`) against what the
                      # provider was really paid, and the delegates behind each.
                      "charged_usd": m.audit.charged_usd, "provider_usd": m.audit.provider_usd,
                      "calls": m.audit.calls, "hits": m.hits, "fills": m.fills,
                      "unfunded_calls": m.unfunded_calls,
                      "exhausted": m.exhausted, "futile": m.futile,
                      "resumed": m.resumed,
                      # D18: which key the arm ran on, which endpoint served each model it bought
                      # live, and the pairs no owner-key arm was served by (§11-1 evidence).
                      "own_key": m.own_key,
                      "endpoints": {model: list(served) for model, served in m.endpoints.items()},
                      "drift": list(m.drift)} for m in meters],
        # §5.2c's diagnostics. The table's STATS, never its rows: D15 publishes the arms' traces.
        "outcome_table": report.table,
        "retest": retest,
        "seen_unseen": None if splits is None else dict(splits),
        "traces": {"king": [episode_json(r) for r in report.king_results],
                   **{arm.name: [episode_json(r) for r in arm.results]
                      for arm in report.reference}},
    }


def _endpoints(served: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    """Per model, the endpoints that served this arm's fills: the provider's name, or
    `provider:served_model` when the response named a different model than the one asked for."""
    seen: dict[str, set[str]] = {}
    for row in served:
        label = (row["provider"] if row["served_model"] in ("", row["model"])
                 else f"{row['provider']}:{row['served_model']}")
        seen.setdefault(row["model"], set()).add(label)
    return {model: tuple(sorted(labels)) for model, labels in sorted(seen.items())}


def _endpoint_audit(meters: Sequence[ArmAudit]) -> tuple[ArmAudit, ...]:
    """D18's §11-1 evidence. For every arm on a miner's own key, the `model@endpoint` pairs it was
    served by that no arm on the OWNER's key was served by this window. The reference arms run
    first and on the owner's key, so a model they bought has a baseline; a model only the miner's
    arm bought has none and every endpoint of it is listed. Evidence, not a gate: a listed pair
    is a miner's account routing differently from the owner's, which is exactly the residual §11-1
    names, and the reveal is where it becomes visible."""
    owner_seen: dict[str, set[str]] = {}
    for meter in meters:
        if not meter.own_key:
            for model, served in meter.endpoints.items():
                owner_seen.setdefault(model, set()).update(served)
    return tuple(
        meter if not meter.own_key else replace(meter, drift=tuple(
            f"{model}@{endpoint}" for model, served in meter.endpoints.items()
            for endpoint in served if endpoint not in owner_seen.get(model, ())))
        for meter in meters)


def _arm_json(arm: ArmScore | None) -> dict | None:
    if arm is None:
        return None
    return {"quality": arm.quality, "final": arm.final,
            "per_benchmark": [{"benchmark": row.benchmark, "n_tasks": row.n_tasks,
                               "quality": row.quality, "spend": row.spend, "final": row.final}
                              for row in arm.per_benchmark],
            "excluded": [[name, list(flags)] for name, flags in arm.excluded]}


def _outcome_json(outcome: Outcome) -> dict:
    verdict = outcome.verdict
    return {"hotkey": outcome.hotkey, "commit_block": outcome.commit_block,
            "status": outcome.status, "detail": outcome.detail,
            "shot_spent": outcome.shot_spent, "won": outcome.won,
            "exhausted": outcome.exhausted,
            "arm": _arm_json(outcome.arm),
            "verdict": None if verdict is None else {
                "delta": verdict.delta, "lcb": verdict.lcb, "eps": verdict.eps,
                "loo_min": verdict.loo_min, "loo_dropped": verdict.loo_dropped,
                "median": verdict.median, "sign_wins": verdict.sign_wins,
                "sign_p": verdict.sign_p, "n_tasks": verdict.n_tasks,
                "by_benchmark": [list(row) for row in verdict.by_benchmark],
                "loo": [list(row) for row in verdict.loo]}}


# --- the entry point ---------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thirtyspokes-validator",
        description="Run the owner-run v3 validator: open the committed window, gate it, admit the "
                    "queue, measure the king ONCE, duel every challenger, crown, set weights and "
                    "publish the reveal. Needs a chain, an R2 bucket, an owner wallet, a funded "
                    "gateway and a served model. The offline path is `thirtyspokes-sim`.")
    parser.add_argument("--state", type=Path, required=True, metavar="DIR",
                        help="where history, checkpoints, model trees and local reveals live")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet", required=True, help="the owner's wallet name")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--genesis-block", type=int, required=True,
                        help="the block window 1 opens at; its hash is window 1's nonce (§6.3)")
    parser.add_argument("--window-blocks", type=int, required=True,
                        help="must exceed the chain's weights_rate_limit (§8b.6)")
    parser.add_argument("--windows", type=int, required=True, metavar="N",
                        help="how many windows the committed schedule covers (§6.3)")
    parser.add_argument("--immunity-blocks", type=int, required=True,
                        help="the subnet's immunity_period; must cover three windows (§8b.1)")
    parser.add_argument("--world", metavar="module:attr", required=True,
                        help="a zero-argument callable returning the published `Pins` — M3a's "
                             "lambda/C table, King0 and its contrast, over the real corpus (§5.1b)")
    parser.add_argument("--reference-tree", type=Path, required=True, metavar="DIR",
                        help="the owner's pinned reference model tree (§1.1)")
    parser.add_argument("--serve-url", metavar="URL[,URL…]",
                        help="operator-managed serving: the OpenAI-compatible endpoint(s) the "
                             "artifacts under test are ALREADY served on; comma-separated when "
                             "there is one server per card, the first that lists a committed name "
                             "serves it. Required unless --serve-host is given")
    parser.add_argument("--serve-host", metavar="SSH_TARGET",
                        help="validator-managed serving: the serving host, reached by ssh; the "
                             "daemon launches vLLM there for each tree it must serve and evicts "
                             "least-recently-used. The host gets a launch script and nothing else")
    parser.add_argument("--serve-cards", default="0:8002,1:8003", metavar="GPU:PORT[,GPU:PORT]",
                        help="with --serve-host: the cards and the ports their servers listen "
                             "on, reached here at http://127.0.0.1:PORT/v1 (docs/VALIDATOR.md §2)")
    parser.add_argument("--serve-trees", default="/var/v3/trees", metavar="DIR",
                        help="with --serve-host: the directory ON THE SERVING HOST that holds the "
                             "trees this daemon fetches into <state>/trees (the shared mount)")
    parser.add_argument("--serve-preamble", default=DEFAULT_PREAMBLE, metavar="SHELL",
                        help="with --serve-host: the exports the launch script runs before vLLM "
                             "(toolchain paths; never a credential)")
    parser.add_argument("--sandbox-host", required=True, metavar="ENDPOINT",
                        help="the docker endpoint EVERY container this process starts lands on — "
                             "the graders and the worker read channel alike (§8b.9): "
                             "`ssh://user@host` for the sandbox fleet, or the literal `local` to "
                             "state out loud that this box executes miner-written code itself")
    parser.add_argument("--grade-dir", type=Path, metavar="DIR",
                        help="where container payloads are built. Must resolve at the IDENTICAL "
                             "absolute path on this box and on the sandbox host, because `docker "
                             "run -v` is resolved by the DAEMON; required when --sandbox-host names "
                             "another machine")
    parser.add_argument("--r2-endpoint", required=True)
    parser.add_argument("--r2-bucket", required=True,
                        help="the PUBLIC store: window files, reveals and credential envelopes")
    parser.add_argument("--r2-private-model-bucket", required=True, metavar="BUCKET",
                        help="where every submission is uploaded and fetched from; must not be "
                             "publicly readable")
    parser.add_argument("--r2-public-model-bucket", required=True, metavar="BUCKET",
                        help="where a winner's model is copied, content-addressed, once it has won")
    parser.add_argument("--per-benchmark", type=int, required=True,
                        help="tasks drawn per benchmark per window (§6.3b)")
    parser.add_argument("--minimum", type=int, required=True,
                        help="below this a benchmark is dropped and the window is NARROWED (§6.3b)")
    parser.add_argument("--king-zero-uid", type=int, default=None, metavar="UID",
                        help="pay King0's crown to this UID and name its hotkey as the reigning "
                             "king in the reveal, instead of §5.5's default of burning it. Set it "
                             "to the burn UID to leave the on-chain slate identical and change "
                             "only what the published record claims")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--check", action="store_true",
                        help="run the launch-time gates against the chain and exit, spending "
                             "nothing and scoring nobody")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """`thirtyspokes-validator`. **Wiring only, and it refuses to guess anything that spends.**

    Every seam is a required argument or a required environment variable and there is no offline
    default anywhere in it: a validator that silently fell back to a mock chain, an unfunded gateway
    or a stand-in corpus would set weights on a real subnet from a simulation. `thirtyspokes-sim` is
    a different program on purpose.

    It carries no logic, exactly as `store.r2_bucket` carries none, and for the same reason — the
    alternative to untested configuration is untested configuration plus untested behaviour. Every
    decision it could make has been pulled out into something with a test: the launch gates are
    `check_launch`, the nonce is `chain_beacon`, the serving check is `local_serving`, and the window
    itself is `Validator.run_window`.

    `--check` is the one mode that is safe to point at a live chain: it reads `weights_rate_limit`,
    runs `check_launch`, and exits without opening a window, touching R2 or spending a cent. It runs
    the sandbox gate too, which costs one `docker version` and no container.

    WHERE A MINER'S CODE EXECUTES IS AN ARGUMENT HERE, NOT AN INHERITED ENVIRONMENT VARIABLE, and
    that is the §8b.9 split expressed at the one place a deployment is chosen. This process grades —
    it applies a model's patch and runs a repository's test suite, and with the read channel live it
    starts a container per read as well — so the machine whose daemon answers `--sandbox-host` is
    the machine an escape lands on. `argparse` requires it, so a validator cannot be started without
    saying which one that is; it is written into `V3_DOCKER_HOST` because `sandbox.docker_host()`
    reads that at call time and stays the single source of truth for every docker call site. The
    same for `--grade-dir` and `V3_GRADE_DIR`. What the flags cannot do is move the PROCESS: see
    the sandbox record §1.3 for which arrangements that leaves and which one is deployed.
    """
    import os
    import sys
    from datetime import datetime, timedelta, timezone

    from ..gateway.signing import Signer
    from ..koth.reference import keypair_verifier, wallet_signer
    from .access import Credential
    from .admission import describe
    from .benchmarks import sandbox
    from .chain import BittensorChain
    from .devkit import _load          # ONE spelling of `module:attr` across both CLIs
    from .openrouter import OpenRouterClient
    from .owner import mailbox_seed
    from .store import r2_bucket

    def never_mint(prefix: str) -> Credential:
        """The daemon reads the one-shot ledger and never writes a credential into it. Issuing is
        the owner's separate mailbox tool; a daemon that could mint would be able to hand out write
        access to a submission prefix from inside the loop that scores it."""
        raise AccessError(f"the validator daemon does not issue credentials (asked for {prefix})")

    args = _parser().parse_args(argv)
    buckets = (args.r2_bucket, args.r2_private_model_bucket, args.r2_public_model_bucket)
    if len(set(buckets)) != len(buckets):
        sys.exit(f"thirtyspokes-validator: --r2-bucket, --r2-private-model-bucket and "
                 f"--r2-public-model-bucket must be three distinct buckets, got {buckets}: a "
                 f"submission in a public bucket is downloadable before it has won anything")
    os.environ[sandbox.DOCKER_HOST_ENV] = args.sandbox_host
    if args.grade_dir is not None:
        os.environ[sandbox.GRADE_DIR_ENV] = str(args.grade_dir)
    print(f"[v3-validator] {sandbox.check_grading_host()}")
    cadence = Cadence(genesis_block=args.genesis_block, window_blocks=args.window_blocks,
                      windows=tuple(range(1, args.windows + 1)),
                      immunity_blocks=args.immunity_blocks)
    chain = BittensorChain(netuid=args.netuid, wallet_name=args.wallet, network=args.network,
                           hotkey=args.hotkey)
    if args.check:
        check_chain_launch(chain, cadence)
        print(f"[v3-validator] launch gates pass at netuid {args.netuid}")
        return

    from bittensor_wallet import Wallet
    wallet = Wallet(name=args.wallet, hotkey=args.hotkey)
    credential = Credential(
        endpoint=args.r2_endpoint, bucket=args.r2_bucket,
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        session_token=os.environ.get("R2_SESSION_TOKEN", ""),
        expires_at=datetime.now(timezone.utc) + timedelta(days=1))
    # The journal is derived from --state, the same path `thirtyspokes-owner credit` writes, so
    # the two cannot name different files and the daemon cannot start against a wallet the owner
    # believes is full. Without it the balances are process memory and one restart zeroes every
    # allowance while the checkpoint's `<arm>.budget` still claims the money is there.
    if args.serve_host:
        serving = remote_serving(args.serve_host, args.serve_cards, trees=args.serve_trees,
                                 preamble=args.serve_preamble)
    elif args.serve_url:
        serving = local_serving(args.serve_url)
    else:
        sys.exit("thirtyspokes-validator: one of --serve-url (operator-managed serving) or "
                 "--serve-host (validator-managed serving) is required")
    gateway = OwnerGateway(OpenRouterClient(os.environ["OPENROUTER_API_KEY"]),
                           journal=allowance_journal(args.state))
    validator = Validator(
        pins=_load(args.world), reference=describe(args.reference_tree), chain=chain,
        store=r2_bucket(credential),
        private_models=r2_bucket(replace(credential, bucket=args.r2_private_model_bucket)),
        public_models=r2_bucket(replace(credential, bucket=args.r2_public_model_bucket)),
        mailbox=Mailbox(args.state / "mailbox.json", Signer(), never_mint),
        gateway=gateway,
        owner=Owner(ss58=wallet.hotkey.ss58_address, sign=wallet_signer(wallet),
                    verify_sig=keypair_verifier()),
        cadence=cadence, serve=serving,
        beacon=chain_beacon(chain._substrate, cadence), netuid=args.netuid, root=args.state,
        per_benchmark=args.per_benchmark, minimum=args.minimum,
        king_zero_uid=args.king_zero_uid,
        # D18: the mailbox key's seed opens the keys miners sealed to `thirtyspokes-owner key`.
        key_seed=mailbox_seed(args.state))
    validator.run_forever(poll_seconds=args.poll_seconds,
                          on_reveal=lambda reveal: print(format_reveal(reveal)))


if __name__ == "__main__":
    main()
