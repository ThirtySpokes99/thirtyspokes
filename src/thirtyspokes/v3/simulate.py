"""The whole mechanism as one offline window — `orchestra-sim` (the build plan M9 exit 1).

Every module built so far meets here: the committed window (`window`), the pinned harness
(`scaffold`), the fixed policies and the power gate (`reference`), the priced score (`score`), the
three-condition verdict (`duel`) and the crown's payroll (`emissions`). Nothing is re-expressed —
this module is wiring plus a report, exactly as `devkit` is for the miner's side, and for the same
reason:
arithmetic that exists twice is arithmetic that will eventually disagree with itself, and here the
disagreement moves a crown.

THE PHASE ORDER IS THE MECHANISM, NOT A PIPELINE (§8 steps 2-5). Each boundary closes a way the
arena could be drained, stalled or made incoherent:

  1. **Open the window, then gate it.** The power gate runs before any artifact is touched, because
     a window that cannot separate two FIXED policies cannot rank two competitors either — and a
     refused window must cost a delay rather than a miner's one shot (§5.6, §8 step 2).
  2. **Admit every queued challenger BEFORE the king's arm is run** (§8b.2). The load check exists
     so that "a broken challenger never costs the king money", and that is only true if the king's
     arm is not started until at least one challenger has passed. A window whose whole queue is
     invalid therefore spends the king nothing at all.
  3. **The king's arm is computed ONCE and reused by every duel (§5.2a, D13) — a CORRECTNESS rule.**
     Every duel in a window runs on that window's single slice, so the king's performance on it is
     one quantity; re-running per challenger does not measure it again, it measures the same thing
     with fresh noise. Challengers A and B would then be compared against *different* king
     measurements, and §5.2's batch coronation — which ranks challengers against each other — would
     be picking a champion by run-to-run variance. It also collapses the king-griefing ratio from
     ~1:1 to N:1. Here it is structural: `king_results` is computed above the loop and the same
     tuple is handed to every duel.
  4. **Every arm runs before any verdict is computed.** A grader failure excludes its task from
     BOTH arms and from the denominator (§6.3c, §8b.3), and the set of failed tasks is not complete
     until the last arm has run. Judging duel 1 before challenger 2's arm exposed a flaky grader
     would score two challengers on two different denominators — the asymmetric exclusion §6.3c
     exists to forbid, arriving by the back door.
  5. **Persist, set weights, publish** (§8 step 5). Re-judging is not idempotent under the one-shot
     rule, so the cheap half must be the half a crash repeats.

WHAT A SHOT COSTS, AND WHAT IT DOES NOT. A hotkey gets one submission ever (§7), so this module is
where the "Kind" property is spent or preserved:

  * an **invalid artifact spends the shot** — it was judged, and the answer was no (§8b.2);
  * a **spent hotkey is skipped**, never re-judged, and spends nothing a second time;
  * a **failed power gate defers everyone** with their shots intact, to roll over to the next window
    (§8 step 2). Our inability to measure is never charged to a miner.

WHAT THIS SIMULATION DELIBERATELY DOES NOT COVER, so nobody reads its green run as more than it is:
the per-duel wall clock and mid-duel restart (§8b.2, §8b.5, the build plan M9 exits 7 and 9) both need
per-task control of the arm, which `Scaffold.run_window` owns; they belong to the daemon that
replaces this loop's `_arm`. The per-EPISODE wall clock is enforced, by `Scaffold` itself, and
§8b.1's two queue caps are enforced here, in `run_window`. Nothing here touches a network, a key, a
GPU or a chain: the worker is `MockWorker`, the chain is `MockChain`, and the model trees admission
runs over are a few hundred bytes of real safetensors headers.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..koth import holdout_feed
from ..subnet.chain import MockChain
from .admission import Reference, admit, describe
from .archetypes import Copier, Specialist
from .benchmarks.base import check_channel, check_protocol
from .benchmarks.mock import MockBenchmark, worker_answers
from .conductor import Conductor
from .config import MAX_DUELS_PER_WINDOW, MAX_QUEUE_DEPTH, MIN_SLICE_REACHED
from .devkit import LocalWorld, suite_grade
from .duel import BenchmarkPair, Contender, DuelVerdict, FinalB, champion, duel
from .emissions import KING0, Lineage, emission_weights
from .reference import (FixedPolicy, PowerVerdict, ReferenceArm, always_cheapest, always_strongest, fit_pool,
                        cascade, contrast_for, power_gate, subsample)
from .scaffold import OutcomeTable, Scaffold
from .score import ArmScore, Exchange, score_arm
from .types import Catalog, CatalogEntry, EpisodeResult, Observation, TaskSpec, ToolCall
from .window import Paired, Window, build, exclude, schedule, verify
from .worker import MockWorker, Worker

# What happened to one queued challenger. Statuses rather than a pair of booleans because the five
# cases differ in what they cost the miner, and a reader of the reveal must be able to tell "we
# refused you" from "we could not measure this window" without inferring it.
DUELLED = "duelled"     # judged; shot spent
REFUSED = "refused"     # invalid artifact: judged, the answer was no; shot spent (§8b.2)
SKIPPED = "skipped"     # this hotkey has already been judged once, ever (§7); nothing happens
DEFERRED = "deferred"   # not rankable here, or the window was full; rolls over, shot intact (§5.6,
                        # §8b.1)
UNQUEUED = "unqueued"   # the queue was already at its cap: refused entry, never queued (§8b.1)


@dataclass(frozen=True)
class Submission:
    """One miner's entry: an artifact, a queue position, and an allowance (§7, §4).

    `model_tree` and `serve` are the two halves of §8b.2's pre-flight, and they are separate because
    they fail for different reasons and the refusal must say which: `admit` reads headers and
    refuses a tree that is not the pinned architecture, while `serve` is the serving stack — it
    raises on a load timeout or an OOM, which is a fact about the artifact rather than about its
    shape. Both spend the shot; neither costs the king anything, because both run first.

    `serve` is a callable rather than a loaded Conductor so that the load is what the validator
    does, at the moment it decides to. A field holding an already-live model would move the failure
    to whoever built the queue, where nothing is watching for it.

    `budget_usd` is the miner's own allowance, snapshotted at window open (§4). It is a value here
    and not a source to poll, for the reason `Scaffold.run_window` documents: a mid-duel raise once
    spend climbs would otherwise be free.
    """

    hotkey: str
    commit_block: int
    model_tree: Path
    serve: Callable[[], Conductor]
    budget_usd: float


@dataclass(frozen=True)
class Pins:
    """M3a's published constants: the λ/C table, King₀'s identity, and its contrast (§5.1b, §5.5).

    They are one object because they are measured together, from one fixed-policy sweep, and
    published together — λ_b is "pinned per corpus, not recomputed per window" (§5.1b) precisely so
    that scores are comparable across windows, and King₀ is "whichever fixed policy turns out to be
    best" (§5.5) under that same table. Deriving either one per window would move the target under
    every miner while looking like an ordinary score.

    `world` is `devkit.LocalWorld` — the SAME type the dev kit hands a miner (M6b). That is the
    "Kind" property made structural rather than promised: the validator scores the world a miner can
    hold locally, so there is no shape in which local and scored behaviour could differ.
    """

    world: LocalWorld
    king_zero: FixedPolicy
    contrast: FixedPolicy


@dataclass(frozen=True)
class Outcome:
    """One queued challenger's fate this window — the row the reveal publishes for it.

    `shot_spent` and `won` are derived rather than stored: a recorded outcome that disagreed with
    the verdict it came from would be undetectable, and a miner's single submission is the last
    place in this system where a duplicated field may drift (`render.py`'s precedent).
    """

    hotkey: str
    commit_block: int
    status: str
    detail: str
    verdict: DuelVerdict | None = None
    arm: ArmScore | None = None
    # §5.8: how many tasks this arm's allowance never reached. Stored rather than derived because
    # the traces it comes from are the CHALLENGER's, and D15 publishes the king's and the reference
    # arms' traces — not a losing miner's. When one arm is zeroed on a tail and the other is not,
    # the verdict was decided by funding rather than by routing; that is legitimate under D5 but it
    # must be legible in the reveal instead of buried inside an aggregate that reads as skill.
    exhausted: int = 0

    @property
    def shot_spent(self) -> bool:
        """Judged, either way. A deferred or skipped challenger keeps its one submission (§7)."""
        return self.status in (DUELLED, REFUSED)

    @property
    def won(self) -> bool:
        return self.verdict is not None and self.verdict.challenger_wins


@dataclass(frozen=True)
class WindowReport:
    """One window's reveal (§8 step 5, §5.8, D15) — everything a third party needs to re-derive it.

    The traces live inside `king_results` and `reference`: `EpisodeResult.steps` carries the
    rendered-state digest, the action, the outcome and the cost per step, which is the corpus D15
    publishes so that entering this subnet does not require first buying a training set.

    `king` is the king's arm as every duel saw it — the exclusion set is window-wide and complete
    before any verdict is computed, so all duels share one denominator and this is that one.
    `graders_failed` is published beside it because a window that dropped many tasks has degraded
    power and the reader must be told (§8b.3), not left to infer it from a task count.
    """

    window: int
    nonce: str
    benchmarks: tuple[str, ...]
    n_tasks: int
    freshness: dict
    power: PowerVerdict
    reference: tuple[ReferenceArm, ...]
    king_hotkey: str                                  # the START-OF-WINDOW king (§5.2, batch)
    king: ArmScore | None
    king_results: tuple[EpisodeResult, ...]
    graders_failed: tuple[str, ...]
    outcomes: tuple[Outcome, ...]
    crowned: str | None
    pensioners: tuple[str, ...]
    weights: dict[int, float]
    metagraph: dict[int, str]                         # what the schedule was resolved against
    # The window's outcome table as `OutcomeTable.stats` reports it (§5.2c): rows, fills, hits,
    # the stored-failure rate and the provider's real outlay. The STATS, never the table — D15
    # publishes the arms' traces, not the draw behind every key. None on a report that predates it.
    table: dict | None = None

    @property
    def scored(self) -> bool:
        """Did this window rank anyone? False means no duel ran and no shot was spent."""
        return self.power.separates

    @property
    def king_exhausted(self) -> int:
        """Tasks the king's allowance never reached (§4, §5.8). The king must stay funded (D6): an
        unfunded king is zeroed on the tail and loses, and this is the number that says so."""
        return _exhausted(self.king_results)


def priced(exchange: Mapping[str, Exchange]) -> FinalB:
    """§5.1b's `final_b` as the duel's `FinalB` — computed BY `score.score_arm`, not beside it.

    The duel needs the score as a *function* of `(benchmark, quality, spend)` because the paired
    bootstrap re-scores both arms on every resample and a number cannot be resampled. The obvious
    adapter is to re-type the formula here, and that one line is the most dangerous line this module
    could contain: it would price the duel while `score_arm` prices the published arm, and the day
    §5.1b changes, the crown and the leaderboard would quietly disagree with no test able to see it.

    So the adapter *calls* `score_arm` on a single row whose graded score and spend are the means it
    was handed. With one benchmark `ArmScore.final` is exactly `final_b` (the mean of one value is
    that value, bit for bit), so this is not an approximation of the scored path — it is the scored
    path, and a benchmark missing from the table raises here exactly as it does there (§5.1b's
    forbidden silent fallback).
    """
    def final_b(benchmark: str, quality: float, spend: float) -> float:
        row = EpisodeResult(task_id="<benchmark mean>", benchmark=benchmark, steps=(),
                            graded_score=quality, spend_usd=spend, stopped_reason="stop")
        return score_arm([row], exchange).final
    return final_b


def pin_corpus(catalog: Catalog, tasks: Sequence[TaskSpec],
               grade: Callable[[TaskSpec, str], float], worker: Worker, *,
               inspect: Callable | None = None) -> Pins:
    """The M3a sweep: pick King₀, then fit λ_b to it (§5.5, §5.1b, §6.1).

    Run once, before any window, and published — never re-run per window (see `Pins`). The sweep is
    UNBUDGETED for `devkit._sweep`'s reason: this measures the pool's own cost/quality slope, and a
    slope fitted on an arm that ran out of money halfway is fitted on zeroes.

    Three fixed policies are swept because §6.1 names three and §5.5 defines King₀ as the best of
    them. λ_b is then the slope from the floor to King₀ — the pair the fit makes tie, and the same
    pair §5.6's gate measures the band on — so the cascade, the free win that needs no miner, is
    both the policy the subnet has to beat and the policy whose own rate prices spend.

    `inspect` is the corpus's read channel and it is swept with everything else, because λ_b is a
    slope through the ARM a policy really runs: fitting it on one-completion delegates and then
    scoring windows whose workers read files would price a different episode shape than the one that
    was measured.
    """
    # ONE TABLE ACROSS THE SWEEP (§5.2c): the three fixed policies are then measured against the
    # same draws, so the slope between the floor and King₀ is a difference of policies and not
    # of coin flips — M3a's replicated cheapest arm moved the dominator flag between draws.
    table = OutcomeTable()

    def sweep(policy: FixedPolicy) -> tuple[EpisodeResult, ...]:
        scaffold = Scaffold(catalog, policy, worker, grade, inspect=inspect, table=table)
        return tuple(episode.result for episode in
                     scaffold.run_window(tasks, nonce="m3a-sweep", budget_usd=float("inf")))

    policies = {policy.name: policy for policy in
                (always_cheapest(catalog), always_strongest(catalog), cascade(catalog))}
    # King0 first, then lambda fitted to King0: `reference.fit_pool` is the one definition of that
    # order and of the pair, shared with the dev kit, the archetype world and the M3a script.
    fit = fit_pool({name: sweep(policy) for name, policy in policies.items()})
    return Pins(world=LocalWorld(catalog=catalog, tasks=tuple(tasks), grade=grade, worker=worker,
                                 exchange=fit.exchange, inspect=inspect),
                king_zero=policies[fit.king_zero], contrast=policies[contrast_for(fit.king_zero)])


@dataclass
class Validator:
    """The single owner-run validator (D1), holding what persists between windows.

    The persistent state is deliberately small and is all append-only history: which hotkeys have
    been judged (`spent`, §7's one shot), who has ever been crowned (`coronations`, the pension's
    only input), who reigns (`king`), and which tasks have been scored before (`revealed`, D12's
    published staleness). Everything else about a window is derived from it, which is what lets §8
    step 5 persist the cheap half and re-derive the rest after a crash.

    The whole SCHEDULE is committed at construction, not per window (§6.3): the chain gives each
    hotkey one commitment slot and `set_commitment` overwrites it, so a per-window write would erase
    the owner's governance record. One chained-manifest root covers every window, and each window's
    slice is then re-derived and compared rather than trusted.
    """

    pins: Pins
    reference: Reference
    chain: MockChain
    windows: tuple[int, ...]
    per_benchmark: int
    minimum: int
    # The owner's allowance: it pays for the reference arms every window (§5.6) and for the king's
    # arm while King₀ reigns, since King₀ is a policy in the owner's scaffold rather than a miner's
    # model (§8b.8). Every other arm spends the miner's own allowance.
    owner_budget_usd: float = 1_000.0
    clock: Callable[[], float] = time.monotonic

    spent: set[str] = field(default_factory=set)
    coronations: list[str] = field(default_factory=list)
    king: Submission | None = None
    revealed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        entries = schedule(self.windows, tasks=self.pins.world.tasks,
                           per_benchmark=self.per_benchmark, minimum=self.minimum)
        self._entries = {entry["epoch"]: entry for entry in entries}
        self._manifest = holdout_feed.manifest(entries)

    @property
    def king_hotkey(self) -> str:
        """Who reigns. `KING0` — the best fixed policy — until somebody beats it (§5.5, §8b.8)."""
        return self.king.hotkey if self.king is not None else KING0

    def run_window(self, window: int, submissions: Sequence[Submission]) -> WindowReport:
        """One window, end to end. See the module docstring for why the phases are in this order."""
        opened = self._open(window)
        # Before the gate, because this is free and the gate is not: a task whose prompt and whose
        # declared verbs disagree either wastes the read channel or has its `READ` line graded as
        # its answer, and both are decided by the slice rather than by a policy (`benchmarks.base`).
        check_protocol(opened.tasks)
        guard = _GraderGuard(self.pins.world.grade, self.pins.world.inspect)
        # ONE OUTCOME TABLE FOR THE WINDOW (§5.2c, D17): every arm below — the two references, the
        # king, every challenger, in that order — is scored against the same sampled draws.
        table = OutcomeTable()
        self._resolve_reign()              # §5.5: a king that has left the metagraph is not one
        king_hotkey = self.king_hotkey     # the START-OF-WINDOW king; every duel faces this one

        # PHASE 1 — the gate (§5.6). Two fixed policies over a subsample; if they are
        # indistinguishable here, nothing this window crowns would be anything but noise.
        arms = tuple(ReferenceArm(policy.name, self._arm(
            opened, policy, self.owner_budget_usd, guard,
            subsample(opened.tasks, nonce=opened.nonce), table))
            for policy in (self.pins.king_zero, self.pins.contrast))
        power = power_gate(arms[0], arms[1], nonce=opened.nonce)

        queue = sorted(submissions, key=lambda s: (s.commit_block, s.hotkey))
        # §8b.1's DEPTH cap, applied at the door. The cap is a drain time wearing a headcount's
        # clothes — `2 * MAX_DUELS_PER_WINDOW` is two windows from the back of the queue to the
        # front — because Bittensor's immunity period protects a UID earning nothing for a fixed
        # number of blocks and no longer: a queue that takes longer than immunity to drain
        # deregisters challengers BEFORE they are ever evaluated, and they paid a registration burn
        # for nothing. So a commit beyond the cap is REFUSED rather than accepted. Taking the money
        # first and queueing them anyway is worse than turning them away, not kinder; refused at the
        # door costs nothing — no download, no arm — and its one shot (§7) is still there to spend
        # once the queue has drained.
        turned_away = [Outcome(sub.hotkey, sub.commit_block, UNQUEUED,
                               f"the queue was already at MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH}; a "
                               f"commit beyond the cap is refused, not queued (§8b.1)")
                       for sub in queue[MAX_QUEUE_DEPTH:]]
        queue = queue[:MAX_QUEUE_DEPTH]

        if not power.separates:
            # No duels, no shots spent, everyone still standing rolls over (§8 step 2). The failure
            # is OURS. A hotkey that was already judged is still reported as spent rather than as
            # deferred: telling a miner they are queued when they can never be judged again would
            # be our ambiguity costing them a wait, which is the "Kind" property in miniature.
            return self._publish(opened, power, arms, king_hotkey, None, (), guard,
                                 tuple([_skipped(sub) if sub.hotkey in self.spent
                                        else Outcome(sub.hotkey, sub.commit_block, DEFERRED,
                                                     power.reason)
                                        for sub in queue] + turned_away), None, table=table)

        # PHASE 2 — admission, for the WHOLE queue, before the king's arm is touched (§8b.2).
        outcomes: list[Outcome] = list(turned_away)
        admitted: list[tuple[Submission, Conductor]] = []
        for sub in queue:
            if sub.hotkey in self.spent:
                outcomes.append(_skipped(sub))
                continue
            if len(admitted) == MAX_DUELS_PER_WINDOW:
                # §8b.1's DUEL cap: the validator's wall clock, not its taste. Challenger arms are
                # serial and each is bounded by the two-hour per-duel clock, so a window holds a
                # fixed number of them. The remainder ROLL OVER with their shot unspent — a queue is
                # never dropped, only deferred (§7 gives a hotkey exactly one of the thing a drop
                # would spend) — and they are deferred BEFORE `_prepare`, so a full window costs
                # them the wait and not a ~70 GB pull. Counted against admitted challengers rather
                # than against every commit examined: an artifact refused on its headers never runs
                # an arm, and deferring a live challenger because someone else's upload was broken
                # would spend the clock on nothing.
                outcomes.append(Outcome(
                    sub.hotkey, sub.commit_block, DEFERRED,
                    f"the window is full at MAX_DUELS_PER_WINDOW={MAX_DUELS_PER_WINDOW}; rolls "
                    f"over to the next window with its shot unspent (§8b.1)"))
                continue
            try:
                conductor = self._prepare(sub)
            except Exception as exc:  # noqa: BLE001 — see `_prepare`: OOM and load timeouts too
                self.spent.add(sub.hotkey)
                outcomes.append(Outcome(sub.hotkey, sub.commit_block, REFUSED, str(exc)))
                continue
            admitted.append((sub, conductor))

        # PHASE 3 — the king's arm, ONCE, and only if something is left to duel it (§5.2a, D13).
        king_results: tuple[EpisodeResult, ...] = ()
        if admitted:
            king_results = self._arm(opened, self._king_conductor(), self._king_budget(), guard,
                                     opened.tasks, table)

        # PHASE 4 — every challenger's arm, then every verdict. Splitting the two is what makes the
        # exclusion set window-wide: a grader that dies during the last arm still drops its task
        # from the first duel's denominator (§6.3c).
        challenger_arms = [(sub, self._arm(opened, conductor, sub.budget_usd, guard, opened.tasks,
                                           table))
                           for sub, conductor in admitted]
        final_b = priced(self.pins.world.exchange)
        contenders: list[Contender] = []
        king_arm: ArmScore | None = None
        for sub, results in challenger_arms:
            paired = exclude(king_results, results, failed=sorted(guard.failed))
            verdict = duel(_pairs(paired, self.pins.world.exchange, opened.tasks), final_b,
                           nonce=opened.nonce)
            # Reassigned every iteration and identical every time: all arms ran the same task list
            # and the exclusion set is now final, so every duel drops exactly the same rows. Scoring
            # the king from INSIDE the loop is what guarantees the published king arm is the one the
            # duels were computed on, rather than a differently-filtered lookalike.
            king_arm = score_arm(paired.king, self.pins.world.exchange)
            self.spent.add(sub.hotkey)
            # §4, and the same rule the daemon carries: funding does not decide the crown. An arm
            # that never reached its slice scores 0 quality at ~0 spend, which `final` puts at
            # exactly 0.0 — above any king whose priced spend exceeds its quality. It is still
            # scored and still published, because §4 zeroes a starved tail and §5.8 wants that cut
            # legible; it simply may not win. Mirrored here rather than left to the daemon because
            # this module IS the mechanism and the daemon is its deployment: a rule that lives in
            # only one of them is the drift this file's docstring exists to forbid.
            reached = sum(1 for r in results if r.stopped_reason != "budget_exhausted")
            eligible = (sum(r.spend_usd for r in results) > 0.0
                        and reached >= MIN_SLICE_REACHED * len(results))
            why = _why(verdict)
            if verdict.challenger_wins and not eligible:
                why = (f"{why} — but the crown is withheld: this arm reached {reached} of "
                       f"{len(results)} tasks, and a challenger may not take the throne by "
                       f"declining to buy anything")
            outcomes.append(Outcome(sub.hotkey, sub.commit_block, DUELLED, why,
                                    verdict=verdict,
                                    arm=score_arm(paired.challenger, self.pins.world.exchange),
                                    exhausted=_exhausted(results)))
            if eligible:
                contenders.append(Contender(sub.hotkey, sub.commit_block, verdict))

        # PHASE 5 — batch coronation: the largest delta among the winners, ties on earliest commit
        # block. Well defined only because every one of those deltas was measured against the same
        # king arm (§5.2a).
        crowned = champion(contenders)
        # Back into queue order for the reveal. The phases append in phase order — refusals before
        # duels — and a reveal that printed that would claim a queue order it did not use.
        outcomes.sort(key=lambda outcome: (outcome.commit_block, outcome.hotkey))
        return self._publish(opened, power, arms, king_hotkey, king_arm, king_results, guard,
                             tuple(outcomes), crowned,
                             submissions={sub.hotkey: sub for sub, _ in admitted},
                             table=table)

    # --- the steps, each small enough to read against the spec ----------------------------------

    def _open(self, window: int) -> Window:
        """Build this window's file and verify it against the committed schedule (§6.2, §6.3).

        The nonce is the chain's beacon, which is the residual `window.verify` deliberately does not
        check: whoever picks the nonce picks the slice, so it must come from the chain rather than
        from the owner. Building and then verifying in one process is not a tautology — `verify`
        RE-DERIVES the slice from the nonce and the pool and compares, so the round trip is what
        proves the published file is the one the schedule pinned.
        """
        record = build(self._entries[window], tasks=self.pins.world.tasks,
                       catalog=self.pins.world.catalog, nonce=self.chain.beacon(window),
                       revealed=sorted(self.revealed))
        return verify(record, window=window, man=self._manifest, tasks=self.pins.world.tasks)

    def _resolve_reign(self) -> None:
        """§5.5: the crown reverts to King₀ the moment the reigning hotkey stops resolving.

        The bug this closes is invisible in the burn total, which is why it survived: an unfunded or
        deregistered king is paid nothing either way, so the slate still sums to 1.0 and still burns
        the same 0.85. What moves is the PENSION. §5.7 excludes the reigning hotkey from its own
        lineage — one hotkey, one share — so a king still recorded as reigning after it has left the
        metagraph is excluded from a lineage it should be at the top of, and every surviving
        pensioner is then paid one rank too high (predecessor #1 draws 0.05 where it is owed 0.04)
        while the sixth most recent drops off the end for nothing.

        Resolution is against the LIVE metagraph, exactly as §5.7 resolves pensioners, and for the
        same reason: hotkeys are the identity, UIDs are recycled.

        The reign is dropped rather than recomputed per window because a reversion is NOT a
        coronation (§5.5): nothing is appended to the lineage, the deposed king keeps the rank-1
        slot it was dethroned into — a slot whose share then burns, since the hotkey does not
        resolve — and the throne stays vacant until a challenger clears the verdict against King₀,
        exactly as at cold start (§8b.8). From here on it is King₀ that defends, on the owner's
        allowance, because King₀ is a policy in the owner's scaffold rather than a miner's model.
        """
        if self.king is not None and self.king.hotkey not in self.chain.hotkeys().values():
            self.king = None

    def _prepare(self, sub: Submission) -> Conductor:
        """§8b.2's pre-flight, run before the king's arm: the tree, then the load.

        The `except` at the call site is broad because the two failures it must catch are of
        different kinds — `AdmissionError` from the checks, and whatever a serving stack raises on a
        load timeout or an OOM — and §8b.2 gives them one consequence: the submission is invalid,
        the shot is spent, the king is not charged. Narrowing it to `AdmissionError` would let an
        OOM crash the window instead, which spends every OTHER queued miner's shot on our failure.
        """
        admit(sub.model_tree, self.reference)
        return sub.serve()

    def _king_conductor(self) -> Conductor:
        return self.king.serve() if self.king is not None else self.pins.king_zero

    def _king_budget(self) -> float:
        return self.king.budget_usd if self.king is not None else self.owner_budget_usd

    def _arm(self, opened: Window, conductor: Conductor, budget_usd: float, guard: _GraderGuard,
             tasks: Sequence[TaskSpec], table: OutcomeTable | None = None) -> tuple[EpisodeResult, ...]:
        """One arm over the window's slice, on the COMMITTED catalog snapshot (§6.2).

        The catalog comes from the verified record rather than from this validator's live copy: that
        is the whole point of freezing it, since king and challenger evaluated minutes apart would
        otherwise face different action spaces and the duel would not be paired.
        """
        scaffold = Scaffold(opened.catalog, conductor, self.pins.world.worker, guard,
                            clock=self.clock, inspect=guard.inspecting, table=table)
        return tuple(episode.result for episode in
                     scaffold.run_window(tasks, nonce=opened.nonce, budget_usd=budget_usd))

    def _publish(self, opened: Window, power: PowerVerdict, arms: tuple[ReferenceArm, ...],
                 king_hotkey: str, king_arm: ArmScore | None,
                 king_results: tuple[EpisodeResult, ...], guard: _GraderGuard,
                 outcomes: tuple[Outcome, ...], crowned: Contender | None,
                 submissions: Mapping[str, Submission] | None = None,
                 table: OutcomeTable | None = None) -> WindowReport:
        """§8 step 5, in its required order: persist history, set weights, publish the reveal.

        Persisting first is not tidiness — re-judging is not idempotent under the one-shot rule, so
        a crash must be able to repeat the cheap half (weights, reveal) and never the half that
        spent a miner's submission.

        Weights are set even when the window scored nobody: a refused window is not owner downtime
        (§8b.6), the schedule simply does not change, and the king keeps earning while the corpus is
        re-measured next window.
        """
        if crowned is not None:
            self.coronations.append(crowned.hotkey)
            self.king = (submissions or {})[crowned.hotkey]
        if power.separates:
            self.revealed.update(task.task_id for task in opened.tasks)

        lineage = Lineage.from_coronations(self.coronations)
        metagraph = self.chain.hotkeys()
        weights = emission_weights(lineage, self.king_hotkey, metagraph)
        self.chain.set_weights(weights)

        return WindowReport(
            window=opened.epoch, nonce=opened.nonce, benchmarks=opened.benchmarks,
            n_tasks=len(opened.tasks), freshness=opened.record["freshness"], power=power,
            reference=arms, king_hotkey=king_hotkey, king=king_arm, king_results=king_results,
            graders_failed=tuple(sorted(guard.failed)), outcomes=outcomes,
            crowned=None if crowned is None else crowned.hotkey,
            pensioners=lineage.pensioners(self.king_hotkey), weights=weights, metagraph=metagraph,
            table=None if table is None else table.stats())


@dataclass
class _GraderGuard:
    """Where §8b.3's harness failure reaches the caller — and stops being scored as routing.

    A worker model erroring is an OUTCOME: it is information about that rung, `Scaffold` records it
    as a failed step, and the episode continues. A grader or sandbox failing is the opposite — a
    fact about the validator's infrastructure — and `benchmarks.base.grader` therefore catches
    nothing, so the exception arrives here, at the one place that can act on it: the task is
    remembered and dropped from BOTH arms and from the denominator (§6.3c).

    The 0.0 handed back to the episode is never scored. It exists only so that one dead sandbox does
    not abandon the other 249 tasks of an arm that has already been paid for; the task it belongs to
    leaves every arm's denominator before any verdict is computed. Scoring it as a zero instead
    would inject the owner's Docker problems into a miner's result, which is the failure this whole
    distinction exists to prevent.

    IT GUARDS THE READ CHANNEL FOR THE SAME REASON AND IT HAD TO, because a read is the other place
    a container is started (`tools.in_sandbox`) and it is the one a MINER can drive: a miner routes
    to a model they control, and that model emits `FIND <rare string>` over an eleven-repository
    tree until the container passes `TOOL_TIMEOUT_SECONDS` — an unguarded `SandboxError` there left
    `Scaffold.run_episode`, left `_arm`, and was caught only by `Validator.run`'s "one bad window
    must not end the reign", so ONE read abandoned the whole window, the money already spent on
    every arm in it, and the challenger's shot went unspent so it could do it again the next window
    and forever. Measured before it was closed (`tests/test_scaffold.py`). The accident half of
    that sentence is now measured and is SMALLER than it was written: the corpus's worst honest FIND
    is 18.4 s against a `TOOL_TIMEOUT_SECONDS` of 90, so a real read hits this clock only on a host
    or an image well outside what has been measured (`config.py`). The guard is not weakened by
    that — the deliberate stall and a dead daemon both still arrive here.

    `None` from `inspecting` is the read loop's own "this reply is the submission", so the delegate
    answers normally and the episode finishes — and the task is in `failed`, so it leaves both arms.
    That is exactly the grading path's shape: keep the arm alive, drop the task symmetrically.
    """

    grade: Callable[[TaskSpec, str], float]
    inspect: Callable[[TaskSpec, ToolCall, str], Observation] | None = None
    failed: set[str] = field(default_factory=set)
    # WHERE AN EXCLUSION IS MADE DURABLE (§8b.5). `failed` is per-WINDOW state — §6.3c drops the
    # task from both arms and from the denominator — but it lived only in this object, which the
    # daemon rebuilds on every restart. A mid-window crash therefore forgot every exclusion made
    # before it, and the 0.0 handed back below, which this docstring says is "never scored", became
    # a scored zero: the owner's dead sandbox recorded as a miner's miss, in whichever arm had
    # already run. `on_exclude` is the seam the daemon persists through; None keeps the offline
    # simulation and the dev kit exactly as they were.
    on_exclude: Callable[[str], None] | None = None

    def _exclude(self, task_id: str) -> None:
        self.failed.add(task_id)
        if self.on_exclude is not None:
            self.on_exclude(task_id)

    def __call__(self, task: TaskSpec, answer: str) -> float:
        try:
            return self.grade(task, answer)
        except Exception:  # noqa: BLE001 — a dead sandbox is the validator's, not the miner's
            self._exclude(task.task_id)
            return 0.0

    def inspecting(self, task: TaskSpec, call: ToolCall,
                   response_sha256: str) -> Observation | None:
        """`Scaffold.inspect`, guarded. None means the read did not happen and the task is dropped."""
        try:
            return None if self.inspect is None else self.inspect(task, call, response_sha256)
        except Exception:  # noqa: BLE001 — the same rule, at the other container
            self._exclude(task.task_id)
            return None


def _groups(rows: Sequence[tuple[EpisodeResult, EpisodeResult]],
            group_of: Mapping[str, str | None]) -> tuple[str, ...] | None:
    """One benchmark's per-task cluster labels, or None where the benchmark has no grouping.

    `duel` resamples whole groups, and None is its every-task-its-own-cluster default — the resample
    that shipped. So a benchmark whose adapter supplies no `TaskSpec.group` lands there unchanged,
    and only a benchmark that has a real grouping (R2E-Gym's repository) gets a clustered interval.
    A benchmark labelled on only some of its tasks is refused by `BenchmarkPair` rather than
    half-clustered, which would be the too-narrow interval arriving by the back door.
    """
    labels = tuple(group_of[king.task_id] for king, _ in rows)
    return None if all(label is None for label in labels) else labels


def _pairs(paired: Paired, exchange: Mapping[str, Exchange],
           tasks: Sequence[TaskSpec]) -> tuple[BenchmarkPair, ...]:
    """Row-aligned arms -> the duel's per-benchmark pairs, over the benchmarks §5.1b ADMITS.

    Alignment is not re-established here: `window.exclude` already returns both arms in the king's
    order over the identical task list, and `BenchmarkPair` pairs positionally. Grouping by
    benchmark is all that is left, in sorted order for `duel`'s reason — a mean of means depends on
    summation order in its last bits, so the verdict must be a function of the admitted set rather
    than of the order tasks happened to land in.

    `tasks` is the window's slice, and it is here for one field: `TaskSpec.group`, the duel's
    resample unit. An `EpisodeResult` carries what an arm DID and the cluster is a fact about the
    task, so it is read from the pinned slice rather than copied into every episode — where two
    copies could disagree about which repository a row came from.

    A benchmark whose §5.1b guard fired is dropped BEFORE the duel sees it, because §5.1b excludes
    it from scoring outright: its λ is not a price, so `score_arm` leaves it out of the published
    arm, and a duel run over the wider set would price a crown on it and would run `loo` and the
    median over a set the published arm does not have. Dropping it here rather than inside `duel`
    keeps one definition of the admitted set — `score_arm`'s — which is the same reason `priced`
    calls `score_arm` instead of re-typing §5.1b.

    A benchmark that is simply MISSING from the table is not dropped: it is passed through so that
    `score_arm` refuses it by name (§5.1b's forbidden silent fallback), which a guard filter that
    swallowed unknown benchmarks would turn into exactly that fallback.
    """
    group_of = {task.task_id: task.group for task in tasks}
    rows: dict[str, list[tuple[EpisodeResult, EpisodeResult]]] = {}
    for king, challenger in zip(paired.king, paired.challenger):
        rows.setdefault(king.benchmark, []).append((king, challenger))
    return tuple(
        BenchmarkPair(benchmark,
                      np.array([k.graded_score for k, _ in pairs], dtype=float),
                      np.array([k.spend_usd for k, _ in pairs], dtype=float),
                      np.array([c.graded_score for _, c in pairs], dtype=float),
                      np.array([c.spend_usd for _, c in pairs], dtype=float),
                      _groups(pairs, group_of))
        for benchmark, pairs in sorted(rows.items())
        if benchmark not in exchange or not exchange[benchmark].flags)


def _exhausted(results: Sequence[EpisodeResult]) -> int:
    """Tasks whose arm ran out of allowance before reaching them (§4) — §5.8's published number."""
    return sum(1 for result in results if result.stopped_reason == "budget_exhausted")


def _skipped(sub: Submission) -> Outcome:
    """Never re-judged: one hotkey, one submission, ever (§7).

    Reported rather than dropped, so a returning miner sees why nothing happened — and nothing is
    spent a second time, because there is nothing left to spend.
    """
    return Outcome(sub.hotkey, sub.commit_block, SKIPPED,
                   "this hotkey has already been judged; one shot per hotkey (§7)")


def _why(verdict: DuelVerdict) -> str:
    """Which of §5.2's conditions decided this duel — derived from the verdict, never stored.

    A challenger that loses on breadth and one that loses on the margin need different work, and
    "you lost" tells them nothing about which. The four clauses are the three conditions, in the
    order the spec states them, with the margin's two halves separated.

    The median clause is not decoration: §5.2's condition 3 refuses a challenger that leave-one-out
    lets through (Phase 1 measured the pair — a win of `+0.315` on two benchmarks of twelve and
    nothing on the other ten clears LOO at `+0.0286` and has a median of exactly zero). Without it,
    a challenger refused by the median alone is handed an EMPTY string, which is the one refusal in
    this module that would tell a miner nothing whatsoever.
    """
    if verdict.challenger_wins:
        return "clears eps, the bootstrap lower bound, breadth and the median"
    refusals = []
    if verdict.delta <= verdict.eps:
        refusals.append(f"delta {verdict.delta:+.4f} does not clear eps {verdict.eps:.2f}")
    if verdict.lcb <= 0.0:
        refusals.append(f"lower bound {verdict.lcb:+.4f} is not separable from zero")
    if verdict.loo_min <= 0.0:
        refusals.append(f"the win does not survive dropping {verdict.loo_dropped}")
    if verdict.median <= 0.0:
        refusals.append(f"the median benchmark's delta {verdict.median:+.4f} is not positive: this "
                        "win lives in a minority of the corpus")
    return "; ".join(refusals)


# --- the report -----------------------------------------------------------------------------------


def format_window(report: WindowReport) -> str:
    """The window's reveal as text: what was measured, who was judged, and who gets paid.

    Everything published here is either the gate's own sentence, a per-benchmark row, or a verdict
    field — no number is recomputed for printing, so the report cannot disagree with the record.
    """
    return "\n".join([*_header(report), "",
                      *_king_section(report), "",
                      *_queue_section(report), "",
                      *_emission_section(report)])


def _header(report: WindowReport) -> list[str]:
    stale = report.freshness.get("staleness")
    lines = [f"# WINDOW {report.window} — nonce {report.nonce}",
             f"slice: {report.n_tasks} tasks over {len(report.benchmarks)} benchmarks "
             f"({', '.join(report.benchmarks)})",
             f"already scored in an earlier window: "
             f"{'n/a' if stale is None else f'{stale:.1%}'} (D12: tasks recur by design)",
             f"reference arms: {', '.join(arm.name for arm in report.reference)}",
             f"POWER GATE — {report.power.reason}",
             f"grader/sandbox failures excluded from BOTH arms: {len(report.graders_failed)}"]
    if report.table is not None:
        t = report.table
        lines.append(
            f"worker delegates: {t['fills'] + t['hits']} — {t['fills']} bought live, {t['hits']} "
            f"replayed from the window's outcome table (§5.2c, D17; before it every one was "
            f"bought); stored provider failures {t['failed_rows']} of {t['rows']} rows "
            f"({t['failure_rate']:.1%})")
    return lines


def _king_section(report: WindowReport) -> list[str]:
    who = report.king_hotkey or "King0 (best fixed policy — its 0.85 burns)"
    if report.king is None:
        return [f"# KING ARM — {who}",
                "not run: no challenger passed admission, so the king was charged nothing (§8b.2)"]
    return [f"# KING ARM — {who}. Computed ONCE and reused by every duel below (§5.2a, D13).",
            f"tasks the king's allowance never reached: {report.king_exhausted} (§5.8)",
            *_score_rows(report.king)]


def _score_rows(arm: ArmScore) -> list[str]:
    rows = [f"  {'benchmark':<18}{'n':>4}{'quality':>10}{'$/task':>12}{'final':>10}"]
    rows += [f"  {row.benchmark:<18}{row.n_tasks:>4}{row.quality:>10.4f}"
             f"{row.spend:>12.6f}{row.final:>10.4f}" for row in arm.per_benchmark]
    # Equal weight per BENCHMARK, never per task (§5.1) — the ALL row is the mean of the rows above
    # it, which is the number the duel runs on.
    rows.append(f"  {'ALL':<18}{sum(r.n_tasks for r in arm.per_benchmark):>4}"
                f"{arm.quality:>10.4f}{'':>12}{arm.final:>10.4f}")
    return rows


def _queue_section(report: WindowReport) -> list[str]:
    lines = ["# QUEUE — evaluated in (commit_block, hotkey) order (§8b.1)"]
    for outcome in report.outcomes:
        lines.append(f"  block {outcome.commit_block:>6}  {outcome.hotkey:<14} "
                     f"{outcome.status.upper():<9} shot_spent={outcome.shot_spent}")
        if outcome.verdict is None:
            lines.append(f"    {outcome.detail}")
            continue
        lines.append(f"    tasks its allowance never reached: {outcome.exhausted} "
                     f"(the king's: {report.king_exhausted}) — §5.8")
        verdict = outcome.verdict
        lines.append(
            f"    delta {verdict.delta:+.4f} (eps {verdict.eps:.2f})  lcb {verdict.lcb:+.4f}  "
            f"breadth {verdict.loo_min:+.4f} (worst without {verdict.loo_dropped})  "
            # §5.2's third condition, printed as a condition and not as a diagnostic: it refuses
            # wins leave-one-out lets through, so a reveal that omitted it would show three numbers
            # none of which explains the verdict.
            f"median {verdict.median:+.4f}  "
            # Published beside the rules that DID gate, never as a rule: a sign test over N noisy
            # bits is weak in both directions, and ≥7 of 12 happens by coin flip 39% of the time
            # (§5.3). Printing both is what makes a disagreement between them visible.
            f"sign {verdict.sign_wins}/{len(verdict.benchmarks)} p={verdict.sign_p:.3f}")
        lines.append(f"    -> {'WINS' if outcome.won else 'REFUSED'}: {outcome.detail}")
        lines.append("    per-benchmark delta: " +
                     "  ".join(f"{name} {value:+.3f}" for name, value in verdict.by_benchmark))
    crowned = report.crowned or "nobody — the crown does not move"
    lines.append(f"# CORONATION — {crowned}"
                 + ("" if report.crowned is None else
                    " (largest delta among the winners; ties break on earliest commit block)"))
    return lines


def _emission_section(report: WindowReport) -> list[str]:
    king_uid = {hotkey: uid for uid, hotkey in report.metagraph.items()}.get(
        report.crowned or report.king_hotkey)
    pension_uid = {hotkey: rank for rank, hotkey in enumerate(report.pensioners, 1)}
    lines = ["# EMISSIONS — king 0.85, a five-deep decaying pension, unfilled slots BURN (§5.7)"]
    for uid, share in sorted(report.weights.items()):
        hotkey = report.metagraph.get(uid, "")
        if uid == king_uid:
            role = "reigning king"
        elif hotkey in pension_uid:
            role = f"ex-king, {pension_uid[hotkey]} back"
        else:
            role = "BURN — never reflowed to the crown, which is what the pension exists to prevent"
        lines.append(f"  uid {uid:<4} {hotkey or '-':<14} {share:.4f}  {role}")
    return lines


# --- the offline arena ----------------------------------------------------------------------------
# A world in which routing skill EXISTS and is not the obvious thing, because a demonstration in a
# world where the cheapest model wins would demonstrate nothing about the mechanism. Prices here are
# deliberately NOT a capability ranking: `mock/premium` is the most expensive model in the catalog
# and only the second strongest, so `reference._by_price` — which is honest that price is a proxy —
# builds a cascade that climbs cheap -> heavy -> premium and never tries `mock/mid`. `mock/mid` is
# strong enough for everything and costs half of `mock/heavy`, so the entire capturable band is
# "know that the middle rung is enough", which no fixed policy in the catalog's own price order can
# reach. That is what this repository's measurements say a real pool would have to look like for the
# mechanism to have anything to buy (ROUTING_MEASUREMENTS: the blocker is the model market).
_CATALOG = Catalog(entries=(
    CatalogEntry("mock/cheap", 0.20, 0.80, 128_000),
    CatalogEntry("mock/mid", 0.40, 1.60, 200_000),
    CatalogEntry("mock/heavy", 1.00, 4.00, 300_000),
    CatalogEntry("mock/premium", 1.20, 4.80, 400_000),
))
_STRENGTH = {"mock/cheap": 1, "mock/mid": 3, "mock/heavy": 3, "mock/premium": 2}
_COST = {"mock/cheap": 0.04, "mock/mid": 0.08, "mock/heavy": 0.16, "mock/premium": 0.20}

# Three benchmarks the price ladder cannot price — one where the cheapest model already suffices and
# two that the priciest model cannot do at all — so on each of them `always_strongest` buys no
# quality over `always_cheapest`, the DOMINATOR guard fires and §5.1b EXCLUDES them from scoring.
# That is this repository's own five-times-measured condition (ROUTING_MEASUREMENTS), kept in the
# cast so the exclusion is exercised and named in the reveal rather than only described. The other
# five are where the middle rung is the whole prize, and they are the corpus every duel below is
# actually decided on. `automation-bench` grades out of four subgoals so partial credit is exercised.
_BENCHMARKS = (
    MockBenchmark("terminal-bench", _STRENGTH, n_tasks=8, difficulty=(1, 1)),
    MockBenchmark("frontier-swe", _STRENGTH, n_tasks=8, difficulty=(3, 3)),
    MockBenchmark("cybergym", _STRENGTH, n_tasks=8, difficulty=(3, 3)),
    MockBenchmark("nl2repo", _STRENGTH, n_tasks=8, difficulty=(2, 2)),
    MockBenchmark("programbench", _STRENGTH, n_tasks=8, difficulty=(2, 2)),
    MockBenchmark("exploitbench", _STRENGTH, n_tasks=8, difficulty=(2, 2)),
    MockBenchmark("swe-marathon", _STRENGTH, n_tasks=8, difficulty=(2, 2)),
    MockBenchmark("automation-bench", _STRENGTH, n_tasks=8, difficulty=(2, 2), subgoals=4),
)

_LEARNABLE = ("nl2repo", "programbench", "exploitbench", "swe-marathon", "automation-bench")


def learned_router(fallback: Conductor, benchmarks: Sequence[str], *rungs: str) -> Conductor:
    """The cast's stand-in for a trained Conductor: it knows which ladder to run on `benchmarks`.

    Stacked out of `archetypes.Specialist` — the shipped lookup-table adversary — one benchmark at a
    time, rather than written as a second benchmark-conditional policy with a second copy of the
    render coupling. A router that has learned N benchmarks IS N specialisations over the free
    cascade, and that is also the honest description of what a trained Conductor would be doing
    here: measurement 16 found every positive capture ever measured in this repository to be a
    router recognising the ask's CATEGORY, and the category is written in the prompt.

    `rungs` is a ladder rather than a single model because under §5.2's third condition the axis a
    cast can show progress along is NOT how many benchmarks a router has memorised: a router that
    covers k of N and ties on the rest has a median delta of zero for every k below half the corpus,
    and two successive coronations by coverage alone are arithmetically impossible (a king already
    holding a majority leaves a minority for its successor to improve on). What a router can be
    progressively better at is reaching the same quality for less money — which is what §5.1b prices
    and the only thing it pays for.

    What separates it from the specialist §5.2's breadth rule refuses is still how many rows it has
    learned — that is the axis leave-one-out grades on, and why one row is not enough.
    """
    if not rungs:
        raise ValueError("a learned router needs at least one rung to route to")
    policy = fallback
    for benchmark in benchmarks:
        ladder = FixedPolicy(f"{'+'.join(rungs)}-on-{benchmark}", rungs)
        policy = Specialist(benchmark, ladder, policy)
    return policy


def mock_arena(root: Path) -> tuple[Validator, dict[int, list[Submission]]]:
    """The offline cast: a validator, and who challenges in each window.

    Three routers that have learned progressively more of the pool's structure, a copy of King₀ that
    must tie and lose (D14 prices derivatives rather than detecting them), and a broken artifact
    whose shot is spent and which is skipped when it tries again. `root` holds the model trees the
    real `admission.admit` runs over — a few hundred bytes each, no weights.

    "MORE OF THE POOL'S STRUCTURE" IS CHEAPER ROUTING, NOT WIDER COVERAGE, AND §5.2's THIRD
    CONDITION IS WHY. All three routers cover the whole admitted corpus and differ only in the
    ladder they run on it: `mock/heavy` (knows to escalate, picks the wrong rung), cheap-then-mid
    (finds the right rung, pays for one failed call first), and `mock/mid` outright (the whole
    capturable band). Each is better than the last on EVERY admitted benchmark, which is what a
    median-gated verdict now requires of a successor — an earlier cast whose routers covered 2, 3
    and 5 benchmarks of the five would today be refused at the first two, and the chain of two
    coronations it exists to show could not happen at any coverage (`learned_router`). The premium
    rung is deliberately absent: it IS `always_strongest`, so §5.1b's fit makes it tie King₀ exactly.
    """
    worker = MockWorker(answers=worker_answers(_STRENGTH), costs=_COST)
    check_channel(_BENCHMARKS)               # the read channel's halves agree, before anything runs
    tasks = tuple(task for benchmark in _BENCHMARKS for task in benchmark.load())
    pins = pin_corpus(_CATALOG, tasks, suite_grade(_BENCHMARKS), worker)

    valid, broken = _write_tree(root / "valid"), _write_tree(root / "broken", pickled=True)
    chain = MockChain()
    chain.register("burn")                    # uid 0, the burn address (§5.7)
    validator = Validator(pins=pins, reference=describe(valid), chain=chain, windows=(1, 2, 3),
                          per_benchmark=5, minimum=3)

    def router(*rungs: str) -> Conductor:
        return learned_router(pins.king_zero, _LEARNABLE, *rungs)

    def submission(hotkey: str, block: int, conductor: Conductor,
                   tree: Path = valid) -> Submission:
        chain.register(hotkey)
        return Submission(hotkey=hotkey, commit_block=block, model_tree=tree,
                          serve=lambda: conductor, budget_usd=100.0)

    broken_sub = submission("dud", 3, pins.king_zero, tree=broken)
    return validator, {
        1: [submission("router-a", 10, router("mock/heavy")),
            submission("router-b", 11, router("mock/cheap", "mock/mid")),
            # A DIFFERENT artifact that behaves identically to the king (D14): it ties on every
            # task, so it ties on every resample, and a tie keeps the crown where it is.
            submission("copycat", 12, Copier(pins.king_zero)),
            broken_sub],
        2: [broken_sub],                      # one shot per hotkey: skipped, not re-judged
        3: [submission("router-c", 30, router("mock/mid"))],
    }


def _write_tree(root: Path, *, pickled: bool = False) -> Path:
    """A minimal model tree for `admission.admit`: a config, one safetensors header, a tokenizer.

    No tensor data is written because none is ever read — admission parses headers only, which is
    what lets it admit a 70 GB tree for a few kilobytes of I/O. `pickled=True` adds the one file
    §1.1 refuses outright, which is how the cast gets an invalid artifact without inventing a
    failure mode the real gate does not have.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
         "num_hidden_layers": 4, "hidden_size": 64, "vocab_size": 128}))
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [128, 64],
                                                       "data_offsets": [0, 16384]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (root / "tokenizer.json").write_text('{"version": "1.0"}')
    (root / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    if pickled:
        (root / "pytorch_model.bin").write_bytes(b"not really a pickle, and never opened")
    return root


def run_simulation(root: Path, *, verbose: bool = True) -> list[WindowReport]:
    """Three windows: a batch coronation, a window that costs the king nothing, and a pension."""
    validator, queue = mock_arena(root)
    reports = []
    for window in validator.windows:
        report = validator.run_window(window, queue.get(window, ()))
        reports.append(report)
        if verbose:
            print(format_window(report))
            print()
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="orchestra-sim",
        description="Run the whole v3 mechanism offline: a committed window, the power gate, one "
                    "king arm shared by every duel, the three-condition verdict, batch coronation "
                    "and the emission schedule. No network, no key, no GPU, no chain.")
    parser.add_argument("--root", type=Path, metavar="DIR",
                        help="where to write the cast's model trees (default: a temp dir)")
    args = parser.parse_args(argv)
    if args.root:
        run_simulation(args.root)
        return
    with tempfile.TemporaryDirectory(prefix="v3-sim-") as tmp:
        run_simulation(Path(tmp))


if __name__ == "__main__":
    main()
