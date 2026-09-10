"""The fixed reference policies, King₀, and the power gate (§5.5, §5.6; the build plan M8 exit 11).

THIS MODULE IS WHAT MAKES THE $100 PRE-LAUNCH BUDGET DEFENSIBLE. An earlier draft of the plan bought
the band with a $1,300–2,500 upfront experiment. The cheap path replaces that with a measurement
that runs *inside* the mechanism: two fixed policies on a subsample of every window's slice, priced
at ordinary operating cost. the build plan §0 is explicit that this makes the rule load-bearing rather
than nice-to-have — "neither may be dropped for expedience" — because without it the subnet is
launched not knowing whether a capturable prize exists and with nothing that would notice.

KING₀ IS THE BEST FIXED POLICY, NOT THE UNTRAINED REFERENCE MODEL (§5.5). The tempting baseline is
"the base model, untrained" — what zero miner contribution buys. It is the wrong bar, because **the
real alternative to running this subnet is shipping a fixed cascade**: it needs no miner, no
training, no arena and no emissions. If a trained Conductor cannot beat one, the subnet is paying
for something the owner could deploy for free. So King₀ is whichever of the fixed policies here
scores best under §5.1b's own rule (`king_zero`), and which one that is, is a v3 measurement rather
than an inheritance from this repo's single-turn experiments.

WHAT THE OLD POWER GATE GUARANTEED, AND HOW THE REPLACEMENT KEEPS IT. `koth/arena.py::power_of` (removed with v2, 2026-09-07)
built a synthetic challenger that plays the per-ask best action some fraction of the time and duels
it with the real machinery. v3 deletes the precomputed outcome matrix that made a per-ask best
action knowable (§9), so it cannot run — but three of its properties are not optional and are
preserved here:

  * **Band alone is the wrong question.** Measured on that build, a 20-ask slice with a band of
    +0.1500 — seven times the scalar floor it replaced — had power 0.00 to detect a challenger
    capturing half of it. So the gate is a *statistical separation* test run with the duel's own
    bootstrap (`duel.duel`), not a magnitude threshold. It keeps a magnitude floor as well, for the
    complementary reason below, and both must clear.
  * **Power is a property of the DATA, so it is never asked about the incumbent.** Probing against
    the sitting king conflates "this slice is too small to resolve anything" (a real problem) with
    "the king is already excellent" (a success), and a near-optimal king would fail the probe and
    stall the arena forever — found exactly that way, by a window-2 test where the incumbent already
    captured ~1.05 of the band. Both arms here are FIXED policies; the gate never touches the crown.
  * **A failed window touches nothing.** No hotkey is spent, no artifact consumed, no duel scored;
    challengers stay queued and roll over (§8 step 2). That is the whole reason the gate can be
    conservative: refusing a live window costs a delay, while scoring a dead one spends a miner's
    single submission on noise.

WHY THE MAGNITUDE FLOOR IS THERE TOO. This repository has already paid to learn the cost of skipping
it: ROUTING_MEASUREMENTS §15 ran an epoch whose oracle beat best-single by +0.0078 and the mechanism
paid out anyway, because nothing asked whether the epoch was worth running. A spread that small is
perfectly separable from zero given enough tasks — a purely statistical gate waves it through — and
it still cannot host a coronation, because a challenger has to clear `eps` (§5.2b). So the floor is
tied to `EPS` rather than chosen: below it, whatever this window crowns is noise.

THE SPREAD IS MEASURED ON QUALITY, NOT ON `final`. This is the subtlety that would quietly invert
the gate. §5.1b fits `λ_b` precisely so that always-cheapest and always-strongest score *the same* —
the score's iso-lines run parallel to the pool's own quality/dollars line, which is what stops the
subnet becoming a frugality contest or a spending contest (M8 exit 3). A gate run on `final` would
therefore ask whether two policies the scoring rule is *designed to tie* nevertheless differ, and a
window where paying more buys a great deal of quality — the richest possible signal — would refuse
itself. The gate's question is not the duel's. It is not "is this policy better?" but **"does which
policy you run change what comes back at all?"**, and that is a question about the graded outcomes.
It also means the gate needs no `λ_b` table, so it works in window 1, before λ is re-derived from
production data (§5.1b).

A SUBSAMPLE, NOT A FULL ARM (§5.6). Detecting *gross signal failure* needs far less data than
ranking two close competitors: ~60 tasks per reference arm against ~250 for a duel. Run **once per
window and shared by every duel in it**, exactly as the king's arm is (§5.2a) — nothing in this
module re-runs it per challenger.

The reference arms' traces are published every window (D15) as the entry ramp for miners. They carry
no REASON text, and that is deliberate: a fixed policy has no reasoning to record, and inventing
plausible-looking reasoning would put owner-written prose into the corpus miners imitate.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .config import EPS, N_BOOT
from .duel import BenchmarkPair, duel
from .scaffold import task_order
from .score import Exchange, derive_exchange, pinned_cost, raw_quality, score_arm
from .types import Catalog, CatalogEntry, EpisodeResult, TaskSpec

ALWAYS_CHEAPEST = "always-cheapest"
ALWAYS_STRONGEST = "always-strongest"
RANDOM_OVER_CATALOG = "random-over-catalog"
CASCADE = "cascade"

# Rungs in the fixed verify-and-escalate cascade — cheapest, a middle tier, most expensive. Two
# rungs is a plain escalation; the middle one is what makes it a cascade, and it is where the
# +0.0982 that a prior single-turn measurement saw came from. More rungs buy little and cost a great
# deal: each additional rung is another full episode's spend on a task the cheap rungs already
# failed, on every task they fail. `MAX_STEPS = 12` leaves ample room for three.
CASCADE_RUNGS = 3

# Tasks per reference arm (§5.6). A duel needs ~250 to rank two close competitors (§5.4); detecting
# that a window cannot separate anyone at all is a far coarser question and ~60 answers it. Two arms
# at 60 against N duels at 250 amortises to a small fraction of window cost, and falls further the
# more challengers a window carries.
POWER_SUBSAMPLE = 60

# The measured spread below which the window is refused, on the quality scale.
#
# TIED TO `EPS` ON PURPOSE, not chosen: a challenger must beat the king by `eps` to be crowned
# (§5.2, D10), so a window in which two *fixed* policies — the whole range the pool gives away for
# free — differ by less than that cannot produce a coronation that is anything but noise. This is
# the measurement-15 condition, and stating it in terms of an existing pin is what stops it becoming
# one more number somebody guessed.
#
# UNLIKE `EPS` ITSELF, THIS IS AN ORDINARY GOVERNANCE KNOB and may move in either direction at a
# window boundary: it prices no miner's submission and nobody trains against it. If production
# windows refuse persistently while the corpus is demonstrably alive, the owner lowers it and
# records why. Erring toward refusal is the safe direction — a refused window costs a delay and
# spends no hotkey (§8 step 2), a scored dead window spends a miner's one shot on a coin flip.
POWER_SPREAD_FLOOR = EPS

# --- reading the rendered state -------------------------------------------------------------------
# A fixed policy sees exactly what a miner's model sees: the rendered prompt, and nothing else
# (`conductor.Conductor`). That is a requirement rather than an inconvenience — the reference arms
# are the baseline every duel is judged against, so they must run through the identical scaffold, or
# the comparison is between two harnesses rather than two policies.
#
# So the ladder position is read back out of the render. Two rules make that robust rather than
# fragile. The heading is taken from the RIGHT (`rpartition`): the task statement is untrusted
# benchmark text that appears earlier in the prompt and could contain anything, while the history
# section is last and its rows cannot contain the heading — they are formatted from a step record
# whose fields are a catalog slug, a verb and a float. And the step index is read from the
# render's own explicit sentence rather than by counting rows, so a row format change cannot shift a
# reference arm's rung silently.
#
# These three literals mirror `render._history`. The coupling is pinned by a test that renders real
# states and asserts a policy reads them back, so a template change fails there rather than quietly
# mis-stepping King₀.
_HISTORY_HEADING = "\n# HISTORY\n"
_STEP_INDEX = re.compile(r"^this is step (\d+) of at most ", re.MULTILINE)
_SUCCEEDED = "-> succeeded"


def _progress(prompt: str) -> tuple[int, bool]:
    """`(1-based index of the step about to be taken, has any step solved the task)`.

    Both refusals are loud. A policy that quietly defaulted to step 1 would delegate to its cheapest
    rung forever and still look like a working cascade in the reveal.
    """
    _, heading, history = prompt.rpartition(_HISTORY_HEADING)
    if not heading:
        raise ValueError("the rendered state carries no history section: a fixed policy cannot "
                         "know its position on the ladder, and guessing one would silently score a "
                         "policy nobody chose")
    match = _STEP_INDEX.search(history)
    if match is None:
        raise ValueError("the rendered history does not state the step index: `render._history` "
                         "and this module have drifted apart")
    return int(match.group(1)), _SUCCEEDED in history


# --- the fixed policies ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedPolicy:
    """A ladder of model IDs, climbed until one solves the task (§5.5, §6.1).

    All three non-random baselines are this shape. A ONE-RUNG LADDER IS EXACTLY AN ALWAYS-X POLICY:
    it delegates to its single model and stops, because the ladder is exhausted. That unification is
    not tidiness — it means always-cheapest, always-strongest and the cascade differ only in their
    rungs, so a duel between them compares the choice of models and not the shape of the harness.

    IT ESCALATES ON ANYTHING SHORT OF A SOLVE. `render` reports a step as succeeded iff the graded
    score reached 1.0 (`scaffold.run_episode`), so partial credit escalates. That is the same call
    the scaffold already made: §5.1's partial credit is a variance-reduction device for the duel,
    not a report that the work is done, and §3's argument for a sequential policy is that the
    Conductor *observes that a model failed*.
    """

    name: str
    rungs: tuple[str, ...]

    def act(self, prompt: str) -> str:
        step, solved = _progress(prompt)
        if solved or step > len(self.rungs):
            return "STOP"
        # DELEGATE opens the episode, RETRY follows an observed outcome (§2). The distinction is
        # recorded in the published traces, so a miner training on them sees escalation as a
        # different act from an opening choice.
        return f"{'DELEGATE' if step == 1 else 'RETRY'} {self.rungs[step - 1]}"


@dataclass(frozen=True)
class RandomOverCatalog:
    """§6.1's third admission policy: one model per episode, drawn from the whole catalog.

    The no-information baseline. It is what "the action space is wide" buys with no policy at all,
    and a benchmark where it matches the considered policies is a benchmark where the scaffold does
    all the work (§6.1 point 3).

    THE DRAW IS REPRODUCIBLE FROM THE PUBLISHED REVEAL, not merely from a seed the validator
    remembers. It is keyed by the window nonce and by `sha256(prompt)` — which is exactly the value
    the trace already records as `StepRecord.rendered_state_digest` — so anyone holding the reveal
    can recompute every model this arm chose. A policy nobody can re-derive is a baseline nobody can
    check.
    """

    catalog: Catalog
    nonce: str
    name: str = RANDOM_OVER_CATALOG

    def act(self, prompt: str) -> str:
        step, solved = _progress(prompt)
        if solved or step > 1:
            return "STOP"
        return f"DELEGATE {self._pick(prompt)}"

    def _pick(self, prompt: str) -> str:
        state = hashlib.sha256(prompt.encode()).hexdigest()
        draw = hashlib.sha256(f"{self.nonce}|{state}".encode()).hexdigest()
        # `entries` is a tuple and its order is part of the committed snapshot (`types.Catalog`);
        # indexing `ids` instead would be the `scoreable_ids` bug, and the arm would choose
        # different models in different processes.
        return self.catalog.entries[int(draw[:8], 16) % len(self.catalog.entries)].model_id


def always_cheapest(catalog: Catalog) -> FixedPolicy:
    """The floor, and King₀'s natural analogue (the build plan M3a)."""
    return FixedPolicy(ALWAYS_CHEAPEST, (_by_price(catalog)[0].model_id,))


def always_strongest(catalog: Catalog) -> FixedPolicy:
    """Does paying more help at all — the contrast when King₀ is always-cheapest (§5.6).

    It is NOT the contrast for a cascade, whose top rung is this same model; see `contrast_for`.
    """
    return FixedPolicy(ALWAYS_STRONGEST, (_by_price(catalog)[-1].model_id,))


def cascade(catalog: Catalog, *, rungs: int = CASCADE_RUNGS) -> FixedPolicy:
    """Cheap first, stronger on failure — **the free win that needs no miner** (§5.5, M3a).

    This is the policy the whole subnet has to beat. It needs no training, no arena and no
    emissions, so a Conductor that cannot outscore it is asking to be paid for work the owner could
    deploy for nothing.
    """
    if rungs < 2:
        raise ValueError(f"a cascade needs at least two rungs to escalate, not {rungs}")
    ordered = _by_price(catalog)
    picks = [ordered[round(i * (len(ordered) - 1) / (rungs - 1))].model_id for i in range(rungs)]
    # Deduplicated in order, for a catalog with fewer models than rungs. `dict.fromkeys` over a list
    # is insertion-ordered and fully deterministic — this is not the set-iteration hazard.
    return FixedPolicy(CASCADE, tuple(dict.fromkeys(picks)))


def _by_price(catalog: Catalog) -> tuple[CatalogEntry, ...]:
    """The catalog cheapest-first, by the sum of its two token prices.

    PRICE IS THE ONLY CAPABILITY SIGNAL A COMMITTED SNAPSHOT CARRIES — it holds `id / $in / $out /
    ctx` and nothing else (§3) — so "strongest" here means "most expensive", and that is a proxy,
    stated as one. This repository's central finding is that the proxy fails: the cheapest model
    dominating on quality-per-dollar is what closed the routing thesis five times over
    (ROUTING_MEASUREMENTS), and `score.derive_exchange` flags exactly that condition. Nothing here
    reads the ordering as a quality ranking; it is used only to construct a *contrast* whose spread
    the gate then measures. If the pool has a dominator, the measured spread says so.

    Input and output prices are summed rather than blended by a weight, because any monotone blend
    picks the same two extremes in practice and a weight would be one more constant nobody measured.
    Ties break on model ID, so the ordering is total and the same in every process.
    """
    if not catalog.entries:
        raise ValueError("an empty catalog is not an action space: no fixed policy can be built")
    return tuple(sorted(catalog.entries,
                        key=lambda e: (e.price_in_per_mtok + e.price_out_per_mtok, e.model_id)))


# --- the window's reference arms ------------------------------------------------------------------


def subsample(tasks: Sequence[TaskSpec], *, nonce: str,
              size: int = POWER_SUBSAMPLE) -> tuple[TaskSpec, ...]:
    """The slice the reference arms run: `size` tasks, nonce-derived and stratified (§5.6, §6.3b).

    STRATIFIED, NOT A FLAT DRAW. The aggregate weights benchmarks equally (§5.1), and the gate's
    bootstrap resamples within each benchmark (`duel.duel`), so a benchmark missing from the
    subsample would be silently reweighted to zero and its per-benchmark spread — which is what
    M2b's continuous admission reads to drop a benchmark that carries no signal — would not exist.
    Round-robin over the benchmarks gives near-equal counts; when `size` is not a multiple of
    the benchmark count the remainder goes to the alphabetically first, which is arbitrary but
    deterministic.

    Ordering within a benchmark is `scaffold.task_order`, so the draw depends only on *which* tasks
    the slice holds and never on the order a caller listed them in — a seeded shuffle would agree
    between two callers only for as long as everything upstream also agreed, which is the
    `scoreable_ids` failure mode wearing a new hat.
    """
    if not tasks:
        raise ValueError("no tasks to subsample: a window with an empty slice cannot be gated")
    groups: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        groups.setdefault(task.benchmark, []).append(task)
    ordered = [task_order(group, nonce) for _, group in sorted(groups.items())]

    picked: list[TaskSpec] = []
    for depth in range(max(len(group) for group in ordered)):
        for group in ordered:
            if depth < len(group):
                picked.append(group[depth])
                if len(picked) == size:
                    return task_order(picked, nonce)
    return task_order(picked, nonce)


@dataclass(frozen=True)
class ReferenceArm:
    """One fixed policy's episodes over the power subsample, named so the reveal can say which."""

    name: str
    results: tuple[EpisodeResult, ...]


def king_zero(arms: Sequence[ReferenceArm]) -> ReferenceArm:
    """**King₀: the best fixed policy** (§5.5) — the one with the highest quality.

    ON QUALITY, NOT ON `final`, AND THE REASON IS CIRCULARITY. λ is fitted to make King₀ and the
    floor tie in `final` (`fit_pool`), so a King₀ chosen under the fitted table would be choosing
    between two arms the table was built to make indistinguishable. Quality is what exists before λ
    does. The older worry — that a King₀ chosen on quality sets the bar at "outspend me" — is
    answered by the fit rather than by the pick: whatever policy sits on the throne, λ prices spend at
    that policy's own rate, so beating it means buying quality more cheaply than it does, never
    merely spending more (`test_reference`).

    Ties on quality break toward the LOWER spend, then on name. Equal quality at higher spend is the
    pool buying nothing with money, and the cheaper arm is the honest bar for it; and a cascade
    starts on the cheapest rung, so at equal quality it has strictly more spend and always-cheapest
    wins — which is why King₀ is the cascade only when the cascade is strictly better
    (`test_invariant2`). This selection runs once, at the M3a admission gate, and its answer is
    then a published pin (§5.5: "whichever that turns out to be").
    """
    if not arms:
        raise ValueError("King₀ is the best of the fixed policies; none were run")
    return min(arms, key=lambda arm: (-raw_quality(arm.results),
                                      sum(r.spend_usd for r in arm.results), arm.name))


@dataclass(frozen=True)
class PoolFit:
    """What the §6.1 sweep pins: King₀, the pair λ is fitted between, and the table itself."""

    king_zero: str
    floor: str
    top: str
    exchange: dict[str, Exchange]


def fit_pool(arms: Mapping[str, Sequence[EpisodeResult]]) -> PoolFit:
    """§5.1b + §5.5 in one place: pick King₀, then fit λ between the floor and King₀.

    THE ONE DEFINITION OF WHICH TWO ARMS λ IS A SLOPE BETWEEN. Four callers used to fit it
    themselves — `simulate.pin_corpus`, `devkit.mock_world`, `archetypes.exchange` and the M3a
    script — and every one hard-coded always-strongest as the top arm. That pair measures whether a
    stronger SINGLE MODEL buys quality, which on the live pool it does not (paired difference
    +0.010 ± 0.054 on livecodebench, −0.051 ± 0.066 on hle, measured 2026-09-06 over three replicated
    draws). The mechanism's question is whether the BEST FIXED POLICY beats the floor, and the
    cascade did, by +0.148 and +0.139 with intervals clear of zero. Fitted to that pair λ is +0.19
    and +0.17 and both benchmarks price; fitted to the old one it was a coin flip and the corpus
    could not launch. It is also the pair `contrast_for` already gives §5.6's power gate, so the
    price and the gate describe one line.

    The floor is always-cheapest. The top is King₀ — unless King₀ IS the floor, in which case there
    is nothing above it to fit to and the top is always-strongest, so that `derive_exchange`'s
    dominator guard fires and names the finding rather than this function guessing a slope.

    `arms` is keyed by policy name and must carry the floor and always-strongest; the cascade is
    optional only so the dev kit's stand-in world can be minimal, and a caller that omits it has
    declared that King₀ is one of the two extremes.
    """
    missing = [name for name in (ALWAYS_CHEAPEST, ALWAYS_STRONGEST) if name not in arms]
    if missing:
        raise ValueError(f"the fixed-policy sweep needs both extremes to fit a slope; missing "
                         f"{missing} from {sorted(arms)}")
    best = king_zero([ReferenceArm(name, tuple(results)) for name, results in arms.items()])
    top = best.name if best.name != ALWAYS_CHEAPEST else contrast_for(best.name)
    floor_results, top_results = arms[ALWAYS_CHEAPEST], arms[top]
    return PoolFit(king_zero=best.name, floor=ALWAYS_CHEAPEST, top=top,
                   exchange=derive_exchange(floor_results, top_results,
                                            pinned_cost([*floor_results, *top_results])))


def contrast_for(king_zero_name: str) -> str:
    """The reference policy King₀ is measured against — chosen against its LADDER (§5.6).

    THE CONTRAST MUST DIFFER FROM KING₀ IN THE MODEL ITS EPISODES END ON, which is a structural
    requirement and not a naming convention. A ladder is climbed until something solves the task and
    the episode's score is the best any step reached (§3), so a ladder's quality is `max` over its
    rungs and is governed by its TOP rung — the model every unsolved episode ends on. Two policies
    that share a top rung have identical quality wherever price tracks capability, and the gate
    scores quality deliberately (see the module docstring).

    SO ALWAYS-STRONGEST IS NOT A CONTRAST FOR A CASCADE, and that is by construction on every
    catalog: `cascade` builds its rungs from `_by_price` and ends on the last entry, which is exactly
    `always_strongest`'s only rung. §5.6 named always-strongest, and §5.5 makes King₀ the cascade
    whenever the cascade wins the sweep — the normal case — so the shipped default compared a policy
    with itself by a second route. Measured on this repo's archetype world: cascade vs
    always-strongest, spread **+0.0000** against a floor of 0.0500, every window refuses itself and
    the arena never opens while the corpus is perfectly healthy. The same King₀ against
    always-cheapest measures **+0.6542** (lcb +0.6042) on that same slice.

    The contrast is therefore always-cheapest — the floor the whole mechanism is measured against,
    and the only fixed policy that ends at the BOTTOM of the price order — except when King₀ *is*
    always-cheapest, where it is always-strongest. Both pairs ask the gate's question of two policies
    that can answer it, and both keep §5.6's two-arm saving: King₀'s arm has to run regardless
    (§8b.8: in window 1 King₀ *is* the king), so the gate never costs a third arm.

    A King₀ whose ladder this module did not build is REFUSED rather than handed always-cheapest on
    the assumption that it ends somewhere else. The defect above was a contrast picked from a name,
    and a rule that guesses for names it does not know is that same rule wearing a different label.
    """
    if king_zero_name not in (ALWAYS_CHEAPEST, ALWAYS_STRONGEST, CASCADE):
        raise ValueError(
            f"no contrast for {king_zero_name!r}: the contrast is chosen against King₀'s ladder, "
            f"and this module builds ladders only for {ALWAYS_CHEAPEST!r}, {ALWAYS_STRONGEST!r} "
            f"and {CASCADE!r} (§5.6)")
    return ALWAYS_STRONGEST if king_zero_name == ALWAYS_CHEAPEST else ALWAYS_CHEAPEST


# --- the power gate -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerVerdict:
    """One window's measured signal — published whether it passes or fails.

    `spread` is oriented: it is the *higher* arm's quality minus the lower one's, so it is never
    negative and `higher` says which policy that was. The gate's question is two-sided — are these
    two distinguishable — and orienting the pair by the observed sign is how a one-sided lower bound
    answers it. The honest reading of that is a two-sided test at 10% rather than a one-sided one at
    5%; it is stated here rather than hidden, and at the magnitudes the floor admits it changes
    nothing.
    """

    king_zero: str
    contrast: str
    higher: str
    spread: float
    lcb: float
    by_benchmark: tuple[tuple[str, float], ...]   # oriented per-benchmark spread, for M2b
    n_tasks: int                                  # per arm
    floor: float

    @property
    def separates(self) -> bool:
        """Both conditions, and they close different holes: the lower bound refuses a spread the
        slice cannot separate from noise, and the floor refuses one that is real but too small to
        host a coronation (§5.2b, ROUTING_MEASUREMENTS §15)."""
        return self.spread > self.floor and self.lcb > 0.0

    @property
    def reason(self) -> str:
        """The sentence the reveal carries. A window that refuses must say what it measured, or a
        reader cannot tell a dead corpus from a broken validator."""
        verdict = "separates policies" if self.separates else "CANNOT separate policies"
        return (f"{verdict}: {self.higher} leads by {self.spread:+.4f} in quality "
                f"(lcb {self.lcb:+.4f}) over {self.n_tasks} tasks per arm, against a floor of "
                f"{self.floor:.4f}. Reference arms: {self.king_zero} vs {self.contrast}.")


def power_gate(king_zero_arm: ReferenceArm, contrast_arm: ReferenceArm, *,
               nonce: str | None = None, seed: int | None = None,
               floor: float = POWER_SPREAD_FLOOR, n_boot: int = N_BOOT,
               failed: Collection[str] = ()) -> PowerVerdict:
    """Can this window separate policies at all? (§5.6) Run ONCE per window, shared by every duel.

    Asked with the duel's own machinery — the paired bootstrap over benchmark-stratified resamples —
    rather than a proxy, for the reason `koth/arena.py::power_of` was built that way: a large band
    over too few tasks still cannot rank anyone, and only the instrument that decides crowns can say
    whether this data would let one be decided. What differs is the score: quality, not
    `final`, because λ is fitted to make the fixed policies tie on `final` (see the module docstring
    — this is the one place where reusing the verdict's score would invert the gate).

    On refusal the caller runs no duels, spends no hotkey and rolls challengers over to the next
    window (§8 step 2, the build plan M8 exit 11).

    `failed` IS §6.3c, AND WITHOUT IT THE GUARD'S PLACEHOLDER IS SCORED HERE. `_GraderGuard` hands a
    dead grader's task back as 0.0 and says of that zero that it "is never scored", because the task
    leaves every arm's denominator before a verdict. That was true of the duels and not of this
    gate, which paired every task it was given. So a sandbox dying during the reference arms pushed
    a zero into both of them, narrowed the measured spread, and could refuse a window the corpus
    could actually have separated — the owner's own infrastructure deciding that nobody is judged.
    """
    if king_zero_arm.name == contrast_arm.name:
        raise ValueError(
            f"both reference arms are {king_zero_arm.name!r}: a policy compared with itself has a "
            "spread of zero and would refuse every window forever — see `contrast_for`")

    pairs = _pairs(king_zero_arm.results, contrast_arm.results, failed=failed)
    # Oriented by the observed sign so the one-sided lower bound answers the two-sided question. The
    # contrast is not required to be the better policy: King₀ is the best fixed policy on `final`,
    # and on QUALITY — which is what is measured here — always-strongest may well lead it.
    lead = sum(float(p.challenger_quality.mean()) for p in pairs) - \
        sum(float(p.king_quality.mean()) for p in pairs)
    higher = contrast_arm.name if lead >= 0.0 else king_zero_arm.name
    if lead < 0.0:
        pairs = tuple(BenchmarkPair(p.benchmark, p.challenger_quality, p.challenger_spend,
                                    p.king_quality, p.king_spend) for p in pairs)

    # The duel's aggregate conditions are reused; its BREADTH rule deliberately is not. A window
    # whose signal is concentrated in one benchmark can still rank policies — whether a WIN may be
    # that concentrated is §5.2's question about a challenger, asked later and about a different
    # thing. Refusing such a window here would refuse a live one, which is why `separates` reads
    # `delta` and `lcb` rather than `verdict.challenger_wins`.
    verdict = duel(pairs, _quality_only, eps=floor, n_boot=n_boot, seed=seed,
                   nonce=None if nonce is None else f"{nonce}|v3-power-gate")
    return PowerVerdict(king_zero=king_zero_arm.name, contrast=contrast_arm.name, higher=higher,
                        spread=verdict.delta, lcb=verdict.lcb,
                        by_benchmark=verdict.by_benchmark, n_tasks=verdict.n_tasks, floor=floor)


def _quality_only(benchmark: str, quality: float, spend: float) -> float:
    """The gate's score: graded quality, spend deliberately unpriced (see the module docstring).

    Satisfies `duel.FinalB` so the gate runs through the verdict's own bootstrap unchanged.
    """
    return quality


def _pairs(king: Sequence[EpisodeResult], contrast: Sequence[EpisodeResult], *,
           failed: Collection[str] = ()) -> tuple[BenchmarkPair, ...]:
    """Row-align the two reference arms by task (§5.2's pairing, §6.3c's symmetry).

    Both arms ran the identical subsample, so a mismatch is a bug in the caller, refused loudly
    rather than intersected: silently scoring the overlap would compare two policies on different
    task sets, which is the asymmetric exclusion §6.3c exists to forbid — and here it would corrupt
    the one number that decides whether the window is scored at all.
    """
    # Dropped from BOTH arms or neither — §6.3c's symmetry is the whole point, and dropping from one
    # is the asymmetric exclusion this function already refuses in its other form.
    if failed:
        king = [r for r in king if r.task_id not in failed]
        contrast = [r for r in contrast if r.task_id not in failed]
    left, right = _index(king), _index(contrast)
    if set(left) != set(right):
        raise ValueError(f"the reference arms cover different benchmarks: {sorted(left)} vs "
                         f"{sorted(right)} — they must run the identical subsample")
    pairs = []
    for benchmark in sorted(left):
        rows, other = left[benchmark], right[benchmark]
        if set(rows) != set(other):
            raise ValueError(f"benchmark {benchmark!r}: the reference arms ran different tasks — "
                             "the gate is paired, so both arms must run the identical subsample")
        task_ids = sorted(rows)
        pairs.append(BenchmarkPair(
            benchmark,
            np.array([rows[t].graded_score for t in task_ids], dtype=float),
            np.array([rows[t].spend_usd for t in task_ids], dtype=float),
            np.array([other[t].graded_score for t in task_ids], dtype=float),
            np.array([other[t].spend_usd for t in task_ids], dtype=float)))
    return tuple(pairs)


def _index(results: Sequence[EpisodeResult]) -> dict[str, dict[str, EpisodeResult]]:
    if not results:
        raise ValueError("a reference arm with no episodes measures nothing")
    indexed: dict[str, dict[str, EpisodeResult]] = {}
    for result in results:
        rows = indexed.setdefault(result.benchmark, {})
        if result.task_id in rows:
            raise ValueError(f"task {result.task_id!r} appears twice in one reference arm")
        rows[result.task_id] = result
    return indexed
