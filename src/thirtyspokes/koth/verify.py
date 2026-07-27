"""Validator-side proof verification + the KOTH scoring formula (docs/DESIGN.md §4-6).

Cheap and re-execution-free. `verify_proof`:
  1. checks the hardware quote, the approved measurement, the payload binding, the
     issued (epoch, nonce), the miner hotkey, and — the KOTH binding — that the
     attested `source_hash`/`weights_hash` equal the hashes the validator recomputes
     from the independently-downloaded public bundle + on-chain commit;
  2. re-derives the SAME nonce-seeded task slice the runtime ran, so it knows the
     public gold for every task without re-running the agent;
  3. grades the attested answers (missing/substituted tasks count as wrong) and
     returns per-benchmark accuracy + a Wilson lower confidence bound and the
     reign scalar `Q_lcb = Σ_b w_b·lcb_b`. Cost is NOT folded into the score — it is
     a budget ceiling applied separately by `eligible`.

`dethrone_guard` is the Pareto-with-tolerance slot-1 takeover rule. The default
anti-cheat backstops are pure proof-inspection (the validator runs no miner code):
`grounding_check` (every scored answer must derive from a logged pool response —
the memorization defense) and `behavioral_duplicates` copy-dedup over the attested
answers, plus `scan_source`/`adjudicate_challenge`. The re-execution memorization
probe (`memorization_collapsed*` + the validator sandbox) is the opt-in upgrade path
(docs/DESIGN.md §6). See docs/DESIGN.md for the grounding vs null-pool trade-off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import sqrt
from statistics import NormalDist

import numpy as np

from ..gateway import signing
from ..gateway.grader import extract_number
from ..tee.attestation import verify_quote
from .benchmarks import Benchmark, bench_seed, extract_choice, grade_choice, grade_patch
from .evidence import wilson_lcb
from .proof import Proof
from .store import hash_source


@dataclass
class BenchStat:
    n: int
    acc: float          # point accuracy over the assigned slice
    lcb: float          # Wilson lower confidence bound (noise can't dethrone)
    cost_usd: float


@dataclass
class ProofVerdict:
    valid: bool
    reason: str
    per_bench: dict[str, BenchStat] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    score: float = 0.0          # Q_lcb = Σ w·lcb, the bounded [0,1] reign scalar
    total_score: float = 0.0    # Σ w·acc (point) — display + per-benchmark floor
    eligible: bool = True        # passes cost budget + floors + pool-call gate


def _bootstrap_lcb(correct: list[float], alpha: float, boot: int, seed: int) -> float:
    """Lower confidence bound on the slice accuracy. Wilson, NOT the resampling bootstrap.

    The bootstrap is DEGENERATE on the all-same slices that matter most here: resampling 8 identical
    values yields 8 identical means, so a perfect 8/8 slice reported `lcb = 1.000` — zero claimed
    uncertainty from eight questions. That defeated the entire premise of scoring on the lower bound
    (`dethrone_guard` then cleared `lcb_c > acc_king + margin` against ANY king below 0.97), making
    "resubmit every epoch until a slice comes up perfect" a free lottery ticket for the crown.
    Wilson is exact on the same counts and correctly prices small samples: 8/8 -> 0.747, and it is
    already what the accumulate path uses (`evidence.wilson_lcb`), so the two modes now agree.

    `boot` is retained for call compatibility and unused; `seed` likewise (Wilson is deterministic,
    which is a bonus — two validators no longer depend on drawing the same resamples).
    """
    if not correct:
        return 0.0
    z = NormalDist().inv_cdf(1.0 - alpha)        # one-sided, matching the accumulate path
    return wilson_lcb(float(np.sum(correct)), float(len(correct)), z)


def verify_proof(
    proof: Proof,
    *,
    approved_measurements: set[str],
    platform_public_hex: str,
    expect_epoch: int,
    expect_nonce: str,
    expect_hotkey: str,
    expect_source_hash: str,
    expect_weights_hash: str,
    suite: list[Benchmark],
    n_per_bench: int,
    alpha: float = 0.05,
    boot: int = 1000,
    approved_mrtd: set[str] | None = None,
    approved_rtmr: dict[int, str] | None = None,
    tcb_accept: frozenset[str] | None = None,
    collateral: str | None = None,
    pccs_url: str | None = None,
    now: int | None = None,
    enforce: bool = False,
) -> ProofVerdict:
    def bad(reason: str) -> ProofVerdict:
        return ProofVerdict(False, reason)

    q = proof.quote
    if q is None:
        return bad("bad_platform_quote")
    is_tdx = q.platform_sig.startswith("tdx:")
    if enforce:
        # PRODUCTION FAIL-CLOSED (docs/DESIGN.md §8). The miner's own CVM samples AND runs the
        # benchmark, so the score is trustworthy ONLY if the audited, measured runtime
        # provably ran — the one thing a trustless miner cannot forge. A tampered runtime
        # can forge every payload field (source/weights hash, cost, answers, and the
        # self-reported `measurement` string), so the sole real anchor is the hardware
        # quote gated on an owner-pinned measured image (MRTD + boot RTMR1/2 + runtime
        # RTMR3) under a TCB policy. A missing gate is a config error that would silently
        # admit a forged runtime — so it DQs here rather than skipping the check.
        if not is_tdx:
            return bad("mock_quote_rejected")             # mock vendor key = zero security
        if not approved_mrtd:
            return bad("mrtd_gate_unset")
        if not approved_rtmr or not {1, 2, 3} <= set(approved_rtmr):
            return bad("rtmr_gate_unset")                 # need boot RTMR1/2 + runtime RTMR3
        if tcb_accept is None:
            return bad("tcb_policy_unset")                # enforce ⇒ full DCAP (TCB/CRL/QE)
        if not proof.confined:
            # The agent ran with network egress. The measured-image gates above prove WHICH image
            # booted, not what it did inside — and an unconfined agent can reach an off-allow-list
            # model using a key embedded in its own weights, voiding the pool pinning, the metered
            # cost and the budget ceiling while still satisfying `no_pool_call` with one token call.
            # `run_agent_confined` degrades silently when the namespace probe fails, so this is
            # gated on the ATTESTED fact rather than on the miner's configuration.
            return bad("unconfined_agent")
    if is_tdx:                                            # WS7: real Intel-TDX hardware quote
        # `tcb_accept` set -> full DCAP (TCB status/CRL/QE-identity via dcap-qvl, H1);
        # else the offline crypto-chain-only path (cert chain to Intel root + binding).
        if tcb_accept is not None:
            from .tdx import verify_tdx_quote_field_full
            vq = verify_tdx_quote_field_full(
                q, approved_mrtd=approved_mrtd, approved_rtmr=approved_rtmr,
                tcb_accept=tcb_accept, collateral=collateral, pccs_url=pccs_url, now=now)
        else:
            from .tdx import verify_tdx_quote_field     # lazy: needs `cryptography` on real HW
            vq = verify_tdx_quote_field(q, approved_mrtd=approved_mrtd)
        if not vq.ok:
            if vq.reason == "mrtd_not_approved" or vq.reason.startswith("rtmr"):
                return bad("unapproved_runtime")       # hardware image not owner-approved
            if vq.reason.startswith("tcb_"):
                return bad(vq.reason)                  # surface the TCB status distinctly
            return bad("bad_platform_quote")
    elif not verify_quote(q, platform_public_hex):     # offline/dev mock vendor key
        return bad("bad_platform_quote")
    if q.measurement != proof.measurement:
        return bad("measurement_mismatch")
    if q.report_data != proof.report_data():
        return bad("report_data_mismatch")            # payload tampered post-attestation
    if proof.measurement not in approved_measurements:
        return bad("unapproved_runtime")
    if (proof.epoch, proof.nonce) != (expect_epoch, expect_nonce):
        return bad("epoch_nonce_mismatch")            # replay / stale / best-of-N
    if proof.hotkey != expect_hotkey:
        return bad("hotkey_mismatch")                 # a copier resubmitting someone's proof
    if proof.source_hash != expect_source_hash or proof.weights_hash != expect_weights_hash:
        return bad("artifact_binding_mismatch")       # not the downloaded/committed artifact
    c = proof.total_cost_usd
    if not (c == c) or c < 0:                          # NaN / negative
        return bad("bad_cost")

    # re-derive the assigned slice + gold, and index the attested answers
    submitted = {(r.benchmark, r.task_id): r for r in proof.results}
    per_bench: dict[str, BenchStat] = {}
    q_lcb = 0.0          # Σ w·lcb  (bounded reign scalar)
    total_score = 0.0    # Σ w·acc  (point)
    seen: set[tuple[str, str]] = set()
    for bench in suite:
        seed = bench_seed(proof.nonce, proof.epoch, bench.name)
        assigned = bench.sample(n_per_bench, seed)
        correct, cost = [], 0.0
        for t in assigned:
            key = (bench.name, t.task_id)
            seen.add(key)
            r = submitted.get(key)
            ans = r.answer if r is not None else ""    # missing task -> graded wrong
            if r is not None:
                cost += float(r.cost_usd)
            correct.append(bench.grade(ans, t.gold))
        lcb = _bootstrap_lcb(correct, alpha, boot,
                             seed=int(signing.sha256_hex(f"{proof.nonce}|{bench.name}")[:8], 16))
        acc = float(np.mean(correct)) if correct else 0.0
        per_bench[bench.name] = BenchStat(len(assigned), acc, lcb, cost)
        q_lcb += bench.weight * lcb
        total_score += bench.weight * acc
    # a result for a task that was NOT assigned = task substitution
    if any(k not in seen for k in submitted):
        return bad("unexpected_task")

    return ProofVerdict(True, "ok", per_bench, c, q_lcb, total_score)


def dethrone_guard(
    challenger: ProofVerdict,
    king: ProofVerdict,
    *,
    margin: float = 0.03,
    tol: float = 0.02,
    cost_tol: float = 0.10,
    min_tasks: int = 5,
    cost_margin: float = 0.10,
) -> tuple[bool, str]:
    """Affine-style Pareto-with-tolerance. To take slot 1 a challenger must be not-worse on EVERY
    benchmark and pass a per-benchmark sample gate, and then win on ONE of two axes:

      (a) QUALITY — confidently (LCB) dominant on >= 1 benchmark, or
      (b) COST    — at quality parity, at least `cost_margin` cheaper than the king.

    (b) is what keeps the subnet alive once the suite saturates. With quality-only dominance, an
    incumbent sitting at the accuracy ceiling can NEVER be dethroned — no challenger can be
    "confidently better" than a perfect score — so emissions freeze on the earliest committer and
    the competitive gradient dies. Cost is the axis that never saturates (you can always be cheaper),
    and it is also where routing/orchestration actually has headroom, so it is the right second axis.

    Any single failure blocks — the validator clamps on ANY `not ok`."""
    quality_dominant = False
    for name, cs in challenger.per_bench.items():
        ks = king.per_bench.get(name)
        ka = ks.acc if ks else 0.0
        if cs.n < min_tasks:
            return False, f"thin_eval:{name}"
        if cs.acc < ka - tol:
            return False, f"regression:{name}"
        if cs.lcb > ka + margin:
            quality_dominant = True
    # (b) parity + materially cheaper. The loop above already established not-worse-anywhere.
    cost_dominant = (king.total_cost_usd > 0
                     and challenger.total_cost_usd <= king.total_cost_usd * (1 - cost_margin))
    if not (quality_dominant or cost_dominant):
        return False, "no_confident_gain"
    # A king-RELATIVE cost ceiling (challenger <= king * (1 + cost_tol)) used to veto here. It is a
    # one-way ratchet: once a cheap-but-weak miner takes a vacant crown, no better-but-pricier agent
    # can ever dethrone it, however large the quality gap — observed live, a challenger at Q_lcb
    # 0.999 clamped forever under a king at 0.282. That is the ossification bug in reverse (the same
    # failure the COST axis above was added to fix, mirrored). Cost policy belongs where the owner
    # actually sets it: the ABSOLUTE per-slice `budget`, enforced upstream by `eligible`, plus the
    # small `cost_tiebreak` that orders equals. So a quality-dominant challenger inside the budget is
    # allowed to cost more; it just pays the tiebreak for it.
    if not quality_dominant and king.total_cost_usd > 0 \
            and challenger.total_cost_usd > king.total_cost_usd * (1 + cost_tol):
        return False, "cost_regression"
    return True, "ok"


def eligible(vd: ProofVerdict, *, budget: float, f_min: float) -> tuple[bool, str]:
    """Cost-budgeted-quality eligibility (WS3): under the owner's per-slice budget AND
    above a minimum accuracy on EVERY benchmark (so answering only the cheap benchmark
    is disqualifying, not rewarded). The ≥1-pool-call-per-task gate is enforced by the
    validator from the trace. Ineligible miners are EXCLUDED from the reign, never fed score 0."""
    if vd.total_cost_usd > budget:
        return False, "over_budget"
    for name, bs in vd.per_bench.items():
        if bs.acc < f_min:
            return False, f"below_floor:{name}"
    return True, "ok"


# --- DEFAULT memorization backstop: the grounding check (pure proof-inspection; the
#     validator runs NO miner code). Every scored answer must derive from a pool response
#     logged in the hash-attested trace; a memorizer that ignores the pool and returns a
#     memorized answer matches no response -> flagged. See docs/DESIGN.md. ----------------
def _bench_kind(bench) -> str:
    """Answer type of a benchmark, read from its grader: 'choice' (A-D) | 'patch' | 'number'."""
    declared = getattr(bench, "answer_kind", None)
    if declared is not None:
        return declared
    g = getattr(bench, "_grade", None)
    if g is grade_choice:
        return "choice"
    if g is grade_patch:
        return "patch"
    return "number"                                  # grade_math / real GSM8K grade / default


def answer_token(text, kind: str):
    """Canonical, comparable token of an answer/response — using the SAME parser the grader
    uses, so grounding and grading agree. None if unparseable."""
    if kind == "choice":
        return extract_choice(text)
    if kind == "number":
        v = extract_number(text)
        return None if v is None else format(v, ".6g")
    s = " ".join(str(text).split())                  # patch/text: whitespace-normalized
    return s or None


# Answer kinds where the answer is NOT structurally present in the prompt, so "who produced it
# first" is a meaningful question. MULTIPLE CHOICE IS EXCLUDED ON PURPOSE: every option is in the
# prompt by construction, so `extract_choice(prompt)` already returns the gold and a provenance rule
# would flag every honest agent. That is not a tuning problem — it means no prompt/response rule can
# ever defend an MCQ benchmark against memorization, which is why the scored suite ranks on
# free-form answers (see `benchmarks.real_suite`).
_PROVENANCE_KINDS = frozenset({"number", "patch"})


def _grounded_one(answer, calls: list[dict], kind: str) -> tuple[bool, str]:
    """Provenance of ONE scored answer, walking that task's calls in order.

    Grounded  — the token first appears in a RESPONSE: the pool produced it.
    Laundered — the token first appears in a PROMPT: the agent already had it and used the pool as
                an echo chamber, which is exactly how a memorizer defeats plain grounding.
    Ungrounded— it never appears at all: the agent answered without the pool.
    """
    tok = answer_token(answer, kind)
    check_prompts = kind in _PROVENANCE_KINDS
    for e in calls:                                   # trace is in call order
        if check_prompts and answer_token(e.get("prompt", ""), kind) == tok:
            return False, "laundered"
        # an unparseable answer relaying an equally-unparseable response is still grounded
        if answer_token(str(e.get("response", "")), kind) == tok:
            return True, "ok"
    return False, "ungrounded"


def grounding_check(proof: Proof, trace: list[dict], suite: list[Benchmark], *,
                    max_ungrounded: float = 0.15) -> tuple[bool, str]:
    """Each scored answer must DERIVE FROM the pool — not merely appear next to it.

    Reads only the proof + the hash-attested trace: no re-execution, no secret bank, no private
    data, deterministic across validators. DQs when the unexplained fraction exceeds
    `max_ungrounded` (slack for synthesis orchestration and parser noise); the reported reason is
    whichever failure dominates.

    Plain "answer ∈ some response" was defeatable by laundering — put the memorized answer in your
    own prompt and the model echoes it back, so the answer is 'grounded' in a response the agent
    itself authored. Ordering closes that: information the agent already had surfaces on the prompt
    side first, information the pool supplied surfaces on the response side first. Verified against
    honest verify-loops and cheap→strong escalation, which both keep passing because their answer
    still originates in a response.

    Residual: a few-shot prompt whose trailing number happens to equal the task's answer reads as
    laundered. `extract_number` takes the LAST number, which is normally part of the question rather
    than an example's answer, and `max_ungrounded` absorbs the occasional collision.
    """
    kind = {b.name: _bench_kind(b) for b in suite}
    calls_by_task: dict[str, list[dict]] = {}
    for e in trace:
        calls_by_task.setdefault(e.get("task_id"), []).append(e)
    total = laundered = ungrounded = 0
    for r in proof.results:
        total += 1
        ok, why = _grounded_one(r.answer, calls_by_task.get(r.task_id, []),
                                kind.get(r.benchmark, "number"))
        if not ok:
            if why == "laundered":
                laundered += 1
            else:
                ungrounded += 1
    if total == 0:
        return True, "ok"
    frac = (laundered + ungrounded) / total
    if frac <= max_ungrounded:
        return True, "ok"
    return False, "laundered" if laundered >= ungrounded else "ungrounded"


# --- optimistic anti-cheat backstops (applied by the validator on the artifact it
#     downloads itself, so they are bound to the committed public artifact) --------

def memorization_collapsed_relative(claimed_acc: float, n_c: int, fresh_acc: float, n_f: int,
                                    allowance: float = 0.0, z_crit: float = 2.33) -> bool:
    """Difficulty-normalized one-sided two-proportion test.

    `allowance` is the COHORT's median (claimed - fresh) drop — i.e. how much harder the
    held-out probe is *for everyone this epoch*. We flag a miner only when its own drop
    exceeds the cohort's by more than sampling noise.

    Why this matters (and why the plain absolute test is unsafe with a secret probe): once
    the probe is drawn from an owner-curated bank (`koth/heldout.py`) rather than a held-out
    slice of the SAME distribution, its difficulty need not match the scored benchmark. An
    honest, perfectly-generalizing router then shows `claimed > fresh` purely from difficulty,
    and the absolute test fires — measurably so: at n=64 a 15-point difficulty gap false-DQs an
    honest miner, and the power *grows* with more probe data. Normalizing by the cohort
    separates "the probe is hard" (everyone drops) from "this miner memorized" (only they do).

    `allowance=0.0` reduces exactly to the original absolute test.
    """
    if n_c == 0 or n_f == 0:
        return True                                   # fail closed: no evidence of generalization
    p = (claimed_acc * n_c + fresh_acc * n_f) / (n_c + n_f)
    se = sqrt(p * (1.0 - p) * (1.0 / n_c + 1.0 / n_f))
    excess = (claimed_acc - fresh_acc) - allowance    # drop beyond what the cohort also suffered
    if se == 0.0:
        return excess > 0.05                          # degenerate (all-same); tiny tolerance
    return excess / se > z_crit


def memorization_collapsed(claimed_acc: float, n_c: int, fresh_acc: float, n_f: int,
                           z_crit: float = 2.33) -> bool:
    """Absolute (un-normalized) memorization test — valid only when the probe is drawn from the
    SAME difficulty distribution as the scored slice. Kept as the `allowance=0` special case."""
    return memorization_collapsed_relative(claimed_acc, n_c, fresh_acc, n_f, 0.0, z_crit)


def cohort_probe_allowance(drops: list[float], *, min_cohort: int, max_drop: float) -> float:
    """The difficulty allowance for this epoch: the cohort's MEDIAN claimed-minus-fresh drop.

    Median (not mean) so a minority of real memorizers can't inflate it. Capped at `max_drop`
    so a colluding all-memorizer cohort can't raise the allowance enough to hide — the cap is the
    owner's declaration of "no honest probe is harder than this".

    With fewer than `min_cohort` miners there is nothing to calibrate against, so we fall back to
    that same owner-declared `max_drop` rather than to 0.0: an allowance of 0 would apply the
    strict absolute test and false-DQ honest miners whenever the probe is harder than the public
    benchmark — fatal for a bootstrapping subnet with 1-2 miners. A memorizer's drop far exceeds
    `max_drop`, so it is still caught. (Set `max_drop=0` for a difficulty-matched probe to recover
    the strict test exactly.)"""
    vals = sorted(d for d in drops if d == d)         # drop NaNs
    if len(vals) < min_cohort:
        return max_drop
    mid = len(vals) // 2
    med = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    return min(max(med, 0.0), max_drop)


def _agreement(a: tuple, b: tuple) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def behavioral_duplicates(fingerprints: dict[str, tuple], commit_block: dict[str, int],
                          *, agree: float = 0.95) -> dict[str, str]:
    """Cluster miners by their answers on a SHARED probe slice; near-identical answer
    vectors are copies (robust to weight-space perturbation, unlike a weight hash).
    Keep the earliest-committed representative; return {loser_hotkey: winner_hotkey}."""
    order = sorted(fingerprints, key=lambda h: (commit_block.get(h, 0), h))  # earliest commit first
    reps: list[tuple[str, tuple]] = []
    losers: dict[str, str] = {}
    for hk in order:
        fp = fingerprints[hk]
        match = next((rhk for rhk, rfp in reps if _agreement(fp, rfp) >= agree), None)
        if match is None:
            reps.append((hk, fp))
        else:
            losers[hk] = match
    return losers


_HARDCODE = re.compile(r"ANSWER_TABLE|GOLD_ANSWERS|hardcoded|answers\s*=\s*\{", re.IGNORECASE)


def scan_source(source_text: str) -> tuple[bool, str]:
    """Flag literal hardcoded-answer tables in the PUBLIC source. A pattern scan is a
    stand-in for an AST/human audit; the validator runs it on the artifact it
    downloads, so it is bound to the committed source by construction."""
    return (True, "hardcoded_answers") if _HARDCODE.search(source_text) else (False, "clean")


def _decoys(golds: list[str], salt: str) -> list[str]:
    """One deterministic control per gold, with the SAME digit-length — so a decoy is exactly as
    likely to occur by chance as the real answer it stands in for. Derived from a shared salt, so
    every validator computes the identical control set."""
    out = []
    for i, s in enumerate(golds):
        n = len(s)
        h = int(signing.sha256_hex(f"{salt}|{i}|{s}")[:8], 16)
        d = str(h % (10 ** n)).zfill(n)
        if d == s:                                  # never let a control BE the answer
            d = str((h + 1) % (10 ** n)).zfill(n)
        out.append(d)
    return out


def scan_weights(weights: bytes, golds: list, *, salt: str = "", min_golds: int = 4,
                 min_hit: float = 0.5, ratio: float = 3.0) -> tuple[bool, str]:
    """Flag an answer table hidden in the OPAQUE weights blob.

    `scan_source` only ever sees `source_text`, so `weights.bin` was the obvious place to keep a
    lookup table — public, bound, and never examined.

    Naive substring search is useless: a short gold like "8" occurs in essentially any binary. So
    the real answers are measured against DECOYS OF THE SAME DIGIT-LENGTH and flagged only when they
    hit substantially more often than the controls. Single-digit golds then hit in both sets and
    correctly discriminate nothing, while a table of multi-digit answers stands out sharply.

    Only digit-bearing golds are usable: an MCQ letter has no same-shape decoy (and "A" matches
    almost any text), so a choice benchmark is skipped entirely rather than measured without a
    control — getting that wrong flagged every honest miner in the offline sim.

    This raises the bar on lazy memorization; it is NOT a proof of absence, since a compressed or
    learned encoding defeats it. `grounding_check` is the defence that targets the behaviour
    regardless of how the answers are stored.
    """
    if not weights:
        return False, "clean"
    # only golds that are numeric enough to build a same-shape control for
    usable = [s for s in (str(g) for g in golds) if s.isdigit()]
    if len(usable) < min_golds:
        return False, "clean"
    text = weights.decode("latin-1")
    present = lambda vals: sum(1 for v in vals if v in text) / len(vals)   # noqa: E731
    real, ctrl = present(usable), present(_decoys(usable, salt))
    if real >= min_hit and real >= ratio * max(ctrl, 1e-9):
        return True, "answers_in_weights"
    return False, "clean"


@dataclass(frozen=True)
class Challenge:
    challenger_hotkey: str
    target_hotkey: str
    source_text: str          # the target's PUBLIC source (anyone can fetch + read it)


def adjudicate_challenge(ch: Challenge, committed_source_hash: str) -> tuple[bool, str]:
    """External fraud-proof path. BINDS first: the challenged source must hash to the
    target's committed source_hash, so a griefer cannot submit fabricated source
    against an honest miner (review D4). Production adds challenger stake + slash."""
    if hash_source(ch.source_text) != committed_source_hash:
        return (False, "unbound_source")
    return scan_source(ch.source_text)


# --- the router-agent scalar: "best answer at the lowest price, for a given ask" ---------------
# The old scalar (Q_lcb - lambda*cost) has three defects for a ROUTING competition:
#   1. It scores the OUTCOME, not the DECISION. "Always call the frontier model" and a router that
#      correctly predicts which asks the cheap model handles score almost identically at lambda=0.02.
#   2. It is ABSOLUTE, not baseline-relative. On a 95%-accurate pool every router scores ~95%, so
#      ~98% of the number measures the POOL and ~2% measures the miner.
#   3. It is a POINT, not a frontier. A fixed lambda bakes in one quality/price exchange rate -- the
#      owner's -- but that tradeoff belongs to the user: a throwaway ask wants the cheapest adequate
#      answer, a critical one wants the best available.
#
# `router_headroom` fixes all three: 0.0 = no better than randomising over fixed pool models AT YOUR
# PRICE, 1.0 = matched the budget-constrained per-query oracle there. It needs a POOL REFERENCE for
# the epoch's slice -- every (task, pool-model) score and cost -- which the OWNER publishes, since a
# validator runs no inference and cannot know what other models would have answered.

def _upper_hull(points):
    """Non-decreasing upper convex hull of (cost, quality). Every interpolated point is physically
    realisable by randomising between the two neighbouring policies."""
    out, best = [], -1.0
    for c, q in sorted(points):
        best = max(best, q)
        out.append((c, best))
    h = []
    for c, q in out:
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (c - h[-2][0]) <= (q - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append((c, q))
    return h


def _on_hull(h, cost):
    if not h or cost <= h[0][0]:
        return h[0][1] if h else 0.0
    for (c1, q1), (c2, q2) in zip(h, h[1:]):
        if c1 <= cost <= c2:
            t = (cost - c1) / (c2 - c1) if c2 > c1 else 0.0
            return q1 + t * (q2 - q1)
    return h[-1][1]


def zero_frontier(pool_scores, pool_costs):
    """RouterBench's Zero router: what a FEATURELESS policy reaches at each price, by randomising
    over fixed pool models. The bar a real router must clear to have demonstrated anything."""
    S, C = np.asarray(pool_scores, float), np.asarray(pool_costs, float)
    return _upper_hull([(C[:, j].mean(), S[:, j].mean()) for j in range(S.shape[1])])


def oracle_frontier(pool_scores, pool_costs):
    """The budget-constrained PER-QUERY upper bound: start from the cheapest model on each ask, then
    buy the cheapest per-ask upgrades first. For binary scores the only upgrade worth buying on an
    ask is to the cheapest model correct there, so this is exact rather than greedy-approximate."""
    S, C = np.asarray(pool_scores, float), np.asarray(pool_costs, float)
    T, M = S.shape
    base = C.argmin(axis=1)
    cur_s, cur_c = S[np.arange(T), base].copy(), C[np.arange(T), base].copy()
    ups = []
    for t in range(T):
        better = [j for j in range(M) if S[t, j] > cur_s[t]]
        if better:
            j = min(better, key=lambda j: C[t, j])
            ups.append((C[t, j] - cur_c[t], t, j))
    pts = [(cur_c.mean(), cur_s.mean())]
    for _dc, t, j in sorted(ups):
        cur_c[t], cur_s[t] = C[t, j], S[t, j]
        pts.append((cur_c.mean(), cur_s.mean()))
    return _upper_hull(pts)


def router_headroom(acc: float, cost: float, pool_scores, pool_costs) -> float:
    """Share of the ACHIEVABLE headroom this router captured, at its own price.

      0.0  -> no better than randomising over fixed pool models at this cost
      1.0  -> matched the budget-constrained per-query oracle at this cost
      <0   -> worse than the featureless baseline

    Unlike a quality-only measure, matching quality at a fraction of the price scores WELL rather
    than negative -- which is the whole point of "best answer at the lowest price".
    """
    z = _on_hull(zero_frontier(pool_scores, pool_costs), cost)
    o = _on_hull(oracle_frontier(pool_scores, pool_costs), cost)
    return float((acc - z) / (o - z)) if o - z > 1e-12 else 0.0


def per_ask_regret(miner_scores, miner_costs, pool_scores, pool_costs):
    """Per-ASK attribution: what did this router give up, on each individual question?

      quality_regret[q] = max_m s(q,m) - s(q, chosen)      # a better model was available
      cost_regret[q]    = cost(q, chosen) - cheapest_correct(q)

    Why per-ask rather than the aggregate `router_headroom` takes: aggregates cannot separate a
    router that made good decisions from one that got lucky overall, and they waste information.
    Binary accuracy carries ~1 bit per ask; regret is CONTINUOUS and graded against every known
    alternative, so a paired comparison needs materially fewer asks to resolve the same gap. That
    matters directly at n_per_bench=8.

    An ask nobody in the pool solves contributes ZERO quality regret to everyone -- routing cannot
    fix it, so it must not drag every miner down the way an accuracy average does.
    """
    S, C = np.asarray(pool_scores, float), np.asarray(pool_costs, float)
    ms, mc = np.asarray(miner_scores, float), np.asarray(miner_costs, float)
    qual = S.max(axis=1) - ms
    cheapest_ok = np.array([
        min([C[t, j] for j in range(S.shape[1]) if S[t, j] >= S[t].max()], default=C[t].min())
        for t in range(S.shape[0])])
    return qual, mc - cheapest_ok


def trajectory_stats(proof, trace: list[dict]) -> dict:
    """Diagnostics for the INTELLIGENCE LAYER — the half of "routing model + intelligence layer"
    that the scalar cannot see.

    The mechanism scores the final answer and the total cost, so it cannot tell an agent that
    escalated BECAUSE its verifier caught an error from one that escalated blindly, nor one that
    stopped early because it was confident from one that got lucky. The trace already records every
    call and is hash-attested (`call_log_hash`), so the signal is present and simply discarded.

    Reported as DIAGNOSTICS, never as reward terms: any of these becomes trivially gameable the
    moment it pays (a miner would escalate constantly to farm `escalations`), exactly as
    RouterEval's selection-entropy metric is maximised by a random router.
    """
    by_task: dict = {}
    for e in trace:
        by_task.setdefault(e.get("task_id"), []).append(e)
    final = {r.task_id: r.answer for r in proof.results}
    n_esc = n_rec = n_waste = 0
    calls = []
    for tid, es in by_task.items():
        calls.append(len(es))
        costs = [float(e.get("cost_usd", 0.0)) for e in es]
        for i in range(1, len(es)):                       # escalation = a pricier call after a cheaper
            if costs[i] > costs[i - 1] > 0:
                n_esc += 1
                # did the escalation actually change the answer that was finally submitted?
                if str(es[i].get("response", "")).strip() == str(final.get(tid, "")).strip() \
                        and str(es[i - 1].get("response", "")).strip() != str(final.get(tid, "")).strip():
                    n_rec += 1
        # a call whose response never became the answer and was not followed by an escalation
        for e in es[:-1]:
            if str(e.get("response", "")).strip() != str(final.get(tid, "")).strip():
                n_waste += 1
    return {
        "calls_per_ask": round(float(np.mean(calls)), 2) if calls else 0.0,
        "escalations": n_esc,
        "escalations_that_changed_the_answer": n_rec,
        "escalation_yield": round(n_rec / n_esc, 3) if n_esc else 0.0,
        "superseded_calls": n_waste,
        "latency_s": round(float(getattr(proof, "latency_s", 0.0)), 3),
        "tokens_out": int(getattr(proof, "tokens_out", 0)),
    }
