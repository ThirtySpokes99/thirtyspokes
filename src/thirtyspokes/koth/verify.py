"""Validator-side proof verification + the KOTH scoring formula (docs/DESIGN.md §4-6).

Cheap and re-execution-free. `verify_proof`:
  1. checks the hardware quote, the approved measurement, the payload binding, the
     issued (epoch, nonce), the miner hotkey, and — the KOTH binding — that the
     attested `source_hash`/`weights_hash` equal the hashes the validator recomputes
     from the independently-downloaded public bundle + on-chain commit;
  2. re-derives the SAME nonce-seeded task slice the runtime ran, so it knows the
     public gold for every task without re-running the agent;
  3. grades the attested answers (missing/substituted tasks count as wrong) and
     returns per-benchmark accuracy + a bootstrap lower confidence bound and the
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

import numpy as np

from ..gateway import signing
from ..gateway.grader import extract_number
from ..tee.attestation import verify_quote
from .benchmarks import Benchmark, bench_seed, extract_choice, grade_choice, grade_patch
from .proof import Proof
from .store import hash_source


@dataclass
class BenchStat:
    n: int
    acc: float          # point accuracy over the assigned slice
    lcb: float          # bootstrap lower confidence bound (noise can't dethrone)
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
    x = np.asarray(correct, dtype=float)
    if x.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(boot, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha))


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


def grounding_check(proof: Proof, trace: list[dict], suite: list[Benchmark], *,
                    max_ungrounded: float = 0.15) -> tuple[bool, str]:
    """Each scored answer must match a pool response logged for its task. Reads only the proof
    + the hash-attested trace (no re-execution). DQs `ungrounded` when the ungrounded fraction
    exceeds `max_ungrounded` (slack for synthesis orchestration + parser noise). KNOWN residual:
    prompt-injection laundering (the agent tells the model the answer and the model echoes it)
    passes — the null-pool re-execution probe is the counterfactual upgrade path."""
    kind = {b.name: _bench_kind(b) for b in suite}
    resp_by_task: dict[str, list[str]] = {}
    for e in trace:
        resp_by_task.setdefault(e.get("task_id"), []).append(str(e.get("response", "")))
    total = ungrounded = 0
    for r in proof.results:
        total += 1
        k = kind.get(r.benchmark, "number")
        at = answer_token(r.answer, k)
        # grounded iff the answer's token matches SOME logged response's token (an unparseable
        # answer that relays an equally-unparseable response is still grounded: None in {None}).
        toks = {answer_token(resp, k) for resp in resp_by_task.get(r.task_id, [])}
        if at not in toks:
            ungrounded += 1
    if total == 0:
        return True, "ok"
    frac = ungrounded / total
    return (frac <= max_ungrounded, "ok" if frac <= max_ungrounded else "ungrounded")


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
