"""Adversarial competition simulation — the Phase-0 mechanism stress test.

Spawns a cast of adversarial miner archetypes and runs them through the real
reign + fingerprint + sanity machinery for many epochs on freshly-sampled eval
batches (drawn from a reservoir disjoint from the training corpus, modelling
"never reuse questions"). The point is to show the mechanism rewards genuine
improvement and starves everything else:

  honest_improver   submits a steadily better router each phase   -> should win
  early_bird        good router, submitted early                  -> strong cumulative
  hoarder           same-quality router, withheld until near-end  -> loses to early_bird
  camper            decent router at t0, never improves           -> slides down slots
  overfitter        trained on a tiny slice, memorizes noise      -> weak on fresh evals
  copier            clones the champion at a later block          -> zeroed by dedup
  degenerate        constant "always cheapest" policy             -> rejected by sanity gate
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .cache import OutcomeCache
from .reign import Reign, Submission
from .router import RouterHead
from .scorer import score_choice
from .train import TrainedRouter, train_router
from .validate import calibrate_threshold, fingerprint, is_duplicate, sanity_gate


@dataclass
class Artifact:
    miner_id: str
    hotkey: str
    commit_block: int
    head: RouterHead
    theta: np.ndarray


@dataclass
class LiveSub:
    art: Artifact
    fp: np.ndarray
    disqualified: bool = False
    dq_reason: str = ""


@dataclass
class CompetitionReport:
    lam: float
    epochs: int
    dup_threshold: float
    cumulative: dict[str, float]           # miner_id -> summed emission weight
    disqualifications: dict[str, str]      # miner_id -> reason
    per_epoch_weights: list[dict[str, float]]
    slot_history: list[list[str]]
    final_scores: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Adversarial competition report", "==============================",
                 f"  lambda={self.lam}  epochs={self.epochs}  dup_threshold={self.dup_threshold:.4f}",
                 "",
                 "  cumulative emissions (share of total paid to miners):"]
        total = sum(v for k, v in self.cumulative.items() if k != "uid0") or 1.0
        for mid, val in sorted(self.cumulative.items(), key=lambda kv: -kv[1]):
            tag = ""
            if mid in self.disqualifications:
                tag = f"   [DQ: {self.disqualifications[mid]}]"
            if mid == "uid0":
                tag = "   [burned]"
            share = val / total if mid != "uid0" else float("nan")
            share_s = f"{share*100:5.1f}%" if mid != "uid0" else "   -- "
            lines.append(f"    {mid:16s} {val:7.3f}  {share_s}{tag}")
        lines.append("")
        lines.append("  notes:")
        lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def admit_artifact(
    a: Artifact,
    live: dict[str, "LiveSub"],
    probe: np.ndarray,
    dup_threshold: float,
    disqualifications: dict[str, str],
    notes: list[str],
) -> None:
    """Sanity-gate + order-symmetric dedup (review fixes H1 and M1).

    H1: a miner's own prior artifact is exempt — refining and resubmitting your
    own router is the honest-improvement loop, never a "copy"; and a rejected
    resubmission never destroys the miner's previously-admitted artifact (it
    just doesn't replace it).
    M1: duplicates resolve by commit block symmetrically — the LATER commit
    loses regardless of admission order, so a copier claiming an earlier block
    cannot dodge comparison by arriving first. (On-chain, commit-reveal + salted
    hashes make block spoofing infeasible; the sim's dedup still must not depend
    on arrival order.)
    """
    sr = sanity_gate(a.head, a.theta, probe)
    if not sr.ok:
        disqualifications[a.miner_id] = f"sanity:{sr.reason}"
        return
    fp = fingerprint(a.head, a.theta, probe)
    duplicate_of: str | None = None
    for other_id, other in live.items():
        if other.disqualified or other_id == a.miner_id:
            continue
        if not is_duplicate(fp, other.fp, dup_threshold):
            continue
        if other.art.commit_block <= a.commit_block:
            duplicate_of = other_id           # incoming artifact is the later copy
            break
        # incoming artifact has the EARLIER block -> the existing one is the copy
        other.disqualified = True
        other.dq_reason = f"copy_of:{a.miner_id}"
        disqualifications[other_id] = f"copy_of:{a.miner_id}"
    if duplicate_of is not None:
        if a.miner_id in live and not live[a.miner_id].disqualified:
            notes.append(f"resubmission by {a.miner_id} rejected as copy_of:"
                         f"{duplicate_of}; prior artifact retained")
        else:
            live[a.miner_id] = LiveSub(a, fp, True, f"copy_of:{duplicate_of}")
            disqualifications[a.miner_id] = f"copy_of:{duplicate_of}"
        return
    live[a.miner_id] = LiveSub(a, fp)


def _sample_fresh(reservoir: OutcomeCache, epoch: int, batch_q: int) -> OutcomeCache:
    """Freshly (block-hash-seeded) sampled eval batch from the held-out reservoir.

    Without replacement within an epoch (review L4), and inheriting the
    reservoir's lambda normalizer (review L2) so scores are comparable across
    epochs. Cross-epoch overlap remains possible with a finite reservoir — the
    real subnet's "never reuse questions" needs a growing/synthetic source.
    """
    rng = np.random.default_rng(1000 + epoch)
    idx = rng.choice(reservoir.Q, size=min(batch_q, reservoir.Q), replace=False)
    return reservoir.subset(idx)


def run_competition(
    master: OutcomeCache,
    lam: float,
    epochs: int = 24,
    batch_q: int = 2500,
    probe_size: int = 256,
    seed: int = 0,
    verbose_setup: bool = False,
) -> CompetitionReport:
    corpus, reservoir = master.split(eval_frac=0.5, seed=seed)
    probe = reservoir.embeddings[:probe_size]          # validator's private probe set
    notes: list[str] = []

    # --- pre-train archetype routers on the public corpus -------------------
    # Quality is controlled by (hidden capacity, CMA-ES iters, training-data frac);
    # every archetype is an INDEPENDENT training run (distinct seed) so honest
    # routers are never mistaken for copies of one another.
    def trq(hidden: int, iters: int, frac: float, s: int) -> TrainedRouter:
        c = corpus
        if frac < 1.0:
            n = max(60, int(corpus.Q * frac))
            idx = np.random.default_rng(s).choice(corpus.Q, n, replace=False)
            c = corpus.subset(idx)
        return train_router(c, lam=lam, hidden=hidden, iters=iters, seed=s)

    # honest_improver: genuinely weak -> strong over four independent stages
    improver_stages = [
        trq(4, 15, 0.4, seed + 0),
        trq(8, 80, 1.0, seed + 1),
        trq(16, 180, 1.0, seed + 2),
        trq(24, 280, 1.0, seed + 3),
    ]
    early_bird_r = trq(24, 280, 1.0, seed + 20)   # strong, submitted early
    hoarder_r = trq(24, 280, 1.0, seed + 21)      # strong, withheld (distinct theta)
    camper_r = trq(8, 70, 0.7, seed + 22)         # mediocre, never improves
    overfit_r = trq(24, 450, 0.02, seed + 23)     # memorizes a tiny slice
    indies = [trq(16, 140, 1.0, seed + 40 + i) for i in range(3)]

    indie_fps = [fingerprint(r.head, r.theta, probe) for r in indies]
    cal = calibrate_threshold(indie_fps, margin=0.5)
    dup_threshold = cal["recommended_threshold"]
    notes.append(f"dedup calibration: min natural TV between independents="
                 f"{cal['min_natural_tv']:.3f}, threshold={dup_threshold:.3f}")

    # verify genuine quality ordering on a held-out sample (goes into notes)
    check = _sample_fresh(reservoir, 5555, batch_q)
    q_scores = {
        "improver[0]": improver_stages[0].evaluate(check).score,
        "improver[3]": improver_stages[3].evaluate(check).score,
        "early_bird": early_bird_r.evaluate(check).score,
        "camper": camper_r.evaluate(check).score,
        "overfitter": overfit_r.evaluate(check).score,
    }
    notes.append("held-out quality: " + ", ".join(f"{k}={v:.3f}" for k, v in q_scores.items()))
    if verbose_setup:
        print("[setup] " + notes[-1])

    def art(mid: str, block: int, r: TrainedRouter) -> Artifact:
        return Artifact(mid, hotkey=f"hk_{mid}", commit_block=block, head=r.head, theta=r.theta)

    # --- submission schedule: epoch -> list of Artifacts (copier handled live) -
    schedule: dict[int, list[Artifact]] = defaultdict(list)
    schedule[0] += [art("early_bird", 100, early_bird_r),
                    art("camper", 101, camper_r),
                    art("overfitter", 102, overfit_r)]
    schedule[1] += [art("honest_improver", 200, improver_stages[0])]
    schedule[5] += [art("honest_improver", 500, improver_stages[1])]
    schedule[10] += [art("honest_improver", 1000, improver_stages[2])]
    schedule[16] += [art("honest_improver", 1600, improver_stages[3])]
    hoarder_epoch = epochs - 3
    schedule[hoarder_epoch] += [art("hoarder", 100 + hoarder_epoch * 100, hoarder_r)]
    copier_epoch = 8
    degenerate_epoch = 2

    reign = Reign()
    live: dict[str, LiveSub] = {}
    cumulative: dict[str, float] = defaultdict(float)
    disqualifications: dict[str, str] = {}
    per_epoch_weights: list[dict[str, float]] = []
    slot_history: list[list[str]] = []

    def admit(a: Artifact) -> None:
        admit_artifact(a, live, probe, dup_threshold, disqualifications, notes)

    for e in range(epochs):
        for a in schedule.get(e, []):
            admit(a)
        # degenerate miner tries a constant "always cheapest" policy
        if e == degenerate_epoch:
            deg_head = RouterHead(master.D, master.K, hidden=8)
            deg_theta = deg_head.init_theta(seed=99, scale=0.0)  # all-zero -> uniform, then bias
            w1, b1, w2, b2 = deg_head.unpack(deg_theta)
            b2[0] = 50.0  # force argmax to model 0 everywhere -> constant policy
            deg_theta = np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])
            admit(Artifact("degenerate", "hk_degenerate", 100 + e * 100, deg_head, deg_theta))
        # copier clones the current champion at a later block
        if e == copier_epoch and reign.members:
            champ_id = reign.members[0].sub.miner_id
            if champ_id in live and not live[champ_id].disqualified:
                src = live[champ_id].art
                noisy = src.theta + np.random.default_rng(7).normal(0, 1e-3, size=src.theta.shape)
                admit(Artifact("copier", "hk_copier", 100 + e * 100 + 50, src.head, noisy))

        batch = _sample_fresh(reservoir, e, batch_q)
        candidates: list[Submission] = []
        for mid, ls in live.items():
            if ls.disqualified:
                continue
            choice = ls.art.head.choose(ls.art.theta, batch.embeddings)
            s = score_choice(batch, choice, lam).score
            candidates.append(Submission(mid, ls.art.hotkey, ls.art.commit_block, s))
        if not candidates:
            per_epoch_weights.append({})
            slot_history.append([""] * reign.n_slots)
            continue
        res = reign.update(candidates)
        for mid, w in res.weights.items():
            cumulative[mid] += w
        per_epoch_weights.append(res.weights)
        slot_history.append(res.slots)

    # final held-out scores
    final = _sample_fresh(reservoir, 9999, batch_q)
    final_scores: dict[str, float] = {}
    for mid, ls in live.items():
        if ls.disqualified:
            continue
        final_scores[mid] = score_choice(
            final, ls.art.head.choose(ls.art.theta, final.embeddings), lam).score

    # narrative notes
    if "copier" in disqualifications:
        notes.append(f"copier caught by fingerprint dedup: {disqualifications['copier']}")
    if "degenerate" in disqualifications:
        notes.append(f"degenerate rejected pre-eval: {disqualifications['degenerate']}")
    eb, ho = cumulative.get("early_bird", 0.0), cumulative.get("hoarder", 0.0)
    if eb > 0 or ho > 0:
        notes.append(f"early_bird {eb:.2f} vs hoarder {ho:.2f} cumulative "
                     f"(same router quality) -> hoarding forfeits {eb-ho:.2f} of emissions")
    notes.append(f"honest_improver final held-out score={final_scores.get('honest_improver', float('nan')):.4f}, "
                 f"overfitter={final_scores.get('overfitter', float('nan')):.4f}")

    return CompetitionReport(
        lam=lam, epochs=epochs, dup_threshold=dup_threshold,
        cumulative=dict(cumulative), disqualifications=disqualifications,
        per_epoch_weights=per_epoch_weights, slot_history=slot_history,
        final_scores=final_scores, notes=notes,
    )
