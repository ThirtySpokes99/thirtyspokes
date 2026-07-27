"""Rank-weighted top-5 reign — the emission mechanism (architecture 6).

Design goals it balances:
  * anti-hoarding pension tail  — 5 paid slots, so submitting early keeps paying
    even after you are surpassed (the SN9 winner-take-all hoarding failure).
  * anti-camping gradient       — slots pay ~40/25/15/12/8, a 5x slot1:slot5
    ratio, so there is always a reason to climb (Albedo's even split invited
    camping).
  * epsilon hysteresis          — an OUTSIDE challenger must beat a member's raw
    score by eps to be ranked ahead of it. eps decays with the member's
    ARTIFACT age, so fresh entrants are noise-protected while long incumbents
    can't squat on a stale marginal lead.
  * earliest-commit tiebreak    — exact ties resolve to the earlier commit block.
  * burn-to-UID-0               — empty slots and slots held by deregistered
    hotkeys pay nobody.

Epsilon semantics, post-review. The first implementation added eps as a score
bonus to every incumbent, which the adversarial review broke three ways:
  (M3) two incumbents of different ages got different bonuses, so a fresher
       member could dethrone an older champion with a strictly LOWER raw score;
  (M4) age was "epochs continuously held", so cycling out for one epoch reset
       age and refreshed full protection;
  (6f) a member re-committing a new artifact silently kept its old age.
Fixed by separating the two things eps was conflating:
  * protection is applied ONLY between members and outside challengers — members
    rank against each other by raw score alone;
  * age is keyed to the ARTIFACT (miner_id, commit_block) and measured from the
    epoch the artifact first appeared as a candidate, so leaving and re-entering
    the reign does not refresh protection;
  * a re-commit is a new artifact: it enters as an unprotected challenger with a
    fresh age clock (the architecture-6f deterrent), and its later commit block
    forfeits seniority tiebreaks.
Commit seniority (anti-copy economics) stays where it always was: in the
earliest-commit tiebreak and the fingerprint dedup upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SPLIT = (0.40, 0.25, 0.15, 0.12, 0.08)

ArtifactKey = tuple[str, int]  # (miner_id, commit_block)


@dataclass(frozen=True)
class Submission:
    miner_id: str
    hotkey: str
    commit_block: int
    score: float

    @property
    def artifact(self) -> ArtifactKey:
        return (self.miner_id, self.commit_block)


@dataclass
class Member:
    sub: Submission
    age: int  # epochs since this ARTIFACT first appeared as a candidate


@dataclass
class EpochResult:
    epoch: int
    weights: dict[str, float]        # miner_id -> emission weight (slots)
    burn: float                      # weight routed to UID 0
    slots: list[str]                 # miner_id per slot, "" if empty/burned
    coronation: bool                 # did slot 1 change hands this epoch


class Reign:
    def __init__(
        self,
        split: tuple[float, ...] = DEFAULT_SPLIT,
        eps0: float = 0.02,
        eps_floor: float = 0.002,
        tau: float = 8.0,
        burn_uid: str = "uid0",
    ):
        self.split = split
        self.n_slots = len(split)
        self.eps0 = eps0
        self.eps_floor = eps_floor
        self.tau = tau
        self.burn_uid = burn_uid
        self.members: list[Member] = []          # ordered slot 1..n
        self._first_seen: dict[ArtifactKey, int] = {}  # artifact -> epoch first candidate
        self._epoch = 0

    # --- persistence (production restart-safety): the reign standings + artifact ages must
    #     survive a validator restart, else the incumbency/eps history resets each restart ---
    def snapshot(self) -> dict:
        return {
            "epoch": self._epoch,
            "members": [{"miner_id": m.sub.miner_id, "hotkey": m.sub.hotkey,
                         "commit_block": m.sub.commit_block, "score": m.sub.score, "age": m.age}
                        for m in self.members],
            "first_seen": [{"miner_id": k[0], "commit_block": k[1], "epoch": v}
                           for k, v in self._first_seen.items()],
        }

    def restore(self, snap: dict) -> None:
        self._epoch = int(snap.get("epoch", 0))
        self.members = [Member(Submission(m["miner_id"], m["hotkey"], int(m["commit_block"]),
                                          float(m["score"])), int(m["age"]))
                        for m in snap.get("members", [])]
        self._first_seen = {(e["miner_id"], int(e["commit_block"])): int(e["epoch"])
                            for e in snap.get("first_seen", [])}

    def eps(self, age: int) -> float:
        """Protection for a member artifact: high when fresh, decays with age."""
        return self.eps_floor + (self.eps0 - self.eps_floor) * float(np.exp(-age / self.tau))

    def _artifact_age(self, sub: Submission) -> int:
        """Epochs of contest exposure since the artifact first appeared.

        Offset so an artifact first offered (and crowned) at epoch e has age 0 at
        its first defense in epoch e+1 (full eps0), decaying thereafter. Age is
        keyed to the artifact and keeps counting while the artifact remains a
        candidate — leaving the reign does not pause or reset it (review M4).
        On-chain, a commitment can't be paused-and-resumed either: withdrawing
        means re-committing, which is a new artifact at a later block.
        """
        return max(0, self._epoch - self._first_seen[sub.artifact] - 1)

    def update(
        self,
        candidates: list[Submission],
        deregistered: set[str] | None = None,
        live: set[str] | None = None,
    ) -> EpochResult:
        """Advance one epoch: re-rank all candidates, apply eps hysteresis, pay.

        `live` (the miners that produced a VALID proof this epoch) gates payout exactly as it does
        in `KingChain` — a slot held by an absent miner burns rather than pays. `None` treats every
        candidate as live, which is the offline-sim shape and leaves existing callers unchanged.
        """
        ids = [c.miner_id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate miner_id in candidates (one submission per miner)")
        self._epoch += 1
        dereg = deregistered or set()
        live_ids = set(live) if live is not None else {c.miner_id for c in candidates}
        prev_champion = self.members[0].sub.miner_id if self.members else ""

        # register first-seen epoch per artifact; prune artifacts no longer offered
        offered = {c.artifact for c in candidates}
        for c in candidates:
            self._first_seen.setdefault(c.artifact, self._epoch)
        self._first_seen = {k: v for k, v in self._first_seen.items() if k in offered}

        # split candidates into current members (SAME artifact as the one holding
        # the slot — a re-commit is a new artifact and enters as a challenger)
        member_artifacts = {m.sub.artifact for m in self.members}
        member_subs = [c for c in candidates if c.artifact in member_artifacts]
        challenger_subs = [c for c in candidates if c.artifact not in member_artifacts]

        by_raw = lambda s: (-s.score, s.commit_block, s.miner_id)  # noqa: E731
        member_subs.sort(key=by_raw)          # members vs members: raw score only (M3)
        challenger_subs.sort(key=by_raw)

        # merge: a challenger is placed ahead of a member only if it beats the
        # member's raw score by that member's (artifact-age) eps
        merged: list[Submission] = []
        mi = ci = 0
        while mi < len(member_subs) and ci < len(challenger_subs):
            m, c = member_subs[mi], challenger_subs[ci]
            beats = c.score > m.score + self.eps(self._artifact_age(m)) + 1e-12
            ties = c.score == m.score and c.commit_block < m.commit_block
            if beats or ties:
                merged.append(c)
                ci += 1
            else:
                merged.append(m)
                mi += 1
        merged.extend(member_subs[mi:])
        merged.extend(challenger_subs[ci:])

        new_top = merged[: self.n_slots]
        self.members = [Member(sub, self._artifact_age(sub)) for sub in new_top]

        # emissions: split across slots; burn empty / deregistered-hotkey slots
        weights: dict[str, float] = {}
        slots: list[str] = []
        burn = 0.0
        for i in range(self.n_slots):
            w = self.split[i]
            if (i < len(self.members) and self.members[i].sub.hotkey not in dereg
                    and self.members[i].sub.miner_id in live_ids):
                mid = self.members[i].sub.miner_id
                weights[mid] = weights.get(mid, 0.0) + w
                slots.append(mid)
            else:
                burn += w
                slots.append("")
        if burn > 0:
            weights[self.burn_uid] = weights.get(self.burn_uid, 0.0) + burn

        champion = self.members[0].sub.miner_id if self.members else ""
        return EpochResult(
            epoch=self._epoch,
            weights=weights,
            burn=burn,
            slots=slots,
            coronation=(champion != prev_champion and champion != ""),
        )


# --- king + equal-share chain (the KOTH emission mechanism) ----------------------------------
KING_CHAIN_SIZE = 5          # the king plus up to 4 recent ex-kings


class KingChain:
    """Single king + an equal-share chain of recent ex-kings.

    Emissions are split EQUALLY across the current king and the last `chain_size - 1` miners who
    held the crown and are still registered (nobody registered -> burn). Drop-in for `Reign`: same
    `members` / `burn_uid` / `update` / `snapshot` / `restore` surface, so the validator is unchanged.

    Why a chain rather than winner-take-all: a dethroned king that KEEPS COMPETING goes on earning
    while it decays out of the chain, which is the anti-hoarding pension tail — a miner beaten by
    0.001 does not drop to zero, so there is no cliff to camp against. Why equal-share rather than a
    graded split: the reward for climbing is *entering the chain at all*, and the only door into it
    is taking the crown, so the gradient lives at the coronation rather than in the slot weights.

    LIVENESS (the pension is earned per epoch, never vested). A seat pays only while its holder is
    still submitting valid proofs. Without this the mechanism paid for *membership* rather than
    *work*, and three exploits followed, all reproduced end-to-end against this class:
      * a miner that took the crown once and then went dark drew its full share forever (measured:
        one epoch of mining with the CHEAPEST pool model captured 54% of emissions over 12 epochs,
        out-earning the honest miner that worked every epoch);
      * because the king earns exactly what an idle ex-king earns, a cartel could rotate the crown
        through 5 self-owned hotkeys and then stop entirely, capping an arbitrarily better honest
        miner at 1/5 of the pot;
      * withholding a single epoch used to hand the crown to ANY submitter, unguarded — a 21-epoch
        champion lost its crown to a 0.01-scoring agent by missing one upload.
    So: `live` (the miners that produced a valid proof THIS epoch) gates every payout; a seat missing
    `absent_grace` consecutive epochs is evicted; and an absent king keeps its title (unpaid) and
    goes on setting the eps bar until its grace runs out, so absence can never be a cheap handoff.
    A king that loses the crown by going dark does NOT get a pension seat — it is delinquent by
    construction.

    What carries over from `Reign` unchanged, because the adversarial review depends on it:
      * eps hysteresis — an OUTSIDE challenger must beat the king's raw score by eps, so noise
        cannot flip the crown;
      * earliest-commit seniority — exact ties go to the earlier commit block.
    Copy-mining is still killed upstream (`behavioral_duplicates` DQs a copy as `copy_of:<orig>`
    before it ever reaches here), so a copy cannot take the crown and cannot enter the chain.
    """

    def __init__(
        self,
        chain_size: int = KING_CHAIN_SIZE,
        eps0: float = 0.02,
        eps_floor: float = 0.002,
        tau: float = 8.0,
        burn_uid: str = "uid0",
        absent_grace: int = 3,
    ):
        if absent_grace < 1:
            raise ValueError("absent_grace must be >= 1")
        self.chain_size = chain_size
        self.eps0 = eps0
        self.eps_floor = eps_floor
        self.tau = tau
        self.burn_uid = burn_uid
        # consecutive epochs a seat may miss before it is evicted / the crown vacates. >1 because a
        # real miner's confidential-VM boot is flaky (~30% of cold boots failed on testnet 526), so a
        # single missed upload must not cost a working miner its seat.
        self.absent_grace = absent_grace
        self.king: Member | None = None
        self.chain: list[Submission] = []              # ex-kings, most recent first
        self._first_seen: dict[ArtifactKey, int] = {}
        self._epoch = 0
        self._misses: dict[str, int] = {}              # miner_id -> consecutive epochs absent
        self._king_since = 0                           # epoch the CURRENT king's miner took the crown
        self._last_weights: dict[str, float] = {}      # last payout, for the standings feed

    # `members` exists so a KingChain duck-types as a Reign for the validator: members[0] is the king.
    @property
    def members(self) -> list[Member]:
        out = [self.king] if self.king else []
        out += [Member(s, 0) for s in self.chain]
        return out

    def eps(self, age: int) -> float:
        return max(self.eps_floor, self.eps0 * float(np.exp(-age / self.tau)))

    def _artifact_age(self, sub: Submission) -> int:
        return self._epoch - self._first_seen.get(sub.artifact, self._epoch)

    def _king_age(self) -> int:
        """Epochs the CURRENT miner has held the crown — the eps clock.

        Keyed to the REIGN, not the artifact. Artifact-keyed age silently waived a king's earned
        protection the moment it re-committed: a 15-epoch champion that improved its agent lost the
        crown to a challenger it still out-scored, because the new artifact key made the incumbent
        lookup miss. Punishing the exact behavior the subnet wants is backwards. Reign-keyed age
        also can't be refreshed by re-committing every epoch, so it opens no squatting vector.
        """
        return max(0, self._epoch - self._king_since)

    def snapshot(self) -> dict:
        def enc(s: Submission) -> dict:
            return {"miner_id": s.miner_id, "hotkey": s.hotkey,
                    "commit_block": s.commit_block, "score": s.score}
        return {
            "epoch": self._epoch,
            "king": ({**enc(self.king.sub), "age": self.king.age} if self.king else None),
            "chain": [enc(s) for s in self.chain],
            "first_seen": [[list(k), v] for k, v in self._first_seen.items()],
            # liveness state must survive a restart, else every restart forgives every absence and
            # the pension exploit reopens for `absent_grace` epochs at a time.
            "misses": dict(self._misses),
            "king_since": self._king_since,
            "last_weights": dict(self._last_weights),
        }

    def restore(self, snap: dict) -> None:
        self._epoch = int(snap.get("epoch", 0))
        k = snap.get("king")
        self.king = (Member(Submission(k["miner_id"], k["hotkey"], int(k["commit_block"]),
                                       float(k["score"])), int(k.get("age", 0)))
                     if k else None)
        self.chain = [Submission(e["miner_id"], e["hotkey"], int(e["commit_block"]),
                                 float(e["score"])) for e in snap.get("chain", [])]
        self._first_seen = {(str(k[0]), int(k[1])): int(v) for k, v in snap.get("first_seen", [])}
        self._misses = {str(m): int(n) for m, n in snap.get("misses", {}).items()}
        self._king_since = int(snap.get("king_since", self._epoch))
        self._last_weights = {str(m): float(w) for m, w in snap.get("last_weights", {}).items()}

    def update(
        self,
        candidates: list[Submission],
        deregistered: set[str] | None = None,
        live: set[str] | None = None,
    ) -> EpochResult:
        """Advance one epoch. `live` = the miners that produced a VALID proof this epoch; only they
        can be paid or crowned. It is passed explicitly rather than inferred from `candidates`
        because under `scoring_mode="accumulate"` a miner that missed the epoch is still scored
        (miss=0) and so still appears as a candidate — inferring liveness there would pay absentees.
        `live=None` (offline sims, the graduated-`Reign` call shape) treats every candidate as live.
        """
        ids = [c.miner_id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate miner_id in candidates (one submission per miner)")
        self._epoch += 1
        dereg = deregistered or set()
        prev_king_id = self.king.sub.miner_id if self.king else ""
        live_ids = set(live) if live is not None else {c.miner_id for c in candidates}

        offered = {c.artifact for c in candidates}
        for c in candidates:
            self._first_seen.setdefault(c.artifact, self._epoch)
        self._first_seen = {k: v for k, v in self._first_seen.items() if k in offered}

        # --- liveness bookkeeping: a seat's miss counter resets only by actually submitting ---
        seated = ([self.king.sub.miner_id] if self.king else []) + [s.miner_id for s in self.chain]
        for mid in seated:
            self._misses[mid] = 0 if mid in live_ids else self._misses.get(mid, 0) + 1
        self._misses = {m: n for m, n in self._misses.items() if m in seated}   # bounded
        # evict delinquent ex-kings BEFORE the coronation, so a freed slot is usable this epoch
        self.chain = [s for s in self.chain if self._misses.get(s.miner_id, 0) < self.absent_grace]
        king_absent = self.king is not None and self.king.sub.miner_id not in live_ids
        king_delinquent = king_absent and self._misses.get(
            self.king.sub.miner_id, 0) >= self.absent_grace

        # --- coronation: only a LIVE miner can be crowned, and only by clearing the king's eps ---
        ranked = sorted((c for c in candidates if c.miner_id in live_ids),
                        key=lambda s: (-s.score, s.commit_block, s.miner_id))
        top = ranked[0] if ranked else None
        # the incumbent's OWN submission this epoch (matched by miner_id, not artifact — see
        # `_king_age`: a re-commit must not forfeit the crown it already holds)
        incumbent = next((c for c in ranked
                          if self.king and c.miner_id == self.king.sub.miner_id), None)

        def _clears(challenger: Submission, held: Submission) -> bool:
            return (challenger.score > held.score + self.eps(self._king_age()) + 1e-12
                    or (challenger.score == held.score
                        and challenger.commit_block < held.commit_block))

        if self.king is None:
            new_king_sub = top                              # genuinely vacant: best live miner takes it
        elif incumbent is not None:
            new_king_sub = (top if top is not None and top.miner_id != incumbent.miner_id
                            and _clears(top, incumbent) else incumbent)
        elif not king_delinquent:
            # ABSENT but inside grace: the king keeps the title (it is NOT paid below) and still sets
            # the bar with its last score, so withholding an epoch cannot hand the crown to a worse
            # agent. This is the fix for the one-epoch-handoff takeover.
            new_king_sub = (top if top is not None and _clears(top, self.king.sub)
                            else self.king.sub)
        else:
            new_king_sub = top                              # grace exhausted: the crown truly vacates

        if new_king_sub is not None:
            if prev_king_id and new_king_sub.miner_id != prev_king_id and self.king is not None:
                # a king dethroned WHILE COMPETING earns the pension seat; one that lost the crown by
                # going dark does not — it is delinquent by construction.
                if not king_delinquent:
                    past = self.king.sub                      # dethroned -> front of the chain
                    self.chain = [s for s in self.chain if s.miner_id != past.miner_id]
                    self.chain.insert(0, past)
                    self.chain = self.chain[: self.chain_size - 1]
                self._king_since = self._epoch                # new reign -> fresh eps clock
            # the sitting king can never also be an ex-king entry
            self.chain = [s for s in self.chain if s.miner_id != new_king_sub.miner_id]
            self.king = Member(new_king_sub, self._king_age())
        elif king_delinquent:
            self.king = None                                  # dead king, no live successor

        # --- payout: EQUAL share across the king + ex-kings that are BOTH registered AND live ---
        paid: list[str] = []
        if (self.king and self.king.sub.hotkey not in dereg
                and self.king.sub.miner_id in live_ids):
            paid.append(self.king.sub.miner_id)
        for s in self.chain:
            if s.hotkey not in dereg and s.miner_id in live_ids and s.miner_id not in paid:
                paid.append(s.miner_id)

        weights: dict[str, float] = {}
        burn = 0.0
        if paid:
            w = 1.0 / len(paid)
            for mid in paid:
                weights[mid] = w
        else:
            burn = 1.0
            weights[self.burn_uid] = 1.0
        self._last_weights = dict(weights)

        king_id = self.king.sub.miner_id if self.king else ""
        slots = ([king_id] if king_id else []) + [s.miner_id for s in self.chain]
        slots += [""] * (self.chain_size - len(slots))
        return EpochResult(
            epoch=self._epoch,
            weights=weights,
            burn=burn,
            slots=slots[: self.chain_size],
            coronation=(king_id != prev_king_id and king_id != ""),
        )
