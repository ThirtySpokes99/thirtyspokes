"""The KOTH validator — verify a proof, grade the reports, crown the king (docs/DESIGN.md §4).

Per epoch, for each committed miner the validator:
  1. downloads the PUBLIC bundle itself, recomputes source/weights hashes, and checks
     them against the on-chain commit (`verify_commit`) — the binding is enforced here,
     not taken on the miner's word;
  2. `verify_proof`s the attested proof against those recomputed hashes (no inference);
  3. runs ONE shared held-out probe over the *downloaded* artifact (the only inference
     the validator does — a bounded spot-check), feeding both the statistical
     memorization test and behavioral copy-dedup;
  4. scans the downloaded source for hardcoded answers;
  5. applies the Pareto dethrone guard (clamping on ANY failure) against the persisted
     king baseline, then feeds the cost-aware scalar + commit-block seniority to `KingChain`.

Reuses `reign.KingChain` (king + equal-share ex-king chain) + `subnet.chain` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from collections import Counter

from ..gateway import signing
from ..gateway.gateway import ModelBackend
from ..reign import Reign, Submission
from . import commit as commitmod
from .benchmarks import bench_seed
from .epoch import EPOCH_BLOCKS, current_epoch, epoch_nonce
from .evidence import EvidenceStore
from .sandbox import SandboxError, run_agent_probe
from .store import hash_source, hash_weights, is_pinned_revision
from .verify import (
    BenchStat,
    ProofVerdict,
    behavioral_duplicates,
    cohort_probe_allowance,
    dethrone_guard,
    eligible,
    grounding_check,
    memorization_collapsed_relative,
    router_headroom,
    scan_source,
    scan_weights,
    verify_proof,
)


# A registered miner that produced no VALID scored proof this epoch — didn't upload (`no_proof`), or
# uploaded but failed the F7 commit binding. All count as a MISS (n_expected tasks, 0 correct) in
# accumulate mode, so withholding a bad/absent epoch can't inflate the pooled average (docs/DESIGN.md §5b).
HISTORY_LEN = 200          # epochs of rolling history kept for the public feed

_MISS_REASONS = frozenset({"no_proof", "no_proof_commit", "commit_out_of_window", "commit_mismatch"})


@dataclass
class EpochReport:
    epoch: int
    scored: dict = field(default_factory=dict)       # hotkey -> scalar S
    dq: dict = field(default_factory=dict)            # hotkey -> reason
    slots: list = field(default_factory=list)
    weights_by_uid: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


@dataclass
class _MinerEval:
    """Outcome of `_score_one_miner`: a DQ reason, or a scored verdict + its dedup fingerprint
    (+ the probe-mode memorization audit tuple, if any). `sh`/`wh` are the bound artifact hashes,
    set once past the commit check (the accumulator keys on them; None if never bound)."""
    dq: str | None = None
    verdict: ProofVerdict | None = None
    fingerprint: tuple | None = None
    audit: tuple | None = None
    sh: str | None = None
    wh: str | None = None


class KOTHValidator:
    def __init__(self, approved_measurements, platform_public_hex, chain, reign: Reign,
                 suite, store, backend: ModelBackend, *, n_per_bench: int = 8,
                 budget: float = 0.5, f_min: float = 0.1, margin: float = 0.03,
                 tol: float = 0.02, cost_tol: float = 0.10, min_tasks: int = 5,
                 cost_margin: float = 0.10,
                 dedup_agree: float = 0.95, epoch_blocks: int = EPOCH_BLOCKS,
                 pool_spec: dict | None = None, approved_mrtd=None, approved_rtmr=None,
                 tcb_accept: frozenset | None = None, collateral: str | None = None,
                 pccs_url: str | None = None, enforce: bool = False, probe_bank=None,
                 min_cohort: int = 3, max_probe_drop: float = 0.25,
                 audit_mode: str = "grounding", scoring_mode: str = "accumulate",
                 half_life_epochs: float = 200.0, budget_per_task: float = 0.02,
                 n_expected: int | None = None, commit_window: int | None = None,
                 grace_blocks: int = 0, cost_tiebreak: float = 0.02,
                 pool_reference=None):
        # memorization backstop: "grounding" (default) = pure proof-inspection, validator runs NO
        # miner code; "probe" = the legacy re-execution fresh-probe (null-pool/secret-bank upgrade).
        self.audit_mode = audit_mode
        # scoring: "accumulate" (DEFAULT, docs/DESIGN.md §5b) pools per-artifact evidence (Wilson-LCB,
        # EWMA decay, reset-on-recommit, miss=0) so 32 q/epoch ranks stably and decoupled validators
        # agree; budget is cost-PER-TASK there. "per_epoch" scores each slice independently — kept for
        # the offline sim, but it makes the crown a per-epoch lottery: a single lucky slice can
        # dethrone, and nothing costs a miner for the epochs it hides.
        self.scoring_mode = scoring_mode
        self.budget_per_task = budget_per_task
        self._evidence = EvidenceStore(half_life_epochs)
        self.pool_spec = pool_spec or {"kind": "mock"}    # sandbox runs the probe against this pool
        # secret owner-held memorization probe (koth/heldout.py). When present + verified against
        # the on-chain probe_commit, the fresh-probe audit draws from it (miners can't derive it);
        # else it falls back to the derivable public probe. See docs/DESIGN.md §6/§9.
        self.probe_bank = probe_bank
        # enforce=True is production: refuse to score anything but a real hardware quote gated
        # on an owner-pinned measured image (fail-closed, docs/DESIGN.md §8). Default False keeps the
        # offline sim / mock-TEE dev runs working. The deployable daemon sets it True.
        self.enforce = enforce
        self.approved = set(approved_measurements)
        # hardware-attestation gates (H1/H2): MRTD + RTMR image measurements + TCB policy.
        # Off (None) by default; populated by the constructor or per-epoch on-chain governance.
        self.approved_mrtd = set(approved_mrtd) if approved_mrtd else None
        self.approved_rtmr = dict(approved_rtmr) if approved_rtmr else None
        self.tcb_accept = tcb_accept
        self.collateral, self.pccs_url = collateral, pccs_url
        self.platform_pub = platform_public_hex
        self.chain, self.reign, self.suite = chain, reign, suite
        self.store, self.backend = store, backend
        self.n_per_bench, self.budget, self.f_min = n_per_bench, budget, f_min
        self.n_expected = n_expected if n_expected is not None else n_per_bench   # miss=0 slice size
        self.guard = dict(margin=margin, tol=tol, cost_tol=cost_tol, min_tasks=min_tasks,
                          cost_margin=cost_margin)
        # Cost gradient. Q_lcb saturates (a perfect agent scores the ceiling and nothing ranks above
        # it), so with accuracy alone every good miner ties and the reign orders them by commit-block
        # seniority — emissions freeze on the earliest committer. Subtracting a SMALL cost term gives
        # a gradient that never runs out: at equal quality the cheaper agent ranks higher. Kept small
        # (0.02) so it cannot override a genuine per-benchmark accuracy difference — quality first,
        # cost as the tiebreak. `dethrone_guard(cost_margin=…)` is the matching rule for slot 1.
        self.cost_tiebreak = cost_tiebreak
        # ROUTER SCALAR (the product goal: best answer at the lowest price, for a given ask).
        # `pool_reference(epoch, nonce) -> (scores, costs)` gives every (task, pool-model) cell for
        # this epoch's slice, published by the OWNER -- a validator runs no inference and so cannot
        # know what the other models would have answered. With it the scalar becomes
        # `router_headroom`: 0.0 = no better than randomising over fixed pool models AT THIS PRICE,
        # 1.0 = matched the budget-constrained per-query oracle there. That scores the DECISION
        # rather than the outcome, cancels out the pool's own capability (on a 95%-accurate pool the
        # old absolute scalar was ~98% pool and ~2% miner), and prices quality against cost on a
        # frontier instead of at one owner-chosen lambda. Without a reference it falls back to the
        # old Q_lcb - lambda*cost, so offline sims and existing deployments are unchanged.
        self.pool_reference = pool_reference
        self.dedup_agree = dedup_agree
        # cohort-relative memorization test: need >= min_cohort audited miners to calibrate the
        # probe's difficulty; max_probe_drop caps the allowance so a colluding all-memorizer
        # cohort cannot inflate it (the owner's "no honest probe is harder than this").
        self.min_cohort, self.max_probe_drop = min_cohort, max_probe_drop
        self.epoch_blocks = epoch_blocks
        # F7 anti-grind: if set, a scored proof must have been COMMITTED on-chain (report_data) inside
        # [epoch*epoch_blocks, +commit_window] blocks and revealed EXACTLY — binding one run (no
        # post-commit best-of-N swap) and capping pre-commit grinding to the window's wall-clock. Off
        # (None) by default; the mainnet validator sets it (docs/DESIGN.md §5b). Miners must commit_proofs.
        self.commit_window = commit_window
        # F2 grace window: score a SETTLED epoch — the latest whose grace deadline has passed
        # (`(current_block - grace_blocks) // epoch_blocks`) — never the live epoch, so every miner
        # had the full commit+upload window and decoupled validators agree on presence. 0 (default)
        # keeps the synchronous sim scoring the current epoch. Keep grace_blocks < epoch_blocks so a
        # real single-slot chain still holds the scored epoch's proof commit (see BittensorChain).
        self.grace_blocks = grace_blocks
        self._king_id: str | None = None
        self._king_vd: ProofVerdict | None = None
        self._last_gov: dict | None = None      # F9: last-known-good on-chain governance record
        # rolling per-epoch history for the public feed (the dashboard's Q_lcb chart + verdict log).
        # Capped, and persisted with the rest of the state so a restart doesn't blank the chart.
        self._history: list[dict] = []
        # Production neuron transaction state. It stages the completed epoch + weights on disk
        # BEFORE the chain write, then clears them only after inclusion. A kill at any point can
        # therefore resume without rescoring the epoch or silently losing the weight update.
        self._last_scored_epoch: int | None = None
        self._pending_weights: dict[int, float] | None = None
        self._pending_report: EpochReport | None = None
        self._last_submitted_weights: dict[int, float] | None = None

    def register(self, hotkey: str) -> None:
        self.chain.register(hotkey)

    # --- the public leaderboard feed -------------------------------------------------------------
    def standings(self, rep: "EpochReport | None" = None) -> dict:
        """The dashboard payload: the king, the ex-king chain, and this epoch's scores + DQs.

        Identity (uid, repo) is resolved against the LIVE metagraph at flush time, never cached from
        when the proof landed: hotkeys deregister and re-register under a new uid, so a uid stored at
        insert time goes stale and the leaderboard would show the wrong miner.
        """
        uid_of = {hk: uid for uid, hk in self.chain.hotkeys().items()}
        repo_of: dict[str, str] = {}
        for c in self.chain.revealed_commitments():
            parsed = commitmod.parse_commit(c.data)
            if parsed:
                repo_of[c.hotkey] = parsed[0]

        try:
            market = self.chain.market()
        except Exception:  # noqa: BLE001 — presentational; a price read must never break scoring
            market = None

        # Alpha actually earned per hour = this miner's slice of the MINER pot (the subnet's gross
        # emission scaled by the on-chain miner share), not of the gross. USD only if a TAO/USD rate
        # was supplied — we never invent one.
        def pay(share: float) -> dict:
            if not market:
                return {"alpha_per_hour": None, "usd_per_hour": None}
            aph = share * market["alpha_per_block_miners"] * market["blocks_per_hour"]
            usd = market.get("alpha_price_usd")
            return {"alpha_per_hour": round(aph, 6),
                    "usd_per_hour": (round(aph * usd, 4) if usd else None)}

        def entry(hk: str, score: float, *, king: bool, share: float) -> dict:
            return {"hotkey": hk, "uid": uid_of.get(hk), "repo": repo_of.get(hk),
                    "q_lcb": round(float(score), 6), "king": king,
                    "weight": round(share, 9), **pay(share)}

        members = self.reign.members
        # Each seat's share comes from the emission mechanism's ACTUAL last payout, not from
        # 1/len(members): a seat whose holder missed the epoch is seated but unpaid (liveness), so
        # assuming an even split across seats would advertise emissions nobody received.
        last_w = getattr(self.reign, "_last_weights", None) or {}
        sh = lambda m: round(float(last_w.get(m.sub.miner_id, 0.0)), 9)   # noqa: E731
        king = entry(members[0].sub.hotkey, members[0].sub.score, king=True,
                     share=sh(members[0])) if members else None
        chain = [entry(m.sub.hotkey, m.sub.score, king=False, share=sh(m)) for m in members[1:]]

        kvd = self._king_vd
        return {
            "schema": 1,
            # Provenance is part of the feed: a testnet replay must never be presented as mainnet.
            "network": getattr(self.chain, "network", None),
            "netuid": getattr(self.chain, "netuid", None),
            "epoch": (rep.epoch if rep else self.reign._epoch),
            "mechanism": {"kind": "king+equal_share_chain",
                          "chain_size": getattr(self.reign, "chain_size", None),
                          # equal across the seats PAID this epoch (absent seats pay nothing)
                          "weight_each": round(max(last_w.values()), 9) if last_w else 0.0,
                          "n_paid": sum(1 for m in last_w if m != self.reign.burn_uid)},
            "king": ({**king,
                      "cost_usd": round(kvd.total_cost_usd, 6) if kvd else None,
                      "per_bench": ({n: {"acc": round(bs.acc, 4), "lcb": round(bs.lcb, 4), "n": bs.n}
                                     for n, bs in kvd.per_bench.items()} if kvd else {})}
                     if king else None),
            "chain": chain,
            "suite": [{"name": b.name, "weight": round(float(b.weight), 4),
                       "floor": self.f_min, "n_per_epoch": self.n_per_bench}
                      for b in self.suite],
            "budget_usd": self.budget,
            "scored": {hk: round(float(s), 6) for hk, s in (rep.scored if rep else {}).items()},
            "dq": dict(rep.dq) if rep else {},
            "weights_by_uid": {str(u): round(w, 9) for u, w in (rep.weights_by_uid if rep else {}).items()},
            "burn_uid": self.reign.burn_uid,
            "market": market,             # None offline / on a price-read failure -> no $ columns
            "block": self.chain.current_block(),
            "history": self._history[-60:],          # newest last; the dashboard charts this
        }

    def flush_standings(self, repo: str, rep: "EpochReport | None" = None) -> bool:
        """Publish `standings.json` for the dashboard. PRESENTATIONAL — it must NEVER raise into the
        scoring loop: a store outage would otherwise surface as a scoring failure for an epoch that
        was already scored correctly. Returns True if published."""
        import json
        substrate = getattr(getattr(self.chain, "subtensor", None), "substrate", None)
        old_timeout = getattr(substrate, "retry_timeout", None)
        old_retries = getattr(substrate, "max_retries", None)
        try:
            # `standings()` performs live metagraph/market reads. A public RPC stopped responding
            # here during a real epoch and held an already-successful weight transaction open. The
            # feed is presentational, so give its reads one short attempt and let the next epoch
            # refresh it if this one cannot.
            if old_timeout is not None:
                substrate.retry_timeout = min(float(old_timeout), 10.0)
            if old_retries is not None:
                substrate.max_retries = 1
            self.store.upload_standings(repo, json.dumps(self.standings(rep), default=str))
            self._standings_failed = False
            return True
        except Exception as e:  # noqa: BLE001 — presentational; must not break scoring
            # ...but it must not be SILENT either: a permanently-dead feed used to look identical to a
            # healthy one from inside the daemon. Warn once per failure streak.
            if not getattr(self, "_standings_failed", False):
                self._standings_failed = True
                print(f"[koth-validator] WARNING: standings publish to {repo} failed "
                      f"({type(e).__name__}: {e}). The dashboard will show no feed. "
                      f"Scoring and weights are unaffected.")
            return False
        finally:
            if old_timeout is not None:
                substrate.retry_timeout = old_timeout
            if old_retries is not None:
                substrate.max_retries = old_retries

    # --- restart-safety: persist the reign standings + the king dethrone-guard baseline, so a
    #     validator restart does not reset incumbency/eps history or disable the Pareto guard ---
    def snapshot(self) -> dict:
        king = None
        if self._king_vd is not None:
            king = {"id": self._king_id, "total_cost_usd": self._king_vd.total_cost_usd,
                    "score": self._king_vd.score, "total_score": self._king_vd.total_score,
                    "per_bench": {n: {"n": bs.n, "acc": bs.acc, "lcb": bs.lcb, "cost_usd": bs.cost_usd}
                                  for n, bs in self._king_vd.per_bench.items()}}
        pending_report = None
        if self._pending_report is not None:
            rep = self._pending_report
            pending_report = {
                "epoch": rep.epoch, "scored": rep.scored, "dq": rep.dq,
                "slots": rep.slots, "weights_by_uid": rep.weights_by_uid, "audit": rep.audit,
            }
        return {"reign": self.reign.snapshot(), "king": king, "evidence": self._evidence.snapshot(),
                "history": self._history, "last_scored_epoch": self._last_scored_epoch,
                "pending_weights": self._pending_weights, "pending_report": pending_report,
                "last_submitted_weights": self._last_submitted_weights}

    def restore(self, snap: dict) -> None:
        self._history = list(snap.get("history") or [])
        if snap.get("reign"):
            self.reign.restore(snap["reign"])
        if snap.get("evidence"):
            self._evidence.restore(snap["evidence"])
        last = snap.get("last_scored_epoch")
        self._last_scored_epoch = int(last) if last is not None else None
        pending = snap.get("pending_weights")
        self._pending_weights = ({int(uid): float(weight) for uid, weight in pending.items()}
                                 if pending is not None else None)
        pending_report = snap.get("pending_report")
        self._pending_report = (EpochReport(
            int(pending_report["epoch"]), dict(pending_report.get("scored") or {}),
            dict(pending_report.get("dq") or {}), list(pending_report.get("slots") or []),
            {int(uid): float(weight)
             for uid, weight in (pending_report.get("weights_by_uid") or {}).items()},
            dict(pending_report.get("audit") or {}),
        ) if pending_report else None)
        submitted = snap.get("last_submitted_weights")
        self._last_submitted_weights = (
            {int(uid): float(weight) for uid, weight in submitted.items()}
            if submitted is not None else None)
        self._king_id = None
        self._king_vd = None
        k = snap.get("king")
        if k:
            self._king_id = k["id"]
            self._king_vd = ProofVerdict(
                True, "ok",
                {n: BenchStat(bs["n"], bs["acc"], bs["lcb"], bs["cost_usd"])
                 for n, bs in k["per_bench"].items()},
                k["total_cost_usd"], k["score"], k.get("total_score", 0.0))

    def save_state(self, path: str) -> None:
        import json
        import os
        import tempfile

        parent = os.path.dirname(os.path.abspath(path))
        fd, temporary = tempfile.mkstemp(prefix=".koth-validator-state-", dir=parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.snapshot(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_state(self, path: str) -> bool:
        import json
        import os
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.restore(json.load(f))
        return True

    @property
    def last_scored_epoch(self) -> int | None:
        return self._last_scored_epoch

    @property
    def pending_weights(self) -> dict[int, float] | None:
        return (dict(self._pending_weights) if self._pending_weights is not None else None)

    @property
    def pending_report(self) -> EpochReport | None:
        return self._pending_report

    @property
    def last_submitted_weights(self) -> dict[int, float] | None:
        return (dict(self._last_submitted_weights)
                if self._last_submitted_weights is not None else None)

    def record_submitted_weights(self, weights: dict[int, float]) -> None:
        self._last_submitted_weights = dict(weights)

    def complete_pending_weights(self) -> None:
        self._pending_weights = None

    def complete_pending_report(self) -> None:
        self._pending_report = None

    def governance_ready(self) -> tuple[bool, str]:
        """Enforcing-mode preflight the daemon runs before it starts scoring. A validator
        with no owner-approved measured image (MRTD + RTMR1/2/3 + TCB policy) would accept a
        tampered miner runtime, so in enforce mode it must NOT run at all until the owner has
        published the set on-chain. Returns (ok, reason); always ok when not enforcing."""
        if not self.enforce:
            return True, "advisory"
        _approved, mrtd, rtmr, tcb, probe_commit = self._effective_governance()
        if not mrtd:
            return False, "no_approved_mrtd"
        if not rtmr or not {1, 2, 3} <= set(rtmr):
            return False, "no_approved_rtmr"
        if not tcb:
            return False, "no_tcb_policy"
        # Only the re-execution "probe" mode needs a secret bank. If it's in use AND the owner
        # committed one, this validator must hold the matching bank — else its fresh-probe audit is
        # toothless (fail closed). The default "grounding" mode catches memorization by proof
        # inspection, so it needs no bank.
        if self.audit_mode == "probe" and probe_commit:
            if self.probe_bank is None:
                return False, "probe_bank_missing"
            if self.probe_bank.commit() != probe_commit:
                return False, "probe_bank_mismatch"
        return True, "ok"

    def _effective_governance(self):
        """The approved-measurement set actually enforced this epoch: the owner's on-chain
        record if published (rotatable for TCB recovery / image upgrades), else the
        constructor defaults. Returns (runtime_set, mrtd, rtmr, tcb_accept, probe_commit)."""
        approved = set(self.approved)
        approved_mrtd = set(self.approved_mrtd) if self.approved_mrtd else None
        approved_rtmr = dict(self.approved_rtmr) if self.approved_rtmr else None
        tcb_accept = self.tcb_accept
        probe_commit = None
        try:
            gov = self.chain.owner_measurements()
        except Exception:                       # noqa: BLE001 — missing method / transient RPC error
            gov = None
        # F9: a vanished or failed governance read must NOT silently fall back to constructor
        # defaults (→ `mrtd_gate_unset` for the whole subnet under enforce). Cache last-known-good;
        # a genuine owner rotation replaces it, a blip reuses it.
        if gov:
            self._last_gov = gov
        else:
            gov = self._last_gov
        if gov:
            if gov.get("runtime_measurements"):
                approved = set(gov["runtime_measurements"])
            if gov.get("mrtd"):
                approved_mrtd = set(gov["mrtd"])
            if gov.get("rtmr"):
                approved_rtmr = {int(k): v for k, v in gov["rtmr"].items()}
            if gov.get("tcb_accept"):
                tcb_accept = frozenset(gov["tcb_accept"])
            probe_commit = gov.get("probe_commit")
        return approved, approved_mrtd, approved_rtmr, tcb_accept, probe_commit

    def _shared_probe(self, nonce: str, bank=None):
        """One held-out slice, same tasks+order for every miner (feeds memorization
        + dedup). Seeded from the unpredictable epoch nonce. When a verified secret
        `bank` is supplied, the slice is drawn from it (miner-underivable); otherwise
        from the benchmark's public held-out set (`bench.probe`)."""
        out = []
        for bench in self.suite:
            seed = int(signing.sha256_hex(f"probe|{nonce}|{bench.name}")[:8], 16)
            tasks = (bank.draw(bench.name, self.n_per_bench, seed)
                     if bank is not None and bank.has(bench.name)
                     else bench.probe(self.n_per_bench, seed))
            for t in tasks:
                out.append((bench, t))
        return out

    def _audit(self, artifact, probe, verdict: ProofVerdict):
        """Run the DOWNLOADED artifact on the shared probe in a SANDBOXED subprocess
        (never in the validator's process — see koth/sandbox.py). Returns
        (fingerprint, claimed, fresh, n_c, n_f); raises SandboxError on a broken/hostile
        artifact -> caller fails closed. The memorization VERDICT is deferred to a cohort
        pass (`_detect_memorizers`) so probe difficulty can be normalized out."""
        answers = run_agent_probe(artifact.source_text, artifact.weights,
                                  [t.prompt for _, t in probe], pool_spec=self.pool_spec)
        if len(answers) != len(probe):
            raise SandboxError("probe answer count mismatch")
        fresh_correct = sum(bench.grade(a, t.gold) for a, (bench, t) in zip(answers, probe))
        n_f = len(probe)
        n_c = sum(bs.n for bs in verdict.per_bench.values())
        claimed = sum(bs.acc * bs.n for bs in verdict.per_bench.values()) / max(n_c, 1)
        return tuple(answers), claimed, fresh_correct / max(n_f, 1), n_c, n_f

    def _detect_memorizers(self, audits: dict) -> set:
        """Cohort-relative memorization test. The SAME probe runs against every miner, so a
        merely-hard probe drops EVERYONE (median drop) while a memorizer drops much further.
        Subtracting the cohort median removes probe difficulty as a confound — without it, a
        secret bank harder than the public benchmark false-DQs honest miners (at n=64 a
        15-point gap suffices). Capped by `max_probe_drop` so an all-memorizer cohort can't
        inflate the allowance to hide behind."""
        allowance = cohort_probe_allowance(
            [c - f for (c, f, _, _) in audits.values()],
            min_cohort=self.min_cohort, max_drop=self.max_probe_drop)
        return {hk for hk, (claimed, fresh, n_c, n_f) in audits.items()
                if memorization_collapsed_relative(claimed, n_c, fresh, n_f, allowance)}

    def _score_one_miner(self, hk, commit_data, repo, revision, epoch, nonce, get_proof, gov,
                         probe) -> _MinerEval:
        """Verify + gate ONE miner's submission, all bound to the artifact WE downloaded:
        download+recompute hashes → verify_commit → fetch proof → verify_proof → trace binding +
        pool-call gate → public-source scan → memorization backstop (grounding, or the probe) →
        cost-budgeted eligibility. Returns a DQ reason or a scored verdict; the epoch-level cohort /
        dedup / reign passes stay in `run_epoch`."""
        approved, approved_mrtd, approved_rtmr, tcb_accept = gov
        # 1. download the PUBLIC bundle AT THE COMMITTED REVISION and recompute the binding hashes
        #    ourselves. Pinning the revision means every validator grades the same immutable snapshot
        #    (and a later auditor can fetch exactly the bytes that were scored), rather than whatever
        #    the repo's default branch happens to hold when each one looks.
        if not is_pinned_revision(revision):
            # the commit points at a mutable branch (or nothing) -> we cannot know WHICH bytes were
            # scored, so there is nothing to bind. Refuse rather than fetch a moving head.
            return _MinerEval(dq="unpinned_revision")
        try:
            artifact = self.store.download(repo, revision)
        except Exception:  # noqa: BLE001
            return _MinerEval(dq="artifact_unavailable")
        sh, wh = hash_source(artifact.source_text), hash_weights(artifact.weights)
        if not commitmod.verify_commit(commit_data, hk, sh, wh):
            return _MinerEval(dq="bad_commit")
        # from here the artifact is bound to the commit -> carry its hashes on EVERY outcome, so the
        # accumulator can key on (hotkey, source_hash, weights_hash, suite_version).
        def E(**kw):
            return _MinerEval(sh=sh, wh=wh, **kw)
        # 2. fetch the miner's attested proof + trace for this epoch (decoupled)
        try:
            sub = get_proof(hk, epoch, nonce, repo)
        except Exception as e:  # noqa: BLE001
            return E(dq=f"produce_error:{type(e).__name__}")
        if sub is None:
            return E(dq="no_proof")
        proof, trace = sub
        vd = verify_proof(
            proof, approved_measurements=approved, platform_public_hex=self.platform_pub,
            expect_epoch=epoch, expect_nonce=nonce, expect_hotkey=hk,
            expect_source_hash=sh, expect_weights_hash=wh,          # <- recomputed, not miner-supplied
            suite=self.suite, n_per_bench=self.n_per_bench,
            approved_mrtd=approved_mrtd, approved_rtmr=approved_rtmr, tcb_accept=tcb_accept,
            collateral=self.collateral, pccs_url=self.pccs_url, enforce=self.enforce)
        if not vd.valid:
            return E(dq=vd.reason)
        # 2b. F7 anti-grind: the proof must have been COMMITTED on-chain inside the intra-epoch window
        #     and revealed EXACTLY. report_data is quote-bound (verify_proof checked it), so the on-chain
        #     digest ties the attested proof to a timestamp. A revealed proof != the committed one is a
        #     post-commit best-of-N swap; a commit past the window is grinding past the wall-clock bound.
        if self.commit_window is not None:
            pc = self.chain.proof_commit(hk, epoch)
            if pc is None:
                return E(dq="no_proof_commit")
            digest, cblock = pc
            lo = epoch * self.epoch_blocks
            if not (lo <= cblock <= lo + self.commit_window):
                return E(dq="commit_out_of_window")
            if digest != proof.report_data():
                return E(dq="commit_mismatch")
        # 3. the trace must match its binding, then every scored task must have a metered pool call
        #    (you must actually orchestrate the pool, not answer free)
        if signing.sha256_hex(trace) != proof.call_log_hash:
            return E(dq="trace_mismatch")
        calls = Counter(r.get("task_id") for r in trace)
        assigned = [t.task_id for b in self.suite
                    for t in b.sample(self.n_per_bench, bench_seed(nonce, epoch, b.name))]
        if any(calls.get(tid, 0) < 1 for tid in assigned):
            return E(dq="no_pool_call")
        # 4. cheap public-artifact audit (bound: we scanned the artifact we downloaded). Source
        #    catches a literal table; weights catch the same table moved into the opaque blob, which
        #    `scan_source` never looked at.
        hard, reason = scan_source(artifact.source_text)
        if hard:
            return E(dq=reason)
        golds = [t.gold for b in self.suite
                 for t in b.sample(self.n_per_bench, bench_seed(nonce, epoch, b.name))]
        hard, reason = scan_weights(artifact.weights, golds, salt=nonce)
        if hard:
            return E(dq=reason)
        # 5. memorization backstop. DEFAULT "grounding": every scored answer must derive from a
        #    logged pool response (proof-inspection only — the validator runs NO miner code).
        #    "probe": the legacy re-execution fresh-probe, whose verdict defers to the cohort pass.
        audit = None
        if self.audit_mode == "probe":
            try:
                fp, claimed, fresh, n_c, n_f = self._audit(artifact, probe, vd)
            except Exception as e:  # noqa: BLE001
                return E(dq=f"audit_error:{type(e).__name__}")
            audit = (claimed, fresh, n_c, n_f)
        else:
            ok_g, why_g = grounding_check(proof, trace, self.suite)
            if not ok_g:
                return E(dq=why_g)
            # copy-dedup fingerprint = the attested scored-answer vector (no re-execution)
            fp = tuple(r.answer for r in sorted(proof.results, key=lambda r: (r.benchmark, r.task_id)))
        # 6. cost-budgeted eligibility. per_epoch: this slice's cost. accumulate: applied on the
        #    ACCUMULATED cost-per-task in run_epoch, so skip the per-slice check here.
        if self.scoring_mode == "per_epoch":
            ok, why = eligible(vd, budget=self.budget, f_min=self.f_min)
            if not ok:
                return E(dq=why)
        return E(verdict=vd, fingerprint=fp, audit=audit)

    def _router_scalar(self, acc: float, cost: float, epoch: int, nonce: str) -> float | None:
        """Frontier-relative headroom against the owner-published pool reference, or None if no
        reference is available for this epoch (then the caller keeps the legacy scalar)."""
        if self.pool_reference is None:
            return None
        try:
            ref = self.pool_reference(epoch, nonce)
        except Exception:      # noqa: BLE001 — a reference outage must never break scoring
            return None
        if not ref:
            return None
        scores, costs = ref
        # len(), not truthiness: `not scores` raises on a numpy array ("truth value is ambiguous")
        if len(scores) == 0 or len(costs) == 0:
            return None
        return router_headroom(acc, cost, scores, costs)

    def _reign_scalar(self, vd: ProofVerdict, epoch: int | None = None,
                      nonce: str | None = None) -> float:
        """Q_lcb minus a small cost term — the quality-first, cost-as-tiebreak ranking scalar.

        Quality alone saturates: once an agent is at the accuracy ceiling nothing can rank above it,
        every good miner ties, and the reign falls back to commit-block seniority forever. The cost
        term is a gradient that never runs out (you can always be cheaper) and points at the axis
        where routing actually has headroom."""
        if epoch is not None:
            # UNITS: the frontiers are per-ask ($/query), but `total_cost_usd` is the whole slice.
            # Comparing them directly puts the miner far to the right of the frontier and scores it
            # ~0 no matter how well it routed.
            n = sum(bs.n for bs in vd.per_bench.values()) or 1
            h = self._router_scalar(vd.total_score, vd.total_cost_usd / n, epoch, nonce or "")
            if h is not None:
                return h            # frontier-relative headroom: the router scalar
        if self.budget <= 0 or not self.cost_tiebreak:
            return vd.score
        return vd.score - self.cost_tiebreak * min(1.0, vd.total_cost_usd / self.budget)

    def _accumulate(self, evals, dq) -> dict:
        """docs/DESIGN.md §5b: pool each committed artifact's per-epoch evidence into one Wilson-LCB (EWMA
        decay, reset-on-recommit) and return {hotkey: accumulated Q_lcb} for the eligible candidates.
        A no-proof epoch counts as (n_expected, 0) — miss=0, which penalizes withholding; a cheat/copy
        DQ is skipped; eligibility is on the ACCUMULATED cost-per-task + per-benchmark accuracy floor."""
        from .runtime import SUITE_VERSION
        self._evidence.decay_all()
        weights = {b.name: b.weight for b in self.suite}
        scores: dict[str, float] = {}
        for hk, ev in evals.items():
            if ev.sh is None:
                continue                                   # never bound -> stays hard-DQ'd
            reason = dq.get(hk)
            if reason is not None and reason not in _MISS_REASONS:
                continue                                   # cheat / copy / etc. -> excluded
            acc = self._evidence.for_artifact(hk, ev.sh, ev.wh, SUITE_VERSION)
            if reason in _MISS_REASONS:
                for b in self.suite:
                    acc.add(b.name, self.n_expected, 0.0, 0.0)   # miss = 0 correct (anti-withholding)
                dq.pop(hk, None)                           # a miss is a candidate, not a hard DQ
            else:                                          # valid verdict this epoch
                for b, bs in ev.verdict.per_bench.items():
                    acc.add(b, bs.n, bs.acc * bs.n, bs.cost_usd)
            # eligibility on the ACCUMULATED evidence (cost is a per-task ceiling, not a divisor)
            if acc.cost_per_task() > self.budget_per_task:
                dq[hk] = "over_budget"
                continue
            bad = next((b.name for b in self.suite if acc.bench_acc(b.name) < self.f_min), None)
            if bad is not None:
                dq[hk] = f"below_floor:{bad}"
                continue
            # same quality-first / cost-tiebreak gradient as per_epoch (`_reign_scalar`), on the
            # ACCUMULATED cost-per-task — otherwise saturated miners tie and freeze on seniority.
            cost_frac = (min(1.0, acc.cost_per_task() / self.budget_per_task)
                         if self.budget_per_task > 0 else 0.0)
            scores[hk] = acc.q_lcb(weights) - self.cost_tiebreak * cost_frac
        return scores

    def _settle_epoch(self) -> int:
        """F2: the latest epoch whose grace deadline has passed — the one it's safe to score now."""
        return max(0, (self.chain.current_block() - self.grace_blocks) // self.epoch_blocks)

    def run_epoch(self, get_proof, epoch: int | None = None, *, submit_weights: bool = True) -> EpochReport:
        """One scoring epoch. `get_proof(hotkey, epoch, nonce, repo) -> Proof | None`
        supplies the miner's attested proof: in the sim it calls `miner.produce`; in the
        live daemon it downloads the proof the miner uploaded to its repo. `epoch` defaults to the
        settled (past-grace) epoch — never the live one — so submissions have settled first (F2)."""
        if epoch is None:
            epoch = self._settle_epoch()
        nonce = epoch_nonce(epoch, self.chain.beacon(epoch))    # unpredictable in production
        commits = {c.hotkey: c for c in self.chain.revealed_commitments()}
        approved, approved_mrtd, approved_rtmr, tcb_accept, probe_commit = self._effective_governance()
        # The re-execution fresh-probe only runs in "probe" mode. In the default "grounding" mode
        # memorization is caught by proof-inspection (grounding_check) and no probe is built/run.
        probe = None
        if self.audit_mode == "probe":
            # decide which probe to use this epoch: the secret bank iff it verifies against the
            # on-chain commit (rotation-safe); no commit published -> use the bank as-is (dev only).
            bank = None
            if self.probe_bank is not None:
                bank = (self.probe_bank if (not probe_commit or self.probe_bank.commit() == probe_commit)
                        else None)
            if self.enforce and probe_commit and bank is None:
                # a secret probe is REQUIRED but unverifiable here -> the memorization audit can't run
                # honestly, so refuse to score this epoch rather than crown on a toothless audit.
                return EpochReport(epoch, {}, {"*": "probe_bank_unverified"}, [], {},
                                   {"error": "probe_bank_unverified"})
            probe = self._shared_probe(nonce, bank)

        verdicts: dict[str, ProofVerdict] = {}
        dq: dict[str, str] = {}
        commit_block: dict[str, int] = {}
        fingerprints: dict[str, tuple] = {}
        audits: dict[str, tuple] = {}          # hotkey -> (claimed, fresh, n_c, n_f)
        audit_detail: dict[str, dict] = {}
        evals: dict[str, _MinerEval] = {}      # every committed miner's outcome (for accumulation)

        gov = (approved, approved_mrtd, approved_rtmr, tcb_accept)
        for hk, c in commits.items():
            parsed = commitmod.parse_commit(c.data)
            if parsed is None:
                continue                        # not a KOTH commit -> skip (no dq)
            commit_block[hk] = c.block
            ev = self._score_one_miner(hk, c.data, parsed[0], parsed[1], epoch, nonce, get_proof,
                                       gov, probe)
            evals[hk] = ev
            if ev.dq:
                dq[hk] = ev.dq
                continue
            verdicts[hk] = ev.verdict
            fingerprints[hk] = ev.fingerprint
            if ev.audit is not None:
                audits[hk] = ev.audit

        # 5a. cohort-relative memorization (probe mode only): normalize out probe difficulty
        if self.audit_mode == "probe":
            for hk in self._detect_memorizers(audits):
                dq[hk] = "memorization"
                verdicts.pop(hk, None)
                fingerprints.pop(hk, None)      # a DQ'd memorizer must not seed copy-dedup

        # 5. behavioral copy-dedup: near-identical answer vectors -> keep earliest commit
        for loser, winner in behavioral_duplicates(fingerprints, commit_block, agree=self.dedup_agree).items():
            dq[loser] = f"copy_of:{winner[:8]}"
            verdicts.pop(loser, None)

        # 6. reign scores. accumulate: pooled per-artifact Q_lcb (incumbency is endogenous, so the
        #    Pareto guard is not needed). per_epoch: this slice's score + the Pareto dethrone guard.
        if self.scoring_mode == "accumulate":
            scored_out = self._accumulate(evals, dq)
            subs = [Submission(hk, hk, commit_block.get(hk, 0), s) for hk, s in scored_out.items()]
        else:
            king_id = self.reign.members[0].sub.miner_id if self.reign.members else self._king_id
            king_vd = verdicts.get(king_id) or self._king_vd
            subs = []
            for hk, vd in verdicts.items():
                s = self._reign_scalar(vd, epoch, nonce)
                if king_vd is not None and hk != king_id:
                    ok, _why = dethrone_guard(vd, king_vd, **self.guard)
                    if not ok:                                # cannot dethrone by trading a regression
                        s = min(s, self._reign_scalar(king_vd, epoch, nonce)) - 1e-9
                subs.append(Submission(hk, hk, commit_block.get(hk, 0), s))   # commit-block seniority
            scored_out = {hk: self._reign_scalar(vd, epoch, nonce)
                          for hk, vd in verdicts.items()}

        current = set(self.chain.hotkeys().values())
        dereg = {m.sub.hotkey for m in self.reign.members if m.sub.hotkey not in current}
        for hk in dereg:
            self._evidence.drop(hk)                     # deregistered -> forget its accumulated evidence
        # LIVENESS: who actually produced a valid, payable proof THIS epoch. `subs` is not the same
        # set — in accumulate mode an absent miner is still scored (miss=0) and so is still a
        # candidate, and paying on candidacy is what let a miner take the crown once, go dark, and
        # draw its share forever. Only the intersection can be paid or crowned.
        live = set(scored_out) & set(verdicts)
        # ALWAYS settle the epoch, even with nothing to score: skipping `set_weights` leaves the
        # PREVIOUS weights standing on-chain, so a network where everyone stopped submitting kept
        # paying its last slate in full, forever. No live miners -> the reign burns to uid 0.
        res = self.reign.update(subs, deregistered=dereg, live=live)

        weights: dict[int, float] = {}
        hk_to_uid = {hk: uid for uid, hk in self.chain.hotkeys().items()}
        for mid, w in res.weights.items():
            uid = 0 if mid == self.reign.burn_uid else hk_to_uid.get(mid, 0)  # unmapped -> burn
            weights[uid] = weights.get(uid, 0.0) + w
        if self.reign.members:                          # persist the king baseline for next epoch's guard
            self._king_id = self.reign.members[0].sub.miner_id
            self._king_vd = verdicts.get(self._king_id) or self._king_vd

        # rolling history for the public feed — the ONLY record of what happened per epoch, since
        # EpochReport is otherwise discarded by the daemon loop.
        king_hk = self.reign.members[0].sub.hotkey if self.reign.members else None
        self._history.append({
            "epoch": epoch,
            "king": king_hk,
            "king_q_lcb": (round(float(self.reign.members[0].sub.score), 6)
                           if self.reign.members else None),
            "coronation": bool(res.coronation),
            "n_scored": len(scored_out),
            "dq": dict(dq),
        })
        self._history = self._history[-HISTORY_LEN:]

        report = EpochReport(epoch, scored_out, dq, res.slots, weights, audit_detail)
        if submit_weights:
            self.chain.set_weights(weights)
            self.record_submitted_weights(weights)
            self._last_scored_epoch = epoch
        else:
            self._last_scored_epoch = epoch
            self._pending_weights = dict(weights)
            self._pending_report = report
        return report
