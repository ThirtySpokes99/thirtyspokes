# ThirtySpokes — design & mechanism

*The full design of the KOTH-TEE subnet, consolidated: trust model, data models, the
miner/validator flow, the scoring (per-epoch quality, evidence accumulation, and the
anti-grind windows), and the anti-cheat backstops (grounding, copy-dedup, hardware
binding). For how to **run** one, see [`MINER.md`](MINER.md), [`VALIDATOR.md`](VALIDATOR.md), and
[`DEPLOYING.md`](DEPLOYING.md).*

**King-of-the-Hill over miner-run, TEE-attested benchmarks.** Miners run the owner-given
benchmark suite inside their own TEE and publish an attested proof to their own HuggingFace
repo; validators only verify the proof and grade the reports — no validator-side inference,
no centralized API. A math formula over the verified reports crowns the king. The load-bearing
idea: because the artifact (source **and** weights) is public and **cryptographically bound**
to what ran, cheating is publicly detectable rather than something we must prevent by hiding
data — which is what makes **static public benchmarks** (math, MMLU, GPQA, SWE-bench Pro) safe
here. Offline skeleton: `src/thirtyspokes/koth/`, run `orchestra-koth-sim`.

This composes the existing primitives — hardware-root attestation (`tee.attestation`),
in-enclave metering (`tee.runtime.MeteringProxy`), the king + equal-share chain (`reign.KingChain`),
and the chain seam (`subnet.chain`). It adds the artifact **binding**, a
**multi-benchmark** proof, a **Pareto dethrone guard**, and the optimistic anti-cheat
backstops. It supersedes the validator-runs-eval model of ARCHITECTURE.md §3 for the
KOTH design: benchmark execution moves *into the miner's TEE*.

> This spec reflects the fixes from the adversarial review: the
> binding is now enforced end-to-end (the runtime hashes the bytes it loads+runs; the
> validator downloads the bundle and binds it to the on-chain commit), copy-mining is
> stopped by **behavioral dedup** (a copier earned ~86% pre-fix), the Pareto guard
> clamps on *any* failure, and the memorization test is statistical + fail-closed.

## 1. Trust model — the miner-owns / protocol-owns inversion

The reviewed KOTH subnets (Teutonic SN3, Affine SN120, Albedo SN97) all run eval on
trusted infrastructure. KOTH moves it to the miner and keeps it trustworthy with
hardware attestation + public auditability. Who is trusted to do what:

| Layer | Trusted? | Who writes it |
|---|---|---|
| TEE hardware root (Intel TDX / AMD SEV-SNP / NVIDIA CC) | yes (vendor) | hardware vendor |
| KOTH enclave runtime (measured; on the approved list) | yes (audited, public) | subnet owner |
| Benchmark suite + public gold | yes (public) | subnet owner |
| Miner agent: routing model + inference/orchestration source | **no** (public, bound, run under the enclave) | **miner** |
| Validator | verify-only (holds no secret; every check is public) | subnet owner |

The enclave runs the miner's arbitrary agent but hands it *only* a metered
`call_model`, and it stamps the agent's `source_hash`/`weights_hash` into the
attestation. So a validator — and any third party — can check that a given public
artifact produced a given score, without re-running it.

## 2. Data models (canonical, hash-stable) — `koth/proof.py`, `koth/commit.py`

```
BenchmarkResult:                      # one graded item
  benchmark   str                     # "math" | "mmlu" | "gpqa" | "swe"
  task_id     str                     # validator re-derives the gold from the nonce
  answer      str                     # graded validator-side against the PUBLIC gold
  cost_usd    float                   # metered by the runtime — un-forgeable

Proof:                                # the unit validators verify (one per miner-epoch)
  epoch, nonce, hotkey                # validator-issued nonce; hotkey bound (anti-copy-resubmit)
  source_hash, weights_hash, model_id # BINDING: ties the score to the public artifact
  results: [BenchmarkResult]          # the whole suite in one attested run
  total_cost_usd, n_calls, call_log_hash
  measurement                         # enclave image hash; must be on the approved list
  quote  = Platform.quote(measurement, report_data)   # hardware signature
  report_data() = sha256(payload-minus-quote)         # binds EVERY field above

Commit (on-chain, salted commit-reveal):
  koth1|<hf_repo>|<revision>|<sha256(source_hash ‖ weights_hash ‖ hotkey)>
```

One hardware signature over `{measurement, report_data}` asserts both *"this approved
enclave ran"* and *"it produced this exact payload"*. Mutating any field recomputes
`report_data()` and breaks the quote.

## 3. Miner flow

1. Build the agent: routing model + inference/orchestration source (any framework).
2. Run the owner-given suite **inside the TEE** (`KOTHRuntime.run`) → an attested `Proof`
   over the nonce-seeded slice; the runtime meters real cost and binds `source_hash`
   /`weights_hash`.
3. Publish the **bundle** to your OWN HF repo (`HFBundleStore`): weights + source +
   `proof.json`. All public and auditable.
4. Commit `koth1|repo|revision|salted-hash` on-chain via `set_reveal_commitment`
   (`subnet.chain`). The reveal delay defeats front-running; the hotkey salt defeats
   commit-copying.

## 4. Validator flow (cheap; one bounded spot-check) — `koth/validator.py`

Per epoch (nonce = a per-epoch chain beacon, shared by all miners so slices are
comparable), for each committed miner:
1. **download the PUBLIC bundle yourself**, recompute `source_hash`/`weights_hash`, and
   `verify_commit` them against the on-chain commit — the binding is checked here, not
   taken on the miner's word.
2. `verify_proof` the attested proof against those *recomputed* hashes: quote valid →
   measurement approved → `report_data` bound → issued `(epoch, nonce)` → bound hotkey →
   `source_hash`/`weights_hash` == recomputed.
3. re-derive the SAME nonce-seeded slice and **grade the attested answers** against the
   public gold (missing/substituted tasks = wrong). No agent re-run.
4. **public-source scan** on the downloaded source (catches literal hardcoding).
5. **grounding check** (`grounding_check`, default) — every scored answer must derive from a
   pool response logged in the (hash-attested) trace; an artifact that ignores the pool and
   returns a memorized answer matches no response → DQ `ungrounded`. **Proof-inspection only:
   the validator runs NO miner code** (deterministic, $0). *(Opt-in `audit_mode="probe"` swaps
   this for the re-execution held-out probe — §6.)*
6. **behavioral dedup**: near-identical **attested-answer** vectors → keep the earliest-committed,
   DQ the rest.
7. apply the **Pareto dethrone guard** (clamp on *any* failure, strictly below the king)
   against the persisted king baseline, feed the cost-aware scalar + commit-block
   seniority + the epoch's **live** set to `KingChain` → `set_weights`. Only seats whose holder
   submitted a valid proof this epoch are paid, split equally; if no seat is live the epoch burns
   to uid 0 (§5a).

## 5. The scoring formula + parameters

The reign scalar is **quality first, cost as the tiebreak**:

```
S = Q_lcb − λ · min(1, cost / B)        Q_lcb = Σ_b w_b · lcb_b,   λ = cost_tiebreak (0.02)
```

`Q_lcb` is the weighted sum of per-benchmark **lower confidence bounds** (bounded to [0,1]), so a
lucky slice can't dethrone. Cost enters only through a **small** term: λ is far below the value of a
single question, so it can **never override a genuine accuracy difference** — it orders miners who
are otherwise equal. Cost is *also* still a hard **budget ceiling** in `eligible`: a miner earns
nothing that epoch unless `total_cost ≤ B`, accuracy ≥ `f_min` on *every* benchmark, and every scored
task made ≥1 pool call.

> **Why cost is in the score at all.** `Q_lcb` **saturates** — once an agent is at the accuracy
> ceiling nothing can rank above it, every good miner ties, and the reign falls back to commit-block
> seniority, freezing emissions on the earliest committer forever. Cost is the one axis that never
> runs out (you can always be cheaper), and it is where routing/orchestration actually has headroom.

Per benchmark `b` the validator computes point accuracy `acc_b` and a paired-bootstrap lower
confidence bound `lcb_b`. To take **slot 1**, a challenger must additionally pass `dethrone_guard`
vs the current king — not-worse everywhere, then winning on **one of two axes**:

- **not-worse on every benchmark:** `acc_c(b) ≥ acc_king(b) − tol`
- **sample gate:** `n_b ≥ min_tasks` (else that benchmark counts as worse — anti thin/cherry-picked eval)
- **then EITHER —**
  - **(a) quality:** confidently dominant on ≥1 benchmark, `lcb_c(b) > acc_king(b) + margin`, **or**
  - **(b) cost:** at quality parity, materially cheaper — `cost_c ≤ cost_king · (1 − cost_margin)`
- **cost not-worse:** `cost_c ≤ cost_king · (1 + cost_tol)`

A challenger that regresses is clamped to `min(S, S_king)` so it cannot dethrone by
trading a regression — `reign.py` is untouched; the guard lives in the validator.

| Param | Default | Source |
|---|---|---|
| benchmark weights `w_b` (math/mmlu/gpqa/swe) | 0.30 / 0.22 / 0.23 / 0.25 | owner |
| `margin` (confident dominance) | 0.03 | Affine 0.03 |
| `tol` (not-worse band) | 0.02 | Affine not-worse 0.02 |
| `min_tasks` (per-benchmark sample gate) | 5 | Affine thin-eval guard |
| `cost_tol` | 0.10 | this design |
| `cost_margin` (cheaper-at-parity dethrone) | 0.10 | this design — the anti-ossification axis |
| `cost_tiebreak` λ (cost term in the reign scalar) | 0.02 | small enough to never outrank a real accuracy gain |
| `lcb` α (one-sided **Wilson**, not a resampling bootstrap) | 0.05 | `evidence.wilson_lcb`; see the small-sample note below |
| `scoring_mode` | **`accumulate`** (default) · `per_epoch` (sim only) | §5b |
| king-chain size / eps schedule | 5 slots (king + 4 ex-kings), equal share 20% each when all five are paid · eps0 0.02 → floor 0.002, τ 8 | `reign.py` `KingChain` (SN9→IOTA anti-hoarding pension tail) |
| `absent_grace` (consecutive missed epochs before a seat is evicted) | 3 | tolerates the ~30% flaky-CVM-boot rate measured on testnet 526 |
| memorization test `z_crit` | 2.33 (one-sided ~99%, two-proportion) | `memorization_collapsed_relative` |
| `min_cohort` (miners needed to calibrate probe difficulty) | 3 | this design |
| `max_probe_drop` (owner: "no honest probe is harder than this") | 0.25 | this design |
| copy-dedup agreement | 0.95 answer-vector agreement | `behavioral_duplicates` |

### 5a. Emissions pay for work, not for a seat (liveness)

**A seat pays only while its holder is still submitting valid proofs.** `KingChain` receives the set
of miners that produced a valid proof this epoch (`live`) and pays only those; a seat missing
`absent_grace` consecutive epochs is evicted; an absent king keeps its title (**unpaid**) and goes on
setting the eps bar until its grace runs out; a king that loses the crown by going dark gets **no**
pension seat. An epoch with no live miner **burns to uid 0** rather than leaving the previous weights
standing on-chain.

This closes an emission-capture class found by auditing the mining side, where payout consulted only
seat *membership* + registration and never *work*. Measured against the pre-fix mechanism:

- a miner that took the vacant crown with the **cheapest** pool model, then went permanently dark,
  captured **54% of emissions over 12 epochs** — out-earning the honest miner that worked every one
  of them (it now earns 8% for 1 of 12 epochs, i.e. exactly its share of the work);
- because the king earned exactly what an idle ex-king earned, a cartel could rotate the crown
  through 5 self-owned hotkeys and then stop entirely, holding **100%** of emissions and capping an
  arbitrarily better honest miner at 1/5 of the pot;
- withholding a **single** epoch handed the crown to any submitter, unguarded (`incumbent is None`
  ⇒ `ranked[0]` wins): a 21-epoch champion lost to a 0.01-scoring agent by missing one upload;
- a network where every miner stopped submitting kept paying its last slate in full, forever,
  because `set_weights` was skipped when there was nothing to score.

Note the interaction with §5b: under `accumulate` a miner that missed the epoch is still *scored*
(miss=0) and therefore still a *candidate*, so liveness is passed explicitly rather than inferred
from the candidate list — inferring it there would re-open the pension.

> **Small-sample honesty.** The reign scalar's LCB is a one-sided **Wilson** bound, not a resampling
> bootstrap. The bootstrap is degenerate on the all-same slices that matter most: resampling 8
> identical values yields 8 identical means, so a perfect 8/8 slice reported `lcb = 1.000` — zero
> claimed uncertainty from eight questions. That cleared `dethrone_guard` against any king below
> 0.97, making "resubmit every epoch until a slice comes up perfect" a free lottery ticket for the
> crown. Wilson prices the sample correctly (8/8 → 0.747, 32/32 → 0.922), is deterministic (two
> validators no longer depend on drawing the same resamples), and is already what §5b pools on — so
> the two modes now agree.

## 5b. Scoring over time — evidence accumulation + anti-grind

A single epoch samples only ~32 tasks (8 per benchmark), which is noisy: per-epoch scoring lets a
lucky slice dethrone and makes two decoupled validators — drawing independent slices — disagree.
**Evidence accumulation is therefore the default**; the two anti-grind windows below remain opt-in
per validator. The simulation that motivated them is `scripts/scoring_v2_sim.py`.

- **Evidence accumulation** (`scoring_mode="accumulate"`, **the default**, `koth/evidence.py`). Each epoch's *verified*
  per-benchmark result for a **fixed artifact** is pooled into one decayed binomial, keyed by
  `(hotkey, source_hash, weights_hash, suite_version)`; the reign scalar is the **Wilson lower bound**
  on the pooled counts. An EWMA (`half_life_epochs`, default 200) gives liveness — stop submitting and
  your evidence bleeds away — and **re-committing a new artifact resets the accumulator** (its key
  changes), so dethroning costs real attested epochs. A **missing epoch counts as `n_expected` tasks,
  0 correct** (miss=0), which makes selective publication (hide your bad epochs) strictly dominated.
  Simulation vs per-epoch scoring: validator divergence ↓5.5×, crown churn ↓20×, mis-crown 71%→6%,
  time-to-crown monotone in the true edge.

- **F7 — intra-epoch commit window** (`commit_window=W`). Accumulation banks a *grinder's* inflated
  draws rather than diluting them, so it needs a wall-clock bound. Each epoch the miner commits its
  proof's `report_data()` on-chain (keyed by `(hotkey, epoch)`) within `[epoch·epoch_blocks, +W]`, then
  reveals the proof; the validator scores it only if the revealed `report_data()` **matches** the
  on-chain commit and the commit landed **in-window**. That binds one attested run (no post-commit
  best-of-N swap → `commit_mismatch`) and caps pre-commit grinding to `W`'s wall-clock (a late commit →
  `commit_out_of_window`; none → `no_proof_commit`). Residual: only as tight as `W` — a short window is
  the actual lever, calibrated to one suite run.

  **Hard constraint — one commitment slot.** The chain gives each hotkey a single
  `CommitmentOf[(netuid, hotkey)]`, and *both* the proof commit (`set_commitment`) and the artifact
  commit (`set_reveal_commitment`) write it. A proof commit issued while the artifact commit is still
  timelocked **overwrites it**, so the artifact never reveals, validators bind no artifact, and nobody
  is scored — verified on testnet 526, where the `TimelockEncrypted` field was replaced by the plain
  one 4 blocks later. So the miner holds proof commits back until its artifact commit has **revealed**
  (`KOTHMinerNeuron._artifact_revealed`); once revealed, the record lives in the separate append-only
  `RevealedCommitments` map, which a later proof commit cannot touch. Re-publishing an artifact
  re-enters that window and pauses proof commits again, automatically. `--commit-proofs` is therefore
  opt-in and only meaningful when validators actually run `--commit-window`.

- **F2 — grace window** (`grace_blocks=G`). Scoring the *live* epoch sees miners still running →
  false-misses everyone, non-deterministically. Instead the validator scores a **settled** epoch — the
  latest whose grace deadline has passed, `(current_block − G) // epoch_blocks` — so every miner had the
  full commit+upload window and two validators polling at different blocks compute the same result. The
  miss=0 presence decision is then a settled read. Keep `G < epoch_blocks` (a real single-slot chain
  holds only the scored epoch's commit) and `G` above the store's upload latency.

Enable `commit_window` + `grace_blocks` together once `W` and `G` have been measured for the live suite.

## 6. Static public benchmarks are safe — binding + optimistic audit

A TEE proves *execution*, not *task honesty*. Binding + publicness turns cheating into
something detectable and punishable, in three layers:

1. **Binding** (`source_hash`/`weights_hash` in `report_data`, checked in step 4.2):
   the score is provably from *this exact public artifact* — no secret code, no swapped
   weights, no off-enclave computation.
2. **Literal hardcoding** (answer tables in code) → visible in the PUBLIC source → any
   validator/rival files a fraud proof (`adjudicate_challenge`) → dethrone + slash.
3. **Weight-memorization** of a static public test set (invisible to source inspection — the
   table hides in the *opaque weights*) → the **grounding check** (default): every scored answer
   must derive from a pool response logged in the trace, so an artifact that ignores the pool and
   returns a memorized answer is DQ'd `ungrounded`. It is pure proof-inspection, so **the validator
   runs no miner code** (deterministic, $0, no secret data), and it closes the *total* memorizer
   (the memorize-everything race). Residual (prompt-injection laundering — inject the answer so a pool
   model echoes it — and partial semantic memorization) is the documented public-data limit. The **opt-in
   re-execution fresh-probe** (`audit_mode="probe"`, plus its recommended null-pool variant) is the
   upgrade for when real emissions attract a competent adversary: the validator runs the *downloaded
   bound* artifact on a held-out slice and applies a **one-sided two-proportion test**
   (`memorization_collapsed_relative`), **fail-closed**. Two properties make that opt-in path sound:
   - **The probe must be miner-underivable**, else a miner memorizes it too. In production the
     slice is drawn from a **secret, owner-held, rotated probe bank** (`koth/heldout.py`) committed
     on-chain (`probe_commit`) and distributed only to validators; the validator verifies its bank
     against the commit and **fails closed** without it (`governance_ready`,
     `probe_bank_unverified`). Off that, it falls back to the public held-out slice (offline/dev only —
     derivable, so weaker).
   - **The test must be difficulty-normalized.** A secret bank need not match the public
     benchmark's difficulty, and an *absolute* claimed-vs-fresh test then flags honest routers
     purely for probe hardness (measured: at n=64 a **15-point** gap false-DQs an honest miner, and
     the power *grows* with more probe data). Since the SAME probe runs against every miner, the
     validator subtracts the **cohort median drop** (`cohort_probe_allowance`) — a hard probe drops
     everyone; a memorizer drops much further. The allowance is capped by `max_probe_drop` so a
     colluding all-memorizer cohort can't inflate it, and with `< min_cohort` miners it falls back
     to that owner-declared cap (not 0, which would false-DQ a bootstrapping subnet).

**The honest caveat.** Layer 2 is fully general; layer 3 needs a *small* fresh held-out
probe — the one place freshness is still required — because weights can hide what source
cannot. This is far lighter than a hidden main benchmark: the leaderboard runs on the
standard public benchmark (comparable to published SOTA), and a small probe backstops
memorization. Re-run audits require reproducibility (temp=0 greedy, pinned pool-model
snapshots, seeded RNG); remote-LLM non-determinism makes them **statistical** (agreement
within a tolerance), not bit-exact.

The sim demonstrates all of this: `hardcoder` → `hardcoded_answers`, `memorizer` →
`memorization`, while the honest miners pass.

## 7. Anti-copy (the load-bearing defense, since every artifact is public)

The review showed copy-mining is the attack the whole design must survive: pre-fix, 4
clones of the best public artifact captured ~86% of emissions. Defense, in order of weight:

- **Behavioral copy-dedup** (`behavioral_duplicates`, the primary): cluster miners by their
  answer vector on the SHARED probe; near-identical vectors are copies (robust to
  weight-space perturbation, unlike a weight-geometry hash). Keep the **earliest-committed**
  representative, DQ the rest. Caveat: two *independently excellent* models agree on correct
  answers, so the agreement threshold is high and the probe should be discriminative
  (error-revealing); pure argmax-agreement is weakest when everyone is near-perfect.
- **Commit-block seniority**: the reign tiebreak + eps age now key on the on-chain **commit
  block** (not the registration block), so the original author wins over a later copier.
- **Salted commit-reveal** `koth1|repo|revision|sha256(source_hash‖weights_hash‖hotkey)`
  + reveal delay: stops commit-string replay and front-running (the copier must run the
  artifact in its own TEE, at which point dedup catches it).

Marginal-contribution (Numerai-MMC) scoring is the deeper moat if behavioral dedup proves
insufficient at scale — a copy earns ~0 orthogonal contribution even if it evades clustering.

## 8. Residual trust assumptions (stated honestly)

- **Hardware vendor root.** Trust reduces to the TEE vendor's attestation service and
  the absence of a full hardware break (side-channels, physical extraction).
- **Measurement governance.** The owner curates + rotates the approved-runtime list (TCB
  recovery). A careless approval admits a lying runtime; this is an owner responsibility.
  *Landed:* published + rotated **on-chain** (`orchestra-koth-owner` → `chain.owner_measurements()`,
  read per-epoch); the validator enforces MRTD + RTMR1/2/3 + a TCB-status policy (`dcap-qvl`).
- **Fail-closed enforcement (the load-bearing invariant).** Because the miner's own CVM samples
  *and* runs the benchmark, the only thing standing between a trustless miner and a fully-forged
  score is the hardware measured-image gate. The production validator therefore runs `enforce=True`
  (`verify_proof`/`KOTHValidator`): it **rejects the mock vendor key** (`mock_quote_rejected`) and
  **refuses to score at all** unless a genuine hardware quote is gated on an owner-pinned MRTD +
  RTMR1/2/3 under a TCB policy — a missing/empty gate DQs (`mrtd_gate_unset`/`rtmr_gate_unset`/
  `tcb_policy_unset`) instead of silently accepting. The deployable daemon `governance_ready()`
  preflight won't even start an enforcing validator until the owner has published the set on-chain.
  `--insecure` (mock TEE, gate off) exists for offline/dev bring-up only, is explicitly no-security, and
  the daemon **refuses it on mainnet** — so production has no no-security path.
  **Precondition:** this gate is only *discriminating* once the reproducible measured image
  (`DEPLOYING.md` §Measured image) is built and its MRTD/RTMR1/2 pinned — a stock-CVM MRTD is shared by all code,
  so the measured-image build is the mainnet prerequisite, not an optional nicety.
- **Provider TLS + no-egress.** The agent runs with **zero network egress** (a netns; `koth/confine.py`)
  and reaches the pool only via the metered `call_model`; the provider is trusted for content, not
  cost (metered in-enclave). *Residual:* **prompt-based exfiltration** — the agent can smuggle the
  nonce/task into a *legitimate* prompt to the pinned model, reaching a confederate through the
  sanctioned channel. netns can't close this; partial mitigation is parent-side prompt inspection.
- **Audit reproducibility.** Fresh-probe/re-run audits are statistical, not bit-exact,
  because remote LLMs aren't deterministic; the collapse threshold absorbs that.
- **TEE barrier to entry.** Miners need a confidential VM (cheap CPU CVM; no GPU if the
  pool is remote). This is a real but modest cost of admission.

## 9. What the offline skeleton shows vs. what a live subnet enforces

**Build now (offline-runnable, `orchestra-koth-sim`, `tests/test_koth.py`):** the whole
protocol shape — binding, multi-benchmark verify + grading, Pareto guard, fresh-probe +
fraud-proof audits, reign + chain + burn. The four benchmarks are computable synthetic
pools so it runs with no network.

**Seams — Intel-TDX path LANDED + verified on a GCP C3 (91 tests); others need external access:**
- **Real TEE** — ✅ `koth/tdx.py` generates a genuine TDX quote (configfs TSM) + `verify_quote_full`
  does full DCAP (chain→pinned Intel root **+ TCB/CRL/QE-identity via `dcap-qvl`**); the runtime is
  bound into **RTMR3** (`koth/rtmr.py`) and gated. SEV-SNP = same shape, different parser (deferred).
- **Approved-measurement list on-chain** — ✅ `chain.owner_measurements()` + `orchestra-koth-owner`.
- **No-egress enclave** — ✅ `koth/confine.py` (netns, verified BLOCKED on the C3); prompt-exfil residual.
- **Reproducible measured image** — recipe in `DEPLOYING.md` §Measured image; first build is the remaining ops step.
- **Real benchmarks** — `koth.mmlu.load_mmlu` (`cais/mmlu`), `koth.gpqa.load_gpqa`
  (`Idavidrein/gpqa`, gated → `HF_TOKEN`), math via `eval.math_tasks`/`eval.hard_tasks`,
  SWE via `eval.swe_tasks.SWEGrader` (Docker + hidden tests).
- **Live pool backend** — `gateway.gateway.OpenRouterBackend` (needs `OPENROUTER_API_KEY`)
  in place of `MockBackend`.
- **HF bundle** — `koth.store.HFBundleStore`; chain — `subnet.chain.BittensorChain`.
- **Hardening seams:**
  - ✅ **Secret, owner-held, rotated held-out probe** — the *mechanism* is landed
    (`koth/heldout.py`: on-chain `probe_commit`, validator-verified bank, fail-closed, rotation).
    *Residual:* sourcing probe **data** that is genuinely outside pool models' training sets (a
    data-curation task, not code).
  - ✅ **Unpredictable beacon** for the epoch nonce — `BittensorChain.beacon` seeds from the
    epoch-start **block hash** (unpredictable before the epoch); mock chains stay deterministic.
  - ✅ **No-egress sandbox** — `koth/confine.py` (netns; verified BLOCKED on the C3). The
    prompt-exfil residual is bounded further by running the validator's probe audit against a
    **trusted** pool (no confederate on the far end of `call_model`).
  - ⏳ **Challenger stake + slash** on the fraud-proof path — the adjudication + binding are in
    place (`adjudicate_challenge`); economic stake/slash is a **subnet-token seam** (not native to
    Bittensor weight-setting) and is left un-faked pending the chain design.
