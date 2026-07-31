# ThirtySpokes — validating guide

A validator **verifies proofs and scores routing decisions — it runs no miner code and does no
inference.** Miners compete on a **routing model**: a small head that, per task, chooses which pool
model should answer. Each epoch you download each miner's weights, check the attestation and binding,
grade the attested answers, score the *decisions* against the owner's pool reference, and set weights. Everything you check is public
and deterministic, so honest validators converge on the same king. For *how the mechanism works* see
[`DESIGN.md`](DESIGN.md); to stand up a subnet see [`DEPLOYING.md`](DEPLOYING.md).

> **Currently deployed on Bittensor TESTNET — netuid 526** (pass `--network test`; the code
> default is mainnet `finney`, netuid 99, which remains the eventual destination).
>
> The daemon runs **fail-closed** by default: it scores a
> proof only if the hardware quote matches the owner's pinned measured image. The no-gate (`--insecure`)
> mode is **refused on mainnet** — it exists for offline development only.

## What you do each epoch

The nonce is a per-epoch chain beacon (same for all miners, so slices are comparable). For each
committed miner:

1. **Download the public bundle yourself**, recompute `source_hash` / `weights_hash`, and `verify_commit`
   them against the on-chain commit — the binding is checked here, not taken on the miner's word.
2. **`verify_proof`**: hardware quote valid → runtime measurement on the owner-approved set (MRTD +
   RTMR1/2/3) under an in-date TCB → `report_data` bound → issued `(epoch, nonce)` + hotkey match → hashes match.
3. **Re-derive the same nonce-seeded slice and grade** the attested answers against public gold
   (missing/substituted tasks count wrong). No agent re-run.
4. **Score the DECISION, not just the outcome.** On the routing path the miner's whole contribution is
   *which rung it entered*; the answer is the pool's work. `decision_regret` compares each attested
   `chosen_rung` against what every other entry point would have produced, using the owner's pool
   reference for that epoch. A validator runs no inference, so without that reference it cannot know
   what the other models would have said — see the pool-reference note below.
5. **Dedup** — on the routing path this fingerprints the head's **soft distribution**, not the answers:
   answers come back from the pool verbatim, so two honest miners choosing the same rung have
   *identical* answer vectors. It ships **off** (`--dedup-max-l1 0`), because honest routers converge
   as they improve and no threshold separates convergence from copying at scale; the reign's
   earliest-commit tiebreak handles copies instead, which a copy cannot beat.
6. **Score + reign** — the cost-budgeted `Q_lcb`, the Pareto dethrone guard vs the persisted king,
   commit-block seniority → `KingChain` → `set_weights` (equal split across the seats that are
   registered **and submitted this epoch**; a seat absent `absent_grace`=3 consecutive epochs is
   evicted, and an epoch with no live miner burns to uid 0 — see [`DESIGN.md`](DESIGN.md) §5a).

Scoring detail and the evidence-accumulation / anti-grind refinements are in [`DESIGN.md`](DESIGN.md) §5–5b.

### The pool reference, and the one way it fails silently

The decision score needs to know what *every* pool model would have scored on this epoch's asks. You
run no inference, so you cannot compute that — the **owner publishes it** once per epoch
(`orchestra-koth-reference`), signed, and you read it from the bucket.

**Its columns are the action space.** A head emits rung indices into the harness's pinned
`ROUTING_POOL`; `decision_regret` discards any rung past the reference's width. So a reference built
over a *different* pool does not error — it silently deletes most of every miner's contribution and
computes a confident number from what is left. The live cron was in exactly that state (a 2-model
reference against a 7-rung ladder), so the validator now **refuses** to score decisions when the
reference's models disagree with its `routing_pool`, recording `decision_skipped` in the audit
diagnostics. If you see that key, the owner's reference is misconfigured — do not "fix" it by
widening your own pool.

Absent a reference entirely, scoring degrades to the absolute scalar rather than stalling: quieter,
and it is why the owner's cron failing is worth alerting on.

## Run fail-closed (the default)

You run **`enforce=True`** — a proof is scored only if its hardware quote matches the owner-pinned image
under an accepted TCB policy. An **unset gate disqualifies** (`mrtd_gate_unset` / `rtmr_gate_unset` /
`tcb_policy_unset`), not a silent accept; the mock TEE and stock CVMs are rejected. A
`governance_ready()` preflight won't let the daemon start until the owner has published the
approved-measurement set on-chain, which you read **per epoch** (so an owner TCB-recovery or image
rotation takes effect without a restart).

> ⚠️ `--insecure` disables the whole security model (mock TEE accepted, measured-image gate skipped →
> miners could forge every score). The daemon **refuses it on mainnet** (`finney`); it exists only for
> offline/dev bring-up on a non-mainnet network.

## Setup + run

**Run the preflight before the daemon.** Every check in it exists because that exact shape silently
broke a live run, and each cost hours because the symptom always surfaced somewhere else — a grading
failure looked like bad miners, a slice mismatch looked like cheating, an unreadable image looked
like an unapproved runtime. It is read-only, needs no chain writes and takes seconds.

```bash
orchestra-koth-doctor --netuid 526 --network test --wallet validator
```

It refuses to start the daemon on a blocking problem (`--skip-preflight` overrides, and does not make
the problem go away). On a fresh server, three of its checks are the ones that bite:

- **`docker` must be the CLIENT.** `apt install docker.io` ships the *daemon*; the client is
  `docker-cli`, a `Recommends` that `--no-install-recommends` drops — so the package installs
  cleanly and leaves no `/usr/bin/docker`, and the ranked code benchmark cannot be graded at all.
- **The sandbox mount must resolve in the DAEMON's namespace.** The grader bind-mounts a scratch
  directory by path, and the path is resolved by whatever daemon you talk to. A containerised
  validator using the host daemon mounts a path that does not exist there and binds nothing. Set
  `KOTH_GRADE_DIR` to a directory mounted identically on both sides.
- **Mount the checkout at `KOTH_REPO_ROOT`** if you run in a container, or the slice-agreement check
  can only compare the process against itself and will say so.

```bash
uv pip install -e ".[chain,eval,tee]"     # bittensor + huggingface_hub + datasets + dcap-qvl
# NO LLM key needed: the default grounding mode does ZERO inference — it verifies the miner's
# attested proof and runs no miner code. Set OPENROUTER_API_KEY *only* if you opt into
# `--audit-mode probe`, which re-executes every miner's agent and bills YOU for it.
huggingface-cli login                     # read miners' public bundles + proofs
btcli wallet new_coldkey && btcli wallet new_hotkey
btcli subnet register --netuid 526 --wallet.name validator --subtensor.network test
# you need stake + a vpermit to set weights
orchestra-koth-validator --netuid 526 --network test --wallet validator
```

**Do not set `--grace-blocks`.** It defaults to the value the OWNER publishes on chain, and that is
deliberate: miners size their runs against the published number, so a validator holding its own copy
drifts from it. Measured here — a compose file defaulted to 50 while governance published 85, every
`up --force-recreate` silently reverted a hand-set value, and the validator scored seven minutes
early, disqualifying proofs that met the published deadline as `no_proof` while the log blamed the
miners. A validator that scores EARLIER than the published grace is now refused at startup.

**One standings publisher.** `--standings-repo` feeds the public dashboard; if a subnet already has a
validator publishing it, leave the flag off rather than racing it.

**Use a state path of your own** (`--state`). The reign, king baseline and accumulator live there,
and sharing one file between validators makes them one validator with extra steps.
It defaults to mainnet (`--network finney`), starts **secure by default**, and refuses to start until
the owner's measured-image governance is published on-chain.

## Run with Docker

A `Dockerfile` + `docker-compose.yml` ship in the repo root for a containerized validator (verify-only,
so no GPU / no CVM). It mounts your `~/.bittensor` (the hotkey signs `set_weights`), persists reign state
in a named volume across restarts, and injects an explicit credential allow-list — `OWNER_HF_TOKEN`
for the optional public standings feed, `HF_TOKEN` for compatibility/gated datasets, and
`OPENROUTER_API_KEY` only for the opt-in `--audit-mode probe` — rather than exposing the rest of
`.env` to the container.

```bash
cp .env.example .env          # set VALIDATOR_WALLET; add standings credentials if publishing
                              # (NETUID=526 and NETWORK=test for the live testnet deployment)
docker compose up -d --build  # build the image + start the daemon
docker compose logs -f        # watch it verify + score each epoch
```

Config comes from `.env` (`NETUID` = 526 and `NETWORK` = `test` for the live deployment; set
`VALIDATOR_WALLET` and `VALIDATOR_HOTKEY`, plus the optional standings and
`COMMIT_WINDOW` / `GRACE_BLOCKS` settings). Enforce mode is on by default.

Epoch state and weight publication are crash-safe. The daemon atomically persists the completed chain
epoch, its new reign, and pending weights before submitting the extrinsic. It waits for bounded chain
inclusion (not an unbounded finalization subscription), then publishes standings and clears the
pending transaction. After a process or RPC failure it replays the same distribution idempotently;
if that distribution is already on-chain, it does not submit it again or rescore the epoch.
Unchanged per-epoch distributions are also skipped from the persisted last successful submission, so
they do not reset Bittensor's weight-rate-limit clock. Expected rate limits use bounded backoff; other
RPC failures replace the connection and retry from the atomically saved state. Validator membership
is read from the minimal on-chain `SubtensorModule.Keys` map rather than the much larger metagraph RPC.
The container handles `SIGTERM` explicitly: it wakes an idle poll immediately or finishes the active
atomic score/save/weight step, then exits within Compose's 30-second stop grace period.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--insecure` | off | **offline/dev only — REFUSED on mainnet.** Accepts the mock TEE and skips the measured-image gate (no security) |
| `--state` | `koth_validator_state.json` | persist reign standings + king baseline so a restart keeps incumbency / eps history |
| `--standings-repo` | — | publish the real per-epoch `standings.json` feed to a public HF repo; requires `OWNER_HF_TOKEN` (preferred for the official validator) or `HF_TOKEN` with write access |
| `--audit-mode` | `grounding` | memorization backstop. `grounding`: pure proof-inspection, runs **no** miner code, costs $0. `probe`: re-executes each miner's agent on a held-out slice — costs inference **and** runs untrusted code in the sandbox |
| `--probe-bank` | — | the owner's secret held-out bank matching the on-chain `probe_commit`. **Implies `--audit-mode probe`** |
| `--commit-window` | off | **F7 anti-grind:** require each proof committed on-chain within N blocks of the epoch open. Miners must opt in with `--commit-proofs`, so **announce this before enabling it** or you will DQ everyone. Calibrate N to one benchmark run's wall-clock first |
| `--grace-blocks` | `0` | **F2:** score an epoch only N blocks after it opens, so submissions settle and validators agree. Keep N < 100 (epoch length); set it above the store's upload latency |
| `--no-pool-reference` | off | score absolute accuracy instead of frontier-relative routing headroom. On by default because the router scalar is the point of the subnet — use this only to reproduce a pre-reference score |
| `--min-headroom-gap` | `0.05` | refuse to score frontier-relatively when the owner's reference shows less achievable headroom than this. Below it the ratio's denominator is under the sampling noise, so the ranking would be noise ([`DESIGN.md`](DESIGN.md) §5.0) |
| `--dedup-max-l1` | `0` (off) | router copy-dedup on soft distributions. Off deliberately: honest routers converge as they improve, and the measured honest-vs-honest distance falls *below* honest-vs-copier at scale, so any non-zero value eventually disqualifies real miners. Earliest-commit handles copies |
| `--n-per-bench` / `--poll` | `16` / `12` | tasks per benchmark per epoch / seconds between chain polls. **This must match the owner's reference builder**, or the record will not line up with your slice |

The `--commit-window` (F7) and `--grace-blocks` (F2) paths are landed but **off by default**, pending a
calibration of `W` and `G`; enable them together once measured. Mechanism in [`DESIGN.md`](DESIGN.md) §5b.

**You do not fetch the pool reference yourself** — the daemon reads it each epoch from the owner's
bucket, resolving the owner from `SubnetOwnerHotkey` on-chain and rejecting anything that key did not
sign. Every failure path (no reference published, unreachable bucket, bad signature, a record for
another slice) falls back to the absolute scalar rather than stalling: a reference outage is the
owner's problem and must never cost a miner its epoch. Watch `diagnostics.pool_reference` in
`standings.json` — `routable: false` means this epoch's traffic was too saturated to score routing on.

## Hardware

Measured on two live validators (one containerised, one on the host):

| | |
|---|---|
| CPU | ~0.05% steady — grounding mode runs **zero inference**; it verifies quotes and grades |
| RAM | 400–600 MB resident, **plus 1 GB per concurrent code grade** (`docker run --memory 1g`) |
| Disk | ~8 GB — validator image 1.4 GB, HF benchmark cache ~6 GB, grader base 189 MB, state 32 KB |
| Network | proof/trace JSON from HF + chain RPC every 12s; ~176 MB over several hours |
| GPU / CVM | **none** — confidential compute is the MINER's requirement, not yours |

**2 vCPU / 4 GB RAM / 40 GB SSD** is comfortable. The constraint that actually scales is grading
concurrency, and it scales with the number of MINERS, not with your hardware: each code grade is a
1 GB container, and every miner's ranked tasks must be graded inside one grace window. At a handful
of miners this is trivially serial; at fifty it is the first thing that will not fit.

## What you need

A **Bittensor wallet** with stake + a vpermit, registered on the subnet, plus TAO; a **HuggingFace**
account; and **Docker**. **No LLM API key, no GPU, and no confidential VM** — validation is verify-only,
and the default grounding mode does zero inference (a plain CPU VM is enough). An **OpenRouter key is
needed only** if you opt into `--audit-mode probe`, which re-executes miner agents and bills you for
their inference. Install the `tee` extra for full DCAP quote verification; off-TDX boxes verify captured
quotes fine.

**Why Docker.** The ranked benchmark is LiveCodeBench ([`DESIGN.md`](DESIGN.md) §5e), so grading an
answer means *executing* a program, not parsing one. Each submission runs in a throwaway
`python:3.11-slim` container with `--network none` and a 1 GB memory cap — the code is untrusted model
output, so it gets no egress and cannot outlive its container. Budget ~8 short-lived containers per
miner per epoch. If Docker is unavailable the validator reports `grading_unavailable` and **excludes**
those miners from the epoch rather than scoring them 0 — your broken infrastructure must never be
charged to a miner — so a Docker outage costs the subnet an epoch, not a miner its rank.
