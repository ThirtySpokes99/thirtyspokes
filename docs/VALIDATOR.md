# ThirtySpokes — validating guide

A validator **verifies proofs and grades answers — it runs no miner code and does no inference.** Each
epoch you download each miner's public bundle, check its attestation and binding, grade the attested
answers against public gold, score the eligible miners, and set weights. Everything you check is public
and deterministic, so honest validators converge on the same king. For *how the mechanism works* see
[`DESIGN.md`](DESIGN.md); to stand up a subnet see [`DEPLOYING.md`](DEPLOYING.md).

> **Live on Bittensor mainnet — netuid 99.** The daemon runs **fail-closed** by default: it scores a
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
4. **Public-source scan** + the **grounding check** (every scored answer must derive from a logged pool
   response, else `ungrounded`). Proof-inspection only — no miner code runs.
5. **Behavioral dedup** — near-identical attested-answer vectors keep the earliest commit (`copy_of:…`).
6. **Score + reign** — the cost-budgeted `Q_lcb`, the Pareto dethrone guard vs the persisted king,
   commit-block seniority → `KingChain` → `set_weights` (equal split across the king + registered
   ex-kings; burns to uid 0 only if nobody in the chain is still registered).

Scoring detail and the evidence-accumulation / anti-grind refinements are in [`DESIGN.md`](DESIGN.md) §5–5b.

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

```bash
uv pip install -e ".[chain,eval,tee]"     # bittensor + huggingface_hub + datasets + dcap-qvl
# NO LLM key needed: the default grounding mode does ZERO inference — it verifies the miner's
# attested proof and runs no miner code. Set OPENROUTER_API_KEY *only* if you opt into
# `--audit-mode probe`, which re-executes every miner's agent and bills YOU for it.
huggingface-cli login                     # read miners' public bundles + proofs
btcli wallet new_coldkey && btcli wallet new_hotkey
btcli subnet register --netuid 99 --wallet.name validator --subtensor.network finney
# you need stake + a vpermit to set weights
orchestra-koth-validator --netuid 99 --wallet validator
```
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
                              # (NETUID=99 and NETWORK=finney are pre-filled)
docker compose up -d --build  # build the image + start the daemon
docker compose logs -f        # watch it verify + score each epoch
```

Config comes from `.env` (`NETUID` = 99 and `NETWORK` = `finney` are pre-filled; set
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
| `--n-per-bench` / `--poll` | `8` / `12` | tasks per benchmark per epoch / seconds between chain polls |

The `--commit-window` (F7) and `--grace-blocks` (F2) paths are landed but **off by default**, pending a
calibration of `W` and `G`; enable them together once measured. Mechanism in [`DESIGN.md`](DESIGN.md) §5b.

## What you need

A **Bittensor wallet** with stake + a vpermit, registered on the subnet, plus TAO; and a **HuggingFace**
account. **No LLM API key, no GPU, and no confidential VM** — validation is verify-only, and the default
grounding mode does zero inference (a plain CPU VM is enough). An **OpenRouter key is needed only** if you
opt into `--audit-mode probe`, which re-executes miner agents and bills you for their inference. Install the `tee` extra for full DCAP
quote verification; off-TDX boxes verify captured quotes fine.
