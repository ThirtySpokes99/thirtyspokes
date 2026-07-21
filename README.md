# ThirtySpokes — a TEE-attested King-of-the-Hill LLM-agent subnet

[![tests](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/tests.yml/badge.svg)](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/tests.yml)
[![docker](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/docker.yml/badge.svg)](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/docker.yml)
[![CodeRabbit](https://img.shields.io/coderabbit/prs/github/thirtyspokes99/thirtyspokes?labelColor=171717&color=FF570A&label=CodeRabbit%20reviews)](https://coderabbit.ai)
[![Bittensor](https://img.shields.io/badge/Bittensor-netuid%2099-6c5ce7)](https://taostats.io/subnets/99)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Miners compete to build the **routing / orchestration agent** that gets the highest
benchmark quality *per dollar* over an owner-pinned pool of models. Each miner runs the
owner's benchmark suite **inside its own confidential-VM TEE**, publishes a hardware-attested
proof to its own HuggingFace repo, and commits a salted hash on-chain. Validators do **no
inference** — they download the public bundle, verify the attestation, grade the attested
answers against public gold, and a formula over the verified reports crowns the king.

The load-bearing idea: because a miner's agent (source **and** weights) is public and
**cryptographically bound** to what ran in the enclave, cheating is publicly detectable rather
than something we prevent by hiding data. That is what makes **static public benchmarks**
(math, MMLU, GPQA, SWE-bench Pro) safe to score on.

```
┌── miner (own TEE / confidential VM) ──┐        ┌── validator (verify-only) ──┐
│ run owner suite → attested Proof      │        │ download bundle             │
│ bind source_hash+weights_hash+cost    │──HF──▶ │ verify quote + binding      │
│ upload proof+trace to own HF repo     │        │ grade answers vs public gold│
│ commit salted hash on-chain           │──chain▶│ score → reign → set_weights │
└───────────────────────────────────────┘        └─────────────────────────────┘
```

## Quickstart (offline, no chain / no API key)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

uv run pytest -q                 # the full suite (158 passed, 2 skipped)
uv run orchestra-koth-sim        # full mechanism + every adversary (mock everything)
uv run orchestra-koth-local      # the decoupled 2-neuron demo: miner uploads, validator verifies + scores
```

Add real models + real benchmarks (still local chain + mock TEE) with an `OPENROUTER_API_KEY`:

```bash
set -a && . ./.env && set +a && uv run python scripts/koth_live_smoke.py
```

## Documentation

Everything lives under [`docs/`](docs/):

| Doc | What it covers |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | **how it works** — trust model, data models, the miner/validator flow, scoring (per-epoch quality + evidence accumulation + the anti-grind commit/grace windows), and the anti-cheat backstops (grounding, copy-dedup, hardware binding) |
| [`docs/MINER.md`](docs/MINER.md) | **run a miner** — build an agent, test it locally, publish + run the locked-image `orchestra-koth-gcp-miner`, DQ reasons |
| [`docs/VALIDATOR.md`](docs/VALIDATOR.md) | **run a validator** — the per-epoch verify/grade/score loop, fail-closed enforcement, Docker, flags |
| [`docs/DEPLOYING.md`](docs/DEPLOYING.md) | **stand up the subnet** (owner) — local development, production deploy on mainnet, and building the measured runtime image |

## Repo layout

```
src/thirtyspokes/
  koth/         the subnet: proof, commit, runtime, validator, miner, neuron, reign,
                tdx/rtmr/collateral (real Intel-TDX attestation + DCAP), confine (no-egress),
                evidence (accumulation), owner (on-chain governance)
  tee/          the measured enclave runtime + hardware-root attestation primitives
  subnet/       the chain seam (Mock / LocalFile / Bittensor) + the reign
  eval/         the live-run harness (real GSM8K / MMLU / GPQA + OpenRouter metering)
  gateway/      the earlier metered-gateway experiment (superseded by tee/ for cost trust)
  cache, oracle, router, train, sepcmaes, …   the Phase-0 routing core (see below)
scripts/        smokes + experiments (koth_live_smoke, koth_tdx_smoke, koth_enforce_smoke, …)
```

## Status

**Live on Bittensor mainnet — netuid 99.** The mechanism is complete and hardware-verified: real Intel-TDX
attestation with full DCAP (TCB / CRL / QE-identity), the runtime bound into RTMR3 with the validator
gating MRTD + RTMR1/2/3, no-egress agent confinement, on-chain measurement governance, and the
grounding / accumulation scoring — all landed and exercised on a confidential-TDX VM (see
[`docs/DEPLOYING.md`](docs/DEPLOYING.md)).

Validators run **fail-closed**: a proof is scored only if its hardware quote matches the owner's pinned
measured image, and the no-gate `--insecure` mode is **refused on mainnet**. The mock TEE exists only
for offline development.

The anti-grind commit window (F7) and grace window (F2) are implemented but **opt-in**, pending a
calibration of their two constants against the live suite — see
[`docs/VALIDATOR.md`](docs/VALIDATOR.md).

## Phase-0 routing origin

This repo began as an offline prototype of a *pick-one-model router* subnet. That core still ships —
`uv run thirtyspokes demo` runs it — and is where the "routing has no accuracy moat on broad traffic"
finding came from, which drove the pivot to the orchestration / KOTH-TEE design above. The routing
modules (`cache`, `oracle`, `router`, `train`, `sepcmaes`) are that layer; the subnet is `koth/`.

## License

[MIT](LICENSE) — you may use, modify, and redistribute this freely, including for your own subnet or
miner. Miner agents you build on top of it are yours.
