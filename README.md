# Thirty Spokes — a routing subnet where miners ship the conductor and the validator runs it

[![tests](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/tests.yml/badge.svg)](https://github.com/thirtyspokes99/thirtyspokes/actions/workflows/tests.yml)
[![CodeRabbit](https://img.shields.io/coderabbit/prs/github/thirtyspokes99/thirtyspokes?labelColor=171717&color=FF570A&label=CodeRabbit%20reviews)](https://coderabbit.ai)
[![Bittensor](https://img.shields.io/badge/Bittensor-netuid%2099-6c5ce7)](https://taostats.io/subnets/99)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Every model is a spoke; the hub holds nothing. Miners train a **conductor** — full weights on a
pinned `Qwen3.6-35B-A3B` — that, given an agentic task, delegates it to a real model on the catalog,
watches the outcome, and retries or stops. It never writes an answer and never touches the question.
A single owner-run validator runs each challenger against the sitting king on the same fresh slice
of benchmarks, paid on each miner's own OpenRouter key, and scores **quality minus what that quality
cost** at an exchange rate measured from the pool rather than chosen. The crown moves only to a
challenger that wins broadly; until someone beats the best *fixed* policy, the king's share burns.

**Status.** Registered on Bittensor mainnet as netuid 99. The mechanism (v3, v3)
is complete and tested and is being rehearsed end to end on testnet 526 before it opens on 99. The
first-generation mechanism was retired on 2026-09-07 and removed from this tree.

```
┌── miner ─────────────────────────────┐        ┌── validator (owner-run) ───────────────────────┐
│ train full weights, pinned arch      │        │ commit the schedule root; draw the slice from  │
│ check locally (the validator's gate) │──S3──▶ │   a chain beacon; snapshot the catalog          │
│ upload ~70 GB, commit the digest     │──chain▶│ power gate → king's arm once → each challenger │
│ register your OpenRouter key, sealed │        │ verdict (Δ>ε, lcb>0, LOO>0, median>0)          │
└──────────────────────────────────────┘        │ set weights → publish the signed reveal        │
                                                └────────────────────────────────────────────────┘
```

## Quickstart (offline — no chain, no key, no GPU)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

uv run pytest -q                 # the invariant suite
uv run orchestra-sim         # one window against the archetype cast, nothing mocked in the mechanism
uv run orchestra-dev --help  # the miner's dev kit: the validator's own gate and scaffold, locally
```

The live window, on real benchmarks with an `OPENROUTER_API_KEY`:

```bash
set -a && . ./.env && set +a && uv run python scripts/live_smoke.py --help
```

## Documentation

[`docs/`](docs/) holds three documents and a brief README:

| document | read it for |
|---|---|
| [`WHITEPAPER.md`](docs/WHITEPAPER.md) | **the whitepaper** — the artifact, the action space, the episode, money, scoring and the four-condition verdict, the corpus, the queue and every operational rule, the owner decisions |
| [`MINER.md`](docs/MINER.md) | **mine** — the ed25519 hotkey, `check`, the one-shot `submit`, budget, the queue |
| [`VALIDATOR.md`](docs/VALIDATOR.md) | **validate** (owner) — the grading host, launching, the owner account and the miners' keys, issuing credentials |

The measurements every constant traces to are kept with the owner's operating records, outside this tree.

## Repo layout

```
src/thirtyspokes/
  v3/           the mechanism: action grammar, scaffold, conductor seam, worker gateway + metering,
                admission, store (S3), mailbox credentials, chain seam, window schedule, score,
                duel, emissions, validator daemon, miner / owner / dev-kit / simulator CLIs
  v3/benchmarks/  the corpus adapters (LiveCodeBench, HLE, SWE-bench Pro, R2E-Gym, GDPval) + the Docker sandbox
  serve/        the OpenAI-compatible serving path and the enclave app for confidential serving
  koth/         the library v3 imports: TDX quotes + DCAP, sealed keys, the chained manifest, harness
  gateway/, subnet/chain, tee/attestation   signing, the Bittensor SDK seam, attestation primitives
scripts/        live_smoke (the live window), m3a_spread (the λ measurement), the reference
                pinning + census tools, build_router_image.sh (the measured serving image)
runs/           the pinned λ table, the M3a report, the first live window's signed reveal
```

## Command-line tools

`orchestra-miner` (`hotkey` · `identity` · `check` · `register-key` · `submit`), `orchestra-owner` (`issue` ·
`credit` · `balances` · `status` · `key` · `commit-schedule`), `orchestra-validator`,
`orchestra-dev`, `orchestra-sim`; `orchestra-serve`, `orchestra-enclave`,
`orchestra-serving-governance`. Every seam of the validator is a required argument — there is no
offline default that could set weights on a real subnet from a simulation.
