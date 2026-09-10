# Thirty Spokes — Bittensor subnet 99

Every model is a spoke; the hub holds nothing. Miners train a **conductor** — full weights on a
pinned `Qwen3.6-35B-A3B` — that delegates each agentic task to a real model on the catalog, watches
the outcome, and retries or stops. It never writes an answer and never touches the question. One
owner-run validator runs every challenger against the sitting king on the same fresh slice of
benchmarks, paid on each miner's own OpenRouter key, and scores **quality minus what that quality
cost** at an exchange rate measured from the pool rather than chosen. The crown moves only to a
challenger that wins broadly — Δ > ε, bootstrap lower bound > 0, leave-one-out > 0, median > 0 —
and until someone beats the best *fixed* policy, the king's 0.85 burns. The five most recent
ex-kings share 0.15.

**Status.** Registered on Bittensor mainnet as netuid 99. The mechanism is complete and tested and
is being rehearsed end to end on testnet 526 before it opens on 99.

| document | for whom |
|---|---|
| [`WHITEPAPER.md`](WHITEPAPER.md) | **the whitepaper** — the artifact, the action space, the episode, money, scoring and the verdict, the corpus, the queue and every operational rule, the owner decisions |
| [`MINER.md`](MINER.md) | **miners** — the ed25519 hotkey, registering, the local admission gate, the sealed OpenRouter key, the one-shot `submit`, budget, the queue |
| [`VALIDATOR.md`](VALIDATOR.md) | **the validator (owner)** — the grading host, launching, the owner account and the miners' keys, issuing credentials, what is published |

Tools: `orchestra-miner` (`hotkey` · `identity` · `check` · `register-key` · `submit`), `orchestra-owner`
(`issue` · `credit` · `balances` · `status` · `key` · `commit-schedule`), `orchestra-validator`,
`orchestra-dev` (the miner's local dev kit), `orchestra-sim` (one offline window).

The measurements every constant traces to (the λ fit, the corpus census, the first live window)
are kept with the owner's operating records, outside this tree.
