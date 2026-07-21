#!/usr/bin/env python
"""Stage-0b live smoke: the decoupled KOTH subnet with REAL components.

Real model backend (OpenRouter), real benchmarks (MMLU + GSM8K with real gold), the
reference single-shot agent. One miner publishes + produces an attested proof by
actually calling a model; the validator independently downloads it, grades the
answers against real gold, runs the fresh-probe audit (real calls), and scores.
Mock TEE + local file-chain still (Stages 1-2 swap those). Needs OPENROUTER_API_KEY.
Tiny by design (n_per_bench=3, 1 miner) to bound cost.
"""
from __future__ import annotations

import tempfile
import warnings

warnings.filterwarnings("ignore")

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.koth.benchmarks import real_suite
from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch, epoch_nonce
from thirtyspokes.koth.miner import KOTHMinerNeuron, reference_artifact
from thirtyspokes.koth.neuron import store_get_proof
from thirtyspokes.koth.proof import Proof
from thirtyspokes.koth.runtime import runtime_measurement
from thirtyspokes.koth.store import LocalBundleStore
from thirtyspokes.koth.validator import KOTHValidator
from thirtyspokes.reign import KingChain
from thirtyspokes.subnet.chain import LocalFileChain
from thirtyspokes.tee.attestation import Platform

MODEL = "openai/gpt-4o-mini"
N_PER_BENCH = 3


def main():
    cfg = config.LiveConfig(); cfg.require_key()
    from thirtyspokes.koth.pool import PinnedBackend
    backend = PinnedBackend(OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=120.0,
                                              max_retries=2, price_fn=config.price_for), {MODEL})
    root = tempfile.mkdtemp(prefix="koth_live_")
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    platform = Platform()
    print(f"loading real MMLU + GSM8K...", flush=True)
    suite = real_suite(n_load=12, seed=1)

    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        suite, store, backend, n_per_bench=N_PER_BENCH, budget=0.5, f_min=0.05,
                        pool_spec={"kind": "openrouter"})
    miner = KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                            reference_artifact(MODEL), "ref/repo", n_per_bench=N_PER_BENCH)

    print(f"miner ({MODEL}) publishing + producing proof (real calls)...", flush=True)
    miner.publish()
    chain.advance(EPOCH_BLOCKS)
    epoch = current_epoch(chain)
    miner.run_once(epoch)                                    # real model calls, uploads proof

    print("validator downloading + grading + auditing...", flush=True)
    rep = val.run_epoch(store_get_proof(store))

    # inspect the graded proof
    proof = Proof.from_json(store.download_proof("ref/repo", epoch))
    from thirtyspokes.koth.verify import verify_proof
    from thirtyspokes.koth.store import hash_source, hash_weights
    a = store.download("ref/repo")
    vd = verify_proof(proof, approved_measurements={runtime_measurement()},
                      platform_public_hex=platform.public_hex, expect_epoch=epoch,
                      expect_nonce=epoch_nonce(epoch, chain.beacon(epoch)), expect_hotkey=miner.hotkey,
                      expect_source_hash=hash_source(a.source_text),
                      expect_weights_hash=hash_weights(a.weights), suite=suite,
                      n_per_bench=N_PER_BENCH)
    print("\n=== LIVE decoupled result ===")
    print(f"  proof valid: {vd.valid} ({vd.reason})")
    for b, bs in vd.per_bench.items():
        print(f"  {b:6s} acc={bs.acc:.2f} (n={bs.n})  cost=${bs.cost_usd:.4f}")
    print(f"  total metered cost: ${vd.total_cost_usd:.4f}   Q_lcb={vd.score:.3f}")
    print(f"  scored (validator): {rep.scored}   dq: {rep.dq}")
    print(f"  weights on chain:   {rep.weights_by_uid}")


if __name__ == "__main__":
    main()
