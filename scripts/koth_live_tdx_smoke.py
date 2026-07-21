#!/usr/bin/env python
"""The whole real path in one run (Stage-2), INSIDE the confidential VM.

Real model backend (OpenRouter, real metered `usage.cost`), real benchmarks (MMLU +
GSM8K with real gold), attested by the REAL Intel-TDX hardware root (not the mock
vendor key), then independently verified with `verify_proof` — DCAP chain to the
Intel SGX Root CA + report_data binding + MRTD gate + answer grading + Q_lcb scoring.

This is the un-forgeable claim end-to-end: a miner cannot forge the cost, the answers,
the artifact binding, or the hardware it ran on. Needs OPENROUTER_API_KEY + a TDX guest.
Tiny by design (n_per_bench=2) to bound spend.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.koth import tdx
from thirtyspokes.koth.benchmarks import real_suite
from thirtyspokes.koth.miner import reference_artifact
from thirtyspokes.koth.pool import PinnedBackend
from thirtyspokes.koth.runtime import KOTHRuntime, runtime_measurement
from thirtyspokes.koth.store import hash_source, hash_weights
from thirtyspokes.koth.verify import eligible, verify_proof

MODEL = "openai/gpt-4o-mini"
N_PER_BENCH = 2
EPOCH, NONCE, HOTKEY = 7, "beacon-live-abc", "miner-tdx"


def main() -> None:
    if not tdx.tdx_available():
        raise SystemExit("not on a TDX guest — run this inside the confidential VM")
    cfg = config.LiveConfig(); cfg.require_key()

    backend = PinnedBackend(
        OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=120.0, max_retries=2,
                          price_fn=config.price_for), {MODEL})
    print("loading real MMLU + GSM8K + producing a hardware-attested proof (real calls)...", flush=True)
    suite = real_suite(n_load=8, seed=1)
    art = reference_artifact(MODEL)

    # miner side: run the suite in the enclave, attest with the REAL TDX root
    runtime = KOTHRuntime(backend, tdx.TDXPlatform())
    proof, trace = runtime.run(art, hotkey=HOTKEY, epoch=EPOCH, nonce=NONCE,
                               suite=suite, n_per_bench=N_PER_BENCH)
    mrtd = tdx.verify_tdx_quote_field(proof.quote).mr_td      # owner would pin this on-chain
    print(f"  real calls: {proof.n_calls}   real metered cost: ${proof.total_cost_usd:.4f}")
    print(f"  hardware MRTD (attested): {mrtd[:24]}...")

    # validator side: independent verify_proof over the REAL quote + approved MRTD
    vd = verify_proof(
        proof, approved_measurements={runtime_measurement()}, platform_public_hex="unused-for-tdx",
        expect_epoch=EPOCH, expect_nonce=NONCE, expect_hotkey=HOTKEY,
        expect_source_hash=hash_source(art.source_text), expect_weights_hash=hash_weights(art.weights),
        suite=suite, n_per_bench=N_PER_BENCH, approved_mrtd={mrtd})
    ok, why = eligible(vd, budget=0.5, f_min=0.0) if vd.valid else (False, vd.reason)

    print("\n=== LIVE + TDX result (real models, real cost, real hardware) ===")
    print(f"  proof valid: {vd.valid} ({vd.reason})")
    for b, bs in vd.per_bench.items():
        print(f"  {b:6s} acc={bs.acc:.2f} (n={bs.n})  cost=${bs.cost_usd:.4f}")
    print(f"  total metered cost: ${vd.total_cost_usd:.4f}   Q_lcb={vd.score:.3f}   eligible={ok}")

    # the MRTD gate is real: an un-approved hardware image is refused
    bad = verify_proof(
        proof, approved_measurements={runtime_measurement()}, platform_public_hex="x",
        expect_epoch=EPOCH, expect_nonce=NONCE, expect_hotkey=HOTKEY,
        expect_source_hash=hash_source(art.source_text), expect_weights_hash=hash_weights(art.weights),
        suite=suite, n_per_bench=N_PER_BENCH, approved_mrtd={"00" * 48})
    print(f"  unapproved-MRTD proof: valid={bad.valid} ({bad.reason})")
    assert vd.valid and not bad.valid and bad.reason == "unapproved_runtime"
    print("\nOK — real hardware-attested, real-cost, independently-verified KOTH proof.")


if __name__ == "__main__":
    main()
