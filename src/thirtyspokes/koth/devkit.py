"""Local miner dev kit (KOTH v2, WS5) — `orchestra-koth-dev`.

A miner points this at their artifact + a pool (offline MockPool, or the real pinned
OpenRouter pool) and sees the EXACT per-benchmark accuracy/LCB, total cost, Q_lcb, and
eligibility the validator will compute — by reusing `verify.verify_proof`/`eligible`
(not a reimplementation), so "what I see locally" == "what the validator scores". Lets
you iterate on routing/orchestration/inference before ever committing on-chain.
"""

from __future__ import annotations

from ..tee.attestation import Platform
from .runtime import Artifact, KOTHRuntime, runtime_measurement
from .store import hash_source, hash_weights
from .verify import eligible, verify_proof


def evaluate(artifact: Artifact, *, pool_backend, suite, n_per_bench: int = 8,
             budget: float = 0.5, f_min: float = 0.1, nonce: str = "dev-seed") -> dict:
    """Run + score `artifact` exactly as a validator would (verify_proof + eligible)."""
    platform = Platform()
    proof, trace = KOTHRuntime(pool_backend, platform).run(
        artifact, hotkey="dev", epoch=0, nonce=nonce, suite=suite, n_per_bench=n_per_bench)
    vd = verify_proof(
        proof, approved_measurements={runtime_measurement()}, platform_public_hex=platform.public_hex,
        expect_epoch=0, expect_nonce=nonce, expect_hotkey="dev",
        expect_source_hash=hash_source(artifact.source_text),
        expect_weights_hash=hash_weights(artifact.weights), suite=suite, n_per_bench=n_per_bench)
    ok, why = eligible(vd, budget=budget, f_min=f_min) if vd.valid else (False, vd.reason)
    return {
        "valid": vd.valid, "reason": vd.reason,
        "per_bench": {b: {"acc": round(bs.acc, 3), "lcb": round(bs.lcb, 3),
                          "cost": round(bs.cost_usd, 5)} for b, bs in vd.per_bench.items()},
        "total_cost": round(vd.total_cost_usd, 5), "Q_lcb": round(vd.score, 4),
        "total_score": round(vd.total_score, 4), "n_pool_calls": len(trace),
        "eligible": ok, "eligibility": why,
    }


def main() -> None:
    import argparse
    import json

    from .benchmarks import default_suite
    from .miner import reference_artifact
    from .pool import MockPool

    p = argparse.ArgumentParser(description="KOTH miner dev kit — score your artifact locally")
    p.add_argument("--source", help="path to your build_agent(weights) source (default: reference router)")
    p.add_argument("--weights", help="path to your weights.bin")
    p.add_argument("--model", default="strong", help="reference router's pool model")
    p.add_argument("--n-per-bench", type=int, default=8)
    p.add_argument("--budget", type=float, default=0.5)
    args = p.parse_args()

    if args.source:
        weights = open(args.weights, "rb").read() if args.weights else b'{"model": "strong"}'
        art = Artifact(open(args.source).read(), weights, "dev")
    else:
        art = reference_artifact(args.model)
    r = evaluate(art, pool_backend=MockPool(), suite=default_suite(),
                 n_per_bench=args.n_per_bench, budget=args.budget)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
