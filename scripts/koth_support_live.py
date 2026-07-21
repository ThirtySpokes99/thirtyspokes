#!/usr/bin/env python
"""Run the fitted two-model support router through the full local KOTH path.

Uses unseen Bitext rows, real OpenRouter calls, confined child execution, the
local chain/store, mock attestation, independent proof verification, and the
validator's grounding/cost gates.  Mock attestation is a local integration test,
not evidence of Intel TDX hardware security.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import tempfile
import time

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.koth.confine import confinement_available
from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch, epoch_nonce
from thirtyspokes.koth.miner import KOTHMinerNeuron
from thirtyspokes.koth.neuron import store_get_proof
from thirtyspokes.koth.pool import PinnedBackend
from thirtyspokes.koth.proof import Proof
from thirtyspokes.koth.runtime import runtime_measurement
from thirtyspokes.koth.store import LocalBundleStore, hash_source, hash_weights
from thirtyspokes.koth.support import (
    SUPPORT_MODELS,
    SUPPORT_PRICE_PER_1K,
    build_support_artifact,
    load_bitext_support_benchmark,
)
from thirtyspokes.koth.validator import KOTHValidator
from thirtyspokes.koth.verify import verify_proof
from thirtyspokes.reign import KingChain
from thirtyspokes.subnet.chain import LocalFileChain
from thirtyspokes.tee.attestation import Platform


class TimingBackend:
    def __init__(self, inner, total: int, progress_every: int):
        self.inner = inner
        self.total = total
        self.progress_every = progress_every
        self.calls = 0
        self.provider_seconds = 0.0
        self.cost = 0.0
        self.by_model = defaultdict(lambda: {"calls": 0, "cost": 0.0, "seconds": 0.0,
                                              "tokens_in": 0, "tokens_out": 0})

    def complete(self, model, messages, params):
        started = time.monotonic()
        result = self.inner.complete(model, messages, params)
        elapsed = time.monotonic() - started
        _text, tokens_in, tokens_out, cost = result
        self.calls += 1
        self.provider_seconds += elapsed
        self.cost += cost
        row = self.by_model[model]
        row["calls"] += 1
        row["cost"] += cost
        row["seconds"] += elapsed
        row["tokens_in"] += tokens_in
        row["tokens_out"] += tokens_out
        if self.calls % self.progress_every == 0 or self.calls == self.total:
            print(f"calls={self.calls}/{self.total} cost=${self.cost:.6f} "
                  f"provider_seconds={self.provider_seconds:.1f}", flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/bitext-support-scored-v2.jsonl")
    parser.add_argument("--n", type=int, default=2_160)
    parser.add_argument("--pool-size", type=int, default=10_800)
    parser.add_argument("--heldout-size", type=int, default=10_800)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.04)
    parser.add_argument("--train-iters", type=int, default=250)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7_200.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--baseline-cost-per-ticket", type=float, default=0.000240201)
    args = parser.parse_args()
    if args.n < 1 or args.progress_every < 1:
        raise ValueError("n and progress-every must be positive")

    live = config.LiveConfig()
    live.require_key()
    setup_started = time.monotonic()
    build = build_support_artifact(
        args.matrix, lam=args.lam, train_iters=args.train_iters, seed=args.train_seed
    )
    bench = load_bitext_support_benchmark(
        exclude_source_indices=build.source_indices,
        n_pool=args.pool_size,
        n_heldout=args.heldout_size,
        seed=args.dataset_seed,
    )
    suite = [bench]
    setup_seconds = time.monotonic() - setup_started

    openrouter = OpenRouterBackend(
        live.api_key, live.base_url, timeout=60.0, max_retries=2,
        price_fn=lambda model: SUPPORT_PRICE_PER_1K[model],
    )
    timed = TimingBackend(openrouter, args.n, args.progress_every)
    backend = PinnedBackend(timed, set(SUPPORT_MODELS))
    root = tempfile.mkdtemp(prefix="koth_support_")
    chain = LocalFileChain(f"{root}/chain")
    store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    platform = Platform()
    hotkey = Signer().public_hex
    model_budget = args.baseline_cost_per_ticket * args.n / 2.0
    validator = KOTHValidator(
        {runtime_measurement()}, platform.public_hex, chain, KingChain(), suite, store, backend,
        n_per_bench=args.n, budget=model_budget, f_min=0.90,
        pool_spec={"kind": "openrouter"},
    )
    miner = KOTHMinerNeuron(
        hotkey, backend, platform, suite, chain, store, build.artifact, "support/router",
        n_per_bench=args.n, confine=True, confine_timeout=args.timeout,
    )

    print("=== support KOTH live holdout ===", flush=True)
    print(f"training_tickets={build.training_tickets} route_mix={json.dumps(build.route_mix)}",
          flush=True)
    print(f"pool={args.pool_size} heldout={args.heldout_size} scored={args.n} "
          f"hardened_confinement={confinement_available()} setup_seconds={setup_seconds:.3f}",
          flush=True)
    print(f"mock_attestation=true model_budget=${model_budget:.6f}", flush=True)
    print(f"run_root={root}", flush=True)

    miner.publish()
    chain.advance(EPOCH_BLOCKS)
    epoch = current_epoch(chain)
    run_started = time.monotonic()
    miner.run_once(epoch)
    proof_seconds = time.monotonic() - run_started

    validation_started = time.monotonic()
    report = validator.run_epoch(store_get_proof(store), epoch=epoch)
    validation_seconds = time.monotonic() - validation_started
    proof = Proof.from_json(store.download_proof("support/router", epoch))
    # LocalBundleStore models the immutable HF revision as the source hash prefix.
    artifact = store.download("support/router", build.artifact.source_hash[:40])
    nonce = epoch_nonce(epoch, chain.beacon(epoch))
    verdict = verify_proof(
        proof,
        approved_measurements={runtime_measurement()},
        platform_public_hex=platform.public_hex,
        expect_epoch=epoch,
        expect_nonce=nonce,
        expect_hotkey=hotkey,
        expect_source_hash=hash_source(artifact.source_text),
        expect_weights_hash=hash_weights(artifact.weights),
        suite=suite,
        n_per_bench=args.n,
    )
    stat = verdict.per_bench[bench.name]

    model_cost_per_ticket = proof.total_cost_usd / stat.n
    allowed_total_per_ticket = args.baseline_cost_per_ticket / 2.0
    infra_headroom_per_ticket = allowed_total_per_ticket - model_cost_per_ticket
    proof_wall_per_ticket = proof_seconds / stat.n
    local_seconds = max(0.0, proof_seconds - timed.provider_seconds)
    local_seconds_per_ticket = local_seconds / stat.n

    def break_even_hourly(seconds_per_ticket: float) -> float | None:
        if infra_headroom_per_ticket <= 0 or seconds_per_ticket <= 0:
            return None
        return infra_headroom_per_ticket * 3_600.0 / seconds_per_ticket

    full_wall_rate = break_even_hourly(proof_wall_per_ticket)
    local_work_rate = break_even_hourly(local_seconds_per_ticket)
    advantage_before_infra = (args.baseline_cost_per_ticket / model_cost_per_ticket
                              if model_cost_per_ticket > 0 else float("inf"))
    passes = (verdict.valid and stat.acc >= 0.90 and advantage_before_infra >= 2.0 and
              hotkey in report.scored)

    print("\n=== proof and validator ===")
    print(f"proof_valid={verdict.valid} reason={verdict.reason} n={stat.n} "
          f"accuracy={stat.acc:.6f} lcb={stat.lcb:.6f}")
    print(f"validator_scored={hotkey in report.scored} dq={report.dq.get(hotkey)} "
          f"mock_quote=true calls={proof.n_calls}")
    print(f"artifact_source_hash={artifact.source_hash} weights_hash={artifact.weights_hash}")
    print("\n=== metering and timing ===")
    for model in SUPPORT_MODELS:
        row = timed.by_model[model]
        print(f"{model}: calls={row['calls']} cost=${row['cost']:.6f} "
              f"mean_provider_seconds={row['seconds'] / max(row['calls'], 1):.4f} "
              f"tokens_in={row['tokens_in']} tokens_out={row['tokens_out']}")
    print(f"model_cost=${proof.total_cost_usd:.6f} "
          f"model_cost_per_ticket=${model_cost_per_ticket:.9f} "
          f"advantage_before_infra={advantage_before_infra:.3f}x")
    print(f"setup_seconds={setup_seconds:.3f} proof_seconds={proof_seconds:.3f} "
          f"provider_seconds={timed.provider_seconds:.3f} local_seconds={local_seconds:.3f} "
          f"validator_seconds={validation_seconds:.3f}")
    print("\n=== 2x infrastructure gate ===")
    print(f"baseline_cost_per_ticket=${args.baseline_cost_per_ticket:.9f} "
          f"allowed_total_per_ticket=${allowed_total_per_ticket:.9f}")
    print(f"infra_headroom_per_ticket=${infra_headroom_per_ticket:.9f} "
          f"infra_headroom_per_million=${infra_headroom_per_ticket * 1_000_000:.3f}")
    print(f"break_even_vm_hourly_full_wall="
          f"{'none' if full_wall_rate is None else f'${full_wall_rate:.6f}/h'}")
    print(f"break_even_vm_hourly_local_work_only="
          f"{'none' if local_work_rate is None else f'${local_work_rate:.6f}/h'}")
    print(f"gate_before_infra={'PASS' if passes else 'FAIL'}")


if __name__ == "__main__":
    main()
