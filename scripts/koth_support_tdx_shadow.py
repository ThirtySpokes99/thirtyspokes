#!/usr/bin/env python
"""Run support-router proofs on fresh TDX VMs across consecutive testnet epochs.

This is deliberately shadow-only: testnet supplies each epoch's block-hash beacon and nonce, but
the script performs no chain writes, proof uploads, governance changes, or weight publication.
Each proof is verified locally with production ``enforce=True`` DCAP/TCB and image gates.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time

import koth_support_tdx_canary as canary

from thirtyspokes.gateway import signing
from thirtyspokes.koth import rtmr, tdx
from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch, epoch_nonce
from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement
from thirtyspokes.koth.support import (
    SUPPORT_MODELS,
    build_support_artifact,
    load_bitext_support_benchmark,
)
from thirtyspokes.koth.verify import _bootstrap_lcb, grounding_check, verify_proof
from thirtyspokes.subnet.chain import BittensorChain


def _select_epoch(block: int, last_epoch: int | None, min_blocks: int) -> int | None:
    epoch = block // EPOCH_BLOCKS
    remaining = (epoch + 1) * EPOCH_BLOCKS - block
    if last_epoch is not None and epoch <= last_epoch:
        return None
    if last_epoch is not None and epoch != last_epoch + 1:
        raise RuntimeError(f"testnet skipped from epoch {last_epoch} to {epoch}")
    return epoch if remaining >= min_blocks else None


def _wait_for_epoch(chain, last_epoch: int | None, min_blocks: int, poll_seconds: float) -> dict:
    announced = None
    while True:
        block = chain.current_block()
        epoch = _select_epoch(block, last_epoch, min_blocks)
        if epoch is not None:
            beacon = chain.beacon(epoch)
            return {
                "block": block,
                "epoch": epoch,
                "blocks_remaining": (epoch + 1) * EPOCH_BLOCKS - block,
                "beacon": beacon,
                "nonce": epoch_nonce(epoch, beacon),
            }
        state = (block // EPOCH_BLOCKS, (block + 1) % EPOCH_BLOCKS)
        if announced != state[0]:
            target = ((last_epoch + 1) if last_epoch is not None
                      else block // EPOCH_BLOCKS + 1)
            print(f"[shadow] waiting for usable epoch {target}; current block={block}", flush=True)
            announced = state[0]
        time.sleep(poll_seconds)


def _shadow_instances(zone: str) -> list[str]:
    found = canary._run([
        "gcloud", "compute", "instances", "list",
        f"--filter=name~'^koth-support-shadow-' AND zone:({zone})",
        "--format=value(name)",
    ], check=False)
    if found.returncode != 0:
        raise RuntimeError(f"could not audit shadow instances: {found.stderr[-1000:]}")
    return [line.strip() for line in found.stdout.splitlines() if line.strip()]


def _cleanup_shadow_instances(zone: str) -> list[str]:
    leaked = _shadow_instances(zone)
    for name in leaked:
        canary._gcloud(zone, "compute", "instances", "delete", name, "--quiet", check=False)
    return leaked


def _verify_run(
    run: dict,
    *,
    expected_measurements: dict[str, str],
    epoch: int,
    nonce: str,
    hotkey: str,
    artifact,
    suite,
    n: int,
    min_accuracy: float,
    min_epoch_lcb: float,
) -> dict:
    for key in ("mrtd", "rtmr1", "rtmr2", "rtmr3"):
        if run["measurements"].get(key) != expected_measurements.get(key):
            raise RuntimeError(f"epoch {epoch} disagrees with approved {key}")

    proof, trace = run["proof"], run["trace"]
    if signing.sha256_hex(trace) != proof.call_log_hash:
        raise RuntimeError(f"epoch {epoch} trace hash differs from proof")
    if proof.n_calls != n or len(trace) != n:
        raise RuntimeError(f"epoch {epoch} expected {n} calls and trace rows")
    if set(entry["model"] for entry in trace) - set(SUPPORT_MODELS):
        raise RuntimeError(f"epoch {epoch} used a model outside the approved support pool")

    raw_quote = tdx.parse_quote(base64.b64decode(proof.quote.platform_sig[4:]))
    quoted = {
        "mrtd": raw_quote.mr_td.hex(),
        **{f"rtmr{i}": value.hex() for i, value in enumerate(raw_quote.rtmrs)},
    }
    if any(quoted[key] != run["measurements"][key]
           for key in ("mrtd", "rtmr1", "rtmr2", "rtmr3")):
        raise RuntimeError(f"epoch {epoch} quote differs from serial measurements")

    approved_rtmr = {i: expected_measurements[f"rtmr{i}"] for i in (1, 2, 3)}
    gate = dict(
        approved_measurements={runtime_measurement()},
        platform_public_hex="unused-for-tdx",
        expect_epoch=epoch,
        expect_nonce=nonce,
        expect_hotkey=hotkey,
        expect_source_hash=artifact.source_hash,
        expect_weights_hash=artifact.weights_hash,
        suite=suite,
        n_per_bench=n,
        enforce=True,
        approved_mrtd={expected_measurements["mrtd"]},
        approved_rtmr=approved_rtmr,
        tcb_accept=tdx.DEFAULT_TCB_ACCEPT,
    )
    verdict = verify_proof(proof, **gate)
    if not verdict.valid:
        raise RuntimeError(f"epoch {epoch} enforcing verification failed: {verdict.reason}")
    grounded = grounding_check(proof, trace, suite)
    if grounded != (True, "ok"):
        raise RuntimeError(f"epoch {epoch} grounding failed: {grounded[1]}")

    wrong_gate = dict(gate)
    wrong_gate["approved_rtmr"] = {**approved_rtmr, 1: "00" * 48}
    rejected = verify_proof(proof, **wrong_gate)
    if rejected.valid or rejected.reason != "unapproved_runtime":
        raise RuntimeError(f"epoch {epoch} wrong RTMR1 was not rejected: {rejected.reason}")

    stat = verdict.per_bench[suite[0].name]
    if stat.acc < min_accuracy or stat.lcb < min_epoch_lcb:
        raise RuntimeError(
            f"epoch {epoch} quality gate failed: accuracy={stat.acc:.4%}, lcb={stat.lcb:.4%}"
        )
    return {
        "accuracy": stat.acc,
        "accuracy_lcb": stat.lcb,
        "correct": round(stat.acc * stat.n),
        "calls": proof.n_calls,
        "model_calls": dict(Counter(entry["model"] for entry in trace)),
        "model_cost_usd": proof.total_cost_usd,
        "call_log_hash": proof.call_log_hash,
        "source_hash": proof.source_hash,
        "weights_hash": proof.weights_hash,
        "wrong_rtmr1_rejection": rejected.reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="koth-support-shadow-v16")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--network", default="test")
    parser.add_argument("--netuid", type=int, default=526)
    parser.add_argument("--wallet", default="mini-val1")
    parser.add_argument("--hotkey-name", default="hotkey1")
    parser.add_argument("--proof-hotkey", default="support-tdx-shadow")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n", type=int, default=720)
    parser.add_argument("--min-blocks", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=12.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--matrix", default="data/bitext-support-scored-v2.jsonl")
    parser.add_argument("--measurement-summary",
                        default="data/koth-support-tdx-v16-rerun/summary.json")
    parser.add_argument("--output", default="data/koth-support-tdx-shadow")
    parser.add_argument("--vm-hourly", type=float, default=0.2224888)
    parser.add_argument("--baseline-cost-per-ticket", type=float, default=0.000240201)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-epoch-lcb", type=float, default=0.0,
                        help="optional small-slice LCB floor; aggregate LCB is the decisive gate")
    parser.add_argument("--min-aggregate-lcb", type=float, default=0.90)
    parser.add_argument("--resume-serial",
                        help="recover one completed first epoch from a prior interrupted gate")
    parser.add_argument("--resume-created-unix", type=float,
                        help="GCE instance-name timestamp for --resume-serial economics")
    parser.add_argument("--resume-start-block", type=int,
                        help="observed start block for --resume-serial")
    args = parser.parse_args()
    if min(args.epochs, args.n, args.min_blocks, args.max_attempts) < 1:
        raise ValueError("epochs, n, min-blocks, and max-attempts must be positive")
    if args.min_blocks >= EPOCH_BLOCKS:
        raise ValueError("min-blocks must be below the 100-block epoch length")
    if args.vm_hourly <= 0 or args.baseline_cost_per_ticket <= 0:
        raise ValueError("prices must be positive")
    if args.resume_serial and args.resume_created_unix is None:
        raise ValueError("--resume-serial requires --resume-created-unix")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    approved_summary = json.loads(Path(args.measurement_summary).read_text(encoding="utf-8"))
    approved = approved_summary["measurement_boot"]["measurements"]
    expected_rtmr3 = rtmr.owner_expected_rtmr3(
        runtime_measurement=runtime_measurement(), suite_version=SUITE_VERSION,
        pool_allow_list=SUPPORT_MODELS,
    )
    if approved["rtmr3"] != expected_rtmr3:
        raise RuntimeError("approved summary RTMR3 differs from the owner-derived runtime value")

    build = build_support_artifact(args.matrix)
    bench = load_bitext_support_benchmark(exclude_source_indices=build.source_indices)
    suite = [bench]
    if (build.artifact.source_hash != approved_summary["proof"]["source_hash"]
            or build.artifact.weights_hash != approved_summary["proof"]["weights_hash"]):
        raise RuntimeError("current support artifact differs from the independently measured artifact")

    chain = BittensorChain(args.netuid, args.wallet, args.network, hotkey=args.hotkey_name)
    all_results = []
    retries = 0
    last_epoch = None
    run_started = time.time()

    def finish_run(run: dict, slot: dict, index: int, errors: list[str],
                   end_block: int | None) -> dict:
        checked = _verify_run(
            run, expected_measurements=approved, epoch=slot["epoch"], nonce=slot["nonce"],
            hotkey=args.proof_hotkey, artifact=build.artifact, suite=suite, n=args.n,
            min_accuracy=args.min_accuracy, min_epoch_lcb=args.min_epoch_lcb,
        )
        vm_cost = run["vm_lifetime_seconds"] / 3_600.0 * args.vm_hourly
        per_million = (checked["model_cost_usd"] + vm_cost) / args.n * 1_000_000
        result = {
            "sequence": index + 1,
            "epoch": slot["epoch"],
            "beacon": slot["beacon"],
            "nonce": slot["nonce"],
            "start_block": slot["block"],
            "end_block": end_block,
            "attempts": len(errors) + 1,
            "attempt_errors": errors,
            "instance": run["name"],
            "measurements": run["measurements"],
            "run_seconds": run["run_seconds"],
            "vm_lifetime_seconds": run["vm_lifetime_seconds"],
            "vm_cost_usd": vm_cost,
            "total_cost_per_million_usd": per_million,
            "passes_2x": per_million <= args.baseline_cost_per_ticket * 500_000,
            **checked,
        }
        epoch_dir = output / f"epoch-{slot['epoch']}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "proof.json").write_text(run["proof"].to_json(), encoding="utf-8")
        (epoch_dir / "trace.json").write_text(json.dumps(run["trace"]), encoding="utf-8")
        (epoch_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
        )
        print(
            f"[shadow] epoch {slot['epoch']} PASS acc={checked['accuracy']:.4%} "
            f"lcb={checked['accuracy_lcb']:.4%} all_in=${per_million:.2f}/M", flush=True,
        )
        return result

    if args.resume_serial:
        serial_path = Path(args.resume_serial)
        serial = serial_path.read_text(encoding="utf-8")
        parsed = canary._parse_serial(serial)
        proof = canary.Proof.from_json(parsed["proof_json"])
        trace = json.loads(parsed["trace_json"])
        epoch = proof.epoch
        beacon = chain.beacon(epoch)
        slot = {
            "block": args.resume_start_block or epoch * EPOCH_BLOCKS,
            "epoch": epoch,
            "blocks_remaining": None,
            "beacon": beacon,
            "nonce": epoch_nonce(epoch, beacon),
        }
        if proof.nonce != slot["nonce"]:
            raise RuntimeError("resumed proof nonce differs from the testnet epoch beacon")
        resumed = {
            **parsed,
            "proof": proof,
            "trace": trace,
            "name": f"recovered-{serial_path.stem}",
            "vm_lifetime_seconds": canary._serial_compute_window(
                serial, args.resume_created_unix,
            ),
        }
        all_results.append(finish_run(resumed, slot, 0, [], None))
        last_epoch = epoch
    with tempfile.TemporaryDirectory(prefix="support_tdx_shadow_metadata_") as td:
        temp = Path(td)
        source_b64 = temp / "source.b64"
        weights_b64 = temp / "weights.b64"
        secrets = temp / "secrets.env"
        source_b64.write_text(base64.b64encode(build.artifact.source_text.encode()).decode())
        weights_b64.write_text(base64.b64encode(build.artifact.weights).decode())
        secrets.write_text(f"OPENROUTER_API_KEY={api_key}\n")
        secrets.chmod(0o600)

        try:
            for index in range(len(all_results), args.epochs):
                slot = _wait_for_epoch(chain, last_epoch, args.min_blocks, args.poll_seconds)
                epoch = slot["epoch"]
                print(
                    f"[shadow] epoch {epoch} start block={slot['block']} "
                    f"remaining={slot['blocks_remaining']} nonce={slot['nonce']}", flush=True,
                )
                errors = []
                run = None
                for attempt in range(1, args.max_attempts + 1):
                    try:
                        run = canary._boot(
                            image=args.image, zone=args.zone,
                            label=f"shadow-e{epoch}-a{attempt}", epoch=epoch,
                            nonce=slot["nonce"], hotkey=args.proof_hotkey, n=args.n,
                            source_b64=source_b64, weights_b64=weights_b64, secrets=secrets,
                            output=output / f"epoch-{epoch}", deadline_seconds=1_800,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 — retry one fresh VM on any boot failure
                        errors.append(f"{type(exc).__name__}: {exc}")
                        if attempt == args.max_attempts:
                            raise
                        retries += 1
                        print(f"[shadow] epoch {epoch} attempt {attempt} failed; retrying", flush=True)
                        time.sleep(15)
                assert run is not None
                if _shadow_instances(args.zone):
                    raise RuntimeError(f"epoch {epoch} VM remained after _boot cleanup")

                end_block = chain.current_block()
                if current_epoch(chain) != epoch:
                    raise RuntimeError(f"epoch {epoch} proof crossed into block {end_block}")
                result = finish_run(run, slot, index, errors, end_block)
                all_results.append(result)
                last_epoch = epoch
        finally:
            leaked = _cleanup_shadow_instances(args.zone)
            if leaked:
                print(f"[shadow] final cleanup removed: {', '.join(leaked)}", flush=True)

    epochs = [result["epoch"] for result in all_results]
    total_calls = sum(result["calls"] for result in all_results)
    total_correct = sum(result["correct"] for result in all_results)
    combined_correct = ([1.0] * total_correct
                        + [0.0] * (total_calls - total_correct))
    aggregate_lcb = _bootstrap_lcb(combined_correct, 0.05, 1_000, seed=0x5A17)
    total_model_cost = sum(result["model_cost_usd"] for result in all_results)
    total_vm_cost = sum(result["vm_cost_usd"] for result in all_results)
    all_in_per_million = (total_model_cost + total_vm_cost) / total_calls * 1_000_000
    summary = {
        "verdict": "PASS",
        "mode": "testnet-read-only shadow; no chain/store writes",
        "network": args.network,
        "netuid": args.netuid,
        "image": args.image,
        "zone": args.zone,
        "hardware": "GCP c3-standard-4 Intel TDX",
        "epochs": epochs,
        "consecutive_epochs": all(b == a + 1 for a, b in zip(epochs, epochs[1:])),
        "runs": all_results,
        "aggregate": {
            "calls": total_calls,
            "correct": total_correct,
            "accuracy": total_correct / total_calls,
            "accuracy_lcb": aggregate_lcb,
            "model_cost_usd": total_model_cost,
            "vm_cost_usd": total_vm_cost,
            "total_cost_usd": total_model_cost + total_vm_cost,
            "total_cost_per_million_usd": all_in_per_million,
            "baseline_per_million_usd": args.baseline_cost_per_ticket * 1_000_000,
            "advantage": args.baseline_cost_per_ticket * 1_000_000 / all_in_per_million,
            "passes_2x": all_in_per_million <= args.baseline_cost_per_ticket * 500_000,
            "retries": retries,
            "wall_seconds_including_epoch_waits": time.time() - run_started,
        },
        "approved_measurements": approved,
        "cloud_cleanup": {"remaining_shadow_instances": _shadow_instances(args.zone)},
    }
    if (len(all_results) != args.epochs or not summary["consecutive_epochs"]
            or aggregate_lcb < args.min_aggregate_lcb
            or not all(result["passes_2x"] for result in all_results)
            or not summary["aggregate"]["passes_2x"]
            or summary["cloud_cleanup"]["remaining_shadow_instances"]):
        summary["verdict"] = "FAIL"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                          encoding="utf-8")
    print("\n=== MULTI-EPOCH TESTNET TDX SHADOW ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["verdict"] != "PASS":
        raise SystemExit("multi-epoch shadow gate failed")


if __name__ == "__main__":
    main()
