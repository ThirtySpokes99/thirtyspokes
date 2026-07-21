#!/usr/bin/env python
"""Run and enforce-verify the support router on a real GCP C3 Intel TDX image.

The image must have been built with ``KOTH_SUITE=support`` using
``build_koth_image_prod.sh``. Two fresh VMs are used: a one-task measurement boot,
then the full holdout boot. Measurements approved for the second proof therefore
come from an independent boot of the same dm-verity image.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

from thirtyspokes.gateway import signing
from thirtyspokes.koth import rtmr, tdx
from thirtyspokes.koth.commit import commit_string
from thirtyspokes.koth.epoch import EPOCH_BLOCKS, epoch_nonce
from thirtyspokes.koth.neuron import NoInferenceBackend, store_get_proof
from thirtyspokes.koth.proof import Proof
from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement
from thirtyspokes.koth.store import LocalBundleStore
from thirtyspokes.koth.support import (
    SUPPORT_MODELS,
    build_support_artifact,
    load_bitext_support_benchmark,
)
from thirtyspokes.koth.validator import KOTHValidator
from thirtyspokes.koth.verify import grounding_check, verify_proof
from thirtyspokes.reign import KingChain
from thirtyspokes.subnet.chain import LocalFileChain


_NEXT_START = re.compile(r"--start=(\d+)")
# The kernel audit console can interleave bytes into journald's ``KOTH-`` prefix.
# Match from the semantic marker onward; chunk framing and strict Base64/JSON
# validation below still prevent partial or unrelated console text from passing.
_CHUNK = re.compile(
    r"(?P<kind>PROOF|TRACE)(?P<gzip>-GZIP)?-CHUNK\s+"
    r"(?P<index>\d+)/(?P<total>\d+)\s+"
    r"(?:sha256=(?P<digest>[0-9a-f]{64})\s+)?(?P<data>[A-Za-z0-9+/=]+)"
)
_MEASURE = re.compile(r"MEASURE\s+(mrtd|rtmr[0-3])=([0-9a-f]+)")
_TIMING = re.compile(r"TIMING\s+run_seconds=([0-9.]+)")
_AUDIT_CLOCK = re.compile(
    r"\[\s*([0-9.]+)\][^\n]*?audit:.*?audit\(([0-9]+\.[0-9]+):"
)
_DONE_CLOCK = re.compile(r"\[\s*([0-9.]+)\][^\n]*KOTH-DONE")


def _run(cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def _gcloud(zone: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["gcloud", *args, f"--zone={zone}"], check=check)


def _decode_chunks(serial: str, kind: str) -> str:
    candidates: dict[int, list[tuple[str, str | None]]] = {}
    total = None
    compressed = None
    for match in _CHUNK.finditer(serial):
        if match.group("kind") != kind:
            continue
        index, seen_total = int(match.group("index")), int(match.group("total"))
        if total is not None and total != seen_total:
            raise RuntimeError(f"inconsistent {kind} chunk totals")
        total = seen_total
        seen_compressed = bool(match.group("gzip"))
        if compressed is not None and compressed != seen_compressed:
            raise RuntimeError(f"inconsistent {kind} chunk compression")
        compressed = seen_compressed
        candidates.setdefault(index, []).append((match.group("data"), match.group("digest")))
    if total is None or set(candidates) != set(range(1, total + 1)):
        raise RuntimeError(f"incomplete {kind} chunks: got {len(candidates)}, expected {total}")

    chunks = {}
    for index, copies in candidates.items():
        valid = [
            (data, digest) for data, digest in copies
            if digest is None or hashlib.sha256(data.encode()).hexdigest() == digest
        ]
        if not valid:
            raise RuntimeError(f"{kind} chunk {index} has no digest-valid copy")
        # Prefer a digest-authenticated copy. For legacy chunks, a console interruption only
        # shortens the Base64 token, so retaining the longest copy is the safest reconstruction.
        chunks[index] = max(valid, key=lambda item: (item[1] is not None, len(item[0])))[0]
    encoded = "".join(chunks[i] for i in range(1, total + 1))
    raw = base64.b64decode(encoded, validate=True)
    if compressed:
        raw = gzip.decompress(raw)
    return raw.decode()


def _payloads_complete(serial: str) -> bool:
    """Return true only after both console payloads are complete, valid JSON."""
    try:
        json.loads(_decode_chunks(serial, "PROOF"))
        json.loads(_decode_chunks(serial, "TRACE"))
    except (RuntimeError, binascii.Error, UnicodeError, json.JSONDecodeError):
        return False
    return True


def _chunks_emitted(serial: str) -> bool:
    """Return true once both declared chunk sequences have emitted every index."""
    for kind in ("PROOF", "TRACE"):
        matches = [match for match in _CHUNK.finditer(serial) if match.group("kind") == kind]
        totals = {int(match.group("total")) for match in matches}
        if len(totals) != 1:
            return False
        total = totals.pop()
        if {int(match.group("index")) for match in matches} != set(range(1, total + 1)):
            return False
    return True


def _parse_serial(serial: str) -> dict:
    measures = {match.group(1): match.group(2) for match in _MEASURE.finditer(serial)}
    if not {"mrtd", "rtmr1", "rtmr2", "rtmr3"} <= set(measures):
        raise RuntimeError(f"serial output is missing measurements: {sorted(measures)}")
    timing = _TIMING.search(serial)
    return {
        "measurements": measures,
        "run_seconds": float(timing.group(1)) if timing else None,
        "proof_json": _decode_chunks(serial, "PROOF"),
        "trace_json": _decode_chunks(serial, "TRACE"),
    }


def _serial_compute_window(serial: str, created_unix: float) -> float:
    """Derive GCE create-to-proof-complete seconds from attested-run console clocks."""
    offsets = [float(wall) - float(monotonic) for monotonic, wall
               in _AUDIT_CLOCK.findall(serial)]
    done = _DONE_CLOCK.search(serial)
    if not offsets or done is None:
        raise RuntimeError("serial output lacks clocks needed to derive the compute window")
    boot_unix = statistics.median(offsets)
    window = boot_unix + float(done.group(1)) - created_unix
    if window <= 0:
        raise RuntimeError(f"invalid serial-derived compute window: {window}")
    return window


def _load_replay(path: Path, created_unix: float, label: str) -> dict:
    serial = path.read_text(encoding="utf-8")
    parsed = _parse_serial(serial)
    parsed["proof"] = Proof.from_json(parsed["proof_json"])
    parsed["trace"] = json.loads(parsed["trace_json"])
    parsed["name"] = f"replay-{label}"
    window = _serial_compute_window(serial, created_unix)
    parsed["create_to_done_seconds"] = window
    parsed["vm_lifetime_seconds"] = window
    parsed["lifetime_basis"] = "serial-derived GCE create to KOTH-DONE"
    return parsed


def _boot(
    *,
    image: str,
    zone: str,
    label: str,
    epoch: int,
    nonce: str,
    hotkey: str,
    n: int,
    source_b64: Path,
    weights_b64: Path,
    secrets: Path,
    output: Path,
    deadline_seconds: float,
) -> dict:
    name = f"koth-support-{label}-{int(time.time())}"
    pool_csv = ",".join(SUPPORT_MODELS)
    metadata = (f"^@^koth-epoch={epoch}@koth-nonce={nonce}@koth-hotkey={hotkey}"
                f"@koth-pool={pool_csv}@koth-n-per-bench={n}@koth-confine-timeout=7200")
    files = (f"koth-secrets={secrets},koth-agent-source={source_b64},"
             f"koth-agent-weights={weights_b64}")
    create_started = time.monotonic()
    print(f"[{label}] creating {name} ({n} task{'s' if n != 1 else ''})", flush=True)
    created = _gcloud(
        zone, "compute", "instances", "create", name,
        "--machine-type=c3-standard-4", "--confidential-compute-type=TDX",
        "--maintenance-policy=TERMINATE", f"--image={image}",
        "--network-interface=nic-type=GVNIC", "--boot-disk-size=50GB",
        f"--metadata={metadata}", f"--metadata-from-file={files}",
        check=False,
    )
    if created.returncode != 0:
        raise RuntimeError(f"VM creation failed: {created.stderr[-2000:]}")

    serial_parts: list[str] = []
    start = 0
    done_at = None
    announced_run = False
    try:
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            time.sleep(15)
            got = _gcloud(
                zone, "compute", "instances", "get-serial-port-output", name,
                f"--start={start}", check=False,
            )
            if got.returncode != 0:
                continue
            if got.stdout:
                serial_parts.append(got.stdout)
            match = _NEXT_START.search(got.stderr)
            start = int(match.group(1)) if match else start + len(got.stdout.encode())
            serial = "".join(serial_parts)
            run = re.search(r"KOTH-RUN tasks=(\d+) calls=(\d+) cost=([0-9.]+)", serial)
            if run and not announced_run:
                print(f"[{label}] runtime finished: tasks={run.group(1)} calls={run.group(2)} "
                      f"cost=${run.group(3)}; collecting attested payload", flush=True)
                announced_run = True
            if "KOTH-ERROR" in serial:
                raise RuntimeError(serial[serial.rfind("KOTH-ERROR"):][-2000:])
            # GCE's incremental serial offsets can occasionally omit bytes even though
            # a complete start=0 snapshot is available. Once the run is over, prefer
            # that validated snapshot instead of waiting on a poisoned incremental view.
            if run and not _payloads_complete(serial):
                snapshot = _gcloud(
                    zone, "compute", "instances", "get-serial-port-output", name,
                    "--start=0", check=False,
                )
                if snapshot.returncode == 0 and _payloads_complete(snapshot.stdout):
                    serial_parts = [snapshot.stdout]
                    serial = snapshot.stdout
                elif snapshot.returncode == 0 and _chunks_emitted(snapshot.stdout):
                    output.mkdir(parents=True, exist_ok=True)
                    (output / f"{label}-invalid-serial.log").write_text(
                        snapshot.stdout, encoding="utf-8",
                    )
                    raise RuntimeError(
                        f"{name} emitted all proof/trace chunks, but strict payload validation failed"
                    )
            if _payloads_complete(serial):
                done_at = time.monotonic()
                break
        else:
            raise TimeoutError(f"{name} did not emit complete proof and trace payloads "
                               "before the deadline")

        serial = "".join(serial_parts)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{label}-serial.log").write_text(serial, encoding="utf-8")
        parsed = _parse_serial(serial)
        parsed["proof"] = Proof.from_json(parsed["proof_json"])
        parsed["trace"] = json.loads(parsed["trace_json"])
        parsed["name"] = name
        parsed["create_to_done_seconds"] = done_at - create_started
        parsed["lifetime_basis"] = "host-observed GCE create through delete"
    finally:
        print(f"[{label}] deleting {name}", flush=True)
        _gcloud(zone, "compute", "instances", "delete", name, "--quiet", check=False)
    parsed["vm_lifetime_seconds"] = time.monotonic() - create_started
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="koth-support-v15")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--matrix", default="data/bitext-support-scored-v2.jsonl")
    parser.add_argument("--n", type=int, default=2_160)
    parser.add_argument("--output", default="data/koth-support-tdx")
    parser.add_argument("--vm-hourly", type=float, default=0.2224888,
                        help="on-demand C3-standard-4 + Intel TDX list price")
    parser.add_argument("--baseline-cost-per-ticket", type=float, default=0.000240201)
    parser.add_argument("--replay-serial", action="store_true",
                        help="validate existing measure/full serial logs without new VMs")
    parser.add_argument("--measure-created-unix", type=float,
                        help="measurement VM name timestamp; required for replay")
    parser.add_argument("--full-created-unix", type=float,
                        help="full VM name timestamp; required for replay")
    parser.add_argument("--measurement-only", action="store_true",
                        help="run and full-DCAP verify only the independent one-task image ceremony")
    args = parser.parse_args()
    if args.n < 1 or args.vm_hourly <= 0 or args.baseline_cost_per_ticket <= 0:
        raise ValueError("n, vm-hourly, and baseline cost must be positive")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not args.replay_serial and not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    if args.replay_serial and (args.measure_created_unix is None
                               or args.full_created_unix is None):
        raise ValueError("replay requires --measure-created-unix and --full-created-unix")
    if args.replay_serial and args.measurement_only:
        raise ValueError("--replay-serial and --measurement-only are mutually exclusive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    build = build_support_artifact(args.matrix)
    bench = load_bitext_support_benchmark(exclude_source_indices=build.source_indices)
    suite = [bench]
    hotkey = "support-tdx-canary"

    # Build the local validator state before boot so the full proof's epoch/nonce is independently
    # determined rather than chosen after seeing an answer slice.
    local_root = tempfile.mkdtemp(prefix="support_tdx_validator_")
    chain = LocalFileChain(f"{local_root}/chain")
    store = LocalBundleStore(f"{local_root}/store")
    chain.register("burn")
    chain.register(hotkey)
    revision = store.upload("support/router", build.artifact)
    chain.commit(hotkey, commit_string(hotkey, "support/router", revision,
                                       build.artifact.source_hash, build.artifact.weights_hash))
    full_epoch = 1
    full_nonce = epoch_nonce(full_epoch, chain.beacon(full_epoch))

    if args.replay_serial:
        measure = _load_replay(output / "measure-serial.log",
                               args.measure_created_unix, "measure")
        full = _load_replay(output / "full-serial.log", args.full_created_unix, "full")
    else:
        with tempfile.TemporaryDirectory(prefix="support_tdx_metadata_") as td:
            temp = Path(td)
            source_b64 = temp / "source.b64"
            weights_b64 = temp / "weights.b64"
            secrets = temp / "secrets.env"
            source_b64.write_text(base64.b64encode(build.artifact.source_text.encode()).decode())
            weights_b64.write_text(base64.b64encode(build.artifact.weights).decode())
            secrets.write_text(f"OPENROUTER_API_KEY={api_key}\n")
            secrets.chmod(0o600)

            measure = _boot(
                image=args.image, zone=args.zone, label="measure", epoch=0,
                nonce="support-measurement-boot", hotkey=hotkey, n=1,
                source_b64=source_b64, weights_b64=weights_b64, secrets=secrets,
                output=output, deadline_seconds=900,
            )
            if not args.measurement_only:
                full = _boot(
                    image=args.image, zone=args.zone, label="full", epoch=full_epoch,
                    nonce=full_nonce, hotkey=hotkey, n=args.n,
                    source_b64=source_b64, weights_b64=weights_b64, secrets=secrets,
                    output=output, deadline_seconds=4_200,
                )

    expected_rtmr3 = rtmr.owner_expected_rtmr3(
        runtime_measurement=runtime_measurement(), suite_version=SUITE_VERSION,
        pool_allow_list=SUPPORT_MODELS,
    )
    if measure["measurements"]["rtmr3"] != expected_rtmr3:
        raise RuntimeError("measurement boot RTMR3 differs from the owner-derived value")

    if args.measurement_only:
        approved_rtmr = {i: measure["measurements"][f"rtmr{i}"] for i in (1, 2, 3)}
        proof: Proof = measure["proof"]
        verdict = verify_proof(
            proof,
            approved_measurements={runtime_measurement()},
            platform_public_hex="unused-for-tdx",
            expect_epoch=0,
            expect_nonce="support-measurement-boot",
            expect_hotkey=hotkey,
            expect_source_hash=build.artifact.source_hash,
            expect_weights_hash=build.artifact.weights_hash,
            suite=suite,
            n_per_bench=1,
            enforce=True,
            approved_mrtd={measure["measurements"]["mrtd"]},
            approved_rtmr=approved_rtmr,
            tcb_accept=tdx.DEFAULT_TCB_ACCEPT,
        )
        if not verdict.valid:
            raise RuntimeError(f"measurement proof failed full DCAP verification: {verdict.reason}")
        grounded = grounding_check(proof, measure["trace"], suite)
        if grounded != (True, "ok"):
            raise RuntimeError(f"measurement grounding failed: {grounded[1]}")
        summary = {
            "image": args.image,
            "zone": args.zone,
            "hardware": "GCP C3 Intel TDX",
            "attestation": "real TDX v4; enforce=True; full DCAP/TCB",
            "measurement_boot": {
                "measurements": measure["measurements"],
                "run_seconds": measure["run_seconds"],
                "vm_lifetime_seconds": measure["vm_lifetime_seconds"],
                "cost_usd": proof.total_cost_usd,
            },
            "proof": {
                "source_hash": proof.source_hash,
                "weights_hash": proof.weights_hash,
                "call_log_hash": proof.call_log_hash,
                "calls": proof.n_calls,
                "cost_usd": proof.total_cost_usd,
            },
        }
        (output / "proof.json").write_text(proof.to_json(), encoding="utf-8")
        (output / "trace.json").write_text(json.dumps(measure["trace"]), encoding="utf-8")
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        print("\n=== REAL GCP INTEL TDX SUPPORT MEASUREMENT ===")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    for key in ("mrtd", "rtmr1", "rtmr2", "rtmr3"):
        if full["measurements"][key] != measure["measurements"][key]:
            raise RuntimeError(f"independent boots disagree on {key}")
    proof: Proof = full["proof"]
    trace = full["trace"]
    if signing.sha256_hex(trace) != proof.call_log_hash:
        raise RuntimeError("trace hash does not match the attested proof")
    raw_quote = tdx.parse_quote(base64.b64decode(proof.quote.platform_sig[4:]))
    quoted = {
        "mrtd": raw_quote.mr_td.hex(),
        **{f"rtmr{i}": value.hex() for i, value in enumerate(raw_quote.rtmrs)},
    }
    if any(quoted[key] != full["measurements"][key]
           for key in ("mrtd", "rtmr1", "rtmr2", "rtmr3")):
        raise RuntimeError("signed quote measurements differ from the guest's reported registers")

    approved_rtmr = {i: measure["measurements"][f"rtmr{i}"] for i in (1, 2, 3)}
    gate = dict(
        approved_measurements={runtime_measurement()},
        platform_public_hex="unused-for-tdx",
        expect_epoch=full_epoch,
        expect_nonce=full_nonce,
        expect_hotkey=hotkey,
        expect_source_hash=build.artifact.source_hash,
        expect_weights_hash=build.artifact.weights_hash,
        suite=suite,
        n_per_bench=args.n,
        enforce=True,
        approved_mrtd={measure["measurements"]["mrtd"]},
        approved_rtmr=approved_rtmr,
        tcb_accept=tdx.DEFAULT_TCB_ACCEPT,
    )
    verdict = verify_proof(proof, **gate)
    if not verdict.valid:
        raise RuntimeError(f"enforcing proof verification failed: {verdict.reason}")
    grounded = grounding_check(proof, trace, suite)
    if grounded != (True, "ok"):
        raise RuntimeError(f"grounding failed: {grounded[1]}")

    # Exercise the complete KOTHValidator control flow with the same independently captured gates.
    owner_record = {
        "version": 1,
        "runtime_measurements": [runtime_measurement()],
        "mrtd": [measure["measurements"]["mrtd"]],
        "rtmr": {str(k): v for k, v in approved_rtmr.items()},
        "tcb_accept": sorted(tdx.DEFAULT_TCB_ACCEPT),
        "pool_allow_list": list(SUPPORT_MODELS),
    }
    chain.publish_owner_measurements(owner_record)
    store.upload_proof("support/router", full_epoch, proof.to_json())
    store.upload_trace("support/router", full_epoch, json.dumps(trace))
    chain.advance(EPOCH_BLOCKS)
    validator = KOTHValidator(
        {runtime_measurement()}, "unused-for-tdx", chain, KingChain(), suite, store,
        NoInferenceBackend(), n_per_bench=args.n,
        budget=args.baseline_cost_per_ticket * args.n / 2.0, f_min=0.90, enforce=True,
    )
    ready = validator.governance_ready()
    if ready != (True, "ok"):
        raise RuntimeError(f"enforcing validator is not governance-ready: {ready[1]}")
    report = validator.run_epoch(store_get_proof(store), epoch=full_epoch)
    if hotkey not in report.scored:
        raise RuntimeError(f"enforcing validator did not score proof: {report.dq.get(hotkey)}")

    # One fail-closed negative proves the captured image gate is active, not merely configured.
    wrong_gate = dict(gate)
    wrong_gate["approved_rtmr"] = {**approved_rtmr, 1: "00" * 48}
    rejected = verify_proof(proof, **wrong_gate)
    if rejected.valid or rejected.reason != "unapproved_runtime":
        raise RuntimeError(f"wrong RTMR1 was not rejected: {rejected.reason}")

    stat = verdict.per_bench[bench.name]
    model_cost_per_ticket = proof.total_cost_usd / stat.n
    route_cost_per_million = model_cost_per_ticket * 1_000_000
    vm_cost = full["vm_lifetime_seconds"] / 3_600.0 * args.vm_hourly
    vm_cost_per_million = vm_cost / stat.n * 1_000_000
    total_per_million = route_cost_per_million + vm_cost_per_million
    baseline_per_million = args.baseline_cost_per_ticket * 1_000_000
    models = Counter(entry["model"] for entry in trace)
    summary = {
        "image": args.image,
        "zone": args.zone,
        "hardware": "GCP C3 Intel TDX",
        "attestation": "real TDX v4; enforce=True; full DCAP/TCB",
        "measurement_boot": {
            "measurements": measure["measurements"],
            "run_seconds": measure["run_seconds"],
            "vm_lifetime_seconds": measure["vm_lifetime_seconds"],
            "lifetime_basis": measure["lifetime_basis"],
            "cost_usd": measure["proof"].total_cost_usd,
        },
        "full_boot": {
            "measurements": full["measurements"],
            "run_seconds": full["run_seconds"],
            "create_to_done_seconds": full["create_to_done_seconds"],
            "vm_lifetime_seconds": full["vm_lifetime_seconds"],
            "lifetime_basis": full["lifetime_basis"],
        },
        "proof": {
            "source_hash": proof.source_hash,
            "weights_hash": proof.weights_hash,
            "call_log_hash": proof.call_log_hash,
            "calls": proof.n_calls,
            "model_calls": dict(models),
            "cost_usd": proof.total_cost_usd,
            "accuracy": stat.acc,
            "accuracy_lcb": stat.lcb,
            "validator_scored": True,
            "wrong_rtmr1_rejection": rejected.reason,
        },
        "economics": {
            "vm_hourly_usd": args.vm_hourly,
            "vm_lifetime_seconds": full["vm_lifetime_seconds"],
            "vm_lifetime_basis": full["lifetime_basis"],
            "vm_cost_usd": vm_cost,
            "route_cost_per_million_usd": route_cost_per_million,
            "vm_cost_per_million_usd": vm_cost_per_million,
            "total_per_million_usd": total_per_million,
            "baseline_per_million_usd": baseline_per_million,
            "advantage": baseline_per_million / total_per_million,
            "passes_2x": total_per_million <= baseline_per_million / 2.0,
        },
    }
    (output / "proof.json").write_text(proof.to_json(), encoding="utf-8")
    (output / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                          encoding="utf-8")
    print("\n=== REAL GCP INTEL TDX SUPPORT CANARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["economics"]["passes_2x"]:
        raise SystemExit("hardware proof passed, but the end-to-end 2x economics gate failed")


if __name__ == "__main__":
    main()
