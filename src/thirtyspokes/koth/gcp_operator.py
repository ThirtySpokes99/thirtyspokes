"""Operator for the owner's one-proof-per-boot GCP Intel TDX runtime image.

The measured image has no shell or daemon.  This process runs outside the VM: it derives the live
epoch, boots a fresh C3 TDX instance, strictly reconstructs the attested serial payload, uploads it
to the miner's committed Hugging Face repo, and deletes the instance.  Boot/runtime failures are
retried on a fresh VM before the epoch deadline; every attempt is cleaned up in ``finally``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .commit import parse_commit, verify_commit
from .epoch import EPOCH_BLOCKS, epoch_nonce
from .proof import Proof
from .store import HFBundleStore
from ..subnet.chain import BittensorChain


_CHUNK = re.compile(
    r"(?P<kind>PROOF|TRACE)(?P<gzip>-GZIP)?-CHUNK\s+"
    r"(?P<index>\d+)/(?P<total>\d+)\s+"
    r"(?:sha256=(?P<digest>[0-9a-f]{64})\s+)?(?P<data>[A-Za-z0-9+/=]+)"
)
_EARLY_FAILURES = ("KOTH-ERROR", "SandboxError", "Traceback (most recent call last)")


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
        raise RuntimeError(f"incomplete {kind} chunks")

    chunks = {}
    for index, copies in candidates.items():
        valid = [(data, digest) for data, digest in copies
                 if digest is None or hashlib.sha256(data.encode()).hexdigest() == digest]
        if not valid:
            raise RuntimeError(f"{kind} chunk {index} has no digest-valid copy")
        chunks[index] = max(valid, key=lambda item: (item[1] is not None, len(item[0])))[0]
    raw = base64.b64decode("".join(chunks[i] for i in range(1, total + 1)), validate=True)
    return gzip.decompress(raw).decode() if compressed else raw.decode()


def _decode_legacy(serial: str, kind: str) -> str:
    marker = f"KOTH-{kind} "
    for line in reversed(serial.splitlines()):
        if marker not in line:
            continue
        raw = base64.b64decode(line.split(marker, 1)[1].strip(), validate=True).decode()
        json.loads(raw)
        return raw
    raise RuntimeError(f"missing legacy {kind} payload")


def decode_payloads(serial: str) -> tuple[str, str]:
    """Strictly decode both new repeated gzip chunks and the older v14 full-line format."""
    payloads = []
    for kind in ("PROOF", "TRACE"):
        try:
            raw = _decode_chunks(serial, kind)
        except (RuntimeError, binascii.Error, UnicodeError, gzip.BadGzipFile):
            raw = _decode_legacy(serial, kind)
        json.loads(raw)
        payloads.append(raw)
    return payloads[0], payloads[1]


def _gcloud(zone: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + ":/opt/google-cloud-sdk/bin"
    return subprocess.run(
        ["gcloud", *args, f"--zone={zone}"], capture_output=True, text=True,
        check=False, timeout=timeout, env=env,
    )


def _delete_instance(zone: str, name: str) -> None:
    last = None
    for attempt in range(3):
        try:
            result = _gcloud(zone, "compute", "instances", "delete", name, "--quiet")
            last = result.stderr
            if result.returncode == 0 or "was not found" in result.stderr:
                return
        except subprocess.TimeoutExpired as exc:
            last = str(exc)
        if attempt < 2:
            time.sleep(5)
    raise RuntimeError(f"could not delete GCP instance {name}: {last}")


def _boot_once(*, zone: str, image: str, machine_type: str, name: str, epoch: int,
               nonce: str, hotkey: str, pool: list[str], n_per_bench: int,
               source_b64: Path, weights_b64: Path, secrets: Path, deadline: float,
               poll: float, log_path: Path) -> tuple[str, str]:
    # A prior operator crash may have left this deterministic attempt name behind. Delete it before
    # recreating; the same finally cleanup below covers every normal failure path.
    try:
        _delete_instance(zone, name)
    except RuntimeError as exc:
        if "was not found" not in str(exc):
            raise

    metadata = (
        f"^@^koth-epoch={epoch}@koth-nonce={nonce}@koth-hotkey={hotkey}"
        f"@koth-pool={','.join(pool)}@koth-n-per-bench={n_per_bench}"
    )
    files = (f"koth-secrets={secrets},koth-agent-source={source_b64},"
             f"koth-agent-weights={weights_b64}")
    serial = ""
    shown = 0                 # heartbeat lines already echoed (serial is re-read from 0 each poll)
    try:
        created = _gcloud(
            zone, "compute", "instances", "create", name,
            f"--machine-type={machine_type}", "--confidential-compute-type=TDX",
            "--maintenance-policy=TERMINATE", f"--image={image}",
            "--network-interface=nic-type=GVNIC", "--boot-disk-size=50GB",
            f"--metadata={metadata}", f"--metadata-from-file={files}", timeout=180,
        )
        if created.returncode != 0:
            raise RuntimeError(f"VM creation failed: {created.stderr[-2000:]}")
        stop_at = time.monotonic() + deadline
        while time.monotonic() < stop_at:
            time.sleep(poll)
            got = _gcloud(
                zone, "compute", "instances", "get-serial-port-output", name, "--start=0",
            )
            if got.returncode != 0:
                continue
            serial = got.stdout
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(serial, encoding="utf-8")
            # Surface the enclave's per-task heartbeat. Without it a booted VM is silent until the
            # whole suite finishes, so "still working" and "hung" look identical from out here — the
            # reason three healthy runs were killed as hangs and a dead 49-minute epoch could not be
            # placed. Only newly-arrived lines are printed; the serial buffer is re-read from 0.
            # SUBSTRING, not startswith: the serial console prefixes every line with a kernel
            # timestamp and the emitting process, e.g.
            #   "[   87.957190] python[427]: KOTH-TASK 1/6 rung=1 rungs_used=1 cost=0.00040"
            # A prefix match silently echoed nothing while the enclave was printing perfectly.
            ticks = [ln for ln in serial.splitlines() if "KOTH-TASK " in ln]
            for ln in ticks[shown:]:
                print(f"[{name}] {ln}", flush=True)
            shown = max(shown, len(ticks))
            try:
                return decode_payloads(serial)
            except (RuntimeError, binascii.Error, UnicodeError, json.JSONDecodeError):
                pass
            marker = next((item for item in _EARLY_FAILURES if item in serial), None)
            if marker:
                tail = serial[serial.rfind(marker):][-2000:]
                raise RuntimeError(f"early guest failure ({marker}): {tail}")
        raise TimeoutError(f"{name} did not emit a complete proof and trace within {deadline}s")
    finally:
        _delete_instance(zone, name)


def _wait_for_epoch(chain: BittensorChain, last_epoch: int | None,
                    min_blocks: int, poll: float) -> tuple[int, str]:
    while True:
        # A TRANSIENT CHAIN RPC FAILURE MUST NOT KILL THE MINER. Same defect, same signature as the
        # one fixed in the validator's run_forever, in a second place nobody checked: the substrate
        # node returns a body with no "result", bittensor's `_get_block_number` subscripts None, and
        # the TypeError escapes here — killing a continuous mining run outright. Observed on testnet
        # 526 after the validator fix, which is why "I fixed that bug" was not the same as "that bug
        # is fixed". A miner that cannot read a block should wait and ask again.
        try:
            block = chain.current_block()
            epoch = block // EPOCH_BLOCKS
            remaining = EPOCH_BLOCKS - block % EPOCH_BLOCKS
            if epoch != last_epoch and remaining >= min_blocks:
                return epoch, epoch_nonce(epoch, chain.beacon(epoch))
        except Exception as exc:                # noqa: BLE001 — chain/network flakiness
            print(f"[koth-gcp] chain unreadable ({type(exc).__name__}: {str(exc)[:100]}) "
                  f"— retrying in {poll:.0f}s", flush=True)
        time.sleep(poll)


def _committed_artifact(chain: BittensorChain, store: HFBundleStore, repo: str | None,
                        revision: str | None):
    hotkey = chain.hotkey_ss58()
    commit = next((item for item in chain.revealed_commitments() if item.hotkey == hotkey), None)
    parsed = parse_commit(commit.data) if commit else None
    if parsed is None:
        raise RuntimeError(f"no revealed KOTH artifact commitment for miner hotkey {hotkey}")
    committed_repo, committed_revision, _salt = parsed
    if repo and repo != committed_repo:
        raise RuntimeError(f"--repo {repo} differs from on-chain repo {committed_repo}")
    if revision and revision != committed_revision:
        raise RuntimeError("--revision differs from the immutable revision committed on-chain")
    artifact = store.download(committed_repo, committed_revision)
    if not verify_commit(commit.data, hotkey, artifact.source_hash, artifact.weights_hash):
        raise RuntimeError("downloaded artifact does not satisfy the miner's on-chain binding")
    return committed_repo, committed_revision, artifact


def _existing_epoch(store: HFBundleStore, repo: str, epoch: int, nonce: str,
                    hotkey: str) -> bool:
    proof_json = store.download_proof(repo, epoch)
    trace_json = store.download_trace(repo, epoch)
    if proof_json is None or trace_json is None:
        return False
    try:
        proof = Proof.from_json(proof_json)
        json.loads(trace_json)
        return proof.epoch == epoch and proof.nonce == nonce and proof.hotkey == hotkey
    except (ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:  # pragma: no cover — real GCP/chain/HF operator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--image", required=True, help="approved GCP measured-image name")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--machine-type", default="c3-standard-4")
    parser.add_argument("--repo", help="optional safety check; normally discovered from chain")
    parser.add_argument("--revision", help="optional safety check; normally discovered from chain")
    # MUST EQUAL THE VALIDATOR'S. This is the THIRD place the slice size lives (miner daemon,
    # validator daemon, this operator) and the third chance to get it wrong. The validator
    # re-derives the slice and rejects any proof whose task set differs — `unexpected_task`, a DQ —
    # so an operator left at 8 against a validator at 16 silently disqualifies every epoch it runs.
    # Pinned by a test alongside the other three.
    parser.add_argument("--n-per-bench", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=0, help="0 runs continuously")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-blocks", type=int, default=50,
                        help="do not start an epoch with fewer blocks remaining")
    parser.add_argument("--attempt-deadline", type=float, default=900,
                        help="abandon a VM after this long. Keep it UNDER the validators' grace "
                             "(--grace-blocks x 12s): a run that finishes after the grace point is "
                             "unscorable, so waiting longer only burns the VM. Sized for the "
                             "harness RUN_BUDGET_S plus boot and upload.")
    parser.add_argument("--poll", type=float, default=15)
    parser.add_argument("--output", default="logs/koth-gcp-operator")
    parser.add_argument("--strict", action="store_true",
                        help="exit on a missed epoch (use for launch-gate soaks)")
    args = parser.parse_args()
    if (args.n_per_bench < 1 or args.max_attempts < 1 or args.min_blocks < 1
            or args.min_blocks > EPOCH_BLOCKS or args.epochs < 0):
        raise ValueError("invalid n-per-bench, attempts, min-blocks, or epochs")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    token = (os.environ.get("MINER_HF_TOKEN") or os.environ.get("OWNER_HF_TOKEN")
             or os.environ.get("HF_TOKEN"))
    if not api_key or not token:
        raise RuntimeError("OPENROUTER_API_KEY and a miner-writable HF token are required")

    chain = BittensorChain(args.netuid, args.wallet, args.network, hotkey=args.hotkey)
    store = HFBundleStore(token=token)
    repo, revision, artifact = _committed_artifact(chain, store, args.repo, args.revision)
    governance = chain.owner_measurements()
    pool = list((governance or {}).get("pool_allow_list") or [])
    if not pool:
        raise RuntimeError("owner governance has no pinned pool allow-list")
    hotkey = chain.hotkey_ss58()
    prefix = hashlib.sha256(hotkey.encode()).hexdigest()[:8]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[koth-gcp] hotkey={hotkey} repo={repo}@{revision} image={args.image} "
          f"netuid={args.netuid} pool={','.join(pool)}", flush=True)

    completed = 0
    last_epoch = None
    with tempfile.TemporaryDirectory(prefix="koth-gcp-metadata-") as temporary:
        root = Path(temporary)
        source_b64, weights_b64, secrets = root / "source.b64", root / "weights.b64", root / "secrets.env"
        source_b64.write_text(base64.b64encode(artifact.source_text.encode()).decode())
        weights_b64.write_text(base64.b64encode(artifact.weights).decode())
        secrets.touch(mode=0o600)
        secrets.write_text(f"OPENROUTER_API_KEY={api_key}\n")

        while args.epochs == 0 or completed < args.epochs:
            epoch, nonce = _wait_for_epoch(chain, last_epoch, args.min_blocks, args.poll)
            last_epoch = epoch
            if _existing_epoch(store, repo, epoch, nonce, hotkey):
                print(f"[koth-gcp] epoch {epoch} already uploaded; resuming without a VM", flush=True)
                completed += 1
                continue
            errors = []
            uploaded = False
            for attempt in range(1, args.max_attempts + 1):
                if chain.current_block() // EPOCH_BLOCKS != epoch:
                    errors.append("epoch expired before retry")
                    break
                name = f"koth-{prefix}-e{epoch}-a{attempt}"
                print(f"[koth-gcp] epoch={epoch} attempt={attempt} creating {name}", flush=True)
                try:
                    proof_json, trace_json = _boot_once(
                        zone=args.zone, image=args.image, machine_type=args.machine_type,
                        name=name, epoch=epoch, nonce=nonce, hotkey=hotkey, pool=pool,
                        n_per_bench=args.n_per_bench, source_b64=source_b64,
                        weights_b64=weights_b64, secrets=secrets,
                        deadline=args.attempt_deadline, poll=args.poll,
                        log_path=output / f"{name}.serial.log",
                    )
                    proof = Proof.from_json(proof_json)
                    if (proof.epoch, proof.nonce, proof.hotkey) != (epoch, nonce, hotkey):
                        raise RuntimeError("proof epoch/nonce/hotkey binding mismatch")
                    if (proof.source_hash, proof.weights_hash) != (
                            artifact.source_hash, artifact.weights_hash):
                        raise RuntimeError("proof artifact hashes differ from the on-chain bundle")
                    if chain.current_block() // EPOCH_BLOCKS != epoch:
                        raise RuntimeError("proof completed after its epoch expired")
                    store.upload_proof(repo, epoch, proof_json)
                    store.upload_trace(repo, epoch, trace_json)
                    summary = {
                        "epoch": epoch, "nonce": nonce, "attempt": attempt, "repo": repo,
                        "revision": revision, "hotkey": hotkey, "n_calls": proof.n_calls,
                        "n_results": len(proof.results), "cost_usd": proof.total_cost_usd,
                        "report_data": proof.report_data(), "errors_before_success": errors,
                    }
                    (output / f"epoch-{epoch}.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n"
                    )
                    print(f"[koth-gcp] epoch={epoch} uploaded calls={proof.n_calls} "
                          f"cost=${proof.total_cost_usd:.8f}", flush=True)
                    uploaded = True
                    completed += 1
                    break
                except Exception as exc:  # noqa: BLE001 — retry only on a completely fresh VM
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    print(f"[koth-gcp] epoch={epoch} attempt={attempt} failed: {error}", flush=True)
            if not uploaded:
                (output / f"epoch-{epoch}-failed.json").write_text(
                    json.dumps({"epoch": epoch, "nonce": nonce, "errors": errors},
                               indent=2, sort_keys=True) + "\n"
                )
                if args.strict:
                    raise RuntimeError(f"epoch {epoch} failed after {len(errors)} errors")


if __name__ == "__main__":
    main()
