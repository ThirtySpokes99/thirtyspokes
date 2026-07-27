"""The per-epoch POOL REFERENCE — what every pool model scored, on the asks the miners ran.

`verify.router_headroom` scores a router against what was ACHIEVABLE at its own price: the Zero
frontier (randomising over fixed pool models) and the budget-constrained per-ask oracle. Both need
every `(ask, model)` cell for the epoch's slice — and a validator runs NO inference, so it cannot
know what the other models would have answered. Without this the router scalar silently falls back
to the old absolute one.

So the OWNER measures the pool once per epoch and publishes the result, using the same shape as the
governance record (`koth/governance.py`): the body is content-addressed in the owner's public bucket,
and only its sha256 goes on-chain. Validators fetch and verify by hash, so a substituted or tampered
reference is rejected rather than trusted.

WHY A SIGNATURE AND NOT AN ON-CHAIN HASH. The first cut committed the record's sha256 on-chain, the
way `koth/governance.py` does. That cannot work, for a reason the design doc already records about
miners: the chain gives each hotkey exactly ONE commitment slot (`CommitmentOf[(netuid, hotkey)]`),
and `set_commitment` overwrites it. The owner's slot already holds the approved-measurement record
(`kothgov1|…`), so writing a reference digest there each epoch would erase the subnet's governance —
every validator would read `mrtd_gate_unset` and refuse to score at all. A per-epoch on-chain anchor
is therefore not available to the owner.

So the record is SIGNED with the owner's hotkey and served from the owner's bucket at a path both
sides derive from `(epoch, nonce)`. Integrity is still rooted on-chain: validators resolve the owner
hotkey from `SubnetOwnerHotkey` and reject any record that key did not sign. This is strictly cheaper
too — no per-epoch extrinsic — and it leaves the governance slot alone.

WHY THE VALIDATOR MUST CHECK MORE THAN THE SIGNATURE. A signature only proves the owner wrote the
body — not that it describes THIS epoch. A reference from an easier epoch would make every miner look
good (a low Zero frontier is easy to beat) and a reference from a harder one would make everyone look
bad. So the record carries `(epoch, nonce, suite_version, n_per_bench, task_ids)` and `matches`
rejects any mismatch. The nonce is chain-derived and unpredictable, so the owner cannot pre-select a
flattering slice either.

COST. `n_per_bench x |pool|` calls per epoch — 8 x 6 = 48, well under a dollar — and a slice may be
reused across several epochs to amortise it further.
"""

from __future__ import annotations

import hashlib
import json

from .benchmarks import Benchmark, bench_seed


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def record_path(epoch: int, nonce: str) -> str:
    """Where this epoch's reference lives. Derived from `(epoch, nonce)` so a validator can address
    it without being told a hash — the nonce is chain-derived, so the path is unforgeable-by-guessing
    and unique per epoch."""
    return f"reference/{int(epoch)}-{str(nonce)[:32]}.json"


def envelope(record: dict, sign) -> dict:
    """Wrap a record with the owner's signature over its canonical bytes."""
    return {"v": 1, "record": record, "sig": sign(canonical(record))}


def open_envelope(raw: bytes, *, owner_ss58: str, verify_sig) -> dict:
    """Unwrap + authenticate. Raises unless the OWNER's key signed exactly these bytes."""
    env = json.loads(raw)
    rec, sig = env.get("record"), env.get("sig")
    if not isinstance(rec, dict) or not sig:
        raise ValueError("malformed pool-reference envelope")
    if not verify_sig(canonical(rec), sig, owner_ss58):
        raise ValueError("pool reference not signed by the subnet owner")
    return rec


def wallet_signer(wallet):
    """Sign with a Bittensor wallet's hotkey — the same identity `SubnetOwnerHotkey` names."""
    return lambda data: wallet.hotkey.sign(data).hex()


def keypair_verifier():
    """Verify an sr25519 signature against an ss58 address. Lazily imported so the offline test
    path (and any validator that never sees a reference) needs no wallet library."""
    def verify_sig(data: bytes, sig: str, ss58: str) -> bool:
        try:
            from bittensor_wallet import Keypair
            return bool(Keypair(ss58_address=ss58).verify(data, bytes.fromhex(sig)))
        except Exception:                   # noqa: BLE001 — unverifiable == not verified
            return False
    return verify_sig


def build(suite: list[Benchmark], *, epoch: int, nonce: str, n_per_bench: int,
          models: list[str], backend, max_tokens: int = 16384,
          reasoning: dict | None = None, workers: int = 8,
          deadline_s: float | None = None, progress=None) -> dict:
    """Run every pool model over the epoch's slice and record what each scored and cost.

    The slice is re-derived with `bench_seed(nonce, epoch, bench.name)` — byte-identical to what the
    miners' runtimes sampled — so the reference describes the same asks that were scored.

    `reasoning` is passed through because `max_tokens` counts THINKING tokens: without it a reasoning
    model spends its whole budget deliberating and returns an empty answer, which would enter the
    reference as a legitimate-looking failure and understate that model on the frontier.

    `deadline_s` BOUNDS THE WHOLE BUILD, and it is not optional in production. This runs per epoch:
    overrun the epoch and the reference describes a slice nobody is being scored on any more, so it
    is dead on arrival. Provider calls can also hang past their own timeout — observed here, a single
    request left idle with the connection ESTABLISHED and no response, holding the build for over
    half an hour on 3.5 seconds of CPU. Cells still outstanding at the deadline are treated exactly
    like failed cells (their rows drop), so a hang costs coverage rather than the whole epoch.

    `progress(done, total)` is called as cells land. A silent hour and a working hour look identical
    from a cron log otherwise.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # RANKED benchmarks only. The frontier weights each row by `w_b / n_b`, so a weight-0
    # eligibility floor (MMLU, GSM8K) contributes exactly nothing to either baseline — measuring the
    # pool on it would buy the owner a 3x bill every epoch for rows the scalar discards.
    tasks = []
    for bench in suite:
        if getattr(bench, "weight", 0.0) <= 0.0:
            continue
        for t in bench.sample(n_per_bench, bench_seed(nonce, epoch, bench.name)):
            tasks.append((bench, t))

    params = {"max_tokens": max_tokens}
    if reasoning:
        params["reasoning"] = reasoning

    cells: dict = {}

    def one(job):
        i, (bench, t), model = job
        try:
            text, _tin, _tout, cost = backend.complete(
                model, [{"role": "user", "content": t.prompt}], dict(params))
            return i, model, float(bench.grade(text, t.gold)), float(cost)
        except Exception:                       # noqa: BLE001 — a dead cell must not sink the epoch
            return i, model, None, 0.0

    jobs = [(i, bt, m) for i, bt in enumerate(tasks) for m in models]
    started = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(one, j) for j in jobs]
        done = 0
        try:
            for f in as_completed(futures, timeout=deadline_s):
                i, model, score, cost = f.result()
                cells[(i, model)] = (score, cost)
                done += 1
                if progress:
                    progress(done, len(jobs))
        except TimeoutError:
            for f in futures:
                f.cancel()
            if progress:
                progress(done, len(jobs))
            print(f"[koth-reference] deadline reached after {time.monotonic() - started:.0f}s "
                  f"with {done}/{len(jobs)} cells; the rest count as failed", flush=True)
    finally:
        # do NOT block on in-flight calls: a hung provider request would otherwise re-impose the
        # unbounded wait the deadline exists to remove.
        ex.shutdown(wait=False, cancel_futures=True)

    # Rows with ANY failed or MISSING cell are dropped: a partial row would silently distort both
    # frontiers (a missing expensive model makes the pool look cheaper AND weaker than it is), and
    # after a deadline some cells were never recorded at all.
    keep = [i for i in range(len(tasks))
            if all(cells.get((i, m), (None, 0.0))[0] is not None for m in models)]
    from .runtime import SUITE_VERSION
    return {
        "v": 1,
        "epoch": epoch,
        "nonce": nonce,
        "suite_version": SUITE_VERSION,
        "n_per_bench": n_per_bench,
        "models": list(models),
        "task_ids": [tasks[i][1].task_id for i in keep],
        # which benchmark each row came from. The validator needs it to weight the rows the same way
        # the miner's own score is weighted (`Σ_b w_b·acc_b`) — without it a weight-0 eligibility
        # benchmark like MMLU would build half the frontier the miner is judged against.
        "benchmarks": [tasks[i][0].name for i in keep],
        "scores": [[cells[(i, m)][0] for m in models] for i in keep],
        "costs": [[cells[(i, m)][1] for m in models] for i in keep],
    }


def publish(record: dict, sign, *, bucket: str | None = None, token: str | None = None) -> str:
    """Sign and upload to the owner's bucket at this epoch's derived path. Returns the path."""
    import os
    import pathlib
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    from . import imagestore

    path = record_path(record["epoch"], record["nonce"])
    staged = pathlib.Path(tempfile.mkdtemp(prefix="koth_ref_"))
    try:
        p = staged / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(envelope(record, sign)))
        api = HfApi(token=token or os.environ.get("OWNER_HF_TOKEN") or os.environ.get("HF_TOKEN"))
        api.sync_bucket(str(staged), imagestore.bucket_uri(bucket), delete=False)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return path


def fetch(epoch: int, nonce: str, *, owner_ss58: str, verify_sig, bucket: str | None = None,
          urls=None) -> dict:
    import urllib.request

    from . import imagestore
    last = None
    for url in list(urls or [imagestore.public_url(record_path(epoch, nonce), bucket=bucket)]):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return open_envelope(r.read(), owner_ss58=owner_ss58, verify_sig=verify_sig)
        except Exception as e:                  # noqa: BLE001 — try the next mirror
            last = e
    raise RuntimeError(f"could not fetch pool reference for epoch {epoch}: {last}")


def matches(record: dict, *, epoch: int, nonce: str, n_per_bench: int) -> bool:
    """Is this reference actually about the slice being scored? The hash alone does not say so —
    an owner-signed record from an EASIER epoch would lower the Zero frontier and flatter every
    miner. The nonce is chain-derived and unpredictable, so a flattering slice cannot be chosen."""
    from .runtime import SUITE_VERSION
    return (record.get("epoch") == epoch and record.get("nonce") == nonce
            and record.get("n_per_bench") == n_per_bench
            and record.get("suite_version") == SUITE_VERSION
            and bool(record.get("scores")) and bool(record.get("costs")))


def chain_reader(chain, *, n_per_bench: int, owner_hotkey: str | None = None,
                 bucket: str | None = None, verify_sig=None):
    """The `pool_reference=` callable a validator passes to `KOTHValidator`.

    Resolves the owner from the CHAIN (never from local config — a validator that can be told who
    the owner is can be pointed at an attacker's key), fetches this epoch's record from the owner's
    bucket, checks the owner signed it, and checks it describes THIS slice. Returns the whole
    verified RECORD — the validator needs `task_ids` to align per-ask regret and `benchmarks` to
    weight the frontier rows, not just the two matrices.

    Returns None on ANY failure so scoring degrades to the legacy scalar instead of the subnet
    stalling: a reference outage is an owner problem, not a miner one.
    """
    cache: dict = {}
    check = verify_sig or keypair_verifier()

    def read(epoch: int, nonce: str):
        if epoch in cache:
            return cache[epoch]
        out = None
        try:
            owner = owner_hotkey or chain.owner_hotkey()
            if owner:
                rec = fetch(epoch, nonce, owner_ss58=owner, verify_sig=check, bucket=bucket)
                if matches(rec, epoch=epoch, nonce=nonce, n_per_bench=n_per_bench):
                    out = rec
        except Exception:                       # noqa: BLE001 — never break scoring on a fetch error
            out = None
        cache[epoch] = out
        return out

    return read


def main() -> None:
    """`orchestra-koth-reference` — the owner-side producer.

    Measures every pinned pool model on the epoch's slice and publishes the signed result, which is
    what turns the validators' scalar from absolute accuracy into frontier-relative headroom. Run it
    once per epoch (a cron next to the validator is the intended shape); if it misses an epoch, that
    epoch simply scores absolutely.
    """
    import argparse

    from ..eval import config
    from ..gateway.gateway import OpenRouterBackend
    from .benchmarks import real_suite
    from .epoch import current_epoch, epoch_nonce

    p = argparse.ArgumentParser(description="Build + publish the KOTH per-epoch pool reference")
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument("--wallet", default="owner")
    p.add_argument("--hotkey", default="default", help="hotkey NAME inside the wallet")
    p.add_argument("--network", default="finney")
    p.add_argument("--pool", required=True, help="comma-separated pinned model allow-list")
    p.add_argument("--n-per-bench", type=int, default=8,
                   help="MUST match the validators' --n-per-bench, or the record will not match")
    p.add_argument("--epoch", type=int, help="default: the current epoch")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--reasoning-effort", default="low",
                   help="'' to disable; max_tokens counts THINKING tokens, so an unbounded "
                        "reasoning model returns an empty answer and understates itself")
    p.add_argument("--deadline-s", type=float, default=900.0,
                   help="hard wall-clock bound on the whole build. A reference that outruns its "
                        "epoch describes a slice nobody is scored on any more; cells still "
                        "outstanding count as failed and their rows drop")
    p.add_argument("--call-timeout", type=float, default=180.0,
                   help="per-request timeout to the provider")
    p.add_argument("--dry-run", action="store_true", help="build and print; do not publish")
    args = p.parse_args()

    from ..subnet.chain import BittensorChain
    chain = BittensorChain(args.netuid, args.wallet, network=args.network, hotkey=args.hotkey)
    epoch = args.epoch if args.epoch is not None else current_epoch(chain)
    nonce = epoch_nonce(epoch, chain.beacon(epoch))
    models = [m.strip() for m in args.pool.split(",") if m.strip()]

    cfg = config.LiveConfig()
    cfg.require_key()
    backend = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=args.call_timeout,
                                max_retries=2, price_fn=config.price_for)
    print(f"measuring {len(models)} models on the epoch slice "
          f"(deadline {args.deadline_s:.0f}s)...", flush=True)
    rec = build(real_suite(), epoch=epoch, nonce=nonce, n_per_bench=args.n_per_bench,
                models=models, backend=backend, max_tokens=args.max_tokens,
                reasoning=({"effort": args.reasoning_effort} if args.reasoning_effort else None),
                deadline_s=args.deadline_s,
                progress=lambda d, t: print(f"  {d}/{t} cells", flush=True) if d % 5 == 0 or d == t
                else None)

    n_rows, n_models = len(rec["scores"]), len(rec["models"])
    print(f"epoch {epoch} nonce {nonce[:16]}… — {n_rows} complete asks x {n_models} models")
    if not n_rows:
        raise SystemExit("[koth-reference] every row had a failed cell — nothing to publish")
    # The go/no-go number: how much a perfect per-ask router could add over a featureless one.
    # Below the validators' floor the reference is still published (it is the evidence for the
    # judgement) but nobody will be scored frontier-relatively on it.
    from .verify import achievable_gap, row_weights_for
    w = row_weights_for(rec["benchmarks"], {b.name: b.weight for b in real_suite()})
    print(f"  achievable gap (oracle - zero): {achievable_gap(rec['scores'], rec['costs'], w):.4f}")
    if args.dry_run:
        print(json.dumps(rec, indent=2)[:2000])
        return
    path = publish(rec, wallet_signer(chain.wallet))
    print(f"  published {path} (signed by {chain.wallet.hotkey.ss58_address})")


if __name__ == "__main__":
    main()
