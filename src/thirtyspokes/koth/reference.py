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

WHY THE VALIDATOR MUST CHECK MORE THAN THE HASH. A hash only proves the body is the one the owner
committed — not that it describes THIS epoch. A reference from an easier epoch would make every
miner look good (a low Zero frontier is easy to beat) and a reference from a harder one would make
everyone look bad. So the record carries `(epoch, nonce, suite_version, n_per_bench, task_ids)` and
`load_for` rejects any mismatch. The nonce is chain-derived and unpredictable, so the owner cannot
pre-select a flattering slice either.

COST. `n_per_bench x |pool|` calls per epoch — 8 x 6 = 48, well under a dollar — and a slice may be
reused across several epochs (`--reuse`) to amortise it further.
"""

from __future__ import annotations

import hashlib
import json


from .benchmarks import Benchmark, bench_seed

REF_PREFIX = "kothref1|"


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def commit_string(record_digest: str) -> str:
    """What goes on-chain: 9 + 64 = 73 bytes, so it fits a plain (immediate) commitment."""
    return f"{REF_PREFIX}{record_digest}"


def parse_commit(data: str) -> str | None:
    if not isinstance(data, str) or not data.startswith(REF_PREFIX):
        return None
    d = data[len(REF_PREFIX):].strip()
    return d if len(d) == 64 and all(c in "0123456789abcdef" for c in d.lower()) else None


def record_path(record_digest: str) -> str:
    return f"reference/{record_digest}.json"


def verify(raw: bytes, expect_digest: str) -> dict:
    got = hashlib.sha256(raw).hexdigest()
    if got != expect_digest:
        raise ValueError(f"pool reference hash mismatch: committed {expect_digest}, got {got}")
    return json.loads(raw)


def build(suite: list[Benchmark], *, epoch: int, nonce: str, n_per_bench: int,
          models: list[str], backend, max_tokens: int = 16384,
          reasoning: dict | None = None, workers: int = 8) -> dict:
    """Run every pool model over the epoch's slice and record what each scored and cost.

    The slice is re-derived with `bench_seed(nonce, epoch, bench.name)` — byte-identical to what the
    miners' runtimes sampled — so the reference describes the same asks that were scored.

    `reasoning` is passed through because `max_tokens` counts THINKING tokens: without it a reasoning
    model spends its whole budget deliberating and returns an empty answer, which would enter the
    reference as a legitimate-looking failure and understate that model on the frontier.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    for bench in suite:
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
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            i, model, score, cost = f.result()
            cells[(i, model)] = (score, cost)

    # Rows with ANY failed cell are dropped: a partial row would silently distort both frontiers
    # (a missing expensive model makes the pool look cheaper AND weaker than it is).
    keep = [i for i in range(len(tasks)) if all(cells[(i, m)][0] is not None for m in models)]
    from .runtime import SUITE_VERSION
    return {
        "v": 1,
        "epoch": epoch,
        "nonce": nonce,
        "suite_version": SUITE_VERSION,
        "n_per_bench": n_per_bench,
        "models": list(models),
        "task_ids": [tasks[i][1].task_id for i in keep],
        "scores": [[cells[(i, m)][0] for m in models] for i in keep],
        "costs": [[cells[(i, m)][1] for m in models] for i in keep],
    }


def publish(record: dict, *, bucket: str | None = None, token: str | None = None) -> str:
    """Upload to the owner's bucket, named by its own hash. Returns the digest to commit on-chain."""
    import os
    import pathlib
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    from . import imagestore

    d = digest(record)
    staged = pathlib.Path(tempfile.mkdtemp(prefix="koth_ref_"))
    try:
        p = staged / record_path(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(canonical(record))
        api = HfApi(token=token or os.environ.get("OWNER_HF_TOKEN") or os.environ.get("HF_TOKEN"))
        api.sync_bucket(str(staged), imagestore.bucket_uri(bucket), delete=False)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return d


def fetch(record_digest: str, *, bucket: str | None = None, urls=None) -> dict:
    import urllib.request

    from . import imagestore
    last = None
    for url in list(urls or [imagestore.public_url(record_path(record_digest), bucket=bucket)]):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return verify(r.read(), record_digest)
        except Exception as e:                  # noqa: BLE001 — try the next mirror
            last = e
    raise RuntimeError(f"could not fetch pool reference {record_digest}: {last}")


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
                 bucket: str | None = None):
    """The `pool_reference=` callable a validator passes to `KOTHValidator`.

    Reads the owner's reference commitment for the epoch, fetches the body, verifies it by hash and
    checks it describes THIS slice. Returns None on any failure so scoring degrades to the legacy
    scalar instead of the subnet stalling — a reference outage is an owner problem, not a miner one.
    """
    cache: dict = {}

    def read(epoch: int, nonce: str):
        if epoch in cache:
            return cache[epoch]
        out = None
        try:
            for c in chain.revealed_commitments():
                if owner_hotkey and c.hotkey != owner_hotkey:
                    continue
                d = parse_commit(c.data)
                if not d:
                    continue
                rec = fetch(d, bucket=bucket)
                if matches(rec, epoch=epoch, nonce=nonce, n_per_bench=n_per_bench):
                    out = (rec["scores"], rec["costs"])
                    break
        except Exception:                       # noqa: BLE001 — never break scoring on a fetch error
            out = None
        cache[epoch] = out
        return out

    return read
