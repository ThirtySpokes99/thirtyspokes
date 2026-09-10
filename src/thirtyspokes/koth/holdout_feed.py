"""Per-epoch held-out files: how validators sync the slice that scores a miner.

THE DESIGN (owner's, 2026-08-12). The dataset does not need to be private. The owner prepares one
file PER EPOCH ahead of time and publishes each one only when its epoch opens. Before that moment
the file simply does not exist at its URL; afterwards it is public forever, so every score stays
re-derivable by anyone. Nothing is secret — things are merely not published yet.

WHY THE URL IS NOT COMMITTED ON-CHAIN. It cannot be. The chain gives each hotkey exactly ONE
commitment slot (`CommitmentOf[(netuid, hotkey)]`) and `set_commitment` overwrites it; the owner's
slot already holds the governance record (`kothgov1|…`, the approved MRTD/RTMR measurements), so a
per-epoch write would erase governance and every validator would read `mrtd_gate_unset` and refuse
to score. `reference.py` hit this exact wall. The answer there and here is the same: **derive the
path, don't publish it.** `epoch_path(N)` is a pure function of the epoch, so a validator addresses
the file without being told anything, and authenticity comes from the OWNER'S SIGNATURE checked
against `SubnetOwnerHotkey` on-chain — not from where the bytes were found.

WHY THIS NEEDS NO NONCE, UNLIKE THE PUBLIC BANK. `benchmarks.bench_seed` draws a slice from a public
pool by chain nonce, because a slice everyone can see must at least be unpredictable. Here the file
is not public until it is scored, so unpredictability buys nothing: the owner fixes the contents in
advance. That removes the draw, the burn bookkeeping, and the secret-bank distribution in
`koth/heldout.py` (`probe_commit`, `probe_bank_unverified`, the fail-closed hand-off to validators)
all at once — the file IS the epoch's slice.

WHAT STILL NEEDS PINNING. An owner who prepares files epoch-by-epoch could pick an easy slice to
flatter a favoured miner, or swap a file after seeing who committed. `manifest()` closes that with
ONE commitment for the whole schedule: the digests of every prepared epoch, chained. The owner
publishes the chain root once (it fits in the governance record), and thereafter each epoch file is
checked against it. No per-epoch extrinsic, and no discretion left at scoring time.

OWNER LIVENESS IS A HARD DEPENDENCY, BY DECISION. No file, no score. This is accepted: a dead owner
is a dead subnet either way, since governance and the pool reference already depend on it. The
validator-side rule is `fail closed` — skip the epoch, leave weights untouched — rather than score
against a stale or substituted file.
"""

from __future__ import annotations

import json

import numpy as np

from ..gateway import signing
# `OutcomeMatrix` is imported inside the three arena builders that need it, not here: v3
# (`v3/window.py`, `v3/validator.py`, `v3/owner.py`, `v3/simulate.py`) uses only
# `epoch_path`, `manifest`, `in_manifest` and `publish`, and a module-level import dragged the
# retired matmul path — matrix, duel, fugu, verify — back into v3's closure (M10, 2026-09-07).
from .reference import canonical, digest, envelope, open_envelope


def epoch_path(epoch: int) -> str:
    """Where epoch N's held-out file lives. A pure function of the epoch: no link to distribute,
    nothing to look up, and the file is simply absent until the owner publishes it."""
    return f"holdout/{int(epoch)}.json"


def manifest(records: list[dict]) -> dict:
    """One commitment covering the WHOLE schedule, so the owner keeps no discretion at scoring time.

    Without this, an owner preparing files epoch-by-epoch could choose an easy slice to flatter a
    favoured miner, or swap a file after seeing who committed. The chain here is order-dependent, so
    inserting, reordering or editing any epoch changes the root.
    """
    chain = ""
    digests = {}
    for record in sorted(records, key=lambda r: r["epoch"]):
        d = digest(record)
        digests[str(record["epoch"])] = d
        chain = signing.sha256_hex(f"{chain}|{record['epoch']}|{d}")
    return {"v": 1, "root": chain, "digests": digests,
            "epochs": [int(r["epoch"]) for r in sorted(records, key=lambda r: r["epoch"])]}


def in_manifest(record: dict, man: dict) -> bool:
    """Was this exact file the one pinned for this epoch before the schedule started?"""
    return man.get("digests", {}).get(str(record.get("epoch"))) == digest(record)


def publish(record: dict, sign, *, bucket: str | None = None, token: str | None = None) -> str:
    """Owner-side: sign and upload epoch N's file. Call this WHEN N OPENS, never earlier — a file
    published before miners commit is a file miners can train on, which is the whole property.

    Staged-directory sync rather than a byte upload, matching `reference.publish`, so both owner
    feeds go through the same path and inherit the same auth and retry behaviour.
    """
    import os
    import pathlib
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    from . import imagestore

    path = epoch_path(record["epoch"])
    staged = pathlib.Path(tempfile.mkdtemp(prefix="koth_holdout_"))
    try:
        target = staged / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(envelope(record, sign)))
        api = HfApi(token=token or os.environ.get("OWNER_HF_TOKEN") or os.environ.get("HF_TOKEN"))
        api.sync_bucket(str(staged), imagestore.bucket_uri(bucket), delete=False)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return path


def canonical_bytes(record: dict) -> bytes:
    """Exposed so an auditor can recompute a digest exactly the way the manifest did."""
    return canonical(record)
