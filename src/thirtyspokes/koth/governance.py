"""Owner governance: the approved-measurement record, committed on-chain by HASH.

THE CHAIN IS THE TRUST ROOT; THE BUCKET IS DUMB TRANSPORT.

The record (approved MRTD + RTMR1/2/3 + TCB policy + pinned pool) is what every validator gates every
proof on — it decides who earns. So its authority must live on-chain: signed by the owner key,
immutable, and identical for every validator. But the record is ~657 bytes and the chain's commitment
field only has `Raw<N>` variants up to 128 bytes ("Value 'Raw657' not present in type_mapping"), so it
cannot be stored on-chain directly.

So we commit its SHA-256 (73 bytes with the prefix — fits a plain, IMMEDIATELY-VISIBLE commitment) and
publish the bytes in the owner's public bucket, named by their own hash. A tampered record fails the
hash and is rejected, so the bucket needs no trust — the same asymmetry as the runtime image.

Why not the timelocked reveal-commitment (what this used to use)? It was chosen only because the
larger payload fit. It buys nothing: the payload is ENCRYPTED until reveal, so miners get no advance
warning of a change — only a 72-minute delay before it takes effect. That delay is a real cost: an
emergency TCB tightening (Intel discloses a vulnerability) could not take effect for 72 minutes. If you
want to give miners notice of a planned rotation, put an explicit `effective_from` block INSIDE the
record — visible immediately, active later. That is real notice; the timelock never gave any.
"""

from __future__ import annotations

import hashlib
import json

GOV_PREFIX = "kothgov1|"


def canonical(record: dict) -> bytes:
    """The exact bytes that get hashed and published — stable key order, no incidental whitespace."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def digest(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def commit_string(record_digest: str) -> str:
    """What goes on-chain: 9 + 64 = 73 bytes, so it fits a plain (immediate) commitment."""
    return f"{GOV_PREFIX}{record_digest}"


def parse_commit(data: str) -> str | None:
    """-> the record's sha256, or None if this isn't a governance commit."""
    if not isinstance(data, str) or not data.startswith(GOV_PREFIX):
        return None
    d = data[len(GOV_PREFIX):].strip()
    return d if len(d) == 64 and all(c in "0123456789abcdef" for c in d.lower()) else None


def record_path(record_digest: str) -> str:
    """Content-addressed: the record lives at a path derived from its own hash, so it cannot be
    silently swapped and old records stay retrievable for audit."""
    return f"governance/{record_digest}.json"


def verify(raw: bytes, expect_digest: str) -> dict:
    """Parse a fetched record and check it against the on-chain hash. Raises if it doesn't match."""
    got = hashlib.sha256(raw).hexdigest()
    if got != expect_digest:
        raise ValueError(f"governance record hash mismatch: on-chain {expect_digest}, got {got}")
    return json.loads(raw)


def publish(record: dict, *, bucket: str | None = None, token: str | None = None) -> str:
    """Upload the record to the owner's bucket, named by its hash. Returns the digest to commit."""
    import os
    import pathlib
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    from . import imagestore

    d = digest(record)
    staged = pathlib.Path(tempfile.mkdtemp(prefix="koth_gov_"))
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
    """Fetch the record by its on-chain hash and verify it. The URL is DERIVED from the hash, so a
    tampered record cannot even be addressed, let alone accepted."""
    import urllib.request

    from . import imagestore

    candidates = list(urls or [imagestore.public_url(record_path(record_digest), bucket=bucket)])
    last = None
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return verify(r.read(), record_digest)
        except Exception as e:  # noqa: BLE001 — try the next mirror; a bad record must not crash
            last = e
    raise RuntimeError(f"could not fetch governance record {record_digest}: {last}")
