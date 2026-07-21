"""DCAP verification collateral — fetch + cache (H1).

Full TDX quote verification needs Intel PCS *collateral* keyed by the platform's FMSPC:
the signed TCB Info, the QE (enclave) Identity, and the CRLs. This module fetches it via
`dcap-qvl` (from Phala's PCCS or Intel PCS) and caches it in-process until the collateral's
own `nextUpdate`, so a validator scoring many proofs per epoch doesn't refetch per quote.

For deterministic / offline verification the owner can pin a collateral bundle (JSON from
`QuoteCollateralV3.to_json()`) and pass it straight to `verify_quote_full(collateral=...)`,
bypassing this module entirely — that is what the offline test fixture does.
"""

from __future__ import annotations

import asyncio
import inspect
import json


def _cache_key(raw_quote: bytes) -> str:
    """The collateral cache key = a hash of the quote's PCK leaf certificate.

    NOT the FMSPC: verified empirically on GCP TDX that the PCCS collateral is effectively
    PCK-cert-specific — two DIFFERENT physical CPUs with the *same* FMSPC each need their own
    collateral, and verifying one CPU's quote with another's collateral fails ("Signature is
    invalid for qe_report"). A validator scoring many miners (each a different CPU) must therefore
    key per-PCK, else all-but-the-first proof fails. The PCK leaf is per-CPU, so hashing it is the
    correct, safe key (still caches repeat verifications of the same miner across epochs)."""
    import hashlib
    try:
        from .tdx import parse_quote
        return hashlib.sha256(parse_quote(raw_quote).cert_pems[0]).hexdigest()
    except Exception:
        return hashlib.sha256(raw_quote[636:]).hexdigest()[:16]     # fallback: separate platforms


async def _afetch(raw_quote: bytes, pccs_url: str):
    import dcap_qvl
    r = dcap_qvl.get_collateral(pccs_url, raw_quote)
    return await r if inspect.isawaitable(r) else r


def _next_update_ts(col) -> int:
    """The earliest `nextUpdate` across the collateral's TCB Info + QE Identity (unix秒),
    so the cache expires exactly when Intel says the collateral goes stale. 0 if unknown."""
    import calendar
    import time as _t

    def parse(field: str) -> int:
        try:
            obj = json.loads(field)
            body = obj.get("tcbInfo") or obj.get("enclaveIdentity") or obj
            nu = body.get("nextUpdate")
            if not nu:
                return 0
            return calendar.timegm(_t.strptime(nu, "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            return 0

    cands = [t for t in (parse(col.tcb_info), parse(col.qe_identity)) if t > 0]
    return min(cands) if cands else 0


_CACHE: dict[str, tuple[int, object]] = {}   # PCK-leaf hash -> (expiry_unix, QuoteCollateralV3)


def get_collateral_cached(raw_quote: bytes, *, pccs_url: str, now: int):
    """Return DCAP collateral for `raw_quote`, fetching + caching per PCK cert until nextUpdate."""
    key = _cache_key(raw_quote)
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    col = asyncio.run(_afetch(raw_quote, pccs_url))
    expiry = _next_update_ts(col) or (now + 3600)     # honor nextUpdate; else a safe 1h TTL
    _CACHE[key] = (expiry, col)
    return col
