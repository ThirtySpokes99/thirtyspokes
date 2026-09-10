"""OpenRouter key provenance — the gate that makes the privacy claim enforceable.

docs/CONFIDENTIAL_SERVING.md §4.3a. The design rests on a measured negative: `GET /api/v1/key`
returns usage, limits and `creator_user_id`, but **nothing about data policy or prompt logging**.
So an enclave holding a miner-owned key cannot tell whether that account is logging every request,
and no attestation fixes it — the fact is not exposed. If the miner owns the account, privacy from
the miner is unenforceable.

The design therefore inverts key ownership. The OWNER provisions per-miner keys under an account
whose data policy it sets once (ZDR on, prompt logging off) and the miner cannot change. The miner
funds a spend limit but never owns the key. The enclave then verifies, before serving and
periodically after, that the key it was handed really was created by the owner's account:

    creator_user_id     == the owner account baked into the measured image
    is_provisioning_key == False        (an inference key, not a management key)
    limit               is not None     (an unlimited key is a blank cheque on the owner)
    limit_remaining     > 0             (the miner's funded budget is not exhausted)

The first check is the load-bearing one. It is what refuses a miner that swaps in its own key —
the attack that would otherwise void the entire privacy guarantee while leaving every other part of
the system working perfectly.

TWO THINGS THAT MAKE THIS MORE THAN A FORMALITY.

**The miner controls the enclave's network, so this call must be un-spoofable.** Introspection goes
over the miner's host. A miner that MITMs it could return a forged response naming the owner's
account and walk straight through the gate. The defence is the one `gateway/gateway.py` already
documents for the pool client: `trust_env=False` plus the pinned `certifi` roots, so the miner owns
the network but cannot forge a certificate for `openrouter.ai`. Passing an unhardened `fetch` into
this module reopens exactly that hole, which is why the default builds its own client rather than
accepting an ambient one.

**It fails closed.** A miner that blocks introspection does not thereby escape the check — an
unreachable API is a refusal to serve, not a pass. That is the only safe direction: the alternative
lets any miner disable its own supervision by dropping one host.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

BASE_URL = "https://openrouter.ai/api/v1"
RECHECK_SECONDS = 900.0


def _hardened_get(url: str, api_key: str, timeout: float = 15.0) -> dict:
    """GET with the miner explicitly out of the trust path.

    `trust_env=False` ignores `SSL_CERT_FILE` / `HTTPS_PROXY` / `REQUESTS_CA_BUNDLE` from the
    environment, which is precisely how `eval/config.py` documents a miner MITMing the enclave's
    provider calls. The CA bundle is the one shipped inside the measured image.
    """
    import ssl  # noqa: PLC0415

    import certifi  # noqa: PLC0415
    import httpx  # noqa: PLC0415

    ctx = ssl.create_default_context(cafile=certifi.where())
    with httpx.Client(trust_env=False, timeout=timeout, verify=ctx) as c:
        r = c.get(url, headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        return r.json()


def introspect(api_key: str, *, fetch=None) -> dict:
    """`GET /api/v1/key` -> the `data` object. `fetch` is for tests ONLY (see module docstring)."""
    fn = fetch or (lambda url: _hardened_get(url, api_key))
    payload = fn(f"{BASE_URL}/key")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError("unexpected /key response shape")
    return data


@dataclass(frozen=True)
class KeyVerdict:
    ok: bool
    reason: str
    creator_user_id: str = ""
    limit: float | None = None
    limit_remaining: float | None = None
    usage: float = 0.0
    checked_at: float = 0.0

    def receipt_fields(self) -> dict:
        """What goes into the public serving receipt. Deliberately excludes the key and its label:
        a receipt is published, and a key label is a partial credential."""
        return {"key_ok": self.ok, "key_reason": self.reason,
                "key_creator": self.creator_user_id,
                "key_limit_remaining": self.limit_remaining,
                "key_checked_at": round(self.checked_at, 3)}


def verify_key(info: dict, *, owner_account_id: str, min_remaining: float = 0.0) -> KeyVerdict:
    """Is this key one the owner issued, still funded, and not a management key?"""
    now = time.time()
    creator = str(info.get("creator_user_id") or "")
    limit = info.get("limit")
    remaining = info.get("limit_remaining")
    base = {"creator_user_id": creator, "limit": limit, "limit_remaining": remaining,
            "usage": float(info.get("usage") or 0.0), "checked_at": now}

    def no(reason: str) -> KeyVerdict:
        return KeyVerdict(ok=False, reason=reason, **base)

    if not owner_account_id:
        # Refusing here rather than defaulting is deliberate: an empty expected account would make
        # the comparison below vacuously true and silently disable the whole gate.
        return no("no owner account configured — refusing to serve rather than skip the check")
    if creator != owner_account_id:
        return no(f"key was created by {creator or '<unknown>'}, not the owner account")
    if info.get("is_provisioning_key") or info.get("is_management_key"):
        return no("management/provisioning key supplied where an inference key is required")
    if limit is None:
        return no("key has no spend limit — an unlimited key is a blank cheque on the owner")
    if remaining is None or float(remaining) <= min_remaining:
        return no(f"funded budget exhausted (remaining={remaining})")
    return KeyVerdict(ok=True, reason="ok", **base)


@dataclass
class KeyGate:
    """Enclave-side gate: verify on startup, re-verify periodically, fail closed.

    Re-verification matters because provenance is not a one-time property — the owner can revoke a
    key, and a miner's funded budget runs out mid-operation. A gate that only checked at boot would
    let a revoked key serve until the next restart.
    """

    api_key: str
    owner_account_id: str
    recheck_seconds: float = RECHECK_SECONDS
    fetch: object = None
    _verdict: KeyVerdict | None = field(default=None, repr=False)
    _last: float = field(default=0.0, repr=False)

    def check(self, *, force: bool = False, now=None) -> KeyVerdict:
        t = now() if now else time.time()
        if not force and self._verdict is not None and (t - self._last) < self.recheck_seconds:
            return self._verdict
        try:
            info = introspect(self.api_key, fetch=self.fetch)
            verdict = verify_key(info, owner_account_id=self.owner_account_id)
        except Exception as exc:  # noqa: BLE001 — unreachable API is a refusal, never a pass
            verdict = KeyVerdict(ok=False, reason=f"introspection failed: {type(exc).__name__}",
                                 checked_at=t)
        self._verdict, self._last = verdict, t
        return verdict

    def require(self, *, now=None) -> KeyVerdict:
        """Raise unless the key is currently good. Call this on the serving path."""
        v = self.check(now=now)
        if not v.ok:
            raise PermissionError(f"refusing to serve: {v.reason}")
        return v


# --- owner side: provisioning ---------------------------------------------------------------

def provision(management_key: str, *, label: str, limit_usd: float,
              limit_reset: str | None = None, post=None) -> dict:
    """Create a per-miner inference key under the OWNER's account.

    `limit_usd` is required by this signature rather than optional, because the gate refuses an
    unlimited key and a helper that made it easy to mint one would be a foot-gun aimed at the
    owner's credit balance. `limit_reset` is one of daily/weekly/monthly, or None for a
    non-renewing budget.
    """
    if limit_usd <= 0:
        raise ValueError("limit_usd must be positive — the gate refuses unlimited keys")
    body: dict = {"name": label, "limit": float(limit_usd)}
    if limit_reset:
        body["limit_reset"] = limit_reset
    if post is not None:
        return post(f"{BASE_URL}/keys", body)

    import ssl  # noqa: PLC0415

    import certifi  # noqa: PLC0415
    import httpx  # noqa: PLC0415
    ctx = ssl.create_default_context(cafile=certifi.where())
    with httpx.Client(trust_env=False, timeout=30.0, verify=ctx) as c:
        r = c.post(f"{BASE_URL}/keys",
                   headers={"Authorization": f"Bearer {management_key}",
                            "Content-Type": "application/json"},
                   content=json.dumps(body))
        r.raise_for_status()
        return r.json()
