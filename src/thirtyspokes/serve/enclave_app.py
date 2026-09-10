"""The HTTP surface exposed from inside the enclave (docs/CONFIDENTIAL_SERVING.md §7).

Two endpoints, and the smallness is the design:

    GET  /v1/enclave   the announcement + a TDX quote over its binding — what a client verifies
    POST /v1/sealed    a sealed request in, a sealed response out — opaque to everything between

There is deliberately no plaintext endpoint. A `/v1/chat/completions` that accepted clear JSON
would be far more convenient and would quietly void the entire guarantee, because the miner
operating the ingress could simply prefer it. If a plaintext path is ever wanted it belongs in a
different service with a different name, not as a fallback here.

WHAT THE MINER STILL SEES, and why the relay is not the interesting part. TDX protects guest
memory, so the miner cannot read this process. What the miner *does* control is the network: it can
see that a ciphertext of some bucket size arrived, time it, drop it, or replay it. Padding denies
the size signal (`sealed.BUCKETS`), the AEAD denies replay across images and epochs, and dropping
is a liveness attack that the validator's probes measure rather than something confidentiality
needs to prevent. So the ingress in front of this app requires no trust and no code of ours — any
TCP proxy will do, which is why this module ships no relay.

`/metrics` and `/v1/receipts` publish decisions, hashes and cost. They are public on purpose: the
receipt feed is how a validator checks a miner without seeing traffic, and how a user disputes a
request by revealing a preimage only they hold.
"""

# NB: deliberately no `from __future__ import annotations`. FastAPI resolves endpoint annotations
# at registration time via get_type_hints, and the fastapi types here are imported inside the
# factory to keep the dependency optional. Postponed evaluation turns `request: Request` into an
# unresolvable string, and FastAPI then reads it as a missing QUERY parameter — every POST to
# /v1/sealed answers 422 without the handler ever running.

import json
from collections import deque
from threading import Lock

from ..koth import sealed


def create_enclave_app(server, *, quote_provider=None, receipt_log: int = 512):
    """Wrap a `ConfidentialServer` in its HTTP surface.

    `quote_provider(report_data) -> bytes` is injected rather than called directly so the app can
    be exercised off TDX hardware; on real hardware it is `koth.tdx.get_quote`. Passing None means
    no quote is published, and a correctly-written client refuses such an enclave — which is the
    behaviour we want from a misconfigured deployment, not a silent downgrade.
    """
    from fastapi import FastAPI, Request, Response  # noqa: PLC0415 — optional dep
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    app = FastAPI(title="ThirtySpokes confidential router", version="1")
    receipts: deque = deque(maxlen=receipt_log)
    lock = Lock()
    stats = {"requests": 0, "refused": 0, "failed": 0}

    # The quote covers a key that never changes for the life of this process, so it is computed
    # once. Re-quoting per request would add ~40 ms and prove nothing new.
    quote_hex: str | None = None
    if quote_provider is not None:
        quote_hex = quote_provider(server.enclave_key.report_data()).hex()

    @app.get("/v1/enclave")
    def enclave() -> dict:
        return {"announcement": server.enclave_key.announcement(), "quote": quote_hex,
                "suites": sorted(sealed.SUITES), "buckets": list(sealed.BUCKETS)}

    @app.post("/v1/sealed")
    async def serve_sealed(request: Request):
        ciphertext = await request.body()
        try:
            result = server.handle(ciphertext)
        except PermissionError as exc:
            # the key gate refused — the miner's credential, not the user's request
            with lock:
                stats["refused"] += 1
            return JSONResponse({"error": "enclave refused to serve", "detail": str(exc)},
                                status_code=503)
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["failed"] += 1
            # The message is deliberately coarse. A decrypt failure and a malformed body must not
            # be distinguishable to a caller probing the enclave with crafted ciphertexts.
            return JSONResponse({"error": "request could not be served",
                                 "type": type(exc).__name__}, status_code=400)

        with lock:
            stats["requests"] += 1
            receipts.append(result.receipt)
        return Response(content=result.sealed_response,
                        media_type="application/octet-stream",
                        headers={"x-thirtyspokes-seq": str(result.receipt["seq"])})

    @app.get("/v1/receipts")
    def receipt_feed(since: int = 0) -> dict:
        with lock:
            return {"receipts": [r for r in receipts if r["seq"] > since]}

    @app.get("/metrics")
    def metrics() -> dict:
        with lock:
            served = list(receipts)
            counters = dict(stats)
        escalated = sum(1 for r in served if r.get("escalated"))
        return {**counters, "in_log": len(served),
                "escalated": escalated,
                "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in served), 6),
                "image_measurement": server.enclave_key.image_measurement,
                "epoch": server.enclave_key.epoch,
                "attested": quote_hex is not None}

    @app.get("/health")
    def health() -> dict:
        v = server.gate.check()
        return {"ok": bool(v.ok), "key": v.receipt_fields()}

    return app


def announcement_json(server, quote_provider=None) -> str:
    """The same payload as `GET /v1/enclave`, for publishing out of band (a file, an HF repo, a
    chain commitment). A client verifies it identically — the transport is not part of the trust."""
    quote = quote_provider(server.enclave_key.report_data()) if quote_provider else None
    return json.dumps({"announcement": server.enclave_key.announcement(),
                       "quote": quote.hex() if quote else None})
