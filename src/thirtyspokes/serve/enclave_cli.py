"""`thirtyspokes-enclave` — bring up the confidential router inside a TDX guest.

Boot order matters here, because each step's failure has to be a refusal to serve rather than a
degraded service that still answers:

    1. read the image measurement FROM THE HARDWARE      MRTD + boot RTMRs, never from a flag
    2. verify the OpenRouter key is the owner's          before any key material exists
    3. mint the ephemeral enclave key + quote it         binding covers 1
    4. serve                                             only if 1-3 held

Step 2 comes before step 3 deliberately. A miner supplying its own provider key can read every
request at the provider, which the enclave cannot detect later — so the process must not reach the
point of publishing a key a user might seal to.

WHAT `--insecure-no-tdx` IS FOR. Developing the routing model needs a runnable service on ordinary
hardware. The flag is refused whenever TDX is actually present, prints a banner, and publishes no
quote — so a correctly-written client refuses the resulting endpoint rather than treating it as
private (`serve/client.py` returns "no quote: this is not an enclave"). A dev mode that produced a
plausible-looking but unattested endpoint would be the worst outcome, because nothing downstream
would object.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_LADDER = ("openai/gpt-oss-120b", "openai/gpt-5.4-mini", "openai/gpt-5.4")


def build(args):
    from ..eval.config import LiveConfig  # noqa: PLC0415 — optional deps
    from ..gateway.gateway import OpenRouterBackend  # noqa: PLC0415
    from ..koth import harness, sealed, tdx  # noqa: PLC0415
    from ..koth.orkey import KeyGate  # noqa: PLC0415
    from .confidential import ConfidentialServer  # noqa: PLC0415
    from .enclave_app import create_enclave_app  # noqa: PLC0415

    # 1. the image measurement, from the silicon
    on_tdx = tdx.tdx_available()
    if on_tdx and args.insecure_no_tdx:
        raise SystemExit("--insecure-no-tdx on a TDX host: refusing to discard real attestation.")
    if on_tdx:
        image = tdx.self_measurement()
        quote_provider = tdx.get_quote
    elif args.insecure_no_tdx:
        print("!" * 78, "\n!! NO TDX. This endpoint is NOT confidential and publishes no quote.",
              "\n!! Requests are readable by whoever runs this process.\n" + "!" * 78,
              file=sys.stderr)
        image = "insecure-no-tdx"
        quote_provider = None
    else:
        raise SystemExit("no TDX available. Pass --insecure-no-tdx to run an UNATTESTED "
                         "development endpoint that clients will refuse.")
    # 2. the provider key must be the owner's, checked before any enclave key exists
    owner = args.owner_account or os.environ.get("THIRTYSPOKES_OWNER_ACCOUNT", "")
    cfg = LiveConfig()
    cfg.require_key()
    gate = KeyGate(cfg.api_key, owner, recheck_seconds=args.key_recheck)
    verdict = gate.check()
    if not verdict.ok:
        raise SystemExit(f"provider key refused: {verdict.reason}")
    print(f"provider key ok (creator={verdict.creator_user_id}, "
          f"${verdict.limit_remaining:.2f} remaining)", file=sys.stderr)

    # 3. the ephemeral key, bound to the measured image
    key = sealed.EnclaveKey.generate(image, args.epoch, suite=args.suite)
    print(f"enclave key {key.public_bytes.hex()[:16]}... image={image[:24]}... "
          f"epoch={args.epoch} suite={args.suite}", file=sys.stderr)

    ladder = tuple(args.ladder or DEFAULT_LADDER)
    policy = _load_policy(args.weights, len(ladder))
    backend = OpenRouterBackend(cfg.api_key, base_url=cfg.base_url, timeout=args.timeout)
    server = ConfidentialServer(
        enclave_key=key, gate=gate, policy=policy, backend=backend,
        ladder=ladder,
        encode=harness.encode, max_tokens=args.max_tokens, request_timeout=args.timeout)
    return create_enclave_app(server, quote_provider=quote_provider)


def _load_policy(weights: str | None, k: int, d: int = 384):
    """The miner's routing model, or the cheapest-rung control.

    A missing model is not an error: always entering at the cheapest rung is the baseline every
    measurement in docs/ROUTING_MEASUREMENTS.md (removed with v2, 2026-09-07) is scored against, and on that evidence it is a
    perfectly respectable thing to serve. It is also the fallback when a supplied model misbehaves.

    A KOTH head is a PICK-ONE router: it reads the encoder features and names a rung, ignoring the
    cascade state that `_route` also passes. Feeding it `state[:d]` is therefore the whole
    adaptation, not a shortcut. The consequence is deliberate — such a head names the same rung on
    the second pass, `_route` sees `tried[action]` already set and returns STOP, and the request
    ends after one call. A pick-one router does not cascade, and the enclave should not invent
    escalation on its behalf.
    """
    if not weights:
        return lambda state: 0
    from pathlib import Path  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from ..koth.harness import load_head  # noqa: PLC0415

    head, theta = load_head(Path(weights).read_bytes(), k=k, d=d)

    def policy(state) -> int:
        features = np.asarray(state, dtype=float)[:d].reshape(1, -1)
        return int(head.distribution(theta, features).argmax(axis=1)[0])

    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description="ThirtySpokes confidential router (in-enclave)")
    ap.add_argument("--host", default="0.0.0.0")   # noqa: S104 — the ingress is outside the guest
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--epoch", type=int, default=1,
                    help="rotation counter; bound into the key binding and the AEAD")
    ap.add_argument("--suite", default="x25519", choices=("x25519", "pq"),
                    help="pq = ML-KEM768, for traffic that must resist later decryption")
    ap.add_argument("--owner-account", default=None,
                    help="OpenRouter creator_user_id the provider key must belong to "
                         "(or THIRTYSPOKES_OWNER_ACCOUNT)")
    ap.add_argument("--weights", default=None, help="miner routing model (.npz)")
    ap.add_argument("--ladder", nargs="+", default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--key-recheck", type=float, default=900.0)
    ap.add_argument("--insecure-no-tdx", action="store_true",
                    help="run WITHOUT attestation for development; clients will refuse it")
    args = ap.parse_args()

    import uvicorn  # noqa: PLC0415
    uvicorn.run(build(args), host=args.host, port=args.port)
