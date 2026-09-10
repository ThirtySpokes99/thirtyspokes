"""Probe a live Conductor endpoint for the four properties v3 actually depends on.

Run against a vLLM/SGLang server hosting the pinned reference model. This is the bridge between
"a server is up" and "the server is one a validator may score against" — and the difference matters,
because every failure below is silent: the endpoint answers, the duel completes, and the number is
wrong.

    python scripts/serve_probe.py --base-url http://127.0.0.1:8000/v1

WHAT IT CHECKS, and why each one is load-bearing:

1. IDENTITY. The served model must be the pinned revision (§1.1). A validator scoring against
   different weights than it believes is a total and undetectable failure — every duel is decided by
   the wrong artifact and nothing in the output looks unusual.

2. DETERMINISM (§8b.4). Greedy decode must return byte-identical output for a repeated prompt.
   This model has TWO nondeterminism sources beyond ordinary sampling: MoE expert routing varies
   with batch composition, and the hybrid SSM layers accumulate in float32. §5.2a removes the case
   where this bites hardest — the king is measured once per slice rather than once per opponent —
   but a server that cannot repeat itself at all makes a re-run unable to reproduce a verdict.

3. THE CACHEABLE PREFIX. M0 exit 7 pins the prompt order as [catalog | task | history] precisely so
   the ~12k-token catalog is prefilled ONCE per window rather than on all ~6,000 Conductor calls.
   That only pays off if the server actually reuses the prefix, so this measures the second call's
   prefill against the first rather than assuming the flag worked.

4. THROUGHPUT AND CONCURRENCY. Sizes how much of a duel's wall clock the Conductor consumes. The
   expectation is "almost none" — a duel is dominated by waiting on worker API calls — and that
   expectation is load-bearing for the 2-hour window, so it is measured rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.v3 import config as C  # noqa: E402


def _post(base: str, path: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path, method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(base: str, path: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path,
                                 headers={"Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(base: str, model: str, prompt: str, max_tokens: int = 24) -> dict:
    t0 = time.time()
    out = _post(base, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,          # greedy — §8b.4
        "seed": 0,
        "stream": False,
    })
    out["_elapsed"] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="probe a live Conductor endpoint")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="v3-conductor")
    ap.add_argument("--prefix-tokens", type=int, default=4000,
                    help="stand-in for the catalog prefix, which is constant within a window")
    args = ap.parse_args()

    print("=" * 74)
    print("1. IDENTITY — is the served model the pinned artifact?")
    models = _get(args.base_url, "/models")
    served = [m["id"] for m in models.get("data", [])]
    print(f"   served ids           : {served}")
    print(f"   pinned reference     : {C.REFERENCE_MODEL}")
    print(f"   pinned revision      : {C.REFERENCE_REVISION}")
    root = next((m.get("root") for m in models.get("data", []) if m["id"] == args.model), None)
    print(f"   root of {args.model!r}: {root}")
    ok_identity = root is not None and C.REFERENCE_MODEL.split("/")[-1] in str(root)
    print(f"   -> {'OK' if ok_identity else 'MISMATCH — do not score against this endpoint'}")

    print("\n2. DETERMINISM — greedy decode must repeat byte-for-byte")
    p = "Reply with exactly the word ROUTE and nothing else."
    a = complete(args.base_url, args.model, p)["choices"][0]["message"]["content"]
    b = complete(args.base_url, args.model, p)["choices"][0]["message"]["content"]
    print(f"   call 1: {a!r}")
    print(f"   call 2: {b!r}")
    print(f"   -> {'IDENTICAL' if a == b else 'DIVERGED — a re-run cannot reproduce a verdict'}")

    print("\n3. CACHEABLE PREFIX — is the constant catalog prefill actually reused?")
    prefix = "CATALOG\n" + "\n".join(
        f"model-{i:04d}/v1  $0.10/$0.40  262144" for i in range(args.prefix_tokens // 12))
    first = complete(args.base_url, args.model, prefix + "\n\nTASK A\nReply OK.")
    second = complete(args.base_url, args.model, prefix + "\n\nTASK B\nReply OK.")
    u1, u2 = first.get("usage", {}), second.get("usage", {})
    print(f"   prompt tokens        : {u1.get('prompt_tokens')} then {u2.get('prompt_tokens')}")
    print(f"   cached tokens        : {u1.get('prompt_tokens_details')} / "
          f"{u2.get('prompt_tokens_details')}")
    print(f"   latency              : {first['_elapsed']:.2f}s then {second['_elapsed']:.2f}s")
    speedup = first["_elapsed"] / max(second["_elapsed"], 1e-6)
    print(f"   -> second call {speedup:.1f}x faster "
          f"({'prefix reuse working' if speedup > 1.3 else 'NO clear reuse — check --enable-prefix-caching'})")

    print("\n4. THROUGHPUT — how much of a duel's wall clock does the Conductor cost?")
    lat = []
    for i in range(5):
        r = complete(args.base_url, args.model, prefix + f"\n\nTASK {i}\nReply with one word.")
        lat.append(r["_elapsed"])
    med = statistics.median(lat)
    print(f"   per-call latency     : median {med:.2f}s  (range {min(lat):.2f}-{max(lat):.2f}s)")
    # A duel is ~500 episodes x ~12 steps over both arms (§5.4, config.MAX_STEPS).
    calls = 500 * C.MAX_STEPS
    print(f"   a duel is ~{calls:,} Conductor calls")
    for conc in (1, 16, 40):
        hrs = calls * med / conc / 3600
        print(f"     at {conc:>2}-way concurrency: {hrs:6.2f} h of the 2 h window"
              f"{'   <-- EXCEEDS THE WINDOW' if hrs > 2 else ''}")
    print("\n   The design assumes the Conductor is NOT the bottleneck — a duel is dominated by")
    print("   waiting on worker API calls. If any realistic concurrency exceeds the window, that")
    print("   assumption is wrong and §5.4's sizing needs redoing.")


if __name__ == "__main__":
    main()
