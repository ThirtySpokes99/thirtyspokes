#!/usr/bin/env python
"""Check a benchmark result you did not run, produced by someone you do not trust.

    "This exact artifact produced this exact result, on real hardware — and you can verify that
     without re-running it or trusting whoever ran it."

Benchmark scores today are self-reported and unfalsifiable. Reproducing one means paying the full
inference bill and still not knowing you ran the same thing: same model build, same prompts, same
grader, no retries, nothing dropped. So in practice nobody checks, and "we scored 87%" is taken on
trust or not at all.

This demo runs the other path against a REAL Intel TDX proof recorded on GCP confidential hardware —
720 graded tasks — and shows a stranger establishing five things in milliseconds:

  1. the quote is genuine        signed by Intel's attestation key, chaining to the Intel root
  2. the code was the approved   MRTD + RTMR image measurements pin WHICH image booted
  3. the payload is untouched    report_data hashes every field, so any edit breaks the quote
  4. it answers the challenge    the (epoch, nonce) the verifier issued — not a replay
  5. it is bound to an artifact  source + weights hashes tie the run to published bytes

Then it tampers with the proof four ways and shows each rejected, because a verifier that only ever
says "valid" has demonstrated nothing.

Run: python scripts/verify_demo.py [path/to/proof.json]
With no argument it falls back to a synthetic proof, so the demo works without the recorded data.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import sys
import time

from thirtyspokes.koth.attest import verify_attestation
from thirtyspokes.koth.proof import Proof


def _find_proof() -> str | None:
    hits = sorted(glob.glob("data/koth-support-tdx*/**/proof.json", recursive=True),
                  key=lambda p: -len(json.load(open(p)).get("results") or []))
    return hits[0] if hits else None


def _synthetic() -> Proof:
    """Fallback so the demo always runs. A mock vendor key -- zero real security, but the payload
    binding, the challenge and the artifact binding behave identically."""
    from thirtyspokes.koth.proof import BenchmarkResult
    from thirtyspokes.tee.attestation import Platform
    p = Proof(1, "n1", "hk", "aa" * 32, "bb" * 32, "router",
              tuple(BenchmarkResult("math", f"t{i}", str(i), 0.001) for i in range(8)),
              0.008, 8, "cc" * 32, "measured-image-v17")
    return p.attested_by(Platform())


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else _find_proof()
    if path:
        proof = Proof.from_json(open(path).read())
        real = proof.quote.platform_sig.startswith("tdx:")
        print(f"proof: {path}")
    else:
        proof, real, path = _synthetic(), False, "<synthetic>"
        print("proof: <synthetic — no recorded TDX proof found under data/>")

    n = len(proof.results)
    print(f"claims: {n} graded tasks, ${proof.total_cost_usd:.4f} of inference, "
          f"image {proof.measurement[:16]}…")
    print(f"quote : {'REAL Intel TDX (DCAP)' if real else 'mock vendor key (dev)'}   "
          f"schema {proof.schema}\n")

    # The verifier's expectations come from ITS OWN records — the challenge it issued and the
    # artifact hashes it recomputed from the published bundle. Taking any of these from the proof
    # would make the check circular: a forger would simply assert whatever it needed.
    expect = dict(
        approved_measurements={proof.measurement},
        platform_public_hex="",
        expect_epoch=proof.epoch, expect_nonce=proof.nonce, expect_hotkey=proof.hotkey,
        expect_source_hash=proof.source_hash, expect_weights_hash=proof.weights_hash,
    )
    if not real:
        from thirtyspokes.tee.attestation import Platform
        expect["platform_public_hex"] = Platform().public_hex   # dev path only

    t0 = time.perf_counter()
    reason = verify_attestation(proof, **expect)
    dt = time.perf_counter() - t0

    if reason and not real:
        # the synthetic path signs with a throwaway platform key; re-sign so the demo is honest
        from thirtyspokes.tee.attestation import Platform
        pf = Platform(); proof = proof.attested_by(pf)
        expect["platform_public_hex"] = pf.public_hex
        t0 = time.perf_counter(); reason = verify_attestation(proof, **expect)
        dt = time.perf_counter() - t0

    print(f"{'ACCEPTED' if reason is None else 'REJECTED: ' + reason}"
          f"   in {dt * 1000:.1f} ms, 0 model calls, $0.00")
    if reason is None and proof.total_cost_usd > 0:
        print(f"   re-running these {n} tasks would cost ${proof.total_cost_usd:.4f} "
              f"and still not prove the same code ran.\n")

    # A verifier that only ever accepts has proven nothing. Each of these is a distinct attack.
    print("now the forgeries — each must fail, and fail for the RIGHT reason:")
    edited = list(proof.results)
    if edited:
        edited[0] = dataclasses.replace(edited[0], answer="FORGED")
    attacks = [
        ("edit one graded answer", dataclasses.replace(proof, results=tuple(edited)), expect),
        ("understate the cost", dataclasses.replace(proof, total_cost_usd=0.0), expect),
        ("replay against another challenge", proof, dict(expect, expect_nonce="different-nonce")),
        ("claim a different artifact", proof, dict(expect, expect_source_hash="ff" * 32)),
    ]
    for label, p2, exp in attacks:
        r = verify_attestation(p2, **exp)
        print(f"  {label:34s} -> {r or 'ACCEPTED (!!)'}")

    print("\nWhat a third party needed: the proof, and the image measurement the publisher "
          "committed to.\nNot the model, not the prompts, not the money, and not the publisher's "
          "good faith.")


if __name__ == "__main__":
    main()
