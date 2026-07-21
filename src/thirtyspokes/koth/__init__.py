"""KOTH-TEE subnet — miners run the owner-given benchmarks inside their own TEE
and validators only verify an attested proof (docs/DESIGN.md).

Composes the existing primitives instead of rebuilding them:
  * hardware-root attestation  — `tee.attestation` (Quote / Platform / verify_quote)
  * in-enclave metering        — `tee.runtime.MeteringProxy`
  * king-of-the-hill emissions — `reign.KingChain` (king + equal-share ex-king chain, eps hysteresis)
  * chain seam                 — `subnet.chain` (commit-reveal + set_weights + burn)

What KOTH adds on top: the artifact BINDING (report_data commits to source+weights,
so the score is provably tied to the public artifact), a MULTI-BENCHMARK proof, a
Pareto dethrone guard, and the optimistic anti-cheat backstops (fresh-probe audit
for weight-memorization, fraud-proof for literal hardcoding).
"""
