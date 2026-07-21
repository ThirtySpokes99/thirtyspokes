"""Production fail-closed enforcement, validated end-to-end on REAL Intel-TDX hardware.

Run INSIDE a TDX guest (e.g. a GCP C3). This is the hardware proof of the `enforce=True` gate
(docs/DESIGN.md §8): a proof is scored ONLY if it carries a genuine hardware quote whose measurements
match the owner-pinned measured image, under a TCB policy. Everything else fails CLOSED.

Sequence:
  1. read the hardware measurements (sysfs) and cross-check them against a signed TDX quote;
  2. bind the runtime into RTMR3 (`measure_self`, one-way per boot) and confirm it equals the
     owner-derivable expected value;
  3. produce a real attested KOTH proof and ACCEPT it with the full pinned gate
     (MRTD + RTMR1/2/3 + full DCAP TCB status);
  4. prove every fail-closed path rejects: wrong image (RTMR2), mock quote, and each unset gate.

NOTE: step 2 extends RTMR3, which is one-way per boot. Re-running after a reboot is fine;
re-running in the same boot is idempotent (`ensure_runtime_measured`).
"""

from __future__ import annotations

import dataclasses

from thirtyspokes.koth import rtmr as rtmrmod
from thirtyspokes.koth import tdx
from thirtyspokes.koth.benchmarks import default_suite
from thirtyspokes.koth.miner import reference_artifact
from thirtyspokes.koth.pool import MockPool, PinnedBackend
from thirtyspokes.koth.runtime import KOTHRuntime, runtime_measurement
from thirtyspokes.koth.verify import verify_proof

EPOCH, NONCE, HOTKEY = 11, "beacon-enforce", "miner-tdx"


def _verify(proof, art, **over):
    kw = dict(approved_measurements={runtime_measurement()}, platform_public_hex="tdx-hardware-root",
              expect_epoch=EPOCH, expect_nonce=NONCE, expect_hotkey=HOTKEY,
              expect_source_hash=art.source_hash, expect_weights_hash=art.weights_hash,
              suite=default_suite(), n_per_bench=3)
    kw.update(over)
    return verify_proof(proof, **kw)


def main() -> None:
    if not tdx.tdx_available():
        raise SystemExit("not a TDX guest — run inside the confidential VM")
    print("== KOTH production enforcement on real Intel TDX ==\n")

    # 1. hardware measurements from sysfs
    hw = rtmrmod.read_measurements()
    print("[1] sysfs measurements")
    for k in ("mrtd", "rtmr1", "rtmr2", "rtmr3"):
        print(f"    {k:6s} {hw.get(k, '')[:32]}…")

    pool = MockPool()
    allow = pool.allowed
    backend = PinnedBackend(pool, allow)
    runtime = KOTHRuntime(backend, tdx.TDXPlatform())     # REAL hardware attestation root

    # 2. bind the runtime into RTMR3 (one-way; idempotent within a boot)
    rtmr3 = runtime.measure_self(allow)
    expected = rtmrmod.owner_expected_rtmr3(
        runtime_measurement=runtime_measurement(),
        suite_version=__import__("thirtyspokes.koth.runtime", fromlist=["x"]).SUITE_VERSION,
        pool_allow_list=allow)
    print(f"\n[2] RTMR3 runtime self-measure")
    print(f"    extended  {rtmr3[:32]}…")
    print(f"    owner-exp {expected[:32]}…   {'MATCH' if rtmr3 == expected else 'DIFFER'}")
    assert rtmr3 == expected, "RTMR3 != owner-derivable expected value"

    # 3. produce a real attested proof; cross-check quote vs sysfs.
    #    Re-read sysfs AFTER the RTMR3 extend so both sides describe the same TD state.
    hw = rtmrmod.read_measurements()
    art = reference_artifact("strong")
    proof, _trace = runtime.run(art, hotkey=HOTKEY, epoch=EPOCH, nonce=NONCE,
                                suite=default_suite(), n_per_bench=3)
    raw = tdx.parse_quote(__import__("base64").b64decode(proof.quote.platform_sig[4:]))
    q_mrtd, q_r = raw.mr_td.hex(), [r.hex() for r in raw.rtmrs]
    print(f"\n[3] quote vs sysfs cross-check "
          f"({proof.n_calls} pool calls, cost=${proof.total_cost_usd:.5f})")
    pairs = [("mrtd", q_mrtd, hw["mrtd"])] + [
        (f"rtmr{i}", q_r[i], hw.get(f"rtmr{i}", "")) for i in range(4)]
    same = True
    for name, q, s in pairs:
        m = q == s
        same &= m
        print(f"    {name:6s} quote={q[:24]}…  sysfs={s[:24]}…  {'MATCH' if m else 'DIFFER'}")
    assert same, "signed quote measurements disagree with sysfs"

    # the owner's pinned set (in production: published on-chain via orchestra-koth-owner)
    APPROVED_MRTD = {hw["mrtd"]}
    APPROVED_RTMR = {1: hw["rtmr1"], 2: hw["rtmr2"], 3: rtmr3}
    TCB = tdx.DEFAULT_TCB_ACCEPT
    gate = dict(enforce=True, approved_mrtd=APPROVED_MRTD, approved_rtmr=APPROVED_RTMR,
                tcb_accept=TCB)

    # 4. ACCEPT under the full pinned gate (full DCAP: chain + TCB + CRL + QE identity)
    vd = _verify(proof, art, **gate)
    print(f"\n[4] enforce=True, fully pinned  -> valid={vd.valid} reason={vd.reason}"
          f"  score={vd.score:.3f}")
    assert vd.valid, f"honest miner rejected under enforce: {vd.reason}"

    # 5. fail-closed negatives
    print("\n[5] fail-closed paths (each MUST reject):")
    checks = [
        ("wrong image (RTMR2 mismatch)", dict(gate, approved_rtmr={**APPROVED_RTMR, 2: "de" * 48}),
         "unapproved_runtime"),
        ("MRTD not approved", dict(gate, approved_mrtd={"ab" * 48}), "unapproved_runtime"),
        ("MRTD gate unset", dict(gate, approved_mrtd=None), "mrtd_gate_unset"),
        ("RTMR gate unset", dict(gate, approved_rtmr=None), "rtmr_gate_unset"),
        ("RTMR gate incomplete", dict(gate, approved_rtmr={1: hw["rtmr1"]}), "rtmr_gate_unset"),
        ("TCB policy unset", dict(gate, tcb_accept=None), "tcb_policy_unset"),
    ]
    ok = True
    for name, kw, want in checks:
        v = _verify(proof, art, **kw)
        good = (not v.valid) and v.reason == want
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}  {name:30s} -> {v.reason}")

    # mock (non-hardware) quote must be rejected outright under enforce
    mock = dataclasses.replace(proof, quote=dataclasses.replace(proof.quote, platform_sig="ed25519:x"))
    v = _verify(mock, art, **gate)
    good = (not v.valid) and v.reason == "mock_quote_rejected"
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  {'mock TEE quote':30s} -> {v.reason}")

    # A tampered payload must break the hardware binding even though the quote itself is genuine:
    # the quote stays internally valid, but the mutated proof no longer hashes to the REPORTDATA
    # the hardware signed -> report_data_mismatch (the binding, not the signature, catches it).
    tampered = dataclasses.replace(proof, total_cost_usd=0.0)
    v = _verify(tampered, art, **gate)
    good = (not v.valid) and v.reason == "report_data_mismatch"
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  {'tampered proof (cost=0)':30s} -> {v.reason}")

    print("\n== RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", "==")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
