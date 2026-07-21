"""WS7 live smoke — a REAL Intel-TDX hardware-attested KOTH proof, on a confidential VM.

Run this INSIDE a TDX guest (e.g. GCP C3). It:
  1. runs the actual KOTH runtime (reference router over the owner suite, pinned MockPool)
     but attests with the real `TDXPlatform` instead of the mock vendor key;
  2. the runtime asks the TDX hardware for a genuine quote whose REPORTDATA = the proof's
     report_data() — so the quote certifies *this exact proof* ran;
  3. verifies the quote's full DCAP signature chain up to the pinned Intel SGX Root CA,
     plus the report_data binding and the MRTD gate;
  4. shows the binding is load-bearing: tampering any proof field makes the hardware quote
     reject it (report_data_mismatch), and a wrong approved-MRTD set is refused.

No shared secret anywhere — this is the Stage-2 trust claim made real.
"""

from __future__ import annotations

import dataclasses

from thirtyspokes.koth import tdx
from thirtyspokes.koth.benchmarks import default_suite
from thirtyspokes.koth.miner import reference_artifact
from thirtyspokes.koth.pool import MockPool, PinnedBackend
from thirtyspokes.koth.runtime import KOTHRuntime


def main() -> None:
    print("== WS7: real Intel TDX attestation ==")
    print("tdx_available():", tdx.tdx_available())
    if not tdx.tdx_available():
        raise SystemExit("not on a TDX guest — run this inside the confidential VM")

    pool = MockPool()
    backend = PinnedBackend(pool, pool.allowed)          # owner-pinned pool
    runtime = KOTHRuntime(backend, tdx.TDXPlatform())    # <-- REAL hardware attestation root
    suite = default_suite()

    proof, trace = runtime.run(reference_artifact("strong"), hotkey="miner-tdx",
                               epoch=7, nonce="beacon-abc", suite=suite, n_per_bench=3)
    print(f"\nproduced proof: {len(proof.results)} results, {proof.n_calls} pool calls, "
          f"cost=${proof.total_cost_usd:.5f}")
    print("proof.report_data():", proof.report_data())

    # --- verify the real quote (DCAP chain -> Intel root, binding, MRTD) ---------
    vd = tdx.verify_tdx_quote_field(proof.quote)         # approved_mrtd=None: surface the MRTD
    print("\n[1] quote verifies (chain->Intel root + binding):", vd.ok, "-", vd.reason)
    assert vd.ok, vd.reason
    print("    attested MRTD :", vd.mr_td)
    print("    attested RTMR0:", vd.rtmrs[0])
    approved = {vd.mr_td}                                 # owner would pin this on-chain

    vd2 = tdx.verify_tdx_quote_field(proof.quote, approved_mrtd=approved)
    print("[2] with owner-approved MRTD gate:", vd2.ok, "-", vd2.reason)
    assert vd2.ok

    # --- the binding is load-bearing: tamper the proof -> hardware rejects it -----
    # A cheater keeps the genuine quote but edits an answer/cost in the published proof.
    # The quote's REPORTDATA still commits to the ORIGINAL report_data, so it no longer
    # matches the tampered proof -> rejected.
    tampered = dataclasses.replace(
        proof, results=(dataclasses.replace(proof.results[0], answer="FORGED"),) + proof.results[1:])
    vd_forge = tdx.verify_quote(
        tdx.base64.b64decode(proof.quote.platform_sig[4:]),
        expect_report_data=tdx.report_data_from_hash(tampered.report_data()),
        approved_mrtd=approved)
    print("[3] tampered proof (answer->FORGED) vs same quote:", vd_forge.ok, "-", vd_forge.reason)
    assert not vd_forge.ok and vd_forge.reason == "report_data_mismatch"

    # --- wrong approved-MRTD set is refused --------------------------------------
    vd_mr = tdx.verify_tdx_quote_field(proof.quote, approved_mrtd={"00" * 48})
    print("[4] unapproved MRTD set:", vd_mr.ok, "-", vd_mr.reason)
    assert not vd_mr.ok and vd_mr.reason == "mrtd_not_approved"

    print("\nALL CHECKS PASSED — genuine hardware-attested, un-forgeable KOTH proof.")


if __name__ == "__main__":
    main()
