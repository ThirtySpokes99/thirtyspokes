"""H2 smoke on real Intel-TDX hardware: bind the runtime into RTMR3, prove the hardware quote
carries it, and the validator gate enforces the owner-pinned MRTD + RTMR1/2/3 set.

Run INSIDE the confidential VM. Best after a fresh boot (RTMR3 == 0) so the idempotent extend
starts from zero, matching the owner's `owner_expected_rtmr3`.
"""
from __future__ import annotations

import hashlib
import time

from thirtyspokes.koth import rtmr, tdx
from thirtyspokes.koth.owner import build_record
from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement

POOL = ["openai/gpt-4o-mini"]


def main() -> None:
    if not rtmr.rtmr_extend_available():
        raise SystemExit("no writable RTMR3 — not a TDX guest?")
    m = rtmr.read_measurements()
    print("MRTD :", m["mrtd"][:24], "…")
    print("RTMR1:", m["rtmr1"][:24], "  RTMR2:", m["rtmr2"][:24])
    zero = m["rtmr3"] == rtmr.RTMR3_ZERO
    print("RTMR3 (pre):", m["rtmr3"][:24], "(fresh boot)" if zero else "(already extended)")

    # 1. bind runtime identity -> RTMR3 (idempotent per boot)
    got = rtmr.ensure_runtime_measured(runtime_measurement=runtime_measurement(),
                                       suite_version=SUITE_VERSION, pool_allow_list=POOL)
    want = rtmr.owner_expected_rtmr3(runtime_measurement=runtime_measurement(),
                                     suite_version=SUITE_VERSION, pool_allow_list=POOL)
    print("RTMR3 (post):", got[:24], "  == owner_expected:", got == want)
    assert got == want, "RTMR3 must equal the owner-derived expected value"

    # 2. a fresh hardware quote carries this RTMR3
    rd = tdx.report_data_from_hash(hashlib.sha256(b"rtmr-smoke").hexdigest())
    q = tdx.get_quote(rd)
    assert tdx.parse_quote(q).rtmrs[3].hex() == got
    print("quote RTMR3 matches the extended runtime measurement: True")

    # 3. owner governance record → validator gate enforces MRTD + RTMR1/2/3 + TCB
    rec = build_record(mrtd=[m["mrtd"]], rtmr1=m["rtmr1"], rtmr2=m["rtmr2"], pool_allow_list=POOL)
    approved_mrtd = set(rec["mrtd"])
    approved_rtmr = {int(k): v for k, v in rec["rtmr"].items()}
    now = int(time.time())
    ok = tdx.verify_quote_full(q, expect_report_data=rd, approved_mrtd=approved_mrtd,
                               approved_rtmr=approved_rtmr, now=now)
    print(f"[gate: owner-pinned set] ok={ok.ok} reason={ok.reason} tcb={ok.tcb_status}")
    assert ok.ok

    bad = tdx.verify_quote_full(q, expect_report_data=rd, approved_mrtd=approved_mrtd,
                                approved_rtmr={**approved_rtmr, 3: "00" * 48}, now=now)
    print(f"[gate: wrong RTMR3]     ok={bad.ok} reason={bad.reason}")
    assert not bad.ok and bad.reason == "rtmr3_not_approved"

    print("\nH2 OK — runtime bound into RTMR3, hardware quote carries it, validator gate enforces "
          "the owner-pinned MRTD + RTMR1/2/3 + TCB.")


if __name__ == "__main__":
    main()
