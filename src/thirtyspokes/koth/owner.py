"""Owner governance CLI (H2/H4) — publish + rotate the approved-measurement set on-chain.

The subnet owner pins WHICH hardware images + software runtime + TCB levels validators accept,
so the whole subnet enforces one governed policy and the owner can rotate it on TCB recovery or
an image upgrade. The record binds: approved MRTD + boot RTMR1/2 (from the reproducible measured
image, docs/DEPLOYING.md), the runtime RTMR3 (derived from runtime+suite+pinned pool), the software
`runtime_measurement`, the TCB acceptance policy, and the pool allow-list. Validators read it each
epoch via `chain.owner_measurements()` (see `KOTHValidator._effective_governance`).
"""

from __future__ import annotations

import json

from .rtmr import owner_expected_rtmr3
from .runtime import SUITE_VERSION, runtime_measurement


def build_record(*, mrtd, rtmr1: str, rtmr2: str, pool_allow_list,
                 tcb_accept=("UpToDate", "SWHardeningNeeded"),
                 suite_version: str = SUITE_VERSION, version: int = 1,
                 probe_commit: str | None = None) -> dict:
    rm = runtime_measurement()
    rec = {
        "version": version,
        "runtime_measurements": [rm],
        "mrtd": list(mrtd),
        "rtmr": {"1": rtmr1, "2": rtmr2,
                 "3": owner_expected_rtmr3(runtime_measurement=rm, suite_version=suite_version,
                                           pool_allow_list=pool_allow_list)},
        "tcb_accept": list(tcb_accept),
        "pool_allow_list": sorted(pool_allow_list),
    }
    if probe_commit:                     # commit to the secret memorization probe (koth/heldout.py)
        rec["probe_commit"] = probe_commit
    return rec


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Publish the KOTH owner approved-measurement set")
    p.add_argument("--mrtd", nargs="+", required=True, help="approved image MRTD(s), hex")
    p.add_argument("--rtmr1", required=True, help="approved boot RTMR1 (kernel), hex")
    p.add_argument("--rtmr2", required=True, help="approved boot RTMR2 (initrd/cmdline), hex")
    p.add_argument("--pool", required=True, help="comma-separated pinned model allow-list")
    p.add_argument("--tcb-accept", default="UpToDate,SWHardeningNeeded")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--probe-bank", help="path to the secret memorization-probe bank JSON "
                                        "(koth/heldout.py); its commit hash is published on-chain")
    p.add_argument("--netuid", type=int, help="omit to just print the record (dry run)")
    p.add_argument("--wallet", default="owner")
    p.add_argument("--hotkey", default="default", help="hotkey NAME inside the wallet")
    p.add_argument("--network", default="finney")
    args = p.parse_args()

    pool = [m.strip() for m in args.pool.split(",") if m.strip()]
    probe_commit = None
    if args.probe_bank:
        from .heldout import SecretProbeBank
        probe_commit = SecretProbeBank.from_file(args.probe_bank).commit()
    rec = build_record(mrtd=args.mrtd, rtmr1=args.rtmr1, rtmr2=args.rtmr2, pool_allow_list=pool,
                       tcb_accept=tuple(t.strip() for t in args.tcb_accept.split(",")),
                       version=args.version, probe_commit=probe_commit)
    if args.netuid is None:
        print(json.dumps(rec, indent=2))
        return
    from ..subnet.chain import BittensorChain
    from . import governance
    chain = BittensorChain(args.netuid, args.wallet, network=args.network, hotkey=args.hotkey)
    chain.publish_owner_measurements(rec)      # record -> bucket (by hash); hash -> chain (plain)
    d = governance.digest(rec)
    print(f"published owner measurement record v{rec['version']} to netuid {args.netuid}")
    print(f"  digest (on-chain) : {d}")
    print(f"  record  (bucket)  : {governance.record_path(d)}")
    print("  visible to validators IMMEDIATELY — the record is hash-committed, not timelocked.")


if __name__ == "__main__":
    main()
