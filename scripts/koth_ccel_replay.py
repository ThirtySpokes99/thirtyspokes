"""Replay the TDX boot event log (CCEL) and verify it reproduces the hardware RTMRs.

Why this matters (docs/DEPLOYING.md): the measured-image gate pins RTMR1/RTMR2. Two things must be
true for that to be usable, and this script proves both on real hardware:

  1. **Which components determine RTMR1/RTMR2.** The CCEL log records every measurement the boot
     chain extended, tagged with the PCR index that TDX maps onto an RTMR. Printing the events
     shows *exactly* what is in each register (bootloader vs kernel vs cmdline).
  2. **RTMRs are pre-computable.** If folding the logged SHA-384 digests in order reproduces the
     sysfs RTMR values bit-for-bit, then the owner can compute a candidate image's RTMR1/RTMR2
     offline from its components and pin them WITHOUT booting it (docs/DEPLOYING.md "Predict").

TDX RTMR <- PCR mapping (Intel TDX Virtual Firmware; matches GCP's documented register contents):
    RTMR0 <- PCR 1, 7        (TDVF config: platform config, secure-boot vars)
    RTMR1 <- PCR 2..6        (TD loader: GRUB/shim, EFI applications)
    RTMR2 <- PCR 8, 9        (the kernel, its command line, initrd)
    RTMR3 <- (nothing at boot; reserved for runtime/userspace -> koth/rtmr.py)
The extend is `RTMR = SHA384(RTMR || digest)` from an all-zero base.

Run as root ON the TDX guest:  python3 koth_ccel_replay.py
"""

from __future__ import annotations

import hashlib
import struct

CCEL_DATA = "/sys/firmware/acpi/tables/data/CCEL"
MEAS = "/sys/devices/virtual/misc/tdx_guest/measurements"

ALG_SIZE = {0x0004: 20, 0x000B: 32, 0x000C: 48, 0x000D: 64}   # sha1/256/384/512
SHA384 = 0x000C
PCR_TO_RTMR = {1: 0, 7: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 8: 2, 9: 2}

EV_NAMES = {
    0x00000001: "EV_POST_CODE", 0x00000003: "EV_NO_ACTION", 0x00000004: "EV_SEPARATOR",
    0x00000005: "EV_ACTION", 0x0000000D: "EV_IPL", 0x0000000E: "EV_IPL_PARTITION_DATA",
    0x80000001: "EV_EFI_VARIABLE_DRIVER_CONFIG", 0x80000002: "EV_EFI_VARIABLE_BOOT",
    0x80000003: "EV_EFI_BOOT_SERVICES_APPLICATION", 0x80000004: "EV_EFI_BOOT_SERVICES_DRIVER",
    0x80000006: "EV_EFI_GPT_EVENT", 0x80000007: "EV_EFI_ACTION",
    0x80000008: "EV_EFI_PLATFORM_FIRMWARE_BLOB", 0x80000009: "EV_EFI_HANDOFF_TABLES",
    0x8000000A: "EV_EFI_PLATFORM_FIRMWARE_BLOB2", 0x8000000C: "EV_EFI_VARIABLE_BOOT2",
    0x800000E0: "EV_EFI_VARIABLE_AUTHORITY",
}


def _readable(ev: bytes) -> str:
    """Best-effort human summary of the event blob (UTF-16LE strings are common in EFI events)."""
    for dec in ("utf-16-le", "ascii"):
        try:
            s = ev.decode(dec).strip("\x00").strip()
            if s and all(31 < ord(c) < 127 or c in "\t " for c in s):
                return s[:90]
        except (UnicodeDecodeError, ValueError):
            pass
    printable = "".join(chr(b) if 31 < b < 127 else "." for b in ev[:64])
    return printable


def parse(data: bytes):
    """Yield (pcr, event_type, {alg: digest}, event_bytes). First record is the legacy
    TCG_PCR_EVENT spec-ID header (SHA1-shaped); the rest are crypto-agile TCG_PCR_EVENT2."""
    off = 0
    _pcr, _etype = struct.unpack_from("<II", data, off); off += 8
    off += 20                                              # SHA1 digest of the spec-ID event
    (esize,) = struct.unpack_from("<I", data, off); off += 4 + esize

    events = []
    while off + 12 <= len(data):
        pcr, etype = struct.unpack_from("<II", data, off)
        (count,) = struct.unpack_from("<I", data, off + 8)
        if count == 0 or count > 8:                        # zero-fill past the end of the log
            break
        o = off + 12
        digests = {}
        for _ in range(count):
            (alg,) = struct.unpack_from("<H", data, o); o += 2
            size = ALG_SIZE.get(alg)
            if size is None:
                return events                              # unknown alg -> stop, keep what we have
            digests[alg] = data[o:o + size]; o += size
        (esize,) = struct.unpack_from("<I", data, o); o += 4
        events.append((pcr, etype, digests, data[o:o + esize]))
        off = o + esize
    return events


def read_sysfs() -> dict[str, str]:
    out = {}
    for name in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"):
        try:
            with open(f"{MEAS}/{name}:sha384", "rb") as f:
                out[name] = f.read().hex()
        except OSError:
            pass
    return out


def main() -> None:
    with open(CCEL_DATA, "rb") as f:
        events = parse(f.read())
    print(f"parsed {len(events)} boot events from {CCEL_DATA}\n")

    print("=== what each RTMR actually measures (from the log) ===")
    for target in (0, 1, 2):
        pcrs = sorted(p for p, r in PCR_TO_RTMR.items() if r == target)
        rows = [(p, e, d, ev) for (p, e, d, ev) in events if PCR_TO_RTMR.get(p) == target]
        print(f"\nRTMR{target}  (PCR {pcrs}) — {len(rows)} events")
        for pcr, etype, _d, ev in rows:
            name = EV_NAMES.get(etype, f"0x{etype:08x}")
            desc = _readable(ev)
            print(f"   pcr{pcr} {name:34s} {desc}")

    print("\n=== REPLAY: fold logged SHA-384 digests -> compare to hardware sysfs ===")
    r = {k: b"\x00" * 48 for k in range(4)}
    for pcr, _etype, digests, _ev in events:
        k = PCR_TO_RTMR.get(pcr)
        d = digests.get(SHA384)
        if k is None or d is None:
            continue
        r[k] = hashlib.sha384(r[k] + d).digest()

    sysfs = read_sysfs()
    ok = True
    for k in (0, 1, 2):
        got, want = r[k].hex(), sysfs.get(f"rtmr{k}", "")
        match = got == want
        ok &= match
        print(f"  RTMR{k}: replay={got[:32]}…  sysfs={want[:32]}…  {'MATCH' if match else 'DIFFER'}")
    print(f"  MRTD  : {sysfs.get('mrtd','')[:32]}…  (build-time; not in the event log — GCP's TDVF)")

    print("\n=== VERDICT ===")
    if ok:
        print("  ✅ The event log fully explains RTMR0/1/2 — so RTMR1/RTMR2 are PRE-COMPUTABLE")
        print("     offline from an image's bootloader/kernel/cmdline. The owner can pin a")
        print("     candidate image's measurements without booting it (docs/DEPLOYING.md 'Predict').")
    else:
        print("  ⚠️  Replay does not reproduce the hardware RTMRs — the PCR->RTMR mapping or the")
        print("     log parse is off. Pin measurements by CAPTURE (boot + read sysfs) instead.")


if __name__ == "__main__":
    main()
