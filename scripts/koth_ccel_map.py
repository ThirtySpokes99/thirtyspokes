"""Discover the PCR->RTMR mapping empirically from the hardware, then locate the kernel+cmdline.

`koth_ccel_replay.py` assumed Intel's documented PCR->RTMR mapping and RTMR0 matched, proving the
parse + extend formula (`RTMR = SHA384(RTMR || digest)`) are right. RTMR1/RTMR2 did not match, so
the mapping differs on this platform. Instead of guessing from specs, brute-force it: for each
hardware RTMR, find the subset of PCR indices whose log-order fold reproduces the sysfs value.

Output answers the two questions the measured-image gate depends on (docs/DEPLOYING.md):
  * which RTMR holds the KERNEL + CMDLINE (the per-image anchor we pin), and
  * whether each RTMR is fully explained by the log (=> pre-computable offline, no boot needed).

Run as root ON the TDX guest:  python3 koth_ccel_map.py
"""

from __future__ import annotations

import hashlib
import itertools
import struct

CCEL_DATA = "/sys/firmware/acpi/tables/data/CCEL"
MEAS = "/sys/devices/virtual/misc/tdx_guest/measurements"
ALG_SIZE = {0x0004: 20, 0x000B: 32, 0x000C: 48, 0x000D: 64}
SHA384 = 0x000C
ZERO = b"\x00" * 48


def parse(data: bytes):
    off = 0
    struct.unpack_from("<II", data, off); off += 8 + 20
    (esize,) = struct.unpack_from("<I", data, off); off += 4 + esize
    events = []
    while off + 12 <= len(data):
        pcr, etype = struct.unpack_from("<II", data, off)
        (count,) = struct.unpack_from("<I", data, off + 8)
        if count == 0 or count > 8:
            break
        o, digests = off + 12, {}
        for _ in range(count):
            (alg,) = struct.unpack_from("<H", data, o); o += 2
            size = ALG_SIZE.get(alg)
            if size is None:
                return events
            digests[alg] = data[o:o + size]; o += size
        (esize,) = struct.unpack_from("<I", data, o); o += 4
        events.append((pcr, etype, digests, data[o:o + esize]))
        off = o + esize
    return events


def fold(events, pcrs: set[int]) -> bytes:
    acc = ZERO
    for pcr, _e, digests, _ev in events:
        if pcr in pcrs and SHA384 in digests:
            acc = hashlib.sha384(acc + digests[SHA384]).digest()
    return acc


def sysfs() -> dict[str, str]:
    out = {}
    for n in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"):
        try:
            with open(f"{MEAS}/{n}:sha384", "rb") as f:
                out[n] = f.read().hex()
        except OSError:
            pass
    return out


def main() -> None:
    with open(CCEL_DATA, "rb") as f:
        events = parse(f.read())
    hw = sysfs()
    pcrs = sorted({p for p, _e, d, _v in events if SHA384 in d})
    counts = {p: sum(1 for q, _e, d, _v in events if q == p and SHA384 in d) for p in pcrs}
    print(f"parsed {len(events)} events; PCR indices present (sha384 events): "
          + ", ".join(f"pcr{p}={counts[p]}" for p in pcrs) + "\n")

    print("=== brute-force: which PCR subset folds to each hardware RTMR? ===")
    solved: dict[int, set[int]] = {}
    for k in (0, 1, 2, 3):
        want = hw.get(f"rtmr{k}", "")
        if want == ZERO.hex():
            print(f"  RTMR{k}: all-zero on hardware (nothing extended)")
            continue
        found = None
        for r in range(1, len(pcrs) + 1):                 # smallest explaining subset first
            for sub in itertools.combinations(pcrs, r):
                if fold(events, set(sub)).hex() == want:
                    found = set(sub)
                    break
            if found:
                break
        if found:
            solved[k] = found
            print(f"  RTMR{k}: EXPLAINED by PCR {sorted(found)}  -> pre-computable from the log")
        else:
            print(f"  RTMR{k}: NOT explained by any PCR subset -> extended outside the CCEL log "
                  f"(hw={want[:24]}…) -> must be pinned by CAPTURE (boot + read sysfs)")

    print("\n=== where are the KERNEL and CMDLINE measured? ===")
    def label(ev: bytes) -> str:
        for dec in ("utf-16-le", "ascii"):
            try:
                s = ev.decode(dec).strip("\x00").strip()
                if s and all(31 < ord(c) < 127 or c == " " for c in s):
                    return s
            except (UnicodeDecodeError, ValueError):
                pass
        return ""
    for pcr, _e, d, ev in events:
        if SHA384 not in d:
            continue
        s = label(ev)
        if "kernel_cmdline" in s or s.startswith("/vmlinuz") or "initrd" in s.lower():
            rt = next((k for k, v in solved.items() if pcr in v), None)
            print(f"  pcr{pcr} -> RTMR{rt if rt is not None else '?'} : {s[:88]}")

    print("\n=== VERDICT for the measured-image gate ===")
    kernel_pcrs = {p for p, _e, d, ev in events
                   if SHA384 in d and ("kernel_cmdline" in label(ev) or label(ev).startswith("/vmlinuz"))}
    anchors = sorted({k for k, v in solved.items() if v & kernel_pcrs})
    if anchors:
        print(f"  Kernel+cmdline land in RTMR{anchors} -> pin RTMR{anchors} as the per-image anchor.")
    else:
        print("  Could not attribute kernel/cmdline to a solved RTMR; inspect the dump above.")
    unexplained = [k for k in (0, 1, 2) if hw.get(f"rtmr{k}") != ZERO.hex() and k not in solved]
    if unexplained:
        print(f"  RTMR{unexplained} are extended outside the log -> NOT offline-predictable; the owner")
        print("  must pin them by capturing one controlled boot of the built image.")


if __name__ == "__main__":
    main()
