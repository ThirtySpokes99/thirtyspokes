"""RTMR runtime self-measurement (H2) — bind the running runtime into the TDX measurement.

On a TDX guest the hardware measurements split cleanly:
  * MRTD + RTMR0/1/2  — the measured-boot chain (firmware, kernel, initrd, cmdline). Fixed by
    the owner's reproducible image (docs/DEPLOYING.md); the owner pins them on-chain.
  * RTMR3             — reserved for RUNTIME measurements. This module extends it, once at
    enclave startup, with a digest of the runtime identity (runtime+suite+pinned pool), so the
    hardware quote's RTMR3 reflects exactly what ran. The validator gates RTMR3 against the
    owner-pinned expected value (H2 wiring in `verify_proof`/`validator`).

The kernel extend is `RTMR3 <- SHA384(RTMR3 || digest)`; a fresh boot leaves RTMR3 zeroed, so the
expected post-extend value is `SHA384(0*48 || digest)`.

Honest scope: extending RTMR3 from userspace is load-bearing ONLY inside the owner's locked
measured image (MRTD+RTMR1/2 pin the boot chain; a dm-verity read-only rootfs + no shell mean
nothing but the runtime can run to extend it). On a general-purpose VM it is defense-in-depth +
the measurement plumbing. See docs/DEPLOYING.md.
"""

from __future__ import annotations

import hashlib
import os

from ..gateway import signing

_MEAS_DIR = "/sys/devices/virtual/misc/tdx_guest/measurements"
_RTMR3_PATH = os.path.join(_MEAS_DIR, "rtmr3:sha384")
_NAMES = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")
RTMR3_ZERO = "00" * 48          # a fresh TDX boot leaves RTMR3 all-zero
RTMR3_DIGEST_DOMAIN = "koth-rtmr3-v1"


def rtmr_extend_available() -> bool:
    return os.access(_RTMR3_PATH, os.W_OK)


def read_measurements() -> dict[str, str]:
    """Read MRTD + RTMR0..3 straight from the kernel sysfs (hex). Empty off a TDX guest."""
    out: dict[str, str] = {}
    for name in _NAMES:
        for attr in (f"{name}:sha384", name):
            p = os.path.join(_MEAS_DIR, attr)
            try:
                with open(p, "rb") as f:
                    out[name] = f.read().hex()
                break
            except OSError:
                continue
    return out


def runtime_digest(*, runtime_measurement: str, suite_version: str, pool_allow_list) -> bytes:
    """The 48-byte SHA-384 digest extended into RTMR3 — a canonical binding of the runtime id
    (the measured runtime/suite version + the exact pinned pool allow-list)."""
    payload = "|".join([RTMR3_DIGEST_DOMAIN, runtime_measurement, suite_version,
                        ",".join(sorted(pool_allow_list))])
    return hashlib.sha384(payload.encode()).digest()


def expected_rtmr3(digest48: bytes, *, base_hex: str = RTMR3_ZERO) -> str:
    """RTMR3 after extending `base` once with `digest48`: SHA384(base || digest48)."""
    return hashlib.sha384(bytes.fromhex(base_hex) + digest48).hexdigest()


def extend_rtmr3(digest48: bytes) -> None:
    if len(digest48) != 48:
        raise ValueError("RTMR extend digest must be 48 bytes (SHA-384)")
    with open(_RTMR3_PATH, "wb") as f:
        f.write(digest48)


def ensure_runtime_measured(*, runtime_measurement: str, suite_version: str,
                            pool_allow_list) -> str:
    """Idempotently extend RTMR3 with the runtime identity (once per boot). Returns the
    resulting RTMR3 hex. Raises if RTMR3 was already extended to a DIFFERENT value (tamper /
    double-run). No-op returning the current value if already at the expected one."""
    digest = runtime_digest(runtime_measurement=runtime_measurement,
                            suite_version=suite_version, pool_allow_list=pool_allow_list)
    want = expected_rtmr3(digest)
    cur = read_measurements().get("rtmr3", "")
    if cur == want:
        return cur                       # already measured this boot
    if cur != RTMR3_ZERO:
        raise RuntimeError(f"RTMR3 already extended to an unexpected value: {cur[:16]}…")
    extend_rtmr3(digest)
    got = read_measurements().get("rtmr3", "")
    if got != want:                      # hardware sanity: kernel extend must match our formula
        raise RuntimeError(f"RTMR3 extend mismatch: got {got[:16]}… want {want[:16]}…")
    return got


def owner_expected_rtmr3(*, runtime_measurement: str, suite_version: str, pool_allow_list) -> str:
    """The RTMR3 the owner pins on-chain for a given runtime/suite/pool (validator side)."""
    return expected_rtmr3(runtime_digest(runtime_measurement=runtime_measurement,
                                         suite_version=suite_version, pool_allow_list=pool_allow_list))


def measurement_set(*, runtime_measurement: str, suite_version: str, pool_allow_list,
                    mrtd: str, rtmr1: str, rtmr2: str) -> dict:
    """Assemble the owner's approved hardware-measurement set (MRTD + boot RTMR1/2 pinned to
    the image, RTMR3 pinned to the runtime). Consumed by the governance record (H2/H4)."""
    return {
        "mrtd": mrtd, "rtmr1": rtmr1, "rtmr2": rtmr2,
        "rtmr3": owner_expected_rtmr3(runtime_measurement=runtime_measurement,
                                      suite_version=suite_version, pool_allow_list=pool_allow_list),
        "runtime_measurement": runtime_measurement,
        "digest": signing.sha256_hex({"suite": suite_version,
                                      "pool": sorted(pool_allow_list)}),
    }
