#!/usr/bin/env bash
# Ground-truth TDX measurement probe for a GCP (or any) Intel-TDX confidential VM.
#
# Answers the ONE question the measured-image plan hinges on (docs/DEPLOYING.md §Stage-2, docs/DESIGN.md §8):
#   On this managed-cloud TDX guest, what is actually PINNABLE?
#     * Is MRTD a cloud-fixed constant (guest firmware we don't control) or per-image?
#     * Are RTMR1/RTMR2 populated by measured boot (→ a real per-image fingerprint we can pin),
#       or all-zero (→ the cloud does NOT measure the guest boot chain, leaving only the
#       userspace-extendable RTMR3, which a tampered runtime can forge)?
#     * Does the sysfs measurement match the signed TDX quote (sanity), and can we extend RTMR3?
#
# Zero dependencies (pure bash + xxd/od). Run as root on the CVM:  bash koth_gcp_measure_probe.sh
# Copy off the printed report; the decisive lines are marked  >>> VERDICT.
set -uo pipefail

sep() { printf '\n========== %s ==========\n' "$1"; }
MEAS=/sys/devices/virtual/misc/tdx_guest/measurements
TSM=/sys/kernel/config/tsm/report
ZERO48="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"

sep "host / kernel / boot model"
uname -a
echo "cmdline: $(cat /proc/cmdline 2>/dev/null)"
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|IMAGE_ID|VERSION)=' /etc/os-release
echo "EFI: $([ -d /sys/firmware/efi ] && echo yes || echo no)   secureboot vars: $(ls /sys/firmware/efi/efivars 2>/dev/null | grep -ci SecureBoot)"
echo "bootloader hints:"; ls -1 /boot 2>/dev/null | head;
echo "dm-verity on root: $(grep -o 'verity[^ ]*' /proc/cmdline 2>/dev/null || echo none)"

sep "GCP instance identity (metadata server — is this really a C3/TDX?)"
MD='-H Metadata-Flavor:Google -s --max-time 4'
for k in machine-type image cpu-platform name zone; do
  v=$(curl $MD "http://metadata.google.internal/computeMetadata/v1/instance/$k" 2>/dev/null)
  echo "$k: ${v:-<none>}"
done
echo "confidential-instance-type: $(curl $MD "http://metadata.google.internal/computeMetadata/v1/instance/confidential-instance-type" 2>/dev/null || echo '<none>')"

sep "TDX interfaces present"
echo "configfs TSM ($TSM): $([ -d "$TSM" ] && echo yes || echo NO)"
echo "/dev/tdx_guest: $([ -e /dev/tdx_guest ] && echo yes || echo NO)"
echo "measurements sysfs ($MEAS): $([ -d "$MEAS" ] && echo yes || echo NO)"

sep "RAW MEASUREMENTS (the decisive read)"
declare -A VAL
for name in mrtd rtmr0 rtmr1 rtmr2 rtmr3; do
  hex=""
  for attr in "$name:sha384" "$name"; do
    p="$MEAS/$attr"
    if [ -r "$p" ]; then hex=$(xxd -p "$p" 2>/dev/null | tr -d '\n'); [ -n "$hex" ] && break; fi
  done
  VAL[$name]=$hex
  if [ -z "$hex" ]; then
    printf '%-6s : <unreadable>\n' "$name"
  elif [ "$hex" = "$ZERO48" ]; then
    printf '%-6s : ALL-ZERO   (not measured)\n' "$name"
  else
    printf '%-6s : %s…%s  (populated)\n' "$name" "${hex:0:24}" "${hex: -8}"
  fi
done

sep ">>> VERDICT — what is pinnable on this cloud"
nz() { [ -n "${VAL[$1]}" ] && [ "${VAL[$1]}" != "$ZERO48" ] && echo yes || echo no; }
echo "MRTD populated : $(nz mrtd)   (if yes: check it's per-IMAGE not cloud-fixed — rebuild/relaunch to compare)"
echo "RTMR1 populated: $(nz rtmr1)   <- kernel; the per-image anchor IF the cloud measures boot"
echo "RTMR2 populated: $(nz rtmr2)   <- initrd/cmdline; the other half of the boot anchor"
echo "RTMR3 populated: $(nz rtmr3)   (runtime; userspace-extendable — NOT a standalone anchor)"
if [ "$(nz rtmr1)" = yes ] || [ "$(nz rtmr2)" = yes ]; then
  echo ">>> GOOD: the guest boot chain IS measured into RTMRs → a customer measured image gives a"
  echo "    real per-image fingerprint we can pin (gate on MRTD+RTMR1/2). Proceed with the build."
else
  echo ">>> WARNING: RTMR1/2 are NOT populated → this cloud does NOT measure the guest boot chain."
  echo "    Then MRTD is likely cloud-fixed and RTMR3 is userspace-only → the measured-image gate"
  echo "    would NOT discriminate a tampered runtime here. Need a boot path we control (own"
  echo "    firmware/td-shim direct boot) or a different platform. This is the load-bearing finding."
fi

sep "cross-check: signed TDX quote MRTD/RTMR == sysfs?"
if command -v python3 >/dev/null && python3 -c "import thirtyspokes.koth.tdx" 2>/dev/null; then
  python3 - <<'PY'
from thirtyspokes.koth import tdx, rtmr
try:
    raw = tdx.get_quote(tdx.report_data_from_hash(__import__("hashlib").sha256(b"probe").hexdigest()))
    p = tdx.parse_quote(raw)
    sysfs = rtmr.read_measurements()
    print("quote MRTD  :", p.mr_td.hex()[:24], "…", "  sysfs:", sysfs.get("mrtd","")[:24], "…",
          "  MATCH" if p.mr_td.hex()==sysfs.get("mrtd") else "  <DIFFERS>")
    for i in range(4):
        q = p.rtmrs[i].hex(); s = sysfs.get(f"rtmr{i}","")
        print(f"quote RTMR{i} :", q[:24], "…", "  MATCH" if q==s else "  <DIFFERS>")
except Exception as e:
    print("quote cross-check skipped:", type(e).__name__, e)
PY
else
  echo "(thirtyspokes not importable here — skipping quote cross-check; sysfs read above is sufficient)"
fi

sep "can we extend RTMR3? (needed for the runtime self-measure)"
echo "RTMR3 writable: $([ -w "$MEAS/rtmr3:sha384" ] && echo yes || echo no)  (do NOT extend here — it is one-way per boot)"
echo -e "\nprobe complete."
