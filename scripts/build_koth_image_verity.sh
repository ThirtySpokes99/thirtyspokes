#!/usr/bin/env bash
# KOTH measured image — ITERATION 2: dm-verity read-only rootfs.
# The verity roothash is baked into the UKI cmdline -> measured into RTMR1 -> the ROOTFS is locked
# (a miner can't swap /opt/koth and keep the same RTMR1). Uses mkosi's own verity-capable initrd
# (systemd-veritysetup + gpt-auto), libstdc++6 in-image (numpy), KernelCommandLineExtra so mkosi's
# auto roothash/root wiring stays. Boot fix (from iter-1): copy the UKI to /EFI/BOOT/BOOTX64.EFI.
set -euo pipefail
export PATH="$PATH:/root/.local/bin"

OUT=/root/koth-build-v6
WHEEL=$(ls /root/koth-build/wheels/thirtyspokes-*.whl | head -1)
KVER=6.17.0-1020-gcp
: "${SOURCE_DATE_EPOCH:=1700000000}"; export SOURCE_DATE_EPOCH
rm -rf "$OUT"; mkdir -p "$OUT/mkosi.extra/opt/koth" "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants"

cat > "$OUT/mkosi.conf" <<EOF
[Distribution]
Distribution=ubuntu
Release=noble
Repositories=main,universe

[Output]
Format=disk
ImageId=koth-runtime
# pinned seed -> deterministic partition + erofs filesystem UUIDs (else random per build)
Seed=a1b2c3d4-e5f6-4789-abcd-ef0123456789

[Content]
Packages=linux-image-${KVER},systemd,systemd-boot,systemd-sysv,udev,dbus,ca-certificates,curl,python3,python3-venv,libpython3.12t64,libstdc++6,nvme-cli
RemovePackages=openssh-server
# strip generated non-deterministic state so rebuilds produce an identical rootfs -> identical
# roothash -> identical RTMR1 (regenerated at boot into the writable overlay). See docs/DEPLOYING.md.
RemoveFiles=/etc/nvme/hostid /etc/nvme/hostnqn /var/cache/ldconfig/aux-cache /var/log/alternatives.log
Bootable=yes
Bootloader=systemd-boot
KernelModulesInclude=nvme gve dm-verity erofs overlay
KernelCommandLine=console=ttyS0,115200 systemd.volatile=overlay systemd.mask=getty@tty1.service systemd.mask=serial-getty@ttyS0.service
SourceDateEpoch=$SOURCE_DATE_EPOCH
EOF

# --- dm-verity partition layout: ESP + root(Verity=data,erofs) + root-verity(hash).
# mkosi computes the roothash and auto-injects `roothash=` into the UKI cmdline -> RTMR1. ---
mkdir -p "$OUT/mkosi.repart"
cat > "$OUT/mkosi.repart/00-esp.conf" <<'EOF'
[Partition]
Type=esp
Format=vfat
CopyFiles=/boot:/
CopyFiles=/efi:/
SizeMinBytes=512M
SizeMaxBytes=512M
EOF
cat > "$OUT/mkosi.repart/10-root.conf" <<'EOF'
[Partition]
Type=root
Format=erofs
CopyFiles=/
Verity=data
VerityMatchKey=root
Minimize=best
EOF
cat > "$OUT/mkosi.repart/20-root-verity.conf" <<'EOF'
[Partition]
Type=root-verity
Verity=hash
VerityMatchKey=root
Minimize=best
EOF

# mkosi.finalize runs LAST (after ldconfig / package postinsts regenerate them), so strip the
# generated non-deterministic state here for a bit-reproducible rootfs -> stable roothash -> RTMR1.
cat > "$OUT/mkosi.finalize" <<'EOF'
#!/bin/bash
set -e
rm -f "$BUILDROOT"/etc/nvme/hostid "$BUILDROOT"/etc/nvme/hostnqn \
      "$BUILDROOT"/var/cache/ldconfig/aux-cache "$BUILDROOT"/var/log/alternatives.log
EOF
chmod +x "$OUT/mkosi.finalize"

# self-contained venv (host==image x86_64/py3.12); copied via extra tree
rm -rf /opt/koth/venv
python3 -m venv --copies /opt/koth/venv
/opt/koth/venv/bin/pip install -q --no-cache-dir "$WHEEL" 'cryptography>=42' 'dcap-qvl>=0.5' numpy
cp -a /opt/koth/venv "$OUT/mkosi.extra/opt/koth/venv"

cat > "$OUT/mkosi.extra/opt/koth/report.py" <<'EOF'
import glob
from thirtyspokes.koth import rtmr, tdx, runtime
m = rtmr.read_measurements()
print("KOTH-MEASURE mrtd=%s" % m.get("mrtd",""), flush=True)
for i in (0,1,2,3):
    print("KOTH-MEASURE rtmr%d=%s" % (i, m.get("rtmr%d"%i,"")), flush=True)
print("KOTH-MEASURE runtime_measurement=%s suite=%s" % (runtime.runtime_measurement(), runtime.SUITE_VERSION), flush=True)
# definitive dm-verity check: a verity device's sysfs UUID starts with CRYPT-VERITY
uuids = []
for p in glob.glob("/sys/block/dm-*/dm/uuid"):
    try: uuids.append(open(p).read().strip())
    except OSError: pass
verity = any(u.startswith("CRYPT-VERITY") for u in uuids)
src = fstype = "?"
for line in open("/proc/self/mountinfo"):
    f = line.split()
    if f[4] == "/":
        i = f.index("-"); fstype, src = f[i+1], f[i+2]
print("KOTH-MEASURE verity_active=%s dm_uuids=%s" % (verity, ",".join(uuids)[:80]), flush=True)
print("KOTH-MEASURE root_fstype=%s root_src=%s tdx=%s" % (fstype, src, tdx.tdx_available()), flush=True)
EOF

cat > "$OUT/mkosi.extra/etc/systemd/system/koth-report.service" <<'EOF'
[Unit]
Description=Print KOTH hardware measurements to the serial console
After=multi-user.target
[Service]
Type=oneshot
StandardOutput=journal+console
ExecStart=/opt/koth/venv/bin/python /opt/koth/report.py
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
ln -sf ../koth-report.service "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants/koth-report.service"

echo "=== mkosi build (verity) ==="
cd "$OUT"
mkosi --force build 2>&1 | tail -30
echo "=== artifacts ==="; ls -la "$OUT"/*.raw* 2>/dev/null

# --- POST-BUILD (validated + REPRODUCIBLE on GCP confidential TDX, 2026-07-09) --------------------
# Two independent builds produce a BYTE-IDENTICAL UKI -> identical RTMR1 (Seed pins the erofs/part
# UUIDs; mkosi.finalize strips generated state: nvme hostid/hostnqn, ldconfig aux-cache, alt log).
# systemd-boot does NOT boot on GCP confidential TDVF, so boot the UKI DIRECTLY:
#   LOOP=$(losetup --find --show --partscan "$OUT/koth-runtime.raw"); mount ${LOOP}p1 /mnt/esp
#   cp /mnt/esp/EFI/Linux/*.efi /mnt/esp/EFI/BOOT/BOOTX64.EFI   # deterministic
#   umount /mnt/esp; losetup -d $LOOP
# Package + image + launch (see docs/DEPLOYING.md); read RTMR1 from serial (koth-report prints
# KOTH-MEASURE rtmr1=...; verity_active=True). Pin: orchestra-koth-owner --mrtd <M> --rtmr1 <R1>
#   --rtmr2 0*96 --pool "<allow-list>". Miners rebuild this recipe -> same UKI -> verify RTMR1.
