#!/usr/bin/env bash
# KOTH measured-runtime image build — mkosi 25.x, VERIFIED building on a GCP C3 (2026-07-09).
# Iteration 1: bootable UKI (reproducible RTMR2 by construction), no dm-verity yet, no ssh.
# On first boot a oneshot prints RTMR1/2/3 to the SERIAL console -> read via
# `gcloud compute instances get-serial-port-output` (no ssh into the locked image).
#
# HOST TOOLING (Ubuntu 24.04 build host; Ubuntu's mkosi 20.2 is TOO OLD for noble t64 — use 25.x):
#   apt-get install -y systemd-boot systemd-ukify mtools dosfstools cryptsetup-bin squashfs-tools \
#                      python3-build python3-pip pipx git
#   pipx install 'git+https://github.com/systemd/mkosi.git@v25.3'
#   # then build the wheel:  (cd <repo> && python3 -m build --wheel -o /root/koth-build/wheels)
# Uses the host's prebuilt initrd for the kernel (mkosi-initrd's default pkg list still has
# pre-t64 tss2 names). ITER-2 TODO: reproducible initrd + dm-verity (Verity=) for the rootfs lock.
set -euo pipefail
export PATH="$PATH:/root/.local/bin"

OUT=/root/koth-build
WHEEL=$(ls "$OUT"/wheels/thirtyspokes-*.whl | head -1)
KVER=6.17.0-1020-gcp
: "${SOURCE_DATE_EPOCH:=1700000000}"; export SOURCE_DATE_EPOCH
rm -rf "$OUT/mkosi.extra" "$OUT/mkosi.conf" "$OUT/mkosi.postinst"
mkdir -p "$OUT/mkosi.extra/opt/koth" "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants"

cat > "$OUT/mkosi.conf" <<EOF
[Distribution]
Distribution=ubuntu
Release=noble
Repositories=main,universe

[Output]
Format=disk
ImageId=koth-runtime

[Content]
Packages=linux-image-${KVER},systemd,systemd-boot,systemd-sysv,udev,dbus,ca-certificates,curl,python3,python3-venv,libpython3.12t64,nvme-cli
RemovePackages=openssh-server
Bootable=yes
Bootloader=systemd-boot
# use the host's prebuilt initrd for this exact kernel (has nvme+gve) instead of mkosi-initrd,
# whose default package list still uses pre-t64 tss2 names that don't exist on noble.
Initrds=/boot/initrd.img-${KVER}
KernelCommandLine=quiet ro console=ttyS0,115200 panic=-1 systemd.mask=getty@tty1.service systemd.mask=serial-getty@ttyS0.service
SourceDateEpoch=$SOURCE_DATE_EPOCH
EOF

# --- pre-staged self-contained venv at the final image path (host==image x86_64/py3.12) ---
rm -rf /opt/koth/venv
python3 -m venv --copies /opt/koth/venv
/opt/koth/venv/bin/pip install -q --no-cache-dir "$WHEEL" 'cryptography>=42' 'dcap-qvl>=0.5' numpy
/opt/koth/venv/bin/python -c "import thirtyspokes.koth.tdx, dcap_qvl, numpy, cryptography; print('staged venv OK')"
cp -a /opt/koth/venv "$OUT/mkosi.extra/opt/koth/venv"

# --- boot-time secret fetch (GCP metadata -> tmpfs) ---
cat > "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /run/koth && chmod 700 /run/koth
curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/koth-secrets \
  > /run/koth/secrets.env || echo "# no koth-secrets metadata" > /run/koth/secrets.env
chmod 600 /run/koth/secrets.env
EOF
chmod +x "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh"

# --- serial-console RTMR reporter ---
cat > "$OUT/mkosi.extra/opt/koth/report.py" <<'EOF'
from thirtyspokes.koth import rtmr, tdx, runtime
m = rtmr.read_measurements()
print("KOTH-MEASURE mrtd=%s" % m.get("mrtd",""), flush=True)
for i in (0,1,2,3):
    print("KOTH-MEASURE rtmr%d=%s" % (i, m.get("rtmr%d"%i,"")), flush=True)
print("KOTH-MEASURE runtime_measurement=%s suite=%s" % (runtime.runtime_measurement(), runtime.SUITE_VERSION), flush=True)
print("KOTH-MEASURE tdx_available=%s" % tdx.tdx_available(), flush=True)
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

echo "=== mkosi 25 build ==="
cd "$OUT"
mkosi --force build 2>&1 | tail -25
echo "=== artifacts ==="; ls -la "$OUT"/*.raw 2>/dev/null || ls "$OUT"
