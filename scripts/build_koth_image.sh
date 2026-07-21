#!/usr/bin/env bash
# Reproducible, locked-down KOTH measured-runtime CVM image (H2, see docs/DEPLOYING.md).
#
# Produces a deterministic Intel-TDX guest image: dm-verity read-only rootfs, NO sshd / getty /
# shell, booting straight into the KOTH runtime. Its MRTD + RTMR1/2 are reproducible and become
# the owner-pinned approved measurements (orchestra-koth-owner). This is the recipe/skeleton; the
# first measured build + boot is an ops step (docs/DEPLOYING.md Stage-2).
#
# Requires: mkosi >= 20, a TDX-capable kernel package, systemd-repart (dm-verity).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO_ROOT/build/koth-image}"
: "${SOURCE_DATE_EPOCH:=1700000000}"     # pin for reproducibility (override to your commit date)
export SOURCE_DATE_EPOCH

mkdir -p "$OUT"
cat > "$OUT/mkosi.conf" <<EOF
[Distribution]
Distribution=ubuntu
Release=noble

[Output]
Format=disk
ImageId=koth-runtime
# dm-verity read-only rootfs: the root hash goes on the (measured) kernel cmdline
Verity=signed
SplitArtifacts=yes

[Content]
# minimal userland: python + the pinned wheel, NO openssh-server / no login shells.
# curl is needed by the boot-time secret fetch; nvme/gve (gVNIC) drivers are REQUIRED for a
# GCP Confidential VM to boot + network (see docs/DEPLOYING.md GCP requirements).
Packages=python3,python3-venv,ca-certificates,curl,linux-modules-extra
RemovePackages=openssh-server
Bootable=yes
# Kernel >= 6.6 is a GCP TDX requirement; pin the exact package for reproducible RTMR2.
# Kernel + this cmdline are measured into RTMR2 (the per-image anchor); the dm-verity root
# hash mkosi appends to the cmdline makes RTMR2 transitively commit the whole rootfs.
KernelCommandLine=quiet ro console=ttyS0 systemd.mask=getty@tty1.service systemd.mask=serial-getty@ttyS0.service

[Validation]
# reproducibility: no machine-id / timestamps baked in
SourceDateEpoch=$SOURCE_DATE_EPOCH
EOF

mkdir -p "$OUT/mkosi.extra/etc/systemd/system" "$OUT/mkosi.extra/opt/koth"

# --- boot-time secret injection ------------------------------------------------------------
# The image is byte-identical for every miner (so RTMR1/2 are one pinnable value); each miner's
# OWN secrets (their OpenRouter key, their Bittensor HOTKEY, HF token, netuid/repo/pool) are NOT
# baked in. A oneshot pulls them from the CVM launch metadata into tmpfs (/run, RAM-only) BEFORE
# the miner starts, so they never touch the measured rootfs or change RTMR1/2/3. On GCP the miner
# supplies them at create time: `--metadata-from-file koth-secrets=./my-secrets.env`.
# NOTE: cloud metadata is delivered by the (untrusted) hypervisor — fine here, since these are the
# miner's OWN secrets and the subnet's integrity rests on the measurement, not on their secrecy.
cat > "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /run/koth && chmod 700 /run/koth       # /run is tmpfs: RAM-only, gone on reboot
MD='http://metadata.google.internal/computeMetadata/v1/instance/attributes/koth-secrets'
curl -fsS -H 'Metadata-Flavor: Google' "$MD" > /run/koth/secrets.env
chmod 600 /run/koth/secrets.env
EOF
chmod +x "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh"

cat > "$OUT/mkosi.extra/etc/systemd/system/koth-secrets.service" <<'EOF'
[Unit]
Description=Fetch KOTH miner secrets from CVM metadata into tmpfs
After=network-online.target
Wants=network-online.target
Before=koth-miner.service
[Service]
Type=oneshot
ExecStart=/opt/koth/fetch-secrets.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF

# systemd unit: boot straight into the confined miner (confine=True); secrets from the oneshot
cat > "$OUT/mkosi.extra/etc/systemd/system/koth-miner.service" <<'EOF'
[Unit]
Description=KOTH measured runtime (miner)
After=koth-secrets.service network-online.target
Requires=koth-secrets.service
Wants=network-online.target

[Service]
Type=simple
# secrets fetched into tmpfs by koth-secrets.service — never baked into the measured image.
# The file also carries KOTH_NETUID/KOTH_REPO/KOTH_POOL so the image stays miner-agnostic.
EnvironmentFile=/run/koth/secrets.env
ExecStart=/opt/koth/venv/bin/orchestra-koth-miner --confine \
  --netuid ${KOTH_NETUID} --wallet miner --repo ${KOTH_REPO} \
  --source /opt/koth/my_router.py --weights /opt/koth/my_weights.bin \
  --pool ${KOTH_POOL}
Restart=no

[Install]
WantedBy=multi-user.target
EOF

echo "mkosi config written to $OUT"
echo "RECIPE ONLY — the mkosi stanzas + pinned kernel need verifying on first boot (docs/DEPLOYING.md)."
echo "next: install the pinned thirtyspokes wheel into /opt/koth/venv in mkosi.extra, then:"
echo "  ( cd $OUT && mkosi build )"
echo "then read MRTD/RTMR1/2 from a controlled boot (koth/rtmr.read_measurements) and pin via orchestra-koth-owner."
