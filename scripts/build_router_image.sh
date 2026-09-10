#!/usr/bin/env bash
# ThirtySpokes CONFIDENTIAL ROUTER measured image (docs/CONFIDENTIAL_SERVING.md §8).
#
# Same shape as the v2 recipe build_koth_image_prod.sh (removed with v2, 2026-09-07) — dm-verity read-only rootfs, roothash baked into the UKI
# cmdline and therefore into RTMR1 — because that recipe was debugged on real hardware and every
# reproducibility trap it documents applies here unchanged. Read its comments before editing this.
#
# WHY THIS IS A SEPARATE RECIPE RATHER THAN A FLAG ON THAT ONE.
#
#   1. `cryptography>=42` is not enough. HPKE arrived in 49, and koth/sealed.py -- the whole sealed
#      channel -- fails to import under 42. Bumping the pin in the KOTH recipe would change the MRTD
#      of the image that is ALREADY GOVERNING mainnet, retroactively unapproving every honest miner
#      running it. A different service gets a different image.
#   2. The two runtimes are shaped differently: KOTH runs a benchmark once and exits, this serves
#      requests indefinitely and must come back after a crash.
#   3. No benchmark datasets. The router never loads MMLU/GSM8K, so `datasets` and its cache are
#      omitted -- a smaller image is a smaller measured surface.
#
# WHAT IS DELIBERATELY THE SAME: the frozen encoder is baked in. A routing model's only view of a
# request is that embedding, so it must be inside the measurement — a miner cannot supply it and the
# runtime must not fetch it at boot (HF_HUB_OFFLINE=1; anything unmeasured is outside RTMR1).
#
# Build on a machine with mkosi 25+, then boot the .raw on a TDX host and read the MRTD with
# `tdx.self_mrtd()`. That value is what `orchestra-serving-governance --mrtd` approves.
set -euo pipefail
export PATH="$PATH:/root/.local/bin"

OUT=${OUT:-/root/router-build}
WHEEL=${WHEEL:?set WHEEL=/path/to/thirtyspokes-*.whl}
KVER=${KVER:-6.17.0-1020-gcp}
: "${SOURCE_DATE_EPOCH:=1700000000}"; export SOURCE_DATE_EPOCH

rm -rf "$OUT"
mkdir -p "$OUT/mkosi.extra/opt/router" "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants"

cat > "$OUT/mkosi.conf" <<EOF
[Distribution]
Distribution=ubuntu
Release=noble
Repositories=main,universe

[Output]
Format=disk
ImageId=thirtyspokes-router
# A DIFFERENT pinned seed from the KOTH image on purpose: same seed would give the two images
# identical partition UUIDs, and a confusing pair of artifacts is a bad thing to govern.
Seed=7f3c9e21-4d58-4a6b-9c02-1e5b8a3d7f40

[Content]
Packages=linux-image-${KVER},systemd,systemd-boot,systemd-sysv,udev,dbus,ca-certificates,curl,python3,python3-venv,libpython3.12t64,libstdc++6,nvme-cli,util-linux
RemovePackages=openssh-server
RemoveFiles=/etc/nvme/hostid /etc/nvme/hostnqn /var/cache/ldconfig/aux-cache /var/log/alternatives.log
Bootable=yes
Bootloader=systemd-boot
KernelModulesInclude=nvme gve dm-verity erofs overlay
KernelCommandLine=console=ttyS0,115200 systemd.volatile=overlay systemd.mask=getty@tty1.service systemd.mask=serial-getty@ttyS0.service
SourceDateEpoch=$SOURCE_DATE_EPOCH
EOF

mkdir -p "$OUT/mkosi.repart"
cat > "$OUT/mkosi.repart/00-esp.conf" <<'EOF'
[Partition]
Type=esp
Format=vfat
CopyFiles=/boot:/
CopyFiles=/efi:/
SizeMinBytes=256M
SizeMaxBytes=256M
EOF
cat > "$OUT/mkosi.repart/10-root.conf" <<'EOF'
[Partition]
Type=root
Format=erofs
CopyFiles=/
Verity=data
VerityMatchKey=root
Minimize=guess
EOF
cat > "$OUT/mkosi.repart/20-root-verity.conf" <<'EOF'
[Partition]
Type=root-verity
Verity=hash
VerityMatchKey=root
Minimize=guess
EOF

cat > "$OUT/mkosi.finalize" <<'EOF'
#!/bin/sh
# Runs LAST, after package postinsts regenerate them. See the v2 recipe (removed with v2, 2026-09-07); its notes are in git history.
rm -f "$BUILDROOT/var/cache/ldconfig/aux-cache" "$BUILDROOT/etc/nvme/hostid" \
      "$BUILDROOT/etc/nvme/hostnqn" "$BUILDROOT/var/log/alternatives.log"
EOF
chmod +x "$OUT/mkosi.finalize"

# --- the venv, staged at a FIXED path outside the image tree ------------------------------------
# The staging path leaks into pyvenv.cfg, console-script shebangs, RECORD hashes and every .pyc
# co_filename. All four reach the rootfs -> the roothash -> RTMR1. the v2 recipe (removed with v2, 2026-09-07) documented
# each leak and what it measured; this uses the identical remedy.
VENV_STAGE="${VENV_STAGE:-/var/tmp/router-venv-stage}"
rm -rf "$VENV_STAGE"
python3 -m venv --copies "$VENV_STAGE"

# CPU torch first, so the sentence-transformers resolve cannot pull the CUDA build (~1.2 GB of GPU
# runtime this image can never use, all of it hashed into RTMR1).
"$VENV_STAGE/bin/pip" install -q --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch
# cryptography>=49 is THE pin that distinguishes this image: HPKE, and therefore the sealed channel,
# does not exist before it. Ubuntu's python3-cryptography is older still — measured on a TDX guest,
# it has no `hpke` module at all.
"$VENV_STAGE/bin/pip" install -q --no-cache-dir "$WHEEL" \
  'cryptography>=49' 'dcap-qvl>=0.5' numpy openai sentence-transformers \
  fastapi 'uvicorn[standard]' httpx
cp -a "$VENV_STAGE" "$OUT/mkosi.extra/opt/router/venv"

IMG_VENV="$OUT/mkosi.extra/opt/router/venv"
grep -rlZ --binary-files=without-match "$VENV_STAGE" "$IMG_VENV" 2>/dev/null \
  | xargs -0 -r sed -i "s|$VENV_STAGE|/opt/router/venv|g"
find "$IMG_VENV" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
"$VENV_STAGE/bin/python" -m compileall -q -f --invalidation-mode checked-hash \
  -d /opt/router/venv "$IMG_VENV" >/dev/null 2>&1 || true

if grep -rl "$VENV_STAGE" "$IMG_VENV" >/dev/null 2>&1; then
  echo "FATAL: build path still embedded in the image venv -> RTMR1 would not be reproducible"
  grep -rl "$VENV_STAGE" "$IMG_VENV" | head; exit 1
fi

# --- the frozen encoder, baked in ---------------------------------------------------------------
HFC="$OUT/mkosi.extra/opt/router/hf"
mkdir -p "$HFC"
HF_HOME="$HFC" "$VENV_STAGE/bin/python" - <<'ENCPY' >/dev/null 2>&1 || {
from sentence_transformers import SentenceTransformer

from thirtyspokes.koth.harness import EMBED_DIM, ENCODER
m = SentenceTransformer(ENCODER)
v = m.encode(["warm the cache and prove the dimension"], convert_to_numpy=True,
             normalize_embeddings=True)
assert v.shape[1] == EMBED_DIM, f"encoder gives {v.shape[1]}, harness expects {EMBED_DIM}"
ENCPY
    echo "FATAL: could not pre-populate the routing encoder cache (needs network here)"; exit 1; }

# Cache junk that varies per build: lock filenames embed the absolute path, xet logs carry a
# timestamp and pid, .no_exist markers and __pycache__ vary per run. None is data; each one moves
# the roothash and would reject every rebuild as unapproved.
find "$HFC" -name '*.lock' -delete
rm -rf "$HFC/xet/logs"
find "$HFC" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$HFC" -type d -name '.no_exist' -prune -exec rm -rf {} + 2>/dev/null || true

# --- network ------------------------------------------------------------------------------------
mkdir -p "$OUT/mkosi.extra/etc/systemd/network"
cat > "$OUT/mkosi.extra/etc/systemd/network/20-dhcp.network" <<'EOF'
[Match]
Name=en*
[Network]
DHCP=yes
EOF
cat > "$OUT/mkosi.extra/etc/resolv.conf" <<'EOF'
nameserver 169.254.169.254
nameserver 8.8.8.8
EOF

# --- boot-time secret injection -----------------------------------------------------------------
# The miner supplies an OWNER-ISSUED OpenRouter key through instance metadata. It lands on tmpfs and
# never touches the verity rootfs, which is read-only anyway. The enclave verifies its provenance
# before serving (koth/orkey.py) — metadata is miner-controlled, so arriving here proves nothing.
cat > "$OUT/mkosi.extra/opt/router/fetch-secrets.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
MD='http://169.254.169.254/computeMetadata/v1/instance/attributes'
H='Metadata-Flavor: Google'
mkdir -p /run/router && chmod 700 /run/router
curl -sf -H "$H" "$MD/router-openrouter-key" > /run/router/openrouter.key || {
  echo "ROUTER-SECRETS: no key in metadata"; exit 1; }
curl -sf -H "$H" "$MD/router-owner-account" > /run/router/owner.txt || {
  echo "ROUTER-SECRETS: no owner account in metadata"; exit 1; }
chmod 600 /run/router/openrouter.key
echo "ROUTER-SECRETS ok key_bytes=$(wc -c < /run/router/openrouter.key)"
EOF
chmod +x "$OUT/mkosi.extra/opt/router/fetch-secrets.sh"

# --- boot-time routing-model injection ----------------------------------------------------------
# The miner's model is DATA, injected at runtime, never baked in. That is what keeps one measured
# image able to host every miner: the image is the same for all of them, so they share an MRTD, and
# what differs is a weights blob that reaches the policy through a single numeric call.
cat > "$OUT/mkosi.extra/opt/router/fetch-model.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
MD='http://169.254.169.254/computeMetadata/v1/instance/attributes'
H='Metadata-Flavor: Google'
mkdir -p /run/router/model
if curl -sf -H "$H" "$MD/router-weights" > /run/router/model/weights.b64 2>/dev/null; then
  base64 -d /run/router/model/weights.b64 > /run/router/model/weights.npz
  echo "ROUTER-MODEL ok bytes=$(wc -c < /run/router/model/weights.npz)"
else
  # No model is a legitimate configuration: the enclave then serves the cheapest rung — the
  # always-cheapest floor every v3 measurement is scored against (docs/WHITEPAPER.md §5.1b).
  echo "ROUTER-MODEL none — serving the cheapest-rung baseline"
fi
EOF
chmod +x "$OUT/mkosi.extra/opt/router/fetch-model.sh"

cat > "$OUT/mkosi.extra/opt/router/start.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
export HF_HOME=/opt/router/hf
export HF_HUB_OFFLINE=1                 # the encoder is measured; fetching one at boot is not
export OPENROUTER_API_KEY="$(cat /run/router/openrouter.key)"
export THIRTYSPOKES_OWNER_ACCOUNT="$(cat /run/router/owner.txt)"
ARGS=(--host 0.0.0.0 --port 8080 --epoch "${ROUTER_EPOCH:-1}")
[ -f /run/router/model/weights.npz ] && ARGS+=(--weights /run/router/model/weights.npz)
exec /opt/router/venv/bin/orchestra-enclave "${ARGS[@]}"
EOF
chmod +x "$OUT/mkosi.extra/opt/router/start.sh"

# --- read the image's own measurement out, before anything else runs -----------------------------
# This image has no shell and no ssh, which is the point. An operator still has to learn its MRTD to
# approve it, and a user has to be able to confirm what they are pinning, so the measurement goes to
# the serial console at boot where `gcloud compute instances get-serial-port-output` can read it.
# Printed IN FULL: enclave_cli truncates it for a log line, and a truncated measurement cannot be
# pasted into a governance record.
cat > "$OUT/mkosi.extra/opt/router/measure.py" <<'EOF'
import sys

sys.path.insert(0, "/opt/router/venv/lib/python3.12/site-packages")
from thirtyspokes.koth import tdx

if not tdx.tdx_available():
    print("ROUTER-MEASURE: no TDX on this host — this image is not attesting")
    raise SystemExit(0)
q = tdx.parse_quote(tdx.get_quote(b"\x00" * 64))
print("ROUTER-MRTD:", q.mr_td.hex())
for i, r in enumerate(q.rtmrs):
    print(f"ROUTER-RTMR{i}:", r.hex())
EOF

cat > "$OUT/mkosi.extra/etc/systemd/system/router-measure.service" <<'EOF'
[Unit]
Description=Print this image's TDX measurements to the serial console
Before=router.service
[Service]
Type=oneshot
StandardOutput=journal+console
ExecStart=/opt/router/venv/bin/python /opt/router/measure.py
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
ln -sf ../router-measure.service \
  "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants/router-measure.service"

for u in router-secrets router-model; do
cat > "$OUT/mkosi.extra/etc/systemd/system/$u.service" <<EOF
[Unit]
Description=ThirtySpokes router boot-time ${u#router-} injection (CVM metadata -> tmpfs)
After=network-online.target
Wants=network-online.target
Before=router.service
[Service]
Type=oneshot
StandardOutput=journal+console
ExecStart=/opt/router/${u/router-/fetch-}.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
ln -sf ../$u.service "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants/$u.service"
done

# Restart=always, unlike the KOTH one-shot: this serves user requests, and a crash that left the
# endpoint dead would look to a validator exactly like a miner that went offline. A restart mints a
# FRESH enclave key and a fresh quote — clients re-fetch and re-verify, which is correct, because a
# restart could equally have been an image change.
cat > "$OUT/mkosi.extra/etc/systemd/system/router.service" <<'EOF'
[Unit]
Description=ThirtySpokes confidential router (in-enclave)
After=router-secrets.service router-model.service network-online.target
Wants=router-secrets.service router-model.service
[Service]
Type=simple
StandardOutput=journal+console
ExecStart=/opt/router/start.sh
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
ln -sf ../router.service "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants/router.service"

echo "=== mkosi build (verity) ==="
cd "$OUT"
mkosi --force build 2>&1 | tail -30
# GCE UEFI starts the removable-media fallback path; put the self-contained UKI there.
mcopy -o -i "$OUT/thirtyspokes-router.raw@@1048576" "$OUT/thirtyspokes-router.efi" \
  ::/EFI/BOOT/BOOTX64.EFI
echo "=== artifacts ==="; ls -la "$OUT"/*.raw* "$OUT"/*.efi 2>/dev/null
echo
echo "NEXT: boot this on a TDX host, read the MRTD with tdx.self_mrtd(), and approve it with"
echo "  orchestra-serving-governance --mrtd <value> --owner-key <seed> --out approved.json"
