#!/usr/bin/env bash
# KOTH measured image — ITERATION 2: dm-verity read-only rootfs.
# The verity roothash is baked into the UKI cmdline -> measured into RTMR1 -> the ROOTFS is locked
# (a miner can't swap /opt/koth and keep the same RTMR1). Uses mkosi's own verity-capable initrd
# (systemd-veritysetup + gpt-auto), libstdc++6 in-image (numpy), KernelCommandLineExtra so mkosi's
# auto roothash/root wiring stays. Boot fix (from iter-1): copy the UKI to /EFI/BOOT/BOOTX64.EFI.
set -euo pipefail
export PATH="$PATH:/root/.local/bin"

OUT=${OUT:-/root/koth-build-prod}
WHEEL=${WHEEL:-$(ls /root/koth-build/wheels/thirtyspokes-*.whl | head -1)}
KOTH_SUITE=${KOTH_SUITE:-real}
SUPPORT_MATRIX=${SUPPORT_MATRIX:-}
if [[ "$KOTH_SUITE" != "real" && "$KOTH_SUITE" != "support" ]]; then
  echo "FATAL: KOTH_SUITE must be real or support"; exit 1
fi
if [[ "$KOTH_SUITE" == "support" && ! -f "$SUPPORT_MATRIX" ]]; then
  echo "FATAL: SUPPORT_MATRIX must point to the collected support JSONL"; exit 1
fi
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
# util-linux ships `unshare` — REQUIRED for the no-egress confinement (koth/confine.py); the minimal
# image doesn't pull it reliably, so pin it. As root the runtime confines without an unprivileged
# userns (see confine._argv), so no userns sysctl is needed.
Packages=linux-image-${KVER},systemd,systemd-boot,systemd-sysv,udev,dbus,ca-certificates,curl,python3,python3-venv,libpython3.12t64,libstdc++6,nvme-cli,util-linux
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

# self-contained venv (host==image x86_64/py3.12); copied via extra tree.
#
# THE STAGING PATH MUST BE CONSTANT ACROSS BUILDS, and it must not be a host runtime. It was
# `$OUT/.venv-stage`, which satisfied the second requirement and broke the first: `python -m venv`
# and pip bake their own absolute location into three separate places, and each one leaked the build
# directory into the rootfs -> the verity roothash -> RTMR1. Rewriting them afterwards is not enough,
# because pip hashes the console scripts INTO `RECORD` at install time, before any rewrite can run —
# so the recorded sha256 of `bin/f2py` et al still differed per build directory (measured: 24 RECORD
# files differing, roothash 54f37385 vs 00bfa86c, after the shebang and bytecode fixes had already
# landed). A fixed scratch path outside /opt/koth satisfies both requirements at once.
VENV_STAGE="${VENV_STAGE:-/var/tmp/koth-venv-stage}"
rm -rf "$VENV_STAGE"
python3 -m venv --copies "$VENV_STAGE"
# openai: the pool client (OpenRouterBackend). datasets: the real MMLU/GSM8K loaders.
# sentence-transformers: the ROUTING HARNESS's frozen encoder. A routing head's only view of a task
# is this embedding, so it has to run inside the measured image — a miner cannot supply it and the
# runtime cannot fetch it at boot (HF_HUB_OFFLINE=1, and anything unmeasured is outside RTMR1).
#
# CPU-ONLY TORCH, EXPLICITLY. The default wheel is the CUDA build: ~1.2 GB of GPU runtime this image
# can never use, all of it inside dm-verity and hashed into RTMR1. The CPU wheel is ~200 MB.
# Installed FIRST so the sentence-transformers resolve cannot pull the CUDA one back in.
"$VENV_STAGE/bin/pip" install -q --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch
"$VENV_STAGE/bin/pip" install -q --no-cache-dir "$WHEEL" 'cryptography>=42' 'dcap-qvl>=0.5' numpy \
  openai datasets sentence-transformers
cp -a "$VENV_STAGE" "$OUT/mkosi.extra/opt/koth/venv"

# REWRITE THE STAGING PATH OUT OF THE IMAGE COPY. `python -m venv` bakes its own absolute location
# into pyvenv.cfg and into the shebang of every console script, and staging moved from the fixed
# /opt/koth/venv to $OUT/.venv-stage — so the build directory leaked into the rootfs. That broke two
# things at once:
#   * REPRODUCIBILITY. The path is part of the rootfs -> the verity roothash -> RTMR1. Two builds in
#     different directories produced different RTMR1s (measured: 533c80d7… vs 2d41ca85…), so a miner
#     who rebuilt this recipe anywhere but the owner's exact directory was rejected
#     `unapproved_runtime`. That is the whole point of the recipe being reproducible.
#   * THE ENTRY POINTS. Every /opt/koth/venv/bin/orchestra-* in the image began
#     `#!/root/koth-build-prod/.venv-stage/bin/python3` — an interpreter that does not exist inside
#     the image. Only the direct `venv/bin/python <script>` invocations worked (bin/python is a
#     copied BINARY, not a script), which is why this survived unnoticed.
# The copy is what ships, so rewrite it to its real in-image location and leave $VENV_STAGE working
# (the HF cache below is populated with it).
IMG_VENV="$OUT/mkosi.extra/opt/koth/venv"
grep -rlZ --binary-files=without-match "$VENV_STAGE" "$IMG_VENV" 2>/dev/null \
  | xargs -0 -r sed -i "s|$VENV_STAGE|/opt/koth/venv|g"

# ...and the same path is baked into every .pyc, which the text rewrite above cannot touch. Python
# stores the compile-time source path in each code object (`co_filename`), so ~10k bytecode files
# each carried the build directory. The HEADERS were already fine — pip emits hash-based pycs under
# SOURCE_DATE_EPOCH — but the BODIES differed, and that alone changed the roothash (measured:
# 1ff8ad2d vs 91f2eff1 after the text fix). Recompiling with `-d /opt/koth/venv` sets `co_filename`
# to the path the code will actually live at: constant across builds, and correct in tracebacks.
find "$IMG_VENV" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
"$VENV_STAGE/bin/python" -m compileall -q -f --invalidation-mode checked-hash \
  -d /opt/koth/venv "$IMG_VENV" >/dev/null 2>&1 || true

# Fail the build rather than ship a non-reproducible image: any surviving reference means the
# rewrite missed a file, and the failure mode is silent (a valid-looking image with a wrong RTMR1).
# Checked over TEXT AND BINARY — the bytecode leak above is exactly what a text-only check misses.
if grep -rl "$VENV_STAGE" "$IMG_VENV" >/dev/null 2>&1; then
  echo "FATAL: build path still embedded in the image venv -> RTMR1 would not be reproducible"
  grep -rl "$VENV_STAGE" "$IMG_VENV" | head; exit 1
fi

# BAKE THE BENCHMARK DATA INTO THE IMAGE. Loading the suite from the Hub at boot made the run
# network-dependent and non-deterministic (it failed on hardware), and left the questions OUTSIDE the
# measured rootfs. Baking the HF cache puts the data under dm-verity -> covered by RTMR1 -> pinned,
# and the runtime loads it offline. Populate the cache with EXACTLY what real_suite() loads.
# Use the image's OWN venv (built above): it has thirtyspokes + datasets, so the cache is populated by
# exactly the code that will read it at boot. The system python3 has neither.
HFC="$OUT/mkosi.extra/opt/koth/hf"
mkdir -p "$HFC"
printf '%s\n' "$KOTH_SUITE" > "$OUT/mkosi.extra/opt/koth/suite.txt"
if [[ "$KOTH_SUITE" == "support" ]]; then
  # The owner suite must exclude every row used to fit the injected router. Bake that exact index
  # set under dm-verity, so neither a miner nor boot metadata can choose a flattering overlap.
  "$VENV_STAGE/bin/python" - "$SUPPORT_MATRIX" \
      "$OUT/mkosi.extra/opt/koth/support_exclude.json" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
indices = sorted({int(row["source_index"]) for row in rows})
if len(indices) != len(rows):
    raise SystemExit("support matrix source_index values are missing or duplicated")
open(sys.argv[2], "w").write(json.dumps(indices, separators=(",", ":")))
PY
  HF_HOME="$HFC" "$VENV_STAGE/bin/python" - \
      "$OUT/mkosi.extra/opt/koth/support_exclude.json" <<'PY' >/dev/null 2>&1 || {
import json, sys
from thirtyspokes.koth.support import load_bitext_support_benchmark
excluded = frozenset(json.load(open(sys.argv[1])))
load_bitext_support_benchmark(exclude_source_indices=excluded)
PY
    echo "FATAL: could not pre-populate the Bitext support cache (need network here)"; exit 1; }
else
  HF_HOME="$HFC" "$VENV_STAGE/bin/python" -c \
    "from thirtyspokes.koth.benchmarks import real_suite; real_suite()" >/dev/null 2>&1 || {
    echo "FATAL: could not pre-populate the benchmark cache (need network here)"; exit 1; }
fi

# BAKE THE ROUTING ENCODER. Same argument as the benchmark data: fetched at boot it would be
# network-dependent, unmeasured, and free to change under the subnet. Baked, it sits under dm-verity
# and is pinned by RTMR1, so "which encoder ran" becomes a hardware fact. Downloaded through the
# image's OWN venv so the cache layout is exactly what the runtime will read at boot.
HF_HOME="$HFC" "$VENV_STAGE/bin/python" - <<'ENCPY' >/dev/null 2>&1 || {
from sentence_transformers import SentenceTransformer
from thirtyspokes.koth.harness import ENCODER, EMBED_DIM
m = SentenceTransformer(ENCODER)
v = m.encode(["warm the cache and prove the dimension"], convert_to_numpy=True,
             normalize_embeddings=True)
assert v.shape[1] == EMBED_DIM, f"encoder gives {v.shape[1]}, harness expects {EMBED_DIM}"
ENCPY
    echo "FATAL: could not pre-populate the routing encoder cache (need network here)"; exit 1; }

# Strip the cache's build junk, or the rootfs is NOT reproducible. The dataset bytes themselves are
# deterministic, but `datasets` leaves two things that are not:
#   * .lock files whose FILENAME embeds the absolute cache path (…_root_repro-A_… vs …-B…), so every
#     build directory yields a different filename;
#   * xet/logs/xet_<timestamp>_<pid>.log, stamped with the wall clock and pid.
# Either one changes the rootfs -> the verity roothash -> RTMR1, and a miner who rebuilt this recipe
# would be rejected as `unapproved_runtime`. Neither is data; both are regenerated at need.
find "$HFC" -name '*.lock' -delete
rm -rf "$HFC/xet/logs"
# The encoder cache leaks the same WAY the dataset cache does: HF writes .no_exist marker dirs and
# torch/transformers leave __pycache__, both varying per run. Not data, regenerated on demand, and
# each one changes the rootfs -> the verity roothash -> RTMR1 -> every rebuilt image is rejected.
find "$HFC" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$HFC" -type d -name '.no_exist' -prune -exec rm -rf {} + 2>/dev/null || true

# DHCP for the GCP NIC — the minimal image ships no .network, so networkd never configures the
# interface (metadata unreachable + wait-online times out ~2min). This fixes both.
mkdir -p "$OUT/mkosi.extra/etc/systemd/network"
cat > "$OUT/mkosi.extra/etc/systemd/network/20-dhcp.network" <<'EOF'
[Match]
Name=en* eth*
[Network]
DHCP=yes
EOF

# DNS. The image ships no systemd-resolved and the rootfs is read-only, so NOTHING ever writes
# /etc/resolv.conf: every pool call died with "Temporary failure in name resolution" on hardware.
# (The metadata fetches survived only because 169.254.169.254 is a bare IP.) Bake the resolver in —
# deterministic, and part of the measured rootfs.
cat > "$OUT/mkosi.extra/etc/resolv.conf" <<'EOF'
nameserver 169.254.169.254
nameserver 8.8.8.8
options timeout:2 attempts:3
EOF

cat > "$OUT/mkosi.extra/opt/koth/report.py" <<'EOF'
# Production runtime: load the miner's INJECTED agent (fetched at boot into /run/koth/agent by
# koth-agent.service from CVM metadata) and run it CONFINED. One pinned image (fixed RTMR1) thus
# runs ANY miner's agent; the proof's source_hash/weights_hash bind whichever agent ran.
import base64, glob, gzip, hashlib, os, time
# The benchmark data is baked into the measured rootfs (see the build script), so load it OFFLINE:
# a boot-time Hub download is non-deterministic, needs network, and leaves the questions outside
# what RTMR1 covers. Must be set before `datasets`/`huggingface_hub` are imported.
os.environ.setdefault("HF_HOME", "/opt/koth/hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json, sys, urllib.request
from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.koth import rtmr, tdx
from thirtyspokes.koth.pool import PinnedBackend
from thirtyspokes.koth.runtime import Artifact, KOTHRuntime

MD = "http://169.254.169.254/computeMetadata/v1/instance/attributes"


def meta(key, default=""):
    req = urllib.request.Request("%s/%s" % (MD, key), headers={"Metadata-Flavor": "Google"})
    try:
        return urllib.request.urlopen(req, timeout=10).read().decode().strip()
    except Exception:
        return default


# Secrets were fetched to tmpfs by koth-secrets.service (never baked into the public image). The
# blob is MINER-SUPPLIED, so it is read through an ALLOWLIST: splatting it into the environment let
# a miner set SSL_CERT_FILE/HTTPS_PROXY and MITM its own pool connection, fabricating answers and
# cost behind a genuine attestation. Rejected keys are printed (not their values) so an attempt is
# visible on the attested console rather than silently dropped. See eval/config.py.
_accepted, _rejected = config.load_secrets_env("/run/koth/secrets.env")
_scrubbed = config.scrub_network_env()
print(f"KOTH-SECRETS accepted={sorted(_accepted)} rejected={sorted(_rejected)} "
      f"scrubbed={sorted(_scrubbed)}", flush=True)

epoch = int(meta("koth-epoch", "0"))
nonce = meta("koth-nonce")
hotkey = meta("koth-hotkey")
pool_list = [m for m in meta("koth-pool", "").split(",") if m]
n_per_bench = int(meta("koth-n-per-bench", "8"))
suite_kind = open("/opt/koth/suite.txt").read().strip()

cfg = config.LiveConfig()
if not (cfg.api_key and nonce and hotkey and pool_list):
    print("KOTH-ERROR missing metadata/secrets: need koth-secrets, koth-nonce, koth-hotkey, koth-pool",
          flush=True)
    sys.exit(1)

# the agent's ONLY channel is the metered call_model, restricted to the owner-pinned pool
if suite_kind == "support":
    from thirtyspokes.koth.support import SUPPORT_PRICE_PER_1K
    price_fn = lambda model: SUPPORT_PRICE_PER_1K.get(model, config.price_for(model))
else:
    price_fn = config.price_for
backend = PinnedBackend(
    OpenRouterBackend(cfg.api_key, cfg.base_url, price_fn=price_fn), set(pool_list))
# require_confinement: in the enclave an unconfined agent has network egress and could call
# an off-allow-list model with a key of its own, voiding the pinning and the metering. Refuse
# to run rather than degrade; the actual mode is also stamped into the proof for the validator.
rt = KOTHRuntime(backend, tdx.TDXPlatform(), confine=True, require_confinement=True,
                 confine_timeout=float(meta("koth-confine-timeout", "7200")))
rt.measure_self(pool_list)                                   # bind runtime+suite+pool -> RTMR3

m = rtmr.read_measurements()
for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"):
    print("KOTH-MEASURE %s=%s" % (k, m.get(k, "")), flush=True)
uuids = [open(p).read().strip() for p in glob.glob("/sys/block/dm-*/dm/uuid")]
print("KOTH-MEASURE verity=%s suite=%s pool=%s" % (
    any(u.startswith("CRYPT-VERITY") for u in uuids), suite_kind, ",".join(pool_list)), flush=True)

# the injected agent (NOT baked into the measured image -> RTMR1 is agent-independent)
ap = "/run/koth/agent/source.py"
if os.path.exists(ap) and os.path.getsize(ap) > 0:
    art = Artifact(open(ap).read(), open("/run/koth/agent/weights.bin", "rb").read(), "injected")
    origin = "injected"
else:
    from thirtyspokes.koth.miner import reference_artifact
    art = reference_artifact(pool_list[0]); origin = "fallback"
print("KOTH-ART origin=%s src=%s w=%s" % (origin, art.source_hash, art.weights_hash), flush=True)

if suite_kind == "support":
    from thirtyspokes.koth.support import load_bitext_support_benchmark
    excluded = frozenset(json.load(open("/opt/koth/support_exclude.json")))
    suite = [load_bitext_support_benchmark(exclude_source_indices=excluded)]
else:
    from thirtyspokes.koth.benchmarks import real_suite
    suite = real_suite()
started = time.monotonic()
# DISPATCH ON THE ARTIFACT, exactly as `miner.run_once` does. This script is a SECOND copy of the
# run logic — it lives inside the image, not in the library — and it was still calling rt.run()
# unconditionally long after the routing path was wired everywhere else. The effect was invisible
# and total: a miner would get the router path on its laptop and the agent path on the hardware that
# actually earns, and the image would carry a 750 MB encoder it never touched.
from thirtyspokes.koth.miner import is_routing_artifact
if is_routing_artifact(art):
    from thirtyspokes.koth import harness as H
    # The head emits rung indices into the HARNESS pool. RTMR3 binds the metadata pool_list, so if
    # those two disagree the miner is scored against a ladder its head never saw. Refuse rather than
    # silently route into the wrong action space.
    if sorted(pool_list) != sorted(H.pool_models()):
        raise SystemExit("KOTH-ERROR routing artifact but koth-pool != harness ROUTING_POOL: "
                         "%s vs %s" % (sorted(pool_list), sorted(H.pool_models())))
    print("KOTH-MODE routing harness=%s rungs=%d" % (H.HARNESS_VERSION, len(H.pool_models())),
          flush=True)
    proof, trace = rt.run_router(art.weights, hotkey=hotkey, epoch=epoch, nonce=nonce,
                                 suite=suite, n_per_bench=n_per_bench,
                                 pool=H.pool_models(), price_of=H.price_of)
else:
    print("KOTH-MODE free-agent (legacy)", flush=True)
    proof, trace = rt.run(art, hotkey=hotkey, epoch=epoch, nonce=nonce,
                          suite=suite, n_per_bench=n_per_bench)
run_seconds = time.monotonic() - started
print("KOTH-RUN tasks=%d calls=%d cost=%.5f" % (
    len(proof.results), proof.n_calls, proof.total_cost_usd), flush=True)
print("KOTH-TIMING run_seconds=%.6f" % run_seconds, flush=True)

# Compress each payload before chunking, authenticate every chunk, and emit three copies. Kernel
# audit output can interleave with a serial write; repeated digest-checked copies make that a
# recoverable transport event while remaining far below GCE's retained-byte limit.
def emit(kind, payload, size=3000):
    encoded = base64.b64encode(gzip.compress(payload.encode(), compresslevel=9, mtime=0)).decode()
    chunks = [encoded[i:i + size] for i in range(0, len(encoded), size)]
    for i, chunk in enumerate(chunks, 1):
        digest = hashlib.sha256(chunk.encode()).hexdigest()
        for _copy in range(3):
            print("KOTH-%s-GZIP-CHUNK %d/%d sha256=%s %s" % (
                kind, i, len(chunks), digest, chunk), flush=True)

emit("PROOF", proof.to_json())
emit("TRACE", json.dumps(trace))
print("KOTH-DONE", flush=True)
EOF

# --- boot-time SECRET injection (metadata koth-secrets -> tmpfs; the miner's OpenRouter key etc.) ---
cat > "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /run/koth && chmod 700 /run/koth
curl -fsS -H 'Metadata-Flavor: Google' \
  http://169.254.169.254/computeMetadata/v1/instance/attributes/koth-secrets \
  > /run/koth/secrets.env 2>/dev/null || echo "# no koth-secrets metadata" > /run/koth/secrets.env
chmod 600 /run/koth/secrets.env
EOF
chmod +x "$OUT/mkosi.extra/opt/koth/fetch-secrets.sh"

# --- boot-time AGENT injection (metadata koth-agent-{source,weights} b64 -> /run/koth/agent) ---
cat > "$OUT/mkosi.extra/opt/koth/fetch-agent.sh" <<'EOF'
#!/usr/bin/env bash
mkdir -p /run/koth/agent
MD='http://169.254.169.254/computeMetadata/v1/instance/attributes'
echo "KOTH-AGENT curl=$(command -v curl) base64=$(command -v base64)"
curl -fsS -H 'Metadata-Flavor: Google' "$MD/koth-agent-source"  -o /run/koth/agent/source.b64
echo "KOTH-AGENT src.b64 curl_rc=$? bytes=$(wc -c < /run/koth/agent/source.b64 2>/dev/null)"
base64 -d /run/koth/agent/source.b64 > /run/koth/agent/source.py
echo "KOTH-AGENT decode_rc=$? py_bytes=$(wc -c < /run/koth/agent/source.py 2>/dev/null)"
curl -fsS -H 'Metadata-Flavor: Google' "$MD/koth-agent-weights" -o /run/koth/agent/weights.b64
base64 -d /run/koth/agent/weights.b64 > /run/koth/agent/weights.bin
echo "KOTH-AGENT done w_bytes=$(wc -c < /run/koth/agent/weights.bin 2>/dev/null)"
EOF
chmod +x "$OUT/mkosi.extra/opt/koth/fetch-agent.sh"

for u in koth-secrets koth-agent; do
cat > "$OUT/mkosi.extra/etc/systemd/system/$u.service" <<EOF
[Unit]
Description=KOTH boot-time $u injection (CVM metadata -> tmpfs)
After=network-online.target
Wants=network-online.target
Before=koth-report.service
[Service]
Type=oneshot
StandardOutput=journal+console
ExecStart=/opt/koth/${u/koth-/fetch-}.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
ln -sf ../$u.service "$OUT/mkosi.extra/etc/systemd/system/multi-user.target.wants/$u.service"
done

cat > "$OUT/mkosi.extra/etc/systemd/system/koth-report.service" <<'EOF'
[Unit]
Description=KOTH runtime — run the injected agent, attest a proof, emit it on the serial console
After=koth-agent.service koth-secrets.service network-online.target
Wants=koth-agent.service koth-secrets.service
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
# GCE's UEFI firmware starts the removable-media fallback path. mkosi puts systemd-boot there and
# exports the self-contained UKI separately; booting systemd-boot did not discover that UKI on the
# measured disk. Replace the fallback binary with the UKI, as required by the proven v14-v16 images.
# The deterministic repart layout starts the ESP at sector 2048 (1 MiB).
mcopy -o -i "$OUT/koth-runtime.raw@@1048576" "$OUT/koth-runtime.efi" \
  ::/EFI/BOOT/BOOTX64.EFI
echo "=== artifacts ==="; ls -la "$OUT"/*.raw* 2>/dev/null
