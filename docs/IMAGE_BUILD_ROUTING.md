# Preparing the routing image (Phase 3)

*Everything below is ready to run. It is **not** run: the build needs network, ~30 GB of disk and a
GCP TDX instance to boot on, and the measurements it produces are only meaningful if they come from
the real hardware. This is the checklist and what to watch for at each step.*

## What changed, and why every miner's evidence resets

| | before | now |
|---|---|---|
| `HARNESS_VERSION` | `koth-harness-1` | **`koth-harness-2`** |
| in `runtime_measurement()` | runtime + suite | runtime + suite + **harness** |
| routing pool | not pinned anywhere | `harness.ROUTING_POOL`, 7 rungs |
| embeddings | raw float | rounded to `EMBED_DECIMALS=6` |
| encoder in image | absent | `all-MiniLM-L6-v2` baked under dm-verity |

New measurement to approve on-chain:

```
c8c5d2ffddf5ae57464b37ecc756ea1cc87bb379b084f49355f8a50baabb401f
```

The harness version is now genuinely inside the measurement. It was *documented* as being there and
was not — so before this, changing the routing engine would not have changed RTMR3, and miners could
have been scored under an engine the owner never approved. Folding it in is what makes "which engine
ran" a hardware fact, and it is why every accumulated-evidence record resets: correct behaviour for
an engine change, not a migration to route around.

## Build

```bash
# needs network, ~30 GB free, and root for mkosi
sudo scripts/build_koth_image_prod.sh
```

### What to check, in order

**1. The encoder actually got baked.** The recipe fails loudly if the download did not happen, but
confirm the size — if `hf/` did not grow by ~90 MB the model is not there and every routing run will
fail at boot with `HF_HUB_OFFLINE=1`.

**2. Image size.** Expect roughly **+300 MB**: ~200 MB CPU-only torch, ~50 MB transformers, ~90 MB
model. The recipe pins the CPU wheel explicitly *before* `sentence-transformers`, because the default
resolve pulls the **1.2 GB CUDA build** — GPU runtime this image can never use, hashed into RTMR1.
If the image jumped by more than a gigabyte, that pin failed.

**3. Reproducibility — build TWICE, in DIFFERENT directories.** This has bitten three times already
(venv shebangs, `.pyc` `co_filename`, pip `RECORD`), and torch adds a large new surface:

```bash
BUILD_DIR=/var/tmp/repro-A sudo scripts/build_koth_image_prod.sh
BUILD_DIR=/var/tmp/repro-B sudo scripts/build_koth_image_prod.sh
sha256sum /var/tmp/repro-{A,B}/out/koth.efi        # must be IDENTICAL
```

Compare the **UKI and roothash**, never `koth-runtime.raw` — its ESP differs by ~14 unmeasured bytes
every build. A mismatch means something wrote a path, a timestamp or a pid into the rootfs; the
recipe already scrubs `.lock`, `xet/logs`, `__pycache__` and `.no_exist`, so a new mismatch is a new
leak and needs finding rather than retrying.

**4. Boot on TDX and capture the measurements.**

```bash
gcloud compute instances create koth-img-v18 --machine-type c3-standard-4 \
  --confidential-compute-type=TDX --maintenance-policy=TERMINATE ...
# from the serial console:
MRTD=...  RTMR1=...  RTMR3=...
```

RTMR2 must stay zero — the UKI boots directly as `/EFI/BOOT/BOOTX64.EFI` because systemd-boot does
not start on GCP TDVF, so there is no GRUB command stream. A non-zero RTMR2 means the boot path
changed and the pinned values are wrong.

**5. Publish governance.**

```bash
orchestra-koth-owner approve-measurement --measurement c8c5d2ff... \
  --mrtd $MRTD --rtmr1 $RTMR1 --rtmr3 $RTMR3
```

**6. Smoke the enforcing path** — `scripts/koth_enforce_smoke.py` must accept a proof from the pinned
image and reject wrong-image, mock and tampered variants.

## The residual risk I could not remove

`harness.encode` promises byte-identical embeddings in three places: the miner's enclave, the owner's
reference build, and the trainer a miner runs at home. **Torch CPU kernels are not bit-reproducible
across microarchitectures** — an AVX-512 CVM and a miner's laptop can differ in the last bits of
every component.

`EMBED_DECIMALS=6` rounds far above that noise, which makes the three copies agree in practice and
costs nothing semantically. It does **not** make the promise literally true: a value landing exactly
on a rounding boundary can still differ, and a near-tie in the head's output can still flip. The
effect is rare and shows up as a miner's local dev-kit result occasionally disagreeing with their
attested run. Removing it entirely would mean replacing torch with a fixed-point or ONNX encoder,
which is a real option if miners report it as a problem — and a much larger change than this one.
