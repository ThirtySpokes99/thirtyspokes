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

**2. Image size — MEASURED, and my first estimate was wrong.** I predicted "+300 MB, ~200 MB torch".
The real CPU-only torch is **750 MB**, so the true addition is ~840 MB and the finished image is
**2.8 GB**. Recorded because the prediction was off by 2.5x and the runbook should carry the measured
number, not the guess.

The CPU pin still held, which is the thing that actually matters: the recipe installs CPU-only torch
explicitly *before* `sentence-transformers`, because the default resolve pulls the **1.2 GB CUDA
build** — GPU runtime this image can never use, every byte hashed into RTMR1. Verify the pin by its
signature rather than by total size: **`site-packages/nvidia/` must not exist**. If it does, the pin
failed and the image carries a CUDA stack it will never run.

**3. Reproducibility — build TWICE, in DIFFERENT directories.** This has bitten three times already
(venv shebangs, `.pyc` `co_filename`, pip `RECORD`), and torch adds a large new surface:

```bash
OUT=/var/tmp/repro-A bash scripts/build_koth_image_prod.sh
OUT=/var/tmp/repro-B bash scripts/build_koth_image_prod.sh
sha256sum /var/tmp/repro-{A,B}/koth-runtime.efi     # must be IDENTICAL
```

**Run 2026-07-29 — PASSED.** Both builds produced

```
516ab1799edd7551f9613ca98292a9ddee4609b3affb100b072ed0c34d0298bf  koth-runtime.efi
```

so torch's ~750 MB of new binary surface carries no build-path, timestamp or pid into the rootfs.
(The variable is `OUT`, not `BUILD_DIR` — an earlier draft of this runbook named the wrong one.)

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

### Captured 2026-07-29 — GCP C3, Intel TDX, us-central1-a

| | |
|---|---|
| UKI sha256 | `516ab1799edd7551f9613ca98292a9ddee4609b3affb100b072ed0c34d0298bf` |
| verity roothash | `b4ae50b9ecfb1adb9877e41ac8bef9bd2d00aa7e2415953af9b784ba6cff00a5` |
| runtime measurement | `c8c5d2ffddf5ae57464b37ecc756ea1cc87bb379b084f49355f8a50baabb401f` |
| MRTD | `c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5` |
| RTMR0 | `c49d22aff6edb37cb6178defb05e0e2b512c26960e6ee73b1ea303365a31def807ab2ad71e5874236feca2ca552c6307` |
| **RTMR1** | `f1af6c815e19b58ae2ae3937514f2a7c903fe5f502627ec584e5b6d6a00018286747391dd39f70bd8b0e0dc0a4949f91` |
| RTMR2 | `000…0` ✅ zero — direct-UKI boot, no GRUB command stream |
| RTMR3 | `80f37f6209cb2457933332875c187528dd6984289f5d2848028dbb88072de99419af4e1f959d8c053e3617475994a044` |
| dm-verity | `True` |

All three expectations held:

* **RTMR1 CHANGED** — `f1af6c81…`, against the previous image's `515b759e…`. It is the per-image
  anchor, and the harness, encoder and pool all changed, so a value that had *not* moved would have
  meant the new engine was not actually measured.
* **RTMR2 is zero**, confirming the UKI boots directly as `/EFI/BOOT/BOOTX64.EFI` — systemd-boot does
  not start on GCP TDVF, so there is no GRUB command stream to measure.
* **MRTD is unchanged** at `c1ee9c16…`. Expected: it is Google's TDVF firmware, identical on every GCP
  TDX guest, so it is a coarse "genuine GCP TDX" check and **never** a per-image discriminator.

RTMR3 (`80f37f62…`) binds runtime + suite + **pool**, so it is only reproducible by a guest booting
this image against the same 7-rung `ROUTING_POOL`. Booting with any other pool string yields a
different RTMR3 that no honest miner can match — which is why the measurement boot passed the pinned
pool explicitly.

**5. Publish governance.**

```bash
# NO --measurement and NO --rtmr3: the CLI DERIVES both. `runtime_measurement()` is computed from
# the code you are running, and RTMR3 from `owner_expected_rtmr3(measurement, suite, pool)`. An
# earlier draft of this runbook invented those two flags; they do not exist and the command failed.
# Deriving them is the safer design — the owner cannot publish a measurement that disagrees with
# the code, only one that disagrees with the HARDWARE, which is what the pre-flight below catches.
orchestra-koth-owner \
  --netuid 526 --network test --wallet minirouter --hotkey owner-hotkey \
  --mrtd c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5 \
  --rtmr1 f1af6c815e19b58ae2ae3937514f2a7c903fe5f502627ec584e5b6d6a00018286747391dd39f70bd8b0e0dc0a4949f91 \
  --rtmr2 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 \
  --pool "qwen/qwen3.7-flash,deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro,z-ai/glm-5.2,openai/gpt-5.6-luna,google/gemini-3.6-flash,moonshotai/kimi-k3"
```

**PRE-FLIGHT, do this first.** The owner publishes a *computed* RTMR3; the guest produces a *measured*
one. If they differ, governance rejects every honest miner and the failure looks like the miners are
at fault. Verified 2026-07-29 — computed == measured == `80f37f62…`:

```bash
python -c "
from thirtyspokes.koth.rtmr import owner_expected_rtmr3
from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement
print(owner_expected_rtmr3(runtime_measurement=runtime_measurement(),
      suite_version=SUITE_VERSION, pool_allow_list=POOL))   # must equal the captured RTMR3
```

### Published 2026-07-29 — testnet 526

```
on-chain digest : 9ba743457bdbcedf8b49bbde2a206f78423410e74a2d121e582a04d71862a989
measurement     : c8c5d2ffddf5ae57464b37ecc756ea1cc87bb379b084f49355f8a50baabb401f
rtmr1 / rtmr3   : f1af6c81… / 80f37f62…        (hardware-captured; rtmr3 computed == measured)
pool            : 7 models
```

**The publish appeared to hang for 35 minutes and had actually already succeeded.** `huggingface_hub`
leaves non-daemon uploader threads behind, so the process blocked in a futex at shutdown with its
output still buffered in the pipe — a completed publish that looks identical to a dead one. The retry
only revealed it by returning `Transaction Already Imported`. Fixed in `owner.py` (staged flushed
prints + `os._exit(0)`, the same fix `reference.py` already carried), but if you are on an older
build: **check the chain before re-running.** `chain.governance_digest()` tells you whether it landed;
re-running a publish that already succeeded is harmless but the 35 minutes are not.

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
