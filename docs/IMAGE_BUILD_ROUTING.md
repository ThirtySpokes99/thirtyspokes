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

### Captured 2026-07-29 — v18, SUPERSEDED (see v19 below)

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

### v19 — the shipped image (supersedes v18)

v18 was built, hardware-measured and had governance published for it **while being incapable of
running a routing model.** The image embeds its own runner as a heredoc in the build script, and that
copy still called `rt.run()` unconditionally long after the routing path was wired into miner.py,
devkit.py, validator.py and trainer.py. Nothing tests shell-embedded Python, so it passed every check
that existed. The tell was in v18's own serial log — `KOTH-ART origin=fallback` and no mode line at
all — and it was read past.

v19 dispatches on the artifact exactly as `miner.run_once` does, so **both** paths work: routing heads
through `run_router`, legacy free-agent bundles through `rt.run`. It also refuses to start when the
injected `koth-pool` disagrees with the harness `ROUTING_POOL`, because RTMR3 binds the metadata pool
while the head emits into the harness pool — disagreement means scoring a head against a ladder it
never saw. And it prints `KOTH-MODE`, so a run that cannot say which engine it used is visibly broken.

| | |
|---|---|
| UKI sha256 | `478138cb33db3dd8fbd23145c80a782ea1dd4e7f5d24f57a27505831a2f0f154` (reproducible, 2 dirs) |
| MRTD | `c1ee9c16…` unchanged (GCP TDVF) |
| **RTMR1** | `f208520a724ba67be76a35e3fc080a805d526daa381cd336932e82d1968b4b890a71c77d2fb7595f6f22645de40b1558` |
| RTMR2 | `000…0` |
| RTMR3 | `80f37f62…` **unchanged from v18 — and that is correct**: RTMR3 binds runtime+suite+pool, none of which moved. Only the image's runner changed, which moves RTMR1. An RTMR3 that had also moved would mean the harness contract shifted when it should not have |

**Hardware-verified routing run** (`koth-router-v19`, GCP C3 TDX): a 6,279-param head injected via CVM
metadata produced `KOTH-MODE routing harness=koth-harness-2 rungs=7`, `KOTH-ART origin=injected`,
then `KOTH-RUN tasks=3 calls=3 cost=$0.00777` in 207s. Three tasks, three calls — the head chose a
rung and the verifier accepted first try each time, so nothing escalated.

### Published 2026-07-29 — testnet 526

```
on-chain digest : a65b5f8cc6025674dccb84fb409e0a76365a8ea72ae680d6ef4939b63c8eb191
measurement     : c8c5d2ffddf5ae57464b37ecc756ea1cc87bb379b084f49355f8a50baabb401f
rtmr1 / rtmr3   : f208520a… / 80f37f62…        (v19; rtmr3 computed == measured)
pool            : 7 models
```

The earlier v18 record (`9ba74345…`, rtmr1 `f1af6c81…`) is superseded and pinned an image that could
not run routing models.

**The publish appeared to hang for 35 minutes and had actually already succeeded.** `huggingface_hub`
leaves non-daemon uploader threads behind, so the process blocked in a futex at shutdown with its
output still buffered in the pipe — a completed publish that looks identical to a dead one. The retry
only revealed it by returning `Transaction Already Imported`. Fixed in `owner.py` (staged flushed
prints + `os._exit(0)`, the same fix `reference.py` already carried), but if you are on an older
build: **check the chain before re-running.** `chain.governance_digest()` tells you whether it landed;
re-running a publish that already succeeded is harmless but the 35 minutes are not.

**6. Smoke the enforcing path** — `scripts/koth_enforce_smoke.py`, run **inside a TDX guest** (the
measured image has no shell, so use a stock Ubuntu CVM with the wheel installed).

**Run 2026-07-29 on a GCP C3 — ALL CHECKS PASSED:**

```
[2] RTMR3 self-measure == owner-expected                MATCH
[3] quote vs sysfs, all 5 measurements                  MATCH
[4] enforce=True, fully pinned      valid=True  score=0.526
[5] 8 fail-closed paths, each rejecting for the right reason:
    wrong image (RTMR2)  -> unapproved_runtime      MRTD not approved  -> unapproved_runtime
    MRTD gate unset      -> mrtd_gate_unset         RTMR gate unset    -> rtmr_gate_unset
    RTMR gate incomplete -> rtmr_gate_unset         TCB policy unset   -> tcb_policy_unset
    mock TEE quote       -> mock_quote_rejected     tampered (cost=0)  -> report_data_mismatch
```

The first run **failed its own accept step** with `unconfined_agent`: the smoke built its runtime
without `confine=True`, while production uses `confine=True, require_confinement=True` and
`enforce=True` rejects any proof whose attested `confined` flag is false. The script predated the
confinement gate and nothing re-ran it on hardware afterwards, so a check documented as "the full
enforce gate" could never have passed. Fixed to mirror production.

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


---

## v20 — SHIPPED (2026-07-29)

The published image. v18 could not run routing models; v19 could, but its manifest named a
`recipe_commit` that would **not** rebuild it, because `miner.py` changed after the build. v20 was
built from a clean tree at the commit it claims.

| | |
|---|---|
| bucket | `runtime/v20/` in `thirtyspokes/cvm-runtime-image` |
| image sha256 | `cbf0936ddef265112591835b30215ca69ce643dcf92249bb139b7b1db90d9e26` |
| UKI sha256 | `1b49dec7ae0098705b03318b6ff0cab44628e73b7fcb02fa550b25a5ce901a4d` (reproducible, 2 dirs) |
| verity roothash | `4d776060b394e58b551feda9b71e77738b43818301168104f8182458272a6dd4` |
| recipe commit | `25f50c6816e565a4be62982552b3efcc9fec4f07` |
| **RTMR1** | `76947191ea7fde9f24fe498b17592f97ba0960cd95103ec77c876220be9cbf7d0d1e5c2da05a495f27bf953874425e6b` |
| RTMR3 | `80f37f62…` unchanged across v18→v20, correctly: it binds runtime+suite+pool, and only the image changed |
| on-chain governance | `b024c4161a1bcf41529c8f4aa92658152901902ca78853ef4f335ec62d097771` (testnet 526) |

Hardware-verified: `KOTH-MODE routing harness=koth-harness-2 rungs=7`, `KOTH-ART origin=injected`.

### The chain anyone can check without trusting the publisher

```
recipe_commit 25f50c6 → rebuild → UKI 1b49dec7… → boot on TDX → RTMR1 76947191…
                                                                 ↕ must match
                                    on-chain governance b024c416…
```

### Publish LAST, and the guard that now enforces it

Three images were built this session and two were wrong, both for the same reason: **the tree kept
moving after the build.** An image is a snapshot; every commit under `src/` invalidates it, and
nothing in the process objected.

`publish_runtime_image.py` now **refuses to publish from a dirty tree**, because the failure is
mechanical and should not depend on someone remembering to compare wheel hashes. It does not yet
verify that the baked wheel matches a fresh build of HEAD — that is the stronger check and the one
that actually caught v19; worth adding if this cycle repeats.

Order that works: freeze the tree → tests green → build → measure → publish governance → publish image.

---

## PROVEN END TO END ON TESTNET 526 — 2026-07-30, epoch 76729

Two miners with genuinely different routing policies, competing on real Intel TDX hardware, scored by
an enforcing validator, crowned by the reign, emissions set on chain.

```
epoch 76729: scored=1  dq=4  king=5G1PepNNhF…
  uid 3  head A (lam=0.50, frugal)   k = {mmlu 2, math 2, code 1}   $0.00055/task
  uid 4  head B (lam=0.05, quality)  k = {mmlu 2, math 2, code 2}   $0.00271/task   <- CROWNED
  last_submitted_weights = {'4': 1.0}
```

**The competition was decided on merit.** Head B routes to stronger models and took the ranked code
benchmark 2/2 where A took 1/2 — the same ordering its offline training predicted (code 0.938 vs
0.875). Nothing about the outcome was hardcoded; the reign ranked them on `Q_lcb − cost` and picked
the better router.

Every link verified, none simulated:

| link | evidence |
|---|---|
| miner trains a head offline | `/root/koth-miner-work` (outside the repo), 48 tasks x 7 models, $1.32 |
| publishes weights + commits on chain | `thirtyspokes/koth-miner-router{,-b}`, revealed commit |
| boots the owner's measured image on TDX | `koth-runtime-v22`, RTMR1 `158ed9b1…` matching governance |
| head routes, harness executes the cascade | `KOTH-MODE routing harness=koth-harness-2 rungs=7` |
| hardware-attested proof uploaded | `proofs/<epoch>.json`, real DCAP quote |
| validator binds + verifies under `enforce=True` | no attestation-class DQ |
| grades answers incl. sandboxed code execution | host daemon via socket, `KOTH_GRADE_DIR` aligned |
| evidence accumulates, floor applies | `k > 0`, A still below floor from its miss deficit |
| reign crowns, weights set on chain | `{'4': 1.0}` |

### The nine bugs this run found, none visible offline

1. the measured image ran the AGENT path for routing artifacts (a second copy of the run logic, untested)
2. the reference pool could silently diverge from the router's action space
3. miner/validator slice-size mismatch — `unexpected_task` for every miner
4. slice size made an epoch physically impossible (a proof that overruns is unrecoverable, not late)
5. one malformed provider response destroyed a 49-minute epoch
6. `commit()` reported success on a rejected extrinsic
7. routing proofs attested `confined=False` — the whole path was unscoreable under `enforce=True`
8. the shipped validator container had no docker CLI (`docker.io` ships the DAEMON; the client is `docker-cli`, a dropped *Recommends*)
9. the sandbox bind-mount resolved in the host namespace, so code never graded even with a working client

Each was found only by running the real thing, and several were masked by a *different* component's
error message — `below_floor:mmlu` in particular stood in for "no proof found" for hours, because a
miss clears its own reason and re-fails on the accumulated floor.

### Operational constraints measured here, not documented anywhere before

* **Epoch length caps slice size.** ~90s/pool-call over 3 benchmarks against 100-block (~20 min)
  epochs means `n_per_bench=2`. 16 was impossible; 4 straddled the boundary and landed ~half the time.
* **A quality-biased router is penalised twice** — once by the cost term, again by the epoch clock,
  because bigger models are slower. The achievable policy space is narrower than the pool implies.
* **A registered-but-silent miner accrues `miss=0` evidence** and needs either a re-commit (which
  resets it) or ~10 good epochs to dig out. Miner A is still below the floor for exactly this reason
  while miner B, freshly committed, cleared it immediately.
