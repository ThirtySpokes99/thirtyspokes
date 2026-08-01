# ThirtySpokes — deploying the subnet (owner)

How to stand the subnet up (currently live on **testnet 526**; mainnet is netuid 99) and build the measured
runtime image. To run a single node instead, see [`MINER.md`](MINER.md) / [`VALIDATOR.md`](VALIDATOR.md);
for *how it works*, see [`DESIGN.md`](DESIGN.md). *(Forking this to run your own subnet? Substitute your
own netuid throughout.)*

> **The measured image is the whole security model.** Until you build it and pin its measurements
> on-chain, an enforcing validator cannot tell an honest runtime from a tampered one — so treat
> [§Measured image](#the-measured-runtime-image) as a hard prerequisite, not an optional nicety.

## The trust boundary

The subnet does **not** trust anyone's machine — it trusts the owner-published **measured runtime
image**. A miner's agent runs in the miner's *own* confidential VM, and that VM samples and runs the
benchmark; the only thing standing between a trustless miner and a forged score is the hardware
measured-image gate you establish here.

- The miner boots the owner's published, locked-down image ([§Measured image](#the-measured-runtime-image))
  on a TDX/SEV-SNP CVM. That image loads the miner's public `source.py` + `weights.bin` as the untrusted
  agent, hashes exactly those bytes into the attestation, meters cost, and runs the agent with **zero
  network egress** (its only channel is the metered `call_model`).
- **What a miner competes on is the agent** (routing/orchestration source + weights) — fully public.
  **What is fixed and measured is the runtime.**
- Validators run **fail-closed** (`enforce=True`): a proof is scored only if it carries a genuine
  hardware quote whose measurements (MRTD + RTMR1/2/3) match the owner-approved image under an in-date
  TCB. A modified runtime, a stock CVM, or the mock TEE is **rejected, not scored**.
- The mock TEE and the no-gate `--insecure` mode exist **only for offline development**; the validator
  daemon **refuses `--insecure` on mainnet**, so there is no no-security path in production.

This is *why* static public benchmarks are safe: the score is bound to a public artifact running under
an audited runtime, so any cheat (hardcoding, memorization, off-pool calls) is visible in the public
source or caught by the validator's checks. Full trust model: [`DESIGN.md`](DESIGN.md) §1 + §8.

## Local development (offline)

Before touching the chain, exercise the whole mechanism offline — no infra, no keys. This is the only
place the mock TEE is used; it has no security and mainnet rejects its proofs.
```bash
uv run pytest -q                         # the full suite
uv run orchestra-koth-sim                # full mechanism + all adversaries (mock everything)
uv run orchestra-koth-local              # decoupled 2-neuron demo (miner uploads, validator verifies)
set -a && . ./.env && set +a && uv run python scripts/koth_live_smoke.py   # real models + benchmarks, local chain
```

## Production — Bittensor

> **Currently deployed on Bittensor TESTNET — netuid 526** (`--network test`). Every command below
> shows the live testnet target; mainnet (netuid 99, `--network finney`) remains the eventual
> destination and is the code default, so **pass `--network test` explicitly** while running against
> 526. Governance, the measured image (`v27`) and the per-epoch pool reference are published there.


The trust claim is only real with genuine hardware attestation, so a production subnet needs the
measured image pinned on-chain **before** miners can earn. Order of operations:

**1. Build + pin the measured image** — see [§Measured image](#the-measured-runtime-image) below. This is
the prerequisite: until MRTD + RTMR1/2/3 are published, an enforcing validator has nothing to gate on.

**2. Register on-chain.**
```bash
uv pip install -e ".[chain,eval,tee]"
export OPENROUTER_API_KEY=...            # MINER (runs the benchmark) and OWNER (builds the pool
                                         # reference). A validator needs NO LLM key in the default
                                         # grounding mode — but it DOES need Docker, since the ranked
                                         # benchmark is graded by executing code (DESIGN.md §5e).
huggingface-cli login
btcli wallet new_coldkey && btcli wallet new_hotkey        # one wallet per role
btcli subnet register --netuid 526 --wallet.name miner     --subtensor.network test
btcli subnet register --netuid 526 --wallet.name validator --subtensor.network test
# owner: enable commit-reveal + a sane tempo on the subnet hyperparams; validator needs stake + vpermit
```

**3. Publish governance** (the approved measurements + pinned pool + TCB policy) with
`orchestra-koth-owner` — see [§Build, pin, publish](#build-pin-publish). Validators read this **per
epoch**, so rotating the image or recovering TCB takes effect without a restart.

**4. Publish the per-epoch pool reference.** This is what turns validators' scalar from "how accurate
was this agent" into "how well did it *route*" — see [`DESIGN.md`](DESIGN.md) §5.0/§5c. Validators run
no inference, so only the owner can measure what the other pool models would have answered.
```bash
# once per epoch, next to the validator (a cron / systemd timer is the intended shape)
set -a && . ./.env && set +a            # OPENROUTER_API_KEY — the OWNER pays for this, not miners
orchestra-koth-reference --netuid 526 --network test --wallet owner \
  --pool "$THE_PINNED_POOL" --n-per-bench 8 \     # MUST match the validators' --n-per-bench
  --deadline-s 900 --call-timeout 180
```

**Run it as a supervised service, and check that it is still publishing.** If this stops, nothing
breaks and nothing stops: `_load_reference` degrades on purpose, every miner falls through to the
legacy quality/cost scalar, and validators keep scoring and setting weights on a different quantity
than you intend. The epoch line does say `reference=MISSING(scoring absolute accuracy, NOT routing)`
— but that is a field in a running process's log, and mainnet still ran 48 epochs that way before
anyone read the bucket. `orchestra-koth-doctor --netuid <N> --wallet <w>` now fails on a stale
reference, so make it a startup gate rather than something you have to notice:
```ini
# /etc/systemd/system/koth-reference.service — `--loop` publishes once per epoch and waits
[Service]
EnvironmentFile=/path/to/thirtyspokes/.env
ExecStart=/root/.local/bin/uv run --frozen orchestra-koth-reference \
  --netuid 99 --network finney --wallet owner --hotkey <owner-hotkey> \
  --n-per-bench 8 --call-timeout 180 --loop
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
```
`Restart=always` covers a crash; it does **not** cover a wedged run (a provider call has held a job
for 35+ minutes here on 3.5s of CPU) — which is why the doctor check reads the bucket rather than
the unit state.

**Keep `--deadline-s` inside your epoch.** A reference that outruns its epoch describes a slice nobody
is scored on any more, so it is dead on arrival — and provider calls can hang well past their own
timeout (observed during this build-out: one request sat with the connection ESTABLISHED and no
response, holding the job for 35+ minutes on 3.5 seconds of CPU). Cells outstanding at the deadline
are treated as failed and their rows drop, so a hang costs coverage rather than the epoch. Progress
is printed as cells land, so a wedged run is visible in the cron log rather than looking like a slow
one.
Cost is `n_per_bench × |pool|` calls per epoch — 8 × 6 = 48, well under a dollar. The record is signed
with the owner hotkey and served from the owner's bucket (it cannot go on-chain: the owner's single
commitment slot already holds the governance record, and overwriting it would blank the measured-image
gate for the whole subnet). **Read the `achievable gap` it prints.** Below the validators'
`--min-headroom-gap` (0.05) the traffic is saturated — a perfect router could add less than the
sampling noise — and every validator will decline to score routing on it and publish
`routable: false`. That is a signal about the *benchmark suite*, not about the miners; see
`koth/lcb.py` for the alternative and what adopting it costs.

Skipping this step is safe: epochs without a reference simply score on absolute accuracy.

**5. Run the validator and the external locked-image miner operator.** Both default to
`--network finney`, so pass `--network test` for the live 526 deployment.
```bash
orchestra-koth-validator --netuid 526 --network test --wallet validator                 # verify-only; no GPU, no CVM
orchestra-koth-gcp-miner --netuid 526 --network test --wallet miner --repo YOU/koth-miner \
  --image THE_OWNER_APPROVED_GCP_IMAGE                                  # creates one TDX VM/epoch
```

**Miner CVM:** GCP **C3** (Intel TDX) / N2D (SEV-SNP), or Azure DCesv6 (TDX) / DCasv5 (SEV-SNP).
CPU-only, small (2–4 vCPU), **no GPU** (pool models are remote) — cheap on Spot. The miner must boot the
owner's measured image. The shipped GCP image is dm-verity locked, has no shell/sshd, and runs one proof
per boot; `orchestra-koth-gcp-miner` runs outside it and handles epoch derivation, fresh-VM retry,
serial proof collection, upload, and cleanup. The measured in-image runtime extends RTMR3 and runs the
agent with zero network egress.

**Verify the gate on real hardware:**
```bash
uv run python scripts/koth_tdx_smoke.py          # real quote over a KOTH proof; forgery + MRTD rejected
uv run python scripts/koth_enforce_smoke.py      # the full enforce gate: accept + 8 fail-closed paths
set -a && . ./.env && set +a && uv run python scripts/koth_live_tdx_smoke.py   # + real models & metered cost
```
Attestation internals (all verified on Intel TDX / GCP C3): `koth/tdx.py` generates a genuine TDX v4
quote (kernel configfs TSM) whose REPORTDATA = the proof's `report_data()`; `verify_quote_full` does the
complete DCAP check — chain → pinned Intel SGX Root CA **+ TCB status / CRL / QE-identity via
`dcap-qvl`** — then re-applies the MRTD / RTMR1/2/3 gate + the binding. `koth/rtmr.py` extends **RTMR3**
with `H(runtime+suite+pool)`; `koth/confine.py` runs the agent with zero egress (netns, verified BLOCKED).

**Known gaps (documented):** challenger **stake+slash** economics are a subnet-token seam, not native to
Bittensor weight-setting; SEV-SNP as a second attestation backend (same shape, different parser).

## The measured runtime image

*(The load-bearing anchor for production.)* The hardware quote proves *a TDX guest* produced the proof; it
does not by itself prove the *owner's trusted runtime* ran (on a stock CVM the operator can run arbitrary
code after boot and extend the "right" RTMR3 while running different code). The anchor is a
**reproducible, locked-down measured image** whose boot chain measures into registers the miner cannot forge.

### What the registers measure (hardware-verified on GCP C3)

- **MRTD** — Google's TDVF firmware, **identical on every GCP TDX guest**, so pin it only as a coarse
  "genuine GCP TDX firmware" check, not a per-image discriminator.
- **RTMR1** — the boot chain. With a **direct-UKI boot** (the working GCP config — see below), the
  firmware measures the launched EFI app (our UKI = kernel + cmdline + initrd) here; this is the
  **per-image anchor to pin**. With stock GRUB the anchor is RTMR2 instead.
- **RTMR2** — the GRUB command stream + kernel + cmdline (stock-GRUB boot); **0** under direct-UKI boot.
- **RTMR3** — all-zero at boot, extended in userspace by the KOTH runtime (`koth/rtmr.py`) to
  `SHA384(0 ‖ H(runtime‖suite‖pool))`, then gated by the validator.

The rootfs is **dm-verity read-only** with **no sshd, getty, or shell** — only the KOTH runtime (a
systemd unit) runs, so nothing but the runtime can extend RTMR3. The dm-verity **root hash is baked into
the UKI cmdline**, so any rootfs change changes RTMR1 → DQ (`unapproved_runtime`). Miner secrets
(OpenRouter key, Bittensor hotkey) are injected at boot via encrypted CVM metadata, never baked into the
public image.

> **Reproducibility hazard found on hardware:** stock GRUB writes `grubenv` (`recordfail` / `save_env`),
> so a prior failed boot changes the next boot's command stream → changes RTMR2 → every honest miner is
> DQ'd. The measured image must use a **UKI / static grub.cfg** with no `save_env`. On GCP's confidential
> TDVF, systemd-boot does not start — **boot the UKI directly** as `/EFI/BOOT/BOOTX64.EFI` (no bootloader:
> one measured blob, no nondeterminism).

### Build, pin, publish

`scripts/build_koth_image_prod.sh` drives [`mkosi`](https://github.com/systemd/mkosi) to a deterministic,
dm-verity-locked image. Determinism levers: pinned `SOURCE_DATE_EPOCH` + apt snapshot + `thirtyspokes`/dep
versions; a fixed kernel + cmdline (measured into RTMR1); the dm-verity roothash in the cmdline; a fixed
`Seed=` for partition/erofs UUIDs; a `mkosi.finalize` that strips regenerated state; and a strip of the
baked HF cache's build junk (see below). Two builds from the same commit + lockfile produce a
**byte-identical UKI → identical RTMR1** (verified: two runs from *different build directories* →
identical UKI and identical verity roothash). GCP custom-image requirements: kernel ≥ 6.6, gVNIC ≥ 1.01,
NVMe driver in the initramfs. The kernel pin is load-bearing beyond determinism: RTMR3 extension and
`rtmr.read_measurements()` need the `tdx_guest` measurements sysfs, which older kernels (e.g. 6.8) do not
expose — on those, RTMR3 silently stays zero.

> **Always re-run the two-build check after touching the image contents.** Anything that lands in the
> rootfs lands in the verity roothash → RTMR1. This has now bitten twice, both times silently:
>
> 1. **The baked benchmark cache.** The dataset bytes were identical, but `datasets` also wrote `.lock`
>    files whose **filename embeds the absolute cache path**, and a `xet/logs/…<timestamp>_<pid>.log`.
>    The recipe now deletes both.
> 2. **The staged venv.** When venv staging moved from the fixed `/opt/koth/venv` to `$OUT/.venv-stage`
>    (so a build could not clobber a host runtime), `python -m venv` baked that absolute path into
>    `pyvenv.cfg` **and into the shebang of all ~30 console scripts**. Measured: two builds →
>    roothash `533c80d7…` vs `2d41ca85…`. It also meant every `/opt/koth/venv/bin/orchestra-*` in the
>    image pointed at an interpreter that does not exist there — invisible, because the runner calls
>    `venv/bin/python <script>` directly and `bin/python` is a copied *binary*, not a script. The
>    recipe now rewrites the staging path to `/opt/koth/venv` in the image copy and **fails the build**
>    if any reference survives.
>
> Both bugs produce a perfectly valid-looking image whose only symptom is that miners who rebuilt the
> recipe get rejected `unapproved_runtime`. Reproducibility is a property of the *whole* rootfs, so
> verify it, don't assume it — `sha256sum` the UKI from two builds in **different directories**.

Obtain the measurements either by **predicting** RTMR1 offline from the build artifacts, or by
**capturing** them from one controlled boot (`koth/rtmr.read_measurements()` / `scripts/koth_gcp_measure_probe.sh`).
Then publish the approved set (RTMR3 is *derived* from the runtime/suite/pool, not read):
```bash
orchestra-koth-owner \
  --mrtd <mrtd_hex> --rtmr1 <rtmr1_hex> --rtmr2 <rtmr2_hex> \
  --pool "openai/gpt-4o-mini,anthropic/claude-opus-4.7" \
  --tcb-accept UpToDate,SWHardeningNeeded \
  --netuid 526 --wallet owner --network finney
```

**How it is stored, and why.** The record is ~657 bytes; a plain on-chain commitment's `Raw` field caps
at 128 ("Value 'Raw657' not present in type_mapping" — the chain says so). So the **sha256 goes
on-chain** (73 bytes, a plain commitment) and the **record goes in your public bucket, named by its own
hash**. Validators read the hash, fetch the record, and verify it — a tampered record fails the hash and
is rejected, so the bucket needs no trust. Same asymmetry as the runtime image: *the chain is the trust
root; the bucket is transport.*

> ✅ **Governance is visible to validators IMMEDIATELY** (measured: 1.0s after publish). It used to be
> shoved whole into a *timelocked* reveal-commitment — not by design, but because that was the only
> thing large enough to fit. The timelock bought nothing (the payload is encrypted until reveal, so
> miners got no advance warning of a change) and cost a great deal: an emergency TCB tightening after
> an Intel disclosure could not take effect for ~72 minutes.
>
> If you want to give miners notice of a *planned* rotation, put an explicit `effective_from` block
> **inside** the record — visible at once, active later. That is real notice; the timelock never gave any.

**Validators find you automatically.** They resolve the subnet owner from the chain itself
(`SubnetOwnerHotkey`) and read the record published by *that* hotkey — so you must publish from the
subnet-owner hotkey, and validators need no configuration. (`--owner-hotkey` exists only to override
this on a fork.) This is deliberate: a validator that has to be *told* who the owner is could be
pointed at an attacker's hotkey and made to trust the attacker's image.

**Measured values for the shipped recipe** (`scripts/build_koth_image_prod.sh`, hardware-captured on a
GCP C3 TDX guest — the image boots dm-verity-locked, runs the metadata-injected agent with zero egress,
and an enforcing validator accepts its proof while rejecting a stock CVM):

**v27** (`recipe_commit a06cd1a`, harness `koth-harness-4`, governance record v7):

| | |
|---|---|
| UKI sha256 | `f9d650cc36bf6a589e98ad36db9d610d1ce1faeaab0a1b53ab70b9767e374426` |
| verity roothash | `a05e4d5325ef6b6aa155b29a9b2aea7649390857aba7258d857d046d7ef0ba31` |
| MRTD | `c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5` |
| **RTMR1** | `cf0db5f18e34e2e699a98b256df3fc614019f5a1f5b76493e2aadc77e75b5be3d3b7ab3f409d5619dc70255d60c51444` |
| RTMR2 | `000…0` (zero under direct-UKI boot) |
| RTMR3 | `dd9bc28fae67ef43c5e998679a58ecab02e3042df9d5afa00534a6386948b1954bf4c704641d4a263a0ba897b512a186` (derived) |
| runtime measurement | `1449fadb4821cadef93f7eecc8c3b040e2cd244e01a2607ea531e5f7055c38d8` |

Two consecutive rotations make the register semantics concrete, and they are worth reading together
before judging any change of your own:

* **v22→v23 moved ONE register.** It added a per-task `print` to the enclave runner and nothing else:
  RTMR1 changed (any rootfs byte does that), while MRTD (GCP's TDVF, not ours), RTMR2 and RTMR3 stayed
  identical, and `runtime_measurement` stayed `c8c5d2ff…`. Accumulated miner evidence was **not** reset,
  and miners did **not** need to re-commit.
* **v23→v24 moved three.** It bounded cascade escalation, which is an ENGINE change, so
  `HARNESS_VERSION` bumped to `koth-harness-3`; that changed `runtime_measurement`, which changed the
  derived RTMR3, and the rootfs change moved RTMR1 as usual. Evidence resets, and — the part that is
  easy to miss — **every miner must re-publish and re-commit**, because a routing artifact's
  `source_text` *is* the harness version, so the old commit no longer binds.
* **v26→v27 moved only RTMR1 as well**, and for the same reason: the runtime now enforces the task
  budget with a WATCHDOG instead of asking the provider client to honour a timeout. It turned out an
  httpx read timeout never fires on a response that trickles, so three successive "bounded" versions
  were not bounded and ~10% of miner-runs were lost to a single hung call. Enforcement changed;
  the engine contract did not.
* **v25→v26 moved only RTMR1 — a bug fix, not an engine change.** It made the runtime actually
  enforce the budget harness-4 already specified (the per-call bound leaked through the SDK's own
  retries). `HARNESS_VERSION` stayed, so `runtime_measurement` and RTMR3 stayed, so **evidence was
  not reset and no miner had to re-commit**. That was a deliberate call: evidence in the accumulator
  came only from runs that COMPLETED — a hung run produced no proof at all — and completed runs
  behave identically under the fix, so old and new evidence stay comparable. Resetting would have
  charged miners for the owner's bug.
* **v24→v25 moved the same three, one epoch later.** harness-3's gate bounded only *escalation*, and
  a single in-flight call still cost epoch 76738 outright; harness-4 bounds every call and shares the
  budget across the run. Same three registers, same reset, same re-commit — the cost of an engine
  change is a property of *what* changed, not of how large the diff was.

If a change moves RTMR3 or the runtime measurement, it is an engine change: plan for reset evidence
and a re-commit cycle, not just an image swap.

> **Engine changes have no gap-free ordering.** Once the new commit REVEALS, miners on the old image
> break (they fetch a revision whose `source_text` is the new harness and refuse it — the binding
> working as designed); before it reveals, miners on the new image break for the mirror reason. So
> publish governance and switch miners together, at the reveal, and expect to lose an epoch. Capture
> the new image's RTMR1 *ahead* of the reveal by injecting the new artifact as VM metadata instead of
> going through the chain — the image reads its artifact from metadata, so a capture needs no commit.

> **Capturing RTMR1 from the measured image.** It ships no sshd, so the sysfs probe cannot be run
> inside it. The image's own attested proof is the capture channel — the quote in `proof.quote`
> `platform_sig` (`tdx:<base64>`) carries MRTD and all four RTMRs. Boot one epoch with the operator,
> then parse the serial log it saved. Validate any such extractor against a boot of the *currently
> pinned* image first: it must reproduce the on-chain record exactly before you trust it for a new one.

> **Rotation is a hard cutover.** The record pins exactly ONE `rtmr1`, so the moment you publish, every
> proof from the previous image is `unapproved_runtime`. Publish and restart your miners back to back,
> and expect to lose the epoch in between. (Allowing a list of approved RTMR1s would make rotations
> overlap; today it does not.)

**Distribute the image on HuggingFace** so miners can boot it:

```bash
huggingface-cli login          # or export HF_TOKEN
python scripts/publish_runtime_image.py \
  --version v25 --image /path/koth-runtime-v27.tar.gz --build-dir /root/koth-build-v25a \
  --uki-sha256 <uki> --roothash <roothash> --mrtd <mrtd> --rtmr1 <rtmr1> \
  --pool "openai/gpt-4o-mini,anthropic/claude-opus-4.7"
```
This uploads the raw disk to the public HF bucket `thirtyspokes/cvm-runtime-image` under
`runtime/<version>/`, with a manifest
carrying the image sha256, the UKI hash, the verity roothash, the measurements, and **the git commit of
the recipe that built it** — so a miner can check out that exact revision, rebuild, and confirm they got
your image rather than taking it on trust.

**Announce the image sha256 it prints.** An HF *bucket* is object storage — unlike a repo it has no
immutable commit SHA, so miners cannot pin a revision. The manifest's `image.sha256` is what they
check before booting; RTMR1 on-chain is what catches a bad image after.

**Distribution needs no trust.** The image is self-verifying: you pin RTMR1 on-chain, and the build is
reproducible, so a miner who fetches corrupted or malicious bytes simply fails the gate
(`unapproved_runtime`). The manifest hashes are a convenience — they let a miner find out before
burning a CVM boot, not a security boundary.

> **Single source, no mirror.** HuggingFace is the only distribution channel. If it is down, *new*
> miners cannot onboard (running miners are unaffected — they already have the image). Because the
> image is self-verifying, adding a mirror later costs nothing in security.

Validators read this each epoch and gate every proof's MRTD + RTMR1/2/3 + TCB status against it. Rotating
the image or recovering TCB = publish a new record (bump `--version`); old images stop being accepted at
the next epoch.

**Status: the full loop is hardware-proven end-to-end** (GCP C3, Intel TDX). The measured image builds
**reproducibly** (two independent builds → byte-identical UKI → identical RTMR1), boots on a confidential
TDX VM, is **dm-verity-locked** (roothash in the UKI cmdline → RTMR1), loads the metadata-injected agent,
runs it with **no network egress**, and emits a proof carrying a genuine Intel-signed quote (full DCAP:
chain → pinned SGX Root CA, CRL, QE identity, TCB `UpToDate`). With the image's measurements pinned as
governance, an **enforcing validator accepts the in-image proof (`valid=True`) and rejects a proof from a
stock Ubuntu TDX CVM (`unapproved_runtime`)** — the discrimination the whole trust model rests on. The
runtime's RTMR3 matched the owner's independently-derived expectation exactly, and tampering with a proof
breaks its quote (`report_data_mismatch`).

Two facts worth internalising, both learned on hardware:

- **MRTD does not identify your image.** It is Google's TDVF firmware and is *byte-identical on every GCP
  TDX guest* — the stock Ubuntu CVM and the measured image produced the same MRTD. Pin it only as a coarse
  "genuine GCP TDX firmware" check. **RTMR1 is the per-image anchor** (it differed between the two), and it
  is what actually rejects a stock CVM.
- **The image needs a baked resolver and baked benchmark data.** It ships no `systemd-resolved` and its
  rootfs is read-only, so nothing writes `/etc/resolv.conf` and every pool call dies with *"Temporary
  failure in name resolution"*; and loading the suite from the Hub at boot is network-dependent,
  non-deterministic, and leaves the questions *outside* the measured rootfs. The recipe now bakes both
  (`/etc/resolv.conf` + the HF cache under `/opt/koth/hf`, loaded with `HF_HUB_OFFLINE=1`), which also puts
  the benchmark data under dm-verity → covered by RTMR1 → pinned.
