# ThirtySpokes — deploying the subnet (owner)

How to stand the subnet up on Bittensor mainnet (ThirtySpokes is **netuid 99**) and build the measured
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

## Production — Bittensor mainnet

The trust claim is only real with genuine hardware attestation, so a production subnet needs the
measured image pinned on-chain **before** miners can earn. Order of operations:

**1. Build + pin the measured image** — see [§Measured image](#the-measured-runtime-image) below. This is
the prerequisite: until MRTD + RTMR1/2/3 are published, an enforcing validator has nothing to gate on.

**2. Register on-chain.**
```bash
uv pip install -e ".[chain,eval,tee]"
export OPENROUTER_API_KEY=...            # MINER only (it runs the benchmark and pays for it).
                                         # A validator needs NO LLM key in the default grounding mode.
huggingface-cli login
btcli wallet new_coldkey && btcli wallet new_hotkey        # one wallet per role
btcli subnet register --netuid 99 --wallet.name miner     --subtensor.network finney
btcli subnet register --netuid 99 --wallet.name validator --subtensor.network finney
# owner: enable commit-reveal + a sane tempo on the subnet hyperparams; validator needs stake + vpermit
```

**3. Publish governance** (the approved measurements + pinned pool + TCB policy) with
`orchestra-koth-owner` — see [§Build, pin, publish](#build-pin-publish). Validators read this **per
epoch**, so rotating the image or recovering TCB takes effect without a restart.

**4. Run the validator and the external locked-image miner operator.** Both default to
`--network finney`.
```bash
orchestra-koth-validator --netuid 99 --wallet validator                 # verify-only; no GPU, no CVM
orchestra-koth-gcp-miner --netuid 99 --wallet miner --repo YOU/koth-miner \
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
> rootfs lands in the verity roothash → RTMR1. Baking the benchmark cache in (necessary, above) initially
> *broke* reproducibility: the dataset bytes were identical, but `datasets` also wrote `.lock` files whose
> **filename embeds the absolute cache path** and a `xet/logs/…<timestamp>_<pid>.log`. Two builds in
> different directories therefore produced different RTMR1s — i.e. every miner who rebuilt the recipe
> would have been rejected `unapproved_runtime`. The recipe now deletes both. Reproducibility is a
> property of the *whole* rootfs, so verify it, don't assume it.

Obtain the measurements either by **predicting** RTMR1 offline from the build artifacts, or by
**capturing** them from one controlled boot (`koth/rtmr.read_measurements()` / `scripts/koth_gcp_measure_probe.sh`).
Then publish the approved set (RTMR3 is *derived* from the runtime/suite/pool, not read):
```bash
orchestra-koth-owner \
  --mrtd <mrtd_hex> --rtmr1 <rtmr1_hex> --rtmr2 <rtmr2_hex> \
  --pool "openai/gpt-4o-mini,anthropic/claude-opus-4.7" \
  --tcb-accept UpToDate,SWHardeningNeeded \
  --netuid 99 --wallet owner --network finney
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

| | |
|---|---|
| UKI sha256 | `4757aaabe0cfef16732da7db6c92043cde55552c379d6048b8c7149fa5daff63` |
| verity roothash | `4af8c767a28675e05260b4b81d2a95d0dcb3469dd70f8b7e96b80e9bc30edad7` |
| MRTD | `9bf86e6280ec4282b8b5822d8166410a456cdb720109aa799f0011fa63df1de3ee5e35e293fc410c061433163acb03a6` |
| **RTMR1** | `6d8893eb9255aad4b52045250f6aaecefbf93163ebcc2b9e3149930314df6c0475c06d1883e77e5aba610f0fee42c796` |
| RTMR2 | `000…0` (zero under direct-UKI boot) |

**Distribute the image on HuggingFace** so miners can boot it:

```bash
huggingface-cli login          # or export HF_TOKEN
python scripts/publish_runtime_image.py \
  --version v14 --image /path/koth-runtime-v14.tar.gz \
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
