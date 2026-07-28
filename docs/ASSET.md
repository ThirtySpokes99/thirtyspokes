# What survives the subnet thesis

*The routing economics are closed (`ROUTING_MEASUREMENTS.md`). This is an honest inventory of what
was built that does **not** depend on them, written so the asset is identifiable and extractable
later rather than rediscovered.*

## The capability, stated plainly

> **This exact artifact produced this exact result, on real hardware — and you can verify that
> without re-running it or trusting whoever ran it.**

That is unusual. Benchmark results today are self-reported and unfalsifiable; reproducing them means
paying the full inference cost and still not knowing you ran the same thing. Here the claim is a
hardware signature over a payload that binds the artifact's own bytes, and checking it is a
signature verification plus a hash comparison.

Proven end to end on real Intel TDX, not simulated: a reproducible measured image, a genuine DCAP
quote, and an enforcing verifier that accepted a real run and rejected wrong-image, mock-quote and
tampered variants.

## Inventory

**Reusable — no chain, no economics, no subnet (~1,400 lines + the image recipe):**

| component | what it does |
|---|---|
| `tee/attestation.py`, `tee/runtime.py`, `tee/verify.py` | platform quote abstraction; metered execution proxy |
| `koth/tdx.py` (394) | real TDX v4 quote generation (configfs TSM) + **full DCAP**: chain to pinned Intel SGX Root CA, TCB status, CRL, QE-identity |
| `koth/collateral.py` (76) | DCAP collateral fetch + cache. **Cache key must be the PCK leaf cert, not FMSPC** — two CPUs share an FMSPC but need different collateral; found on hardware |
| `koth/rtmr.py` (110) | measures the runtime into **RTMR3** at startup, binding runtime+config into the quote |
| `koth/proof.py` (105) | the attested payload: `report_data` = sha256 of every field, so tampering breaks the quote |
| `koth/confine.py` (233) | no-egress execution (netns) with secrets hidden, metered parent-side; **the confinement fact is itself attested**, not merely configured |
| `koth/store.py`, `koth/imagestore.py` | content-addressed artifact + image distribution |
| `scripts/build_koth_image_prod.sh` | **reproducible** dm-verity measured image — two builds in different directories produce a byte-identical UKI and roothash |
| `verify_proof`'s attestation half | quote → approved measurement → payload binding → issued epoch/nonce → artifact binding |

**Coupled to the dead thesis (~2,300 lines):** `reign.py` (emissions), `koth/validator.py` (scoring
loop), `koth/reference.py` (pool matrix), `koth/{commit,epoch,owner,governance}.py` (chain), and the
scoring half of `koth/verify.py` (headroom, regret, frontiers, grounding, dedup, eligibility).

Note `koth/verify.py` is **mixed** — attestation verification and economics live in one 797-line
file. Extraction would need to split it; the boundary is clean (roughly `verify_proof` and above vs
`_wmean` and below) but it is not a file move.

## Hard-won details that would be re-learned expensively

- **systemd-boot does not start on GCP TDVF.** Boot the UKI directly as `/EFI/BOOT/BOOTX64.EFI`. This
  moves the per-image anchor to **RTMR1** (RTMR2 stays zero — no GRUB command stream).
- **Reproducibility is fragile and fails silently.** Three separate leaks of the build directory into
  the measured rootfs were found and fixed: venv shebangs + `pyvenv.cfg`; `co_filename` in ~10k
  `.pyc` (binary — a text-only check cannot see it); and pip's `RECORD`, which hashes console scripts
  *before* any rewrite can run. Fixed by pinning the staging path; the build now **fails** if any
  reference survives. Always verify with two builds in **different directories**, comparing the UKI
  and roothash — never `koth-runtime.raw`, whose ESP differs by ~14 unmeasured bytes.
- **The dataset cache must be scrubbed** or the rootfs is not reproducible: `datasets` writes `.lock`
  files whose *filename* embeds the absolute path, plus timestamped `xet/logs`.
- **Kernel ≥ 6.6 is load-bearing**, not just for determinism: RTMR3 extension needs the `tdx_guest`
  measurements sysfs, absent on older kernels — where RTMR3 silently stays zero.

## Honest read on where this could go

The obvious direction is verifiable third-party evaluation — attested benchmark results a regulator,
customer or auditor can check without trusting the claimant.

**One correction to the earlier strategy note, since it changes the sizing.** The EU AI Act
high-risk deadline of **2 Aug 2026 is confirmed**, but third-party conformity assessment *by a
notified body* applies only to **biometric identification and critical-infrastructure safety
components**. Most high-risk categories self-assess via internal control, with documentation and a
declaration of conformity. So the compulsory pull for third-party verification is **narrower** than
"mandatory from August 2026" implies.

That does not make the capability worthless — self-assessment still demands evidence, and
benchmark-score distrust is real and independent of regulation — but it means the next step is a
**customer conversation, not more code**. Nothing in this repo answers whether someone will pay for
it, and this project has now twice learned that building before that question is answered is the
expensive way to find out.

## Current deployed state

Live on **testnet 526**: governance published (MRTD `c1ee9c16…`, RTMR1 `515b759e…`, RTMR3
`c8e9bdc5…`), measured image `v15` published to the HF bucket with a manifest naming the exact
commit that rebuilds it, two registered miners. The per-epoch reference cron has been **stopped**
(78 epochs, ~$1.60) and no GCP instances are running. Nothing is billing; the demonstrator still
stands if it is useful to show.
