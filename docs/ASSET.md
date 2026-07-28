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
| `koth/attest.py` (125) | **the gate, now standalone**: quote → approved measurement → payload binding → issued epoch/nonce → artifact binding. Imports only `tee/attestation` + `proof`; a test asserts it pulls in **no** scoring module, so the boundary cannot silently rot |

**Coupled to the dead thesis (~2,300 lines):** `reign.py` (emissions), `koth/validator.py` (scoring
loop), `koth/reference.py` (pool matrix), `koth/{commit,epoch,owner,governance}.py` (chain), and the
scoring half of `koth/verify.py` (headroom, regret, frontiers, grounding, dedup, eligibility).

~~Note `koth/verify.py` is **mixed**~~ — **DONE.** The attestation gate was extracted to
`koth/attest.py`; `verify_proof` now calls it and keeps only the grading half, with its two jobs
labelled in place. Behaviour is unchanged (a test asserts both entry points reject the same proof for
the same reason) and the separation is pinned by a test that imports the gate in a fresh interpreter
and fails if any economics module appears. The remaining coupling in `verify.py` is scoring-only.

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

## Durability: a defect that would have voided the whole claim

Verifying the recorded hardware proofs while building `scripts/verify_demo.py` found that **none of
them verified under current code**. `report_data` hashed `asdict(proof)`, so each of the seven fields
added since they were attested silently changed their hash. The proofs were authentic and
self-consistent; they failed as `report_data_mismatch` — the same reason a *tampered* proof fails.

That is the worst available failure mode for this capability. "Check it later without trusting the
claimant" is the entire product, and a routine software upgrade would have turned it into a false
accusation of forgery against an honest party.

Fixed by versioning the attested payload (`PROOF_SCHEMA`, each version's field list frozen forever);
proofs written before versioning are verified against the field set they were actually attested with.
All eight recorded TDX proofs verify again. A first version of that fix replayed the stored payload
verbatim, which meant `report_data` ignored the proof's contents and a *tampered* legacy proof
verified — caught immediately by the demo's forgery section, and now pinned by a test. Both are
regression-tested.

**Generalisable lesson for anything built on this:** an attested payload is a wire format, and its
shape is part of the signature. It must be versioned from the first line of code, not when the first
old proof fails.

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
commit that rebuilds it, two registered miners. No GCP instances are running.

**Correction (2026-07-28): an earlier version of this section claimed the reference cron had been
stopped and that "nothing is billing". Both were false.** A check found the cron still installed at
`*/5 * * * *` and publishing — 126 records over epochs 76533–76602 — at 16 paid pool calls each
(8 of them gpt-4o), roughly **$0.02/epoch ≈ $6/day**, funding a frontier for a thesis that is closed.
The validator container was also up, and scoring nothing: every epoch logged `scored=0 dq=5` with
`grading_unavailable: docker not available`, because the image has no Docker to run the code grader.

The lesson is not the money, it is that **"I stopped it" was written down instead of checked.** A
cron's failure mode is silence in both directions: it is equally quiet when it should be running and
when it should not.

To wind down (both reversible — the script and the container are kept):

```bash
crontab -l | grep -v koth-reference-cron | crontab -   # stop the paid reference publishes
docker stop thirtyspokes-validator                     # restart later with `docker start`
```

The demonstrator still stands if it is useful to show; it should just not be running by accident.
