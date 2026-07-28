# Verifiable evaluation — one page

*Material for a customer conversation, not a product pitch. `docs/ASSET.md` is the honest inventory;
`docs/ROUTING_MEASUREMENTS.md` is why the subnet this was built for is closed.*

## The claim

> **This exact code produced this exact result, on real hardware — and you can check that without
> re-running it or trusting whoever ran it.**

Run `python scripts/verify_demo.py` to watch a stranger do it against a real Intel TDX proof from GCP
confidential hardware:

```
proof : 2160 graded tasks, $0.1546 of inference, image 5cb871638f665a9b…
quote : REAL Intel TDX (DCAP)

ACCEPTED   in 24 ms, 0 model calls, $0.00

forgeries, each rejected for the right reason:
  edit one graded answer             -> report_data_mismatch
  understate the cost                -> report_data_mismatch
  replay against another challenge   -> epoch_nonce_mismatch
  claim a different artifact         -> artifact_binding_mismatch
```

## Why it is not just "publish your logs"

Logs are assertions by the party being measured. This is a hardware signature over a payload that
commits to its own inputs, so five things hold at once, and the fifth is the one nobody else offers:

| | established by |
|---|---|
| the attestation is genuine | Intel DCAP quote chaining to the pinned Intel SGX Root CA, with TCB status, CRL and QE-identity checked |
| **the code was the approved code** | MRTD + RTMR1/2/3 measurements pin which image booted — from a **reproducible** build, so anyone can rebuild it and get the same measurement |
| no field was edited afterwards | `report_data` is a hash over every field; any change breaks the quote |
| it answers the challenge that was set | the verifier's own (epoch, nonce) — so it is not a replay or a best-of-N pick |
| it is bound to published bytes | source + weights hashes, recomputed by the verifier from the public artifact |

Cost of checking: **~24 ms and no inference.** Cost of the run being checked: the full bill, plus the
unanswerable question of whether you ran the same thing.

## What is real, and what is not

**Proven on hardware, end to end.** A reproducible dm-verity measured image, a genuine TDX quote from
a GCP C3, and an enforcing verifier that accepted a real run and rejected wrong-image, mock-quote and
tampered variants. Not a simulation.

**Two defects found while writing this page**, both fixed and both regression-tested — recorded here
because they are the kind of thing that decides whether the claim survives contact with a real
auditor:

1. **Proofs did not survive their own software.** `report_data` hashed `asdict(proof)`, so every
   field added to the payload silently invalidated every proof ever attested. All eight recorded
   hardware proofs failed under current code — as `report_data_mismatch`, indistinguishable from
   tampering. A routine upgrade would have become a false accusation of forgery against an honest
   party. The payload is now versioned, with each version's field list frozen.
2. **The backward-compatibility fix initially accepted tampering.** Replaying the old payload shape
   verbatim meant `report_data` ignored the proof's actual contents. The demo's forgery section
   caught it on the first run.

**Not established.** Whether anyone will pay for this. The EU AI Act high-risk deadline of
**2 Aug 2026** is confirmed, but third-party conformity assessment *by a notified body* applies only
to biometric identification and critical-infrastructure safety components; most high-risk categories
self-assess through internal control. So the compulsory pull is **narrower** than "mandatory from
August 2026" suggests. Self-assessment still demands evidence, and distrust of self-reported
benchmark scores is real and independent of regulation — but that is a hypothesis, not a finding.

## What it does not do

- It proves *which code ran*, not that the code is *correct*. A measured image running a biased
  grader produces an honest proof of a bad benchmark. Choosing the benchmark stays a human problem.
- It cannot prove absence of a network side channel by construction; the no-egress confinement is
  itself attested (`koth/confine.py`), but prompt-based exfiltration remains documented and open.
- It needs the verifier to know which image measurement to trust. That is a governance question —
  solved here with on-chain approved measurements, but any registry would do.

## Questions worth asking a prospective user

Ordered so an early "no" saves the later ones:

1. When you read a published eval score, do you act on it, or discount it? If you discount it, what
   do you do instead, and what does that cost you?
2. Has anyone ever asked you to *prove* an eval result — a customer, an auditor, a partner, a
   regulator? What did you send, and did it satisfy them?
3. Would a proof that a specific model build scored X on a benchmark you chose change any decision
   you make, or would you still want to run it yourself?
4. Who is the buyer — the party being evaluated (wants credible claims) or the party relying on it
   (wants to stop trusting)? They have different willingness to pay and only one of them is you.

**If the answers are "discount it, nobody has ever asked, I'd rerun it anyway", that is a no**, and
it is worth much more than another month of building. This project has twice learned that building
before answering this is the expensive way to find out.

## What extraction costs

~1,400 lines are already independent of the dead economics: `koth/attest.py` (the gate, 125 lines,
imports nothing from scoring — pinned by a test), `koth/tdx.py`, `koth/collateral.py`, `koth/rtmr.py`,
`koth/proof.py`, `koth/confine.py`, `tee/*`, plus the reproducible image recipe. `docs/ASSET.md` lists
the hard-won hardware details that would otherwise be re-learned expensively.
