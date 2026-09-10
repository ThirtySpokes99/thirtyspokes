"""Client for the confidential router (docs/CONFIDENTIAL_SERVING.md §7).

This module is the user's half of the privacy claim, and it is the only place the claim is
actually *decided*. Everything on the enclave side can be perfect and still worthless if the
client seals to whatever public key the miner hands back, so the ordering below is the substance:

    fetch announcement + quote
        │
        ├─ 1. verify the quote                  hardware signature, Intel chain, TCB, CRL, QE id
        ├─ 2. gate the measurements             MRTD/RTMR ∈ the owner's approved set
        ├─ 3. recompute the binding             sha512(label ‖ pk ‖ image ‖ epoch) == report_data
        │
        └─ only now: seal

Skipping step 3 is the subtle failure. Steps 1 and 2 prove *an* approved enclave exists on that
machine; they say nothing about whether the key you were just handed came from it. A miner running
a genuine approved enclave alongside a plain HTTP process can serve you a real quote and its own
public key, and you would encrypt every request to the miner. The binding is what ties the key to
the quote, which is why `verify()` refuses to return a sealer unless all three pass, and why there
is no flag to disable any of them.

`insecure_skip_verify` exists for local development against a mock enclave, is refused whenever a
quote is present, and is loud. That is deliberate: the failure mode of a quiet bypass here is total
and silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..koth import sealed

DEFAULT_TIMEOUT = 180.0


@dataclass
class EnclaveVerdict:
    """Why a client did or did not trust an enclave. Worth returning rather than raising, because
    a user-facing product wants to *show* this, not just fail."""

    ok: bool
    reason: str
    announcement: dict | None = None
    mrtd: str | None = None
    rtmrs: dict[int, str] = field(default_factory=dict)   # index -> hex, from the quote
    tcb_status: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def verify(announcement: dict, quote: bytes, *, approved_mrtd: set[str] | None = None,
           approved_rtmr: dict[int, str] | None = None, collateral: str | None = None,
           full: bool = True) -> EnclaveVerdict:
    """Run all three checks in order. Returns a verdict; never raises on an untrusted enclave.

    `approved_mrtd`/`approved_rtmr` come from the owner's on-chain record
    (`chain.owner_measurements()`), not from the miner. Passing None means the caller has chosen
    not to pin an image, which reduces the guarantee to "some TDX enclave" — legitimate for a
    smoke test, wrong for a user.
    """
    from ..koth import tdx

    # 3 first, because it is free and needs no network: does the quote even mention this key?
    try:
        expected = sealed.binding(bytes.fromhex(announcement["public_key"]),
                                  announcement["image_measurement"], int(announcement["epoch"]))
    except (KeyError, ValueError) as exc:
        return EnclaveVerdict(False, f"malformed announcement: {exc}")

    verifier = tdx.verify_quote_full if full else tdx.verify_quote
    kw = {"approved_mrtd": approved_mrtd, "approved_rtmr": approved_rtmr}
    if full:
        kw["collateral"] = collateral
    try:
        verdict = verifier(quote, expect_report_data=expected, **kw)
    except Exception as exc:  # noqa: BLE001 — an unverifiable quote is an untrusted enclave
        return EnclaveVerdict(False, f"quote verification failed: {type(exc).__name__}: {exc}")

    if not verdict.ok:
        return EnclaveVerdict(False, f"quote rejected: {verdict.reason}")

    # 4. the announced image must BE the hardware's measurement, not a label beside it.
    #
    # Without this the `image_measurement` field is decorative: it is bound into the AEAD and into
    # report_data, so it is self-consistent no matter what the enclave puts there. A client that
    # pins nothing would accept any string; a client that pins `approved_mrtd` would catch a wrong
    # image but still display a name the enclave chose. Requiring equality collapses the two into
    # one fact, and means a user reading `image_measurement` is reading the silicon.
    #
    # It is the COMPOSITE identity, not MRTD. MRTD on a cloud TD is the virtual firmware and is
    # shared by every guest on the platform, so pinning it alone would accept a completely different
    # image — see `tdx.measurement_id`.
    expected_image = tdx.measurement_id(verdict.mr_td, verdict.rtmrs)
    if announcement["image_measurement"] != expected_image:
        return EnclaveVerdict(
            False, f"announced image {announcement['image_measurement'][:16]}… is not this "
                   f"enclave's measured identity {expected_image[:16]}…")

    return EnclaveVerdict(True, "ok", announcement=announcement, mrtd=verdict.mr_td,
                          rtmrs=dict(enumerate(verdict.rtmrs)), tcb_status=verdict.tcb_status)


@dataclass
class ConfidentialClient:
    """Talks to one enclave. Verify once, then seal many requests to the same announcement.

    The announcement is cached for the life of this object because re-verifying per request would
    add the ~40 ms quote fetch to every call for no gain — the key is ephemeral and the epoch is in
    the AEAD, so if the enclave rotates or reboots, ciphertexts sealed to the old key simply stop
    opening and `refresh()` is the fix. Failing loudly on a stale key beats silently trusting a new
    one that was never verified.
    """

    base_url: str
    approved: object = None          # serve.governance.Approved — the owner's signed pins
    approved_mrtd: set[str] | None = None
    approved_rtmr: dict[int, str] | None = None
    collateral: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    full_verification: bool = True
    insecure_skip_verify: bool = False
    _announcement: dict | None = field(default=None, repr=False)
    _verdict: EnclaveVerdict | None = field(default=None, repr=False)

    # ---- trust -------------------------------------------------------------------------------
    def refresh(self, *, fetch=None) -> EnclaveVerdict:
        """Fetch the announcement and decide whether to trust it."""
        payload = (fetch or self._http_get)(f"{self.base_url.rstrip('/')}/v1/enclave")
        announcement = payload["announcement"]
        quote_hex = payload.get("quote")

        if self.insecure_skip_verify:
            if quote_hex:
                raise ValueError(
                    "insecure_skip_verify was set but the enclave produced a quote; refusing to "
                    "discard real attestation evidence")
            verdict = EnclaveVerdict(True, "UNVERIFIED (insecure_skip_verify)",
                                     announcement=announcement)
        else:
            if not quote_hex:
                return self._store(EnclaveVerdict(False, "no quote: this is not an enclave"))
            mrtd, rtmr = self._pins()
            verdict = verify(announcement, bytes.fromhex(quote_hex), approved_mrtd=mrtd,
                             approved_rtmr=rtmr, collateral=self.collateral,
                             full=self.full_verification)
            if verdict.ok and self.approved is not None:
                # The epoch floor is the record's revocation lever: an enclave key compromised at
                # epoch N is retired by publishing min_epoch = N+1, and clients stop accepting it
                # without any change to the image or its measurements.
                if int(announcement.get("epoch", 0)) < self.approved.min_epoch:
                    return self._store(EnclaveVerdict(
                        False, f"epoch {announcement.get('epoch')} is below the owner's floor "
                               f"{self.approved.min_epoch}"))
        return self._store(verdict)

    def _pins(self) -> tuple[set[str] | None, dict[int, str] | None]:
        """Prefer the owner's signed record; fall back to whatever the caller passed.

        Both being empty is legitimate but weak — it reduces the guarantee to "some TDX enclave",
        which is fine for a smoke test and wrong for a user, and `verify()`'s docstring says so.
        """
        if self.approved is not None:
            return set(self.approved.mrtd), dict(self.approved.rtmr) or None
        return self.approved_mrtd, self.approved_rtmr

    def _store(self, verdict: EnclaveVerdict) -> EnclaveVerdict:
        self._verdict = verdict
        self._announcement = verdict.announcement if verdict.ok else None
        return verdict

    def _trusted(self) -> dict:
        """The announcement, or a refusal. A failed verification is STICKY.

        Re-fetching after a refusal would let a miner that failed verification simply be asked
        again — and worse, it would do so silently on the next `complete()` call, turning a
        deliberate rejection into a retry loop against the same untrusted host. Recovering requires
        an explicit `refresh()`.
        """
        if self._announcement is None:
            if self._verdict is not None:
                raise PermissionError(f"refusing to send: {self._verdict.reason}")
            verdict = self.refresh()
            if not verdict.ok:
                raise PermissionError(f"refusing to send: {verdict.reason}")
        return self._announcement

    # ---- the round trip ----------------------------------------------------------------------
    def complete(self, messages: list[dict], *, verify_mode: str | None = None,
                 max_tokens: int | None = None, temperature: float = 0.0,
                 suite: str = sealed.DEFAULT_SUITE, post=None) -> dict:
        """Seal, send, open. Returns the OpenAI-shaped response body.

        The reply key is minted here and dropped when this call returns, so one recorded
        ciphertext gives an attacker one answer even if the key later leaks.
        """
        announcement = self._trusted()
        reply_key = sealed.ReplyKey.generate(suite)

        body: dict = {"model": "thirtyspokes", "reply_to": reply_key.public(),
                      "messages": messages, "temperature": temperature}
        if verify_mode:
            body["verify"] = verify_mode
        if max_tokens:
            body["max_tokens"] = int(max_tokens)

        plaintext = json.dumps(body).encode()
        ciphertext = sealed.seal(announcement, plaintext)
        reply = (post or self._http_post)(f"{self.base_url.rstrip('/')}/v1/sealed", ciphertext)
        return json.loads(reply_key.open(reply, announcement["image_measurement"],
                                         int(announcement["epoch"])))

    # ---- transport ---------------------------------------------------------------------------
    def _http_get(self, url: str) -> dict:
        import httpx
        with httpx.Client(timeout=self.timeout, trust_env=False) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()

    def _http_post(self, url: str, ciphertext: bytes) -> bytes:
        import httpx
        with httpx.Client(timeout=self.timeout, trust_env=False) as c:
            r = c.post(url, content=ciphertext,
                       headers={"content-type": "application/octet-stream"})
            r.raise_for_status()
            return r.content
