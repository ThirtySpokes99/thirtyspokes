"""Real Intel TDX attestation — the production replacement for `mock_vendor_platform`.

Two halves, both hardware-real (no shared secret, unlike the mock TEE):

  * GENERATION (runs inside the confidential VM): `get_quote(report_data)` asks the
    Linux kernel's configfs TSM interface (`/sys/kernel/config/tsm/report`) for a
    genuine TDX v4 quote whose 64-byte REPORTDATA carries the proof's `report_data()`.
    The quote is signed by an Intel-provisioned attestation key and embeds the PCK
    certificate chain up to the Intel SGX Root CA.

  * VERIFICATION (runs anywhere — the validator): `verify_quote(...)` checks the full
    DCAP signature chain in pure Python (only `cryptography`):
        TD-quote sig  ←(attestation key)
        attestation key  ←(bound in QE report's report_data)
        QE report sig  ←(PCK leaf cert)
        PCK leaf  ←  Intel SGX Platform CA  ←  Intel SGX Root CA (pinned below)
    plus REPORTDATA == the expected payload hash and MRTD ∈ the owner-approved set.

Residual (needs live Intel PCS collateral, not done here): TCB-status / CRL revocation.
The cryptographic binding to authentic Intel silicon and to *this* proof IS checked.
See docs/DEPLOYING.md §Production.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils

from ..tee.attestation import Quote

# ---------------------------------------------------------------------------
# Intel SGX Root CA — the trust anchor. Fetched over TLS from
# certificates.trustedservices.intel.com and pinned here so verification needs
# no network. The root in every genuine quote's cert chain must equal this key.
# ---------------------------------------------------------------------------
INTEL_SGX_ROOT_CA_PEM = b"""-----BEGIN CERTIFICATE-----
MIICjzCCAjSgAwIBAgIUImUM1lqdNInzg7SVUr9QGzknBqwwCgYIKoZIzj0EAwIw
aDEaMBgGA1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENv
cnBvcmF0aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJ
BgNVBAYTAlVTMB4XDTE4MDUyMTEwNDUxMFoXDTQ5MTIzMTIzNTk1OVowaDEaMBgG
A1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENvcnBvcmF0
aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJBgNVBAYT
AlVTMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEC6nEwMDIYZOj/iPWsCzaEKi7
1OiOSLRFhWGjbnBVJfVnkY4u3IjkDYYL0MxO4mqsyYjlBalTVYxFP2sJBK5zlKOB
uzCBuDAfBgNVHSMEGDAWgBQiZQzWWp00ifODtJVSv1AbOScGrDBSBgNVHR8ESzBJ
MEegRaBDhkFodHRwczovL2NlcnRpZmljYXRlcy50cnVzdGVkc2VydmljZXMuaW50
ZWwuY29tL0ludGVsU0dYUm9vdENBLmRlcjAdBgNVHQ4EFgQUImUM1lqdNInzg7SV
Ur9QGzknBqwwDgYDVR0PAQH/BAQDAgEGMBIGA1UdEwEB/wQIMAYBAf8CAQEwCgYI
KoZIzj0EAwIDSQAwRgIhAOW/5QkR+S9CiSDcNoowLuPRLsWGf/Yi7GSX94BgwTwg
AiEA4J0lrHoMs+Xo5o/sX6O9QWxHRAvZUGOdRQ7cvqRXaqI=
-----END CERTIFICATE-----"""

TSM_REPORT_DIR = "/sys/kernel/config/tsm/report"
TDX_GUEST_DEV = "/dev/tdx_guest"

_TDX_TEE_TYPE = 0x81
_HEADER_LEN = 48
_BODY_LEN = 584          # TD10 report body
# offsets within the TD10 body
_MRTD_OFF = 136
_RTMR0_OFF = 328
_REPORTDATA_OFF = 520
# report_data field offset within an SGX report body (the QE report)
_SGX_REPORTDATA_OFF = 320


# ---------------------------------------------------------------------------
# Generation (inside the CVM)
# ---------------------------------------------------------------------------
def tdx_available() -> bool:
    """True on an Intel TDX guest exposing the configfs TSM quote interface."""
    return os.path.isdir(TSM_REPORT_DIR) and os.path.exists(TDX_GUEST_DEV)


def get_quote(report_data: bytes) -> bytes:
    """Return a genuine TDX v4 quote whose REPORTDATA == `report_data` (64 bytes).

    Uses the kernel configfs TSM interface: write REPORTDATA to `inblob`, read the
    signed quote from `outblob`. No privilege beyond access to configfs is needed.
    """
    if len(report_data) != 64:
        raise ValueError(f"report_data must be 64 bytes, got {len(report_data)}")
    if not tdx_available():
        raise RuntimeError("not a TDX guest (no configfs TSM interface)")
    # deterministic entry name (no Date/random needed): bound to the payload
    entry = os.path.join(TSM_REPORT_DIR, "koth_" + hashlib.sha256(report_data).hexdigest()[:16])
    made = False
    try:
        try:
            os.mkdir(entry)
            made = True
        except FileExistsError:
            pass
        with open(os.path.join(entry, "inblob"), "wb") as f:
            f.write(report_data)
        with open(os.path.join(entry, "outblob"), "rb") as f:
            return f.read()
    finally:
        if made:
            try:
                os.rmdir(entry)
            except OSError:
                pass


def report_data_from_hash(payload_hash_hex: str) -> bytes:
    """Embed a 32-byte payload hash (the proof's `report_data()`) into the 64-byte
    TDX REPORTDATA field (hash in the low 32 bytes, zero-padded)."""
    h = bytes.fromhex(payload_hash_hex)
    if len(h) != 32:
        raise ValueError("expected a 32-byte (64 hex char) payload hash")
    return h + b"\x00" * 32


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedQuote:
    version: int
    tee_type: int
    mr_td: bytes
    rtmrs: tuple[bytes, ...]
    report_data: bytes            # 64-byte REPORTDATA from the TD body
    signed_region: bytes          # header || TD body (what the att key signs)
    td_sig: bytes                 # 64-byte raw r||s over signed_region
    att_pub: bytes                # 64-byte raw X||Y attestation key
    qe_report: bytes              # 384-byte SGX QE report
    qe_sig: bytes                 # 64-byte raw r||s over qe_report (by PCK leaf)
    qe_auth: bytes
    cert_pems: tuple[bytes, ...]  # PCK leaf, intermediate, root


def _u16(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 2], "little")


def _u32(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 4], "little")


def parse_quote(q: bytes) -> ParsedQuote:
    version = _u16(q, 0)
    tee_type = _u32(q, 4)
    body = q[_HEADER_LEN:_HEADER_LEN + _BODY_LEN]
    mr_td = body[_MRTD_OFF:_MRTD_OFF + 48]
    rtmrs = tuple(body[_RTMR0_OFF + 48 * i:_RTMR0_OFF + 48 * (i + 1)] for i in range(4))
    report_data = body[_REPORTDATA_OFF:_REPORTDATA_OFF + 64]
    signed_region = q[0:_HEADER_LEN + _BODY_LEN]

    sig_len = _u32(q, _HEADER_LEN + _BODY_LEN)          # @632
    sd = q[_HEADER_LEN + _BODY_LEN + 4:_HEADER_LEN + _BODY_LEN + 4 + sig_len]
    td_sig = sd[0:64]
    att_pub = sd[64:128]
    # outer certification_data (type 6 = ECDSA sig aux data)
    outer_size = _u32(sd, 130)
    cert_data = sd[134:134 + outer_size]
    qe_report = cert_data[0:384]
    qe_sig = cert_data[384:448]
    qe_auth_size = _u16(cert_data, 448)
    qe_auth = cert_data[450:450 + qe_auth_size]
    o = 450 + qe_auth_size
    inner_size = _u32(cert_data, o + 2)                 # inner type @o (==5 PCK chain)
    pem_blob = cert_data[o + 6:o + 6 + inner_size]
    cert_pems = tuple(re.findall(
        rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", pem_blob, re.S))
    return ParsedQuote(version, tee_type, mr_td, rtmrs, report_data, signed_region,
                       td_sig, att_pub, qe_report, qe_sig, qe_auth, cert_pems)


# ---------------------------------------------------------------------------
# Verification (DCAP crypto chain, pure Python)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TDXVerdict:
    ok: bool
    reason: str
    mr_td: str = ""
    rtmrs: tuple[str, ...] = ()
    report_data: bytes = b""
    tcb_status: str = ""              # full-DCAP path only: UpToDate / OutOfDate / ...
    advisory_ids: tuple[str, ...] = ()


def _pub_from_xy(raw64: bytes) -> ec.EllipticCurvePublicKey:
    x = int.from_bytes(raw64[:32], "big")
    y = int.from_bytes(raw64[32:], "big")
    return ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()


def _verify_raw_ecdsa(pub: ec.EllipticCurvePublicKey, raw_sig64: bytes, msg: bytes) -> bool:
    r = int.from_bytes(raw_sig64[:32], "big")
    s = int.from_bytes(raw_sig64[32:], "big")
    der = asym_utils.encode_dss_signature(r, s)
    try:
        pub.verify(der, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _cert_signed_by(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    try:
        issuer.public_key().verify(
            child.signature, child.tbs_certificate_bytes,
            ec.ECDSA(child.signature_hash_algorithm))
        return True
    except InvalidSignature:
        return False


def verify_quote(q: bytes, *, expect_report_data: bytes,
                 approved_mrtd: set[str] | None = None,
                 intel_root_pem: bytes = INTEL_SGX_ROOT_CA_PEM) -> TDXVerdict:
    """Full DCAP verification of a TDX v4 quote. Returns a verdict with the attested
    MRTD/RTMRs on success. `approved_mrtd` (owner-pinned image measurements) is checked
    when non-empty; pass None to only surface the MRTD without gating on it."""
    try:
        p = parse_quote(q)
    except Exception as e:  # noqa: BLE001 — malformed quote is a verdict, not a crash
        return TDXVerdict(False, f"parse_error:{type(e).__name__}")

    if p.version != 4 or p.tee_type != _TDX_TEE_TYPE:
        return TDXVerdict(False, f"not_tdx_v4:v{p.version}/tee{p.tee_type:#x}")
    if p.report_data != expect_report_data:
        return TDXVerdict(False, "report_data_mismatch")
    if len(p.cert_pems) < 3:
        return TDXVerdict(False, f"cert_chain_len:{len(p.cert_pems)}")

    leaf = x509.load_pem_x509_certificate(p.cert_pems[0])
    inter = x509.load_pem_x509_certificate(p.cert_pems[1])
    root = x509.load_pem_x509_certificate(p.cert_pems[2])
    intel_root = x509.load_pem_x509_certificate(intel_root_pem)

    # 1. cert chain: leaf <- intermediate <- root, root self-signed
    if not _cert_signed_by(leaf, inter):
        return TDXVerdict(False, "pck_leaf_sig")
    if not _cert_signed_by(inter, root):
        return TDXVerdict(False, "pck_intermediate_sig")
    if not _cert_signed_by(root, root):
        return TDXVerdict(False, "root_not_self_signed")
    # 2. pin the chain root to the authentic Intel SGX Root CA key
    if root.public_key().public_numbers() != intel_root.public_key().public_numbers():
        return TDXVerdict(False, "root_not_intel")
    # 3. PCK leaf signs the QE report
    if not _verify_raw_ecdsa(leaf.public_key(), p.qe_sig, p.qe_report):
        return TDXVerdict(False, "qe_report_sig")
    # 4. QE report binds the attestation key: report_data[:32] == SHA256(att_pub || qe_auth)
    qe_rd = p.qe_report[_SGX_REPORTDATA_OFF:_SGX_REPORTDATA_OFF + 64]
    if qe_rd[:32] != hashlib.sha256(p.att_pub + p.qe_auth).digest():
        return TDXVerdict(False, "qe_key_binding")
    # 5. the attestation key signs the TD quote (header || body)
    if not _verify_raw_ecdsa(_pub_from_xy(p.att_pub), p.td_sig, p.signed_region):
        return TDXVerdict(False, "td_quote_sig")
    # 6. owner-approved measurement gate
    mr_td_hex = p.mr_td.hex()
    if approved_mrtd and mr_td_hex not in approved_mrtd:
        return TDXVerdict(False, "mrtd_not_approved")

    return TDXVerdict(True, "ok", mr_td_hex, tuple(r.hex() for r in p.rtmrs), p.report_data)


# ---------------------------------------------------------------------------
# Platform seam — a drop-in for tee.attestation.Platform on real hardware.
# ---------------------------------------------------------------------------
class TDXPlatform:
    """Real hardware attestation root. `.quote(measurement, report_data_hex)` returns a
    `Quote` whose `platform_sig` carries the base64 TDX quote (over the payload hash);
    verify with `verify_tdx_quote_field`. Drop-in for `mock_vendor_platform()`'s Platform
    at the `attested_by` call site inside the CVM."""

    @property
    def public_hex(self) -> str:
        # identity lives in the quote's Intel-rooted cert chain, not a standalone key
        return "tdx-hardware-root"

    def quote(self, measurement: str, report_data: str) -> Quote:
        raw = get_quote(report_data_from_hash(report_data))
        return Quote(measurement=measurement, report_data=report_data,
                     platform_sig="tdx:" + base64.b64encode(raw).decode())


def verify_tdx_quote_field(quote: Quote, *, approved_mrtd: set[str] | None = None) -> TDXVerdict:
    """Verify a `Quote` produced by `TDXPlatform`: decode the raw TDX quote from
    `platform_sig` and check it binds exactly `quote.report_data` (the proof hash)."""
    if not quote.platform_sig.startswith("tdx:"):
        return TDXVerdict(False, "not_a_tdx_quote")
    raw = base64.b64decode(quote.platform_sig[4:])
    return verify_quote(raw, expect_report_data=report_data_from_hash(quote.report_data),
                        approved_mrtd=approved_mrtd)


# ---------------------------------------------------------------------------
# Full DCAP verification (H1) — TCB status + QE identity + CRL revocation.
#
# The pure-Python `verify_quote` above proves the crypto chain to the Intel root +
# the payload binding, but NOT that the platform's TCB is current / not revoked (that
# needs Intel PCS collateral). `verify_quote_full` adds that via `dcap-qvl` (Phala's
# maintained Rust QVL), then re-applies our own MRTD/RTMR gates + payload binding.
# ---------------------------------------------------------------------------

# Owner TCB policy: which reported TCB levels are acceptable. UpToDate is always fine;
# SWHardeningNeeded is commonly accepted (a software mitigation, not a platform break).
# Everything else (OutOfDate*, ConfigurationNeeded*, Revoked) is rejected by default.
DEFAULT_TCB_ACCEPT: frozenset[str] = frozenset({"UpToDate", "SWHardeningNeeded"})


def _tcb_reason(status: str) -> str:
    return "tcb_" + re.sub(r"(?<!^)(?=[A-Z])", "_", status).lower()   # OutOfDate -> tcb_out_of_date


def verify_quote_full(
    raw: bytes,
    *,
    expect_report_data: bytes,
    approved_mrtd: set[str] | None = None,
    approved_rtmr: dict[int, str] | None = None,
    tcb_accept: frozenset[str] = DEFAULT_TCB_ACCEPT,
    collateral: str | None = None,
    now: int | None = None,
    pccs_url: str | None = None,
) -> TDXVerdict:
    """Production TDX verification: our binding/MRTD/RTMR gates + dcap-qvl's full DCAP
    (cert chain, CRL revocation, QE identity, cert validity, and the platform TCB status).

    `collateral` = an owner-pinned `QuoteCollateralV3.to_json()` string for deterministic
    offline verification; if None, it is fetched + cached per FMSPC (see `collateral.py`).
    `now` = the verification instant (unix seconds; defaults to wall-clock) — pass a fixed
    value to verify a captured quote against captured collateral reproducibly.
    """
    import dcap_qvl

    from . import collateral as _col

    # 1. cheap local gates first (no network): structure + payload binding.
    try:
        p = parse_quote(raw)
    except Exception as e:  # noqa: BLE001
        return TDXVerdict(False, f"parse_error:{type(e).__name__}")
    if p.version != 4 or p.tee_type != _TDX_TEE_TYPE:
        return TDXVerdict(False, f"not_tdx_v4:v{p.version}/tee{p.tee_type:#x}")
    if p.report_data != expect_report_data:
        return TDXVerdict(False, "report_data_mismatch")

    # 2. full DCAP via dcap-qvl (chain -> Intel root, CRL, QE identity, TCB status).
    when = int(now) if now is not None else int(__import__("time").time())
    try:
        col = (dcap_qvl.QuoteCollateralV3.from_json(collateral) if collateral is not None
               else _col.get_collateral_cached(raw, pccs_url=pccs_url or dcap_qvl.PHALA_PCCS_URL,
                                                now=when))
    except Exception as e:  # noqa: BLE001 — collateral fetch/parse failure is a verdict
        return TDXVerdict(False, f"collateral_error:{type(e).__name__}")
    try:
        rep = dcap_qvl.verify(raw, col, when)
    except Exception as e:  # noqa: BLE001 — chain/CRL/expiry failure raises
        return TDXVerdict(False, f"dcap_verify_failed:{type(e).__name__}")

    status = rep.status
    if status not in tcb_accept:
        return TDXVerdict(False, _tcb_reason(status), p.mr_td.hex(),
                          tuple(r.hex() for r in p.rtmrs), p.report_data, status,
                          tuple(rep.advisory_ids))

    # 3. our owner-approved measurement gates (MRTD + any pinned RTMRs).
    mr_td_hex = p.mr_td.hex()
    if approved_mrtd and mr_td_hex not in approved_mrtd:
        return TDXVerdict(False, "mrtd_not_approved", mr_td_hex,
                          tuple(r.hex() for r in p.rtmrs), p.report_data, status)
    rtmr_hex = [r.hex() for r in p.rtmrs]
    if approved_rtmr:
        for idx, expected in approved_rtmr.items():
            if rtmr_hex[idx] != expected:
                return TDXVerdict(False, f"rtmr{idx}_not_approved", mr_td_hex,
                                  tuple(rtmr_hex), p.report_data, status)

    return TDXVerdict(True, "ok", mr_td_hex, tuple(rtmr_hex), p.report_data,
                      status, tuple(rep.advisory_ids))


def verify_tdx_quote_field_full(quote: Quote, **kw) -> TDXVerdict:
    """Full-DCAP counterpart of `verify_tdx_quote_field` for a `TDXPlatform` `Quote`."""
    if not quote.platform_sig.startswith("tdx:"):
        return TDXVerdict(False, "not_a_tdx_quote")
    raw = base64.b64decode(quote.platform_sig[4:])
    return verify_quote_full(raw, expect_report_data=report_data_from_hash(quote.report_data), **kw)
