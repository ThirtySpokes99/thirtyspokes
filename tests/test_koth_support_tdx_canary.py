from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "koth_support_tdx_canary.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("koth_support_tdx_canary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canary)

_SHADOW_SPEC = importlib.util.spec_from_file_location(
    "koth_support_tdx_shadow", _SCRIPT.with_name("koth_support_tdx_shadow.py")
)
assert _SHADOW_SPEC is not None and _SHADOW_SPEC.loader is not None
shadow = importlib.util.module_from_spec(_SHADOW_SPEC)
_SHADOW_SPEC.loader.exec_module(shadow)


def test_payload_completion_tolerates_interleaved_koth_prefix():
    serial = "\n".join([
        "KOTH[ audit noise ]-PROOF-CHUNK 1/1 e30=",  # {}
        "KOTH-TRACE-CHUNK 1/1 W10=",                  # []
    ])
    assert canary._chunks_emitted(serial)
    assert canary._payloads_complete(serial)


def test_payload_completion_rejects_missing_or_invalid_json():
    assert not canary._payloads_complete("KOTH-PROOF-CHUNK 1/1 e30=")
    assert not canary._chunks_emitted("KOTH-PROOF-CHUNK 1/1 e30=")
    serial = "\n".join([
        "KOTH-PROOF-CHUNK 1/1 e3A=",  # {p -- valid Base64, invalid JSON
        "KOTH-TRACE-CHUNK 1/1 W10=",
    ])
    assert canary._chunks_emitted(serial)
    assert not canary._payloads_complete(serial)
    with pytest.raises(RuntimeError, match="incomplete PROOF"):
        canary._decode_chunks("KOTH-PROOF-CHUNK 1/2 e30=", "PROOF")


def test_gzip_chunks_recover_a_digest_valid_repeated_copy():
    def line(kind, payload, *, corrupt=False):
        encoded = base64.b64encode(
            gzip.compress(json.dumps(payload).encode(), compresslevel=9, mtime=0)
        ).decode()
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        data = encoded[:-8] if corrupt else encoded
        return f"KOTH-{kind}-GZIP-CHUNK 1/1 sha256={digest} {data}"

    serial = "\n".join([
        line("PROOF", {}, corrupt=True),
        line("PROOF", {}),
        line("TRACE", [], corrupt=True),
        line("TRACE", []),
    ])
    assert canary._chunks_emitted(serial)
    assert canary._payloads_complete(serial)


def test_serial_compute_window_pairs_wall_and_monotonic_clocks():
    serial = "\n".join([
        "[    5.000000] audit: type=1130 audit(1005.000:1): res=success",
        "[   25.000000] python[1]: KOTH-DONE",
    ])
    assert canary._serial_compute_window(serial, created_unix=990.0) == pytest.approx(35.0)


def test_shadow_selects_only_usable_consecutive_epochs():
    assert shadow._select_epoch(7_564_227, None, 50) == 75_642
    assert shadow._select_epoch(7_564_270, None, 50) is None
    assert shadow._select_epoch(7_564_299, 75_642, 50) is None
    assert shadow._select_epoch(7_564_300, 75_642, 50) == 75_643
    with pytest.raises(RuntimeError, match="skipped"):
        shadow._select_epoch(7_564_400, 75_642, 50)
