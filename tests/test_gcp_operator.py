from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from thirtyspokes.koth import gcp_operator


def _chunk(kind: str, payload, *, corrupt: bool = False) -> str:
    encoded = base64.b64encode(
        gzip.compress(json.dumps(payload).encode(), compresslevel=9, mtime=0)
    ).decode()
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    if corrupt:
        encoded = encoded[:-8]
    return f"KOTH-{kind}-GZIP-CHUNK 1/1 sha256={digest} {encoded}"


def test_operator_recovers_digest_valid_repeated_serial_chunks():
    serial = "\n".join([
        _chunk("PROOF", {"proof": 1}, corrupt=True),
        _chunk("PROOF", {"proof": 1}),
        _chunk("TRACE", [{"call": 1}], corrupt=True),
        _chunk("TRACE", [{"call": 1}]),
    ])
    proof, trace = gcp_operator.decode_payloads(serial)
    assert json.loads(proof) == {"proof": 1}
    assert json.loads(trace) == [{"call": 1}]


def test_operator_accepts_the_legacy_v14_full_line_payloads():
    encode = lambda value: base64.b64encode(json.dumps(value).encode()).decode()  # noqa: E731
    serial = f"KOTH-PROOF {encode({'proof': 1})}\nKOTH-TRACE {encode([])}\nKOTH-DONE"
    proof, trace = gcp_operator.decode_payloads(serial)
    assert json.loads(proof) == {"proof": 1}
    assert json.loads(trace) == []


def test_operator_aborts_early_guest_failure_and_always_deletes(monkeypatch, tmp_path: Path):
    deleted = []

    def delete(_zone, name):
        deleted.append(name)

    def gcloud(_zone, *args, **_kwargs):
        if "create" in args:
            return subprocess.CompletedProcess(args, 0, "created", "")
        return subprocess.CompletedProcess(args, 0, "KOTH-ERROR SandboxError: Broken pipe", "")

    monkeypatch.setattr(gcp_operator, "_delete_instance", delete)
    monkeypatch.setattr(gcp_operator, "_gcloud", gcloud)
    monkeypatch.setattr(gcp_operator.time, "sleep", lambda _seconds: None)

    metadata = tmp_path / "metadata"
    metadata.write_text("x")
    with pytest.raises(RuntimeError, match="early guest failure"):
        gcp_operator._boot_once(
            zone="zone", image="image", machine_type="c3-standard-4", name="vm",
            epoch=1, nonce="nonce", hotkey="hotkey", pool=["model"], n_per_bench=8,
            source_b64=metadata, weights_b64=metadata, secrets=metadata,
            deadline=1, poll=0, log_path=tmp_path / "serial.log",
        )
    assert deleted == ["vm", "vm"]
