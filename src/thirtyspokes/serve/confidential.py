"""In-enclave confidential serving (docs/CONFIDENTIAL_SERVING.md §3, §5).

This is where the privacy claim is either structural or merely asserted, so the arrangement of this
module matters more than its length. The owner's image owns every step that touches plaintext —
open, encode, call the provider, seal — and the miner's routing model is reached through exactly
one function, `_route`, which accepts a feature vector and returns an action index.

    ciphertext ──► gate ──► open ──► encode ──► _route(features) ──► ladder ──► seal ──► receipt
                                       │            ▲
                                  plaintext         │  the ONLY call into miner code,
                                  stops here ───────┘  and it takes no text

`koth/confine.py` (removed with v2, 2026-09-07) documented the residual this arrangement removes: *"the agent can embed the
nonce/task in a legitimate prompt to the pinned model, reaching a confederate through the
sanctioned channel."* That was survivable when the prompt was a public benchmark question. With a
user's private request it is the whole ballgame, and confinement cannot close it — a sandbox stops
the network, not code that legitimately holds the plaintext and encodes it into a legitimate call.
Withholding the text is the only closure, which is why `_route` takes an ndarray and why
`test_confidential.py` asserts that no byte of the request ever reaches the policy.

WHAT ESCALATION CAN AND CANNOT DO HERE. The measured +13.39-point gain from verification
(docs/router-benchmarks) comes from escalating on *correctness*, which needs a task with a gold
answer or a mechanical checker. Arbitrary user traffic has neither, so the default trigger is
well-formedness — an empty or unparseable answer — exactly as the beta serving path does today. A
request may opt into a stronger check it can specify for itself (`verify: "json"`), and where a
caller can supply one, the stronger result follows. Claiming the benchmark's verification gain on
free-form traffic would be dishonest, so the code does not pretend to it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from ..gateway import signing
from ..koth import sealed
from ..koth.orkey import KeyGate
from ..koth.sealed import EnclaveKey

STOP = -1


def _non_empty(text: str) -> bool:
    return bool(text and text.strip())


def _valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:  # noqa: BLE001 — any parse failure is a failed check
        return False


VERIFIERS = {"non_empty": _non_empty, "json": _valid_json}


@dataclass
class ServeResult:
    sealed_response: bytes
    receipt: dict


@dataclass
class ConfidentialServer:
    """One enclave's serving loop.

    `policy(features, state) -> int` is the MINER's model and is deliberately typed to receive
    numbers. `encode(list[str]) -> ndarray` is the owner's pinned encoder. `backend.complete` is
    the owner's hardened provider client (`trust_env=False`, pinned roots), never one the miner
    can influence.
    """

    enclave_key: EnclaveKey
    gate: KeyGate
    policy: object
    backend: object
    ladder: tuple[str, ...]
    encode: object
    signer: object = None
    max_tokens: int = 4096
    request_timeout: float = 120.0
    _seq: int = field(default=0, repr=False)

    # ---- the miner boundary ------------------------------------------------------------------
    def _route(self, features: np.ndarray, tried: np.ndarray, failed: np.ndarray,
               spend_ratio: float) -> int:
        """The ONE call into miner-supplied code. Takes numbers; returns an action.

        Everything the miner's model learns about the request arrives through `features`, which is
        the pinned encoder's output — no text, no metadata, no identifiers. A malformed or
        out-of-range action is treated as STOP rather than trusted, because the model is untrusted
        code and an index is an index.
        """
        state = np.concatenate([features, tried, failed, np.array([spend_ratio])])
        try:
            action = int(self.policy(state))
        except Exception:  # noqa: BLE001 — a broken miner model must not take the request down
            return STOP
        if action < 0 or action >= len(self.ladder) or tried[action] > 0:
            return STOP
        return action

    # ---- the serving path --------------------------------------------------------------------
    def handle(self, ciphertext: bytes, *, now=None) -> ServeResult:
        t0 = (now or time.time)()
        self.gate.require()                       # fail closed before anything is decrypted

        plaintext = self.enclave_key.open(ciphertext)
        request = json.loads(plaintext)
        messages = request.get("messages") or []
        if not messages:
            raise ValueError("messages must not be empty")
        prompt = _last_user(messages)
        verify = VERIFIERS.get(request.get("verify") or "non_empty", _non_empty)

        features = np.asarray(self.encode([prompt]), dtype=float)[0]
        k = len(self.ladder)
        tried, failed = np.zeros(k), np.zeros(k)
        spend, attempts, answer, served = 0.0, [], None, None
        params = {"max_tokens": int(request.get("max_tokens") or self.max_tokens),
                  "temperature": request.get("temperature", 0.0),
                  "_timeout": self.request_timeout}

        for _ in range(k):
            action = self._route(features, tried, failed, spend / max(self.gate_budget(), 1e-9))
            if action == STOP:
                break
            model = self.ladder[action]
            tried[action] = 1.0
            try:
                text, tin, tout, cost = self.backend.complete(model, messages, dict(params))
            except Exception as exc:  # noqa: BLE001 — a dead rung is not a dead request
                failed[action] = 1.0
                attempts.append({"model": model, "ok": False,
                                 "error": f"{type(exc).__name__}"[:60]})
                continue
            spend += float(cost)
            ok = verify(text)
            attempts.append({"model": model, "ok": ok, "tokens_in": tin, "tokens_out": tout,
                             "cost_usd": float(cost)})
            if ok:
                answer, served = text, model
                break
            failed[action] = 1.0
            answer, served = answer or text, served or model   # keep the best we have

        if answer is None:
            raise RuntimeError("every rung failed or the policy stopped before any attempt")

        response = json.dumps({"model": "thirtyspokes", "served_model": served,
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": answer}}]})
        # Sealed to the CLIENT's ephemeral reply key, not to the enclave's own. Sealing to
        # ourselves would produce a response only this enclave can read — a mistake unit tests
        # hide, because a test can open it with an enclave key no real client possesses.
        reply_to = request.get("reply_to")
        if not reply_to:
            raise ValueError("request carries no reply_to key; the response would be undeliverable")
        sealed_response = sealed.seal_reply(reply_to, response.encode(),
                                            self.enclave_key.image_measurement,
                                            self.enclave_key.epoch)

        self._seq += 1
        receipt = self._receipt(request_hash=signing.sha256_hex(plaintext),
                                response_hash=signing.sha256_hex(response.encode()),
                                served=served, attempts=attempts, spend=spend,
                                latency=(now or time.time)() - t0)
        return ServeResult(sealed_response=sealed_response, receipt=receipt)

    def gate_budget(self) -> float:
        v = self.gate.check()
        return float(v.limit_remaining or 0.0)

    def _receipt(self, *, request_hash, response_hash, served, attempts, spend, latency) -> dict:
        """The public record. Hashes, decisions and cost — never content.

        `request_hash` is what lets a USER dispute one of their own requests by revealing their own
        plaintext. It is deliberately not a capability the miner has: the miner cannot produce the
        preimage, so publishing the hash discloses nothing to it.
        """
        body = {
            "v": 1, "seq": self._seq, "ts": round(time.time(), 3),
            "image_measurement": self.enclave_key.image_measurement,
            "epoch": self.enclave_key.epoch,
            "enclave_key": self.enclave_key.public_bytes.hex(),
            "request_hash": request_hash, "response_hash": response_hash,
            "served_model": served, "escalated": len(attempts) > 1,
            "decision_path": [a["model"] for a in attempts],
            "attempts": attempts, "cost_usd": round(spend, 8),
            "latency_s": round(latency, 3),
            **self.gate.check().receipt_fields(),
        }
        if self.signer is not None:
            body["sig"] = self.signer.sign(signing.canonical(body))
        return body


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return str(messages[-1].get("content") or "")


def verify_receipt(receipt: dict, public_hex: str) -> bool:
    """Anyone can check a receipt chains to the enclave key that signed it."""
    sig = receipt.get("sig")
    if not sig:
        return False
    body = {k: v for k, v in receipt.items() if k != "sig"}
    return signing.verify(public_hex, signing.canonical(body), sig)
