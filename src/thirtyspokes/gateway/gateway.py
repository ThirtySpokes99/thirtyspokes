"""The metering gateway — the trust anchor that makes cost un-forgeable.

Every miner model call goes through here. The gateway authenticates the miner
(hotkey signature), proxies to a provider backend, reads the REAL cost, and
returns a gateway-signed Receipt. Direct provider calls that skip the gateway
produce no receipt and are therefore unscoreable — miners cannot bypass it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from . import signing
from .receipts import Receipt
from .signing import Signer


@dataclass(frozen=True)
class CallRequest:
    hotkey: str
    nonce: str
    model: str
    messages: list
    params: dict
    miner_sig: str

    def _core(self) -> bytes:
        return signing.canonical(
            {"hotkey": self.hotkey, "nonce": self.nonce, "model": self.model,
             "messages": self.messages, "params": self.params})


@runtime_checkable
class ModelBackend(Protocol):
    def complete(self, model: str, messages: list, params: dict) -> tuple[str, int, int, float]:
        """-> (response_text, tokens_in, tokens_out, cost_usd)."""


class MockBackend:
    """Deterministic offline backend. Cost comes from a price table so the
    metering flow is exercised without a provider."""

    PRICES = {"tiny": 0.00005, "small": 0.0002, "mid": 0.001,
              "large": 0.006, "frontier": 0.02}  # $ per 1k tokens

    def _price(self, model: str) -> float:
        for k, v in self.PRICES.items():
            if k in model:
                return v
        return self.PRICES["mid"]

    def complete(self, model, messages, params):
        tin = max(1, len(signing.canonical(messages)) // 4)
        tout = int(params.get("max_tokens", 64))
        # a real agent's final step emits the answer through the gateway; the mock
        # echoes the last message so the answer↔trace binding is exercised.
        if params.get("finalize"):
            text = str(messages[-1]["content"])
        else:
            text = f"[{model} reply to {signing.sha256_hex(messages)[:8]}]"
        cost = (tin + tout) / 1000.0 * self._price(model)
        return text, tin, tout, cost


class OpenRouterBackend:  # pragma: no cover — live seam (needs OPENROUTER_API_KEY)
    """Real OpenRouter calls with REAL cost.

    Cost priority: (1) `usage.cost` returned when we request `usage={"include":true}`;
    (2) the `/generation?id=` stats endpoint; (3) a price-table fallback. Temperature
    defaults to 0 for reproducibility. Retries transient errors.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 timeout: float = 120.0, max_retries: int = 3, price_fn=None):
        import ssl  # noqa: PLC0415

        import certifi  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415
        # `trust_env=False` + an explicit CA bundle is a SECURITY boundary, not a preference.
        # The default client reads its trust store and proxy routing from the environment
        # (`httpx.create_ssl_context` passes `os.environ["SSL_CERT_FILE"]` straight to
        # `ssl.create_default_context`), and in the measured enclave the environment is fed from a
        # blob the MINER supplies. Setting `SSL_CERT_FILE` + `HTTPS_PROXY` therefore let a miner
        # MITM its own pool: fabricate answers, fabricate `usage.cost` (cost is read out of the
        # response body), pay nothing, and still emit a fully valid attested proof. Pinning the
        # bundled `certifi` roots means the miner still owns the network but cannot forge a cert,
        # which is what makes the metered cost and the graded answers mean anything.
        # `eval.config.scrub_network_env()` clears those vars too; this is the backstop that holds
        # even if something reaches the environment by another route.
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              timeout=timeout, max_retries=max_retries,
                              http_client=httpx.Client(
                                  trust_env=False, timeout=timeout,
                                  verify=ssl.create_default_context(cafile=certifi.where())))
        self._price_fn = price_fn   # model -> (prompt_$per1k, completion_$per1k)

    def complete(self, model, messages, params):
        p = {"temperature": 0.0, **params}
        p.pop("finalize", None)     # gateway-internal flag, not a provider param
        # `reasoning` is an OpenRouter body param, not a top-level kwarg, so it has to be merged
        # into extra_body. Without it a reasoning model spends its ENTIRE max_tokens budget thinking
        # and never emits an answer -- measured on SWE-bench Pro oracle prompts, where raising the
        # cap from 8k to 32k did not help and simply bought more thinking: kimi-k3 burned the full
        # 32k on 5 of 7 cells and returned nothing. Bounding thinking separately from output is the
        # only lever that works, and the failure is silent (an empty answer, graded wrong), so it
        # would otherwise read as a capability difference between models.
        body = {"usage": {"include": True}}
        reasoning = p.pop("reasoning", None)
        if reasoning is not None:
            body["reasoning"] = reasoning
        # A PROVIDER FAILURE IS DATA, NOT A CRASH — the convention everywhere else in this codebase,
        # and the enclave was the one place that ignored it. MEASURED: a live epoch ran 49 minutes,
        # completed its tasks, and died on `httpx ... .json()` raising JSONDecodeError because
        # OpenRouter returned a non-JSON body (an HTML error page or a truncated response). One bad
        # HTTP response destroyed the whole epoch's work and the miner earned nothing.
        #
        # The SDK's own retries do not cover this: it retries connection errors and 5xx statuses, not
        # a 200 whose body will not parse. So retry here, then DEGRADE: an unanswerable rung returns
        # an empty answer at zero cost, which is exactly what the cascade already handles — the
        # verifier rejects it and the ladder escalates.
        import time as _time
        last = None
        for attempt in range(3):
            try:
                r = self._client.chat.completions.create(
                    model=model, messages=messages, extra_body=body, **p)
                break
            except Exception as e:      # noqa: BLE001 — any provider failure, incl. unparseable bodies
                last = e
                if attempt < 2:
                    _time.sleep(1.5 * (attempt + 1))
        else:
            print(f"[pool] {model} failed after 3 attempts ({type(last).__name__}: "
                  f"{str(last)[:120]}) — treating as an unanswered rung", flush=True)
            return "", 0, 0, 0.0
        u = r.usage
        tin, tout = u.prompt_tokens, u.completion_tokens
        cost = None
        extra = getattr(u, "model_extra", None) or {}
        if extra.get("cost") is not None:
            cost = float(extra["cost"])
        if cost is None and self._price_fn is not None:
            pin, pout = self._price_fn(model)
            cost = tin / 1000 * pin + tout / 1000 * pout
        return r.choices[0].message.content or "", tin, tout, float(cost or 0.0)


class MeteringGateway:
    def __init__(self, backend: ModelBackend, registered: set[str]):
        self.backend = backend
        self.registered = set(registered)
        self._gw = Signer()
        self._receipts: dict[str, Receipt] = {}
        self._io: dict[str, tuple] = {}
        self._used_nonces: set[str] = set()
        self._n = 0

    @property
    def public_hex(self) -> str:
        return self._gw.public_hex

    def register(self, hotkey: str) -> None:
        self.registered.add(hotkey)

    def call(self, req: CallRequest) -> tuple[str, Receipt]:
        if not signing.verify(req.hotkey, req._core(), req.miner_sig):
            raise PermissionError("bad_miner_signature")
        if req.hotkey not in self.registered:
            raise PermissionError("hotkey_not_registered")
        if req.nonce in self._used_nonces:
            raise PermissionError("nonce_reused")
        self._used_nonces.add(req.nonce)

        text, tin, tout, cost = self.backend.complete(req.model, req.messages, req.params)
        cid = f"call-{self._n}"
        self._n += 1
        receipt = Receipt(
            call_id=cid, hotkey=req.hotkey, model=req.model,
            prompt_hash=signing.sha256_hex({"m": req.messages, "p": req.params}),
            response_hash=signing.sha256_hex(text), tokens_in=tin, tokens_out=tout,
            cost_usd=cost, ts=self._n,
        ).signed_by(self._gw)
        self._receipts[cid] = receipt
        self._io[cid] = (req.messages, text)
        return text, receipt

    def get_receipt(self, call_id: str) -> Receipt | None:
        return self._receipts.get(call_id)


def gateway_call(gw: MeteringGateway, miner: Signer, model: str, messages: list,
                 params: dict, nonce: str) -> tuple[str, Receipt]:
    """Miner-side SDK: build a signed request and call the gateway."""
    core = {"hotkey": miner.public_hex, "nonce": nonce, "model": model,
            "messages": messages, "params": params}
    req = CallRequest(miner.public_hex, nonce, model, messages, params,
                      miner.sign(signing.canonical(core)))
    return gw.call(req)
