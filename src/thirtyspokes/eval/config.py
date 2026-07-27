"""Live-run configuration: OpenRouter key, the model pool, and cost fallbacks.

Everything the real evaluation needs to be flipped on with one env var. Nothing
here is exercised offline; it's the switchboard for the live seam.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# The sanctioned pool, expressed as OpenRouter model ids across the cost spectrum.
# Miners route among these through the gateway; edit to curate the pool (§5).
DEFAULT_POOL: dict[str, str] = {
    "cheap": "meta-llama/llama-3.1-8b-instruct",
    "mid": "deepseek/deepseek-v3.2-exp",
    "strong": "google/gemini-2.5-pro",
    "frontier": "anthropic/claude-opus-4.7",
}

# Fallback $/1k-token prices, used only if OpenRouter doesn't return usage.cost.
# (prompt, completion) — approximate; the gateway prefers the real cost.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.1-8b-instruct": (0.00002, 0.00003),
    "qwen/qwen-2.5-72b-instruct": (0.0003, 0.0003),
    "anthropic/claude-3.5-sonnet": (0.003, 0.015),
    "openai/gpt-4o": (0.0025, 0.01),
}


@dataclass
class LiveConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    pool: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_POOL))
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: float = 120.0
    max_retries: int = 3

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it to run the live evaluation:\n"
                "  export OPENROUTER_API_KEY=sk-or-...")


def price_for(model: str) -> tuple[float, float]:
    return PRICE_TABLE.get(model, (0.001, 0.001))


# --- enclave secret intake (security boundary) ---------------------------------------------------
# The measured runtime receives its pool credential from a blob the MINER supplies (GCE metadata
# `koth-secrets`). That blob is attacker-controlled, so it is read through an ALLOWLIST — never
# splatted wholesale into the environment.
#
# Why this is load-bearing: the runtime's HTTP client reads its TLS trust store and proxy routing
# FROM THE ENVIRONMENT. `httpx.create_ssl_context` uses `os.environ["SSL_CERT_FILE"]` directly as its
# CA file when `trust_env` is on. So two extra lines in the secrets blob —
#     SSL_CERT_FILE=/run/koth/agent/weights.bin     (a CA smuggled in as "weights")
#     HTTPS_PROXY=http://127.0.0.1:8080
# — made the enclave trust exactly one root, the miner's, and route the pool through the miner's own
# proxy. Every "OpenRouter" response was then authored by the miner: arbitrary answers (inject the
# memorized gold), arbitrary `usage.cost` and token counts (cost is read from the response body), at
# zero real spend. The resulting proof is CRYPTOGRAPHICALLY GENUINE — real quote, real measured
# runtime, real artifact binding, answers that match the logged trace — so grading, the budget
# ceiling, the cost tiebreak and the grounding check all pass while measuring nothing real.
#
# The TEE proves the approved code ran on these exact bytes. It cannot prove where the inputs came
# from; that has to be closed here, at the boundary.
ALLOWED_SECRET_KEYS = frozenset({"OPENROUTER_API_KEY"})

# Env vars that redirect or downgrade outbound TLS/HTTP. Cleared before any pool client is built, so
# a value reaching the environment by any other route still cannot move the trust anchor.
UNSAFE_NET_ENV: tuple[str, ...] = (
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSLKEYLOGFILE",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_PROXY",
)


def scrub_network_env(env: dict | None = None) -> list[str]:
    """Remove every TLS/proxy override from the environment. Returns what was dropped (for the
    attested console log — a miner attempting this should be visible, not silently ignored)."""
    e = os.environ if env is None else env
    return [k for k in UNSAFE_NET_ENV if e.pop(k, None) is not None]


def load_secrets_env(path: str, env: dict | None = None) -> tuple[list[str], list[str]]:
    """Load the miner-supplied secrets blob through `ALLOWED_SECRET_KEYS`.

    Returns `(accepted, rejected)` key names. Never returns values, so the caller can log the
    outcome without leaking the credential."""
    e = os.environ if env is None else env
    accepted, rejected = [], []
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return accepted, rejected
    for line in lines:
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k in ALLOWED_SECRET_KEYS:
            e[k] = v.strip().strip('"').strip("'")
            accepted.append(k)
        else:
            rejected.append(k)
    return accepted, rejected
