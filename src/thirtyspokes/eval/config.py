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
