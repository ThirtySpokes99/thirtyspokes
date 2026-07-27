"""The owner-pinned pool (KOTH v2, WS1).

The subnet owner pins an allow-list of models; the miner's routing/orchestration
plugin may call ONLY those (any other model -> UnpinnedModelError, so there is no
external-model smuggling). `PinnedBackend` is an additive ModelBackend wrapper — it
does NOT modify the shared `tee.MeteringProxy`; cost is still metered by the proxy
over whatever the pinned backend returns (real OpenRouter `usage.cost` in production).

`MockPool` is the offline stand-in: 2-3 models at DIFFERENTIATED (accuracy, price) that
return GRADEABLE answers to the synthetic benchmark prompts, so an honest router that
picks a better model actually scores higher (without this the whole sim is degenerate).
"""

from __future__ import annotations

import ast
import operator
import re

from ..gateway import signing
from ..gateway.gateway import ModelBackend


class UnpinnedModelError(Exception):
    pass


class PinnedBackend:
    """Wraps a backend and rejects any model outside the owner allow-list."""

    def __init__(self, inner: ModelBackend, allowed: set[str]):
        self.inner = inner
        self.allowed = set(allowed)

    def complete(self, model: str, messages: list, params: dict):
        if model not in self.allowed:
            raise UnpinnedModelError(model)
        return self.inner.complete(model, messages, params)


# --- offline solver for the synthetic benchmark prompts (so the pool can answer) ---
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.USub: operator.neg}


def _safe_arith(expr: str) -> int | None:
    """Evaluate an integer +-* () expression (only)."""
    try:
        def ev(n):
            if isinstance(n, ast.Expression): return ev(n.body)
            if isinstance(n, ast.Constant): return n.value
            if isinstance(n, ast.BinOp): return _OPS[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp): return _OPS[type(n.op)](ev(n.operand))
            raise ValueError
        return int(ev(ast.parse(expr, mode="eval")))
    except Exception:
        return None


def solve_synthetic(prompt: str) -> str:
    """Reference solver for koth's synthetic benchmark prompts (math/MC/SWE)."""
    if prompt.startswith("Compute ") and prompt.rstrip().endswith("."):
        v = _safe_arith(prompt[len("Compute "):].rstrip("."))
        return str(v) if v is not None else "?"
    m = re.search(r"value of (.+?)\?", prompt)  # MMLU/GPQA MC
    if m:
        val = _safe_arith(m.group(1))
        for line in prompt.splitlines():
            om = re.match(r"([A-D])\)\s*(-?\d+)", line.strip())
            if om and val is not None and int(om.group(2)) == val:
                return om.group(1)
        return "A"
    if prompt.startswith("[SWE]"):
        return "diff --git a/fix.py b/fix.py\n@@ -1 +1 @@\n-old\n+new  # resolves\n"
    return "?"


class MockPool:
    """Differentiated offline pool: model -> (accuracy, price/1k). Returns the solved
    answer with the model's accuracy (deterministic per prompt), else a wrong answer."""

    def __init__(self, models: dict[str, tuple[float, float]] | None = None):
        self.models = models or {"cheap": (0.55, 0.00005), "mid": (0.80, 0.001), "strong": (1.0, 0.02)}

    @property
    def allowed(self) -> set[str]:
        return set(self.models)

    def backend(self) -> "MockPool":
        return self

    def complete(self, model: str, messages: list, params: dict):
        acc, price = self.models[model]
        prompt = str(messages[-1]["content"])
        # deterministic "does this model get it right": hash(model|prompt) < acc
        h = int(signing.sha256_hex(f"{model}|{prompt}")[:8], 16) / 0xFFFFFFFF
        ans = solve_synthetic(prompt) if h < acc else "wrong"
        tin = max(1, len(signing.canonical(messages)) // 4)
        # bill what was actually PRODUCED, not the cap. `max_tokens` is a ceiling — a real provider
        # charges for the tokens it emitted — and billing the ceiling made an agent's cost depend on
        # a number it never spent. That matters now the ranked benchmark is code: a sane agent asks
        # for a 16k ceiling so a program is not truncated mid-function, and under cap-billing that
        # alone put the reference agent 32x over budget while emitting a two-token answer.
        tout = min(int(params.get("max_tokens", 32)), max(1, len(ans) // 4 + 1))
        cost = (tin + tout) / 1000.0 * price
        return ans, tin, tout, cost
