"""Sandboxed fresh-probe of untrusted miner code (review WS0 RCE fix + H3 no-egress).

The fresh-probe re-runs the miner's artifact to detect memorization/copying. Doing so
in-process (`load_agent` → `exec`) ran arbitrary miner Python inside the validator with the
wallet key + gold in scope. This runs it via `koth/confine.py`: a network-isolated child with
a scrubbed env, resource limits, and (hardened path) tmpfs-hidden secret dirs. The pool
backend + key stay in the trusted parent; the child reaches models only through the metered
`call_model` RPC — so it can neither egress nor read the validator's secrets, and fails
closed on any error/timeout.
"""

from __future__ import annotations

from .confine import SandboxError, run_agent_confined

__all__ = ["SandboxError", "run_agent_probe"]


def _build_backend(spec: dict):
    """Build the pool backend PARENT-side (the key never crosses into the confined child)."""
    kind = spec.get("kind", "mock")
    if kind == "mock":
        from thirtyspokes.gateway.gateway import MockBackend
        return MockBackend()
    if kind in ("mockpool", "pinned"):
        from thirtyspokes.koth.pool import MockPool
        return MockPool()
    if kind == "openrouter":  # pragma: no cover — live pool
        import os
        from thirtyspokes.eval import config
        from thirtyspokes.gateway.gateway import OpenRouterBackend
        return OpenRouterBackend(os.environ["OPENROUTER_API_KEY"], price_fn=config.price_for)
    raise ValueError(f"unknown pool kind: {kind}")


def run_agent_probe(source_text: str, weights: bytes, prompts: list[str],
                    *, pool_spec: dict | None = None, timeout: float = 60.0) -> list[str]:
    """Run the miner's agent on `prompts` in a confined child against the SAME pool the miner
    uses (`pool_spec`; default offline MockBackend). Returns answers in prompt order. Raises
    SandboxError → caller fails closed."""
    backend = _build_backend(pool_spec or {"kind": "mock"})
    tasks = [{"task_id": i, "benchmark": "probe", "prompt": p} for i, p in enumerate(prompts)]
    results, _trace, _hardened = run_agent_confined(source_text, weights, tasks,
                                                    backend=backend, timeout=timeout)
    by_id = {r["task_id"]: r["answer"] for r in results}
    if len(by_id) != len(prompts):
        raise SandboxError("probe produced fewer answers than prompts")
    return [by_id[i] for i in range(len(prompts))]
