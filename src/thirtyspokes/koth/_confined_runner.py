"""Child entrypoint for the no-egress confined agent run (koth/confine.py).

Runs inside `unshare --net` (no external network) with a scrubbed env and no pool key. The
agent's only channel is `call_model`, which RPCs to the trusted parent over stdio. The real
fd 1 is reserved for the framed-JSON protocol; the agent's own stdout is redirected to stderr
so a stray `print()` can't corrupt the channel.
"""

from __future__ import annotations

import base64
import json
import os
import sys

# --- reserve a clean protocol channel, then send the agent's stdout to stderr -------------
_CHAN = os.fdopen(os.dup(1), "w")     # writes here reach the parent's proc.stdout
os.dup2(2, 1)                          # agent print() -> stderr (drained), not the channel


def _hide_secrets() -> None:
    """In the private user+mount namespace we hold CAP_SYS_ADMIN over our own mounts, so we
    overlay tmpfs on dirs that may hold host secrets — the agent then physically cannot read
    the validator/miner wallet, SSH keys, or HF token cache. Best-effort."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except Exception:
        return
    for d in ("/root/.bittensor", "/root/.ssh", "/root/.config",
              os.path.expanduser("~/.bittensor"), os.path.expanduser("~/.ssh"),
              os.path.expanduser("~/.cache/huggingface"), "/home"):
        if os.path.isdir(d):
            libc.mount(b"tmpfs", d.encode(), b"tmpfs", 0, b"")


def _send(obj: dict) -> None:
    _CHAN.write(json.dumps(obj) + "\n")
    _CHAN.flush()


def _recv() -> dict | None:
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def main() -> None:
    if os.environ.get("KOTH_CONFINED") == "1":
        _hide_secrets()
    init = _recv()
    if not init or init.get("type") != "init":
        raise SystemExit("no init message")

    from thirtyspokes.koth.runtime import load_agent
    agent = load_agent(init["source"], base64.b64decode(init["weights"]))

    for t in init["tasks"]:
        tid = t["task_id"]

        def call_model(model, messages, params=None, _tid=tid):
            _send({"type": "call", "task_id": _tid, "model": model,
                   "messages": messages, "params": params})
            r = _recv()
            if not r or r.get("type") != "resp":
                raise RuntimeError("no response from parent")
            return r["response"]

        answer = agent(t["prompt"], call_model)
        _send({"type": "result",
               "result": {"task_id": tid, "benchmark": t.get("benchmark"), "answer": str(answer)}})
    _send({"type": "done"})


if __name__ == "__main__":
    main()
