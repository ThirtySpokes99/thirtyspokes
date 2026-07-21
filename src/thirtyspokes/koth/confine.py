"""No-egress confinement for untrusted agent code (H3).

The miner's agent (and the validator's fresh-probe of it) is arbitrary Python. Running it
in-process lets it `import socket` and exfiltrate the nonce / task-slice to a confederate, or
read on-disk secrets (wallet, HF token). This runs it in a **network-isolated child** whose
ONLY channel is `call_model`, proxied over stdio to the trusted parent:

    parent (trusted)                     child (untrusted, unshare --net)
    ── builds the pool backend           ── load_agent(source, weights)
    ── meters every call (MeteringProxy)  <── {"call", model, messages}     agent(prompt, call_model)
    ── makes the real OpenRouter call     ──> {"resp", response}
    ── records the authoritative trace     ── {"result", answer} per task

The child has NO network (private netns), NO pool key (it never leaves the parent), scrubbed
env, resource limits, and — in the hardened path — tmpfs over the host secret dirs. Egress and
metering are therefore un-bypassable by the agent. `confine=False`/unavailable falls back to a
plain isolated subprocess (still scrubbed env + rlimits) so offline CI works everywhere; the
no-egress *guarantee* holds only where `confinement_available()`.

Residual (documented, not closable here): prompt-based exfiltration — the agent can embed the
nonce/task in a *legitimate* prompt to the pinned model, reaching a confederate through the
sanctioned channel (docs/DESIGN.md §8).
"""

from __future__ import annotations

import base64
import functools
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading


class SandboxError(Exception):
    pass


def _kill(proc: subprocess.Popen) -> None:
    """SIGKILL the whole child process group — `unshare --fork` reparents the python
    grandchild, so killing only `proc` would orphan it (and hold the pipe open)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


_MEM_LIMIT = 2 * 1024 ** 3      # RLIMIT_AS per process
_NPROC_LIMIT = 1024             # fork-bomb backstop (+ PID namespace + watchdog)
_NOFILE_LIMIT = 256


def _ns_flags() -> list[str]:
    """The namespaces the confined child needs. Real root (the measured-image runtime) creates
    net/mount/pid DIRECTLY — it holds CAP_SYS_ADMIN and needs no user namespace, which is what lets
    it bootstrap on the locked image where Ubuntu 24.04 restricts unprivileged userns. Non-root must
    first enter an unprivileged user namespace to gain those caps."""
    ns = ["--net", "--mount", "--pid", "--fork"]
    if os.geteuid() != 0:
        ns = ["--user", "--map-root-user", *ns]
    return ns


@functools.lru_cache(maxsize=1)
def confinement_available() -> bool:
    """True where the no-egress confinement can ACTUALLY be built — probed, not guessed.

    We spawn the real `unshare` with the very flags `_argv` will use and check it succeeds. A
    knob-based guess is NOT sufficient: Ubuntu 24.04 (GitHub runners, many distros) blocks
    unprivileged user namespaces via AppArmor and does not expose the old
    `/proc/sys/kernel/unprivileged_userns_clone`, so the guess returned True and the sandbox then
    died at spawn with `unshare: write failed /proc/self/uid_map: Operation not permitted` — turning
    every audit into `audit_error:SandboxError`.

    Where this returns False, `run_agent_confined` degrades to a plain isolated subprocess (scrubbed
    env + rlimits); the no-egress *guarantee* holds only where it returns True.
    """
    if sys.platform != "linux" or not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", *_ns_flags(), "--", "true"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _preexec(cpu_limit: int):
    for res, lim in ((resource.RLIMIT_AS, _MEM_LIMIT), (resource.RLIMIT_CPU, cpu_limit),
                     (resource.RLIMIT_NPROC, _NPROC_LIMIT), (resource.RLIMIT_NOFILE, _NOFILE_LIMIT)):
        try:
            resource.setrlimit(res, (lim, lim))
        except (ValueError, OSError):
            pass


def _scrubbed_env(hardened: bool) -> dict:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
           "HOME": "/nonexistent"}
    if hardened:
        env["KOTH_CONFINED"] = "1"      # tells the runner to tmpfs-hide secret dirs
    return env


def _argv(hardened: bool) -> list[str]:
    """Confined child argv. The namespaces come from `_ns_flags()` — the SAME flags
    `confinement_available()` probes, so detection can never drift from execution."""
    runner = [sys.executable, "-m", "thirtyspokes.koth._confined_runner"]
    if not hardened:
        return runner
    # net + mount + pid: the child gets only a private loopback (no egress) and its own mount table
    # (so it can tmpfs-hide host secrets).
    return ["unshare", *_ns_flags(), "--", *runner]


def _send(f, obj: dict) -> None:
    f.write((json.dumps(obj) + "\n").encode())
    f.flush()


def run_agent_confined(source_text: str, weights: bytes, tasks: list[dict], *, backend,
                       timeout: float = 120.0, hardened: bool | None = None):
    """Run the agent over `tasks` (`[{task_id, benchmark?, prompt}]`) in a confined child.
    Model calls are metered + executed parent-side; returns `(results, trace)`:
      results = [{task_id, benchmark, answer, cost_usd}]   (cost attributed from the trace)
      trace   = [{task_id, model, prompt, response, cost_usd}]
    Raises SandboxError on crash/timeout → callers fail closed."""
    from ..tee.runtime import MeteringProxy

    if hardened is None:
        hardened = confinement_available()
    proxy = MeteringProxy(backend)
    trace: list[dict] = []
    results: list[dict] = []

    proc = subprocess.Popen(
        _argv(hardened), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_scrubbed_env(hardened), start_new_session=True,     # own pgroup → killable as a unit
        preexec_fn=lambda: _preexec(int(timeout) + 5))
    stderr_buf: dict = {}
    drain = threading.Thread(target=lambda: stderr_buf.setdefault("e", proc.stderr.read()), daemon=True)
    drain.start()
    watchdog = threading.Timer(timeout, _kill, args=(proc,))
    watchdog.start()
    try:
        _send(proc.stdin, {"type": "init", "source": source_text,
                           "weights": base64.b64encode(weights).decode(), "tasks": tasks})
        for raw in proc.stdout:                        # framed JSON, one message per line
            msg = json.loads(raw)
            if msg["type"] == "call":
                c0 = proxy.total_cost_usd
                resp = proxy.call_model(msg["model"], msg["messages"], msg.get("params"))
                trace.append({"task_id": msg.get("task_id"), "model": msg["model"],
                              "prompt": str(msg["messages"][-1]["content"]),
                              "response": str(resp), "cost_usd": proxy.total_cost_usd - c0})
                _send(proc.stdin, {"type": "resp", "response": resp})
            elif msg["type"] == "result":
                results.append(msg["result"])
            elif msg["type"] == "done":
                break
    except (BrokenPipeError, ValueError, KeyError) as e:
        _kill(proc)
        # 1s was too short under real cold-boot I/O contention: the drain thread's blocking read
        # hadn't reached EOF yet, so `err` came back empty and a real crash reason (e.g. unshare
        # denied, OOM) was lost — every occurrence looked like a bare "Broken pipe" with nothing to
        # act on. Seen live on testnet 526 (2026-07-14): ~30% of confined boots, no diagnosable cause.
        drain.join(timeout=5)
        err = (stderr_buf.get("e") or b"").decode(errors="replace").strip()
        raise SandboxError(f"protocol error: {e}" + (f"; child stderr: {err[-400:]}" if err else "")) from e
    finally:
        watchdog.cancel()
        try:
            proc.stdin.close()
        except OSError:
            pass
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill(proc)
        raise SandboxError("timeout")
    drain.join(timeout=5)
    err = (stderr_buf.get("e") or b"").decode(errors="replace").strip()
    if rc != 0:
        raise SandboxError(f"exit={rc}: {err[-300:]}")

    # per-task cost is attributed from the PARENT's authoritative trace — the child can't lie.
    by_task: dict = {}
    for e in trace:
        by_task[e["task_id"]] = by_task.get(e["task_id"], 0.0) + e["cost_usd"]
    for r in results:
        r["cost_usd"] = by_task.get(r["task_id"], 0.0)
    return results, trace
