"""The worker-call seam — where the task text leaves the validator (§2.1, §8b.3).

A *worker* is whichever real model the Conductor chose. The contract is three arguments wide on
purpose:

    complete(model_id, task_text, params) -> (text, cost_usd)

`model_id` is validated against the window's catalog snapshot, `task_text` is the composed worker
text, and `params` is an owner-pinned constant. **There is no fourth argument, and that is the
design.** §2.1 says the only things that reach a worker model are the pinned task text, a validated
model ID, and bytes the benchmark's own environment returned in answer to that same worker's own
read-only requests — so the seam is shaped so that a Conductor-produced string has nowhere to sit: a
prompt prefix, a system message or a `metadata` field would each be a channel through which a miner
could smuggle a memorised solution into the answer path, and no detector would be needed to find it
because it would be a documented parameter.

THE COMPOSITION HAPPENS ABOVE THIS SEAM AND THAT IS WHY THE SEAM DID NOT HAVE TO WIDEN. `task_text`
is `tools.compose(task.prompt, observations)`, a pinned pure function of the benchmark's own
statement and this worker's own reads, with `compose(p, ()) == p` byte for byte — so on every
benchmark that declares no tools, and on the first turn of every delegate, `task_text` IS the
benchmark's statement exactly as it always was. A fourth argument carrying the observations would
have bought nothing and would have cost the sentence above.

FAILURE IS AN OUTCOME, NOT AN ERROR (§8b.3). A worker erroring, refusing or timing out is
information about that rung — a Conductor that keeps delegating to a flaky model should be scored
for it. So the seam signals failure by raising, and the scaffold turns that into a recorded failed
step rather than a crash. (Grader failures are the opposite case and are deliberately NOT caught
there; see `scaffold._call_worker`.)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

# Pinned, identical for every worker call in every arm, and — the load-bearing part — CONSTANT:
# nothing derived from Conductor output is ever merged into it. Temperature 0 for the same reason
# the Conductor decodes greedily (§1): sampling would put run-to-run noise inside a paired
# comparison that exists to remove it, so a duel would partly compare luck rather than routing.
#
# A read-only mapping rather than a plain dict, because this is the one parameter of a worker call
# that is neither the task text nor the model ID: if anything could add a key to it, §2.1 would
# have a channel and this module's whole claim would be a convention instead of a property.
WORKER_PARAMS: Mapping[str, object] = MappingProxyType({"temperature": 0.0})


class WorkerError(Exception):
    """A provider error, refusal or timeout. Data about that rung (§8b.3), never a crash.

    `status` is the HTTP status of a permanent refusal, or None for a transport failure. It exists
    for ONE reader: `OwnerGateway.call` turns a 401/402 — the PAYER's key refused, not the model —
    into a meter refusal that is not an outcome (§5.2c), so an empty account is never memoised as
    a model failing.
    """

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class NotAnOutcome(Exception):
    """A worker-seam failure that says NOTHING about the rung: the meter refused the call — an
    exhausted allowance, a closed duel token — never the model. `Scaffold._call_worker` still
    records it as a dead step (§8b.3 keeps the arm alive), but it must not be memoised: the
    window's outcome table (§5.2c) holds draws of what a MODEL did, and a payer's empty wallet
    stored under a `(task, model, attempt)` key would replay as that model failing for every later
    arm. `gateway.GatewayError` is one.
    """


class Worker(Protocol):
    """One model call. Implementations: `MockWorker` here, `openrouter.OpenRouterClient` live.

    THE LIVE IMPLEMENTATION IS NOT IN THIS MODULE, AND THAT IS DELIBERATE. This file held an
    `OpenRouterWorker` until the seams were reconciled; it was written in parallel with
    `openrouter.py` and diverged from it on the two rules that decide money:

      * it priced a response carrying no `usage.cost` at **zero** (`float(... or 0.0)`), which
        `final_b` reads as an arm that answered for free — the most efficient arm there can be.
        `openrouter.read_completion` refuses such a response instead (§5.1b), because a spend
        figure the validator invented is the owner's arithmetic inside the miner's score.
      * it pinned no `provider` block, so the endpoint was chosen by OpenRouter's load balancing,
        i.e. by the clock — king and challenger minutes apart could face different machines (§11-1).

    Neither was reachable (the class had no caller and no test), but two provider clients in one
    package is how the wrong one gets wired up. The pins live in one place; this module keeps the
    seam and the mock, which is what the offline suite and §2.1's invariant test need.
    """

    def complete(self, model_id: str, task_text: str,
                 params: Mapping[str, object]) -> tuple[str, float]:
        """Send `task_text` — verbatim, alone — to `model_id`. Returns (response, dollars spent)."""

    # OPTIONAL, AND NOT PART OF `complete`'s THREE ARGUMENTS: a seam that meters money may also
    # offer `charge(model_id, task_text, cost_usd, *, tokens_in=0, tokens_out=0, response_text=None)`,
    # which the scaffold calls INSTEAD of `complete` when a delegate is replayed from the window's
    # outcome table (§5.2c) — the same debit as the call it replaces, no provider. A seam without it
    # replays for free, which is what an offline world does anyway. `Scaffold` discovers it with
    # `getattr`, so this protocol stays exactly as wide as §2.1 allows.


@dataclass
class MockWorker:
    """A deterministic, scriptable pool for offline tests.

    `answers` and `costs` are keyed by model ID, so a test writes a pool the way the mechanism
    describes one: this model answers well, that one is cheap, that one is down. A model with no
    scripted answer returns the empty string — an unhelpful response, which is what an unscripted
    rung is.

    `seen` is the independent witness for the §2.1 test. The scaffold keeps its own audit log, but a
    log the scaffold writes cannot prove what the scaffold sent; this records what actually arrived
    at the far end of the seam, and the invariant test asserts on both.
    """

    answers: Mapping[str, str] = field(default_factory=dict)
    costs: Mapping[str, float] = field(default_factory=dict)
    failures: frozenset[str] = frozenset()
    seen: list[tuple[str, str, Mapping[str, object]]] = field(default_factory=list)
    charged: float = 0.0

    def complete(self, model_id: str, task_text: str,
                 params: Mapping[str, object]) -> tuple[str, float]:
        self.seen.append((model_id, task_text, params))
        if model_id in self.failures:
            raise WorkerError(f"{model_id} is down")
        cost = float(self.costs.get(model_id, 0.0))
        self.charged += cost
        return self.answers.get(model_id, ""), cost

    def charge(self, model_id: str, task_text: str, cost_usd: float, *, tokens_in: int = 0,
               tokens_out: int = 0, response_text: str | None = None) -> None:
        """A replayed delegate's debit (§5.2c): no call, the same money — `charged` counts it, so
        the accounting tests see hit and miss priced alike."""
        self.charged += float(cost_usd)
