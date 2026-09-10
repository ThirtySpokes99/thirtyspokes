"""v3 — subnet v3 (docs/WHITEPAPER.md, plan in the build plan).

A miner ships full model weights on a pinned architecture. That model is a *Conductor*: given an
agentic task it reasons internally, delegates the task to a real model, observes the outcome, and
either retries elsewhere or stops. It never writes the answer and never touches the worker's input.

The three words are three properties the mechanism must hold at once, and they show up here as code
rather than as prose:

  * **Kind**   — a miner's one shot is never spent by our ambiguity. Nothing is coerced: an action
                 that is not exactly one valid action fails loudly and is counted (`action.py`).
  * **Knight** — the crown is defended, not held; only a fresh re-evaluation defends it.
  * **Spirit** — what is rewarded is the part that generalises, enforced by leave-one-out breadth.

M0 (this milestone) pins the two things everything downstream is shaped by — the constants
(`config.py`) and the contract every module codes against (`types.py`) — plus the action grammar
(`action.py`) that carries the §2.1 invariant: *the only things that reach a worker model are the
pinned task text and a validated model ID.*

Built ALONGSIDE `koth/`, which is untouched until the cutover (the build plan M10): nothing is deleted
until its replacement runs end to end.
"""
