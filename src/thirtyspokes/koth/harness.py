"""The fixed routing harness — the subnet's engine, not the miner's.

A miner submits ONE thing: the weights of a routing head. Everything that surrounds it — the frozen
encoder, the head architecture, the cascade execution, the verifier, the task sampling — lives here,
inside the owner's measured image, and is bound into RTMR3. A miner cannot change any of it.

WHY THIS SHAPE. When miners shipped arbitrary Python, most of the mechanism existed to police it:
grounding checks, source and weight scans, no-egress confinement, sandboxed re-execution. All of that
was mitigation for a decision — "let miners run code" — and it still could not rule out the central
cheat, since on a static public pool a `prompt -> best model` lookup table is behaviourally identical
to a router. Moving the engine into the harness deletes that whole class rather than mitigating it:

  * a routing head CANNOT emit an answer. It emits a distribution over pool actions; the harness calls
    the chosen model and returns that model's response verbatim. Answer memorisation is impossible by
    construction, so grounding/laundering/source-scan/weight-scan all retire.
  * miner-authored CODE runs nowhere in the system. Weights load through `np.load(allow_pickle=False)`
    into a strict shape check — never pickle, which is code execution wearing a data costume.
  * the PARAM CAP is doing security work, not tidiness. A head this small cannot memorise a large task
    pool; it has to generalise. That is the anti-overfitting bound that replaces the fresh-probe audit.

WHAT A MINER COMPETES ON. Given a task, the head picks WHERE TO ENTER a cheap->expensive ladder. The
harness invokes that rung, runs the pinned verifier on the answer, and escalates while the verifier
rejects. So the miner is learning two things at once: which asks are cheap-solvable, and how far to
trust a cheap answer. `cascade.to_cascade_cache` models exactly this offline over a precomputed
outcome cache, so a miner can train against the same semantics the harness executes without paying
for inference — and the validator can score the decision from the owner's reference matrix without
re-running anything.
"""

from __future__ import annotations

import io

import numpy as np

from ..router import RouterHead
from .benchmarks import Benchmark

# Bumped whenever the harness's OBSERVABLE behaviour changes (action space, verifier, encoder,
# feature construction). Folded into `runtime_measurement()` beside SUITE_VERSION, so a changed
# harness changes RTMR3 and every miner's evidence resets — the same contract a suite rotation has.
HARNESS_VERSION = "koth-harness-1"

# The frozen encoder. Pinned by name because the embedding must be byte-identical in the miner's
# enclave, in the owner's reference build, and in the trainer a miner runs at home; a different
# encoder silently changes every routing decision.
ENCODER = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# A head is d*h + h + h*k + k params. At d=384, k=12, h=16 that is ~6.4K; the cap admits h up to ~128
# while staying far too small to memorise a task pool of thousands. Lowering this tightens the
# anti-memorisation bound; raising it widens the competitive surface. It is a security parameter.
PARAM_CAP = 50_000
DEFAULT_HIDDEN = 16


class ArtifactError(ValueError):
    """The miner's weights blob is unusable. Always a DQ, never a crash: a malformed artifact is a
    miner problem and must not take the runtime down."""


def load_head(weights: bytes, *, k: int, d: int = EMBED_DIM,
              param_cap: int = PARAM_CAP) -> tuple[RouterHead, np.ndarray]:
    """Parse a miner's `weights.npz` into a head + parameter vector.

    Format: an npz holding `theta` (1-D float) and `hidden` (int). `d` and `k` are NOT read from the
    file — they come from the harness's pinned encoder and the owner's pinned pool, so a miner cannot
    redefine the feature space or the action space by editing its artifact.

    `allow_pickle=False` is the load-bearing flag. numpy's pickle path executes arbitrary code at load
    time, which would reintroduce the exact capability this architecture exists to remove.
    """
    try:
        z = np.load(io.BytesIO(weights), allow_pickle=False)
    except Exception as e:  # noqa: BLE001 — any parse failure is the miner's problem
        raise ArtifactError(f"not a readable npz ({type(e).__name__})") from e
    missing = {"theta", "hidden"} - set(z.files)
    if missing:
        raise ArtifactError(f"npz missing {sorted(missing)}; expected theta + hidden")
    try:
        hidden = int(np.asarray(z["hidden"]).reshape(()))
    except Exception as e:  # noqa: BLE001
        raise ArtifactError("hidden must be a scalar int") from e
    if not 1 <= hidden <= 4096:
        raise ArtifactError(f"hidden {hidden} out of range")
    theta = np.asarray(z["theta"], dtype=np.float64).reshape(-1)
    if not np.isfinite(theta).all():
        raise ArtifactError("theta contains NaN or inf")

    head = RouterHead(d, k, hidden)
    if head.n_params > param_cap:
        raise ArtifactError(f"head has {head.n_params} params > cap {param_cap}")
    if theta.size != head.n_params:
        raise ArtifactError(f"theta has {theta.size} params, expected {head.n_params} "
                            f"for d={d} k={k} hidden={hidden}")
    return head, theta


def save_head(theta: np.ndarray, hidden: int) -> bytes:
    """The miner-side counterpart, so the published artifact is produced by the same contract the
    harness parses. Used by the dev kit and the reference router."""
    buf = io.BytesIO()
    np.savez(buf, theta=np.asarray(theta, dtype=np.float32), hidden=np.int32(hidden))
    return buf.getvalue()


def verifier_ok(answer: str, bench: Benchmark) -> bool:
    """The PINNED verifier: does this answer parse under the benchmark's own grader?

    Deliberately deterministic and free. It must return the identical verdict in the miner's enclave
    and in the owner's reference build, or the cascade the miner trained against is not the cascade
    that runs — and a model-based judge is neither reproducible nor cheap enough to sit in the inner
    loop of every escalation.

    It checks WELL-FORMEDNESS, not correctness: a truncated program, an empty answer, a refusal, a
    reasoning model that spent its whole budget thinking. That is a real and common failure of cheap
    models, and it is exactly the signal a cascade needs — "did this rung produce something usable?"
    It cannot catch a fluent wrong answer, which bounds what the ladder can recover and is stated
    plainly in docs/DESIGN.md rather than papered over.
    """
    from .verify import _bench_kind, answer_token
    return answer_token(answer, _bench_kind(bench)) is not None


def rung_order(pool: list[str], price_of) -> list[int]:
    """Pool indices ordered cheap -> expensive. The ladder's rungs, and the action space: action r
    means 'enter at rung r'. Ordering by price (not by name or pool order) is what makes escalation
    monotonically more expensive, which is the whole premise of the cascade."""
    return sorted(range(len(pool)), key=lambda i: (price_of(pool[i]), pool[i]))


def run_cascade(start_rung: int, prompt: str, bench: Benchmark, pool: list[str],
                order: list[int], call_model, params: dict) -> tuple[str, list[int]]:
    """Enter the ladder at `start_rung`, escalate while the verifier rejects, bank on accept.

    Mirrors `cascade.to_cascade_cache` exactly — invoke, verify, escalate, and take whatever the top
    rung produces — so a head trained offline against a precomputed cache behaves identically here.
    Returns the banked answer and the rungs actually invoked (the cost trail, which the proof
    records so a validator can price the run without re-executing it).
    """
    used: list[int] = []
    answer = ""
    for pos in range(max(0, min(start_rung, len(order) - 1)), len(order)):
        idx = order[pos]
        used.append(idx)
        answer = call_model(pool[idx], [{"role": "user", "content": prompt}], dict(params))
        if verifier_ok(answer, bench) or pos == len(order) - 1:
            break
    return answer, used
