"""The Conductor inference seam — a rendered state in, one raw turn out (§3).

    act(prompt) -> raw text

That is the entire contract, and its narrowness is the point. The Conductor is the miner's artifact
and the only miner-supplied thing in the loop; everything it produces leaves through this one return
value and is immediately handed to `action.parse`, which admits a kind and a catalog slug and
nothing else (§2.1). A seam that returned a structured object — a message list, a tool call, a
dict of fields — would be a seam with somewhere for a solution to ride along.

WHAT A LIVE IMPLEMENTATION MUST PIN, all of it already in `config.py`: greedy decoding
(`TEMPERATURE = 0`), `max_tokens = MAX_CONDUCTOR_TOKENS`, batch size 1, and fixed kernels (§8b.4 —
MoE expert routing can vary with batch composition, so "the same model" can otherwise produce
different outputs across runs). The serving stack itself is not built here: M1 runs entirely
offline, and §5.2a already removes the case where bit-exactness would bite hardest by measuring the
king once per slice rather than once per opponent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Conductor(Protocol):
    """One greedy turn against the miner's full weights. `MockConductor` offline."""

    def act(self, prompt: str) -> str:
        """The rendered state -> the raw generation, verbatim and unparsed."""


@dataclass
class MockConductor:
    """A scripted Conductor: the outputs a test wants, in order, malformed ones included.

    THE LAST OUTPUT REPEATS once the script runs out. An episode's length is decided by the loop —
    `MAX_STEPS`, three consecutive parse failures, budget exhaustion, the wall clock — and a mock
    that raised when exhausted would turn every one of those termination tests into a test of the
    script's length instead. Scripting one line and letting it repeat is how "this Conductor never
    stops" is written.

    `prompts` records what it was shown, so a test can assert the scaffold fed it the pinned render
    (and that `StepRecord.rendered_state_digest` is a digest of that exact prompt).
    """

    outputs: tuple[str, ...]
    prompts: list[str] = field(default_factory=list)

    def act(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.outputs[min(len(self.prompts), len(self.outputs)) - 1]
