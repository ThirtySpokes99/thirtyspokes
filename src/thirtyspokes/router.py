"""The router head — a tiny MLP on frozen embeddings, packed as a flat vector.

Contract (architecture 5): input = one frozen-encoder embedding, output = a
probability distribution over the K pool models. Soft output is mandatory — it
feeds the fingerprint deduplication (two heads with near-identical distributions
over a probe set are copies).

The head is deliberately small (<= a param cap) and gradient-free-friendly:
weights live in one flat float vector so separable CMA-ES can optimize them
directly, exactly as TinyRouter / Sakana-Fugu-base do.
"""

from __future__ import annotations

import numpy as np

PARAM_CAP = 1_000_000  # architecture 5: heads must stay under this


class RouterHead:
    """embedding (D) -> tanh hidden (H) -> logits (K) -> softmax.

    A single flat parameter vector packs [W1, b1, W2, b2].
    """

    def __init__(self, d: int, k: int, hidden: int = 16):
        self.d = d
        self.k = k
        self.h = hidden
        self._sizes = [d * hidden, hidden, hidden * k, k]
        self._shapes = [(d, hidden), (hidden,), (hidden, k), (k,)]
        self.n_params = sum(self._sizes)
        if self.n_params > PARAM_CAP:
            raise ValueError(
                f"router head has {self.n_params} params > cap {PARAM_CAP}; "
                f"reduce hidden ({hidden}) or embed dim ({d})"
            )

    # --- flat-vector packing (the CMA-ES interface) ------------------------
    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        assert theta.shape == (self.n_params,), f"expected {self.n_params} params"
        out, i = [], 0
        for size, shape in zip(self._sizes, self._shapes):
            out.append(theta[i:i + size].reshape(shape))
            i += size
        return tuple(out)

    def init_theta(self, seed: int = 0, scale: float = 0.1) -> np.ndarray:
        """Small random init; CMA-ES searches outward from here."""
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, size=self.n_params).astype(np.float64)

    # --- forward ------------------------------------------------------------
    def logits(self, theta: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """(Q, D) embeddings -> (Q, K) raw logits (pre-softmax).

        Exposed for the multi-turn orchestrator, which masks invalid actions
        (e.g. Verifier before any answer exists) before argmax.
        """
        w1, b1, w2, b2 = self.unpack(theta)
        return np.tanh(embeddings @ w1 + b1) @ w2 + b2

    def distribution(self, theta: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """(Q, D) embeddings -> (Q, K) softmax distributions over the pool."""
        logits = self.logits(theta, embeddings)
        logits = logits - logits.max(axis=1, keepdims=True)  # stability
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def choose(self, theta: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Single-shot decision: argmax model index per question -> (Q,)."""
        return self.distribution(theta, embeddings).argmax(axis=1)
