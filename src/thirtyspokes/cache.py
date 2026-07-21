"""The OutcomeCache — the central data structure of the whole design.

Every downstream component (oracle gap, scorer, CMA-ES training, reign
simulation) reads from an OutcomeCache and never calls an LLM. Filling the
cache is the *only* place model inference happens: validators do it once per
epoch (O(K x Q)), and the miner enablement kit ships a pre-filled cache so
miners train without paying per-query API costs.

This is exactly what makes validation flat in the number of miners: scoring any
router is a table lookup over this structure.

A cache can be produced two ways, both yielding the same object:
  - synth.py         -> a synthetic world (offline, no keys, tunable difficulty)
  - backends/*       -> real LLM outcomes (opt-in, needs API access)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PoolModel:
    """One model in the fixed pool, with its published per-token price."""

    name: str
    price_per_token: float  # published price; used verbatim in cost scoring
    role: str = ""          # optional label, e.g. "cheap" / "mid" / "strong"


@dataclass
class OutcomeCache:
    """Precomputed per-(question, model) outcomes over a fixed pool.

    Shapes: Q questions, K models, D embedding dims.
      embeddings   (Q, D) float32  — frozen-encoder vectors, shared by all routers
      correct      (Q, K) bool     — did model k answer question q correctly
      cost         (Q, K) float32  — generated_tokens * price_per_token
      verifier_ok  (Q, K) bool     — does the pinned verifier accept model k's
                                     answer (used by the v1.5 cascade action space;
                                     None until the verifier is filled)
      domain       (Q,)   int      — latent domain id (synthetic bookkeeping only)
    """

    models: list[PoolModel]
    embeddings: np.ndarray
    correct: np.ndarray
    cost: np.ndarray
    verifier_ok: np.ndarray | None = None
    domain: np.ndarray | None = None
    question_ids: list[str] = field(default_factory=list)
    # Season-pinned cost normalizer. When set, cost_max returns this instead of
    # the subset max, so lambda means the same thing across train/eval splits,
    # per-epoch batches, and derived (cascade) caches. Derived/subset caches
    # inherit it from their parent; only a season boundary changes it.
    cost_norm: float | None = None

    def __post_init__(self) -> None:
        q, k = self.correct.shape
        assert self.cost.shape == (q, k), "cost/correct shape mismatch"
        assert self.embeddings.shape[0] == q, "embeddings/questions mismatch"
        assert len(self.models) == k, "models list must match K"
        if self.verifier_ok is not None:
            assert self.verifier_ok.shape == (q, k), "verifier_ok shape mismatch"
        if not self.question_ids:
            self.question_ids = [f"q{i}" for i in range(q)]

    # --- dimensions ---------------------------------------------------------
    @property
    def Q(self) -> int:
        return self.correct.shape[0]

    @property
    def K(self) -> int:
        return self.correct.shape[1]

    @property
    def D(self) -> int:
        return self.embeddings.shape[1]

    # --- cost normalization -------------------------------------------------
    @property
    def cost_max(self) -> float:
        """The lambda normalizer: pinned season constant if set, else subset max.

        Review finding (M2/L2): normalizing each subset/derived cache by its own
        max silently changes lambda's exchange rate between train/eval/epoch
        batches and flatters multi-call (cascade) caches. Subset and derived
        caches therefore inherit the parent's normalizer via `cost_norm`; a
        cascade trajectory may legitimately normalize above 1.0 (it costs more
        than the priciest single call).
        """
        if self.cost_norm is not None:
            return self.cost_norm
        m = float(self.cost.max())
        return m if m > 0 else 1.0

    def normalized_cost(self) -> np.ndarray:
        return (self.cost / self.cost_max).astype(np.float32)

    # --- per-model summary stats (used by oracle gap & reporting) -----------
    def model_accuracy(self) -> np.ndarray:
        """Mean correctness of each model over all questions -> (K,)."""
        return self.correct.mean(axis=0)

    def model_mean_cost(self) -> np.ndarray:
        return self.cost.mean(axis=0)

    # --- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            correct=self.correct,
            cost=self.cost,
            verifier_ok=(self.verifier_ok if self.verifier_ok is not None
                         else np.zeros((0,), dtype=bool)),
            has_verifier=np.array([self.verifier_ok is not None]),
            domain=(self.domain if self.domain is not None
                    else np.zeros((0,), dtype=np.int64)),
            model_names=np.array([m.name for m in self.models], dtype=object),
            model_prices=np.array([m.price_per_token for m in self.models]),
            model_roles=np.array([m.role for m in self.models], dtype=object),
            question_ids=np.array(self.question_ids, dtype=object),
            cost_norm=np.array([self.cost_norm if self.cost_norm is not None else -1.0]),
        )

    @classmethod
    def load(cls, path: str) -> "OutcomeCache":
        d = np.load(path, allow_pickle=True)
        models = [
            PoolModel(str(n), float(p), str(r))
            for n, p, r in zip(d["model_names"], d["model_prices"], d["model_roles"])
        ]
        verifier = d["verifier_ok"] if bool(d["has_verifier"][0]) else None
        domain = d["domain"] if d["domain"].size else None
        saved_norm = float(d["cost_norm"][0]) if "cost_norm" in d else -1.0
        return cls(
            models=models,
            embeddings=d["embeddings"],
            correct=d["correct"],
            cost=d["cost"],
            verifier_ok=verifier,
            domain=domain,
            question_ids=list(d["question_ids"]),
            cost_norm=saved_norm if saved_norm > 0 else None,
        )

    # --- train / eval split -------------------------------------------------
    def split(self, eval_frac: float, seed: int = 0) -> tuple["OutcomeCache", "OutcomeCache"]:
        """Deterministic split into (train, eval) caches sharing the same pool.

        Mirrors the real subnet: miners train on the enablement corpus, validators
        score on freshly sampled held-out questions.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.Q)
        n_eval = int(round(self.Q * eval_frac))
        eval_idx, train_idx = perm[:n_eval], perm[n_eval:]

        def take(idx: np.ndarray) -> "OutcomeCache":
            return OutcomeCache(
                models=self.models,
                embeddings=self.embeddings[idx],
                correct=self.correct[idx],
                cost=self.cost[idx],
                verifier_ok=None if self.verifier_ok is None else self.verifier_ok[idx],
                domain=None if self.domain is None else self.domain[idx],
                question_ids=[self.question_ids[i] for i in idx],
                cost_norm=self.cost_max,  # children share the parent's lambda scale
            )

        return take(train_idx), take(eval_idx)

    def subset(self, idx: np.ndarray) -> "OutcomeCache":
        """Arbitrary row subset, inheriting this cache's lambda normalizer."""
        return OutcomeCache(
            models=self.models,
            embeddings=self.embeddings[idx],
            correct=self.correct[idx],
            cost=self.cost[idx],
            verifier_ok=None if self.verifier_ok is None else self.verifier_ok[idx],
            domain=None if self.domain is None else self.domain[idx],
            question_ids=[self.question_ids[i] for i in idx],
            cost_norm=self.cost_max,
        )
