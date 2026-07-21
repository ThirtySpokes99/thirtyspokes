"""Real-LLM outcome backend seam.

Phase 0 runs entirely on synthetic caches. This module is the single, isolated
place where real model inference would enter: fill an OutcomeCache from actual
pool models + a real encoder + a real verifier. Everything downstream is
identical whether the cache came from synth.py or from here — which is the whole
point of centering the design on the cache.

`build_cache` is backend-agnostic; a concrete backend implements PoolBackend.
The OpenAI-compatible adapter is a sketch (not exercised in Phase 0) and only
imports its optional deps when instantiated, so importing this module is free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .cache import OutcomeCache, PoolModel


@runtime_checkable
class PoolBackend(Protocol):
    def embed(self, questions: list[str]) -> np.ndarray:
        """(N,) questions -> (N, D) frozen-encoder embeddings."""

    def answer(self, question: str, model_name: str) -> tuple[str, int]:
        """Return (answer_text, generated_token_count) for one model."""

    def is_correct(self, question: str, answer: str, reference: str) -> bool:
        """Verifiable-answer grading (exact match / checker / unit test)."""

    def verify(self, question: str, answer: str) -> bool:
        """Pinned-verifier verdict used by the v1.5 cascade (accept/reject)."""


def build_cache(
    questions: list[str],
    references: list[str],
    models: list[PoolModel],
    backend: PoolBackend,
    with_verifier: bool = True,
) -> OutcomeCache:
    """Drive a backend to produce the O(K x Q) outcome cache once.

    This is exactly the per-epoch validator step and the enablement-kit build
    step; miners never run it (they get the resulting cache).
    """
    Q, K = len(questions), len(models)
    embeddings = np.asarray(backend.embed(questions), dtype=np.float32)
    correct = np.zeros((Q, K), dtype=bool)
    cost = np.zeros((Q, K), dtype=np.float32)
    verifier_ok = np.zeros((Q, K), dtype=bool) if with_verifier else None

    for qi, (q, ref) in enumerate(zip(questions, references)):
        for ki, m in enumerate(models):
            ans, ntok = backend.answer(q, m.name)
            correct[qi, ki] = backend.is_correct(q, ans, ref)
            cost[qi, ki] = ntok * m.price_per_token
            if with_verifier:
                verifier_ok[qi, ki] = backend.verify(q, ans)

    return OutcomeCache(
        models=models, embeddings=embeddings, correct=correct, cost=cost,
        verifier_ok=verifier_ok, domain=None,
        question_ids=[f"q{i}" for i in range(Q)],
    )


class OpenAICompatibleBackend:
    """Sketch adapter for OpenAI-compatible pool endpoints + an embedding model.

    Not exercised in Phase 0. Wire real grading (`is_correct`) to your verifiable
    answers, and `verify` to the pinned verifier model. Deps (`openai`) import
    lazily so the rest of the package stays dependency-free.
    """

    def __init__(self, base_url: str, api_key: str, embed_model: str, verifier_model: str):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'thirtyspokes[llm]' to use the real-LLM backend") from e
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._embed_model = embed_model
        self._verifier_model = verifier_model

    def embed(self, questions: list[str]) -> np.ndarray:  # pragma: no cover
        resp = self._client.embeddings.create(model=self._embed_model, input=questions)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def answer(self, question: str, model_name: str) -> tuple[str, int]:  # pragma: no cover
        r = self._client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": question}], temperature=0.0)
        return r.choices[0].message.content or "", r.usage.completion_tokens

    def is_correct(self, question: str, answer: str, reference: str) -> bool:  # pragma: no cover
        # Replace with your verifiable-answer checker (exact match / unit tests / math).
        return answer.strip() == reference.strip()

    def verify(self, question: str, answer: str) -> bool:  # pragma: no cover
        r = self._client.chat.completions.create(
            model=self._verifier_model, temperature=0.0,
            messages=[{"role": "user",
                       "content": f"Question: {question}\nAnswer: {answer}\n"
                                  f"Is the answer correct? Reply only YES or NO."}])
        return (r.choices[0].message.content or "").strip().upper().startswith("YES")
