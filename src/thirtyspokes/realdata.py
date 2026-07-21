"""Load a REAL pool into an OutcomeCache from RouterBench.

RouterBench (withmartian/routerbench) records, for 36k+ real questions across
~11 real LLMs: a per-(question, model) correctness score, the real dollar cost,
the benchmark domain (`eval_name`), and the prompt text. That is exactly the
OutcomeCache — no API calls needed to run the oracle-gap gate on a real pool.

This is the Phase-0 -> Phase-1 hinge: the decisive "does a real pool have
routable headroom?" experiment, runnable offline from public data.

Fetch once:
    from huggingface_hub import hf_hub_download
    hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl",
                    repo_type="dataset", local_dir="data")
"""

from __future__ import annotations

import numpy as np

from .cache import OutcomeCache, PoolModel

# A curated pool spanning cheap -> strong, matching the architecture's "3-5 models
# across the cost spectrum" (the full 11-model pool is also selectable). Order is
# cheap -> expensive by measured mean cost.
CURATED_POOL = [
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
]

ALL_MODELS = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat",
]


def load_routerbench(path: str):
    import pandas as pd  # lazy: only needed for real-data path
    return pd.read_pickle(path)


def domain_onehot_embeddings(df, domains: np.ndarray, n_domains: int) -> np.ndarray:
    """Fallback 'domain-only encoder': one-hot(eval_name), lightly jittered.

    A router on these embeddings can identify the benchmark domain but nothing
    finer — a deliberately conservative encoder. Real sentence embeddings
    (see sentence_embeddings) reveal more and raise the achievable ceiling.
    """
    x = np.zeros((len(df), n_domains), dtype=np.float32)
    x[np.arange(len(df)), domains] = 1.0
    return x


def sentence_embeddings(df, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Real semantic embeddings of the prompts (optional; needs sentence-transformers)."""
    from sentence_transformers import SentenceTransformer  # optional heavy dep
    enc = SentenceTransformer(model_name)
    return np.asarray(enc.encode(list(df["prompt"]), show_progress_bar=True,
                                 batch_size=256), dtype=np.float32)


def to_cache(
    df,
    models: list[str],
    embeddings: np.ndarray,
    correct_threshold: float = 0.5,
) -> OutcomeCache:
    """Build an OutcomeCache from RouterBench rows + a provided embedding matrix.

    correctness = (score >= threshold); RouterBench scores are mostly {0,1} but a
    few benchmarks give partial credit, so we threshold. cost = real dollar cost.
    domain = factorized eval_name (used as the routable-ceiling grouping signal).
    """
    Q = len(df)
    correct = np.zeros((Q, len(models)), dtype=bool)
    cost = np.zeros((Q, len(models)), dtype=np.float32)
    for k, m in enumerate(models):
        correct[:, k] = df[m].astype(float).to_numpy() >= correct_threshold
        cost[:, k] = df[f"{m}|total_cost"].astype(float).to_numpy()

    domains, _ = _factorize(df["eval_name"].to_numpy())
    mean_cost = cost.mean(axis=0)
    order = np.argsort(mean_cost)  # label roles by cost rank
    role = {}
    for rank, k in enumerate(order):
        role[k] = "cheap" if rank < len(models) / 3 else ("strong" if rank >= 2 * len(models) / 3 else "mid")
    pool = [PoolModel(name=m, price_per_token=float(mean_cost[k]), role=role[k])
            for k, m in enumerate(models)]

    return OutcomeCache(
        models=pool,
        embeddings=embeddings,
        correct=correct,
        cost=cost,
        verifier_ok=None,
        domain=domains,
        question_ids=[str(s) for s in df["sample_id"].to_numpy()],
        cost_norm=float(cost.max()),  # pin the lambda normalizer for this pool/season
    )


def _factorize(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    uniq = sorted(set(values.tolist()))
    index = {v: i for i, v in enumerate(uniq)}
    return np.array([index[v] for v in values], dtype=np.int64), uniq
