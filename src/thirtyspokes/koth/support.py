"""Bitext support-intent benchmark and the fitted two-model KOTH artifact.

This is an experimental product/economics suite.  It intentionally does not
replace ``real_suite`` or change the owner-pinned production suite version.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..cache import OutcomeCache
from ..public_support import extract_intent, stratified_indices
from ..support_pilot import fit_linear_router, load_scored_tickets
from .benchmarks import BenchTask, RealBenchmark
from .runtime import Artifact


SUPPORT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SUPPORT_MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5.4",
)
SUPPORT_PRICE_PER_1K = {
    "google/gemini-2.5-flash-lite": (0.00010, 0.00040),
    "openai/gpt-5.4": (0.00250, 0.01500),
}
SUPPORT_PROMPT_VERSIONS = ("v1", "v2")
SUPPORT_INTENTS = (
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
)

_SUPPORT_BASE_SYSTEM_PREFIX = (
    "Classify the customer-support ticket. Return exactly one allowed intent ID "
    "and nothing else. Allowed intent IDs: "
)
SUPPORT_GEMINI_GUIDANCE_V2 = (
    " Disambiguate: check_invoice=find/view existing vs get_invoice=request copy; "
    "delivery_period=general duration vs track_order=existing-order status/ETA; "
    "get_refund=start vs track_refund=status; set_up_shipping_address=add vs "
    "change_shipping_address=edit; switch_account=change tier/type."
)


def support_system_prompt(
    model: str,
    intents: tuple[str, ...] = SUPPORT_INTENTS,
    *,
    version: str = "v1",
) -> str:
    """Return the exact versioned classification prompt used by collection/runtime."""
    if version not in SUPPORT_PROMPT_VERSIONS:
        raise ValueError(f"unknown support prompt version: {version}")
    prompt = _SUPPORT_BASE_SYSTEM_PREFIX + ", ".join(intents)
    if version == "v2" and model == SUPPORT_MODELS[0]:
        prompt += SUPPORT_GEMINI_GUIDANCE_V2
    return prompt


def grade_support_intent(answer: str, gold: object) -> float:
    return float(extract_intent(str(answer), SUPPORT_INTENTS) == str(gold))


class SupportBenchmark(RealBenchmark):
    """RealBenchmark with a text answer kind for the validator grounding check."""

    answer_kind = "text"


def load_bitext_support_benchmark(
    *,
    exclude_source_indices: set[int] | frozenset[int] = frozenset(),
    n_pool: int = 10_800,
    n_heldout: int = 10_800,
    seed: int = 17,
) -> SupportBenchmark:
    """Load two disjoint, label-balanced sets after excluding the training matrix.

    The scored epoch draws from ``pool``. ``heldout`` is retained for the optional
    fresh-probe audit.  Both are disjoint from every source row used to train the
    artifact, preventing an outcome-matrix lookup from flattering the live result.
    """
    from datasets import load_dataset

    dataset = load_dataset(SUPPORT_DATASET, split="train")
    available = [i for i in range(len(dataset)) if i not in exclude_source_indices]
    labels = [dataset[i]["intent"] for i in available]
    if tuple(sorted(set(labels))) != SUPPORT_INTENTS:
        raise ValueError("Bitext intent vocabulary changed")

    pool_rel = stratified_indices(labels, n_pool, seed)
    pool_set = set(pool_rel)
    remaining = [i for i in range(len(available)) if i not in pool_set]
    held_rel = stratified_indices([labels[i] for i in remaining], n_heldout, seed + 1)
    pool_idx = [available[i] for i in pool_rel]
    held_idx = [available[remaining[i]] for i in held_rel]

    def tasks(indices: list[int]) -> list[BenchTask]:
        return [
            BenchTask(f"bitext-{i:05d}", dataset[i]["instruction"], dataset[i]["intent"])
            for i in indices
        ]

    return SupportBenchmark("support_intent", 1.0, tasks(pool_idx), tasks(held_idx),
                            grade_support_intent)


# Self-contained because these bytes are executed inside the confined child. The
# implementation deliberately mirrors support_pilot._hash_features without importing
# project code, so the public artifact consists only of its source + JSON weights.
SUPPORT_ROUTER_SOURCE = r'''
import hashlib
import json
import math
import re

_TOKEN = re.compile(r"[a-z0-9_]+")

def _features(text, category, dims):
    tokens = _TOKEN.findall(text.lower())
    terms = ["word:" + token for token in tokens]
    terms += ["bigram:" + a + ":" + b for a, b in zip(tokens, tokens[1:])]
    terms.append("category:" + category.lower())
    out = [0.0] * dims
    for term in terms:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        out[raw % (dims - 4)] += 1.0 if digest[-1] & 0x80 else -1.0
    out[-4] = math.log1p(len(tokens))
    out[-3] = math.log1p(len(text))
    out[-2] = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    out[-1] = 1.0
    norm = math.sqrt(sum(value * value for value in out))
    return [value / norm for value in out] if norm > 0 else out

def build_agent(weights):
    cfg = json.loads(weights.decode())
    vectors = cfg["weights"]
    bias = cfg["bias"]
    models = cfg["models"]
    intents = cfg["intents"]
    labels = ", ".join(intents)
    system = ("Classify the customer-support ticket. Return exactly one allowed intent ID "
              "and nothing else. Allowed intent IDs: " + labels)

    def agent(prompt, call_model):
        x = _features(prompt, cfg["category"], cfg["feature_dims"])
        scores = [bias[k] + sum(x[d] * vectors[d][k] for d in range(len(x)))
                  for k in range(len(models))]
        model = models[max(range(len(models)), key=lambda k: scores[k])]
        return call_model(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], {"max_tokens": 24})
    return agent
'''.lstrip()

# Preserve SUPPORT_ROUTER_SOURCE byte-for-byte: it is the v17-measured v1 artifact. The v2 source
# changes only the system prompt chosen after routing, leaving the fitted route computation intact.
_V1_PROMPT_BLOCK = '''    labels = ", ".join(intents)
    system = ("Classify the customer-support ticket. Return exactly one allowed intent ID "
              "and nothing else. Allowed intent IDs: " + labels)
'''
_V2_PROMPT_BLOCK = '''    labels = ", ".join(intents)
    base_system = ("Classify the customer-support ticket. Return exactly one allowed intent ID "
                   "and nothing else. Allowed intent IDs: " + labels)
    cheap_system = (base_system +
        " Disambiguate: check_invoice=find/view existing vs get_invoice=request copy; "
        "delivery_period=general duration vs track_order=existing-order status/ETA; "
        "get_refund=start vs track_refund=status; set_up_shipping_address=add vs "
        "change_shipping_address=edit; switch_account=change tier/type.")
'''
if SUPPORT_ROUTER_SOURCE.count(_V1_PROMPT_BLOCK) != 1:
    raise RuntimeError("support v1 prompt block changed; cannot derive v2 source safely")
SUPPORT_ROUTER_SOURCE_V2 = SUPPORT_ROUTER_SOURCE.replace(_V1_PROMPT_BLOCK, _V2_PROMPT_BLOCK).replace(
    '            {"role": "system", "content": system},',
    '            {"role": "system", "content": (cheap_system if model == models[0] '
    'else base_system)},',
)


@dataclass(frozen=True)
class SupportArtifactBuild:
    artifact: Artifact
    training_tickets: int
    route_mix: dict[str, float]
    source_indices: frozenset[int]


def build_support_artifact(
    matrix_path: str | Path,
    *,
    lam: float = 0.04,
    train_iters: int = 250,
    seed: int = 0,
    prompt_version: str = "v1",
) -> SupportArtifactBuild:
    """Fit and serialize the two-model router on the completed public matrix."""
    if prompt_version not in SUPPORT_PROMPT_VERSIONS:
        raise ValueError(f"unknown support prompt version: {prompt_version}")
    records = [json.loads(line) for line in Path(matrix_path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    data = load_scored_tickets(matrix_path)
    positions = {model.name: k for k, model in enumerate(data.cache.models)}
    missing = [model for model in SUPPORT_MODELS if model not in positions]
    if missing:
        raise ValueError(f"matrix is missing support models: {missing}")
    keep = [positions[model] for model in SUPPORT_MODELS]
    cost = data.cache.cost[:, keep]
    cache = OutcomeCache(
        models=[data.cache.models[k] for k in keep],
        embeddings=data.cache.embeddings,
        correct=data.cache.correct[:, keep],
        cost=cost,
        question_ids=data.cache.question_ids,
        cost_norm=float(cost.max()) if float(cost.max()) > 0 else 1.0,
    )
    fitted = fit_linear_router(cache, lam, iters=train_iters, seed=seed)
    choices = fitted.choose(cache.embeddings)
    payload = {
        "version": 1 if prompt_version == "v1" else 2,
        "feature_dims": cache.D,
        "category": "public_support",
        "models": list(SUPPORT_MODELS),
        "intents": list(SUPPORT_INTENTS),
        "lambda": lam,
        "weights": fitted.weights.tolist(),
        "bias": fitted.bias.tolist(),
    }
    artifact = Artifact(
        SUPPORT_ROUTER_SOURCE if prompt_version == "v1" else SUPPORT_ROUTER_SOURCE_V2,
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        f"support-linear-{prompt_version}",
    )
    route_mix = {
        model: float((choices == k).mean()) for k, model in enumerate(SUPPORT_MODELS)
    }
    by_id = {record["ticket_id"]: record for record in records}
    source_indices = frozenset(int(by_id[ticket_id]["source_index"])
                               for ticket_id in data.cache.question_ids)
    return SupportArtifactBuild(artifact, cache.Q, route_mix, source_indices)
