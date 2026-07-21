#!/usr/bin/env python
"""Build a real-model outcome matrix from Bitext customer-support intents.

The output is compatible with ``scripts/support_routing_pilot.py``.  Dataset
timestamps are unavailable, so a seeded stratified shuffle is assigned synthetic
timestamps; this is a public technical holdout, not a production temporal test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend
from thirtyspokes.public_support import extract_intent, stratified_indices
from thirtyspokes.koth.support import SUPPORT_PROMPT_VERSIONS, support_system_prompt


DEFAULT_MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4",
)
PRICE_PER_1K = {
    "google/gemma-3-4b-it": (0.00005, 0.00010),
    "google/gemini-2.5-flash-lite": (0.00010, 0.00040),
    "openai/gpt-5.4-nano": (0.00020, 0.00125),
    "anthropic/claude-sonnet-5": (0.00200, 0.01000),
    "openai/gpt-5.4": (0.00250, 0.01500),
}
_LOCAL = threading.local()


def _backend() -> OpenRouterBackend:
    backend = getattr(_LOCAL, "backend", None)
    if backend is None:
        live = config.LiveConfig()
        live.require_key()
        backend = OpenRouterBackend(
            live.api_key,
            live.base_url,
            timeout=45.0,
            max_retries=2,
            price_fn=lambda model: PRICE_PER_1K[model],
        )
        _LOCAL.backend = backend
    return backend


def _classify(model: str, instruction: str, intents: tuple[str, ...], prompt_version: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": support_system_prompt(model, intents, version=prompt_version),
        },
        {"role": "user", "content": instruction},
    ]
    started = time.monotonic()
    text, tokens_in, tokens_out, cost = _backend().complete(model, messages, {"max_tokens": 24})
    return {
        "prediction": extract_intent(text, intents),
        "cost_usd": cost,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_seconds": time.monotonic() - started,
    }


def _ticket(row: dict, position: int, source_index: int, models: tuple[str, ...],
            intents: tuple[str, ...], prompt_version: str) -> dict:
    outcomes = {}
    for model in models:
        result = _classify(model, row["instruction"], intents, prompt_version)
        outcomes[model] = {
            **result,
            "success": int(result["prediction"] == row["intent"]),
        }
    created = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * position)
    return {
        "ticket_id": f"bitext-{source_index:05d}",
        "created_at": created.isoformat(),
        "category": "public_support",
        "routing_text": row["instruction"],
        "outcomes": outcomes,
        "gold_intent": row["intent"],
        "source_category": row["category"],
        "source_flags": row["flags"],
        "source_index": source_index,
        "prompt_version": prompt_version,
        "split_note": "seeded_stratified_shuffle_with_synthetic_timestamp",
    }


def _indices_from_matrix(path: Path) -> list[int]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    records.sort(key=lambda record: record["created_at"])
    indices = [int(record["source_index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("--indices-from contains duplicate source indices")
    return indices


def _evidence_indices(path: Path) -> set[int]:
    found = set()
    for trace_path in sorted(path.glob("epoch-*/trace.json")):
        for entry in json.loads(trace_path.read_text(encoding="utf-8")):
            task_id = str(entry["task_id"])
            if not task_id.startswith("bitext-"):
                raise ValueError(f"unexpected evidence task ID: {task_id}")
            found.add(int(task_id.removeprefix("bitext-")))
    if not found:
        raise ValueError("--exclude-evidence contains no epoch trace rows")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/bitext-support-scored.jsonl")
    parser.add_argument("--n", type=int, default=2_160,
                        help="number of tickets; 2160 gives 80 per intent")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt-version", choices=SUPPORT_PROMPT_VERSIONS, default="v1")
    parser.add_argument("--indices-from", type=Path,
                        help="reuse the exact source rows and pseudo-time order from a matrix")
    parser.add_argument("--exclude-evidence", type=Path,
                        help="fail if selected rows overlap task IDs in an evidence directory")
    parser.add_argument("--max-new", type=int,
                        help="collect only this many remaining rows (for a resumable smoke)")
    args = parser.parse_args()
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if len(models) < 2:
        raise ValueError("the pilot requires at least two models")
    unknown = [model for model in models if model not in PRICE_PER_1K]
    if unknown:
        raise ValueError(f"add current fallback pricing for models: {unknown}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")

    from datasets import load_dataset
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    intents = tuple(sorted(set(dataset["intent"])))
    indices = (_indices_from_matrix(args.indices_from) if args.indices_from else
               stratified_indices(dataset["intent"], args.n, args.seed))
    if len(indices) != args.n:
        raise ValueError(f"selected {len(indices)} rows but --n is {args.n}")
    if args.exclude_evidence:
        overlap = set(indices) & _evidence_indices(args.exclude_evidence)
        if overlap:
            raise ValueError(f"selected rows overlap {len(overlap)} evidence tasks")
    jobs = [(position, index, dataset[index]) for position, index in enumerate(indices)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = Path(f"{output}.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another collector is writing {output}") from exc
    completed: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("prompt_version", "v1") != args.prompt_version:
                    raise ValueError("output contains a different prompt version")
                if tuple(sorted(record["outcomes"])) != tuple(sorted(models)):
                    raise ValueError("output contains a different model pool")
                completed.add(record["ticket_id"])
    jobs = [job for job in jobs if f"bitext-{job[1]:05d}" not in completed]
    if args.max_new is not None:
        if args.max_new < 1:
            raise ValueError("--max-new must be positive")
        jobs = jobs[:args.max_new]
    print(f"dataset={len(dataset)} intents={len(intents)} selected={args.n} "
          f"scheduled={len(jobs)} already_complete={len(completed)} models={len(models)} "
          f"prompt={args.prompt_version}", flush=True)

    total_cost = 0.0
    done = len(completed)
    failed = 0
    with output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_ticket, row, position, index, models, intents,
                                args.prompt_version): index
                for position, index, row in jobs
            }
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 — leave ticket resumable for the next pass
                    failed += 1
                    print(f"ticket_failed={futures[future]} error={type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                done += 1
                total_cost += sum(item["cost_usd"] for item in record["outcomes"].values())
                if done % 50 == 0 or done == args.n:
                    print(f"completed={done}/{args.n} new_cost=${total_cost:.4f}", flush=True)
    print(f"run_complete={done}/{args.n} scheduled_this_pass={len(jobs)} "
          f"failed_this_pass={failed} new_cost=${total_cost:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
