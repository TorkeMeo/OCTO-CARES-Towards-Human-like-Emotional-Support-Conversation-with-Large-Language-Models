#!/usr/bin/env python3
"""Validate and merge seeker-only summary worker shards in source order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import summarize_escot_bailian_rolecard as base


SUMMARY_PROTOCOL = "escot-seeker-only-prompt-constrained-single-pass-summary-v6"
SUMMARY_STYLE = "first_person_seeker_reddit_narrative"
SUMMARY_PIPELINE = "prompt_constrained_single_pass_narrative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", default="")
    return parser.parse_args()


def read_shard(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary shard: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.worker_count <= 0 or len(args.shard) != args.worker_count:
        raise ValueError("worker-count must equal the number of --shard arguments")

    records = base.records_from_payload(base.read_json(args.input), args.input)
    if args.limit > 0:
        records = records[: args.limit]
    expected_ids = [f"escot_{str(record.get('id') or '').strip()}" for record in records]
    if not expected_ids or any(query_id == "escot_" for query_id in expected_ids):
        raise ValueError("Input has no usable records or contains a missing id")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Input contains duplicate ids")

    by_id: dict[str, dict[str, Any]] = {}
    for worker_index, path in enumerate(args.shard):
        rows = read_shard(path)
        expected_worker_ids = expected_ids[worker_index :: args.worker_count]
        actual_worker_ids = [str(row.get("query_id") or "") for row in rows]
        if actual_worker_ids != expected_worker_ids:
            raise ValueError(
                f"Worker {worker_index} coverage/order mismatch: "
                f"expected={len(expected_worker_ids)} got={len(actual_worker_ids)}"
            )
        for row in rows:
            query_id = str(row.get("query_id") or "")
            if query_id in by_id:
                raise ValueError(f"Duplicate query across summary shards: {query_id}")
            if not str(row.get("summary") or "").strip():
                raise ValueError(f"{query_id}: empty summary")
            if row.get("summary_prompt_version") != SUMMARY_PROTOCOL:
                raise ValueError(f"{query_id}: wrong summary protocol")
            if row.get("summary_model") != args.model:
                raise ValueError(f"{query_id}: wrong summary model")
            if args.model_path and row.get("summary_model_path") != args.model_path:
                raise ValueError(f"{query_id}: wrong summary model path")
            if "summary_evidence" in row:
                raise ValueError(f"{query_id}: v6 must not contain summary_evidence")
            by_id[query_id] = row

    if set(by_id) != set(expected_ids):
        raise ValueError("Merged summary coverage differs from input")
    ordered = [by_id[query_id] for query_id in expected_ids]
    base.write_jsonl(args.output, ordered)
    base.write_json(
        args.manifest,
        {
            "created_at": base.timestamp(),
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "model": args.model,
            "model_path": args.model_path or None,
            "record_count": len(ordered),
            "summary_count": len(ordered),
            "error_count": 0,
            "errors": [],
            "worker_count": args.worker_count,
            "shards": [str(path.resolve()) for path in args.shard],
            "leakage_policy": {
                "dialogue_cut": "through_last_seeker_turn",
                "reference_response_read": False,
                "strategy_read": False,
                "cot_data_read": False,
                "supporter_turns_visible_to_model": True,
                "supporter_advice_controlled_by_prompt": True,
            },
            "summary_style": SUMMARY_STYLE,
            "summary_pipeline": SUMMARY_PIPELINE,
            "summary_format": "natural first-person narrative, 100-400 words",
            "prompt_version": SUMMARY_PROTOCOL,
        },
    )
    print(f"Merged {len(ordered)} summaries from {len(args.shard)} shards -> {args.output}")


if __name__ == "__main__":
    main()
