#!/usr/bin/env python3
"""Validate coverage, ordering, protocols, and prompt-use audits for v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FILES = (
    ("pure_qwen3", "four_methods/top1/responses.pure_qwen3.jsonl", "escot-seeker-only-reddit-four-method-latest-priority-v2"),
    ("attention_post", "four_methods/top1/responses.attention_post.jsonl", "escot-seeker-only-reddit-four-method-latest-priority-v2"),
    ("attention_post_comment", "four_methods/top1/responses.attention_post_comment.jsonl", "escot-seeker-only-reddit-four-method-latest-priority-v2"),
    ("attention_post_random_comment", "four_methods/top1/responses.attention_post_random_comment.jsonl", "escot-seeker-only-reddit-four-method-latest-priority-v2"),
    ("simple_persona", "simple_persona/generation/responses.simple_persona.jsonl", "escot-seeker-only-persona-reply-latest-priority-v2"),
    ("reddit_multiagent", "reddit_multiagent/responses.reddit_multiagent.jsonl", "escot-seeker-only-multiagentesc-reddit-latest-priority-v2"),
)


def read(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or not str(row.get("response") or "").strip():
                raise ValueError(f"failed/empty row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty file: {path}")
    return rows


def read_inputs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("query_id") or "").strip():
                raise ValueError(f"input row lacks query_id at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty input file: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=341)
    args = parser.parse_args()

    input_path = args.root / "input/seeker_only_summary_latest_retrieval.jsonl"
    input_rows = read_inputs(input_path)
    if len(input_rows) != args.expected_count:
        raise ValueError(
            f"mirror input: expected {args.expected_count}, got {len(input_rows)}"
        )
    expected_ids = [str(row["query_id"]) for row in input_rows]
    input_by_id = {str(row["query_id"]): row for row in input_rows}
    if len(input_by_id) != len(input_rows):
        raise ValueError("mirror input has duplicate query IDs")

    reference: list[str] | None = None
    for condition, relative, protocol in FILES:
        path = args.root / relative
        rows = read(path)
        if len(rows) != args.expected_count:
            raise ValueError(f"{condition}: expected {args.expected_count}, got {len(rows)}")
        ids = [str(row.get("query_id") or "") for row in rows]
        if len(set(ids)) != len(ids) or not all(ids):
            raise ValueError(f"{condition}: missing or duplicate query IDs")
        if ids != expected_ids:
            raise ValueError(f"{condition}: query order differs from current mirror input")
        if reference is None:
            reference = ids
        elif ids != reference:
            raise ValueError(f"{condition}: query order differs")
        for row in rows:
            source = input_by_id[str(row.get("query_id") or "")]
            if row.get("condition") != condition:
                raise ValueError(f"{condition}/{row.get('query_id')}: wrong condition")
            if row.get("generation_protocol_version") != protocol:
                raise ValueError(f"{condition}/{row.get('query_id')}: wrong protocol")
            if row.get("summary_used_in_generation_prompt") is not True:
                raise ValueError(f"{condition}/{row.get('query_id')}: summary audit is false")
            if row.get("last_seeker_used_in_generation_prompt") is not True:
                raise ValueError(f"{condition}/{row.get('query_id')}: latest-turn audit is false")
            uses_rag = condition.startswith("attention_")
            if bool(row.get("retrieval_used_in_prompt")) != uses_rag:
                raise ValueError(f"{condition}/{row.get('query_id')}: RAG audit mismatch")
            if uses_rag and not row.get("prompt_memory_ids"):
                raise ValueError(f"{condition}/{row.get('query_id')}: no prompt memory IDs")
            if row.get("dialogue_prefix") != source.get("dialogue_prefix"):
                raise ValueError(
                    f"{condition}/{row.get('query_id')}: stale generation context"
                )
            if row.get("last_seeker") != source.get("last_seeker"):
                raise ValueError(
                    f"{condition}/{row.get('query_id')}: stale latest Seeker turn"
                )
            if "input_summary" in row and row.get("input_summary") != source.get("summary"):
                raise ValueError(
                    f"{condition}/{row.get('query_id')}: stale seeker-only summary"
                )
        print(f"OK {condition}: {len(rows)} rows | {path}")
    print(f"VALIDATION_OK methods={len(FILES)} rows_each={args.expected_count}")


if __name__ == "__main__":
    main()
