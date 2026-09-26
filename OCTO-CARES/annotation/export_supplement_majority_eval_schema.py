#!/usr/bin/env python3
"""Export supplement majority-vote labels into the 8-type eval schema.

The three-model majority file stores labels as majority_labels and preserves
large per-model annotation details. The 8_Type_attention eval code expects the
older compact schema: labels, label_vector, title, body, text, and source_post.

This script only rewrites field names and removes records that are not complete.
It does not relabel or change any 0/1 decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from post_labels import POST_LABEL_KEYS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "outputs"
    / "supplement_three_model_post_labels"
    / "supplement_post_labels_majority_vote.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "outputs"
    / "supplement_three_model_post_labels"
    / "supplement_post_labels_majority_vote_eval_schema.json"
)
DEFAULT_SUMMARY = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keep-model-annotations",
        action="store_true",
        help="Keep full per-model annotation details. Disabled by default to keep eval files small.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"Input JSON must be a list or contain records: {path}")


def coerce_binary(value: Any) -> int | None:
    if value in (0, 0.0, False):
        return 0
    if value in (1, 1.0, True):
        return 1
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def source_post(record: dict[str, Any]) -> dict[str, Any]:
    source_record = record.get("source_record")
    if isinstance(source_record, dict) and isinstance(source_record.get("source_post"), dict):
        return source_record["source_post"]
    direct = record.get("source_post")
    if isinstance(direct, dict):
        return direct
    return {}


def selected_comment(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("selected_comment")
    if isinstance(value, dict):
        return value
    source_record = record.get("source_record")
    if isinstance(source_record, dict) and isinstance(source_record.get("selected_comment"), dict):
        return source_record["selected_comment"]
    return None


def record_text_parts(record: dict[str, Any]) -> tuple[str, str, str]:
    post = source_post(record)
    title = str(record.get("post_title") or record.get("title") or post.get("title") or "").strip()
    body = str(
        record.get("post_body")
        or record.get("body")
        or record.get("selftext")
        or post.get("selftext")
        or post.get("body")
        or ""
    ).strip()
    text = str(record.get("text") or "").strip()
    if not text:
        text = "\n\n".join(part for part in (title, body) if part)
    return title, body, text


def convert_record(record: dict[str, Any], source_index: int, keep_model_annotations: bool) -> tuple[dict[str, Any] | None, str | None]:
    if record.get("status") != "complete":
        return None, "status_not_complete"

    post_id = str(record.get("post_id") or "").strip()
    if not post_id:
        return None, "missing_post_id"

    title, body, text = record_text_parts(record)
    if not text:
        return None, "missing_text"

    raw_labels = record.get("majority_labels") or record.get("labels")
    if not isinstance(raw_labels, dict):
        return None, "missing_majority_labels"

    labels = {key: coerce_binary(raw_labels.get(key)) for key in POST_LABEL_KEYS}
    if any(value is None for value in labels.values()):
        return None, "incomplete_labels"
    label_vector = [int(labels[key]) for key in POST_LABEL_KEYS]

    post = source_post(record)
    subreddit = str(record.get("source_subreddit") or record.get("subreddit") or post.get("subreddit") or "unknown")
    converted = {
        "post_id": post_id,
        "annotation_number": record.get("annotation_number") or source_index,
        "combined_record_number": record.get("combined_record_number"),
        "dataset_name": "supplement_majority_vote_eval_schema",
        "dataset_source": "supplement_three_model_majority_vote",
        "status": "complete",
        "source_name": record.get("source_name"),
        "source_subreddit": subreddit,
        "subreddit": subreddit,
        "title": title,
        "body": body,
        "text": text,
        "labels": labels,
        "label_vector": label_vector,
        "label_order": list(POST_LABEL_KEYS),
        "majority_vote_details": record.get("vote_details") or {},
        "selected_comment_id": record.get("selected_comment_id"),
        "selected_comment": selected_comment(record),
        "source_post": post or {"id": post_id, "title": title, "selftext": body, "subreddit": subreddit},
        "origin_file": record.get("origin_file"),
        "origin_file_index": record.get("origin_file_index"),
        "prompt_version": record.get("prompt_version"),
        "converted_at": timestamp(),
    }
    if keep_model_annotations:
        converted["model_annotations"] = record.get("model_annotations") or {}
    return converted, None


def main() -> None:
    args = parse_args()
    records = records_from_payload(read_json(args.input), args.input)
    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_post_ids: list[str] = []

    for index, record in enumerate(records, start=1):
        item, reason = convert_record(record, index, args.keep_model_annotations)
        if item is None:
            skipped.append({"index": index, "post_id": record.get("post_id"), "reason": reason})
            continue
        if item["post_id"] in seen:
            duplicate_post_ids.append(item["post_id"])
            skipped.append({"index": index, "post_id": item["post_id"], "reason": "duplicate_post_id"})
            continue
        seen.add(item["post_id"])
        converted.append(item)

    label_positive_counts = {
        key: sum(int(record["labels"][key]) for record in converted)
        for key in POST_LABEL_KEYS
    }
    summary = {
        "created_at": timestamp(),
        "input": str(args.input),
        "output": str(args.output),
        "raw_records": len(records),
        "converted_records": len(converted),
        "skipped_records": len(skipped),
        "skipped_first_20": skipped[:20],
        "duplicate_post_id_count": len(duplicate_post_ids),
        "duplicate_post_ids_first_20": duplicate_post_ids[:20],
        "label_order": list(POST_LABEL_KEYS),
        "label_positive_counts": label_positive_counts,
        "label_negative_counts": {key: len(converted) - count for key, count in label_positive_counts.items()},
        "keep_model_annotations": bool(args.keep_model_annotations),
    }
    write_json_atomic(args.output, converted)
    write_json_atomic(args.summary, summary)
    print(f"Wrote eval-schema data: {args.output}")
    print(f"Wrote summary: {args.summary}")
    print(f"Converted records: {len(converted)}/{len(records)}")


if __name__ == "__main__":
    main()
