#!/usr/bin/env python3
"""Strictly join a complete Qwen label cache to source posts and supplied split IDs.

This is a packaging utility, not the historical staged dataset construction.
It never invents labels, a split, or a completeness-based subset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from post_labels import POST_LABEL_KEYS


def records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty record list: {path}")
    return rows


def build(source, labels, split, expected_count, model):
    cache = {}
    for row in labels:
        pid = str(row.get("post_id") or "").strip()
        if not pid or pid in cache:
            raise ValueError("Missing/duplicate label-cache post_id")
        if row.get("status") != "complete" or row.get("model") != model:
            raise ValueError(f"Incomplete/wrong-model annotation: {pid}")
        values = row.get("labels") or {}
        if any(type(values.get(key)) is not int or values[key] not in (0, 1) for key in POST_LABEL_KEYS):
            raise ValueError(f"Invalid eight-label annotation: {pid}")
        cache[pid] = row
    test_ids = split.get("test_post_ids")
    if not isinstance(test_ids, list) or not test_ids or len(test_ids) != len(set(test_ids)):
        raise ValueError("Split manifest must supply unique nonempty test_post_ids")
    if any(not isinstance(pid, str) or not pid for pid in test_ids):
        raise ValueError("Split IDs must be nonempty strings")
    test_ids = set(test_ids)
    output, seen = [], set()
    for row in source:
        post = row.get("source_post") or {}
        pid = str(post.get("id") or "").strip()
        if not pid or pid in seen or pid not in cache:
            raise ValueError(f"Missing/duplicate/unannotated source post: {pid}")
        seen.add(pid)
        title = str(post.get("title") or "").strip()
        body = str(post.get("selftext") or "").strip()
        text = "\n\n".join(part for part in (title, body) if part)
        if not text:
            raise ValueError(f"Empty source post: {pid}")
        annotation = cache[pid]
        # The source may contain an unfilled historical manual-annotation
        # template. It is provenance, never a target label for this pipeline.
        source_fields = {key: value for key, value in row.items() if key != "human_annotation"}
        output.append({**source_fields, **annotation, "post_id": pid, "title": title,
                       "body": body, "text": text, "subreddit": post.get("subreddit"),
                       "label_vector": [annotation["labels"][key] for key in POST_LABEL_KEYS],
                       "fixed_split": "test" if pid in test_ids else "train"})
    if seen != set(cache) or not test_ids < seen or len(output) != expected_count:
        raise ValueError("Coverage/count mismatch or no train partition; no output written")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1600)
    parser.add_argument("--model", default="qwen3.7-max")
    args = parser.parse_args()
    inputs = [args.source, args.labels, args.split_manifest]
    if args.output.exists() or args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Use a new output file; refusing to overwrite")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    rows = build(records(args.source), records(args.labels), split, args.expected_count, args.model)
    payload = {"records": rows, "metadata": {
        "construction": "strict_source_cache_split_join_v1",
        "historical_staging_reproduced": False,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Joined {len(rows)} records -> {args.output}")


if __name__ == "__main__":
    main()
