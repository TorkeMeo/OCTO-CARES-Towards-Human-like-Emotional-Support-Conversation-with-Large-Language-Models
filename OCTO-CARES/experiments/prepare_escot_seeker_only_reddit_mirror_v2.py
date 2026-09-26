#!/usr/bin/env python3
"""Prepare isolated seeker-only-background plus latest-turn mirror rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


PROTOCOL_VERSION = "escot-seeker-only-summary-latest-reddit-mirror-context-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def tagged_context(summary: str, latest: str) -> str:
    return "\n".join(
        [
            "<Seeker-only background>",
            summary,
            "</Seeker-only background>",
            "",
            "<Latest Seeker message>",
            latest,
            "</Latest Seeker message>",
        ]
    )


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            query_id = str(source.get("query_id") or "").strip()
            summary = str(source.get("summary") or "").strip()
            latest = str(source.get("last_seeker") or "").strip()
            prompt_version = str(source.get("summary_prompt_version") or "").strip()
            if not query_id or not summary or not latest:
                raise ValueError(
                    f"Missing query_id/summary/last_seeker at {args.input}:{line_number}"
                )
            if prompt_version != "escot-seeker-only-prompt-constrained-single-pass-summary-v6":
                raise ValueError(
                    f"{query_id}: expected seeker-only summary protocol, got {prompt_version!r}"
                )
            context = tagged_context(summary, latest)
            copied = dict(source)
            copied.update(
                {
                    "target_title": "",
                    "target_body": context,
                    "target_text": context,
                    "dialogue_prefix": context,
                    "generation_context_source": "seeker_only_summary_plus_latest_seeker",
                    "retrieval_query_source": "seeker_only_summary",
                    "mirror_context_protocol_version": PROTOCOL_VERSION,
                    "mirror_context_sha256": hashlib.sha256(
                        context.encode("utf-8")
                    ).hexdigest(),
                    "original_retrieval_file": str(args.input.resolve()),
                }
            )
            rows.append(copied)
            if args.limit and len(rows) >= args.limit:
                break

    if not rows:
        raise ValueError("No rows prepared")
    atomic_write_jsonl(args.output, rows)
    print(f"Prepared {len(rows)} seeker-only mirror rows -> {args.output}")


if __name__ == "__main__":
    main()
