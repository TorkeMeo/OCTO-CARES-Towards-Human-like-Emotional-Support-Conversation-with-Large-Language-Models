#!/usr/bin/env python3
"""Label complete Reddit posts with independent binary Bailian API calls."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from post_labels import DECISION_RULE_TEXT, POST_LABELS


SYSTEM_PROMPT = """You are a careful research annotator.
Your task is to judge only one category for one Reddit post.
Use only the supplied category name, category definition, and Reddit post.
Consider the title and body together, including negation, quoted speech, sarcasm, and who is experiencing the event or emotion.
Return exactly one word: Yes or No.
Do not return explanations, punctuation, bullet points, JSON, Markdown, or any other text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="original_pair_data_re1_all.json",
        help=(
            "Input JSON file, glob pattern, or comma-separated patterns. "
            "Default: original_pair_data_re1_all.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/post_labels_re1.json"))
    parser.add_argument("--model", default=os.getenv("BAILIAN_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_yes_no(content: str) -> int | None:
    normalized = re.sub(r"[^a-z]", "", content.lower())
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    return None


def format_exception(exc: Exception) -> str:
    messages = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause is not None:
        messages.append(f"caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " | ".join(messages)


def make_prompt(text: str, label_name: str, definition: str) -> str:
    return f"""Decide whether the Reddit post contains this one category.

Category:
{label_name}

Definition:
{definition}

Decision rule:
{DECISION_RULE_TEXT}

Reddit post:
<post>
{text}
</post>

Answer:"""


def request_label(
    client: OpenAI,
    model: str,
    text: str,
    label_name: str,
    definition: str,
    max_retries: int,
) -> tuple[int | None, str | None, str | None]:
    last_content: str | None = None
    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": make_prompt(text, label_name, definition),
                    },
                ],
            )
            last_content = response.choices[0].message.content or ""
            parsed = parse_yes_no(last_content)
            if parsed is not None:
                return parsed, last_content, None
            last_error = f"Unparseable response: {last_content!r}"
        except Exception as exc:  # API errors vary by SDK/provider version.
            last_error = format_exception(exc)

        if attempt + 1 < max_retries:
            time.sleep(2**attempt)
    return None, last_content, last_error


def resolve_input_paths(input_pattern: str) -> list[Path]:
    patterns = [item.strip() for item in input_pattern.split(",") if item.strip()]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(path) for path in glob.glob(pattern))
    if matches:
        return sorted(set(matches))

    path = Path(input_pattern)
    if path.exists():
        return [path]

    raise FileNotFoundError(f"No input JSON files matched: {input_pattern}")


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return records
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Input JSON must contain a records array or a record list: {path}")


def extract_posts_from_pair_data(input_pattern: str) -> list[dict[str, Any]]:
    paths = resolve_input_paths(input_pattern)
    posts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for pair in records_from_payload(payload, path):
            post = pair.get("source_post") or {}
            post_id = str(post.get("id") or "").strip()
            if not post_id or post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            title = str(post.get("title") or "").strip()
            body = str(post.get("selftext") or "").strip()
            text = "\n\n".join(part for part in (title, body) if part)
            if not text:
                continue

            posts.append(
                {
                    "post_id": post_id,
                    "annotation_number": pair.get("annotation_number"),
                    "pair_record_number": (
                        pair.get("combined_record_number")
                        or pair.get("record_number")
                        or pair.get("number")
                    ),
                    "input_file": path.name,
                    "subreddit": post.get("subreddit") or pair.get("source_subreddit"),
                    "author": post.get("author"),
                    "created_utc": post.get("created_utc"),
                    "title": title,
                    "body": body,
                    "text": text,
                    "selected_comment_id": (pair.get("selected_comment") or {}).get("id"),
                }
            )
    return posts


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError("Existing output JSON must be an array")


def is_complete_result(item: dict[str, Any]) -> bool:
    return item.get("status") == "complete" and bool(item.get("post_id"))


def dedupe_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep one record per post_id; complete records beat incomplete records."""
    deduped: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    duplicate_count = 0
    for item in results:
        post_id = item.get("post_id")
        if not post_id:
            deduped.append(item)
            continue
        post_id = str(post_id)
        if post_id not in indexes:
            indexes[post_id] = len(deduped)
            deduped.append(item)
            continue
        duplicate_count += 1
        existing_index = indexes[post_id]
        existing = deduped[existing_index]
        if is_complete_result(item) and not is_complete_result(existing):
            deduped[existing_index] = item
        elif is_complete_result(item) == is_complete_result(existing):
            deduped[existing_index] = item
    return deduped, duplicate_count


def completed_ids(results: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["post_id"])
        for item in results
        if item.get("status") == "complete" and item.get("post_id")
    }


def result_indexes_by_post_id(results: list[dict[str, Any]]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, item in enumerate(results):
        post_id = item.get("post_id")
        if post_id:
            indexes[str(post_id)] = index
    return indexes


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise SystemExit("BAILIAN_API_KEY is required")
    if not args.model:
        raise SystemExit("Set BAILIAN_MODEL or pass --model")
    if not args.base_url:
        raise SystemExit("Set BAILIAN_BASE_URL or pass --base-url")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    posts = extract_posts_from_pair_data(args.input)
    if args.limit is not None:
        posts = posts[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = read_existing_results(args.output)
    results, duplicate_count = dedupe_results(results)
    if duplicate_count:
        print(f"Deduplicated {duplicate_count} existing output records by post_id")
        write_results(args.output, results)
    done = completed_ids(results)
    result_indexes = result_indexes_by_post_id(results)

    for index, post in enumerate(posts, start=1):
        post_id = str(post["post_id"])
        if post_id in done:
            continue

        labels: dict[str, int | None] = {}
        raw_responses: dict[str, str | None] = {}
        errors: dict[str, str] = {}
        for definition in POST_LABELS:
            value, raw, error = request_label(
                client=client,
                model=args.model,
                text=post["text"],
                label_name=definition.name_en,
                definition=definition.definition_en,
                max_retries=args.max_retries,
            )
            labels[definition.key] = value
            raw_responses[definition.key] = raw
            if error:
                errors[definition.key] = error

        status = "complete" if not errors else "incomplete"
        result = {
            "post_id": post_id,
            "annotation_number": post.get("annotation_number"),
            "pair_record_number": post.get("pair_record_number"),
            "input_file": post.get("input_file"),
            "subreddit": post.get("subreddit"),
            "selected_comment_id": post.get("selected_comment_id"),
            "labels": labels,
            "label_vector": [labels[item.key] for item in POST_LABELS],
            "status": status,
            "errors": errors,
            "raw_responses": raw_responses,
            "model": args.model,
            "prompt_version": "post-binary-any-evidence-v3",
        }
        if post_id in result_indexes:
            results[result_indexes[post_id]] = result
        else:
            result_indexes[post_id] = len(results)
            results.append(result)
        done.add(post_id)
        write_results(args.output, results)
        print(f"[{index}/{len(posts)}] {post_id}: {status}")


if __name__ == "__main__":
    main()
