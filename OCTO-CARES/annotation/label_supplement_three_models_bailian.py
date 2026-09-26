#!/usr/bin/env python3
"""Label supplement Reddit posts with three Bailian models and majority vote.

This script keeps the original single-category binary annotation style:
each post label is queried independently and must parse as exact Yes/No.

Outputs:
  1. one complete JSON file per model
  2. one majority-vote JSON file combining the three model files

The script is resumable. It saves after every single label response.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from post_labels import DECISION_RULE_TEXT, POST_LABELS, POST_LABEL_KEYS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "origin_pair_data_re1_supplement_all.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "supplement_three_model_post_labels"
PROMPT_VERSION = "post-binary-any-evidence-v3-three-model-supplement"

DEFAULT_MODELS = {
    "qwen37_max": "qwen3.7-max",
    "kimi_k3": "kimi-k3",
    "deepseek_v4_pro_0813": "deepseek-v4-pro-0813",
}

SYSTEM_PROMPT = """You are a careful research annotator.
Your task is to judge only one category for one Reddit post.
Use only the supplied category name, category definition, and Reddit post.
Consider the title and body together, including negation, quoted speech, sarcasm, and who is experiencing the event or emotion.
Return exactly one word: Yes or No.
Do not return explanations, punctuation, bullet points, JSON, Markdown, or any other text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label supplement posts with qwen3.7-max, kimi-k3, deepseek-v4-pro-0813, then majority vote."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL"))
    parser.add_argument("--api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument(
        "--models",
        default=",".join(f"{alias}={model}" for alias, model in DEFAULT_MODELS.items()),
        help="Comma-separated alias=model pairs.",
    )
    parser.add_argument(
        "--only-models",
        default=None,
        help="Optional comma-separated aliases to run. Majority file still uses all configured aliases.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--retry-max-tokens",
        default="128,256,512",
        help="Comma-separated max_tokens values tried across retries for parse/API failures.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per API request timeout in seconds. Prevents one stuck request from blocking the run forever.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Relabel even if a record is already complete.")
    parser.add_argument(
        "--majority-only",
        action="store_true",
        help="Do not call APIs; rebuild only the majority-vote file from existing model files.",
    )
    parser.add_argument(
        "--allow-partial-vote",
        action="store_true",
        help="Allow 2 matching valid model votes to decide a label when the third vote is missing.",
    )
    parser.add_argument(
        "--drop-incomplete-models",
        default="",
        help="Comma-separated model aliases. During majority merge, drop any source record where these aliases have missing labels.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_models(text: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --models item {item!r}; expected alias=model")
        alias, model = item.split("=", 1)
        alias = alias.strip()
        model = model.strip()
        if not alias or not model:
            raise ValueError(f"Bad --models item {item!r}; alias and model are required")
        models[alias] = model
    if not models:
        raise ValueError("--models did not include any models")
    return models


def parse_alias_list(text: str | None, models: dict[str, str]) -> list[str]:
    if not text:
        return list(models)
    aliases = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [alias for alias in aliases if alias not in models]
    if unknown:
        raise ValueError(f"Unknown --only-models aliases: {', '.join(unknown)}")
    return aliases


def parse_positive_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Token retry values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("--retry-max-tokens must include at least one integer")
    return values


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
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
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Input JSON must contain a records array or a record list: {path}")


def selected_comment_id(record: dict[str, Any]) -> str | None:
    selected = record.get("selected_comment") or {}
    value = selected.get("id") or record.get("selected_comment_id")
    return str(value) if value else None


def extract_post_text(record: dict[str, Any]) -> tuple[str, str, str]:
    post = record.get("source_post") or {}
    title = str(post.get("title") or "").strip()
    body = str(post.get("selftext") or "").strip()
    text = "\n\n".join(part for part in (title, body) if part)
    return title, body, text


def post_id_from_record(record: dict[str, Any]) -> str:
    post = record.get("source_post") or {}
    return str(post.get("id") or record.get("post_id") or "").strip()


def load_source_records(input_path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = read_json(input_path)
    records = records_from_payload(payload, input_path)
    extracted: list[dict[str, Any]] = []
    seen_post_ids: set[str] = set()
    duplicate_count = 0

    for fallback_number, record in enumerate(records, start=1):
        post_id = post_id_from_record(record)
        if not post_id:
            continue
        if post_id in seen_post_ids:
            duplicate_count += 1
            continue
        seen_post_ids.add(post_id)

        title, body, text = extract_post_text(record)
        if not text:
            continue
        extracted.append(
            {
                "post_id": post_id,
                "annotation_number": record.get("annotation_number") or fallback_number,
                "combined_record_number": record.get("combined_record_number"),
                "origin_file": record.get("origin_file"),
                "origin_file_index": record.get("origin_file_index"),
                "source_name": record.get("source_name"),
                "source_subreddit": record.get("source_subreddit"),
                "selected_comment_id": selected_comment_id(record),
                "post_title": title,
                "post_body": body,
                "post_text": text,
                "source_record": record,
            }
        )

    if duplicate_count:
        print(f"Skipped duplicate post_id records: {duplicate_count}")
    if limit is not None:
        extracted = extracted[:limit]
    return extracted


def parse_yes_no(content: str) -> int | None:
    """Parse a binary answer without letting hidden reasoning poison content.

    Most Bailian OpenAI-compatible models return the usable final answer in
    message.content and optional chain-of-thought style text in
    reasoning_content. The final answer should be parsed from content first.
    This parser also accepts common final-answer wrappers for retry/fallback
    cases, but it does not treat any random Yes/No inside prose as a label.
    """
    if not content:
        return None
    normalized = re.sub(r"[^a-z]", "", content.lower())
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0

    for line in reversed([line.strip() for line in content.splitlines() if line.strip()]):
        line_normalized = re.sub(r"[^a-z]", "", line.lower())
        if line_normalized == "yes":
            return 1
        if line_normalized == "no":
            return 0

    final_answer_patterns = (
        r"(?im)^\s*(?:answer|final answer|output|label|result)\s*[:：]\s*(yes|no)\s*[.!。]?\s*$",
        r"(?im)^\s*(?:the\s+)?(?:answer|final answer|output|label|result)\s+is\s+(yes|no)\s*[.!。]?\s*$",
    )
    for pattern in final_answer_patterns:
        match = re.search(pattern, content)
        if match:
            return 1 if match.group(1).lower() == "yes" else 0
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


def plain_object(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [plain_object(item) for item in value]
    if isinstance(value, tuple):
        return [plain_object(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain_object(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return plain_object(value.model_dump())
    if hasattr(value, "dict"):
        return plain_object(value.dict())
    return str(value)


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def message_field(message: Any, field_name: str) -> Any:
    if hasattr(message, field_name):
        value = getattr(message, field_name)
        if value is not None:
            return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get(field_name) is not None:
        return extra.get(field_name)
    dumped = plain_object(message)
    if isinstance(dumped, dict):
        return dumped.get(field_name)
    return None


def response_text_from_message(message: Any) -> str:
    # Prefer the final assistant-facing answer. Full message details, including
    # reasoning_content, are still persisted in raw_response_details.
    for field_name in ("content", "text", "reasoning_content"):
        part = stringify_content(message_field(message, field_name)).strip()
        if part:
            return part
    return ""


def parse_yes_no_from_message(message: Any) -> int | None:
    for field_name in ("content", "text", "reasoning_content"):
        part = stringify_content(message_field(message, field_name)).strip()
        parsed = parse_yes_no(part)
        if parsed is not None:
            return parsed
    return None


def response_debug(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    return {
        "finish_reason": getattr(choice, "finish_reason", None),
        "message": plain_object(choice.message),
        "usage": plain_object(getattr(response, "usage", None)),
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
    }


def token_budget_for_attempt(token_budgets: list[int], attempt: int) -> int:
    return token_budgets[min(attempt, len(token_budgets) - 1)]


def request_label(
    client: OpenAI,
    model: str,
    text: str,
    label_name: str,
    definition: str,
    max_retries: int,
    retry_max_tokens: list[int],
    request_timeout: float,
    progress_prefix: str = "",
) -> tuple[int | None, str | None, str | None, dict[str, Any] | None]:
    last_content: str | None = None
    last_error: str | None = None
    last_debug: dict[str, Any] | None = None
    for attempt in range(max_retries):
        max_tokens = token_budget_for_attempt(retry_max_tokens, attempt)
        if progress_prefix:
            print(
                f"{progress_prefix} attempt {attempt + 1}/{max_retries} "
                f"max_tokens={max_tokens}",
                flush=True,
            )
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                timeout=request_timeout,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_prompt(text, label_name, definition)},
                ],
            )
            last_debug = response_debug(response)
            last_debug["attempt"] = attempt + 1
            last_debug["request_max_tokens"] = max_tokens
            last_content = response_text_from_message(response.choices[0].message)
            parsed = parse_yes_no_from_message(response.choices[0].message)
            if parsed is not None:
                return parsed, last_content, None, last_debug
            last_error = f"Unparseable response at max_tokens={max_tokens}: {last_content!r}"
        except Exception as exc:
            last_error = f"max_tokens={max_tokens}: {format_exception(exc)}"

        if progress_prefix:
            print(f"{progress_prefix} attempt {attempt + 1} failed: {last_error[:500]}", flush=True)
        if attempt + 1 < max_retries:
            time.sleep(2**attempt)
    return None, last_content, last_error, last_debug


def model_output_path(output_dir: Path, alias: str) -> Path:
    return output_dir / f"supplement_post_labels_{alias}.json"


def majority_output_path(output_dir: Path) -> Path:
    return output_dir / "supplement_post_labels_majority_vote.json"


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Existing output JSON must be an array: {path}")


def normalize_label_value(value: Any) -> int | None:
    if value in (0, 0.0, False):
        return 0
    if value in (1, 1.0, True):
        return 1
    return None


def has_complete_labels(record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    labels = record.get("labels") or {}
    return all(normalize_label_value(labels.get(key)) is not None for key in POST_LABEL_KEYS)


def is_complete_record(record: dict[str, Any]) -> bool:
    labels = record.get("labels") or {}
    return (
        record.get("status") == "complete"
        and all(normalize_label_value(labels.get(key)) is not None for key in POST_LABEL_KEYS)
    )


def dedupe_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    duplicate_count = 0
    for item in results:
        post_id = str(item.get("post_id") or "").strip()
        if not post_id:
            deduped.append(item)
            continue
        if post_id not in indexes:
            indexes[post_id] = len(deduped)
            deduped.append(item)
            continue
        duplicate_count += 1
        current_index = indexes[post_id]
        current = deduped[current_index]
        if is_complete_record(item) and not is_complete_record(current):
            deduped[current_index] = item
        elif is_complete_record(item) == is_complete_record(current):
            deduped[current_index] = item
    return deduped, duplicate_count


def result_indexes_by_post_id(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(record["post_id"]): index
        for index, record in enumerate(results)
        if record.get("post_id")
    }


def make_empty_model_record(source: dict[str, Any], alias: str, model: str) -> dict[str, Any]:
    return {
        "post_id": source["post_id"],
        "annotation_number": source.get("annotation_number"),
        "combined_record_number": source.get("combined_record_number"),
        "origin_file": source.get("origin_file"),
        "origin_file_index": source.get("origin_file_index"),
        "source_name": source.get("source_name"),
        "source_subreddit": source.get("source_subreddit"),
        "selected_comment_id": source.get("selected_comment_id"),
        "post_title": source.get("post_title"),
        "post_body": source.get("post_body"),
        "selected_comment": (source.get("source_record") or {}).get("selected_comment"),
        "source_record": source.get("source_record"),
        "labels": {key: None for key in POST_LABEL_KEYS},
        "label_vector": [None for _ in POST_LABEL_KEYS],
        "status": "incomplete",
        "errors": {},
        "raw_responses": {},
        "raw_response_details": {},
        "model_alias": alias,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "created_at": timestamp(),
        "updated_at": timestamp(),
    }


def refresh_model_record_metadata(record: dict[str, Any], source: dict[str, Any], alias: str, model: str) -> None:
    """Keep old label progress but ensure full current source data is present."""
    record.update(
        {
            "annotation_number": source.get("annotation_number"),
            "combined_record_number": source.get("combined_record_number"),
            "origin_file": source.get("origin_file"),
            "origin_file_index": source.get("origin_file_index"),
            "source_name": source.get("source_name"),
            "source_subreddit": source.get("source_subreddit"),
            "selected_comment_id": source.get("selected_comment_id"),
            "post_title": source.get("post_title"),
            "post_body": source.get("post_body"),
            "selected_comment": (source.get("source_record") or {}).get("selected_comment"),
            "source_record": source.get("source_record"),
            "model_alias": alias,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }
    )
    record.setdefault("labels", {key: None for key in POST_LABEL_KEYS})
    record.setdefault("raw_responses", {})
    record.setdefault("raw_response_details", {})
    record.setdefault("errors", {})


def update_record_status(record: dict[str, Any]) -> None:
    labels = record.setdefault("labels", {})
    vector = [normalize_label_value(labels.get(key)) for key in POST_LABEL_KEYS]
    record["label_vector"] = vector
    record["status"] = "complete" if all(value is not None for value in vector) else "incomplete"
    record["updated_at"] = timestamp()


def run_one_model(
    client: OpenAI,
    sources: list[dict[str, Any]],
    output_path: Path,
    alias: str,
    model: str,
    max_retries: int,
    retry_max_tokens: list[int],
    request_timeout: float,
    force: bool,
) -> None:
    results, duplicate_count = dedupe_results(read_existing_results(output_path))
    if duplicate_count:
        print(f"[{alias}] Deduplicated existing records: {duplicate_count}")
        write_json(output_path, results)

    indexes = result_indexes_by_post_id(results)
    for source_index, source in enumerate(sources, start=1):
        post_id = source["post_id"]
        if post_id in indexes:
            record = results[indexes[post_id]]
            refresh_model_record_metadata(record, source, alias, model)
        else:
            record = make_empty_model_record(source, alias, model)
            indexes[post_id] = len(results)
            results.append(record)

        if is_complete_record(record) and not force:
            continue

        for label in POST_LABELS:
            existing_value = normalize_label_value((record.get("labels") or {}).get(label.key))
            if existing_value is not None and not force:
                continue

            value, raw, error, debug = request_label(
                client=client,
                model=model,
                text=source["post_text"],
                label_name=label.name_en,
                definition=label.definition_en,
                max_retries=max_retries,
                retry_max_tokens=retry_max_tokens,
                request_timeout=request_timeout,
                progress_prefix=f"[{alias}] [{source_index}/{len(sources)}] {post_id} {label.key}",
            )
            record.setdefault("labels", {})[label.key] = value
            record.setdefault("raw_responses", {})[label.key] = raw
            record.setdefault("raw_response_details", {})[label.key] = debug
            if error:
                record.setdefault("errors", {})[label.key] = error
            else:
                record.setdefault("errors", {}).pop(label.key, None)
            update_record_status(record)
            write_json(output_path, results)
            print(
                f"[{alias}] [{source_index}/{len(sources)}] {post_id} "
                f"{label.key}={value} status={record['status']}"
            )

        update_record_status(record)
        write_json(output_path, results)


def load_model_results(output_dir: Path, models: dict[str, str]) -> dict[str, dict[str, dict[str, Any]]]:
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for alias in models:
        path = model_output_path(output_dir, alias)
        results = read_existing_results(path)
        all_results[alias] = {
            str(record.get("post_id")): record
            for record in results
            if record.get("post_id")
        }
    return all_results


def label_vote(values: list[int | None], allow_partial: bool) -> tuple[int | None, str, int, int]:
    valid = [value for value in values if value in {0, 1}]
    ones = sum(value == 1 for value in valid)
    zeros = sum(value == 0 for value in valid)
    if len(valid) == 3:
        return (1 if ones > zeros else 0), "complete", ones, zeros
    if allow_partial and len(valid) == 2 and ones != zeros:
        return (1 if ones > zeros else 0), "partial_2_of_3", ones, zeros
    return None, "incomplete_vote", ones, zeros


def compact_model_annotation(record: dict[str, Any] | None, alias: str, model: str) -> dict[str, Any]:
    if record is None:
        return {
            "model_alias": alias,
            "model": model,
            "status": "missing",
            "labels": {key: None for key in POST_LABEL_KEYS},
            "label_vector": [None for _ in POST_LABEL_KEYS],
            "errors": {},
            "raw_responses": {},
            "raw_response_details": {},
        }
    return {
        "model_alias": alias,
        "model": record.get("model") or model,
        "status": record.get("status"),
        "labels": record.get("labels") or {},
        "label_vector": record.get("label_vector"),
        "errors": record.get("errors") or {},
        "raw_responses": record.get("raw_responses") or {},
        "raw_response_details": record.get("raw_response_details") or {},
        "prompt_version": record.get("prompt_version"),
        "updated_at": record.get("updated_at"),
    }


def build_majority(
    sources: list[dict[str, Any]],
    output_dir: Path,
    models: dict[str, str],
    allow_partial: bool,
    drop_incomplete_aliases: set[str] | None = None,
) -> list[dict[str, Any]]:
    model_results = load_model_results(output_dir, models)
    majority_records: list[dict[str, Any]] = []
    drop_incomplete_aliases = drop_incomplete_aliases or set()
    dropped_records = 0

    for source in sources:
        post_id = source["post_id"]
        model_annotations: dict[str, dict[str, Any]] = {}
        majority_labels: dict[str, int | None] = {}
        vote_details: dict[str, dict[str, Any]] = {}

        should_drop = any(
            not has_complete_labels(model_results.get(alias, {}).get(post_id))
            for alias in drop_incomplete_aliases
        )
        if should_drop:
            dropped_records += 1
            continue

        for alias, model in models.items():
            model_annotations[alias] = compact_model_annotation(
                model_results.get(alias, {}).get(post_id),
                alias,
                model,
            )

        for label_key in POST_LABEL_KEYS:
            votes = {
                alias: normalize_label_value(annotation.get("labels", {}).get(label_key))
                for alias, annotation in model_annotations.items()
            }
            final_label, vote_status, ones, zeros = label_vote(list(votes.values()), allow_partial)
            majority_labels[label_key] = final_label
            vote_details[label_key] = {
                "votes": votes,
                "votes_for_1": ones,
                "votes_for_0": zeros,
                "valid_vote_count": sum(value in {0, 1} for value in votes.values()),
                "final_label": final_label,
                "status": vote_status,
                "agreement": f"{max(ones, zeros)}/{sum(value in {0, 1} for value in votes.values())}",
            }

        majority_vector = [majority_labels[key] for key in POST_LABEL_KEYS]
        status = "complete" if all(value is not None for value in majority_vector) else "incomplete"
        majority_records.append(
            {
                "post_id": post_id,
                "annotation_number": source.get("annotation_number"),
                "combined_record_number": source.get("combined_record_number"),
                "origin_file": source.get("origin_file"),
                "origin_file_index": source.get("origin_file_index"),
                "source_name": source.get("source_name"),
                "source_subreddit": source.get("source_subreddit"),
                "selected_comment_id": source.get("selected_comment_id"),
                "post_title": source.get("post_title"),
                "post_body": source.get("post_body"),
                "selected_comment": (source.get("source_record") or {}).get("selected_comment"),
                "source_record": source.get("source_record"),
                "model_annotations": model_annotations,
                "majority_labels": majority_labels,
                "majority_label_vector": majority_vector,
                "vote_details": vote_details,
                "label_order": list(POST_LABEL_KEYS),
                "status": status,
                "allow_partial_vote": allow_partial,
                "prompt_version": PROMPT_VERSION,
                "updated_at": timestamp(),
            }
        )

    if dropped_records:
        print(
            "Dropped records with incomplete required model labels: "
            f"{dropped_records} ({', '.join(sorted(drop_incomplete_aliases))})"
        )
    return majority_records


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    aliases_to_run = parse_alias_list(args.only_models, models)
    drop_incomplete_aliases = set(parse_alias_list(args.drop_incomplete_models, models)) if args.drop_incomplete_models else set()
    retry_max_tokens = parse_positive_int_list(args.retry_max_tokens)
    if args.max_tokens not in retry_max_tokens:
        retry_max_tokens = [args.max_tokens] + retry_max_tokens
    sources = load_source_records(args.input, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Posts to process: {len(sources)}")
    print("Configured models: " + ", ".join(f"{alias}={model}" for alias, model in models.items()))
    print(f"Retry max_tokens schedule: {retry_max_tokens[:args.max_retries]}")
    print(f"Request timeout: {args.request_timeout}s")
    print(f"Output dir: {args.output_dir}")

    if not args.majority_only:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is required")
        if not args.base_url:
            raise SystemExit("Set BAILIAN_BASE_URL or pass --base-url")
        client = OpenAI(api_key=api_key, base_url=args.base_url)

        for alias in aliases_to_run:
            model = models[alias]
            path = model_output_path(args.output_dir, alias)
            print(f"\n=== Running {alias}: {model} ===")
            run_one_model(
                client=client,
                sources=sources,
                output_path=path,
                alias=alias,
                model=model,
                max_retries=args.max_retries,
                retry_max_tokens=retry_max_tokens,
                request_timeout=args.request_timeout,
                force=args.force,
            )

    majority = build_majority(
        sources=sources,
        output_dir=args.output_dir,
        models=models,
        allow_partial=args.allow_partial_vote,
        drop_incomplete_aliases=drop_incomplete_aliases,
    )
    majority_path = majority_output_path(args.output_dir)
    write_json(majority_path, majority)
    complete = sum(record.get("status") == "complete" for record in majority)
    print(f"\nMajority vote written: {majority_path}")
    print(f"Majority complete records: {complete}/{len(majority)}")
    for alias in models:
        print(f"{alias} output: {model_output_path(args.output_dir, alias)}")


if __name__ == "__main__":
    main()
