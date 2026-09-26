#!/usr/bin/env python3
"""Create detailed first-person seeker role cards for ESCoT test dialogues.

Only the dialogue prefix ending at the latest seeker turn is sent to Bailian.
The reference response, strategy, and ``cot_data`` are deliberately never
read by the role-card writer.  Results are written as a separate JSONL cache and
can be resumed without touching the original narrative-summary cache.
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "test.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "escot_rolecard_summaries.jsonl"
DEFAULT_MANIFEST = SCRIPT_DIR / "escot_rolecard_summaries.manifest.json"
SUMMARY_PROMPT_VERSION = "escot-first-person-seeker-reddit-narrative-v2"
SUMMARY_STYLE = "first_person_seeker_reddit_narrative"
SUMMARY_PIPELINE = "single_pass_dialogue_to_narrative"

SUMMARY_SYSTEM_PROMPT = """You are writing a detailed first-person account for an emotional-support responder and a memory-retrieval system.
The source is a conversation prefix that ends at the seeker's latest message. Write as if I am naturally telling my own story in a thoughtful Reddit post or a candid message to someone who might understand me. I am the speaker: use I, me, and my throughout, and never describe me as "the seeker," "the user," or "they."

Use conversational, emotionally alive prose rather than a clinical case report, polished therapy note, questionnaire, role-card template, or checklist. Do not use section headings, bullet points, labels, or a fixed four-part structure. Let the account unfold in a natural order, with paragraphs only when they help readability. It should sound like a real person explaining what has been happening, what it has felt like, what has changed, and why they are bringing it up now, while still being detailed enough for a responder to understand the whole situation.

Preserve every material detail supported by the conversation: concrete events and circumstances, people and relationships, timeline and changes, actions already taken, important wording or questions, emotions and their causes, mixed feelings, conflicts, uncertainty, goals, fears, hopes, and the kind of understanding or help I seem to be seeking. Cover the situation/background, feelings, experiences or attempts, and needs/fears/hopes organically within the narrative, not as separate labeled items. Preserve distinctive topics and terms that could match this case to a relevant past experience; do not replace them with vague phrases such as "having problems" or "feeling bad." Do not compress away repetition when it shows escalation, hesitation, or an important emotional shift.

Treat my own seeker statements as the primary facts. Include supporter turns when they materially clarify what happened or what I was responding to, but describe them as something I was asked or told rather than silently converting them into facts about me. Do not turn a supporter's guess, diagnosis, or suggestion into my belief unless I explicitly confirmed it.

Stay faithful to the source and preserve uncertainty when something is unclear (for example, "I am not sure"), rather than guessing. Do not give advice, diagnose anyone, prescribe a support strategy, evaluate the conversation, or write the responder's reply. Do not invent facts, personal history, or emotions. Do not mention a hidden reference answer, this instruction, role cards, RAG, or the summarization process."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=os.getenv("BAILIAN_MODEL", "qwen3.7-max"))
    parser.add_argument(
        "--model-path-provenance",
        default="",
        help="Optional local checkpoint path recorded in cache/manifest provenance.",
    )
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens; the role card is intentionally more detailed than the legacy summary.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            key = str(row.get("query_id") or "").strip()
            if key and row.get("summary"):
                rows[key] = row
    return rows


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    raise ValueError(f"Expected a JSON list in {path}")


def dialogue_prefix(record: dict[str, Any]) -> tuple[str, str, int]:
    original = record.get("original_data")
    if not isinstance(original, dict):
        raise ValueError(f"Record {record.get('id')} has no original_data object")
    dialog = original.get("dialog")
    if not isinstance(dialog, list):
        raise ValueError(f"Record {record.get('id')} has no dialog list")

    seeker_indices = []
    clean_turns: list[tuple[str, str]] = []
    for index, turn in enumerate(dialog):
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content or speaker not in {"seeker", "supporter"}:
            continue
        clean_turns.append((speaker, content))
        if speaker == "seeker":
            seeker_indices.append(len(clean_turns) - 1)
    if not seeker_indices:
        raise ValueError(f"Record {record.get('id')} contains no seeker turn")
    last_index = seeker_indices[-1]
    selected = clean_turns[: last_index + 1]
    lines = [f"{speaker.title()}: {content}" for speaker, content in selected]
    return "\n".join(lines), selected[-1][1], len(selected)


def make_user_prompt(prefix: str) -> str:
    return (
        "Build a detailed first-person seeker role card from the following emotional-support "
        "conversation prefix. It ends at the current seeker's latest message. Describe the "
        "seeker's own experience using I/me/my and preserve all material details.\n\n<conversation>\n"
        + prefix
        + "\n</conversation>"
    )


def response_content(response: Any) -> str:
    message = response.choices[0].message
    for field in ("content", "text", "reasoning_content"):
        content = getattr(message, field, None)
        if content is None:
            extra = getattr(message, "model_extra", None)
            content = extra.get(field) if isinstance(extra, dict) else None
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text") or item.get("content")
                else:
                    value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
            content = "\n".join(parts)
        if content:
            return str(content).strip()
    return ""


def clean_summary(text: str) -> str:
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = re.sub(r"^(?:summary|概述|role\s*card|seeker\s*role\s*card)\s*[:：]?\s*", "", text.strip(), flags=re.IGNORECASE)
    # Keep section breaks so the role-card structure remains visible to the
    # retrieval encoder and to anyone inspecting the JSONL cache.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def call_summary(client: Any, model: str, prefix: str, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": make_user_prompt(prefix)},
                ],
            )
            content = clean_summary(response_content(response))
            if content:
                usage = getattr(response, "usage", None)
                return content, {
                    "attempt": attempt + 1,
                    "model": str(getattr(response, "model", model) or model),
                    "finish_reason": str(getattr(response.choices[0], "finish_reason", "") or ""),
                    "usage": str(usage) if usage is not None else "",
                }
            last_error = "empty response"
        except Exception as exc:  # API clients expose several exception classes
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < max(1, args.max_retries):
            time.sleep(min(16, 2**attempt))
    raise RuntimeError(last_error or "summary request failed")


def main() -> None:
    args = parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count)")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    all_records = records_from_payload(read_json(args.input), args.input)
    selected_records = all_records[: args.limit] if args.limit > 0 else all_records
    records = [
        record
        for index, record in enumerate(selected_records)
        if index % args.worker_count == args.worker_index
    ]

    prior_rows = load_jsonl(args.output)
    if args.limit > 0 and args.output.is_file():
        existing_ids = set(prior_rows)
        selected_ids = {f"escot_{record.get('id')}" for record in records}
        if existing_ids - selected_ids:
            raise ValueError(
                "Refusing --limit resume because it would truncate a larger summary cache. "
                "Use a separate --output path for smoke tests."
            )
    existing = {} if args.force else {
        key: row
        for key, row in prior_rows.items()
        if row.get("summary_model") == args.model
        and row.get("summary_prompt_version") == SUMMARY_PROMPT_VERSION
        and (
            not args.model_path_provenance
            or row.get("summary_model_path") == args.model_path_provenance
        )
    }
    prepared: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("id") or "").strip()
        if not source_id:
            raise ValueError("Every ESCoT record needs an id")
        prefix, last_seeker, turn_count = dialogue_prefix(record)
        prepared.append(
            {
                "query_id": f"escot_{source_id}",
                "source_id": source_id,
                "dialogue_prefix": prefix,
                "last_seeker": last_seeker,
                "turn_count": turn_count,
            }
        )
    prepared_by_id = {row["query_id"]: row for row in prepared}
    existing = {
        key: row
        for key, row in existing.items()
        if key in prepared_by_id
        and row.get("dialogue_prefix") == prepared_by_id[key]["dialogue_prefix"]
    }

    missing = [row for row in prepared if row["query_id"] not in existing]
    client = None
    if missing:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is required for {len(missing)} uncached summaries")
        if not args.base_url:
            raise SystemExit("BAILIAN_BASE_URL (or --base-url) is required")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the OpenAI-compatible Bailian client in the server environment") from exc
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    errors: list[dict[str, Any]] = []
    for number, row in enumerate(prepared, start=1):
        key = row["query_id"]
        if key in existing and existing[key].get("summary") and not args.force:
            continue
        print(f"[{timestamp()}] summary {number}/{len(prepared)} {key}", flush=True)
        try:
            assert client is not None
            summary, debug = call_summary(client, args.model, row["dialogue_prefix"], args)
            existing[key] = {
                **row,
                "summary": summary,
                "summary_model": args.model,
                "summary_model_path": args.model_path_provenance or None,
                "summary_prompt_version": SUMMARY_PROMPT_VERSION,
                "summary_style": SUMMARY_STYLE,
                "summary_pipeline": SUMMARY_PIPELINE,
                "summary_debug": debug,
                "created_at": timestamp(),
            }
            if isinstance(debug, dict) and isinstance(debug.get("evidence_claims"), list):
                existing[key]["summary_evidence"] = debug["evidence_claims"]
        except Exception as exc:
            error = {"query_id": key, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            print(f"[{timestamp()}] ERROR {key}: {error['error']}", flush=True)
        ordered = [existing[item["query_id"]] for item in prepared if item["query_id"] in existing]
        write_jsonl(args.output, ordered)

    ordered = [existing[item["query_id"]] for item in prepared if item["query_id"] in existing]
    write_jsonl(args.output, ordered)
    manifest = {
        "created_at": timestamp(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "model": args.model,
        "model_path": args.model_path_provenance or None,
        "record_count": len(prepared),
        "selected_record_count_before_sharding": len(selected_records),
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "summary_count": len(ordered),
        "error_count": len(errors),
        "errors": errors[:20],
        "leakage_policy": {
            "dialogue_cut": "through_last_seeker_turn",
            "reference_response_read": False,
            "strategy_read": False,
            "cot_data_read": False,
        },
        "summary_style": SUMMARY_STYLE,
        "summary_pipeline": SUMMARY_PIPELINE,
        "summary_format": "natural first-person Reddit-style narrative without fixed sections",
        "prompt_version": SUMMARY_PROMPT_VERSION,
    }
    write_json(args.manifest, manifest)
    if errors:
        raise SystemExit(f"{len(errors)} summaries failed; cached successful rows were preserved")
    print(f"[{timestamp()}] Wrote {len(ordered)} summaries to {args.output}", flush=True)


if __name__ == "__main__":
    main()
