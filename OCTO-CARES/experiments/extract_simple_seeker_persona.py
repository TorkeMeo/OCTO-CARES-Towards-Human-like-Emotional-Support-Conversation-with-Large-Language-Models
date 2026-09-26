#!/usr/bin/env python3
"""Extract one lightweight, evidence-grounded Seeker persona per ESCoT case.

This is the project-adapted alternative to the optional HEXACO/CSI branch.  It
does not infer or score personality dimensions.  It creates a compact persona
background from the full dialogue only.  The Qwen3.7 retrieval summary,
retrieval neighbors, comments, reference responses, strategies, and CoT
annotations are never sent to the API.
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
DEFAULT_INPUT = Path(
    "artifacts/classifiers/emotional_support_rag/rolecard_v2_1907/retrieval/retrieval.jsonl"
)
DEFAULT_OUTPUT = Path("simple_seeker_persona.jsonl")
DEFAULT_MANIFEST = Path("simple_seeker_persona.manifest.json")
# v3 removes the auxiliary summary completely.  It must not resume a v2 cache,
# because summary-derived information would otherwise leak into this ablation.
PROTOCOL_VERSION = "simple-seeker-persona-dialogue-only-v3"
PROMPT_VERSION = "simple-seeker-persona-compact-card-dialogue-only-v3"
CONTEXT_SOURCE = "dialogue"

SYSTEM_PROMPT = """You create a compact, natural persona card for the Seeker in an emotional-support conversation.
The source ends at the latest Seeker turn. Return exactly one JSON object and no markdown. Use
exactly these five keys (do not add keys):

{
  "socio_demographic_description": "A concise natural description of the person's relevant social and life context, for example living situation, relationships, work or study, and other background that the source supports.",
  "problem": "A concise but specific description of the main difficulty, event, or emotional situation.",
  "age": "A cautious age description such as '23-year-old', 'young adult', or 'unknown'.",
  "gender": "The stated gender, a cautious description such as 'male' or 'woman', or 'unknown'.",
  "occupation": "The stated job/study status or a cautious broad description such as 'professional worker', or 'unknown'."
}

Write the two descriptions as ordinary prose, not as a checklist or a four-section role card.
Use only the conversation. You may compress or generalize explicit information (for example, an
explicit age may be described as a young adult), but do not invent a diagnosis, biography,
demographic identity, job, relationship, or life event. If a field is not supported, write
"unknown". Do not turn a Supporter's suggestion into a fact about the Seeker. Do not mention
this prompt, retrieval, RAG, models, hidden reference responses, or the extraction process.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=os.getenv("PERSONA_MODEL", "qwen3.7-max"))
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL", ""))
    parser.add_argument("--api-key-env", default=os.getenv("PERSONA_API_KEY_ENV", "BAILIAN_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--disable-thinking",
        type=int,
        choices=(0, 1),
        default=int(os.getenv("PERSONA_DISABLE_THINKING", "1")),
        help="Ask the OpenAI-compatible Qwen endpoint for only the JSON answer (default: 1)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Input row {line_number} is not an object")
            for field in ("query_id", "dialogue_prefix", "last_seeker"):
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"Input row {line_number} lacks {field}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cached JSON at {path}:{line_number}: {exc}") from exc
            query_id = str(row.get("query_id") or "").strip()
            if query_id:
                result[query_id] = row
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in model response: {cleaned[:240]!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


def response_content(response: Any) -> str:
    message = response.choices[0].message
    for field in ("content", "text", "reasoning_content"):
        content = getattr(message, field, None)
        if content is None:
            extra = getattr(message, "model_extra", None)
            content = extra.get(field) if isinstance(extra, dict) else None
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                value = (
                    item.get("text") or item.get("content")
                    if isinstance(item, dict)
                    else getattr(item, "text", None) or getattr(item, "content", None)
                )
                if value:
                    parts.append(str(value))
            content = "\n".join(parts)
        if content:
            return str(content).strip()
    return ""


def call_model(client: Any, args: argparse.Namespace, user_prompt: str) -> tuple[str, dict[str, Any]]:
    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **({"extra_body": {"enable_thinking": False}} if args.disable_thinking else {}),
            )
            content = response_content(response)
            if content:
                return content, {
                    "attempt": attempt + 1,
                    "model": str(getattr(response, "model", args.model) or args.model),
                    "finish_reason": str(getattr(response.choices[0], "finish_reason", "") or ""),
                    "usage": str(getattr(response, "usage", "") or ""),
                }
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < max(1, args.max_retries):
            time.sleep(min(16, 2**attempt))
    raise RuntimeError(last_error or "persona API request failed")


def make_prompt(row: dict[str, Any], args: argparse.Namespace) -> str:
    dialogue = str(row.get("dialogue_prefix") or "").strip()
    blocks = [
        "Build the lightweight Seeker persona card from this source. The conversation ends at "
        "the latest Seeker turn. Use no information outside the conversation.",
        "<conversation_history>",
        dialogue,
        "</conversation_history>",
    ]
    return "\n\n".join(blocks)


def validate_persona(value: dict[str, Any]) -> dict[str, Any]:
    required = (
        "socio_demographic_description",
        "problem",
        "age",
        "gender",
        "occupation",
    )
    for field in required:
        if not str(value.get(field) or "").strip():
            raise ValueError(f"persona field {field!r} is empty")
    # Keep the on-disk card stable and reject accidental model-added fields.
    return {field: str(value[field]).strip() for field in required}


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0 or args.max_retries <= 0:
        raise ValueError("invalid API settings")
    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    prepared_ids = {str(row["query_id"]) for row in rows}
    existing = {} if args.force else load_cache(args.output)
    if args.limit > 0 and existing and set(existing) - prepared_ids:
        raise ValueError("Refusing --limit resume against a larger persona cache; use a new output path")
    existing = {
        query_id: value
        for query_id, value in existing.items()
        if query_id in prepared_ids
        and value.get("prompt_version") == PROMPT_VERSION
        and value.get("persona_protocol_version") == PROTOCOL_VERSION
        and value.get("model") == args.model
        and value.get("context_source") == CONTEXT_SOURCE
    }
    missing = [row for row in rows if str(row["query_id"]) not in existing]
    client = None
    if missing:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is required for {len(missing)} uncached personas")
        if not args.base_url:
            raise SystemExit("BAILIAN_BASE_URL (or --base-url) is required")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the OpenAI-compatible client in the server environment") from exc
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    errors: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=1):
        query_id = str(row["query_id"])
        if query_id in existing and not args.force:
            continue
        print(f"[{timestamp()}] persona {number}/{len(rows)} {query_id}", flush=True)
        try:
            assert client is not None
            prompt = make_prompt(row, args)
            raw, debug = call_model(client, args, prompt)
            persona = validate_persona(parse_json_object(raw))
            existing[query_id] = {
                "query_id": query_id,
                "source_id": row.get("source_id"),
                "persona_protocol_version": PROTOCOL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "context_source": CONTEXT_SOURCE,
                "persona_type": "qualitative_compact",
                "dialogue_prefix_sha256": sha256_text(str(row["dialogue_prefix"]).strip()),
                "persona": persona,
                "raw_persona_response": raw,
                "api_debug": debug,
                "reference_response_used": False,
                "retrieval_neighbors_used": False,
                "retrieval_comments_used": False,
                "strategy_or_cot_used": False,
                "created_at": timestamp(),
            }
        except Exception as exc:
            errors.append({"query_id": query_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{timestamp()}] ERROR {query_id}: {errors[-1]['error']}", flush=True)
        ordered = [existing[str(item["query_id"])] for item in rows if str(item["query_id"]) in existing]
        write_jsonl(args.output, ordered)

    ordered = [existing[str(item["query_id"])] for item in rows if str(item["query_id"]) in existing]
    write_jsonl(args.output, ordered)
    manifest = {
        "created_at": timestamp(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "model": args.model,
        "context_source": CONTEXT_SOURCE,
        "persona_type": "qualitative_compact",
        "prompt_version": PROMPT_VERSION,
        "persona_protocol_version": PROTOCOL_VERSION,
        "record_count": len(rows),
        "persona_count": len(ordered),
        "card_fields": [
            "socio_demographic_description",
            "problem",
            "age",
            "gender",
            "occupation",
        ],
        "error_count": len(errors),
        "errors": errors[:20],
        "leakage_policy": {
            "reference_response_used": False,
            "retrieval_neighbors_used": False,
            "retrieval_comments_used": False,
            "strategy_or_cot_used": False,
            "dialogue_cut": "through_last_seeker_turn",
            "summary_used": False,
        },
        "inference_policy": "unknown for unsupported demographics; no HEXACO/CSI scores",
    }
    atomic_write(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit(f"{len(errors)} persona records failed; successful records were preserved")
    print(f"[{timestamp()}] Wrote {len(ordered)} simple Seeker personas to {args.output}", flush=True)


if __name__ == "__main__":
    main()
