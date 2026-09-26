#!/usr/bin/env python3
"""Generate the qualitative-persona condition with local Qwen3-8B.

Both the default and optional paired mode use the full dialogue.  The
``simple_persona`` condition additionally receives the compact qualitative
Seeker card; ``dialogue_only`` is available only as an explicit ablation because
the project's existing ``pure_qwen3`` output already is that baseline.  The
Qwen3.7 retrieval summary and all retrieved material are excluded from both
prompts.  The same per-query seed is used for paired conditions, and worker
files are resumable and merged in source order.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from generate_support_replies_lively_topk import generate_one, load_model
from extract_simple_seeker_persona import PROTOCOL_VERSION as PERSONA_PROTOCOL_VERSION
from extract_simple_seeker_persona import sha256_text


DEFAULT_RETRIEVAL = Path(
    "artifacts/classifiers/emotional_support_rag/rolecard_v2_1907/retrieval/retrieval.jsonl"
)
ALL_CONDITIONS = ("dialogue_only", "simple_persona")
PROTOCOL_VERSION = "escot-dialogue-plus-simple-persona-ablation-no-rag-v3"
PROMPT_VERSION = "simple-persona-ablation-responder-next-message-v3"
AUDIT_GENERATION_CONTEXT_SOURCE: str | None = None
AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--personas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="responses")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--local-files-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=[0, 1], default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--conditions",
        default=os.getenv("PERSONA_CONDITIONS", "simple_persona"),
        help="Comma-separated subset of dialogue_only,simple_persona; default: simple_persona",
    )
    return parser.parse_args()


def selected_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    raw = args.conditions
    values = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError("--conditions must contain at least one condition")
    if any(item not in ALL_CONDITIONS for item in values):
        raise ValueError(
            f"Unknown persona condition in {values}; choose from {','.join(ALL_CONDITIONS)}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate persona conditions: {values}")
    return values


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            for field in ("query_id", "dialogue_prefix", "last_seeker"):
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"{path}:{line_number} lacks {field}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def load_personas(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            query_id = str(row.get("query_id") or "").strip()
            persona = row.get("persona")
            if query_id and isinstance(persona, dict):
                result[query_id] = row
    return result


def validate_persona(row: dict[str, Any]) -> None:
    if row.get("persona_protocol_version") != PERSONA_PROTOCOL_VERSION:
        raise ValueError(f"{row.get('query_id')}: unsupported persona protocol")
    persona = row.get("persona")
    required = (
        "socio_demographic_description",
        "problem",
        "age",
        "gender",
        "occupation",
    )
    if not isinstance(persona, dict) or any(not str(persona.get(field) or "").strip() for field in required):
        raise ValueError(f"{row.get('query_id')}: compact persona card is incomplete")


def make_prompt(
    row: dict[str, Any], persona_row: dict[str, Any] | None = None, condition: str = "simple_persona"
) -> str:
    dialogue = str(row["dialogue_prefix"]).strip()
    latest = str(row["last_seeker"]).strip()
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    blocks = [
        "Act as an ordinary person, not an AI agent or universal advice bot. You are the "
        "Responder in the conversation below. Reply in your own voice; any tone, length, "
        "stance, question, suggestion, disagreement, or lack of resolution is allowed. Do "
        "not force comfort, therapy language, a quality checklist, or a fixed reply shape.",
        "<conversation_history>",
        dialogue,
        "</conversation_history>",
    ]
    if condition == "simple_persona":
        if not isinstance(persona_row, dict):
            raise ValueError("simple_persona condition requires a persona row")
        persona = persona_row["persona"]
        card = "\n".join(
            [
                "Socio-demographic description: "
                + str(persona.get("socio_demographic_description") or "unknown"),
                "Problem: " + str(persona.get("problem") or "unknown"),
                f"Age: {persona.get('age', 'unknown')}",
                f"Gender: {persona.get('gender', 'unknown')}",
                f"Occupation: {persona.get('occupation', 'unknown')}",
            ]
        )
        blocks.extend(
            [
                "<seeker_persona>",
                card,
                "</seeker_persona>",
                "The persona describes the Seeker, not the Responder. Use it only as background "
                "to make the reply specific. Do not recite it, mention this profile or its "
                "extraction, or invent a biography from it. Treat it as internal context and "
                "output only the reply.",
            ]
        )
    blocks.extend(
        [
            "<latest_seeker_turn>",
            latest,
            "</latest_seeker_turn>",
            "Write only the single next Responder message. Answer the latest turn first while "
            "using earlier dialogue for context. Stop when you have said what you mean.",
        ]
    )
    return "\n\n".join(blocks)


def output_path(args: argparse.Namespace, condition: str) -> Path:
    return args.output_dir / f"{args.output_prefix}.{condition}.worker{args.worker_index}.jsonl"


def worker_output_path(args: argparse.Namespace, condition: str, worker: int) -> Path:
    return args.output_dir / f"{args.output_prefix}.{condition}.worker{worker}.jsonl"


def final_path(args: argparse.Namespace, condition: str) -> Path:
    return args.output_dir / f"{args.output_prefix}.{condition}.jsonl"


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_id") and row.get("response") and not row.get("error"):
                result[str(row["query_id"])] = row
    return result


def seed_for(row: dict[str, Any], index: int, base: int) -> int:
    source_id = str(row.get("source_id") or "")
    return base + int(source_id) if source_id.isdigit() else base + index * 1009


def can_reuse(row: dict[str, Any], prompt_hash: str, seed: int, args: argparse.Namespace) -> bool:
    return (
        row.get("prompt_sha256") == prompt_hash
        and row.get("generation_protocol_version") == PROTOCOL_VERSION
        and row.get("model_path") == args.model_path
        and row.get("seed") == seed
        and row.get("max_input_tokens") == args.max_input_tokens
        and row.get("max_new_tokens") == args.max_new_tokens
        and row.get("temperature") == args.temperature
        and row.get("top_p") == args.top_p
        and row.get("bf16") == bool(args.bf16)
    )


def merge(args: argparse.Namespace, rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> None:
    expected = [str(row["query_id"]) for row in rows]
    for condition in conditions:
        by_id: dict[str, dict[str, Any]] = {}
        for worker in range(args.worker_count):
            path = worker_output_path(args, condition, worker)
            if not path.is_file():
                raise FileNotFoundError(f"Missing worker output: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("condition") != condition:
                        raise ValueError(f"Wrong condition in {path}:{line_number}")
                    by_id[str(row["query_id"])] = row
        missing = set(expected) - set(by_id)
        failed = [
            query_id for query_id, row in by_id.items() if row.get("error") or not row.get("response")
        ]
        if missing or failed:
            raise RuntimeError(
                f"{condition} generation incomplete: missing={sorted(missing)[:5]} "
                f"failed={failed[:5]}"
            )
        if set(by_id) - set(expected):
            raise ValueError(f"{condition} worker outputs contain IDs outside the selected input")
        ordered = [by_id[query_id] for query_id in expected]
        write_jsonl(final_path(args, condition), ordered)
        print(
            f"[{timestamp()}] merged {len(ordered)} {condition} replies -> "
            f"{final_path(args, condition)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    conditions = selected_conditions(args)
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count)")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid sampling settings")
    rows = read_jsonl(args.retrieval)
    if args.limit > 0:
        rows = rows[: args.limit]
    personas: dict[str, dict[str, Any]] = {}
    if "simple_persona" in conditions:
        personas = load_personas(args.personas)
        for row in rows:
            query_id = str(row["query_id"])
            persona_row = personas.get(query_id)
            if persona_row is None:
                raise ValueError(f"{query_id}: persona is missing")
            validate_persona(persona_row)
            if persona_row.get("dialogue_prefix_sha256") != sha256_text(str(row["dialogue_prefix"]).strip()):
                raise ValueError(f"{query_id}: persona/dialogue hash mismatch")
    if args.merge:
        merge(args, rows, conditions)
        return

    shard = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {condition: output_path(args, condition) for condition in conditions}
    if args.force:
        for path in paths.values():
            path.unlink(missing_ok=True)
    for path in paths.values():
        path.touch(exist_ok=True)
    completed = {condition: load_completed(path) for condition, path in paths.items()}
    print(
        f"[{timestamp()}] dialogue/persona worker={args.worker_index}/{args.worker_count} "
        f"queries={len(shard)} model={args.model_path} summary_used=False rag_used=False",
        flush=True,
    )
    if not shard:
        return
    torch, tokenizer, model = load_model(args)
    for local_index, row in enumerate(shard):
        query_id = str(row["query_id"])
        persona_row = personas.get(query_id)
        query_seed = seed_for(row, local_index, args.seed)
        for condition in conditions:
            prompt = make_prompt(
                row,
                persona_row if condition == "simple_persona" else None,
                condition,
            )
            prompt_hash = sha256_text(prompt)
            if (
                query_id in completed[condition]
                and not args.force
                and can_reuse(completed[condition][query_id], prompt_hash, query_seed, args)
            ):
                print(f"[{timestamp()}] reuse {condition} {query_id}", flush=True)
                continue
            try:
                response, input_count, untruncated_count, output_count, finish_reason, input_truncated = generate_one(
                    torch, tokenizer, model, prompt, args, query_seed
                )
                has_persona = condition == "simple_persona"
                payload = {
                    "query_id": query_id,
                    "source_id": row.get("source_id"),
                    "condition": condition,
                    "response": response,
                    # Retain these source fields for audit, but explicitly mark
                    # that summary was not put in either prompt.
                    "input_summary": row.get("summary"),
                    "dialogue_prefix": row.get("dialogue_prefix"),
                    "last_seeker": row.get("last_seeker"),
                    "generation_context_source": AUDIT_GENERATION_CONTEXT_SOURCE or (
                        "dialogue_prefix_plus_simple_persona" if has_persona else "dialogue_prefix"
                    ),
                    "generation_protocol_scope": "project_adaptation_single_next_reply_no_rag",
                    "dialogue_used_in_generation_prompt": (
                        AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT
                        if AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT is not None
                        else True
                    ),
                    "summary_used_in_generation_prompt": (
                        AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT
                        if AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT is not None
                        else False
                    ),
                    "last_seeker_used_in_generation_prompt": (
                        AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT
                        if AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT is not None
                        else True
                    ),
                    "persona_used_in_generation_prompt": has_persona,
                    "simple_persona_used_in_generation_prompt": has_persona,
                    "persona_type": "qualitative_compact" if has_persona else None,
                    "simple_persona": persona_row.get("persona") if has_persona else None,
                    "retrieval_used_in_prompt": False,
                    "rag_used_in_prompt": False,
                    "persona_profile_sha256": (
                        sha256_text(json.dumps(persona_row["persona"], ensure_ascii=False, sort_keys=True))
                        if has_persona
                        else None
                    ),
                    "persona_protocol_version": persona_row.get("persona_protocol_version") if has_persona else None,
                    "persona_prompt_version": persona_row.get("prompt_version") if has_persona else None,
                    "persona_model": persona_row.get("model") if has_persona else None,
                    "persona_context_source": persona_row.get("context_source") if has_persona else None,
                    "persona_card_fields": (
                        [
                            "socio_demographic_description",
                            "problem",
                            "age",
                            "gender",
                            "occupation",
                        ]
                        if has_persona
                        else []
                    ),
                    "prompt_version": PROMPT_VERSION,
                    "generation_prompt": prompt,
                    "prompt_sha256": prompt_hash,
                    "input_tokens": input_count,
                    "input_tokens_before_truncation": untruncated_count,
                    "input_truncated": input_truncated,
                    "output_tokens": output_count,
                    "finish_reason": finish_reason,
                    "seed": query_seed,
                    "paired_seed": query_seed,
                    "model_path": args.model_path,
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "bf16": bool(args.bf16),
                    "generation_protocol_version": PROTOCOL_VERSION,
                    "reference_response_used": False,
                    "strategy_or_cot_used": False,
                    "created_at": timestamp(),
                }
                if input_truncated:
                    print(
                        f"[{timestamp()}] input truncated {condition} {query_id}: "
                        f"{untruncated_count}->{input_count}",
                        flush=True,
                    )
            except Exception as exc:
                payload = {
                    "query_id": query_id,
                    "source_id": row.get("source_id"),
                    "condition": condition,
                    "error": f"{type(exc).__name__}: {exc}",
                    "prompt_sha256": prompt_hash,
                    "seed": query_seed,
                    "paired_seed": query_seed,
                    "generation_protocol_version": PROTOCOL_VERSION,
                    "created_at": timestamp(),
                }
                print(f"[{timestamp()}] ERROR {condition} {query_id}: {payload['error']}", flush=True)
            append_jsonl(paths[condition], payload)
            if payload.get("response") and not payload.get("error"):
                completed[condition][query_id] = payload
    print(f"[{timestamp()}] dialogue/persona worker={args.worker_index} complete", flush=True)


if __name__ == "__main__":
    main()
