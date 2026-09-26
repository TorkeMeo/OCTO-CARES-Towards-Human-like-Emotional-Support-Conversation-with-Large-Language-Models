#!/usr/bin/env python3
"""Extract compact Seeker personas with four local Qwen3 workers.

Only ``dialogue_prefix`` is placed in the prompt.  The summary, retrieved
posts/comments, reference response, strategy, and CoT are excluded.  Worker
files are append-only and resumable; the launcher merges them in input order.
No API client or API credential is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from generate_support_replies_lively_topk import generate_one, load_model


PROTOCOL_VERSION = "simple-seeker-persona-dialogue-only-v3"
PROMPT_VERSION = "simple-seeker-persona-compact-card-dialogue-only-local-qwen3-v1"
CONTEXT_SOURCE = "dialogue"
# Wrapper modules can override this so spawned workers execute the wrapper
# again and inherit its protocol, prompt, and context customizations.
WORKER_ENTRYPOINT = Path(__file__)
REQUIRED_FIELDS = (
    "socio_demographic_description",
    "problem",
    "age",
    "gender",
    "occupation",
)

INSTRUCTION = """Create a compact, evidence-grounded persona card for the Seeker in the conversation below.
The conversation ends at the latest Seeker turn. Return exactly one JSON object and no markdown,
using exactly these five keys:

{
  "socio_demographic_description": "concise natural description of supported social and life context",
  "problem": "concise but specific description of the main difficulty or emotional situation",
  "age": "supported age description or unknown",
  "gender": "supported gender description or unknown",
  "occupation": "supported work or study description or unknown"
}

Use only the conversation. Do not invent a diagnosis, biography, demographic identity, job,
relationship, or event. Do not turn the Supporter's suggestion into a fact about the Seeker.
Write unknown whenever a field is unsupported. Output only the JSON object."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="simple_seeker_persona")
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--gpu-ids", default=os.getenv("GPU_LIST", "4-7"))
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bf16", type=int, choices=(0, 1), default=1)
    parser.add_argument("--local-files-only", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=(0, 1), default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_gpu_ids(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw or "").replace(";", ",").replace(" ", "").split(","):
        if not item:
            continue
        if item.count("-") == 1 and all(part.isdigit() for part in item.split("-", 1)):
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            values.extend(str(number) for number in range(start, end + step, step))
        else:
            values.append(item)
    if len(values) != len(set(values)) or not values:
        raise ValueError(f"GPU IDs must be distinct: {raw!r}")
    return values


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            dialogue = str(row.get("dialogue_prefix") or "").strip()
            if not query_id or query_id in seen or not dialogue:
                raise ValueError(f"Invalid input row at {path}:{line_number}")
            seen.add(query_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows[:limit] if limit > 0 else rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def worker_path(args: argparse.Namespace, worker: int) -> Path:
    return args.output_dir / f"{args.output_prefix}.worker{worker}.jsonl"


def final_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"{args.output_prefix}.jsonl"


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            persona = row.get("persona")
            if (
                query_id
                and row.get("persona_protocol_version") == PROTOCOL_VERSION
                and row.get("prompt_version") == PROMPT_VERSION
                and isinstance(persona, dict)
                and all(str(persona.get(field) or "").strip() for field in REQUIRED_FIELDS)
            ):
                completed[query_id] = row
    return completed


def parse_persona(text: str) -> dict[str, str]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in output: {cleaned[:240]!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Persona output is not an object")
    persona = {field: str(value.get(field) or "").strip() for field in REQUIRED_FIELDS}
    if any(not persona[field] for field in REQUIRED_FIELDS):
        raise ValueError(f"Incomplete persona fields: {persona}")
    return persona


def prompt_for(row: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            INSTRUCTION,
            "<conversation_history>",
            str(row["dialogue_prefix"]).strip(),
            "</conversation_history>",
        ]
    )


def generation_args(
    args: argparse.Namespace, *, attempt: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        max_input_tokens=args.max_input_tokens,
        # A small minority of cases may spend the first budget without closing
        # the JSON object.  Later attempts get more room and mild sampling so
        # deterministic malformed output does not repeat forever.
        max_new_tokens=args.max_new_tokens * (attempt + 1),
        temperature=0.0 if attempt == 0 else 0.2,
        top_p=1.0 if attempt == 0 else 0.9,
        bf16=args.bf16,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )


def run_worker(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    assert args.worker_index is not None
    path = worker_path(args, args.worker_index)
    path.touch(exist_ok=True)
    completed = load_completed(path)
    shard = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    pending = [row for row in shard if str(row["query_id"]) not in completed]
    print(
        f"[{timestamp()}] local persona worker={args.worker_index}/{args.worker_count} "
        f"rows={len(shard)} pending={len(pending)} API=false",
        flush=True,
    )
    if not pending:
        return
    model_args = generation_args(args)
    torch, tokenizer, model = load_model(model_args)
    for local_index, row in enumerate(pending):
        query_id = str(row["query_id"])
        base_prompt = prompt_for(row)
        base_seed = args.seed + args.worker_index * 100_003 + local_index * 1009
        attempts: list[dict[str, Any]] = []
        try:
            persona = None
            for attempt in range(3):
                local_args = generation_args(args, attempt=attempt)
                seed = base_seed + attempt * 1_000_003
                prompt = base_prompt
                if attempt:
                    prompt += (
                        "\n\nYour previous attempt did not finish valid JSON. Keep each value "
                        "under 30 words. Include all five keys, close every quote and the final "
                        "brace, and emit no reasoning or text outside the JSON object."
                    )
                try:
                    (
                        raw,
                        input_tokens,
                        before_tokens,
                        output_tokens,
                        finish_reason,
                        truncated,
                    ) = generate_one(torch, tokenizer, model, prompt, local_args, seed)
                    persona = parse_persona(raw)
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "seed": seed,
                            "max_new_tokens": local_args.max_new_tokens,
                            "finish_reason": finish_reason,
                            "output_tokens": output_tokens,
                            "valid_json": True,
                        }
                    )
                    break
                except Exception as attempt_exc:
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "seed": seed,
                            "max_new_tokens": local_args.max_new_tokens,
                            "valid_json": False,
                            "error": f"{type(attempt_exc).__name__}: {attempt_exc}",
                        }
                    )
                    if attempt == 2:
                        raise
                    print(
                        f"[{timestamp()}] retry persona {query_id} attempt={attempt + 2}",
                        flush=True,
                    )
            assert persona is not None
            payload = {
                "query_id": query_id,
                "source_id": row.get("source_id"),
                "persona_protocol_version": PROTOCOL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "model": args.model_path,
                "context_source": CONTEXT_SOURCE,
                "persona_type": "qualitative_compact",
                "dialogue_prefix_sha256": sha256_text(str(row["dialogue_prefix"]).strip()),
                "persona": persona,
                "raw_persona_response": raw,
                "prompt_sha256": sha256_text(prompt),
                "extraction_attempts": attempts,
                "input_tokens": input_tokens,
                "input_tokens_before_truncation": before_tokens,
                "input_truncated": truncated,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "seed": seed,
                "reference_response_used": False,
                "summary_used": False,
                "retrieval_neighbors_used": False,
                "retrieval_comments_used": False,
                "strategy_or_cot_used": False,
                "created_at": timestamp(),
            }
        except Exception as exc:
            payload = {
                "query_id": query_id,
                "source_id": row.get("source_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "extraction_attempts": attempts,
                "persona_protocol_version": PROTOCOL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "created_at": timestamp(),
            }
            print(f"[{timestamp()}] ERROR persona {query_id}: {payload['error']}", flush=True)
        append_jsonl(path, payload)
        if payload.get("persona") and not payload.get("error"):
            completed[query_id] = payload
            print(f"[{timestamp()}] persona {query_id} complete", flush=True)


def merge(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for worker in range(args.worker_count):
        path = worker_path(args, worker)
        if not path.is_file():
            raise FileNotFoundError(f"Missing persona worker output: {path}")
        by_id.update(load_completed(path))
    expected = [str(row["query_id"]) for row in rows]
    missing = [query_id for query_id in expected if query_id not in by_id]
    if missing:
        raise RuntimeError(f"Local persona extraction incomplete: missing={missing[:10]}")
    ordered = [by_id[query_id] for query_id in expected]
    atomic_write_jsonl(final_path(args), ordered)
    manifest = {
        "created_at": timestamp(),
        "input": str(args.input.resolve()),
        "output": str(final_path(args).resolve()),
        "record_count": len(ordered),
        "model_path": args.model_path,
        "persona_protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "context_source": CONTEXT_SOURCE,
        "api_used": False,
        "summary_used": False,
        "retrieval_used": False,
        "worker_count": args.worker_count,
    }
    (args.output_dir / f"{args.output_prefix}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{timestamp()}] merged {len(ordered)} local personas -> {final_path(args)}", flush=True)


def launch(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) != args.worker_count:
        raise ValueError(f"Expected {args.worker_count} GPU IDs, got {gpu_ids}")
    processes: list[tuple[int, str, subprocess.Popen[Any], Any, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for worker, gpu in enumerate(gpu_ids):
        out = (args.output_dir / f"persona_worker{worker}.out").open("w", encoding="utf-8")
        err = (args.output_dir / f"persona_worker{worker}.err").open("w", encoding="utf-8")
        command = [
            sys.executable, str(WORKER_ENTRYPOINT),
            "--input", str(args.input),
            "--output-dir", str(args.output_dir),
            "--output-prefix", args.output_prefix,
            "--model-path", args.model_path,
            "--worker-count", str(args.worker_count),
            "--worker-index", str(worker),
            "--max-input-tokens", str(args.max_input_tokens),
            "--max-new-tokens", str(args.max_new_tokens),
            "--seed", str(args.seed),
            "--bf16", str(args.bf16),
            "--local-files-only", str(args.local_files_only),
            "--trust-remote-code", str(args.trust_remote_code),
            "--limit", str(args.limit),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(command, env=environment, stdout=out, stderr=err)
        processes.append((worker, gpu, process, out, err))
    failures: list[str] = []
    for worker, gpu, process, out, err in processes:
        code = process.wait()
        out.close()
        err.close()
        if code:
            failures.append(
                f"worker={worker} gpu={gpu} exit={code} stderr={args.output_dir / f'persona_worker{worker}.err'}"
            )
    if failures:
        raise RuntimeError("; ".join(failures))
    merge(args, rows)


def main() -> None:
    args = parse_args()
    if args.worker_count <= 0 or args.limit < 0:
        raise ValueError("Invalid worker count or limit")
    rows = read_rows(args.input, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args, rows)
    elif args.worker_index is not None:
        if not 0 <= args.worker_index < args.worker_count:
            raise ValueError("worker-index must lie in [0, worker-count)")
        run_worker(args, rows)
    else:
        launch(args, rows)


if __name__ == "__main__":
    main()
