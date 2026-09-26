#!/usr/bin/env python3
"""Run the lively four-condition Qwen3-8B replies for one retrieval depth.

This driver deliberately does not summarize, extract vectors, or call Bailian.
It launches three or four independent instances of the lively generation engine and
binds one physical GPU to each instance.  Worker JSONL files are resumable;
after all workers finish they are validated and merged into four final files.
Invoke it once for each requested top-k (1, 3, or 5); the companion shell
launcher does that automatically.  ``dialogue_plus_summary`` is an additive
context ablation and writes its own protocol metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE = SCRIPT_DIR / "generate_support_replies_lively_topk.py"
GENERATION_PROTOCOL_VERSION = "escot-dialogue-friend-state-topk-v7"
SUMMARY_ONLY_GENERATION_PROTOCOL_VERSION = "escot-summary-only-friend-state-topk-v1"
SUMMARY_PLUS_LATEST_GENERATION_PROTOCOL_VERSION = "escot-summary-plus-latest-experience-topk-v8"
DIALOGUE_PLUS_SUMMARY_GENERATION_PROTOCOL_VERSION = "escot-dialogue-plus-rolecard-summary-topk-v1"
TOP_K_CHOICES = (1, 3, 5)
CONDITIONS = (
    "pure_qwen3",
    "attention_post",
    "attention_post_comment",
    "attention_post_random_comment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="responses")
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--gpu-ids", default=os.getenv("GPU_LIST", "4-7"))
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--local-files-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--context-source",
        choices=("dialogue", "dialogue_plus_summary", "summary_only", "summary_plus_latest"),
        default="dialogue",
        help="Context exposed to the generation model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=TOP_K_CHOICES,
        default=1,
        help="Number of ranked RAG neighbors exposed to the prompt.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip model workers and merge existing worker JSONL files.",
    )
    parser.add_argument("--engine", type=Path, default=ENGINE)
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_gpu_ids(raw: str) -> list[str]:
    cleaned = str(raw or "").replace(";", ",").replace(" ", "")
    values: list[str] = []
    for item in cleaned.split(","):
        if not item:
            continue
        if item.count("-") == 1 and all(part.isdigit() for part in item.split("-", 1)):
            parts = item.split("-", 1)
            start, end = (int(part) for part in parts)
            step = 1 if end >= start else -1
            values.extend(str(number) for number in range(start, end + step, step))
        elif item.isdigit():
            values.append(item)
        else:
            # UUIDs and MIG identifiers can be passed through unchanged.
            values.append(item)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"GPU list must contain distinct entries: {raw!r}")
    return values


def read_retrieval_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing retrieval file: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid retrieval JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not str(row.get("query_id") or "").strip():
                raise ValueError(f"Retrieval row {line_number} lacks query_id")
    if count == 0:
        raise ValueError(f"Retrieval file is empty: {path}")
    return count


def engine_args(args: argparse.Namespace, *, worker_index: int | None = None, merge: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(args.engine),
        "--retrieval",
        str(args.retrieval),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        args.output_prefix,
        "--model-path",
        args.model_path,
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--bf16",
        str(args.bf16),
        "--local-files-only",
        str(args.local_files_only),
        "--trust-remote-code",
        str(args.trust_remote_code),
        "--worker-count",
        str(args.worker_count),
        "--top-k",
        str(args.top_k),
        "--context-source",
        args.context_source,
    ]
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.force:
        command.append("--force")
    if merge:
        command.append("--merge")
    else:
        if worker_index is None:
            raise ValueError("worker_index is required for a worker command")
        command.extend(["--worker-index", str(worker_index)])
    return command


def merge(args: argparse.Namespace, expected_count: int) -> None:
    command = engine_args(args, merge=True)
    print(f"[{timestamp()}] Merging {expected_count} retrieval rows", flush=True)
    subprocess.run(command, check=True)


def run_workers(args: argparse.Namespace, expected_count: int, gpu_ids: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, str, subprocess.Popen[Any], Any, Any]] = []
    try:
        for worker_index, gpu in enumerate(gpu_ids):
            stdout_path = args.output_dir / f"worker{worker_index}.out"
            stderr_path = args.output_dir / f"worker{worker_index}.err"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            command = engine_args(args, worker_index=worker_index)
            print(
                f"[{timestamp()}] Starting generation worker={worker_index} gpu={gpu} "
                f"rows~{(expected_count + len(gpu_ids) - 1) // len(gpu_ids)}",
                flush=True,
            )
            try:
                process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
            except Exception:
                stdout.close()
                stderr.close()
                raise
            processes.append((worker_index, gpu, process, stdout, stderr))
    except Exception:
        for _, _, process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process, stdout, stderr in processes:
            process.wait()
            stdout.close()
            stderr.close()
        raise

    failed: list[tuple[int, str, int, Path]] = []
    try:
        for worker_index, gpu, process, stdout, stderr in processes:
            return_code = process.wait()
            stdout.close()
            stderr.close()
            if return_code != 0:
                failed.append(
                    (
                        worker_index,
                        gpu,
                        return_code,
                        args.output_dir / f"worker{worker_index}.err",
                    )
                )
    except KeyboardInterrupt:
        for _, _, process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
            stdout.close()
            stderr.close()
        raise
    if failed:
        details = "; ".join(
            f"worker={worker} gpu={gpu} exit={code} stderr={path}"
            for worker, gpu, code, path in failed
        )
        raise RuntimeError(f"Generation worker failure: {details}")


def main() -> None:
    args = parse_args()
    if args.worker_count not in (3, 4):
        raise ValueError("This launcher requires three or four independent workers")
    if not args.engine.is_file():
        raise FileNotFoundError(f"Missing generation engine: {args.engine}")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    gpu_ids = [] if args.merge_only else parse_gpu_ids(args.gpu_ids)
    if not args.merge_only and len(gpu_ids) != args.worker_count:
        raise ValueError(f"Expected {args.worker_count} GPU IDs, got {gpu_ids}")
    expected_count = read_retrieval_count(args.retrieval)
    if args.limit > 0:
        expected_count = min(expected_count, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{timestamp()}] Standalone four-condition generation: retrieval={args.retrieval} "
        f"rows={expected_count} top_k={args.top_k} output={args.output_dir} "
        f"context_source={args.context_source} GPUs={','.join(gpu_ids)}",
        flush=True,
    )
    if not args.merge_only:
        run_workers(args, expected_count, gpu_ids)
    merge(args, expected_count)
    print(f"[{timestamp()}] Generation experiment complete", flush=True)
    for condition in CONDITIONS:
        print(
            f"output={args.output_dir / f'{args.output_prefix}.{condition}.jsonl'}",
            flush=True,
        )
    manifest = {
        "created_at": timestamp(),
        "retrieval_file": str(args.retrieval),
        "output_dir": str(args.output_dir),
        "output_prefix": args.output_prefix,
        "row_count": expected_count,
        "top_k": args.top_k,
        "conditions": list(CONDITIONS),
        "model_path": args.model_path,
        "gpu_ids": gpu_ids,
        "worker_count": args.worker_count,
        "generation": {
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "bf16": bool(args.bf16),
        },
        "reference_response_used": False,
        "strategy_or_cot_used": False,
        "generation_protocol_version": (
            SUMMARY_ONLY_GENERATION_PROTOCOL_VERSION
            if args.context_source == "summary_only"
            else SUMMARY_PLUS_LATEST_GENERATION_PROTOCOL_VERSION
            if args.context_source == "summary_plus_latest"
            else DIALOGUE_PLUS_SUMMARY_GENERATION_PROTOCOL_VERSION
            if args.context_source == "dialogue_plus_summary"
            else GENERATION_PROTOCOL_VERSION
        ),
        "generation_context_source": (
            "rolecard_summary"
            if args.context_source == "summary_only"
            else "rolecard_summary_plus_latest"
            if args.context_source == "summary_plus_latest"
            else "dialogue_plus_rolecard_summary"
            if args.context_source == "dialogue_plus_summary"
            else "dialogue_prefix"
        ),
        "summary_used_in_generation_prompt": args.context_source in ("summary_only", "summary_plus_latest", "dialogue_plus_summary"),
        "dialogue_used_in_generation_prompt": args.context_source in ("dialogue", "dialogue_plus_summary"),
        "last_seeker_used_in_generation_prompt": args.context_source in ("dialogue", "summary_plus_latest", "dialogue_plus_summary"),
        "rolecard_summary_used_as_background": args.context_source == "dialogue_plus_summary",
        "persona_summary_used_in_generation_prompt": args.context_source == "dialogue_plus_summary",
        "rolecard_summary_model": (
            os.getenv("ROLECARD_SUMMARY_MODEL", "qwen3.7-max")
            if args.context_source == "dialogue_plus_summary"
            else None
        ),
        "rolecard_summary_prompt_version": (
            os.getenv(
                "ROLECARD_SUMMARY_PROMPT_VERSION",
                "escot-first-person-seeker-reddit-narrative-v2",
            )
            if args.context_source == "dialogue_plus_summary"
            else None
        ),
        "retrieval_query_source": "rolecard_summary",
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
