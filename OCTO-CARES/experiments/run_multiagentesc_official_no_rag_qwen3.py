#!/usr/bin/env python3
"""Launch three or four local-Qwen3 workers for the official-protocol no-RAG baseline."""

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
ENGINE = SCRIPT_DIR / "generate_multiagentesc_official_no_rag_qwen3.py"
CONDITION = "multiagent_esc_official_no_rag"
PROTOCOL_VERSION = "multiagentesc-official-protocol-no-rag-qwen3-v1"
UPSTREAM_REPOSITORY = "https://github.com/MindIntLab-HFUT/MultiAgentESC"
UPSTREAM_COMMIT = "631b7f1961fc7502e547fd9258e847230dbcb973"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="responses")
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--gpu-ids", default=os.getenv("GPU_LIST", "4-7"))
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--decision-max-new-tokens", type=int, default=100)
    parser.add_argument("--analysis-max-new-tokens", type=int, default=400)
    parser.add_argument("--discussion-max-new-tokens", type=int, default=400)
    parser.add_argument("--response-max-new-tokens", type=int, default=100)
    parser.add_argument("--judge-max-new-tokens", type=int, default=400)
    parser.add_argument("--refine-max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", type=int, choices=(0, 1), default=1)
    parser.add_argument("--local-files-only", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=(0, 1), default=1)
    parser.add_argument("--strategy-agent-count", type=int, default=3)
    parser.add_argument(
        "--complexity-gate", choices=("official", "always_multiagent"), default="official"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--engine", type=Path, default=ENGINE)
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_gpu_ids(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw or "").replace(";", ",").replace(" ", "").split(","):
        if not item:
            continue
        if item.count("-") == 1 and all(part.isdigit() for part in item.split("-", 1)):
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            values.extend(str(number) for number in range(start, end + step, step))
        elif item.isdigit():
            values.append(item)
        else:
            raise ValueError(f"Invalid GPU identifier: {item!r}")
    if not values or len(values) != len(set(values)):
        raise ValueError(f"GPU IDs must be non-empty and distinct: {raw!r}")
    return values


def input_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input JSONL: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(row.get("query_id") or "").strip():
                raise ValueError(f"Invalid row at {path}:{line_number}")
            count += 1
    if not count:
        raise ValueError(f"Input JSONL is empty: {path}")
    return count


def engine_command(
    args: argparse.Namespace, *, worker_index: int | None = None, merge: bool = False
) -> list[str]:
    command = [
        sys.executable,
        str(args.engine),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        args.output_prefix,
        "--model-path",
        args.model_path,
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--decision-max-new-tokens",
        str(args.decision_max_new_tokens),
        "--analysis-max-new-tokens",
        str(args.analysis_max_new_tokens),
        "--discussion-max-new-tokens",
        str(args.discussion_max_new_tokens),
        "--response-max-new-tokens",
        str(args.response_max_new_tokens),
        "--judge-max-new-tokens",
        str(args.judge_max_new_tokens),
        "--refine-max-new-tokens",
        str(args.refine_max_new_tokens),
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
        "--strategy-agent-count",
        str(args.strategy_agent_count),
        "--complexity-gate",
        args.complexity_gate,
        "--worker-count",
        str(args.worker_count),
    ]
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.force:
        command.append("--force")
    if merge:
        command.append("--merge")
    else:
        if worker_index is None:
            raise ValueError("worker_index is required")
        command.extend(["--worker-index", str(worker_index)])
    return command


def main() -> None:
    args = parse_args()
    if args.worker_count not in (3, 4):
        raise ValueError("The project launcher requires three or four workers")
    if "qwen3" not in Path(args.model_path).name.lower():
        raise ValueError(f"All roles must use a Qwen3 checkpoint, got: {args.model_path}")
    if args.strategy_agent_count != 3:
        raise ValueError("The official protocol uses exactly three strategy participants")
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    if not args.engine.is_file():
        raise FileNotFoundError(f"Missing engine: {args.engine}")
    expected = input_count(args.input)
    if args.limit > 0:
        expected = min(expected, args.limit)
    gpu_ids = [] if args.merge_only else parse_gpu_ids(args.gpu_ids)
    if not args.merge_only and len(gpu_ids) != args.worker_count:
        raise ValueError(f"Expected {args.worker_count} GPU IDs, got {gpu_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{timestamp()}] official MultiAgentESC no-RAG Qwen3: rows={expected} "
        f"input={args.input} output={args.output_dir} model={args.model_path} "
        f"GPUs={','.join(gpu_ids) if gpu_ids else '<merge-only>'}",
        flush=True,
    )

    if not args.merge_only:
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
                process = subprocess.Popen(
                    engine_command(args, worker_index=worker_index),
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
                processes.append((worker_index, gpu, process, stdout, stderr))
                print(
                    f"[{timestamp()}] started worker={worker_index} gpu={gpu} "
                    f"rows~{(expected + args.worker_count - 1) // args.worker_count}",
                    flush=True,
                )
            failures: list[tuple[int, str, int, Path]] = []
            for worker_index, gpu, process, stdout, stderr in processes:
                code = process.wait()
                stdout.close()
                stderr.close()
                if code:
                    failures.append(
                        (worker_index, gpu, code, args.output_dir / f"worker{worker_index}.err")
                    )
            if failures:
                raise RuntimeError(
                    "; ".join(
                        f"worker={worker} gpu={gpu} exit={code} stderr={path}"
                        for worker, gpu, code, path in failures
                    )
                )
        except KeyboardInterrupt:
            for _, _, process, stdout, stderr in processes:
                if process.poll() is None:
                    process.terminate()
                stdout.close()
                stderr.close()
            raise

    subprocess.run(engine_command(args, merge=True), check=True)
    manifest = {
        "created_at": timestamp(),
        "condition": CONDITION,
        "protocol_version": PROTOCOL_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "input_file": str(args.input),
        "output_file": str(
            args.output_dir / f"{args.output_prefix}.{CONDITION}.jsonl"
        ),
        "row_count": expected,
        "model_path_for_every_role": args.model_path,
        "homogeneous_qwen3_agents": True,
        "retrieval_removed": True,
        "top_k": None,
        "top_k_applicability": "not_applicable",
        "complexity_gate": args.complexity_gate,
        "worker_count": args.worker_count,
        "gpu_ids": gpu_ids,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "bf16": bool(args.bf16),
        "max_input_tokens": args.max_input_tokens,
        "stage_token_budgets": {
            "decision": args.decision_max_new_tokens,
            "analysis": args.analysis_max_new_tokens,
            "discussion": args.discussion_max_new_tokens,
            "response": args.response_max_new_tokens,
            "judge": args.judge_max_new_tokens,
            "refiner": args.refine_max_new_tokens,
        },
        "preserved_components": [
            "official_complexity_gate",
            "sequential_emotion_cause_intention",
            "three_participant_round_robin_strategy_group",
            "one_response_per_distinct_strategy",
            "candidate_bound_debate",
            "separate_reflection_round",
            "majority_vote",
            "dedicated_tie_judge",
            "mandatory_final_refiner",
        ],
        "removed_components": [
            "sbert",
            "top10_esconv_retrieval",
            "retrieved_strategy_response_examples",
        ],
        "runtime_adaptation": (
            "Official deterministic round-robin group semantics are executed directly with "
            "one local Qwen3 Transformers instance per worker; AutoGen is not required."
        ),
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{timestamp()}] complete: {manifest['output_file']}", flush=True)


if __name__ == "__main__":
    main()
