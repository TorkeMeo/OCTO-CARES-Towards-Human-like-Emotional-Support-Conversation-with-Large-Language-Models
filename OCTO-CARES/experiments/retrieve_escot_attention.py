#!/usr/bin/env python3
"""Retrieve post/comment memories for ESCoT queries with the fresh adapters.

The query is the Bailian-generated narrative summary.  A query and every
memory post are represented with the same fresh eight-adapter protocol used by
``attention_weight/eight_classifier_1053_attention_eval.py``.  The default
score is the ``base_hidden_adapter_attention`` family: each classifier's
answer-token attention weights ordinary Qwen final hidden states, and the eight
label-wise cosine scores are averaged.

The script can consume compatible NPZ caches.  If caches are absent it invokes
the existing no-gold extractor once for the missing records, writing canonical
temporary records that contain no labels, reference responses, or CoT fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ATTENTION_EVAL = PROJECT_DIR / "representation" / "eight_classifier_1053_attention_eval.py"
DEFAULT_SUMMARIES = SCRIPT_DIR / "escot_summaries.jsonl"
DEFAULT_OLD_CORPUS = PROJECT_DIR / "representation" / "Eight_attention_dataset.json"
DEFAULT_SUPPLEMENT_CORPUS = (
    PROJECT_DIR
    / "annotation"
    / "outputs"
    / "supplement_three_model_post_labels"
    / "supplement_post_labels_majority_vote_eval_schema.json"
)

# The fresh evaluator writes the first key in each tuple.  The second names
# are accepted for caches produced by the older 8_Type_attention evaluator.
VECTOR_KEYS_BY_METHOD = {
    "base_mean": ("base_hidden_post_mean_vectors", "qwen_base_hidden_post_mean_vectors"),
    "base+attn": (
        "base_hidden_adapter_attention_vectors",
        "qwen_base_hidden_adapter_attention_vectors",
    ),
    "base+delta": (
        "base_hidden_delta_ratio_attention_vectors",
        "qwen_base_hidden_delta_ratio_attention_vectors",
    ),
    "adapter+attn": ("adapter_hidden_attention_vectors", "qwen_hidden_attention_vectors"),
    "adapter+delta": (
        "adapter_hidden_delta_ratio_attention_vectors",
        "qwen_delta_ratio_attention_vectors",
    ),
    "adapter+rate": (
        "adapter_hidden_attention_change_rate_vectors",
        "qwen_hidden_attention_change_rate_vectors",
    ),
}
EXPECTED_LABEL_KEYS = [
    "positive_informational_self_disclosure",
    "negative_informational_self_disclosure",
    "neutral_informational_self_disclosure",
    "positive_emotional_self_disclosure",
    "negative_emotional_self_disclosure",
    "seek_emotional_support",
    "seek_informational_support",
    "seek_companionship",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument(
        "--corpus-file",
        type=Path,
        action="append",
        dest="corpus_files",
        default=None,
        help="Memory JSON file; repeat for the 1600 and 1053 corpora.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs" / "escot_attention")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--query-vectors", type=Path, default=None)
    parser.add_argument("--corpus-vectors", type=Path, default=None)
    parser.add_argument(
        "--source-corpus-vectors",
        type=Path,
        default=None,
        help=(
            "Existing complete predicted/no-gold corpus cache. When supplied, "
            "materialize the requested corpus subset from it instead of re-extracting it."
        ),
    )
    parser.add_argument("--method", choices=list(VECTOR_KEYS_BY_METHOD), default="base+attn")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extract-if-missing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--qwen-model-path", default="models/Qwen3-8B")
    parser.add_argument("--adapter-root", default="artifacts/classifiers")
    parser.add_argument("--attention-eval", type=Path, default=ATTENTION_EVAL)
    parser.add_argument("--max-qwen-length", type=int, default=40960)
    parser.add_argument("--qwen-prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--attention-layers", default="last:1")
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma/range list used for compatible vector extraction. Defaults to CUDA_VISIBLE_DEVICES or 4-7.",
    )
    parser.add_argument("--local-files-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=[0, 1], default=1)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--keep-invalid-comments", action="store_true")
    parser.add_argument(
        "--injectable-comment-required",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use the highest-ranked post with a non-empty, non-AutoModerator comment.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    raise ValueError(f"Cannot find a record list in {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def post_id(record: dict[str, Any]) -> str:
    direct = str(record.get("post_id") or record.get("id") or "").strip()
    if direct:
        return direct
    source = record.get("source_post")
    if isinstance(source, dict):
        return str(source.get("post_id") or source.get("id") or "").strip()
    return ""


def post_text(record: dict[str, Any]) -> tuple[str, str, str]:
    source = record.get("source_post") if isinstance(record.get("source_post"), dict) else {}
    title = str(record.get("title") or source.get("title") or "").strip()
    body = str(
        record.get("body")
        or record.get("selftext")
        or source.get("body")
        or source.get("selftext")
        or ""
    ).strip()
    direct = str(record.get("text") or "").strip()
    text = direct or "\n\n".join(part for part in (title, body) if part)
    return title, body, text


def comment_info(record: dict[str, Any]) -> dict[str, Any]:
    selected = record.get("selected_comment")
    if not isinstance(selected, dict):
        selected = {}
    body = str(selected.get("body") or record.get("selected_comment_body") or "").strip()
    author = str(selected.get("author") or "").strip()
    score_raw = selected.get("score", record.get("selected_comment_score"))
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None
    return {
        "comment_id": str(selected.get("id") or record.get("selected_comment_id") or "").strip(),
        "body": body,
        "author": author,
        "score": score,
        "word_count": word_count(body),
        "permalink": str(selected.get("permalink") or "").strip(),
    }


def comment_quality(comment: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    body = str(comment.get("body") or "")
    if not body:
        reasons.append("empty")
    author = str(comment.get("author") or "").lower()
    if "automoderator" in author:
        reasons.append("automoderator")
    if "automoderator" in body.lower():
        reasons.append("automoderator_text")
    return (not reasons, reasons)


def load_memory_records(paths: list[Path], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    stats: dict[str, Any] = {"files": [], "duplicate_ids": 0, "duplicate_texts": 0, "dropped": 0}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        source_name = "supplement1053" if "supplement" in path.name.lower() else "old1600"
        records = records_from_payload(read_json(path), path)
        file_info = {"path": str(path.resolve()), "source": source_name, "raw_count": len(records), "kept": 0}
        for record in records:
            pid = post_id(record)
            title, body, text = post_text(record)
            if not pid or not text:
                stats["dropped"] += 1
                continue
            if pid in seen_ids:
                stats["duplicate_ids"] += 1
                continue
            comment = comment_info(record)
            valid, reasons = comment_quality(comment)
            if not valid:
                stats.setdefault("invalid_comment_count", 0)
                stats["invalid_comment_count"] += 1
                # Keep the post in the retrieval index.  A filtered
                # comment should not remove an otherwise useful historical
                # post; invalid comments are simply never selected for the
                # paired or random-comment injection.
            normalized = {
                "memory_id": pid,
                "source": source_name,
                "subreddit": str(record.get("subreddit") or record.get("source_subreddit") or "unknown"),
                "title": title,
                "body": body,
                "text": text,
                "comment": comment,
                "comment_quality": {
                    "valid": valid,
                    "injectable": bool(valid or args.keep_invalid_comments),
                    "reasons": reasons,
                },
            }
            text_key = re.sub(r"\s+", " ", text).strip().lower()
            if text_key in seen_texts:
                stats["duplicate_texts"] += 1
            seen_ids.add(pid)
            seen_texts.add(text_key)
            output.append(normalized)
            file_info["kept"] += 1
        stats["files"].append(file_info)
    if not output:
        raise ValueError("No usable memory records remain after comment filtering")
    stats["kept_count"] = len(output)
    return output, stats


def load_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summaries: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            summary = str(row.get("summary") or "").strip()
            if not query_id or not summary:
                raise ValueError(f"Summary row {line_number} lacks query_id or summary")
            if query_id in seen:
                raise ValueError(f"Duplicate summary query_id={query_id}")
            seen.add(query_id)
            rows.append(
                {
                    "query_id": query_id,
                    "source_id": str(row.get("source_id") or query_id).strip(),
                    "summary": summary,
                    "summary_model": str(row.get("summary_model") or "").strip() or None,
                    "summary_prompt_version": str(row.get("summary_prompt_version") or "").strip() or None,
                    "dialogue_prefix": str(row.get("dialogue_prefix") or "").strip(),
                    "last_seeker": str(row.get("last_seeker") or "").strip(),
                    "turn_count": int(row.get("turn_count") or 0),
                }
            )
    if not rows:
        raise ValueError(f"No summaries found in {path}")
    return rows


def canonical_record(identifier: str, text: str, subreddit: str = "unknown") -> dict[str, Any]:
    return {
        "post_id": identifier,
        "subreddit": subreddit,
        "title": "",
        "body": text,
        "text": text,
        "source_post": {"id": identifier, "subreddit": subreddit, "title": "", "selftext": text},
        "status": "complete",
    }


def parse_gpu_ids(raw: str) -> list[str]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        cleaned = os.getenv("CUDA_VISIBLE_DEVICES", "").strip() or os.getenv("GPU_IDS", "").strip() or "4-7"
    cleaned = cleaned.replace(";", ",").replace(" ", "")
    result: list[str] = []
    for item in cleaned.split(","):
        if not item:
            continue
        if re.fullmatch(r"\d+-\d+", item):
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            result.extend(str(value) for value in range(start, end + step, step))
        else:
            result.append(item)
    if not result:
        raise ValueError(f"No GPU IDs parsed from {raw!r}")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate GPU IDs: {result}")
    return result


def np_scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = data[key]
    try:
        value = value.tolist()
    except AttributeError:
        pass
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def load_vector_file(path: Path, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Missing vector cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        key = next((candidate for candidate in VECTOR_KEYS_BY_METHOD[method] if candidate in data), None)
        if "post_ids" not in data or key is None:
            raise ValueError(
                f"Vector cache {path} lacks post_ids or a vector key from "
                f"{VECTOR_KEYS_BY_METHOD[method]}"
            )
        ids = data["post_ids"].astype(str).tolist()
        vectors = np.asarray(data[key], dtype=np.float32)
        metadata = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "record_count": len(ids),
            "vector_key": key,
            "label_keys": np_scalar(data, "label_keys", []),
            "attention_layers": np_scalar(data, "attention_layers", ""),
            "attention_answer_mode": np_scalar(data, "attention_answer_mode", ""),
            "gold_labels_used_for_prompt": np_scalar(data, "gold_labels_used_for_prompt", None),
        }
    if len(ids) != len(set(ids)):
        raise ValueError(f"Vector cache contains duplicate post_ids: {path}")
    if vectors.ndim not in {2, 3} or vectors.shape[0] != len(ids):
        raise ValueError(f"Unexpected vector shape {vectors.shape} in {path}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Vector cache contains NaN/Inf: {path}")
    if metadata["attention_answer_mode"] != "predicted":
        raise ValueError(
            f"Vector cache must explicitly declare attention_answer_mode=predicted: {path}"
        )
    if metadata["gold_labels_used_for_prompt"] not in (False, 0):
        raise ValueError(
            f"Vector cache must explicitly declare gold_labels_used_for_prompt=false: {path}"
        )
    expected_rank = 2 if method == "base_mean" else 3
    if vectors.ndim != expected_rank:
        raise ValueError(
            f"Method {method} requires {expected_rank}-D vectors, got {vectors.shape}: {path}"
        )
    label_keys = metadata["label_keys"]
    if method != "base_mean" and label_keys != EXPECTED_LABEL_KEYS:
        raise ValueError(
            f"Method {method} requires the canonical ordered eight label keys: {path}"
        )
    return {"ids": ids, "vectors": vectors, "index": {pid: i for i, pid in enumerate(ids)}}, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_set_sha256(ids: list[str], texts: list[str]) -> str:
    digest = hashlib.sha256()
    for identifier, value in zip(ids, texts):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(".input.json")


def protocol_fingerprint(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    result: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file():
        result["sha256"] = sha256_file(path)
        return result
    if path.is_dir():
        files = []
        for pattern in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "model*.safetensors",
        ):
            for candidate in sorted(path.rglob(pattern)):
                stat = candidate.stat()
                files.append(
                    {
                        "relative_path": str(candidate.relative_to(path)),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        result["files"] = files
    return result


def expected_cache_metadata(
    ids: list[str], texts: list[str], args: argparse.Namespace, side: str
) -> dict[str, Any]:
    return {
        "side": side,
        "record_count": len(ids),
        "text_sha256": text_set_sha256(ids, texts),
        "method": args.method,
        "qwen_model": protocol_fingerprint(args.qwen_model_path),
        "adapters": protocol_fingerprint(args.adapter_root),
        "attention_layers": args.attention_layers,
        "max_qwen_length": args.max_qwen_length,
        "qwen_prefill_chunk_size": args.qwen_prefill_chunk_size,
        "attention_answer_mode": "predicted",
        "gold_labels_used_for_prompt": False,
        "normalize_output_vectors": True,
        "extractor_sha256": (
            sha256_file(args.attention_eval) if args.attention_eval.is_file() else "unavailable"
        ),
    }


def cache_metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    metadata_path = cache_metadata_path(path)
    if not metadata_path.is_file():
        return False
    try:
        actual = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def run_extractor(records: list[dict[str, Any]], output_path: Path, args: argparse.Namespace) -> None:
    if not records:
        raise ValueError("Cannot extract vectors for an empty record list")
    if not args.attention_eval.is_file():
        raise FileNotFoundError(f"Missing attention extractor: {args.attention_eval}")
    if not Path(args.qwen_model_path).exists():
        raise FileNotFoundError(f"Missing Qwen model: {args.qwen_model_path}")
    if not Path(args.adapter_root).is_dir():
        raise FileNotFoundError(f"Missing adapter root: {args.adapter_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "_vector_input_records.json"
    write_json(input_path, records)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    worker_count = min(len(gpu_ids), len(records))
    if worker_count <= 0:
        raise ValueError("No extraction workers available")

    common = [
            sys.executable,
            str(args.attention_eval),
            "extract",
            "--data-file",
            str(input_path),
            "--source-file",
            "",
            "--label-keys",
            "all",
            "--record-start",
            "0",
            "--max-records",
            "0",
            "--output-file",
            str(output_path),
            "--overwrite",
            "1",
            "--qwen-model-name-or-path",
            args.qwen_model_path,
            "--adapter-root",
            args.adapter_root,
            "--adapter-subdir",
            "final",
            "--local-files-only",
            str(args.local_files_only),
            "--trust-remote-code",
            str(args.trust_remote_code),
            "--attn-implementation",
            "eager",
            "--bf16",
            str(args.bf16),
            "--device-map",
            "",
            "--qwen-max-memory",
            "",
            "--qwen-device",
            "cuda:0",
            "--max-qwen-length",
            str(args.max_qwen_length),
            "--qwen-prefill-chunk-size",
            str(args.qwen_prefill_chunk_size),
            "--attention-layers",
            args.attention_layers,
            "--attention-answer-mode",
            "predicted",
            "--prediction-threshold",
            "0.5",
            "--fail-on-nonfinite",
            "1",
            "--normalize-output-vectors",
            "1",
            "--ratio-eps",
            "1e-6",
            "--max-ratio",
            "64",
            "--compress",
            "1",
        ]
    # Keep the four A6000 workers independent.  Each child sees one physical
    # device and addresses it as cuda:0, exactly like the existing evaluator.
    shard_paths: list[Path] = []
    ranges: list[tuple[int, int, str, Path]] = []
    chunk = (len(records) + worker_count - 1) // worker_count
    for worker in range(worker_count):
        start = worker * chunk
        if start >= len(records):
            continue
        count = min(chunk, len(records) - start)
        shard_path = args.output_dir / f"_vectors_shard{worker}.npz"
        shard_path.unlink(missing_ok=True)
        shard_paths.append(shard_path)
        ranges.append((start, count, gpu_ids[worker], shard_path))

    print(
        f"[{timestamp()}] Extracting compatible vectors for {len(records)} records "
        f"with GPUs={','.join(item[2] for item in ranges)}",
        flush=True,
    )

    def run_one(item: tuple[int, int, str, Path]) -> None:
        start, count, gpu, shard_path = item
        command = list(common)
        command[command.index("--record-start") + 1] = str(start)
        command[command.index("--max-records") + 1] = str(count)
        command[command.index("--output-file") + 1] = str(shard_path)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        stdout_path = shard_path.with_name(f"{shard_path.stem}.worker.out")
        stderr_path = shard_path.with_name(f"{shard_path.stem}.worker.err")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            subprocess.run(command, check=True, env=environment, stdout=stdout, stderr=stderr)

    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [pool.submit(run_one, item) for item in ranges]
        for future in as_completed(futures):
            future.result()

    merge_command = [
        sys.executable,
        str(args.attention_eval),
        "merge",
        "--data-file",
        str(input_path),
        "--source-file",
        "",
        "--label-keys",
        "all",
        "--record-start",
        "0",
        "--max-records",
        "0",
        "--output-file",
        str(output_path),
        "--overwrite",
        "1",
        "--compress",
        "1",
    ]
    for shard_path in shard_paths:
        merge_command.extend(["--shard-file", str(shard_path)])
    subprocess.run(merge_command, check=True)
    # Keep per-worker logs for diagnosing numerical/model issues, but remove
    # bulky transient shards after the merged cache is verified.
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
        shard_path.with_suffix(".summary.json").unlink(missing_ok=True)
    input_path.unlink(missing_ok=True)


def save_subset_vector_file(source_path: Path, destination: Path, ids: list[str], method: str) -> None:
    import numpy as np

    with np.load(source_path, allow_pickle=False) as source:
        source_ids = source["post_ids"].astype(str).tolist()
        locations = {pid: i for i, pid in enumerate(source_ids)}
        missing = [pid for pid in ids if pid not in locations]
        if missing:
            raise ValueError(f"Extractor output is missing {len(missing)} requested IDs: {missing[:5]}")
        selected_key = next(
            (candidate for candidate in VECTOR_KEYS_BY_METHOD[method] if candidate in source),
            None,
        )
        if selected_key is None:
            raise KeyError(f"Extractor output lacks {VECTOR_KEYS_BY_METHOD[method]}")
        row_indices = [locations[pid] for pid in ids]
        payload: dict[str, Any] = {
            selected_key: source[selected_key][row_indices],
        }
        if "subreddits" in source and source["subreddits"].shape[0] == len(source_ids):
            payload["subreddits"] = source["subreddits"][row_indices]
        for name in (
            "created_at",
            "script",
            "label_keys",
            "selected_layers",
            "attention_layers",
            "max_qwen_length",
            "attention_answer_mode",
            "attention_query",
            "gold_labels_used_for_prompt",
            "gold_labels_in_output",
            "labels_accessed_during_inference",
            "classification_metrics_computed",
            "normalize_output_vectors",
            "ratio_eps",
            "max_ratio",
        ):
            if name in source:
                payload[name] = source[name]
        payload["post_ids"] = np.asarray(ids, dtype=str)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)


def ensure_vectors(
    query_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load supplied caches or extract missing query/corpus rows."""

    import numpy as np

    query_ids = [row["query_id"] for row in query_rows]
    memory_ids = [row["memory_id"] for row in memory_rows]
    if set(query_ids) & set(memory_ids):
        raise ValueError("Query and memory post IDs overlap; refusing possible retrieval leakage")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.query_vectors or (args.output_dir / "escot_query_vectors.npz")
    corpus_path = args.corpus_vectors or (args.output_dir / "memory_corpus_vectors.npz")
    if query_path.resolve() == corpus_path.resolve():
        raise ValueError("Query and corpus vector cache paths must be different")
    query_expected = expected_cache_metadata(
        query_ids,
        [row["summary"] for row in query_rows],
        args,
        "query",
    )
    corpus_expected = expected_cache_metadata(
        memory_ids,
        [row["text"] for row in memory_rows],
        args,
        "corpus",
    )
    source_corpus_meta: dict[str, Any] = {}
    if args.source_corpus_vectors is not None:
        source_path = args.source_corpus_vectors
        if source_path.resolve() == corpus_path.resolve():
            raise ValueError("source-corpus-vectors and corpus-vectors must be different paths")
        source_cache, source_corpus_meta = load_vector_file(source_path, args.method)
        if source_corpus_meta.get("attention_layers") != args.attention_layers:
            raise ValueError(
                "Source corpus attention layer mismatch: "
                f"{source_corpus_meta.get('attention_layers')!r} != {args.attention_layers!r}"
            )
        source_missing = [memory_id for memory_id in memory_ids if memory_id not in source_cache["index"]]
        if source_missing:
            raise ValueError(
                f"Source corpus cache lacks {len(source_missing)} requested memories: {source_missing[:5]}"
            )
        if args.force or not cache_metadata_matches(corpus_path, corpus_expected):
            save_subset_vector_file(source_path, corpus_path, memory_ids, args.method)
            write_json(cache_metadata_path(corpus_path), corpus_expected)
    query_cache = None
    corpus_cache = None
    query_meta: dict[str, Any] = {}
    corpus_meta: dict[str, Any] = {}
    if query_path.is_file() and not args.force and cache_metadata_matches(query_path, query_expected):
        query_cache, query_meta = load_vector_file(query_path, args.method)
    elif query_path.is_file() and not args.force:
        print(f"[{timestamp()}] Query vector cache metadata is stale; regenerating {query_path}", flush=True)
    if corpus_path.is_file() and not args.force and cache_metadata_matches(corpus_path, corpus_expected):
        corpus_cache, corpus_meta = load_vector_file(corpus_path, args.method)
    elif corpus_path.is_file() and not args.force:
        print(f"[{timestamp()}] Corpus vector cache metadata is stale; regenerating {corpus_path}", flush=True)

    # A partial cache is not merged in place: replacing it with a complete
    # side-specific cache keeps row order and all metadata unambiguous.
    query_missing_ids = (
        set(query_ids)
        if query_cache is None
        else {row["query_id"] for row in query_rows if row["query_id"] not in query_cache["index"]}
    )
    memory_missing_ids = (
        set(memory_ids)
        if corpus_cache is None
        else {row["memory_id"] for row in memory_rows if row["memory_id"] not in corpus_cache["index"]}
    )
    # If even one row is absent, regenerate that entire side so an old partial
    # file is never silently replaced by a cache containing only the missing
    # suffix.
    missing_query = list(query_rows) if query_missing_ids else []
    missing_memory = list(memory_rows) if memory_missing_ids else []
    if (missing_query or missing_memory) and not args.extract_if_missing:
        raise FileNotFoundError("Compatible query/corpus vectors are missing and extraction was disabled")

    if missing_query or missing_memory:
        extracted_query_ids = [row["query_id"] for row in missing_query]
        extracted_memory_ids = [row["memory_id"] for row in missing_memory]
        # Extract both sides in one four-worker pass.  This loads the base model
        # and eight adapters once per worker instead of once for queries and
        # once again for the memory corpus.
        extracted_path = args.output_dir / "_fresh_combined_vectors.npz"
        extracted_path.unlink(missing_ok=True)
        extraction_rows = [
            canonical_record(row["query_id"], row["summary"], "escot") for row in missing_query
        ] + [
            canonical_record(row["memory_id"], row["text"], row["subreddit"]) for row in missing_memory
        ]
        run_extractor(extraction_rows, extracted_path, args)
        if missing_query:
            save_subset_vector_file(extracted_path, query_path, extracted_query_ids, args.method)
            write_json(cache_metadata_path(query_path), query_expected)
            query_cache, query_meta = load_vector_file(query_path, args.method)
        if missing_memory:
            save_subset_vector_file(extracted_path, corpus_path, extracted_memory_ids, args.method)
            write_json(cache_metadata_path(corpus_path), corpus_expected)
            corpus_cache, corpus_meta = load_vector_file(corpus_path, args.method)
        extracted_path.unlink(missing_ok=True)

    if query_cache is None or corpus_cache is None:
        raise RuntimeError("Failed to materialize both query and corpus vector caches")
    for field in ("label_keys", "attention_layers", "attention_answer_mode"):
        query_value = query_meta.get(field)
        corpus_value = corpus_meta.get(field)
        # Both caches have passed the explicit no-gold gate; their shared
        # protocol fields must also agree exactly.
        if (
            query_value not in (None, "", [])
            and corpus_value not in (None, "", [])
            and query_value != corpus_value
        ):
            raise ValueError(
                f"Query/corpus cache metadata mismatch for {field}: {query_value!r} vs {corpus_value!r}"
            )
    missing_query_ids = [pid for pid in query_ids if pid not in query_cache["index"]]
    missing_memory_ids = [pid for pid in memory_ids if pid not in corpus_cache["index"]]
    if missing_query_ids or missing_memory_ids:
        raise ValueError(f"Vector cache coverage incomplete: query={missing_query_ids[:3]}, memory={missing_memory_ids[:3]}")

    query_indices = [query_cache["index"][pid] for pid in query_ids]
    memory_indices = [corpus_cache["index"][pid] for pid in memory_ids]
    query_vectors = np.asarray(query_cache["vectors"])[query_indices]
    memory_vectors = np.asarray(corpus_cache["vectors"])[memory_indices]
    if query_vectors.ndim != memory_vectors.ndim:
        raise ValueError(f"Query/corpus vector rank mismatch: {query_vectors.shape} vs {memory_vectors.shape}")
    if query_vectors.shape[1:] != memory_vectors.shape[1:]:
        raise ValueError(f"Query/corpus vector shape mismatch: {query_vectors.shape} vs {memory_vectors.shape}")
    if args.method != "base_mean" and (query_vectors.ndim != 3 or query_vectors.shape[1] != 8):
        raise ValueError(
            f"Eight-classifier method {args.method} requires [N,8,H] vectors, got {query_vectors.shape}"
        )
    return (
        {"ids": query_ids, "vectors": query_vectors},
        {"ids": memory_ids, "vectors": memory_vectors},
        {
            "query": query_meta,
            "corpus": corpus_meta,
            "query_path": str(query_path),
            "corpus_path": str(corpus_path),
            "source_corpus": source_corpus_meta or None,
        },
    )


def normalized_scores(query_vectors: Any, memory_vectors: Any) -> Any:
    import numpy as np

    def norm(value: Any) -> Any:
        lengths = np.linalg.norm(value, axis=-1, keepdims=True)
        return value / np.maximum(lengths, 1e-12)

    q = norm(np.asarray(query_vectors, dtype=np.float32))
    m = norm(np.asarray(memory_vectors, dtype=np.float32))
    if q.ndim == 2:
        return q @ m.T
    if q.ndim != 3 or m.ndim != 3 or q.shape[1:] != m.shape[1:]:
        raise ValueError(f"Expected matching [N,8,H] vectors, got {q.shape} and {m.shape}")
    # Calculate one cosine per label, then average the eight label scores.
    return np.einsum("qlh,mlh->qml", q, m, optimize=True).mean(axis=2)


def memory_for_random_comment(
    rows: list[dict[str, Any]], matched_id: str, target_words: int, rng: random.Random
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["memory_id"] != matched_id and row["comment_quality"]["injectable"]
    ]
    if not candidates:
        raise ValueError("No independent valid comment is available for random-comment control")
    # Match the rough length of the paired comment so quality/length is not the
    # only reason the random-comment condition differs.
    candidates.sort(key=lambda row: abs(int(row["comment"]["word_count"]) - target_words))
    pool = candidates[: min(50, len(candidates))]
    return rng.choice(pool)


def compact_memory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": row["memory_id"],
        "source": row["source"],
        "subreddit": row["subreddit"],
        "title": row["title"],
        "body": row["body"],
        "text": row["text"],
        "comment": row["comment"],
        "comment_quality": row["comment_quality"],
    }


def build_retrieval_rows(
    query_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    query_vectors: dict[str, Any],
    memory_vectors: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import numpy as np

    scores = normalized_scores(query_vectors["vectors"], memory_vectors["vectors"])
    if scores.ndim != 2:
        raise ValueError(f"Expected a query-by-memory score matrix, got {scores.shape}")
    top_k = max(1, min(args.top_k, len(memory_rows)))
    output: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_rows):
        row_scores = np.asarray(scores[query_index], dtype=np.float64)
        full_order = np.argsort(-np.where(np.isfinite(row_scores), row_scores, -np.inf), kind="stable")
        order = full_order[:top_k]
        neighbors = []
        for rank, memory_index in enumerate(order, start=1):
            memory = memory_rows[int(memory_index)]
            neighbors.append(
                {
                    "rank": rank,
                    "score": float(row_scores[int(memory_index)]),
                    **compact_memory(memory),
                }
            )
        # Prefer the highest-ranked post with an injectable comment.  This
        # keeps post-only and post+comment conditions on the same memory item
        # while preventing an empty/automoderator comment from entering a
        # prompt.  If none of the returned neighbors has a valid comment,
        # retain rank 1 and let the generator record the missing comment.
        if args.injectable_comment_required:
            matched_position = next(
                (
                    position
                    for position, memory_index in enumerate(full_order)
                    if memory_rows[int(memory_index)]["comment_quality"]["injectable"]
                ),
                0,
            )
        else:
            matched_position = 0
        matched_index = int(full_order[matched_position])
        matched = memory_rows[matched_index]
        matched_payload = compact_memory(matched)
        matched_payload["score"] = float(row_scores[matched_index])
        rng = random.Random(args.seed + query_index * 1009)
        random_memory = memory_for_random_comment(
            memory_rows,
            matched["memory_id"],
            int(matched["comment"]["word_count"]),
            rng,
        )
        output.append(
            {
                "query_id": query["query_id"],
                "source_id": query["source_id"],
                "summary": query["summary"],
                "summary_model": query.get("summary_model"),
                "summary_prompt_version": query.get("summary_prompt_version"),
                "dialogue_prefix": query["dialogue_prefix"],
                "last_seeker": query["last_seeker"],
                "turn_count": query["turn_count"],
                "retrieval_method": args.method,
                "retrieval_formula": (
                    "mean_l cosine(q_l,m_l), l=1..8; q_l and m_l are normalized "
                    "ordinary-Qwen hidden vectors weighted by each fresh adapter's "
                    "predicted answer-token attention"
                ),
                "neighbors": neighbors,
                "matched_memory": matched_payload,
                "matched_rank": int(matched_position + 1),
                "random_comment": {
                    "memory_id": random_memory["memory_id"],
                    "source": random_memory["source"],
                    "comment": random_memory["comment"],
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.corpus_files is None:
        args.corpus_files = [DEFAULT_OLD_CORPUS, DEFAULT_SUPPLEMENT_CORPUS]
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    supplied_caches = bool(
        args.query_vectors
        and args.corpus_vectors
        and args.query_vectors.is_file()
        and args.corpus_vectors.is_file()
    )
    if args.extract_if_missing and not supplied_caches:
        if not args.attention_eval.is_file():
            raise FileNotFoundError(f"Missing attention extractor: {args.attention_eval}")
        if not Path(args.qwen_model_path).exists():
            raise FileNotFoundError(f"Missing Qwen model: {args.qwen_model_path}")
        if not Path(args.adapter_root).is_dir():
            raise FileNotFoundError(f"Missing adapter root: {args.adapter_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_file or (args.output_dir / "retrieval.jsonl")
    manifest_file = args.manifest or (args.output_dir / "retrieval.manifest.json")
    query_rows = load_summaries(args.summaries)
    memory_rows, memory_stats = load_memory_records(args.corpus_files, args)
    print(f"[{timestamp()}] queries={len(query_rows)} usable_memories={len(memory_rows)}", flush=True)
    query_vectors, memory_vectors, vector_meta = ensure_vectors(query_rows, memory_rows, args)
    rows = build_retrieval_rows(query_rows, memory_rows, query_vectors, memory_vectors, args)
    write_jsonl(output_file, rows)
    manifest = {
        "created_at": timestamp(),
        "summaries": str(args.summaries.resolve()),
        "corpus_files": [str(path.resolve()) for path in args.corpus_files],
        "output_file": str(output_file.resolve()),
        "query_count": len(query_rows),
        "corpus_count": len(memory_rows),
        "method": args.method,
        "vector_key_candidates": VECTOR_KEYS_BY_METHOD[args.method],
        "top_k": args.top_k,
        "seed": args.seed,
        "query_memory_id_overlap": False,
        "gold_labels_used_for_prompt": False,
        "reference_response_used": False,
        "strategy_or_cot_used": False,
        "attention_layers": args.attention_layers,
        "memory_stats": memory_stats,
        "vector_cache": vector_meta,
        "formula": (
            "S(q,m) = (1/8) * sum_l cosine(v_q,l, v_m,l); "
            "v_x,l = Norm(sum_t normalized_adapter_attention_x,l,t * "
            "ordinary_Qwen_final_hidden_x,t)"
        ),
        "comment_filter": {
            "requires_nonempty_body": True,
            "rejects_automoderator": True,
            "length_filter_enabled": False,
            "score_filter_enabled": False,
            "keep_invalid_comments": bool(args.keep_invalid_comments),
            "injectable_comment_required": bool(args.injectable_comment_required),
        },
    }
    write_json(manifest_file, manifest)
    print(f"[{timestamp()}] wrote {len(rows)} retrieval rows to {output_file}", flush=True)


if __name__ == "__main__":
    main()
