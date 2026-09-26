#!/usr/bin/env python3
"""Train one post-label binary SFT model on the complete 1440-record train set.

This script intentionally trains a causal-LM LoRA adapter that generates one
token-like answer, "0" or "1". It does not create a classifier layer, a
multi-label head, or any eight-label joint target.

V2 defaults to annotation/outputs/Total_train_data_1440_train.json. The separate
160-record test set is never loaded by this training script. Long posts are
truncated only inside the Reddit post body, preserving the category definition,
</post>, and Answer: prompt suffix.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FRESHNEW_DIR = PROJECT_DIR / "annotation"
if str(FRESHNEW_DIR) not in sys.path:
    sys.path.insert(0, str(FRESHNEW_DIR))

from post_labels import DECISION_RULE_TEXT, POST_LABELS  # noqa: E402


SYSTEM_PROMPT = """You are a careful research annotator.
Your task is to judge only one category for one Reddit post.
Use only the supplied category name, category definition, and Reddit post.
Consider the title and body together, including negation, quoted speech, sarcasm, and who is experiencing the event or emotion.
Return exactly one word: 0 or 1.
Do not return explanations, punctuation, bullet points, JSON, Markdown, or any other text."""

LABEL_DEFS = {item.key: item for item in POST_LABELS}
LABEL_KEYS = [item.key for item in POST_LABELS]


@dataclass
class Example:
    post_id: str
    subreddit: str
    title: str
    body: str
    text: str
    label: int


@dataclass
class EncodedExample:
    example: Example
    prompt_text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_len: int
    was_truncated: bool
    original_text_chars: int
    kept_text_chars: int
    prompt_token_count: int


class ListDataset:
    def __init__(self, rows: list[EncodedExample]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file",
        default=str(PROJECT_DIR / "annotation" / "outputs" / "Total_train_data_1440_train.json"),
    )
    parser.add_argument("--source_file", default="")
    parser.add_argument(
        "--fixed_split_file",
        default="",
        help="Optional split JSON for excluding test post IDs. The complete 1440 train file needs none.",
    )
    parser.add_argument("--label_key", required=True, choices=LABEL_KEYS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", choices=["auto", "none", "local_rank"], default="auto")
    parser.add_argument("--max_length", type=int, default=42000)
    parser.add_argument("--post_truncation_strategy", choices=["head", "tail", "head_tail"], default="head_tail")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=20260809)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--invalid_attempts", type=int, default=3)
    parser.add_argument("--invalid_retry_temperature", type=float, default=0.7)
    parser.add_argument("--invalid_retry_top_p", type=float, default=0.9)
    parser.add_argument(
        "--answer_only_loss",
        type=int,
        default=1,
        help="Compute CE only on non-ignored answer labels. This is equivalent for this SFT format and avoids huge full-sequence CE.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Cannot find records in {path}")


def source_post(record: dict[str, Any]) -> dict[str, Any]:
    post = record.get("source_post")
    return post if isinstance(post, dict) else record


def post_id(record: dict[str, Any]) -> str:
    direct = str(record.get("post_id") or "").strip()
    if direct:
        return direct
    return str(source_post(record).get("id") or "").strip()


def text_parts(record: dict[str, Any], source_texts: dict[str, tuple[str, str]]) -> tuple[str, str, str]:
    post = source_post(record)
    title = str(record.get("title") or post.get("title") or "").strip()
    body = str(record.get("body") or record.get("selftext") or post.get("selftext") or post.get("body") or "").strip()
    text = str(record.get("text") or "").strip()
    if text:
        return title, body, text
    if title or body:
        return title, body, "\n\n".join(part for part in (title, body) if part)
    fallback_title, fallback_body = source_texts.get(post_id(record), ("", ""))
    return fallback_title, fallback_body, "\n\n".join(part for part in (fallback_title, fallback_body) if part)


def load_source_texts(source_file: Path | None) -> dict[str, tuple[str, str]]:
    if source_file is None or not source_file.is_file():
        return {}
    records = records_from_payload(read_json(source_file), source_file)
    texts: dict[str, tuple[str, str]] = {}
    for record in records:
        pid = post_id(record)
        if not pid:
            continue
        post = source_post(record)
        title = str(post.get("title") or record.get("title") or "").strip()
        body = str(post.get("selftext") or post.get("body") or record.get("body") or record.get("selftext") or "").strip()
        texts[pid] = (title, body)
    return texts


def coerce_binary(value: Any) -> int | None:
    if value in (0, 0.0, False):
        return 0
    if value in (1, 1.0, True):
        return 1
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def label_value(record: dict[str, Any], label_key: str) -> int | None:
    labels = record.get("labels")
    if isinstance(labels, dict):
        parsed = coerce_binary(labels.get(label_key))
        if parsed is not None:
            return parsed
    vector = record.get("label_vector")
    if isinstance(vector, list) and label_key in LABEL_KEYS:
        index = LABEL_KEYS.index(label_key)
        if index < len(vector):
            return coerce_binary(vector[index])
    return None


def resolve_records(label_file: Path) -> list[dict[str, Any]]:
    if label_file.exists():
        return records_from_payload(read_json(label_file), label_file)
    part_a = label_file.with_name("updated_400.json")
    part_b = label_file.with_name("new_720_labeled.json")
    if part_a.exists() and part_b.exists():
        return records_from_payload(read_json(part_a), part_a) + records_from_payload(read_json(part_b), part_b)
    raise FileNotFoundError(f"Missing label_file: {label_file}")


def make_prompt(label_key: str, text: str) -> str:
    label_def = LABEL_DEFS[label_key]
    return f"""Decide whether the Reddit post contains this one category.

Category:
{label_def.name_en}

Definition:
{label_def.definition_en}

Decision rule:
{DECISION_RULE_TEXT}

Reddit post:
<post>
{text}
</post>

Answer:"""


def render_prompt_text(tokenizer: Any, label_key: str, text: str) -> str:
    prompt = make_prompt(label_key, text)
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def truncate_post_text(text: str, max_chars: int, strategy: str) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    if strategy == "head":
        return text[:max_chars]
    if strategy == "tail":
        return text[-max_chars:]

    marker = "\n\n[...post truncated...]\n\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head_chars = max(1, remaining // 2)
    tail_chars = max(1, remaining - head_chars)
    return text[:head_chars] + marker + text[-tail_chars:]


def fit_prompt_text(
    tokenizer: Any,
    label_key: str,
    text: str,
    max_prompt_len: int,
    truncation_strategy: str,
) -> tuple[str, list[int], dict[str, Any]]:
    prompt_text = render_prompt_text(tokenizer, label_key, text)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(prompt_ids) <= max_prompt_len:
        return prompt_text, prompt_ids, {
            "was_truncated": False,
            "original_text_chars": len(text),
            "kept_text_chars": len(text),
            "prompt_token_count": len(prompt_ids),
        }

    empty_prompt = render_prompt_text(tokenizer, label_key, "")
    empty_prompt_ids = tokenizer(empty_prompt, add_special_tokens=True, truncation=False)["input_ids"]
    if len(empty_prompt_ids) > max_prompt_len:
        raise ValueError(
            f"The prompt without post text has {len(empty_prompt_ids)} tokens, "
            f"which exceeds max prompt budget {max_prompt_len}"
        )

    low = 0
    high = len(text)
    best_text = ""
    best_prompt_text = empty_prompt
    best_prompt_ids = empty_prompt_ids
    while low <= high:
        mid = (low + high) // 2
        candidate_text = truncate_post_text(text, mid, truncation_strategy)
        candidate_prompt_text = render_prompt_text(tokenizer, label_key, candidate_text)
        candidate_prompt_ids = tokenizer(candidate_prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(candidate_prompt_ids) <= max_prompt_len:
            best_text = candidate_text
            best_prompt_text = candidate_prompt_text
            best_prompt_ids = candidate_prompt_ids
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt_text, best_prompt_ids, {
        "was_truncated": True,
        "original_text_chars": len(text),
        "kept_text_chars": len(best_text),
        "prompt_token_count": len(best_prompt_ids),
    }


def load_examples(label_file: Path, source_file: Path | None, label_key: str) -> tuple[list[Example], dict[str, Any]]:
    records = resolve_records(label_file)
    source_texts = load_source_texts(source_file)
    examples: list[Example] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        status = str(record.get("status") or "").strip()
        if status and status != "complete":
            skipped.append({"index": index, "reason": f"status={status}"})
            continue
        pid = post_id(record)
        value = label_value(record, label_key)
        title, body, text = text_parts(record, source_texts)
        if not pid or not text or value is None:
            skipped.append({"index": index, "post_id": pid, "reason": "missing_post_text_or_label"})
            continue
        if pid in seen:
            skipped.append({"index": index, "post_id": pid, "reason": "duplicate_post_id"})
            continue
        seen.add(pid)
        post = source_post(record)
        examples.append(
            Example(
                post_id=pid,
                subreddit=str(record.get("subreddit") or post.get("subreddit") or "unknown"),
                title=title,
                body=body,
                text=text,
                label=value,
            )
        )
    positives = sum(item.label for item in examples)
    info = {
        "created_at": timestamp(),
        "label_file": str(label_file),
        "source_file": str(source_file) if source_file is not None else "",
        "label_key": label_key,
        "label_name": LABEL_DEFS[label_key].name_en,
        "definition": LABEL_DEFS[label_key].definition_en,
        "raw_records": len(records),
        "loaded_examples": len(examples),
        "positive_count": positives,
        "negative_count": len(examples) - positives,
        "skipped_count": len(skipped),
        "skipped_first_20": skipped[:20],
    }
    return examples, info


def load_fixed_test_post_ids(path: Path | None) -> tuple[set[str], dict[str, Any]]:
    if path is None:
        return set(), {"fixed_split_file": "", "test_post_id_count": 0}
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed_split_file: {path}")
    payload = read_json(path)
    raw_ids = payload.get("test_post_ids") if isinstance(payload, dict) else None
    if not isinstance(raw_ids, list):
        raise ValueError(f"fixed_split_file must contain a test_post_ids list: {path}")
    ids = {str(item).strip() for item in raw_ids if str(item).strip()}
    if len(ids) != len(raw_ids):
        raise ValueError(f"fixed_split_file has duplicate or blank test_post_ids: {path}")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return ids, {
        "fixed_split_file": str(path),
        "test_post_id_count": len(ids),
        "metadata": metadata,
    }


def exclude_fixed_test_examples(
    examples: list[Example],
    fixed_test_post_ids: set[str],
) -> tuple[list[Example], dict[str, Any]]:
    if not fixed_test_post_ids:
        return examples, {
            "fixed_test_post_id_count": 0,
            "excluded_count": 0,
            "remaining_count": len(examples),
            "missing_test_ids_in_label_file_count": 0,
            "missing_test_ids_in_label_file_first_20": [],
            "excluded_first_20": [],
        }
    examples_by_id = {item.post_id: item for item in examples}
    missing = sorted(fixed_test_post_ids - set(examples_by_id))
    excluded = [item for item in examples if item.post_id in fixed_test_post_ids]
    remaining = [item for item in examples if item.post_id not in fixed_test_post_ids]
    return remaining, {
        "fixed_test_post_id_count": len(fixed_test_post_ids),
        "excluded_count": len(excluded),
        "remaining_count": len(remaining),
        "missing_test_ids_in_label_file_count": len(missing),
        "missing_test_ids_in_label_file_first_20": missing[:20],
        "excluded_first_20": [
            {"post_id": item.post_id, "subreddit": item.subreddit, "label": item.label}
            for item in excluded[:20]
        ],
    }


def stratified_split(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example], dict[str, Any]]:
    rng = random.Random(seed)
    positive = [item for item in examples if item.label == 1]
    negative = [item for item in examples if item.label == 0]
    rng.shuffle(positive)
    rng.shuffle(negative)
    total_val = max(1, int(round(len(examples) * val_ratio)))
    pos_val = min(len(positive), int(round(len(positive) * val_ratio)))
    neg_val = max(0, total_val - pos_val)
    if neg_val > len(negative):
        neg_val = len(negative)
        pos_val = min(len(positive), total_val - neg_val)
    val = positive[:pos_val] + negative[:neg_val]
    train = positive[pos_val:] + negative[neg_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    split = {
        "split_seed": seed,
        "val_ratio": val_ratio,
        "train_size": len(train),
        "val_size": len(val),
        "positive_total": len(positive),
        "negative_total": len(negative),
        "positive_train": sum(item.label for item in train),
        "positive_val": sum(item.label for item in val),
    }
    return train, val, split


def encode_example(example: Example, tokenizer: Any, label_key: str, max_length: int, truncation_strategy: str) -> EncodedExample:
    output_ids = tokenizer(str(example.label), add_special_tokens=False, truncation=False)["input_ids"]
    if not output_ids:
        raise ValueError("Tokenizer produced an empty output target")
    max_prompt_len = max_length - len(output_ids)
    if max_prompt_len < 1:
        raise ValueError(f"max_length={max_length} is too small")
    prompt_text, prompt_ids, truncation_info = fit_prompt_text(tokenizer, label_key, example.text, max_prompt_len, truncation_strategy)
    input_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids
    return EncodedExample(
        example=example,
        prompt_text=prompt_text,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        prompt_len=len(prompt_ids),
        was_truncated=bool(truncation_info["was_truncated"]),
        original_text_chars=int(truncation_info["original_text_chars"]),
        kept_text_chars=int(truncation_info["kept_text_chars"]),
        prompt_token_count=int(truncation_info["prompt_token_count"]),
    )


def encode_all(examples: list[Example], tokenizer: Any, label_key: str, max_length: int, truncation_strategy: str) -> list[EncodedExample]:
    return [encode_example(item, tokenizer, label_key, max_length, truncation_strategy) for item in examples]


def tokenizer_model_vocab_info(tokenizer: Any, model: Any) -> dict[str, Any]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_vocab_size = int(getattr(input_embeddings, "num_embeddings", 0) or 0)
    output_vocab_size = 0
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_vocab_size = int(output_embeddings.weight.shape[0])
    config = getattr(model, "config", None)
    config_vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    return {
        "tokenizer_length": int(len(tokenizer)),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "input_embedding_vocab_size": input_vocab_size,
        "output_embedding_vocab_size": output_vocab_size,
        "config_vocab_size": config_vocab_size,
    }


def validate_encoded_ids(
    rows: list[EncodedExample],
    split_name: str,
    vocab_info: dict[str, Any],
    target_token_ids: dict[str, int],
) -> dict[str, Any]:
    input_limit = int(vocab_info["input_embedding_vocab_size"] or vocab_info["config_vocab_size"])
    label_limit = int(vocab_info["output_embedding_vocab_size"] or vocab_info["config_vocab_size"])
    if input_limit <= 0 or label_limit <= 0:
        raise ValueError(f"Could not determine model vocab sizes: {vocab_info}")

    invalid_inputs: list[dict[str, Any]] = []
    invalid_labels: list[dict[str, Any]] = []
    observed_label_ids: set[int] = set()
    max_input_id = -1
    max_label_id = -1
    min_label_id = None

    for index, item in enumerate(rows):
        item_max_input = max(item.input_ids) if item.input_ids else -1
        max_input_id = max(max_input_id, item_max_input)
        bad_inputs = sorted({token_id for token_id in item.input_ids if token_id < 0 or token_id >= input_limit})
        target_ids = [token_id for token_id in item.labels if token_id != -100]
        if target_ids:
            observed_label_ids.update(int(token_id) for token_id in target_ids)
            item_max_label = max(target_ids)
            item_min_label = min(target_ids)
            max_label_id = max(max_label_id, item_max_label)
            min_label_id = item_min_label if min_label_id is None else min(min_label_id, item_min_label)
        bad_labels = sorted({token_id for token_id in target_ids if token_id < 0 or token_id >= label_limit})
        if bad_inputs and len(invalid_inputs) < 20:
            invalid_inputs.append(
                {
                    "split": split_name,
                    "row_index": index,
                    "post_id": item.example.post_id,
                    "max_input_id": item_max_input,
                    "bad_input_ids": bad_inputs[:20],
                    "input_vocab_limit": input_limit,
                }
            )
        if bad_labels and len(invalid_labels) < 20:
            invalid_labels.append(
                {
                    "split": split_name,
                    "row_index": index,
                    "post_id": item.example.post_id,
                    "target_ids": target_ids,
                    "bad_label_ids": bad_labels[:20],
                    "label_vocab_limit": label_limit,
                }
            )

    expected_label_ids = sorted(int(value) for value in target_token_ids.values())
    unexpected_label_ids = sorted(observed_label_ids - set(expected_label_ids))
    info = {
        "split": split_name,
        "row_count": len(rows),
        "input_vocab_limit": input_limit,
        "label_vocab_limit": label_limit,
        "max_input_id": max_input_id,
        "min_label_id": min_label_id,
        "max_label_id": max_label_id,
        "expected_label_ids": expected_label_ids,
        "observed_label_ids": sorted(observed_label_ids),
        "unexpected_label_ids": unexpected_label_ids,
        "invalid_input_count_first_20": len(invalid_inputs),
        "invalid_label_count_first_20": len(invalid_labels),
        "invalid_inputs_first_20": invalid_inputs,
        "invalid_labels_first_20": invalid_labels,
    }
    if invalid_inputs or invalid_labels or unexpected_label_ids:
        raise ValueError("Encoded id validation failed before training:\n" + json.dumps(info, ensure_ascii=False, indent=2))
    return info


def truncation_summary(encoded: list[EncodedExample]) -> dict[str, Any]:
    truncated = [item for item in encoded if item.was_truncated]
    return {
        "count": len(encoded),
        "truncated_count": len(truncated),
        "truncated_rate": len(truncated) / len(encoded) if encoded else 0.0,
        "truncated_first_20": [
            {
                "post_id": item.example.post_id,
                "original_text_chars": item.original_text_chars,
                "kept_text_chars": item.kept_text_chars,
                "prompt_token_count": item.prompt_token_count,
            }
            for item in truncated[:20]
        ],
    }


def collate(batch: list[EncodedExample], tokenizer: Any, torch: Any) -> dict[str, Any]:
    max_len = max(len(item.input_ids) for item in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_id] * pad_len)
        attention_mask.append(item.attention_mask + [0] * pad_len)
        labels.append(item.labels + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def import_training_stack():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    return torch, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, LoraConfig, TaskType, get_peft_model


def normalize_precision_args(args: argparse.Namespace, torch: Any) -> None:
    if bool(args.bf16):
        bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if not bf16_supported:
            print("bf16 requested but unavailable; falling back to fp16.", flush=True)
            args.bf16 = 0
            if not bool(args.fp16):
                args.fp16 = 1


def resolved_device_map(args: argparse.Namespace, torch: Any) -> str | dict[str, int] | None:
    if args.device_map == "none":
        return None
    if args.device_map == "local_rank":
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and torch.cuda.is_available():
            rank = int(local_rank)
            torch.cuda.set_device(rank)
            return {"": rank}
        return "auto"
    return "auto"


def build_model(args: argparse.Namespace, torch: Any, AutoModelForCausalLM: Any, AutoTokenizer: Any, LoraConfig: Any, TaskType: Any, get_peft_model: Any):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if bool(args.fp16) else torch.float32
    model_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    device_map = resolved_device_map(args, torch)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if bool(args.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return tokenizer, model


def binary_target_token_ids(tokenizer: Any) -> dict[str, int]:
    zero_ids = tokenizer("0", add_special_tokens=False, truncation=False)["input_ids"]
    one_ids = tokenizer("1", add_special_tokens=False, truncation=False)["input_ids"]
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(f"Expected 0 and 1 to each be one token, got {zero_ids} and {one_ids}")
    return {"0": int(zero_ids[0]), "1": int(one_ids[0])}


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            candidates.append((int(child.name.split("-")[-1]), child))
        except ValueError:
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def training_args_kwargs_for_version(TrainingArguments: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    if "warmup_ratio" not in params and "warmup_steps" in params:
        kwargs["warmup_steps"] = 0

    supported = set(params) - {"self"}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    required = {
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    }
    missing_required = sorted(required - set(filtered))
    if missing_required:
        raise TypeError(
            "This transformers TrainingArguments version is too incompatible; "
            f"missing required parameters: {', '.join(missing_required)}"
        )

    skipped = sorted(set(kwargs) - set(filtered))
    if skipped:
        print(
            "Skipping unsupported TrainingArguments keys for this transformers version: "
            + ", ".join(skipped),
            flush=True,
        )
    return filtered


def make_training_args(TrainingArguments: Any, args: argparse.Namespace, output_dir: Path) -> Any:
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    if bool(args.bf16):
        kwargs["bf16"] = True
    elif bool(args.fp16):
        kwargs["fp16"] = True
    kwargs = training_args_kwargs_for_version(TrainingArguments, kwargs)
    print("TrainingArguments kwargs: " + ", ".join(sorted(kwargs)), flush=True)
    return TrainingArguments(**kwargs)


def make_answer_only_trainer_class(Trainer: Any, torch: Any) -> Any:
    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            labels = inputs.pop("labels")
            model_inputs = dict(inputs)
            used_logits_to_keep = False
            if labels.shape[0] == 1:
                # Only the final answer token has a real label. Qwen-style models
                # can often avoid materializing logits for every prompt token.
                model_inputs["logits_to_keep"] = 2
                used_logits_to_keep = True
            try:
                outputs = model(**model_inputs)
            except TypeError:
                if not used_logits_to_keep:
                    raise
                model_inputs.pop("logits_to_keep", None)
                outputs = model(**model_inputs)

            logits = outputs.logits
            vocab_size = int(logits.shape[-1])
            labels = labels.to(logits.device)
            padded_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
            if logits.shape[1] == labels.shape[1]:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
            else:
                kept_len = int(logits.shape[1])
                start = int(labels.shape[1]) - kept_len
                if start < 0:
                    raise ValueError(
                        f"Cannot align kept logits with labels: logits_len={kept_len}, labels_len={labels.shape[1]}"
                    )
                shift_logits = logits
                shift_labels = padded_labels[:, start + 1 : start + 1 + kept_len]

            active = shift_labels.ne(-100)
            active_labels = shift_labels[active]
            if active_labels.numel() == 0:
                raise ValueError("No answer labels remained after shifting; cannot compute SFT loss.")
            min_label = int(active_labels.min().item())
            max_label = int(active_labels.max().item())
            if min_label < 0 or max_label >= vocab_size:
                raise ValueError(
                    f"Runtime label id out of range before CE: min_label={min_label}, "
                    f"max_label={max_label}, vocab_size={vocab_size}"
                )

            active_logits = shift_logits[active].float()
            loss = torch.nn.functional.cross_entropy(active_logits, active_labels.to(active_logits.device))
            return (loss, outputs) if return_outputs else loss

    return AnswerOnlyTrainer


def predict_once(
    model: Any,
    tokenizer: Any,
    torch: Any,
    item: EncodedExample,
    max_new_tokens: int,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    target_token_ids: dict[str, int],
) -> tuple[int | None, str, int | None, str]:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([item.input_ids[: item.prompt_len]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    generated_ids = output_ids[0][item.prompt_len :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_token_id = int(generated_ids[0]) if len(generated_ids) else None
    if first_token_id == target_token_ids["0"]:
        return 0, text, first_token_id, tokenizer.decode([first_token_id], skip_special_tokens=False)
    if first_token_id == target_token_ids["1"]:
        return 1, text, first_token_id, tokenizer.decode([first_token_id], skip_special_tokens=False)
    first_token_text = "" if first_token_id is None else tokenizer.decode([first_token_id], skip_special_tokens=False)
    return None, text, first_token_id, first_token_text


def predict(
    model: Any,
    tokenizer: Any,
    torch: Any,
    item: EncodedExample,
    max_new_tokens: int,
    invalid_attempts: int,
    invalid_retry_temperature: float,
    invalid_retry_top_p: float,
    target_token_ids: dict[str, int],
) -> dict[str, Any]:
    attempts = max(1, invalid_attempts)
    raw_generations = []
    first_generated_token_ids = []
    first_generated_token_texts = []
    for attempt_index in range(attempts):
        pred, raw, first_token_id, first_token_text = predict_once(
            model,
            tokenizer,
            torch,
            item,
            max_new_tokens,
            do_sample=attempt_index > 0,
            temperature=invalid_retry_temperature,
            top_p=invalid_retry_top_p,
            target_token_ids=target_token_ids,
        )
        raw_generations.append(raw)
        first_generated_token_ids.append(first_token_id)
        first_generated_token_texts.append(first_token_text)
        if pred is not None:
            return {
                "prediction": pred,
                "raw_generation": raw,
                "raw_generations": raw_generations,
                "first_generated_token_id": first_token_id,
                "first_generated_token_text": first_token_text,
                "first_generated_token_ids": first_generated_token_ids,
                "first_generated_token_texts": first_generated_token_texts,
                "attempts_used": attempt_index + 1,
                "is_invalid": False,
                "recovered_from_invalid": attempt_index > 0,
            }
    return {
        "prediction": None,
        "raw_generation": raw_generations[-1] if raw_generations else "",
        "raw_generations": raw_generations,
        "first_generated_token_id": first_generated_token_ids[-1] if first_generated_token_ids else None,
        "first_generated_token_text": first_generated_token_texts[-1] if first_generated_token_texts else "",
        "first_generated_token_ids": first_generated_token_ids,
        "first_generated_token_texts": first_generated_token_texts,
        "attempts_used": attempts,
        "is_invalid": True,
        "recovered_from_invalid": False,
    }


def evaluate(
    model: Any,
    tokenizer: Any,
    torch: Any,
    encoded: list[EncodedExample],
    max_new_tokens: int,
    invalid_attempts: int,
    invalid_retry_temperature: float,
    invalid_retry_top_p: float,
    target_token_ids: dict[str, int],
) -> dict[str, Any]:
    model.eval()
    rows = []
    y_true = []
    y_pred = []
    invalid_positive = 0
    invalid_negative = 0
    for item in encoded:
        prediction = predict(
            model,
            tokenizer,
            torch,
            item,
            max_new_tokens,
            invalid_attempts,
            invalid_retry_temperature,
            invalid_retry_top_p,
            target_token_ids,
        )
        parsed = prediction["prediction"]
        y_true.append(item.example.label)
        y_pred.append(parsed)
        if parsed is None:
            if item.example.label == 1:
                invalid_positive += 1
            else:
                invalid_negative += 1
        rows.append(
            {
                "post_id": item.example.post_id,
                "subreddit": item.example.subreddit,
                "label": item.example.label,
                "prediction": parsed,
                "is_invalid": prediction["is_invalid"],
                "attempts_used": prediction["attempts_used"],
                "recovered_from_invalid": prediction["recovered_from_invalid"],
                "raw_generation": prediction["raw_generation"],
                "raw_generations": prediction["raw_generations"],
                "first_generated_token_id": prediction["first_generated_token_id"],
                "first_generated_token_text": prediction["first_generated_token_text"],
                "first_generated_token_ids": prediction["first_generated_token_ids"],
                "first_generated_token_texts": prediction["first_generated_token_texts"],
            }
        )
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    invalid_count = invalid_positive + invalid_negative
    valid_count = len(y_true) - invalid_count
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    valid_accuracy = (tp + tn) / valid_count if valid_count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn + invalid_positive) if (tp + fn + invalid_positive) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    recovered_invalid_count = sum(1 for row in rows if row["recovered_from_invalid"])
    return {
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(y_true) if y_true else 0.0,
        "invalid_positive": invalid_positive,
        "invalid_negative": invalid_negative,
        "recovered_invalid_count": recovered_invalid_count,
        "recovered_invalid_rate": recovered_invalid_count / len(y_true) if y_true else 0.0,
        "predictions": rows,
    }


def save_example_rows(path: Path, rows: list[EncodedExample], label_key: str) -> None:
    write_json(
        path,
        [
            {
                "post_id": item.example.post_id,
                "subreddit": item.example.subreddit,
                "label_key": label_key,
                "label": item.example.label,
                "title": item.example.title,
                "body": item.example.body,
                "was_truncated": item.was_truncated,
                "original_text_chars": item.original_text_chars,
                "kept_text_chars": item.kept_text_chars,
                "prompt_token_count": item.prompt_token_count,
                "prompt": item.prompt_text,
            }
            for item in rows
        ],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, LoraConfig, TaskType, get_peft_model = import_training_stack()
    normalize_precision_args(args, torch)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples, load_info = load_examples(Path(args.label_file), Path(args.source_file) if args.source_file else None, args.label_key)
    if not examples:
        raise SystemExit(f"No usable examples for label_key={args.label_key}")
    fixed_split_path = Path(args.fixed_split_file) if args.fixed_split_file else None
    fixed_test_post_ids, fixed_split_info = load_fixed_test_post_ids(fixed_split_path)
    examples, fixed_test_exclusion_info = exclude_fixed_test_examples(examples, fixed_test_post_ids)
    if not examples:
        raise SystemExit(f"No training examples remain after fixed test exclusion for label_key={args.label_key}")
    train_examples, val_examples, split_info = stratified_split(examples, args.val_ratio, args.split_seed)
    split_info["fixed_test_exclusion"] = fixed_test_exclusion_info
    split_info["fixed_split"] = fixed_split_info

    tokenizer, model = build_model(args, torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, TaskType, get_peft_model)
    target_token_ids = binary_target_token_ids(tokenizer)
    train_encoded = encode_all(train_examples, tokenizer, args.label_key, args.max_length, args.post_truncation_strategy)
    val_encoded = encode_all(val_examples, tokenizer, args.label_key, args.max_length, args.post_truncation_strategy)
    vocab_info = tokenizer_model_vocab_info(tokenizer, model)
    encoded_id_validation = {
        "vocab_info": vocab_info,
        "train": validate_encoded_ids(train_encoded, "train", vocab_info, target_token_ids),
        "val": validate_encoded_ids(val_encoded, "val", vocab_info, target_token_ids),
    }
    print("Tokenizer/model vocab info: " + json.dumps(vocab_info, ensure_ascii=False, sort_keys=True), flush=True)
    print(
        "Encoded id validation passed: "
        + json.dumps(
            {
                "train_max_input_id": encoded_id_validation["train"]["max_input_id"],
                "train_label_ids": encoded_id_validation["train"]["observed_label_ids"],
                "val_max_input_id": encoded_id_validation["val"]["max_input_id"],
                "val_label_ids": encoded_id_validation["val"]["observed_label_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    truncation_info = {
        "max_length": args.max_length,
        "post_truncation_strategy": args.post_truncation_strategy,
        "train": truncation_summary(train_encoded),
        "val": truncation_summary(val_encoded),
    }

    save_example_rows(output_dir / "train_examples.json", train_encoded, args.label_key)
    save_example_rows(output_dir / "val_examples.json", val_encoded, args.label_key)
    write_json(output_dir / "load_info.json", load_info)
    write_json(output_dir / "fixed_split_info.json", fixed_split_info)
    write_json(output_dir / "fixed_test_exclusion_info.json", fixed_test_exclusion_info)
    write_json(output_dir / "split_info.json", split_info)
    write_json(output_dir / "truncation_info.json", truncation_info)
    write_json(output_dir / "binary_target_token_ids.json", target_token_ids)
    write_json(output_dir / "encoded_id_validation.json", encoded_id_validation)

    trainer_class = make_answer_only_trainer_class(Trainer, torch) if bool(args.answer_only_loss) else Trainer
    print(f"Answer-only loss: {int(bool(args.answer_only_loss))}", flush=True)
    trainer = trainer_class(
        model=model,
        args=make_training_args(TrainingArguments, args, output_dir),
        train_dataset=ListDataset(train_encoded),
        eval_dataset=ListDataset(val_encoded),
        data_collator=lambda batch: collate(batch, tokenizer, torch),
    )

    checkpoint = latest_checkpoint(output_dir) if args.resume else None
    if checkpoint:
        print(f"Resuming from {checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    val_result = evaluate(
        model,
        tokenizer,
        torch,
        val_encoded,
        args.max_new_tokens,
        args.invalid_attempts,
        args.invalid_retry_temperature,
        args.invalid_retry_top_p,
        target_token_ids,
    )
    metrics = {
        "created_at": timestamp(),
        "label_key": args.label_key,
        "label_name": LABEL_DEFS[args.label_key].name_en,
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "binary_target_token_ids": target_token_ids,
        "load_info": load_info,
        "fixed_split_info": fixed_split_info,
        "fixed_test_exclusion_info": fixed_test_exclusion_info,
        "split_info": split_info,
        "truncation_info": truncation_info,
        "val_metrics": {key: value for key, value in val_result.items() if key != "predictions"},
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "val_predictions.json", val_result["predictions"])
    write_csv(
        output_dir / "val_predictions.csv",
        val_result["predictions"],
        [
            "post_id",
            "subreddit",
            "label",
            "prediction",
            "is_invalid",
            "attempts_used",
            "recovered_from_invalid",
            "raw_generation",
            "raw_generations",
            "first_generated_token_id",
            "first_generated_token_text",
            "first_generated_token_ids",
            "first_generated_token_texts",
        ],
    )

    print(f"Finished label_key={args.label_key}", flush=True)
    print(f"Train/val={len(train_examples)}/{len(val_examples)}", flush=True)
    print(
        f"Fixed test excluded={fixed_test_exclusion_info['excluded_count']} "
        f"remaining_pool={fixed_test_exclusion_info['remaining_count']}",
        flush=True,
    )
    print(
        f"Val accuracy={val_result['accuracy']:.4f} F1={val_result['f1']:.4f} "
        f"invalid={val_result['invalid_count']}",
        flush=True,
    )
    print(f"Saved final adapter to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
