#!/usr/bin/env python3
"""Fresh no-gold eight-classifier 1053 Qwen attention/hidden extraction.

This implementation is kept with the new classifier launcher and intentionally
does not import the existing eval/cache helpers. It may import
``annotation/post_labels.py`` for the canonical label definitions. All vector
extraction, ranking, label-overlap metrics, and summary writing are implemented
here.

The ``predicted`` answer mode first gives each label adapter an answer-less
prompt and selects its own 0/1 token from the two answer logits. That token is
then appended solely to obtain the answer-query attention row. Dataset labels
are never used to construct the prompt in this mode; they are only available to
the separate post-hoc retrieval evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FRESHNEW_DIR = PROJECT_DIR / "annotation"
if str(FRESHNEW_DIR) not in sys.path:
    sys.path.insert(0, str(FRESHNEW_DIR))

from post_labels import DECISION_RULE_TEXT, POST_LABELS  # noqa: E402


LABEL_DEFS = {item.key: item for item in POST_LABELS}
LABEL_KEYS = [item.key for item in POST_LABELS]

SYSTEM_PROMPT = """You are a careful research annotator.
Your task is to judge only one category for one Reddit post.
Use only the supplied category name, category definition, and Reddit post.
Consider the title and body together, including negation, quoted speech, sarcasm, and who is experiencing the event or emotion.
Return exactly one word: 0 or 1.
Do not return explanations, punctuation, bullet points, JSON, Markdown, or any other text."""

METHOD_DEFINITIONS = {
    "qwen_base_hidden_post_mean": {
        "family": "base_post_hidden",
        "core": "Mean-pool base Qwen final hidden states over post-only tokens, no label prompt and no LoRA adapter contribution.",
        "score": "Cosine similarity between one vector per post.",
    },
    "qwen_base_hidden_adapter_attention_mean_label": {
        "family": "base_hidden_adapter_attention",
        "core": "For each label prompt, let that LoRA classifier choose its own 0/1 token, append that predicted token, use its answer-token attention over post tokens, map those weights onto base-model post-only hidden states, then average label-wise cosine scores.",
        "score": "Mean of the eight per-label cosine similarity matrices.",
    },
    "qwen_base_hidden_delta_ratio_attention_mean_label": {
        "family": "base_hidden_delta_ratio_attention",
        "core": "For each label prompt, append the classifier's own predicted 0/1 token, emphasize tokens whose normalized adapter answer-token attention increases versus normalized base answer-token attention, apply those delta-ratio weights to base-model post-only hidden states, then average label-wise cosine scores.",
        "score": "Mean of the eight per-label cosine similarity matrices.",
    },
    "qwen_hidden_attention_mean_label": {
        "family": "adapter_hidden_attention",
        "core": "For each label prompt, append the classifier's own predicted 0/1 token, use LoRA-adapter hidden states weighted by the same prompt's LoRA-adapter answer-token attention over post tokens, then average label-wise cosine scores.",
        "score": "Mean of the eight per-label cosine similarity matrices.",
    },
    "qwen_delta_ratio_attention_mean_label": {
        "family": "adapter_hidden_delta_ratio_attention",
        "core": "For each label prompt, append the classifier's own predicted 0/1 token, emphasize tokens whose normalized adapter answer-token attention increases versus normalized base answer-token attention, apply those weights to LoRA-adapter hidden states, then average label-wise cosine scores.",
        "score": "Mean of the eight per-label cosine similarity matrices.",
    },
    "qwen_hidden_attention_change_rate_mean_label": {
        "family": "adapter_hidden_attention_change_rate",
        "core": "For each label prompt, append the classifier's own predicted 0/1 token, weight LoRA-adapter hidden states by clipped raw answer-token attention change-rate, adapter attention divided by base attention, then average label-wise cosine scores.",
        "score": "Mean of the eight per-label cosine similarity matrices.",
    },
}

VECTOR_FAMILY_TO_METHOD = {
    "base_hidden_post_mean": "qwen_base_hidden_post_mean",
    "base_hidden_adapter_attention": "qwen_base_hidden_adapter_attention_mean_label",
    "base_hidden_delta_ratio_attention": "qwen_base_hidden_delta_ratio_attention_mean_label",
    "adapter_hidden_attention": "qwen_hidden_attention_mean_label",
    "adapter_hidden_delta_ratio_attention": "qwen_delta_ratio_attention_mean_label",
    "adapter_hidden_attention_change_rate": "qwen_hidden_attention_change_rate_mean_label",
}

VECTOR_KEYS = [
    "base_hidden_post_mean",
    "base_hidden_adapter_attention",
    "base_hidden_delta_ratio_attention",
    "adapter_hidden_attention",
    "adapter_hidden_delta_ratio_attention",
    "adapter_hidden_attention_change_rate",
]

# The historical evaluator cleaned non-finite values so that a run could
# continue, which can silently poison attention caches.  This deployment
# evaluator fails before writing vectors.
FAIL_ON_NONFINITE = False


def require_finite_numpy(np: Any, values: Any, name: str) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        count = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name} contains {count} non-finite value(s); refusing to continue")


@dataclass
class PostExample:
    post_id: str
    subreddit: str
    title: str
    body: str
    text: str
    labels: dict[str, int]


@dataclass
class PromptEncoding:
    prompt_text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    post_char_span: tuple[int, int]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Cannot find records in {path}")


def source_post(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("source_post")
    return value if isinstance(value, dict) else record


def post_id_from_record(record: dict[str, Any]) -> str:
    direct = str(record.get("post_id") or "").strip()
    if direct:
        return direct
    return str(source_post(record).get("id") or "").strip()


def coerce_binary(value: Any) -> int | None:
    if value in (0, 0.0, False):
        return 0
    if value in (1, 1.0, True):
        return 1
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def label_value_from_record(record: dict[str, Any], label_key: str) -> int | None:
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


def load_source_texts(source_file: Path | None) -> dict[str, tuple[str, str]]:
    if source_file is None or not source_file.exists():
        return {}
    records = records_from_payload(read_json(source_file), source_file)
    texts: dict[str, tuple[str, str]] = {}
    for record in records:
        pid = post_id_from_record(record)
        if not pid:
            continue
        post = source_post(record)
        title = str(post.get("title") or record.get("title") or "").strip()
        body = str(post.get("selftext") or post.get("body") or record.get("body") or record.get("selftext") or "").strip()
        texts[pid] = (title, body)
    return texts


def text_parts(record: dict[str, Any], source_texts: dict[str, tuple[str, str]]) -> tuple[str, str, str]:
    post = source_post(record)
    title = str(record.get("title") or post.get("title") or "").strip()
    body = str(record.get("body") or record.get("selftext") or post.get("selftext") or post.get("body") or "").strip()
    direct_text = str(record.get("text") or "").strip()
    if direct_text:
        return title, body, direct_text
    if title or body:
        return title, body, "\n\n".join(part for part in (title, body) if part)
    fallback_title, fallback_body = source_texts.get(post_id_from_record(record), ("", ""))
    return fallback_title, fallback_body, "\n\n".join(part for part in (fallback_title, fallback_body) if part)


def parse_label_keys(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(LABEL_KEYS)
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("--label-keys resolved to an empty list")
    bad = [key for key in keys if key not in LABEL_KEYS]
    if bad:
        raise ValueError(f"Unknown label key(s): {bad}")
    return keys


def load_examples(
    data_file: Path,
    source_file: Path | None,
    label_keys: list[str],
    record_start: int = 0,
    max_records: int = 0,
) -> tuple[list[PostExample], dict[str, Any]]:
    records = records_from_payload(read_json(data_file), data_file)
    if record_start > 0:
        records = records[record_start:]
    if max_records > 0:
        records = records[:max_records]
    source_texts = load_source_texts(source_file)
    examples: list[PostExample] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        status = str(record.get("status") or "").strip()
        if status and status != "complete":
            skipped.append({"index": index, "reason": f"status={status}"})
            continue
        pid = post_id_from_record(record)
        title, body, text = text_parts(record, source_texts)
        if not pid or not text:
            skipped.append({"index": index, "post_id": pid, "reason": "missing_post_or_text"})
            continue
        if pid in seen:
            skipped.append({"index": index, "post_id": pid, "reason": "duplicate_post_id"})
            continue
        labels: dict[str, int] = {}
        missing_labels = []
        for label_key in label_keys:
            value = label_value_from_record(record, label_key)
            if value is None:
                missing_labels.append(label_key)
            else:
                labels[label_key] = value
        if missing_labels:
            skipped.append({"index": index, "post_id": pid, "reason": "missing_labels", "labels": missing_labels})
            continue
        seen.add(pid)
        post = source_post(record)
        examples.append(
            PostExample(
                post_id=pid,
                subreddit=str(record.get("subreddit") or post.get("subreddit") or "unknown"),
                title=title,
                body=body,
                text=text,
                labels=labels,
            )
        )
    info = {
        "created_at": timestamp(),
        "data_file": str(data_file),
        "source_file": str(source_file) if source_file is not None else "",
        "record_start": record_start,
        "max_records": max_records,
        "loaded_examples": len(examples),
        "skipped_count": len(skipped),
        "skipped_first_20": skipped[:20],
        "label_keys": label_keys,
    }
    return examples, info


def load_unlabeled_examples(
    data_file: Path,
    record_start: int = 0,
    max_records: int = 0,
) -> tuple[list[PostExample], dict[str, Any]]:
    """Load only post metadata for inference; never read a ``labels`` field."""

    records = records_from_payload(read_json(data_file), data_file)
    if record_start > 0:
        records = records[record_start:]
    if max_records > 0:
        records = records[:max_records]
    examples: list[PostExample] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        # Build a deliberately narrow view before any downstream helper sees a
        # record.  In particular, ``labels``, ``label_vector``, majority-vote
        # fields, and all annotation metadata are absent from the inference
        # object—not merely ignored later by convention.
        source = record.get("source_post")
        source_view = source if isinstance(source, dict) else {}
        inference_record = {
            "status": record.get("status"),
            "post_id": record.get("post_id"),
            "subreddit": record.get("subreddit"),
            "title": record.get("title"),
            "body": record.get("body"),
            "text": record.get("text"),
            "source_post": {
                "id": source_view.get("id"),
                "subreddit": source_view.get("subreddit"),
                "title": source_view.get("title"),
                "selftext": source_view.get("selftext"),
                "body": source_view.get("body"),
            },
        }
        record = inference_record
        status = str(record.get("status") or "").strip()
        if status and status != "complete":
            skipped.append({"index": index, "reason": f"status={status}"})
            continue
        pid = post_id_from_record(record)
        title, body, text = text_parts(record, {})
        if not pid or not text:
            skipped.append({"index": index, "post_id": pid, "reason": "missing_post_or_text"})
            continue
        if pid in seen:
            skipped.append({"index": index, "post_id": pid, "reason": "duplicate_post_id"})
            continue
        seen.add(pid)
        post = source_post(record)
        examples.append(
            PostExample(
                post_id=pid,
                subreddit=str(record.get("subreddit") or post.get("subreddit") or "unknown"),
                title=title,
                body=body,
                text=text,
                labels={},
            )
        )
    info = {
        "created_at": timestamp(),
        "data_file": str(data_file),
        "record_start": record_start,
        "max_records": max_records,
        "loaded_examples": len(examples),
        "skipped_count": len(skipped),
        "skipped_first_20": skipped[:20],
        "labels_loaded_for_inference": False,
    }
    return examples, info


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


def build_prompt_text(tokenizer: Any, label_key: str, text: str) -> str:
    prompt = make_prompt(label_key, text)
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            # Match the prompt used by the new SFT script; Qwen3 otherwise may
            # insert a thinking block and shift the answer-token position.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer versions do not expose enable_thinking.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def find_post_span(prompt_text: str) -> tuple[int, int]:
    open_index = prompt_text.find("<post>")
    close_index = prompt_text.rfind("\n</post>")
    answer_index = prompt_text.rfind("Answer:")
    if open_index < 0 or close_index < 0:
        raise ValueError("Could not find <post>...</post> markers in rendered prompt")
    if answer_index < 0 or answer_index < close_index:
        raise ValueError("Could not find trailing Answer: marker after </post>")
    content_start = prompt_text.find("\n", open_index)
    if content_start < 0 or content_start >= close_index:
        raise ValueError("Malformed <post> marker block")
    return content_start + 1, close_index


def unbatched_token_list(values: Any, field_name: str = "input_ids") -> list[int]:
    """Normalize HF BatchEncoding/tensor/list output for one input string."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ValueError(f"Tokenizer did not return {field_name} as a list")
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError(f"Expected one tokenizer row, got {len(values)}")
        values = list(values[0])
    if any(isinstance(value, (list, tuple, dict)) for value in values):
        raise ValueError(f"Tokenizer returned nested {field_name}")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tokenizer {field_name} contains a non-integer value") from exc


def encode_prompt(tokenizer: Any, label_key: str, text: str, max_qwen_length: int, answer_text: str | None = None) -> PromptEncoding:
    prompt_text = build_prompt_text(tokenizer, label_key, text)
    post_span = find_post_span(prompt_text)
    encoded = tokenizer(prompt_text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    input_ids = unbatched_token_list(encoded["input_ids"])
    raw_offsets = encoded["offset_mapping"]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    if raw_offsets and isinstance(raw_offsets[0], (list, tuple)) and raw_offsets[0] and isinstance(raw_offsets[0][0], (list, tuple)):
        if len(raw_offsets) != 1:
            raise ValueError(f"Expected one tokenizer offset row, got {len(raw_offsets)}")
        raw_offsets = raw_offsets[0]
    offsets = [tuple(item) for item in raw_offsets]
    if answer_text is not None:
        answer_ids = unbatched_token_list(
            tokenizer(str(answer_text), add_special_tokens=False, truncation=False)["input_ids"]
        )
        if not answer_ids:
            raise ValueError(f"Tokenizer produced no ids for answer_text={answer_text!r}")
        answer_start = len(prompt_text)
        answer_end = answer_start + len(str(answer_text))
        input_ids.extend(int(token_id) for token_id in answer_ids)
        offsets.extend((answer_start, answer_end) for _ in answer_ids)
    if len(input_ids) != len(offsets):
        raise ValueError("Qwen token ids and offsets have different lengths")
    if len(input_ids) > max_qwen_length:
        raise ValueError(f"Prompt has {len(input_ids)} Qwen tokens, over --max-qwen-length={max_qwen_length}")
    return PromptEncoding(prompt_text=prompt_text, input_ids=input_ids, offsets=offsets, post_char_span=post_span)


def answer_text_for_mode(mode: str, example: PostExample, label_key: str) -> str | None:
    normalized = mode.strip().lower()
    if normalized == "none":
        return None
    if normalized == "gold":
        return str(int(example.labels[label_key]))
    if normalized in {"0", "zero"}:
        return "0"
    if normalized in {"1", "one"}:
        return "1"
    raise ValueError(f"Unknown attention answer mode: {mode!r}")


def encode_post_only(tokenizer: Any, text: str, max_qwen_length: int) -> PromptEncoding:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    input_ids = unbatched_token_list(encoded["input_ids"])
    raw_offsets = encoded["offset_mapping"]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    if raw_offsets and isinstance(raw_offsets[0], (list, tuple)) and raw_offsets[0] and isinstance(raw_offsets[0][0], (list, tuple)):
        if len(raw_offsets) != 1:
            raise ValueError(f"Expected one tokenizer offset row, got {len(raw_offsets)}")
        raw_offsets = raw_offsets[0]
    offsets = [tuple(item) for item in raw_offsets]
    if len(input_ids) != len(offsets):
        raise ValueError("Qwen post-only token ids and offsets have different lengths")
    if len(input_ids) > max_qwen_length:
        raise ValueError(f"Post has {len(input_ids)} Qwen tokens, over --max-qwen-length={max_qwen_length}")
    return PromptEncoding(prompt_text=text, input_ids=input_ids, offsets=offsets, post_char_span=(0, len(text)))


def post_token_indices(encoding: PromptEncoding) -> list[int]:
    post_start, post_end = encoding.post_char_span
    indices: list[int] = []
    for token_index, (start_raw, end_raw) in enumerate(encoding.offsets):
        start = int(start_raw)
        end = int(end_raw)
        if end <= start:
            continue
        if min(end, post_end) <= max(start, post_start):
            continue
        indices.append(token_index)
    if not indices:
        raise ValueError("No post-overlapping Qwen tokens were found")
    return indices


def import_runtime_stack() -> tuple[Any, Any, Any, Any]:
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return np, torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def adapter_path(adapter_root: Path, adapter_subdir: str, label_key: str) -> Path:
    label_root = adapter_root / label_key
    candidates: list[Path] = []
    subdir = adapter_subdir.strip()
    if subdir and subdir != ".":
        candidates.append(label_root / subdir)
    candidates.append(label_root)
    if label_root.exists():
        candidates.extend(
            sorted(
                [child for child in label_root.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")],
                key=checkpoint_step,
                reverse=True,
            )
        )
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
        if (candidate / "adapter_config.json").is_file():
            return candidate
    checked = ", ".join(str(item) for item in unique[:8])
    raise FileNotFoundError(f"Missing adapter_config.json for {label_key}; checked: {checked}")


def parse_max_memory(raw: str, torch: Any) -> dict[Any, str]:
    stripped = raw.strip()
    if not stripped:
        return {}
    if stripped.lower() == "auto":
        if not torch.cuda.is_available():
            raise ValueError("--qwen-max-memory=auto requires CUDA")
        output: dict[Any, str] = {}
        for index in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(index)
            output[index] = f"{max(1, int(free_bytes / (1024 ** 3)) - 1)}GiB"
        return output
    output = {}
    for item in stripped.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid max-memory item: {item!r}")
        device_raw, memory = item.split(":", 1)
        device_key: Any = int(device_raw.strip()) if device_raw.strip().isdigit() else device_raw.strip()
        output[device_key] = memory.strip()
    return output


def model_input_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def align_peft_adapter_devices(model: Any) -> dict[str, Any]:
    adapter_attributes = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
        "modules_to_save",
    )
    aligned = 0
    placements: dict[str, int] = {}
    mismatches = []
    for module_name, module in model.named_modules():
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            continue
        base_parameter = next(base_layer.parameters(), None)
        if base_parameter is None:
            continue
        target_device = base_parameter.device
        found = False
        for attribute in adapter_attributes:
            container = getattr(module, attribute, None)
            if container is None or not hasattr(container, "to"):
                continue
            container.to(target_device)
            found = True
            for parameter_name, parameter in container.named_parameters(recurse=True):
                if parameter.device != target_device:
                    mismatches.append((f"{module_name}.{attribute}.{parameter_name}", str(parameter.device), str(target_device)))
        if found:
            aligned += 1
            name = str(target_device)
            placements[name] = placements.get(name, 0) + 1
    if mismatches:
        raise RuntimeError(f"Failed to align PEFT adapters. First mismatches: {mismatches[:10]}")
    return {"aligned_modules": aligned, "placements": placements}


def load_qwen_and_adapters(args: argparse.Namespace, label_keys: list[str], torch: Any, AutoModelForCausalLM: Any, AutoTokenizer: Any, PeftModel: Any) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Qwen tokenizer must be fast because offset_mapping is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    qwen_device = str(args.qwen_device or "").strip()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if bool(args.bf16) else torch.float16,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    if qwen_device:
        if qwen_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {qwen_device}, but CUDA is not available")
    else:
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        max_memory = parse_max_memory(args.qwen_max_memory, torch)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model_name_or_path, **model_kwargs
        )
    except TypeError as exc:
        raise RuntimeError(
            "The installed Transformers/Qwen stack did not accept the requested "
            "attention implementation.  Eager attention is required for this "
            "test; no silent fallback is allowed."
        ) from exc
    base_model.config.use_cache = True

    context_limit = None
    for context_key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        context_value = getattr(base_model.config, context_key, None)
        if isinstance(context_value, int) and context_value > 0:
            context_limit = context_value
            break
    if context_limit is not None and args.max_qwen_length > context_limit:
        print(
            f"[{timestamp()}] Requested max_qwen_length={args.max_qwen_length} exceeds "
            f"model context limit {context_limit}; using {context_limit}.",
            flush=True,
        )
        args.max_qwen_length = context_limit
    if args.max_qwen_length < 2:
        raise ValueError("max_qwen_length must leave room for a prompt and answer token")

    adapter_root = Path(args.adapter_root)
    first_key = label_keys[0]
    first_adapter = adapter_path(adapter_root, args.adapter_subdir, first_key)
    model = PeftModel.from_pretrained(base_model, str(first_adapter), adapter_name=first_key, is_trainable=False)
    for label_key in label_keys[1:]:
        model.load_adapter(str(adapter_path(adapter_root, args.adapter_subdir, label_key)), adapter_name=label_key, is_trainable=False)

    if qwen_device:
        model.to(qwen_device)
        expected_device = torch.device(qwen_device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        mismatched = [(name, str(parameter.device)) for name, parameter in model.named_parameters() if parameter.device != expected_device]
        if mismatched:
            raise RuntimeError(f"Qwen placement failed for {expected_device}. First mismatches: {mismatched[:10]}")
        print(f"[{timestamp()}] Qwen base and adapters placed on {expected_device}", flush=True)
    else:
        alignment = align_peft_adapter_devices(model)
        print(f"[{timestamp()}] PEFT adapter device alignment: {alignment}", flush=True)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    assert_loaded_adapter_parameters_finite(model, torch)
    return tokenizer, model


def set_adapters_enabled(model: Any, enabled: bool) -> None:
    if enabled:
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        else:
            raise RuntimeError("PEFT model does not expose enable_adapter_layers")
    else:
        if hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()
        else:
            raise RuntimeError("PEFT model does not expose disable_adapter_layers")


def assert_loaded_adapter_parameters_finite(model: Any, torch: Any) -> None:
    checked = 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name and "modules_to_save" not in name:
            continue
        checked += 1
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise ValueError(f"Loaded adapter parameter contains NaN/Inf: {name}")
    if checked == 0:
        raise ValueError("No LoRA parameters were found after loading adapters")


def normalize_weights(np: Any, values: Any, max_value: float = 1e6) -> Any:
    weights = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        if FAIL_ON_NONFINITE:
            raise ValueError("Attention weights contain NaN/Inf; refusing to write vectors")
        nonfinite = int(np.size(weights) - np.count_nonzero(np.isfinite(weights)))
        print(
            f"[{timestamp()}] Warning: normalize_weights cleaned non-finite values; "
            f"nonfinite_count={nonfinite}, size={weights.size}",
            flush=True,
        )
        weights = np.nan_to_num(weights, nan=0.0, posinf=float(max_value), neginf=0.0)
    total = float(weights.sum())
    if total <= 1e-12 or not np.isfinite(total):
        if len(weights) == 0:
            return weights.astype(np.float32)
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float32)
    return (weights / total).astype(np.float32)


def l2_normalize_rows(np: Any, array: Any) -> Any:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (array / norms).astype(np.float32)


def l2_normalize_label_vectors(np: Any, array: Any) -> Any:
    norms = np.linalg.norm(array, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (array / norms).astype(np.float32)


def maybe_normalize_vector(np: Any, vector: Any, enabled: bool) -> Any:
    vector = vector.astype(np.float32)
    if not np.all(np.isfinite(vector)):
        if FAIL_ON_NONFINITE:
            raise ValueError("Hidden vector contains NaN/Inf; refusing to write vectors")
        nonfinite = int(vector.size - np.count_nonzero(np.isfinite(vector)))
        print(
            f"[{timestamp()}] Warning: cleaned non-finite vector values before normalization; "
            f"nonfinite_count={nonfinite}, size={vector.size}",
            flush=True,
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not enabled:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = (vector / norm).astype(np.float32)
    return vector


def prompt_attention_hidden(
    model: Any,
    torch: Any,
    np: Any,
    input_ids: list[int],
    attention_layers: str,
    prefill_chunk_size: int,
    collect_hidden: bool,
) -> tuple[Any, Any | None, list[int], Any]:
    if len(input_ids) < 2:
        raise ValueError("Prompt must contain at least two tokens")
    device = model_input_device(model)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    prefix = ids[:-1]
    last = ids[-1:].view(1, 1)
    past_key_values = None
    chunk_size = max(1, int(prefill_chunk_size))
    hidden_chunks = []
    with torch.no_grad():
        for start in range(0, int(prefix.numel()), chunk_size):
            chunk = prefix[start : start + chunk_size].view(1, -1)
            attention_mask = torch.ones((1, start + int(chunk.numel())), dtype=torch.long, device=device)
            outputs = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=collect_hidden,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            if collect_hidden:
                chunk_hidden = outputs.hidden_states[-1][0].detach().float().cpu().numpy()
                if chunk_hidden.shape[0] != int(chunk.numel()):
                    raise ValueError(
                        "Cached prompt hidden-state length does not match input chunk "
                        f"({chunk_hidden.shape[0]} != {int(chunk.numel())})"
                    )
                hidden_chunks.append(chunk_hidden)
            del outputs
        final_attention_mask = torch.ones((1, len(input_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=last,
            attention_mask=final_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=collect_hidden,
            return_dict=True,
        )
        attentions = outputs.attentions
        if not attentions:
            raise ValueError("Model did not return attentions. Try --attn-implementation eager")
        selected_layers = parse_attention_layers(attention_layers, len(attentions))
        total = None
        for layer_index in selected_layers:
            layer_attention = attentions[layer_index]
            if layer_attention.ndim != 4 or layer_attention.shape[0] != 1:
                raise ValueError(
                    f"Unexpected attention shape at layer {layer_index}: {list(layer_attention.shape)}"
                )
            if layer_attention.shape[2] < 1 or layer_attention.shape[3] != len(input_ids):
                raise ValueError(
                    f"Answer attention key length mismatch at layer {layer_index}: "
                    f"{list(layer_attention.shape)} expected key length {len(input_ids)}"
                )
            values = layer_attention[0, :, -1, : len(input_ids)].detach().float().mean(dim=0).cpu()
            total = values if total is None else total + values
        attention = (total / len(selected_layers)).numpy().astype(np.float32)
        if not np.all(np.isfinite(attention)):
            if FAIL_ON_NONFINITE:
                raise ValueError("Qwen attention contains NaN/Inf")
            nonfinite = int(attention.size - np.count_nonzero(np.isfinite(attention)))
            print(
                f"[{timestamp()}] Warning: Qwen attention contains non-finite values; nonfinite_count={nonfinite}, "
                f"size={attention.size}, selected_layers={selected_layers}; cleaning to zero",
                flush=True,
            )
            attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        hidden = None
        if collect_hidden:
            answer_hidden = outputs.hidden_states[-1][0].detach().float().cpu().numpy()
            if answer_hidden.shape[0] != 1:
                raise ValueError(
                    f"Answer hidden-state length mismatch: {answer_hidden.shape[0]} != 1"
                )
            hidden_chunks.append(answer_hidden)
            hidden = np.concatenate(hidden_chunks, axis=0).astype(np.float32)
            if not np.all(np.isfinite(hidden)):
                if FAIL_ON_NONFINITE:
                    raise ValueError("Qwen hidden states contain NaN/Inf")
                nonfinite = int(hidden.size - np.count_nonzero(np.isfinite(hidden)))
                print(
                    f"[{timestamp()}] Warning: Qwen hidden states contain non-finite values; "
                    f"nonfinite_count={nonfinite}, shape={hidden.shape}; cleaning to zero",
                    flush=True,
                )
                hidden = np.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        logits = outputs.logits[0, -1].detach().float().cpu().numpy()
        if not np.all(np.isfinite(logits)):
            if FAIL_ON_NONFINITE:
                raise ValueError("Qwen logits contain NaN/Inf")
            nonfinite = int(logits.size - np.count_nonzero(np.isfinite(logits)))
            print(
                f"[{timestamp()}] Warning: Qwen logits contain non-finite values; "
                f"nonfinite_count={nonfinite}, size={logits.size}; cleaning to zero",
                flush=True,
            )
            logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        del outputs
    return attention, hidden, selected_layers, logits


def binary_answer_token_ids(tokenizer: Any) -> dict[str, int]:
    """Return the distinct single-token IDs for the classifier answers."""

    def one_token(text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        values = encoded["input_ids"] if hasattr(encoded, "__getitem__") else None
        if values is None and hasattr(encoded, "input_ids"):
            values = encoded.input_ids
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, list):
            raise ValueError(f"Tokenizer did not return input_ids for answer {text!r}")
        if values and isinstance(values[0], (list, tuple)):
            if len(values) != 1:
                raise ValueError(f"Tokenizer returned multiple rows for answer {text!r}")
            values = values[0]
        return [int(value) for value in values]

    zero = one_token("0")
    one = one_token("1")
    if len(zero) != 1 or len(one) != 1 or zero[0] == one[0]:
        raise ValueError(
            f"Qwen tokenizer must encode 0 and 1 as distinct single tokens: {zero}, {one}"
        )
    return {"0": zero[0], "1": one[0]}


def prompt_next_logits(
    model: Any,
    torch: Any,
    input_ids: list[int],
    prefill_chunk_size: int,
) -> Any:
    """Get logits for the token immediately following an answer-less prompt.

    The final prompt token is ``Answer:`` (or the chat-template generation
    marker).  Its causal-LM logits predict the first answer token.  This pass
    deliberately requests no attention: attention is extracted in a second
    pass after the classifier's own predicted token has been appended.
    """

    if not input_ids:
        raise ValueError("Cannot classify an empty prompt")
    device = model_input_device(model)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    past_key_values = None
    chunk_size = max(1, int(prefill_chunk_size))
    final_logits = None
    with torch.no_grad():
        for start in range(0, int(ids.numel()), chunk_size):
            chunk = ids[start : start + chunk_size].view(1, -1)
            attention_mask = torch.ones(
                (1, start + int(chunk.numel())), dtype=torch.long, device=device
            )
            outputs = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            final_logits = outputs.logits[0, -1].detach().float().cpu()
            if not bool(torch.isfinite(final_logits).all().item()):
                raise ValueError("Classifier answer logits contain NaN/Inf")
            past_key_values = outputs.past_key_values
            del outputs
    if final_logits is None:
        raise ValueError("Classifier prompt produced no logits")
    return final_logits


def predicted_answer_from_logits(
    np: Any,
    logits: Any,
    answer_token_ids: dict[str, int],
    threshold: float = 0.5,
) -> tuple[int, float]:
    """Select 0/1 using only the adapter's own answer-token logits."""

    values = np.asarray(logits, dtype=np.float64)
    zero_logit = float(values[int(answer_token_ids["0"])])
    one_logit = float(values[int(answer_token_ids["1"])])
    if not np.isfinite(zero_logit) or not np.isfinite(one_logit):
        raise ValueError("Classifier answer logits contain NaN/Inf")
    # sigmoid(one-zero) is stable and equivalent to softmax over the two
    # binary answer tokens.
    difference = one_logit - zero_logit
    if difference >= 0:
        probability_one = 1.0 / (1.0 + math.exp(-min(difference, 700.0)))
    else:
        exp_difference = math.exp(max(difference, -700.0))
        probability_one = exp_difference / (1.0 + exp_difference)
    predicted = int(probability_one >= float(threshold))
    return predicted, float(probability_one)


def append_answer_token(
    encoding: PromptEncoding,
    answer_token_id: int,
    answer_text: str,
) -> PromptEncoding:
    """Append a manually selected single token without retokenizing the prompt."""

    answer_start = len(encoding.prompt_text)
    answer_end = answer_start + len(answer_text)
    return PromptEncoding(
        prompt_text=encoding.prompt_text,
        input_ids=list(encoding.input_ids) + [int(answer_token_id)],
        offsets=list(encoding.offsets) + [(answer_start, answer_end)],
        post_char_span=encoding.post_char_span,
    )


def post_only_hidden_tokens(
    model: Any,
    torch: Any,
    np: Any,
    tokenizer: Any,
    text: str,
    max_qwen_length: int,
    prefill_chunk_size: int,
    normalize_output: bool,
) -> tuple[list[tuple[int, int]], Any, Any]:
    encoding = encode_post_only(tokenizer, text, max_qwen_length)
    device = model_input_device(model)
    ids = torch.tensor(encoding.input_ids, dtype=torch.long, device=device)
    past_key_values = None
    chunk_size = max(1, int(prefill_chunk_size))
    spans: list[tuple[int, int]] = []
    vectors = []
    with torch.no_grad():
        for start in range(0, int(ids.numel()), chunk_size):
            chunk = ids[start : start + chunk_size].view(1, -1)
            attention_mask = torch.ones((1, start + int(chunk.numel())), dtype=torch.long, device=device)
            outputs = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1][0].detach().float().cpu().numpy()
            if hidden.shape[0] != int(chunk.numel()):
                raise ValueError(
                    "Post-only cached hidden-state length does not match input chunk "
                    f"({hidden.shape[0]} != {int(chunk.numel())})"
                )
            if not np.all(np.isfinite(hidden)):
                if FAIL_ON_NONFINITE:
                    raise ValueError("Post-only hidden states contain NaN/Inf")
                nonfinite = int(hidden.size - np.count_nonzero(np.isfinite(hidden)))
                print(
                    f"[{timestamp()}] Warning: post-only hidden states contain non-finite values; "
                    f"nonfinite_count={nonfinite}, shape={hidden.shape}; cleaning to zero",
                    flush=True,
                )
                hidden = np.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            offsets = encoding.offsets[start : start + int(chunk.numel())]
            for local_index, (span_start_raw, span_end_raw) in enumerate(offsets):
                span_start = int(span_start_raw)
                span_end = int(span_end_raw)
                if span_end <= span_start:
                    continue
                spans.append((span_start, span_end))
                vectors.append(hidden[local_index])
            past_key_values = outputs.past_key_values
            del outputs
    if not vectors:
        raise ValueError("Post-only Qwen tokenizer produced no non-special text tokens")
    matrix = np.stack(vectors).astype(np.float32)
    mean_vector = maybe_normalize_vector(np, matrix.mean(axis=0), normalize_output)
    return spans, matrix, mean_vector


def parse_attention_layers(spec: str, num_layers: int) -> list[int]:
    raw = spec.strip().lower()
    if raw == "all":
        return list(range(num_layers))
    if raw.startswith("last:"):
        count = int(raw.split(":", 1)[1])
        if count <= 0:
            raise ValueError("last:N requires N > 0")
        return list(range(max(0, num_layers - count), num_layers))
    if raw.startswith("first:"):
        count = int(raw.split(":", 1)[1])
        if count <= 0:
            raise ValueError("first:N requires N > 0")
        return list(range(min(count, num_layers)))
    if "," in raw:
        items = [int(item.strip()) for item in raw.split(",") if item.strip()]
    elif ":" in raw:
        start_raw, end_raw = raw.split(":", 1)
        start = int(start_raw) if start_raw else 0
        end = int(end_raw) if end_raw else num_layers
        if start < 0:
            start = num_layers + start
        if end < 0:
            end = num_layers + end
        items = list(range(start, end))
    else:
        items = [int(raw)]
    output = []
    for item in items:
        index = item + num_layers if item < 0 else item
        if index < 0 or index >= num_layers:
            raise ValueError(f"Layer index {item} outside [0, {num_layers - 1}]")
        output.append(index)
    if not output:
        raise ValueError("--attention-layers selected no layers")
    return output


def entries_from_prompt_weights(encoding: PromptEncoding, token_indices: list[int], weights: Any) -> list[tuple[int, int, float]]:
    post_start, post_end = encoding.post_char_span
    entries = []
    for local_index, token_index in enumerate(token_indices):
        start_raw, end_raw = encoding.offsets[token_index]
        start = int(start_raw)
        end = int(end_raw)
        overlap_start = max(start, post_start)
        overlap_end = min(end, post_end)
        if overlap_end <= overlap_start:
            continue
        entries.append((overlap_start - post_start, overlap_end - post_start, float(weights[local_index])))
    return entries


def transfer_weights_to_post_tokens(np: Any, entries: list[tuple[int, int, float]], post_spans: list[tuple[int, int]]) -> tuple[Any, float]:
    weights = np.zeros(len(post_spans), dtype=np.float32)
    if not entries or not post_spans:
        return weights, 0.0
    m_index = 0
    mapped_mass = 0.0
    for q_start, q_end, q_weight in entries:
        while m_index < len(post_spans) and post_spans[m_index][1] <= q_start:
            m_index += 1
        overlaps = []
        overlap_total = 0
        scan = m_index
        while scan < len(post_spans) and post_spans[scan][0] < q_end:
            m_start, m_end = post_spans[scan]
            overlap = max(0, min(q_end, m_end) - max(q_start, m_start))
            if overlap > 0:
                overlaps.append((scan, overlap))
                overlap_total += overlap
            scan += 1
        if overlap_total <= 0:
            continue
        mapped_mass += q_weight
        for target_index, overlap in overlaps:
            weights[target_index] += float(q_weight) * (overlap / overlap_total)
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.full(len(post_spans), 1.0 / max(1, len(post_spans)), dtype=np.float32)
    else:
        weights /= total
    return weights.astype(np.float32), float(mapped_mass)


def weighted_hidden_vector(np: Any, hidden: Any, indices: Any, weights: Any, normalize_output: bool) -> Any:
    vector = np.matmul(weights.astype(np.float32), hidden[indices].astype(np.float32)).astype(np.float32)
    if not np.all(np.isfinite(vector)):
        if FAIL_ON_NONFINITE:
            raise ValueError("Weighted hidden vector contains NaN/Inf")
        nonfinite = int(vector.size - np.count_nonzero(np.isfinite(vector)))
        print(
            f"[{timestamp()}] Warning: weighted vector contains non-finite values after matmul; "
            f"nonfinite_count={nonfinite}, size={vector.size}; cleaning to zero",
            flush=True,
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return maybe_normalize_vector(np, vector, normalize_output)


def build_vectors_for_prompt(
    np: Any,
    encoding: PromptEncoding,
    base_attention: Any,
    adapter_attention: Any,
    base_post_spans: list[tuple[int, int]],
    base_post_hidden: Any,
    adapter_hidden: Any,
    ratio_eps: float,
    max_ratio: float,
    normalize_output: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    indices = post_token_indices(encoding)
    index_array = np.array(indices, dtype=np.int64)
    base_raw = np.asarray(base_attention, dtype=np.float64)[index_array]
    adapter_raw = np.asarray(adapter_attention, dtype=np.float64)[index_array]
    require_finite_numpy(np, base_raw, "base post attention")
    require_finite_numpy(np, adapter_raw, "adapter post attention")
    if np.any(base_raw < 0) or np.any(adapter_raw < 0):
        raise ValueError("Attention weights contain a negative value")
    base_weights = normalize_weights(np, base_raw)
    adapter_weights = normalize_weights(np, adapter_raw)
    safe_eps = max(float(ratio_eps), 1e-6)

    ratio_denominator = np.maximum(base_weights.astype(np.float64), safe_eps)
    ratio = adapter_weights.astype(np.float64) / ratio_denominator
    if max_ratio > 0:
        ratio = np.clip(ratio, 0.0, float(max_ratio))
    else:
        ratio = np.maximum(ratio, 0.0)
    delta_ratio_weights = normalize_weights(np, adapter_weights.astype(np.float64) * ratio)

    change_rate_denominator = np.maximum(base_raw.astype(np.float64), safe_eps)
    change_rate = (adapter_raw.astype(np.float64) + safe_eps) / change_rate_denominator
    if max_ratio > 0:
        change_rate = np.clip(change_rate, 0.0, float(max_ratio))
    else:
        change_rate = np.maximum(change_rate, 0.0)
    change_rate_weights = normalize_weights(np, change_rate)

    adapter_entries = entries_from_prompt_weights(encoding, indices, adapter_weights)
    delta_entries = entries_from_prompt_weights(encoding, indices, delta_ratio_weights)
    base_hidden_adapter_weights, adapter_mapped_mass = transfer_weights_to_post_tokens(np, adapter_entries, base_post_spans)
    base_hidden_delta_weights, delta_mapped_mass = transfer_weights_to_post_tokens(np, delta_entries, base_post_spans)
    base_index = np.arange(len(base_post_spans), dtype=np.int64)

    vectors = {
        "base_hidden_adapter_attention": weighted_hidden_vector(np, base_post_hidden, base_index, base_hidden_adapter_weights, normalize_output),
        "base_hidden_delta_ratio_attention": weighted_hidden_vector(np, base_post_hidden, base_index, base_hidden_delta_weights, normalize_output),
        "adapter_hidden_attention": weighted_hidden_vector(np, adapter_hidden, index_array, adapter_weights, normalize_output),
        "adapter_hidden_delta_ratio_attention": weighted_hidden_vector(np, adapter_hidden, index_array, delta_ratio_weights, normalize_output),
        "adapter_hidden_attention_change_rate": weighted_hidden_vector(np, adapter_hidden, index_array, change_rate_weights, normalize_output),
    }
    for vector_name, vector in vectors.items():
        require_finite_numpy(np, vector, vector_name)
    metrics = {
        "post_token_count": float(len(indices)),
        "base_attention_mass": float(base_raw.sum()),
        "adapter_attention_mass": float(adapter_raw.sum()),
        "adapter_max_weight": float(adapter_weights.max()) if len(adapter_weights) else 0.0,
        "delta_ratio_max_weight": float(delta_ratio_weights.max()) if len(delta_ratio_weights) else 0.0,
        "change_rate_max_weight": float(change_rate_weights.max()) if len(change_rate_weights) else 0.0,
        "base_hidden_adapter_mapped_mass": adapter_mapped_mass,
        "base_hidden_delta_ratio_mapped_mass": delta_mapped_mass,
        "ratio_denominator_floor_count": float(np.count_nonzero(base_weights.astype(np.float64) < safe_eps)),
        "change_rate_denominator_floor_count": float(np.count_nonzero(base_raw.astype(np.float64) < safe_eps)),
        "ratio_mean": float(ratio.mean()) if len(ratio) else 0.0,
        "ratio_max": float(ratio.max()) if len(ratio) else 0.0,
        "change_rate_mean": float(change_rate.mean()) if len(change_rate) else 0.0,
        "change_rate_max": float(change_rate.max()) if len(change_rate) else 0.0,
        "attention_delta_max": float((adapter_weights.astype(np.float64) - base_weights.astype(np.float64)).max()) if len(adapter_weights) else 0.0,
        "attention_delta_min": float((adapter_weights.astype(np.float64) - base_weights.astype(np.float64)).min()) if len(adapter_weights) else 0.0,
    }
    return vectors, metrics


def safe_empty_cache(torch: Any) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        print(f"[{timestamp()}] Warning: torch.cuda.empty_cache failed: {exc}", flush=True)


def save_npz(np: Any, path: Path, compress: bool, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def extract_vectors(args: argparse.Namespace) -> None:
    global FAIL_ON_NONFINITE
    FAIL_ON_NONFINITE = bool(args.fail_on_nonfinite)
    if not FAIL_ON_NONFINITE:
        raise ValueError("This evaluator requires --fail-on-nonfinite 1")
    if not 0.0 <= float(args.prediction_threshold) <= 1.0:
        raise ValueError("--prediction-threshold must be between 0 and 1")
    if args.attention_answer_mode != "predicted":
        raise ValueError(
            "This deployment evaluator is no-gold only: "
            "--attention-answer-mode must be predicted"
        )
    start_time = time.time()
    label_keys = parse_label_keys(args.label_keys)
    examples, load_info = load_unlabeled_examples(
        Path(args.data_file), args.record_start, args.max_records
    )
    if not examples:
        raise RuntimeError(f"No examples loaded: {load_info}")
    output_file = Path(args.output_file)
    if output_file.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing output_file without --overwrite 1: {output_file}")

    np, torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = import_runtime_stack()
    torch.set_grad_enabled(False)
    tokenizer, model = load_qwen_and_adapters(args, label_keys, torch, AutoModelForCausalLM, AutoTokenizer, PeftModel)
    answer_token_ids = binary_answer_token_ids(tokenizer)

    arrays: dict[str, Any] = {}
    prompt_lengths = None
    post_token_counts = None
    predicted_answer_ids = None
    predicted_probability_one = None
    metric_arrays: dict[str, Any] = {}
    selected_layers_reference: list[int] | None = None
    metric_names = [
        "post_token_count",
        "base_attention_mass",
        "adapter_attention_mass",
        "adapter_max_weight",
        "delta_ratio_max_weight",
        "change_rate_max_weight",
        "base_hidden_adapter_mapped_mass",
        "base_hidden_delta_ratio_mapped_mass",
        "ratio_denominator_floor_count",
        "change_rate_denominator_floor_count",
        "ratio_mean",
        "ratio_max",
        "change_rate_mean",
        "change_rate_max",
        "attention_delta_max",
        "attention_delta_min",
    ]

    try:
        for example_index, example in enumerate(examples):
            print(f"[{timestamp()}] Fresh extract {example_index + 1}/{len(examples)} post_id={example.post_id}", flush=True)
            model.set_adapter(label_keys[0])
            set_adapters_enabled(model, False)
            try:
                base_post_spans, base_post_hidden, base_post_mean = post_only_hidden_tokens(
                    model,
                    torch,
                    np,
                    tokenizer,
                    example.text,
                    args.max_qwen_length,
                    args.qwen_prefill_chunk_size,
                    bool(args.normalize_output_vectors),
                )
            finally:
                set_adapters_enabled(model, True)

            if not arrays:
                hidden_size = int(base_post_mean.shape[0])
                arrays["base_hidden_post_mean"] = np.zeros((len(examples), hidden_size), dtype=np.float32)
                for key in VECTOR_KEYS:
                    if key == "base_hidden_post_mean":
                        continue
                    arrays[key] = np.zeros((len(examples), len(label_keys), hidden_size), dtype=np.float32)
                prompt_lengths = np.zeros((len(examples), len(label_keys)), dtype=np.int32)
                post_token_counts = np.zeros(len(examples), dtype=np.int32)
                predicted_answer_ids = np.full(
                    (len(examples), len(label_keys)), -1, dtype=np.int8
                )
                predicted_probability_one = np.full(
                    (len(examples), len(label_keys)), np.nan, dtype=np.float32
                )
                metric_arrays = {name: np.zeros((len(examples), len(label_keys)), dtype=np.float32) for name in metric_names}

            arrays["base_hidden_post_mean"][example_index] = base_post_mean
            post_token_counts[example_index] = len(base_post_spans)

            for label_index, label_key in enumerate(label_keys):
                # First pass: the adapter classifies the answer-less prompt.
                # No dataset label is read here.  The selected token is then
                # appended manually for the answer-query attention pass.
                model.set_adapter(label_key)
                set_adapters_enabled(model, True)
                prompt_encoding = encode_prompt(
                    tokenizer,
                    label_key,
                    example.text,
                    args.max_qwen_length - 1,
                    answer_text=None,
                )
                prediction_logits = prompt_next_logits(
                    model,
                    torch,
                    prompt_encoding.input_ids,
                    args.qwen_prefill_chunk_size,
                )
                predicted_value, probability_one = predicted_answer_from_logits(
                    np,
                    prediction_logits,
                    answer_token_ids,
                    float(args.prediction_threshold),
                )
                answer_text = str(predicted_value)
                encoding = append_answer_token(
                    prompt_encoding,
                    answer_token_ids[answer_text],
                    answer_text,
                )
                predicted_answer_ids[example_index, label_index] = predicted_value
                predicted_probability_one[example_index, label_index] = probability_one
                del prediction_logits
                if len(encoding.input_ids) > args.max_qwen_length:
                    raise ValueError(
                        f"Prompt plus selected answer has {len(encoding.input_ids)} tokens, "
                        f"over --max-qwen-length={args.max_qwen_length}"
                    )
                prompt_lengths[example_index, label_index] = len(encoding.input_ids)

                model.set_adapter(label_key)
                set_adapters_enabled(model, False)
                try:
                    base_attention, _, selected_layers, _ = prompt_attention_hidden(
                        model,
                        torch,
                        np,
                        encoding.input_ids,
                        args.attention_layers,
                        args.qwen_prefill_chunk_size,
                        collect_hidden=False,
                    )
                finally:
                    set_adapters_enabled(model, True)

                model.set_adapter(label_key)
                adapter_attention, adapter_hidden, selected_layers, _ = prompt_attention_hidden(
                    model,
                    torch,
                    np,
                    encoding.input_ids,
                    args.attention_layers,
                    args.qwen_prefill_chunk_size,
                    collect_hidden=True,
                )
                if adapter_hidden is None:
                    raise RuntimeError("adapter_hidden was not collected")
                if selected_layers_reference is None:
                    selected_layers_reference = selected_layers

                vector_bundle, vector_metrics = build_vectors_for_prompt(
                    np,
                    encoding,
                    base_attention,
                    adapter_attention,
                    base_post_spans,
                    base_post_hidden,
                    adapter_hidden,
                    float(args.ratio_eps),
                    float(args.max_ratio),
                    bool(args.normalize_output_vectors),
                )
                for key, vector in vector_bundle.items():
                    arrays[key][example_index, label_index] = vector
                for name in metric_names:
                    metric_arrays[name][example_index, label_index] = float(vector_metrics[name])
                del base_attention, adapter_attention, adapter_hidden
            del base_post_hidden
            safe_empty_cache(torch)

    finally:
        try:
            del model
            del tokenizer
        except UnboundLocalError:
            pass
        gc.collect()
        safe_empty_cache(torch)

    for vector_name, vector_array in arrays.items():
        require_finite_numpy(np, vector_array, f"{vector_name} vectors")
    for metric_name, metric_array in metric_arrays.items():
        require_finite_numpy(np, metric_array, f"metric_{metric_name}")
    if predicted_answer_ids is not None:
        if np.any(predicted_answer_ids < 0):
            raise ValueError("At least one classifier prediction was not populated")
        require_finite_numpy(np, predicted_answer_ids, "predicted answer IDs")
    if predicted_probability_one is not None:
        require_finite_numpy(np, predicted_probability_one, "predicted answer probabilities")
        if np.any(predicted_probability_one < 0.0) or np.any(predicted_probability_one > 1.0):
            raise ValueError("Predicted answer probabilities must lie in [0, 1]")

    payload: dict[str, Any] = {
        "created_at": np.array([timestamp()], dtype=str),
        "script": np.array([Path(__file__).name], dtype=str),
        "post_ids": np.array([example.post_id for example in examples], dtype=str),
        "subreddits": np.array([example.subreddit for example in examples], dtype=str),
        "label_keys": np.array(label_keys, dtype=str),
        "selected_layers": np.array(selected_layers_reference or [], dtype=np.int32),
        "attention_layers": np.array([args.attention_layers], dtype=str),
        "max_qwen_length": np.array([args.max_qwen_length], dtype=np.int32),
        "qwen_prefill_chunk_size": np.array([args.qwen_prefill_chunk_size], dtype=np.int32),
        "attention_answer_mode": np.array(["predicted"], dtype=str),
        "attention_query": np.array(["predicted_answer_token"], dtype=str),
        "gold_labels_used_for_prompt": np.array([0], dtype=np.int8),
        "gold_labels_in_output": np.array([0], dtype=np.int8),
        "labels_accessed_during_inference": np.array([0], dtype=np.int8),
        "classification_metrics_computed": np.array([0], dtype=np.int8),
        "attention_extracted": np.array([1], dtype=np.int8),
        "hidden_states_extracted": np.array([1], dtype=np.int8),
        "raw_attention_maps_saved": np.array([0], dtype=np.int8),
        "answer_token_ids_saved": np.array([1], dtype=np.int8),
        "answer_token_ids": np.array(
            [answer_token_ids["0"], answer_token_ids["1"]], dtype=np.int64
        ),
        "normalize_output_vectors": np.array([int(bool(args.normalize_output_vectors))], dtype=np.int8),
        "ratio_eps": np.array([float(args.ratio_eps)], dtype=np.float64),
        "max_ratio": np.array([float(args.max_ratio)], dtype=np.float64),
        "record_start": np.array([args.record_start], dtype=np.int32),
        "max_records": np.array([args.max_records], dtype=np.int32),
        "prompt_lengths": prompt_lengths,
        "post_token_counts": post_token_counts,
        **{f"{key}_vectors": value for key, value in arrays.items()},
        **{f"metric_{key}": value for key, value in metric_arrays.items()},
    }
    # The selected answer is an internal query coordinate, not an output
    # classification result.  Keep the saved artifact limited to the vector
    # families and provenance metadata requested by this experiment.
    save_npz(np, output_file, bool(args.compress), **payload)
    summary = {
        "created_at": timestamp(),
        "output_file": str(output_file),
        "record_start": args.record_start,
        "max_records": args.max_records,
        "example_count": len(examples),
        "label_keys": label_keys,
        "selected_layers": selected_layers_reference or [],
        "attention_layers": args.attention_layers,
        "attention_answer_mode": "predicted",
        "attention_query": "predicted_answer_token",
        "gold_labels_used_for_prompt": False,
        "gold_labels_in_output": False,
        "labels_accessed_during_inference": False,
        "classification_metrics_computed": False,
        "attention_extracted": True,
        "hidden_states_extracted": True,
        "raw_attention_maps_saved": False,
        "saved_vector_keys": list(VECTOR_KEYS),
        "answer_token_ids_saved": True,
        "load_info": load_info,
        "method_definitions": METHOD_DEFINITIONS,
        "runtime_seconds": time.time() - start_time,
    }
    write_json(output_file.with_suffix(".summary.json"), summary)
    print(f"[{timestamp()}] Fresh extraction wrote {output_file}", flush=True)


def label_matrix(np: Any, examples: list[PostExample], label_keys: list[str]) -> Any:
    return np.array([[example.labels[label_key] for label_key in label_keys] for example in examples], dtype=np.int8)


def semantic_scores(np: Any, vectors: Any) -> Any:
    normalized = l2_normalize_rows(np, vectors)
    return (normalized @ normalized.T).astype(np.float32)


def mean_label_scores(np: Any, vectors: Any) -> Any:
    normalized = l2_normalize_label_vectors(np, vectors)
    scores = np.zeros((vectors.shape[0], vectors.shape[0]), dtype=np.float32)
    for label_index in range(vectors.shape[1]):
        scores += normalized[:, label_index, :] @ normalized[:, label_index, :].T
    scores /= vectors.shape[1]
    return scores.astype(np.float32)


def ranked_indices_excluding_same_post_id(np: Any, scores: Any, post_ids: list[str], max_k: int) -> list[Any]:
    score_matrix = np.asarray(scores)
    post_id_values = np.array(post_ids, dtype=str)
    rows = []
    for query_index, query_post_id in enumerate(post_id_values):
        candidate_indices = np.flatnonzero(post_id_values != query_post_id)
        if len(candidate_indices) == 0:
            rows.append(candidate_indices)
            continue
        candidate_scores = score_matrix[query_index, candidate_indices]
        candidate_scores = np.where(np.isfinite(candidate_scores), candidate_scores, -np.inf)
        limit = min(max_k, len(candidate_indices))
        order = np.argsort(-candidate_scores, kind="stable")[:limit]
        rows.append(candidate_indices[order])
    return rows


def label_match_fraction(np: Any, query_vector: Any, neighbor_vectors: Any) -> Any:
    return np.mean(neighbor_vectors == query_vector, axis=1)


def positive_jaccard(np: Any, query_vector: Any, neighbor_vectors: Any) -> Any:
    query_positive = query_vector.astype(bool)
    neighbor_positive = neighbor_vectors.astype(bool)
    intersection = np.logical_and(neighbor_positive, query_positive).sum(axis=1)
    union = np.logical_or(neighbor_positive, query_positive).sum(axis=1)
    return np.where(union == 0, 1.0, intersection / np.maximum(union, 1)).astype(np.float32)


def overlap_counts(np: Any, query_vector: Any, neighbor_vector: Any) -> dict[str, int]:
    query_bool = query_vector.astype(bool)
    neighbor_bool = neighbor_vector.astype(bool)
    return {
        "same_label_count": int(np.sum(query_vector == neighbor_vector)),
        "different_label_count": int(np.sum(query_vector != neighbor_vector)),
        "query_positive_count": int(np.sum(query_bool)),
        "neighbor_positive_count": int(np.sum(neighbor_bool)),
        "shared_positive_count": int(np.sum(np.logical_and(query_bool, neighbor_bool))),
        "missing_query_positive_count": int(np.sum(np.logical_and(query_bool, ~neighbor_bool))),
        "extra_neighbor_positive_count": int(np.sum(np.logical_and(~query_bool, neighbor_bool))),
    }


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def evaluate_method(
    np: Any,
    method_name: str,
    scores: Any,
    examples: list[PostExample],
    labels: Any,
    top_ks: list[int],
    query_results_top_k: int,
    max_rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    post_ids = [example.post_id for example in examples]
    subreddits = np.array([example.subreddit for example in examples], dtype=str)
    max_k = max(max(top_ks), query_results_top_k, max_rank)
    ranks = ranked_indices_excluding_same_post_id(np, scores, post_ids, max_k)
    same_post_counts = Counter(post_ids)

    retrieval_counts = {k: 0 for k in top_ks}
    topk_label_values = {
        k: {
            "label_match_max": [],
            "label_match_mean": [],
            "label_match_min": [],
            "positive_jaccard_max": [],
            "positive_jaccard_mean": [],
            "positive_jaccard_min": [],
            "exact_label_hit": [],
        }
        for k in top_ks
    }
    rank_buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "same_subreddit": 0,
            "same_post_id": 0,
            "label_match_fraction": [],
            "positive_jaccard": [],
            "exact_label_match": 0,
            "any_shared_positive": 0,
            "same_label_count": [],
            "different_label_count": [],
            "query_positive_count": [],
            "neighbor_positive_count": [],
            "shared_positive_count": [],
            "missing_query_positive_count": [],
            "extra_neighbor_positive_count": [],
        }
    )
    query_rows = []

    for query_index, neighbors in enumerate(ranks):
        query_subreddit = str(subreddits[query_index])
        query_post_id = post_ids[query_index]
        retrieved_subreddits = subreddits[neighbors]
        query_labels = labels[query_index]
        neighbor_labels = labels[neighbors]
        label_match_values = label_match_fraction(np, query_labels, neighbor_labels)
        jaccard_values = positive_jaccard(np, query_labels, neighbor_labels)

        for k in top_ks:
            if bool(np.any(retrieved_subreddits[:k] == query_subreddit)):
                retrieval_counts[k] += 1
            top_match = label_match_values[:k]
            top_jaccard = jaccard_values[:k]
            topk_label_values[k]["label_match_max"].append(float(np.max(top_match)) if len(top_match) else 0.0)
            topk_label_values[k]["label_match_mean"].append(float(np.mean(top_match)) if len(top_match) else 0.0)
            topk_label_values[k]["label_match_min"].append(float(np.min(top_match)) if len(top_match) else 0.0)
            topk_label_values[k]["positive_jaccard_max"].append(float(np.max(top_jaccard)) if len(top_jaccard) else 0.0)
            topk_label_values[k]["positive_jaccard_mean"].append(float(np.mean(top_jaccard)) if len(top_jaccard) else 0.0)
            topk_label_values[k]["positive_jaccard_min"].append(float(np.min(top_jaccard)) if len(top_jaccard) else 0.0)
            topk_label_values[k]["exact_label_hit"].append(float(np.any(top_match == 1.0)) if len(top_match) else 0.0)

        top_neighbors = []
        for rank, neighbor_index in enumerate(neighbors[:query_results_top_k], start=1):
            neighbor_vector = labels[neighbor_index]
            match_value = float(label_match_values[rank - 1])
            jaccard_value = float(jaccard_values[rank - 1])
            counts = overlap_counts(np, query_labels, neighbor_vector)
            same_subreddit = bool(subreddits[neighbor_index] == query_subreddit)
            same_post_id = bool(post_ids[neighbor_index] == query_post_id)
            top_neighbors.append(
                {
                    "rank": rank,
                    "post_id": post_ids[neighbor_index],
                    "subreddit": str(subreddits[neighbor_index]),
                    "score": float(scores[query_index, neighbor_index]),
                    "same_subreddit": same_subreddit,
                    "same_post_id": same_post_id,
                    "label_match_fraction": match_value,
                    "positive_jaccard": jaccard_value,
                    "label_vector": [int(value) for value in neighbor_vector.tolist()],
                }
            )
            if rank <= max_rank:
                bucket = rank_buckets[rank]
                bucket["count"] += 1
                bucket["same_subreddit"] += int(same_subreddit)
                bucket["same_post_id"] += int(same_post_id)
                bucket["label_match_fraction"].append(match_value)
                bucket["positive_jaccard"].append(jaccard_value)
                bucket["exact_label_match"] += int(match_value >= 1.0)
                bucket["any_shared_positive"] += int(counts["shared_positive_count"] > 0)
                for key, value in counts.items():
                    bucket[key].append(float(value))

        query_rows.append(
            {
                "method": method_name,
                "query_index": int(query_index),
                "post_id": query_post_id,
                "subreddit": query_subreddit,
                "candidate_count": int(len(examples) - same_post_counts[query_post_id]),
                "excluded_same_post_id_count": int(same_post_counts[query_post_id]),
                "query_label_vector": [int(value) for value in query_labels.tolist()],
                "top_neighbors": top_neighbors,
            }
        )

    total = len(examples)
    summary = {
        "method": method_name,
        "family": METHOD_DEFINITIONS[method_name]["family"],
        "core": METHOD_DEFINITIONS[method_name]["core"],
        "score_definition": METHOD_DEFINITIONS[method_name]["score"],
        "total": total,
        "candidate_count_min": min(len(examples) - same_post_counts[pid] for pid in post_ids) if post_ids else 0,
        "candidate_count_max": max(len(examples) - same_post_counts[pid] for pid in post_ids) if post_ids else 0,
        "candidate_count_mean": mean_or_zero([float(len(examples) - same_post_counts[pid]) for pid in post_ids]),
        "self_match_policy": "exclude all candidates whose post_id equals the query post_id",
    }
    for k in top_ks:
        summary[f"top{k}_accuracy"] = rate(retrieval_counts[k], total)
        summary[f"top{k}_correct"] = int(retrieval_counts[k])
        for metric_name, values in topk_label_values[k].items():
            suffix = "rate" if metric_name == "exact_label_hit" else "query_mean"
            summary[f"top{k}_{metric_name}_{suffix}"] = mean_or_zero(values)

    by_rank_rows = []
    for rank_value in range(1, max_rank + 1):
        bucket = rank_buckets[rank_value]
        count = int(bucket["count"])
        by_rank_rows.append(
            {
                "method": method_name,
                "rank": rank_value,
                "count": count,
                "same_subreddit_rate": rate(bucket["same_subreddit"], count),
                "same_post_id_rate": rate(bucket["same_post_id"], count),
                "exact_label_match_rate": rate(bucket["exact_label_match"], count),
                "any_shared_positive_rate": rate(bucket["any_shared_positive"], count),
                "label_match_fraction_mean": mean_or_zero(bucket["label_match_fraction"]),
                "positive_jaccard_mean": mean_or_zero(bucket["positive_jaccard"]),
                "same_label_count_mean": mean_or_zero(bucket["same_label_count"]),
                "different_label_count_mean": mean_or_zero(bucket["different_label_count"]),
                "query_positive_count_mean": mean_or_zero(bucket["query_positive_count"]),
                "neighbor_positive_count_mean": mean_or_zero(bucket["neighbor_positive_count"]),
                "shared_positive_count_mean": mean_or_zero(bucket["shared_positive_count"]),
                "missing_query_positive_count_mean": mean_or_zero(bucket["missing_query_positive_count"]),
                "extra_neighbor_positive_count_mean": mean_or_zero(bucket["extra_neighbor_positive_count"]),
            }
        )
    return summary, by_rank_rows, query_rows


def load_shard_npz(np: Any, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing shard file: {path}")
    print(f"[{timestamp()}] Loading shard {path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        required = {"post_ids", "subreddits", "label_keys", *{f"{key}_vectors" for key in VECTOR_KEYS}}
        missing = sorted(key for key in required if key not in data)
        if missing:
            raise KeyError(f"Shard {path} is missing keys: {missing}")
        materialized = {key: data[key] for key in data.files}
    post_ids = materialized["post_ids"].astype(str).tolist()
    print(f"[{timestamp()}] Loaded shard {path} rows={len(post_ids)}", flush=True)
    return {"path": path, "data": materialized, "post_ids": post_ids}


def merge_shards_for_examples(np: Any, shard_paths: list[Path], examples: list[PostExample], label_keys: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"[{timestamp()}] Merging {len(shard_paths)} shard file(s) for {len(examples)} examples", flush=True)
    shards = [load_shard_npz(np, path) for path in shard_paths]
    by_post_id: dict[str, tuple[dict[str, Any], int]] = {}
    duplicates = []
    for shard in shards:
        for index, post_id in enumerate(shard["post_ids"]):
            if post_id in by_post_id:
                duplicates.append(post_id)
            by_post_id[post_id] = (shard, index)
    if duplicates:
        raise ValueError(f"Duplicate post_ids across shards: {duplicates[:20]}")
    target_ids = [example.post_id for example in examples]
    missing = [post_id for post_id in target_ids if post_id not in by_post_id]
    if missing:
        raise ValueError(f"{len(missing)} target post_ids missing from shards. First missing: {missing[:20]}")
    target_id_set = set(target_ids)
    extra = [post_id for post_id in by_post_id if post_id not in target_id_set]
    if extra:
        raise ValueError(f"{len(extra)} shard post_ids are not in the target corpus. First extra: {extra[:20]}")

    first = shards[0]["data"]
    shard_label_keys = first["label_keys"].astype(str).tolist()
    if shard_label_keys != label_keys:
        raise ValueError(f"label_keys mismatch: shard={shard_label_keys}, data={label_keys}")
    shard_modes = []
    shard_gold_flags = []
    for shard in shards:
        shard_data = shard["data"]
        current_keys = shard_data["label_keys"].astype(str).tolist()
        if current_keys != label_keys:
            raise ValueError(f"label_keys mismatch in shard {shard['path']}: {current_keys}")
        mode = str(shard_data["attention_answer_mode"].astype(str)[0]) if "attention_answer_mode" in shard_data else ""
        gold_flag = bool(int(shard_data["gold_labels_used_for_prompt"][0])) if "gold_labels_used_for_prompt" in shard_data else None
        shard_modes.append(mode)
        shard_gold_flags.append(gold_flag)
    if any(mode != "predicted" for mode in shard_modes) or any(flag is not False for flag in shard_gold_flags):
        raise ValueError("All shards must be predicted-token, no-gold shards")
    if any("labels" in shard["data"] for shard in shards):
        raise ValueError("A no-gold shard contains a labels array")
    arrays: dict[str, Any] = {}
    for key in VECTOR_KEYS:
        source_array = first[f"{key}_vectors"]
        arrays[key] = np.empty((len(examples), *source_array.shape[1:]), dtype=source_array.dtype)
    for output_index, post_id in enumerate(target_ids):
        shard, source_index = by_post_id[post_id]
        for key in VECTOR_KEYS:
            arrays[key][output_index] = shard["data"][f"{key}_vectors"][source_index]
        if (output_index + 1) % 100 == 0 or output_index + 1 == len(target_ids):
            print(f"[{timestamp()}] Merged {output_index + 1}/{len(target_ids)} examples", flush=True)
    selected_layers = first["selected_layers"].astype(int).tolist() if "selected_layers" in first else []
    metadata = {
        "shard_files": [str(path) for path in shard_paths],
        "shard_count": len(shard_paths),
        "target_count": len(examples),
        "selected_layers": selected_layers,
        "attention_layers": str(first["attention_layers"].astype(str)[0]) if "attention_layers" in first else "",
        "attention_answer_mode": str(first["attention_answer_mode"].astype(str)[0]) if "attention_answer_mode" in first else "missing_pre_answer_mode",
        "attention_query": str(first["attention_query"].astype(str)[0]) if "attention_query" in first else "",
        "gold_labels_used_for_prompt": bool(int(first["gold_labels_used_for_prompt"][0])) if "gold_labels_used_for_prompt" in first else None,
        "shard_attention_answer_modes": shard_modes,
        "shard_gold_labels_used_for_prompt": shard_gold_flags,
    }
    return arrays, metadata


def require_predicted_shards(shard_info: dict[str, Any], shard_paths: list[Path], shard_materialized: list[dict[str, Any]] | None = None) -> None:
    """Reject any shard that could have used a gold/configured answer token."""

    modes = shard_info.get("shard_attention_answer_modes") or []
    gold_flags = shard_info.get("shard_gold_labels_used_for_prompt") or []
    if not modes or any(mode != "predicted" for mode in modes):
        raise ValueError("No-gold output requires every shard to have attention_answer_mode=predicted")
    if not gold_flags or any(flag is not False for flag in gold_flags):
        raise ValueError("No-gold output requires every shard to declare gold_labels_used_for_prompt=false")
    if shard_materialized is not None:
        leaked = [str(path) for path, data in zip(shard_paths, shard_materialized) if "labels" in data]
        if leaked:
            raise ValueError(f"No-gold shard unexpectedly contains labels: {leaked[:3]}")


def merge_row_fields(
    np: Any,
    shard_paths: list[Path],
    examples: list[PostExample],
    fields: list[str],
) -> dict[str, Any]:
    """Merge optional row-aligned metadata (predictions, lengths, metrics)."""

    loaded = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            loaded.append({key: data[key] for key in data.files})
    locations: dict[str, tuple[dict[str, Any], int]] = {}
    for data in loaded:
        post_ids = data["post_ids"].astype(str).tolist()
        for row_index, post_id in enumerate(post_ids):
            locations[post_id] = (data, row_index)
    output: dict[str, Any] = {}
    target_ids = [example.post_id for example in examples]
    for field in fields:
        present = [field in data for data in loaded]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Row field {field} is missing from some shards")
        first = loaded[0][field]
        if getattr(first, "ndim", 0) < 1:
            continue
        merged = np.empty((len(target_ids), *first.shape[1:]), dtype=first.dtype)
        for output_index, post_id in enumerate(target_ids):
            data, row_index = locations[post_id]
            value = data[field]
            if value.shape[0] <= row_index:
                raise ValueError(f"Row field {field} is shorter than post_ids in a shard")
            merged[output_index] = value[row_index]
        output[field] = merged
    return output


def merge_vectors_no_gold(args: argparse.Namespace) -> None:
    """Assemble worker shards into one complete label-free NPZ file."""

    import numpy as np

    label_keys = parse_label_keys(args.label_keys)
    examples, load_info = load_unlabeled_examples(
        Path(args.data_file), args.record_start, args.max_records
    )
    if not examples:
        raise RuntimeError(f"No examples loaded: {load_info}")
    output_file = Path(args.output_file)
    if output_file.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Refusing to overwrite existing output_file without --overwrite 1: {output_file}"
        )
    shard_paths = [Path(path) for path in args.shard_file]
    arrays, shard_info = merge_shards_for_examples(np, shard_paths, examples, label_keys)
    # ``merge_shards_for_examples`` already validates no-gold provenance and
    # rejects serialized label arrays in every shard.
    require_predicted_shards(shard_info, shard_paths)
    row_fields = ["prompt_lengths", "post_token_counts"]
    with np.load(shard_paths[0], allow_pickle=False) as first_shard:
        row_fields.extend(key for key in first_shard.files if key.startswith("metric_"))
    payload: dict[str, Any] = {
        "created_at": np.array([timestamp()], dtype=str),
        "script": np.array([Path(__file__).name], dtype=str),
        "post_ids": np.array([example.post_id for example in examples], dtype=str),
        "subreddits": np.array([example.subreddit for example in examples], dtype=str),
        "label_keys": np.array(label_keys, dtype=str),
        "selected_layers": np.array(shard_info.get("selected_layers", []), dtype=np.int32),
        "attention_layers": np.array([shard_info.get("attention_layers", "")], dtype=str),
        "attention_answer_mode": np.array(["predicted"], dtype=str),
        "attention_query": np.array(["predicted_answer_token"], dtype=str),
        "gold_labels_used_for_prompt": np.array([0], dtype=np.int8),
        "gold_labels_in_output": np.array([0], dtype=np.int8),
        "labels_accessed_during_inference": np.array([0], dtype=np.int8),
        "classification_metrics_computed": np.array([0], dtype=np.int8),
        "answer_token_ids_saved": np.array([1], dtype=np.int8),
        "source_shard_count": np.array([len(shard_paths)], dtype=np.int32),
        "attention_extracted": np.array([1], dtype=np.int8),
        "hidden_states_extracted": np.array([1], dtype=np.int8),
        "raw_attention_maps_saved": np.array([0], dtype=np.int8),
        **{f"{key}_vectors": value for key, value in arrays.items()},
    }
    with np.load(shard_paths[0], allow_pickle=False) as first_shard:
        for key in (
            "max_qwen_length",
            "qwen_prefill_chunk_size",
            "answer_token_ids",
            "normalize_output_vectors",
            "ratio_eps",
            "max_ratio",
        ):
            if key in first_shard.files:
                payload[key] = first_shard[key]
    payload.update(merge_row_fields(np, shard_paths, examples, row_fields))
    save_npz(np, output_file, bool(args.compress), **payload)
    summary = {
        "created_at": timestamp(),
        "output_file": str(output_file),
        "example_count": len(examples),
        "label_keys": label_keys,
        "attention_answer_mode": "predicted",
        "attention_query": "predicted_answer_token",
        "gold_labels_used_for_prompt": False,
        "gold_labels_in_output": False,
        "labels_accessed_during_inference": False,
        "classification_metrics_computed": False,
        "saved_vector_keys": list(VECTOR_KEYS),
        "answer_token_ids_saved": "answer_token_ids" in payload,
        "max_qwen_length": int(payload["max_qwen_length"][0]) if "max_qwen_length" in payload else None,
        "qwen_prefill_chunk_size": int(payload["qwen_prefill_chunk_size"][0]) if "qwen_prefill_chunk_size" in payload else None,
        "load_info": load_info,
        "shard_info": shard_info,
        "attention_extracted": True,
        "hidden_states_extracted": True,
        "raw_attention_maps_saved": False,
        "source_shard_files": [str(path) for path in shard_paths],
    }
    write_json(output_file.with_suffix(".summary.json"), summary)
    print(f"[{timestamp()}] Wrote merged no-gold vectors: {output_file}", flush=True)


def parse_top_ks(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--top-ks must contain positive integers")
    return values


def method_scores(np: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        "qwen_base_hidden_post_mean": semantic_scores(np, arrays["base_hidden_post_mean"]),
        "qwen_base_hidden_adapter_attention_mean_label": mean_label_scores(np, arrays["base_hidden_adapter_attention"]),
        "qwen_base_hidden_delta_ratio_attention_mean_label": mean_label_scores(np, arrays["base_hidden_delta_ratio_attention"]),
        "qwen_hidden_attention_mean_label": mean_label_scores(np, arrays["adapter_hidden_attention"]),
        "qwen_delta_ratio_attention_mean_label": mean_label_scores(np, arrays["adapter_hidden_delta_ratio_attention"]),
        "qwen_hidden_attention_change_rate_mean_label": mean_label_scores(np, arrays["adapter_hidden_attention_change_rate"]),
    }


def vector_diagnostics(np: Any, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family, array in arrays.items():
        method = VECTOR_FAMILY_TO_METHOD[family]
        flat = array.reshape(array.shape[0], -1) if array.ndim == 3 else array
        norms = np.linalg.norm(flat, axis=1)
        finite = np.isfinite(array)
        rows.append(
            {
                "method": method,
                "family": family,
                "shape": "x".join(str(value) for value in array.shape),
                "finite_value_rate": float(np.count_nonzero(finite) / array.size) if array.size else 0.0,
                "nan_value_count": int(np.count_nonzero(np.isnan(array))),
                "inf_value_count": int(np.count_nonzero(np.isinf(array))),
                "l2_norm_mean": float(np.mean(norms)),
                "l2_norm_min": float(np.min(norms)),
                "l2_norm_max": float(np.max(norms)),
                "l2_norm_std": float(np.std(norms)),
                "zero_vector_count": int(np.sum(norms <= 1e-12)),
            }
        )
    return rows


def score_diagnostics(np: Any, scores_by_method: dict[str, Any], post_ids: list[str], top_k: int) -> list[dict[str, Any]]:
    post_id_values = np.array(post_ids, dtype=str)
    valid_mask = post_id_values[:, None] != post_id_values[None, :]
    rows = []
    for method_name, scores in scores_by_method.items():
        offdiag = np.asarray(scores)[valid_mask]
        finite_mask = np.isfinite(offdiag)
        finite_values = offdiag[finite_mask]
        row_tie_counts = []
        row_finite_counts = []
        for query_index, query_post_id in enumerate(post_ids):
            candidate_indices = np.flatnonzero(post_id_values != query_post_id)
            candidate_scores = np.asarray(scores)[query_index, candidate_indices]
            finite_row_scores = candidate_scores[np.isfinite(candidate_scores)]
            row_finite_counts.append(int(finite_row_scores.size))
            if finite_row_scores.size:
                best = float(np.max(finite_row_scores))
                row_tie_counts.append(int(np.count_nonzero(np.isclose(finite_row_scores, best, rtol=1e-7, atol=1e-8))))
            else:
                row_tie_counts.append(0)
        unique_rounded = int(np.unique(np.round(finite_values.astype(np.float64), 8)).size) if finite_values.size else 0
        rows.append(
            {
                "method": method_name,
                "score_shape": "x".join(str(value) for value in np.asarray(scores).shape),
                "offdiag_count": int(offdiag.size),
                "finite_score_rate": float(np.count_nonzero(finite_mask) / offdiag.size) if offdiag.size else 0.0,
                "nonfinite_score_count": int(offdiag.size - np.count_nonzero(finite_mask)),
                "finite_score_mean": float(np.mean(finite_values)) if finite_values.size else "",
                "finite_score_std": float(np.std(finite_values)) if finite_values.size else "",
                "finite_score_min": float(np.min(finite_values)) if finite_values.size else "",
                "finite_score_max": float(np.max(finite_values)) if finite_values.size else "",
                "finite_score_unique_rounded_8dp": unique_rounded,
                "all_finite_scores_equal_8dp": bool(unique_rounded <= 1),
                "row_finite_count_min": int(min(row_finite_counts)) if row_finite_counts else 0,
                "row_finite_count_mean": mean_or_zero([float(value) for value in row_finite_counts]),
                "top_score_tie_count_mean": mean_or_zero([float(value) for value in row_tie_counts]),
                f"top_score_tie_count_ge_{top_k}_rate": rate(sum(1 for value in row_tie_counts if value >= top_k), len(row_tie_counts)),
            }
        )
    return rows


def pairwise_method_checks(np: Any, scores_by_method: dict[str, Any], post_ids: list[str], top_k: int) -> list[dict[str, Any]]:
    methods = list(scores_by_method)
    ranks = {name: ranked_indices_excluding_same_post_id(np, scores, post_ids, top_k) for name, scores in scores_by_method.items()}
    post_id_values = np.array(post_ids, dtype=str)
    valid_mask = post_id_values[:, None] != post_id_values[None, :]
    rows = []
    for left_index, left_name in enumerate(methods):
        for right_name in methods[left_index + 1:]:
            left_scores = scores_by_method[left_name]
            right_scores = scores_by_method[right_name]
            diff = np.abs(left_scores[valid_mask] - right_scores[valid_mask])
            same_top1 = []
            same_ordered_topk = []
            set_jaccards = []
            for query_index in range(len(post_ids)):
                left_rank = [int(value) for value in ranks[left_name][query_index]]
                right_rank = [int(value) for value in ranks[right_name][query_index]]
                same_top1.append(float(bool(left_rank and right_rank and left_rank[0] == right_rank[0])))
                same_ordered_topk.append(float(left_rank[:top_k] == right_rank[:top_k]))
                left_set = set(left_rank[:top_k])
                right_set = set(right_rank[:top_k])
                union = len(left_set | right_set)
                set_jaccards.append(float(len(left_set & right_set) / union) if union else 0.0)
            rows.append(
                {
                    "left_method": left_name,
                    "right_method": right_name,
                    "top1_same_rate": mean_or_zero(same_top1),
                    f"top{top_k}_ordered_same_rate": mean_or_zero(same_ordered_topk),
                    f"top{top_k}_set_jaccard_mean": mean_or_zero(set_jaccards),
                    "score_mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
                    "score_max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                    "scores_exactly_equal": bool(np.array_equal(left_scores, right_scores)),
                }
            )
    return rows


def evaluate_fresh(args: argparse.Namespace) -> None:
    start_time = time.time()
    import numpy as np

    label_keys = parse_label_keys(args.label_keys)
    examples, load_info = load_examples(Path(args.data_file), Path(args.source_file) if args.source_file else None, label_keys, args.record_start, args.max_records)
    if not examples:
        raise RuntimeError(f"No examples loaded: {load_info}")
    print(f"[{timestamp()}] Fresh eval loaded {len(examples)} examples", flush=True)
    shard_paths = [Path(path) for path in args.shard_file]
    arrays, shard_info = merge_shards_for_examples(np, shard_paths, examples, label_keys)
    print(f"[{timestamp()}] Building labels and score matrices", flush=True)
    labels = label_matrix(np, examples, label_keys)
    top_ks = parse_top_ks(args.top_ks)
    max_rank = max(max(top_ks), int(args.max_rank))
    output_dir = Path(args.output_dir)
    prefix = args.output_prefix
    scores_by_method = method_scores(np, arrays)
    print(f"[{timestamp()}] Built {len(scores_by_method)} method score matrices", flush=True)

    method_summaries = []
    by_rank_rows = []
    query_rows = []
    for method_name, scores in scores_by_method.items():
        print(f"[{timestamp()}] Evaluating method {method_name}", flush=True)
        summary, rank_rows, rows = evaluate_method(
            np,
            method_name,
            scores,
            examples,
            labels,
            top_ks,
            args.query_results_top_k,
            max_rank,
        )
        method_summaries.append(summary)
        by_rank_rows.extend(rank_rows)
        query_rows.extend(rows)

    vector_rows = vector_diagnostics(np, arrays)
    score_rows = score_diagnostics(np, scores_by_method, [example.post_id for example in examples], max(top_ks))
    pair_rows = pairwise_method_checks(np, scores_by_method, [example.post_id for example in examples], max(top_ks))
    print(f"[{timestamp()}] Writing fresh eval outputs to {output_dir}", flush=True)

    method_csv_fields = ["method", "family", "total"]
    method_csv_fields.extend(f"top{k}_accuracy" for k in top_ks)
    method_csv_fields.extend(f"top{k}_correct" for k in top_ks)
    for metric_name in (
        "label_match_max_query_mean",
        "label_match_mean_query_mean",
        "label_match_min_query_mean",
        "positive_jaccard_max_query_mean",
        "positive_jaccard_mean_query_mean",
        "positive_jaccard_min_query_mean",
        "exact_label_hit_rate",
    ):
        method_csv_fields.extend(f"top{k}_{metric_name}" for k in top_ks)
    method_csv_fields.extend(["candidate_count_mean", "core", "score_definition"])
    method_rows = [{field: summary.get(field, "") for field in method_csv_fields} for summary in method_summaries]

    write_csv(output_dir / f"{prefix}_method_summary.csv", method_rows, method_csv_fields)
    write_csv(output_dir / f"{prefix}_by_rank.csv", by_rank_rows)
    write_csv(output_dir / f"{prefix}_vector_diagnostics.csv", vector_rows)
    write_csv(output_dir / f"{prefix}_score_diagnostics.csv", score_rows)
    write_csv(output_dir / f"{prefix}_method_pair_checks.csv", pair_rows)
    write_jsonl(output_dir / f"{prefix}_query_results.jsonl", query_rows)
    write_json(output_dir / f"{prefix}_method_definitions.json", METHOD_DEFINITIONS)
    write_json(
        output_dir / f"{prefix}_summary.json",
        {
            "created_at": timestamp(),
            "script": Path(__file__).name,
            "data_file": args.data_file,
            "source_file": args.source_file,
            "output_dir": str(output_dir),
            "output_prefix": prefix,
            "query_count": len(examples),
            "corpus_count": len(examples),
            "label_keys": label_keys,
            "top_ks": top_ks,
            "max_rank": max_rank,
            "method_definitions": METHOD_DEFINITIONS,
            "methods": method_summaries,
            "by_rank_file": str(output_dir / f"{prefix}_by_rank.csv"),
            "method_summary_file": str(output_dir / f"{prefix}_method_summary.csv"),
            "query_results_file": str(output_dir / f"{prefix}_query_results.jsonl"),
            "vector_diagnostics_file": str(output_dir / f"{prefix}_vector_diagnostics.csv"),
            "score_diagnostics_file": str(output_dir / f"{prefix}_score_diagnostics.csv"),
            "method_pair_checks_file": str(output_dir / f"{prefix}_method_pair_checks.csv"),
            "load_info": load_info,
            "shard_info": shard_info,
            "runtime_seconds": time.time() - start_time,
        },
    )
    print(f"[{timestamp()}] Fresh eval wrote {output_dir / f'{prefix}_summary.json'}", flush=True)
    for row in method_rows:
        top_parts = " ".join(f"top{k}={float(row.get(f'top{k}_accuracy') or 0):.4f}" for k in top_ks)
        print(f"{row['method']}: {top_parts}", flush=True)


def collect_root(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("last*/*_method_summary.csv")):
        layer = path.parent.name
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"layer": layer, **row, "source_file": str(path)})
    out = root / "fresh_layer_grid_method_summary.csv"
    if rows:
        write_csv(out, rows)
        print(f"[{timestamp()}] Wrote {out}", flush=True)
    else:
        print(f"[{timestamp()}] No method summaries found under {root}", flush=True)


def count_records(args: argparse.Namespace) -> None:
    label_keys = parse_label_keys(args.label_keys)
    if bool(getattr(args, "ignore_labels", 0)):
        examples, _ = load_unlabeled_examples(
            Path(args.data_file), args.record_start, args.max_records
        )
    else:
        examples, _ = load_examples(
            Path(args.data_file),
            Path(args.source_file) if args.source_file else None,
            label_keys,
            args.record_start,
            args.max_records,
        )
    print(len(examples))


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--source-file", default="")
    parser.add_argument("--label-keys", default="all")
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count")
    add_shared_args(count_parser)
    count_parser.add_argument(
        "--ignore-labels",
        type=int,
        choices=[1],
        default=1,
        help="Required no-gold preflight mode; label fields are never inspected.",
    )

    extract_parser = subparsers.add_parser("extract")
    add_shared_args(extract_parser)
    extract_parser.add_argument("--output-file", required=True)
    extract_parser.add_argument("--overwrite", type=int, default=0)
    extract_parser.add_argument("--qwen-model-name-or-path", default="models/Qwen3-8B")
    extract_parser.add_argument("--adapter-root", default="artifacts/classifiers")
    extract_parser.add_argument("--adapter-subdir", default="final")
    extract_parser.add_argument("--local-files-only", type=int, default=1)
    extract_parser.add_argument("--trust-remote-code", type=int, default=1)
    extract_parser.add_argument("--attn-implementation", default="eager")
    extract_parser.add_argument("--bf16", type=int, default=1)
    extract_parser.add_argument("--device-map", default="")
    extract_parser.add_argument("--qwen-max-memory", default="")
    extract_parser.add_argument("--qwen-device", default="")
    extract_parser.add_argument("--max-qwen-length", type=int, default=40960)
    extract_parser.add_argument("--qwen-prefill-chunk-size", type=int, default=2048)
    extract_parser.add_argument("--attention-layers", default="last:1")
    extract_parser.add_argument(
        "--attention-answer-mode",
        default="predicted",
        choices=["predicted"],
        help=(
            "Obtain the adapter's own 0/1 decision from the answer-less prompt "
            "and use that token as the attention query; dataset labels are never "
            "read during extraction."
        ),
    )
    extract_parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for --attention-answer-mode predicted.",
    )
    extract_parser.add_argument(
        "--fail-on-nonfinite",
        type=int,
        choices=[1],
        default=1,
        help="Required fail-fast behavior for non-finite forward/vector values.",
    )
    extract_parser.add_argument("--normalize-output-vectors", type=int, default=1)
    extract_parser.add_argument("--ratio-eps", type=float, default=1e-6)
    extract_parser.add_argument("--max-ratio", type=float, default=64.0)
    extract_parser.add_argument("--compress", type=int, default=1)

    eval_parser = subparsers.add_parser("evaluate")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--shard-file", action="append", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--output-prefix", default="fresh_qwen_pure")
    eval_parser.add_argument("--top-ks", default="1,3,5")
    eval_parser.add_argument("--query-results-top-k", type=int, default=5)
    eval_parser.add_argument("--max-rank", type=int, default=5)
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge no-gold worker NPZ files into one complete vector file.",
    )
    add_shared_args(merge_parser)
    merge_parser.add_argument("--shard-file", action="append", required=True)
    merge_parser.add_argument("--output-file", required=True)
    merge_parser.add_argument("--overwrite", type=int, default=0)
    merge_parser.add_argument("--compress", type=int, default=1)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--root", required=True)

    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.command == "count":
        count_records(args)
    elif args.command == "extract":
        extract_vectors(args)
    elif args.command == "evaluate":
        evaluate_fresh(args)
    elif args.command == "merge":
        merge_vectors_no_gold(args)
    elif args.command == "collect":
        collect_root(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
