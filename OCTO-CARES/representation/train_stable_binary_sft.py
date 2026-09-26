#!/usr/bin/env python3
"""Numerically guarded, single-label Qwen3 LoRA SFT.

This is a new implementation for the eight independent post classifiers.  It
does not import or resume any v2 adapter.  A label is represented as one
causal-LM answer token (``0`` or ``1``), and the loss is computed only at the
answer position.  The prompt is still processed by the backbone; answer-only
loss does not remove the cost of the long-context forward pass.

The default launcher fits all 1600 records (the records marked
``fixed_split=train`` *and* ``fixed_split=test``) and does not evaluate or hold
out the fixed 160-record partition.  This is intentional: the caller owns the
downstream test protocol.  An internal validation holdout and evaluation of
the fixed partition remain opt-in command-line features.

Important safety properties:

* one fresh base model and one fresh LoRA adapter per invocation;
* no silent fallback to full-sequence vocabulary logits;
* final-position hidden-state projection (or explicitly requested
  ``logits_to_keep``) for the answer loss;
* BF16 by default, conservative learning rate, gradient clipping, and
  finite-value checks on model outputs, gradients, adapter parameters, and
  optimizer state;
* a non-finite value aborts the run and writes ``nonfinite_abort.json``.  It is
  never converted to zero and no final adapter is saved after such an abort.

The companion Bash launcher binds one process to each GPU and runs two labels
sequentially on that GPU.  Run the launcher or this Python file on the A6000
server; this development environment does not execute model training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FRESHNEW_DIR = PROJECT_DIR / "annotation"
if str(FRESHNEW_DIR) not in sys.path:
    sys.path.insert(0, str(FRESHNEW_DIR))

from post_labels import DECISION_RULE_TEXT, POST_LABELS  # noqa: E402


LABEL_DEFS = {item.key: item for item in POST_LABELS}
LABEL_KEYS = [item.key for item in POST_LABELS]
DEFAULT_DATA_FILE = SCRIPT_DIR / "Eight_attention_dataset.json"
DEFAULT_MODEL_PATH = "models/qwen3-8B"

SYSTEM_PROMPT = """You are a careful research annotator.
Your task is to judge only one category for one Reddit post.
Use only the supplied category name, category definition, and Reddit post.
Consider the title and body together, including negation, quoted speech, sarcasm, and who is experiencing the event or emotion.
Return exactly one word: 0 or 1.
Do not return explanations, punctuation, bullet points, JSON, Markdown, or any other text."""


@dataclass(frozen=True)
class Example:
    post_id: str
    subreddit: str
    text: str
    label: int


@dataclass
class EncodedExample:
    example: Example
    input_ids: list[int]
    attention_mask: list[int]
    target_token_id: int
    prompt_len: int
    was_truncated: bool
    original_text_chars: int
    kept_text_chars: int


class NumericalError(RuntimeError):
    """Raised when a tensor, gradient, parameter, or optimizer value is non-finite."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one numerically guarded Qwen3 binary answer-token adapter."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--label-key", required=True, choices=LABEL_KEYS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--local-files-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=[0, 1], default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--answer-projection",
        choices=["hidden", "logits_to_keep"],
        default="hidden",
        help=(
            "hidden projects only the final prompt hidden state through lm_head; "
            "logits_to_keep asks the Qwen causal-LM forward for one final logit row."
        ),
    )
    parser.add_argument("--max-length", type=int, default=42000)
    parser.add_argument("--strict-context", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--post-truncation-strategy",
        choices=["head", "tail", "head_tail"],
        default="head_tail",
    )
    parser.add_argument(
        "--data-split",
        choices=["train", "all"],
        default="all",
        help=(
            "Fit fixed_split=train only, or use all fixed train+test records. "
            "The default is all because the caller supplies the test protocol."
        ),
    )
    parser.add_argument(
        "--expected-train-count",
        type=int,
        default=1440,
        help="Expected fixed_split=train count used for dataset integrity checks.",
    )
    parser.add_argument(
        "--expected-test-count",
        type=int,
        default=160,
        help="Expected fixed_split=test count used for dataset integrity checks.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help=(
            "Internal validation fraction from the selected fit data; 0 keeps "
            "all selected records in the optimizer (default: 0)."
        ),
    )
    parser.add_argument(
        "--evaluate-fixed-test",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Evaluate the records marked fixed_split=test after fitting. "
            "This is disabled by default because the caller owns testing."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260829)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Balance 0/1 contributions in the training answer-token NLL.",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=5.0,
        help="Upper bound before re-normalizing inverse-frequency class weights.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--gradient-checkpointing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--detect-anomaly", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing final adapter in this new output directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load/tokenize/check the model and data, then exit without optimizer steps.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot decode UTF-8 JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    else:
        raise ValueError(f"{path} must contain a JSON list or a records list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains a non-object record")
    return records  # type: ignore[return-value]


def coerce_binary(value: Any) -> int | None:
    if value in (0, 0.0, False):
        return 0
    if value in (1, 1.0, True):
        return 1
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def post_id(record: dict[str, Any]) -> str:
    for key in ("post_id", "id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    source_post = record.get("source_post")
    if isinstance(source_post, dict):
        return str(source_post.get("id") or source_post.get("post_id") or "").strip()
    return ""


def label_value(record: dict[str, Any], label_key: str) -> int | None:
    labels = record.get("labels")
    if isinstance(labels, dict):
        value = coerce_binary(labels.get(label_key))
        if value is not None:
            return value
    vector = record.get("label_vector")
    if isinstance(vector, list) and len(vector) == len(LABEL_KEYS):
        return coerce_binary(vector[LABEL_KEYS.index(label_key)])
    return None


def record_text(record: dict[str, Any]) -> str:
    value = record.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    title = str(record.get("title") or "").strip()
    body = str(record.get("body") or "").strip()
    return "\n\n".join(part for part in (title, body) if part)


def load_examples(
    data_file: Path,
    label_key: str,
    data_split: str,
    expected_train_count: int,
    expected_test_count: int,
) -> tuple[list[Example], list[Example], dict[str, Any]]:
    records = records_from_payload(read_json(data_file), data_file)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": [], "other": []}
    for record in records:
        fixed_split = str(record.get("fixed_split") or "").strip()
        if fixed_split == "train":
            by_split["train"].append(record)
        elif fixed_split == "test":
            by_split["test"].append(record)
        else:
            by_split["other"].append(record)

    if len(by_split["train"]) != expected_train_count:
        raise ValueError(
            f"Expected {expected_train_count} fixed_split=train records, "
            f"found {len(by_split['train'])} in {data_file}"
        )
    if len(by_split["test"]) != expected_test_count:
        raise ValueError(
            f"Expected {expected_test_count} fixed_split=test records, "
            f"found {len(by_split['test'])} in {data_file}"
        )
    if by_split["other"]:
        raise ValueError(f"Found {len(by_split['other'])} records without fixed_split=train/test")

    seen: set[str] = set()
    skipped: list[dict[str, Any]] = []

    def convert(source_records: list[dict[str, Any]], split_name: str) -> list[Example]:
        output: list[Example] = []
        for index, record in enumerate(source_records):
            identifier = post_id(record)
            value = label_value(record, label_key)
            text = record_text(record)
            if not identifier or not text or value is None:
                skipped.append(
                    {
                        "split": split_name,
                        "index": index,
                        "post_id": identifier,
                        "reason": "missing_post_id_text_or_label",
                    }
                )
                continue
            if identifier in seen:
                raise ValueError(f"Duplicate post_id={identifier} in {data_file}")
            seen.add(identifier)
            source_post = record.get("source_post") if isinstance(record.get("source_post"), dict) else {}
            subreddit = str(
                record.get("subreddit")
                or record.get("source_subreddit")
                or source_post.get("subreddit")
                or "unknown"
            )
            output.append(Example(identifier, subreddit, text, value))
        return output

    train_examples = convert(by_split["train"], "train")
    test_examples = convert(by_split["test"], "test")
    if not train_examples:
        raise ValueError(f"No usable training records for {label_key}")
    if not test_examples:
        raise ValueError(f"No usable test records for {label_key}")

    fit_examples = train_examples if data_split == "train" else train_examples + test_examples
    info = {
        "data_file": str(data_file.resolve()),
        "label_key": label_key,
        "label_name": LABEL_DEFS[label_key].name_en,
        "definition": LABEL_DEFS[label_key].definition_en,
        "decision_rule": DECISION_RULE_TEXT,
        "raw_record_count": len(records),
        "fixed_train_count": len(train_examples),
        "fixed_test_count": len(test_examples),
        "fit_source": data_split,
        "fit_count_before_validation_split": len(fit_examples),
        "test_is_seen_during_fit": data_split == "all",
        "skipped_count": len(skipped),
        "skipped_first_20": skipped[:20],
    }
    return fit_examples, test_examples, info


def split_train_validation(
    examples: list[Example], validation_ratio: float, seed: int
) -> tuple[list[Example], list[Example], dict[str, Any]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 (inclusive) and 1")
    positives = [item for item in examples if item.label == 1]
    negatives = [item for item in examples if item.label == 0]
    if not positives or not negatives:
        raise ValueError("Each label needs at least one positive and one negative example")
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if validation_ratio == 0.0:
        # No internal holdout: every selected fit record contributes to the
        # optimizer.  Keep the class-stratified ordering deterministic, while
        # leaving the actual per-epoch shuffle to ``batch_iter``.
        train = positives + negatives
        rng.shuffle(train)
        return train, [], {
            "split_seed": seed,
            "validation_ratio": validation_ratio,
            "train_count": len(train),
            "validation_count": 0,
            "train_positive": sum(item.label for item in train),
            "train_negative": sum(item.label == 0 for item in train),
            "validation_positive": 0,
            "validation_negative": 0,
            "train_post_ids": [item.post_id for item in train],
            "validation_post_ids": [],
        }
    validation_size = max(1, int(round(len(examples) * validation_ratio)))
    positive_validation = min(len(positives) - 1, max(1, int(round(len(positives) * validation_ratio))))
    negative_validation = validation_size - positive_validation
    if negative_validation < 1:
        negative_validation = 1
        positive_validation = validation_size - negative_validation
    if positive_validation < 1:
        positive_validation = 1
        negative_validation = validation_size - positive_validation
    if positive_validation >= len(positives) or negative_validation >= len(negatives):
        raise ValueError("Validation split would remove an entire class from training")

    validation = positives[:positive_validation] + negatives[:negative_validation]
    train = positives[positive_validation:] + negatives[negative_validation:]
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation, {
        "split_seed": seed,
        "validation_ratio": validation_ratio,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_positive": sum(item.label for item in train),
        "train_negative": sum(item.label == 0 for item in train),
        "validation_positive": sum(item.label for item in validation),
        "validation_negative": sum(item.label == 0 for item in validation),
        "train_post_ids": [item.post_id for item in train],
        "validation_post_ids": [item.post_id for item in validation],
    }


def calculate_class_weights(
    examples: list[Example], mode: str, max_weight: float
) -> dict[str, float | int | str]:
    if max_weight <= 0 or not math.isfinite(max_weight):
        raise ValueError("--max-class-weight must be a positive finite number")
    positive = sum(item.label for item in examples)
    negative = len(examples) - positive
    if positive <= 0 or negative <= 0:
        raise ValueError("Cannot calculate binary class weights with an empty class")
    if mode == "none":
        w0 = w1 = 1.0
    else:
        total = len(examples)
        # Inverse-frequency weights with expected mean approximately one.
        w0 = total / (2.0 * negative)
        w1 = total / (2.0 * positive)
        w0 = min(w0, max_weight)
        w1 = min(w1, max_weight)
        mean_weight = (negative * w0 + positive * w1) / total
        w0 /= mean_weight
        w1 /= mean_weight
    return {
        "mode": mode,
        "positive_count": positive,
        "negative_count": negative,
        "weight_for_0": float(w0),
        "weight_for_1": float(w1),
        "max_class_weight_before_renormalization": float(max_weight),
    }


def make_prompt(label_key: str, text: str) -> str:
    label = LABEL_DEFS[label_key]
    return f"""Decide whether the Reddit post contains this one category.

Category:
{label.name_en}

Definition:
{label.definition_en}

Decision rule:
{DECISION_RULE_TEXT}

Reddit post:
<post>
{text}
</post>

Answer:"""


def render_prompt(tokenizer: Any, label_key: str, text: str) -> str:
    prompt = make_prompt(label_key, text)
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
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


def truncate_text(text: str, max_chars: int, strategy: str) -> str:
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


def token_ids(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    """Return one unbatched list of token IDs from any HF tokenizer output.

    ``AutoTokenizer`` normally returns a ``transformers.BatchEncoding``.  That
    object is mapping-like but is *not* necessarily an ``isinstance(..., dict)``
    object, which made the earlier implementation reject perfectly valid
    Qwen output.  Some environments additionally return a one-row tensor or
    numpy array, so normalize those representations here without importing
    torch/numpy into the tokenizer-only path.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )

    values: Any = None
    if hasattr(encoded, "__getitem__"):
        try:
            values = encoded["input_ids"]
        except (KeyError, IndexError, TypeError):
            values = None
    if values is None and hasattr(encoded, "input_ids"):
        values = encoded.input_ids
    if values is None and hasattr(encoded, "get"):
        values = encoded.get("input_ids")

    # Convert torch tensors, numpy arrays, and similar scalar containers to
    # ordinary Python objects.  ``detach``/``cpu`` are intentionally checked
    # dynamically so this helper remains usable with tokenizer-only tests.
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)

    if not isinstance(values, list):
        output_type = type(encoded).__name__
        value_type = type(values).__name__ if values is not None else "None"
        raise ValueError(
            "Tokenizer did not return input_ids as a list-like value "
            f"(encoding_type={output_type}, value_type={value_type})"
        )

    # A tokenizer called on one string should return [seq].  Accept a one-row
    # batch defensively, but reject a multi-row result because it would make
    # prompt/record alignment ambiguous.
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError(
                f"Expected one unbatched tokenizer result, got {len(values)} rows"
            )
        values = list(values[0])
    if any(isinstance(value, (list, tuple, dict)) for value in values):
        raise ValueError("Tokenizer returned nested input_ids beyond a single row")
    try:
        output = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Tokenizer input_ids contains a non-integer token ID") from exc
    if not output:
        raise ValueError("Tokenizer returned an empty input_ids sequence")
    return output


def fit_prompt(
    tokenizer: Any,
    label_key: str,
    text: str,
    max_prompt_length: int,
    strategy: str,
) -> tuple[str, list[int], dict[str, Any]]:
    full_prompt = render_prompt(tokenizer, label_key, text)
    full_ids = token_ids(tokenizer, full_prompt, add_special_tokens=True)
    if len(full_ids) <= max_prompt_length:
        return full_prompt, full_ids, {
            "was_truncated": False,
            "original_text_chars": len(text),
            "kept_text_chars": len(text),
        }

    empty_prompt = render_prompt(tokenizer, label_key, "")
    empty_ids = token_ids(tokenizer, empty_prompt, add_special_tokens=True)
    if len(empty_ids) > max_prompt_length:
        raise ValueError(
            f"Prompt template alone needs {len(empty_ids)} tokens, "
            f"but max prompt length is {max_prompt_length}"
        )

    low = 0
    high = len(text)
    best_text = ""
    best_prompt = empty_prompt
    best_ids = empty_ids
    while low <= high:
        middle = (low + high) // 2
        candidate_text = truncate_text(text, middle, strategy)
        candidate_prompt = render_prompt(tokenizer, label_key, candidate_text)
        candidate_ids = token_ids(tokenizer, candidate_prompt, add_special_tokens=True)
        if len(candidate_ids) <= max_prompt_length:
            best_text = candidate_text
            best_prompt = candidate_prompt
            best_ids = candidate_ids
            low = middle + 1
        else:
            high = middle - 1
    return best_prompt, best_ids, {
        "was_truncated": True,
        "original_text_chars": len(text),
        "kept_text_chars": len(best_text),
    }


def binary_target_token_ids(tokenizer: Any) -> dict[str, int]:
    zero = token_ids(tokenizer, "0", add_special_tokens=False)
    one = token_ids(tokenizer, "1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero[0] == one[0]:
        raise ValueError(f"Qwen tokenizer must encode 0 and 1 as distinct single tokens: {zero}, {one}")
    return {"0": zero[0], "1": one[0]}


def encode_examples(
    examples: list[Example],
    tokenizer: Any,
    label_key: str,
    max_prompt_length: int,
    truncation_strategy: str,
    target_token_ids: dict[str, int],
) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []
    for example in examples:
        _, prompt_ids, info = fit_prompt(
            tokenizer,
            label_key,
            example.text,
            max_prompt_length,
            truncation_strategy,
        )
        encoded.append(
            EncodedExample(
                example=example,
                input_ids=prompt_ids,
                attention_mask=[1] * len(prompt_ids),
                target_token_id=target_token_ids[str(example.label)],
                prompt_len=len(prompt_ids),
                was_truncated=bool(info["was_truncated"]),
                original_text_chars=int(info["original_text_chars"]),
                kept_text_chars=int(info["kept_text_chars"]),
            )
        )
    return encoded


def collate(
    batch: list[EncodedExample],
    tokenizer: Any,
    torch: Any,
) -> dict[str, Any]:
    max_len = max(len(item.input_ids) for item in batch)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id")
    input_ids = []
    attention_masks = []
    target_token_ids = []
    binary_labels = []
    post_ids = []
    for item in batch:
        padding = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [int(tokenizer.pad_token_id)] * padding)
        attention_masks.append(item.attention_mask + [0] * padding)
        target_token_ids.append(item.target_token_id)
        binary_labels.append(item.example.label)
        post_ids.append(item.example.post_id)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "target_token_ids": torch.tensor(target_token_ids, dtype=torch.long),
        "binary_labels": torch.tensor(binary_labels, dtype=torch.long),
        "post_ids": post_ids,
    }


def batch_iter(
    rows: list[EncodedExample], batch_size: int, shuffle: bool, seed: int
) -> Iterable[list[EncodedExample]]:
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_training_stack() -> tuple[Any, ...]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, LoraConfig, TaskType, get_peft_model


def require_single_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script requires one A6000 per process")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible CUDA device per process, "
            f"found {torch.cuda.device_count()}. Bind the process with CUDA_VISIBLE_DEVICES."
        )
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def precision_settings(args: argparse.Namespace, torch: Any) -> tuple[Any, bool]:
    if args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this CUDA device; choose --precision fp16 explicitly")
        return torch.bfloat16, True
    if args.precision == "fp16":
        return torch.float16, True
    return torch.float32, False


def model_context_limit(config: Any) -> int | None:
    for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0 and value < 10**8:
            return value
    return None


def load_model(
    args: argparse.Namespace,
    torch: Any,
    AutoConfig: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    LoraConfig: Any,
    TaskType: Any,
    get_peft_model: Any,
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    model_path = str(Path(args.model_name_or_path).expanduser())
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    dtype, autocast_enabled = precision_settings(args, torch)
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    context_limit = model_context_limit(config)
    if context_limit is not None and args.max_length > context_limit:
        message = (
            f"Requested max_length={args.max_length} exceeds model context limit "
            f"max_position_embeddings={context_limit}; effective prompt limit will be capped."
        )
        if bool(args.strict_context):
            raise ValueError(message)
        print(f"WARNING: {message}", flush=True)
    effective_max_length = min(args.max_length, context_limit) if context_limit else args.max_length
    if effective_max_length < 2:
        raise ValueError("Effective max_length must leave at least one prompt token and one answer token")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "attn_implementation": args.attn_implementation,
    }
    try:
        language_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "This Transformers version/model did not accept the requested attention implementation. "
            "Upgrade the environment or choose a supported --attn-implementation; "
            "the stable script will not silently fall back to eager attention."
        ) from exc
    language_model.to(device)
    language_model.config.use_cache = False
    if bool(args.gradient_checkpointing):
        if not hasattr(language_model, "gradient_checkpointing_enable"):
            raise RuntimeError("Model does not expose gradient_checkpointing_enable")
        language_model.gradient_checkpointing_enable()
        if hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(language_model, lora_config)
    model.to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        # PEFT names newly created LoRA matrices with ``lora_``.  Re-enable
        # only those matrices after freezing the complete base model.
        if "lora_" in name:
            parameter.requires_grad = True
            # Keep the small trainable adapter in FP32; autocast handles its
            # matrix multiplications while AdamW retains stable FP32 state.
            parameter.data = parameter.data.float()
            if not parameter.data.is_contiguous():
                parameter.data = parameter.data.contiguous()
            trainable_names.append(name)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA produced no trainable parameters")
    model_info = {
        "model_name_or_path": model_path,
        "model_type": getattr(config, "model_type", ""),
        "context_limit": context_limit,
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "prompt_max_length": effective_max_length - 1,
        "precision": args.precision,
        "torch_dtype": str(dtype),
        "autocast_enabled": autocast_enabled,
        "attention_implementation": args.attn_implementation,
        "answer_projection": args.answer_projection,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_names_first_20": trainable_names[:20],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    return model, tokenizer, model_info


def all_parameters_finite(model: Any, torch: Any, include_frozen: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if not include_frozen and not parameter.requires_grad:
            continue
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise NumericalError(
                f"Non-finite parameter detected: {name}",
                {"phase": "parameter_check", "parameter": name},
            )


def require_finite_tensor(
    tensor: Any,
    name: str,
    context: dict[str, Any],
    torch: Any,
) -> None:
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        details = dict(context)
        details.update(
            {
                "tensor": name,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "nonfinite_count": int((~finite).sum().item()),
                "nan_count": int(torch.isnan(detached).sum().item()),
                "inf_count": int(torch.isinf(detached).sum().item()),
            }
        )
        raise NumericalError(f"Non-finite tensor detected: {name}", details)


def require_finite_gradients(
    model: Any, torch: Any, context: dict[str, Any]
) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        require_finite_tensor(parameter.grad, f"gradient:{name}", context, torch)


def require_finite_optimizer_state(
    optimizer: Any, torch: Any, context: dict[str, Any]
) -> None:
    for state_index, state in enumerate(optimizer.state.values()):
        for state_name, value in state.items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                require_finite_tensor(
                    value,
                    f"optimizer_state:{state_index}:{state_name}",
                    context,
                    torch,
                )


def unwrap_base_causal_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def hidden_answer_logits(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    base = unwrap_base_causal_model(model)
    core = None
    lm_head = None
    # Depending on PEFT version, get_base_model() may return the causal-LM
    # itself or a wrapper around it.  Select a pair from the same wrapper so
    # LoRA hooks remain active and the final projection is never materialized
    # for every sequence position.
    candidates = [base]
    for attribute in ("model", "base_model"):
        candidate = getattr(base, attribute, None)
        if candidate is not None and all(candidate is not existing for existing in candidates):
            candidates.append(candidate)
    for candidate in candidates:
        candidate_head = getattr(candidate, "lm_head", None)
        candidate_core = None
        for attribute in ("model", "transformer", "decoder"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                candidate_core = nested
                break
        if candidate_head is not None and candidate_core is not None:
            lm_head = candidate_head
            core = candidate_core
            break
    if core is None or lm_head is None:
        raise RuntimeError(
            "Cannot locate the base transformer and lm_head for hidden answer projection; "
            "use a Qwen3 CausalLM checkpoint with the standard model/lm_head structure."
        )
    outputs = core(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, (tuple, list)):
        hidden = outputs[0]
    if hidden is None:
        raise RuntimeError("Base transformer returned no last_hidden_state")
    last_indices = attention_mask.long().sum(dim=1) - 1
    if bool((last_indices < 0).any().item()):
        raise RuntimeError("An input row has no non-padding token")
    batch_indices = torch_arange_like(last_indices, hidden)
    pooled = hidden[batch_indices, last_indices.to(hidden.device)]
    return lm_head(pooled)


def torch_arange_like(indices: Any, reference: Any) -> Any:
    # Keep the row indices on the same device as the hidden states.  A tensor
    # of zeros is not sufficient here: with batch_size > 1 it would always
    # select row zero from the hidden-state batch.
    import torch

    return torch.arange(indices.shape[0], device=reference.device, dtype=indices.dtype)


def answer_logits(
    model: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
) -> Any:
    if args.answer_projection == "hidden":
        return hidden_answer_logits(model, batch["input_ids"], batch["attention_mask"])
    try:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    except TypeError as exc:
        raise RuntimeError(
            "The selected model does not support logits_to_keep=1. "
            "Use --answer-projection hidden; full-sequence logits fallback is disabled."
        ) from exc
    logits = getattr(outputs, "logits", None)
    if logits is None or logits.ndim != 3 or logits.shape[1] != 1:
        shape = None if logits is None else list(logits.shape)
        raise RuntimeError(f"logits_to_keep=1 returned unexpected logits shape: {shape}")
    return logits[:, 0, :]


def weighted_answer_loss(
    logits: Any,
    target_token_ids: Any,
    binary_labels: Any,
    class_weights: dict[str, Any],
    torch: Any,
) -> tuple[Any, Any]:
    logits_float = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits_float, dim=-1)
    per_example_nll = -log_probs.gather(1, target_token_ids.to(log_probs.device).view(-1, 1)).squeeze(1)
    weights = torch.where(
        binary_labels.to(log_probs.device).eq(1),
        torch.as_tensor(class_weights["weight_for_1"], dtype=log_probs.dtype, device=log_probs.device),
        torch.as_tensor(class_weights["weight_for_0"], dtype=log_probs.dtype, device=log_probs.device),
    )
    weighted_loss = (per_example_nll * weights).mean()
    unweighted_loss = per_example_nll.mean()
    return weighted_loss, unweighted_loss


def autocast_context(torch: Any, precision: str, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def binary_metrics(
    labels: list[int], probabilities: list[float], threshold: float, losses: list[float]
) -> dict[str, Any]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(y == 1 and prediction == 1 for y, prediction in zip(labels, predictions))
    tn = sum(y == 0 and prediction == 0 for y, prediction in zip(labels, predictions))
    fp = sum(y == 0 and prediction == 1 for y, prediction in zip(labels, predictions))
    fn = sum(y == 1 and prediction == 0 for y, prediction in zip(labels, predictions))
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * positive_recall / (precision + positive_recall) if precision + positive_recall else 0.0
    return {
        "count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "precision": precision,
        "recall": positive_recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_unweighted_answer_nll": sum(losses) / len(losses) if losses else None,
        "predictions": predictions,
    }


def evaluate_model(
    model: Any,
    rows: list[EncodedExample],
    tokenizer: Any,
    args: argparse.Namespace,
    torch: Any,
    device: Any,
    target_token_ids: dict[str, int],
    class_weights: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    losses: list[float] = []
    weighted_losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    zero_id = int(target_token_ids["0"])
    one_id = int(target_token_ids["1"])
    autocast_enabled = args.precision in {"bf16", "fp16"}
    with torch.no_grad():
        for batch_items in batch_iter(rows, args.batch_size, shuffle=False, seed=0):
            batch = move_batch(collate(batch_items, tokenizer, torch), device)
            context = {"phase": "evaluation", "split": split_name, "post_ids": batch["post_ids"]}
            with autocast_context(torch, args.precision, autocast_enabled):
                logits = answer_logits(model, batch, args, torch)
            require_finite_tensor(logits, "answer_logits", context, torch)
            weighted_loss, unweighted_loss = weighted_answer_loss(
                logits,
                batch["target_token_ids"],
                batch["binary_labels"],
                class_weights,
                torch,
            )
            require_finite_tensor(unweighted_loss, "unweighted_loss", context, torch)
            require_finite_tensor(weighted_loss, "weighted_loss", context, torch)
            pair_logits = logits.float()[:, [zero_id, one_id]]
            pair_probabilities = torch.softmax(pair_logits, dim=-1)[:, 1]
            global_top_ids = logits.float().argmax(dim=-1)
            require_finite_tensor(pair_probabilities, "binary_probability", context, torch)
            labels.extend(int(value) for value in batch["binary_labels"].detach().cpu().tolist())
            probabilities.extend(float(value) for value in pair_probabilities.detach().cpu().tolist())
            losses.extend(float(value) for value in (-torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(1, batch["target_token_ids"].view(-1, 1)).squeeze(1)).detach().cpu().tolist())
            weighted_losses.extend([float(weighted_loss.detach().cpu())] * len(batch_items))
            for item, probability, top_id in zip(
                batch_items,
                pair_probabilities.detach().cpu().tolist(),
                global_top_ids.detach().cpu().tolist(),
            ):
                prediction_rows.append(
                    {
                        "post_id": item.example.post_id,
                        "label": item.example.label,
                        "probability_1_within_0_1": float(probability),
                        "prediction": int(float(probability) >= args.threshold),
                        "global_top_token_id": int(top_id),
                        "global_top_token_is_binary_answer": int(int(top_id) in {zero_id, one_id}),
                        "was_truncated": item.was_truncated,
                    }
                )
    metrics = binary_metrics(labels, probabilities, args.threshold, losses)
    metrics["split"] = split_name
    metrics["mean_weighted_answer_nll"] = sum(weighted_losses) / len(weighted_losses) if weighted_losses else None
    metrics["invalid_global_top_token_count"] = sum(
        not bool(row["global_top_token_is_binary_answer"]) for row in prediction_rows
    )
    metrics["invalid_global_top_token_rate"] = (
        metrics["invalid_global_top_token_count"] / len(prediction_rows) if prediction_rows else 0.0
    )
    metrics["predictions"] = prediction_rows
    return metrics


def truncation_info(rows: list[EncodedExample]) -> dict[str, Any]:
    lengths = [len(row.input_ids) for row in rows]
    truncated = [row for row in rows if row.was_truncated]
    return {
        "count": len(rows),
        "truncated_count": len(truncated),
        "truncated_rate": len(truncated) / len(rows) if rows else 0.0,
        "token_length_min": min(lengths) if lengths else None,
        "token_length_median": statistics.median(lengths) if lengths else None,
        "token_length_p95": sorted(lengths)[max(0, math.ceil(len(lengths) * 0.95) - 1)] if lengths else None,
        "token_length_max": max(lengths) if lengths else None,
        "truncated_first_20": [
            {
                "post_id": row.example.post_id,
                "original_text_chars": row.original_text_chars,
                "kept_text_chars": row.kept_text_chars,
                "prompt_tokens": row.prompt_len,
            }
            for row in truncated[:20]
        ],
    }


def capture_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: Any, state: dict[str, Any], torch: Any) -> None:
    current = dict(model.named_parameters())
    for name, value in state.items():
        if name not in current:
            raise RuntimeError(f"Best-state parameter disappeared before restore: {name}")
        current[name].data.copy_(value.to(current[name].device, dtype=current[name].dtype))
    all_parameters_finite(model, torch, include_frozen=False)


def make_optimizer_and_scheduler(
    model: Any,
    args: argparse.Namespace,
    torch: Any,
    total_updates: int,
) -> tuple[Any, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (0.9, 0.95),
        "eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "foreach": False,
    }
    try:
        optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("foreach")
        optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    warmup_steps = max(0, min(warmup_steps, max(0, total_updates - 1)))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        remaining = max(1, total_updates - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def optimizer_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    args: argparse.Namespace,
    torch: Any,
    context: dict[str, Any],
) -> float:
    if scaler is not None:
        scaler.unscale_(optimizer)
    require_finite_gradients(model, torch, context)
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
            error_if_nonfinite=True,
        )
    except TypeError:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        require_finite_tensor(grad_norm, "gradient_norm", context, torch)
    require_finite_tensor(grad_norm, "gradient_norm", context, torch)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    all_parameters_finite(model, torch, include_frozen=False)
    require_finite_optimizer_state(optimizer, torch, context)
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().cpu())


def save_predictions(path: Path, metrics: dict[str, Any]) -> None:
    write_json(path, metrics.get("predictions", []))


def run(args: argparse.Namespace) -> None:
    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than 1")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0 or args.num_epochs <= 0:
        raise ValueError("batch size, gradient accumulation, and epochs must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be a positive finite number")
    if args.max_grad_norm <= 0 or not math.isfinite(args.max_grad_norm):
        raise ValueError("--max-grad-norm must be a positive finite number")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    output_dir = args.output_dir.expanduser().resolve()
    final_dir = output_dir / "final"
    if final_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing adapter: {final_dir}. Use --overwrite only for this new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_started.json",
        {"created_at": timestamp(), "label_key": args.label_key, "pid": os.getpid()},
    )

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, LoraConfig, TaskType, get_peft_model = import_training_stack()
    device = require_single_cuda(torch)
    set_seed(args.seed, torch)
    if bool(args.detect_anomaly):
        torch.autograd.set_detect_anomaly(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    fit_examples, test_examples, load_info = load_examples(
        args.data_file.expanduser().resolve(),
        args.label_key,
        args.data_split,
        args.expected_train_count,
        args.expected_test_count,
    )
    expected_fit_count = (
        args.expected_train_count + args.expected_test_count
        if args.data_split == "all"
        else args.expected_train_count
    )
    if len(fit_examples) != expected_fit_count:
        raise ValueError(
            f"Selected fit data contains {len(fit_examples)} usable records, "
            f"expected {expected_fit_count}; no records may be skipped for this run"
        )
    train_examples, validation_examples, split_info = split_train_validation(
        fit_examples,
        args.validation_ratio,
        args.split_seed,
    )
    class_weights = calculate_class_weights(train_examples, args.class_weight, args.max_class_weight)
    model, tokenizer, model_info = load_model(
        args,
        torch,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
        device,
    )
    all_parameters_finite(model, torch, include_frozen=True)
    target_token_ids = binary_target_token_ids(tokenizer)
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    if any(value < 0 or value >= vocab_size for value in target_token_ids.values()):
        raise ValueError(f"Target token IDs are outside output vocabulary: {target_token_ids}, vocab={vocab_size}")
    max_prompt_length = int(model_info["prompt_max_length"])
    train_encoded = encode_examples(
        train_examples,
        tokenizer,
        args.label_key,
        max_prompt_length,
        args.post_truncation_strategy,
        target_token_ids,
    )
    validation_encoded = encode_examples(
        validation_examples,
        tokenizer,
        args.label_key,
        max_prompt_length,
        args.post_truncation_strategy,
        target_token_ids,
    )
    test_encoded: list[EncodedExample] = []
    if bool(args.evaluate_fixed_test):
        test_encoded = encode_examples(
            test_examples,
            tokenizer,
            args.label_key,
            max_prompt_length,
            args.post_truncation_strategy,
            target_token_ids,
        )
    for rows in (train_encoded, validation_encoded, test_encoded):
        for row in rows:
            if not row.input_ids or len(row.input_ids) > max_prompt_length:
                raise ValueError(f"Encoded prompt length is invalid for post_id={row.example.post_id}")

    run_info = {
        "created_at": timestamp(),
        "label_key": args.label_key,
        "label_name": LABEL_DEFS[args.label_key].name_en,
        "definition": LABEL_DEFS[args.label_key].definition_en,
        "decision_rule": DECISION_RULE_TEXT,
        "system_prompt": SYSTEM_PROMPT,
        "args": {key: str(value) for key, value in vars(args).items()},
        "load_info": load_info,
        "model_info": model_info,
        "target_token_ids": target_token_ids,
        "vocab_size": vocab_size,
        "class_weights": class_weights,
        "split_info": split_info,
        "fit_source": args.data_split,
        "fit_record_count": len(fit_examples),
        "optimizer_train_count": len(train_examples),
        "validation_count": len(validation_examples),
        "fixed_test_count": len(test_examples),
        "fixed_test_evaluation_enabled": bool(args.evaluate_fixed_test),
        "external_test_protocol_required": not bool(args.evaluate_fixed_test),
        "truncation": {
            "train": truncation_info(train_encoded),
            "validation": truncation_info(validation_encoded),
            "test": truncation_info(test_encoded),
        },
    }
    write_json(output_dir / "run_info.json", run_info)
    write_json(output_dir / "split_info.json", split_info)
    write_json(output_dir / "class_weights.json", class_weights)
    write_json(output_dir / "truncation_info.json", run_info["truncation"])
    write_json(output_dir / "target_token_ids.json", target_token_ids)
    (output_dir / "prompt_preview.txt").write_text(
        render_prompt(tokenizer, args.label_key, "Example post text.")
        + "\n0/1 target token IDs: "
        + json.dumps(target_token_ids)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"load": load_info, "model": model_info, "class_weights": class_weights}, ensure_ascii=False), flush=True)
    if args.preflight_only:
        probe = max(train_encoded, key=lambda item: len(item.input_ids))
        probe_batch = move_batch(collate([probe], tokenizer, torch), device)
        probe_context = {
            "phase": "preflight",
            "post_id": probe.example.post_id,
            "prompt_tokens": probe.prompt_len,
        }
        model.eval()
        with torch.no_grad(), autocast_context(
            torch,
            args.precision,
            args.precision in {"bf16", "fp16"},
        ):
            probe_logits = answer_logits(model, probe_batch, args, torch)
        require_finite_tensor(probe_logits, "preflight_answer_logits", probe_context, torch)
        if probe_logits.ndim != 2 or probe_logits.shape[0] != 1 or probe_logits.shape[1] != vocab_size:
            raise RuntimeError(
                f"Preflight answer projection returned unexpected shape {list(probe_logits.shape)}; "
                f"expected [1, {vocab_size}]"
            )
        write_json(
            output_dir / "preflight_result.json",
            {
                "created_at": timestamp(),
                "post_id": probe.example.post_id,
                "prompt_tokens": probe.prompt_len,
                "answer_logits_shape": list(probe_logits.shape),
                "answer_logits_finite": True,
            },
        )
        print(
            f"Preflight completed on longest encoded training prompt ({probe.prompt_len} tokens); "
            "no optimizer step was run.",
            flush=True,
        )
        return

    updates_per_epoch = math.ceil(len(train_encoded) / args.batch_size / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.num_epochs)
    optimizer, scheduler = make_optimizer_and_scheduler(model, args, torch, total_updates)
    scaler = None
    if args.precision == "fp16":
        scaler_type = getattr(torch.cuda, "amp", None)
        if scaler_type is None:
            raise RuntimeError("FP16 requested but torch.cuda.amp is unavailable")
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = -float("inf")
    stale_epochs = 0
    global_step = 0
    autocast_enabled = args.precision in {"bf16", "fp16"}
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            weighted_losses: list[float] = []
            unweighted_losses: list[float] = []
            grad_norms: list[float] = []
            batches = list(batch_iter(train_encoded, args.batch_size, shuffle=True, seed=args.seed + epoch))
            for batch_number, batch_items in enumerate(batches, start=1):
                batch = move_batch(collate(batch_items, tokenizer, torch), device)
                context = {
                    "phase": "training",
                    "epoch": epoch,
                    "batch_number": batch_number,
                    "global_step": global_step,
                    "post_ids": batch["post_ids"],
                }
                with autocast_context(torch, args.precision, autocast_enabled):
                    logits = answer_logits(model, batch, args, torch)
                    require_finite_tensor(logits, "answer_logits", context, torch)
                    weighted_loss, unweighted_loss = weighted_answer_loss(
                        logits,
                        batch["target_token_ids"],
                        batch["binary_labels"],
                        class_weights,
                        torch,
                    )
                require_finite_tensor(weighted_loss, "weighted_loss", context, torch)
                require_finite_tensor(unweighted_loss, "unweighted_loss", context, torch)
                weighted_losses.append(float(weighted_loss.detach().cpu()))
                unweighted_losses.append(float(unweighted_loss.detach().cpu()))
                scaled_loss = weighted_loss / args.gradient_accumulation
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    batch_number % args.gradient_accumulation == 0 or batch_number == len(batches)
                )
                if should_step:
                    step_context = dict(context)
                    step_context["global_step"] = global_step + 1
                    grad_norms.append(
                        optimizer_step(model, optimizer, scheduler, scaler, args, torch, step_context)
                    )
                    global_step += 1

            validation_metrics: dict[str, Any] | None = None
            if validation_encoded:
                validation_metrics = evaluate_model(
                    model,
                    validation_encoded,
                    tokenizer,
                    args,
                    torch,
                    device,
                    target_token_ids,
                    class_weights,
                    "validation",
                )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_train_weighted_loss": sum(weighted_losses) / len(weighted_losses),
                "mean_train_unweighted_loss": sum(unweighted_losses) / len(unweighted_losses),
                "mean_gradient_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
                "validation": (
                    {key: value for key, value in validation_metrics.items() if key != "predictions"}
                    if validation_metrics is not None
                    else None
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            write_json(output_dir / "history.json", history)
            if validation_metrics is None:
                print(
                    f"[{timestamp()}] label={args.label_key} epoch={epoch}/{args.num_epochs} "
                    f"train_weighted_loss={row['mean_train_weighted_loss']:.6f} "
                    "validation=disabled (all selected records remain in optimizer)",
                    flush=True,
                )
            else:
                print(
                    f"[{timestamp()}] label={args.label_key} epoch={epoch}/{args.num_epochs} "
                    f"train_weighted_loss={row['mean_train_weighted_loss']:.6f} "
                    f"val_balanced_accuracy={validation_metrics['balanced_accuracy']:.4f} "
                    f"val_f1={validation_metrics['f1']:.4f}",
                    flush=True,
                )
                score = float(validation_metrics["balanced_accuracy"])
                if score > best_score + 1e-8:
                    best_score = score
                    best_state = capture_trainable_state(model)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
                        print(f"[{timestamp()}] early stopping after epoch {epoch}", flush=True)
                        break
    except NumericalError as exc:
        details = dict(exc.details)
        details.update(
            {
                "created_at": timestamp(),
                "label_key": args.label_key,
                "message": str(exc),
                "model_info": model_info,
            }
        )
        write_json(output_dir / "nonfinite_abort.json", details)
        raise

    if best_state is not None:
        restore_trainable_state(model, best_state, torch)
    model.eval()
    validation_metrics = None
    if validation_encoded:
        validation_metrics = evaluate_model(
            model,
            validation_encoded,
            tokenizer,
            args,
            torch,
            device,
            target_token_ids,
            class_weights,
            "validation_best",
        )
    test_metrics = None
    if bool(args.evaluate_fixed_test):
        test_metrics = evaluate_model(
            model,
            test_encoded,
            tokenizer,
            args,
            torch,
            device,
            target_token_ids,
            class_weights,
            "test_160_qwen_labels",
        )
    all_parameters_finite(model, torch, include_frozen=False)
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if validation_metrics is not None:
        save_predictions(output_dir / "validation_predictions.json", validation_metrics)
    if test_metrics is not None:
        save_predictions(output_dir / "test_predictions.json", test_metrics)
    metrics_payload: dict[str, Any] = {
        "created_at": timestamp(),
        "label_key": args.label_key,
        "label_name": LABEL_DEFS[args.label_key].name_en,
        "fit_source": args.data_split,
        "fit_record_count": len(fit_examples),
        "optimizer_train_count": len(train_examples),
        "validation_count": len(validation_examples),
        "fixed_test_count": len(test_examples),
        "fixed_test_evaluation_enabled": bool(args.evaluate_fixed_test),
        "external_test_protocol_required": not bool(args.evaluate_fixed_test),
        "best_validation": (
            {key: value for key, value in validation_metrics.items() if key != "predictions"}
            if validation_metrics is not None
            else None
        ),
        "class_weights": class_weights,
        "history": history,
        "final_dir": str(final_dir),
    }
    if test_metrics is not None:
        metrics_payload["test_160_qwen_labels"] = {
            key: value for key, value in test_metrics.items() if key != "predictions"
        }
        metrics_payload["warning"] = (
            "The 160-record fixed test labels were included in fit and are not a held-out estimate."
        )
    else:
        metrics_payload["test_160_qwen_labels"] = None
        metrics_payload["warning"] = (
            "All 1600 records were used for fit; fixed_split=test was not evaluated. "
            "Use the caller's external test protocol."
        )
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(
        output_dir / "run_finished.json",
        {
            "created_at": timestamp(),
            "label_key": args.label_key,
            "global_steps": global_step,
            "fit_record_count": len(fit_examples),
            "optimizer_train_count": len(train_examples),
            "validation_count": len(validation_examples),
            "fixed_test_evaluation_enabled": bool(args.evaluate_fixed_test),
        },
    )
    print(f"[{timestamp()}] saved fresh adapter: {final_dir}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except NumericalError as exc:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        details = dict(exc.details)
        details.update(
            {
                "created_at": timestamp(),
                "label_key": args.label_key,
                "message": str(exc),
            }
        )
        write_json(output_dir / "nonfinite_abort.json", details)
        print(f"NUMERICAL ABORT: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
