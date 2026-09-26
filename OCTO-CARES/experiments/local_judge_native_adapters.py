#!/usr/bin/env python3
"""Model-specific Transformers adapters for the three local human-likeness judges."""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTER_VERSION = "native-three-adapters-v4-tokenized-template-overflow-guard"
OUTPUT_PREFILL = "Choice:"
SUPPORTED_ADAPTERS = ("glm4", "gemma3", "qwen35")


@dataclass
class LoadedJudge:
    adapter: str
    model: Any
    tokenizer: Any
    chat_source: Any
    backend: str
    input_device: Any
    tokenizer_size: int
    input_embedding_rows: int
    output_embedding_rows: int
    effective_vocab_limit: int
    special_token_ids: list[int]


def _native_model_class(transformers: Any, adapter: str) -> Any:
    class_names = {
        "glm4": "Glm4ForCausalLM",
        "gemma3": "Gemma3ForConditionalGeneration",
        "qwen35": "Qwen3_5ForConditionalGeneration",
    }
    class_name = class_names[adapter]
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not expose {class_name}; "
            "use an environment that natively supports this checkpoint"
        )
    return model_class


def _flatten_token_ids(*values: Any) -> list[int]:
    result: list[int] = []
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            number = int(candidate)
            if number not in result:
                result.append(number)
    return result


def _embedding_rows(embedding: Any) -> int:
    rows = int(getattr(embedding, "num_embeddings", 0) or 0)
    if not rows and getattr(embedding, "weight", None) is not None:
        rows = int(embedding.weight.shape[0])
    return rows


def _embedding_device(model: Any) -> Any:
    import torch

    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0")


def load_native_judge(
    adapter: str,
    model_path: Path,
    attn_implementation: str = "",
    load_in_8bit: bool = False,
) -> LoadedJudge:
    import torch
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    if adapter not in SUPPORTED_ADAPTERS:
        raise ValueError(f"Unsupported adapter={adapter!r}; choose from {SUPPORTED_ADAPTERS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native local judge adapters")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Each native judge must see exactly two GPUs; visible={torch.cuda.device_count()}"
        )

    common: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "balanced",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        common["attn_implementation"] = attn_implementation
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        common["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    model_class = _native_model_class(transformers, adapter)
    if adapter == "glm4":
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        chat_source = tokenizer
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"{adapter} AutoProcessor does not expose a tokenizer")
        chat_source = processor

    model = model_class.from_pretrained(model_path, **common)
    model.eval()
    device_map = getattr(model, "hf_device_map", None) or {}
    forbidden = {
        str(device)
        for device in device_map.values()
        if str(device) in ("cpu", "disk")
    }
    if forbidden:
        raise RuntimeError(
            f"{adapter} device_map offloaded modules to {sorted(forbidden)}; "
            "CPU/disk offload is disabled for this evaluation"
        )
    mapped_cuda_devices: set[int] = set()
    for device in device_map.values():
        text = str(device)
        if isinstance(device, int) or text.isdigit():
            mapped_cuda_devices.add(int(device))
        elif text.startswith("cuda:") and text[5:].isdigit():
            mapped_cuda_devices.add(int(text[5:]))
    if len(mapped_cuda_devices) != 2:
        raise RuntimeError(
            f"{adapter} did not shard across both visible GPUs: "
            f"mapped_cuda_devices={sorted(mapped_cuda_devices)}"
        )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"{adapter} tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_rows = _embedding_rows(model.get_input_embeddings())
    output_embedding = model.get_output_embeddings()
    output_rows = _embedding_rows(output_embedding) if output_embedding is not None else input_rows
    tokenizer_size = int(len(tokenizer))
    if min(input_rows, output_rows, tokenizer_size) <= 0:
        raise RuntimeError(
            f"Invalid vocabulary dimensions: tokenizer={tokenizer_size}, "
            f"input={input_rows}, output={output_rows}"
        )
    if tokenizer_size > min(input_rows, output_rows):
        raise RuntimeError(
            f"Tokenizer exceeds model vocabulary: tokenizer={tokenizer_size}, "
            f"input={input_rows}, output={output_rows}"
        )

    special_ids = _flatten_token_ids(
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        getattr(model.generation_config, "pad_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    )
    invalid_special = [
        token_id for token_id in special_ids if token_id < 0 or token_id >= min(input_rows, output_rows)
    ]
    if invalid_special:
        raise RuntimeError(
            f"Special token IDs exceed model vocabulary: {invalid_special}; "
            f"input={input_rows}, output={output_rows}"
        )
    # Some checkpoints reserve output rows beyond tokenizer length. Greedy
    # decoding can select them, produce garbage, then poison later CUDA calls.
    # Preserve every declared special token while masking only the undecodable tail.
    effective_limit = max([tokenizer_size, *(token_id + 1 for token_id in special_ids)])
    return LoadedJudge(
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        chat_source=chat_source,
        backend=model_class.__name__,
        input_device=_embedding_device(model),
        tokenizer_size=tokenizer_size,
        input_embedding_rows=input_rows,
        output_embedding_rows=output_rows,
        effective_vocab_limit=effective_limit,
        special_token_ids=special_ids,
    )


def _messages(adapter: str, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    combined = system_prompt + "\n\n" + user_prompt
    if adapter in ("gemma3", "qwen35"):
        return [{"role": "user", "content": [{"type": "text", "text": combined}]}]
    return [{"role": "user", "content": combined}]


def native_prompt_token_ids(
    judge: LoadedJudge,
    system_prompt: str,
    user_prompt: str,
) -> list[int]:
    messages = _messages(judge.adapter, system_prompt, user_prompt)
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if judge.adapter == "qwen35":
        kwargs["enable_thinking"] = False
    try:
        encoded = judge.chat_source.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        # Some local processor versions accept plain string content even though
        # the native current template uses typed multimodal content.
        plain = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        encoded = judge.chat_source.apply_chat_template(plain, **kwargs)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not encoded or not all(
        isinstance(value, int) for value in encoded
    ):
        raise RuntimeError(
            f"Native chat template returned unexpected input_ids: {type(encoded).__name__}"
        )
    prefill_ids = judge.tokenizer.encode(OUTPUT_PREFILL, add_special_tokens=False)
    if not prefill_ids:
        raise RuntimeError("Tokenizer produced no IDs for the Choice: assistant prefill")
    return [*encoded, *[int(value) for value in prefill_ids]]


def generate_native_batch(
    judge: LoadedJudge,
    prompts: list[str],
    system_prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class UndecodableTailGuard(LogitsProcessor):
        def __init__(self, limit: int, adapter: str, tokenizer: Any) -> None:
            self.limit = limit
            self.adapter = adapter
            self.tokenizer = tokenizer
            self.steps = 0
            self.started = time.monotonic()
            self.top_history: list[int] = []
            self.sanitization_events: list[dict[str, Any]] = []

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            self.steps += 1
            nan_mask = torch.isnan(scores)
            positive_inf_mask = torch.isposinf(scores)
            invalid_mask = nan_mask | positive_inf_mask
            invalid_count = int(invalid_mask.sum().item())
            if invalid_count:
                valid_region_invalid = int(invalid_mask[..., : self.limit].sum().item())
                tail_invalid = int(invalid_mask[..., self.limit :].sum().item())
                invalid_ratio = invalid_count / scores.numel()
                invalid_ids = torch.nonzero(invalid_mask[0], as_tuple=False).flatten()[:16]
                samples = [
                    {
                        "id": int(token_id),
                        "text": self.tokenizer.decode(
                            [int(token_id)], skip_special_tokens=False
                        ),
                    }
                    for token_id in invalid_ids.tolist()
                ]
                event = {
                    "step": self.steps,
                    "nan": int(nan_mask.sum().item()),
                    "positive_inf": int(positive_inf_mask.sum().item()),
                    "inside_decodable_vocab": valid_region_invalid,
                    "outside_decodable_vocab": tail_invalid,
                    "ratio": invalid_ratio,
                    "sample_tokens": samples,
                }
                self.sanitization_events.append(event)
                print(
                    f"native_generation_sanitize adapter={self.adapter} event={event}",
                    flush=True,
                )
                if invalid_ratio > 0.005:
                    raise RuntimeError(
                        f"Too many invalid generation logits at step={self.steps}: "
                        f"count={invalid_count} ratio={invalid_ratio:.6f}"
                    )
                scores.masked_fill_(invalid_mask, -float("inf"))
            if scores.shape[-1] > self.limit:
                scores[..., self.limit :] = -float("inf")
            valid_scores = scores[..., : self.limit]
            if not bool(torch.isfinite(valid_scores).any(dim=-1).all().item()):
                raise RuntimeError(
                    f"No finite logits remain inside decodable vocabulary at step={self.steps}"
                )
            finite_values = valid_scores[torch.isfinite(valid_scores)]
            finite_abs_max = float(finite_values.abs().max().item())
            if finite_abs_max > 1.0e4:
                raise RuntimeError(
                    f"Generation logits numerically exploded at step={self.steps}: "
                    f"finite_abs_max={finite_abs_max:.6g}"
                )
            top_id = int(torch.argmax(valid_scores[0]).item())
            self.top_history.append(top_id)
            if len(self.top_history) >= 4 and len(set(self.top_history[-4:])) == 1:
                token_text = self.tokenizer.decode([top_id], skip_special_tokens=False)
                raise RuntimeError(
                    f"Degenerate greedy decoding: token_id={top_id} "
                    f"token={token_text!r} repeated four steps"
                )
            if self.steps == 1 or self.steps % 8 == 0:
                elapsed = time.monotonic() - self.started
                top_values, top_ids = torch.topk(
                    valid_scores[0].float(), k=min(5, self.limit)
                )
                top_tokens = [
                    {
                        "id": int(token_id),
                        "text": self.tokenizer.decode(
                            [int(token_id)], skip_special_tokens=False
                        ),
                        "score": round(float(score), 6),
                    }
                    for token_id, score in zip(top_ids.tolist(), top_values.tolist())
                ]
                print(
                    f"native_generation_progress adapter={self.adapter} "
                    f"new_tokens={self.steps} seconds={elapsed:.2f} top={top_tokens}",
                    flush=True,
                )
            return scores

    token_rows = [
        native_prompt_token_ids(judge, system_prompt, prompt) for prompt in prompts
    ]
    longest = max(len(row) for row in token_rows)
    if longest > max_input_tokens:
        raise RuntimeError(
            f"Native chat template input has {longest} tokens, exceeding "
            f"max_input_tokens={max_input_tokens}; refusing to truncate evaluation options"
        )
    input_ids = torch.full(
        (len(token_rows), longest),
        fill_value=int(judge.tokenizer.pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(token_rows):
        width = len(row)
        input_ids[index, -width:] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, -width:] = 1
    input_min = int(input_ids.min().item())
    input_max = int(input_ids.max().item())
    if input_min < 0 or input_max >= judge.input_embedding_rows:
        raise RuntimeError(
            f"Input token range {input_min}..{input_max} exceeds "
            f"embedding rows={judge.input_embedding_rows}"
        )
    encoded = {
        "input_ids": input_ids.to(judge.input_device),
        "attention_mask": attention_mask.to(judge.input_device),
    }
    input_width = int(encoded["input_ids"].shape[1])
    print(
        f"native_generation_input adapter={judge.adapter} batch={len(prompts)} "
        f"input_tokens={input_width} max_new_tokens={max_new_tokens}",
        flush=True,
    )
    logits_processors = LogitsProcessorList(
        [
            UndecodableTailGuard(
                judge.effective_vocab_limit,
                judge.adapter,
                judge.tokenizer,
            )
        ]
    )
    with torch.inference_mode():
        generated = judge.model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=judge.tokenizer.pad_token_id,
            eos_token_id=getattr(judge.model.generation_config, "eos_token_id", None),
            logits_processor=logits_processors,
        )
    if generated.ndim != 2 or generated.shape[0] != len(prompts):
        raise RuntimeError(f"Unexpected generated tensor shape: {tuple(generated.shape)}")
    includes_prompt = (
        generated.shape[1] >= input_width
        and torch.equal(generated[:, :input_width], encoded["input_ids"])
    )
    new_tokens = generated[:, input_width:] if includes_prompt else generated
    decoded = judge.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    decoded_special = judge.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
    rows: list[dict[str, Any]] = []
    for index in range(new_tokens.shape[0]):
        rows.append(
            {
                "text": OUTPUT_PREFILL + decoded[index],
                "text_with_special_tokens": OUTPUT_PREFILL + decoded_special[index],
                "generated_token_ids": [int(value) for value in new_tokens[index].tolist()],
                "input_width": input_width,
                "returned_sequence_width": int(generated.shape[1]),
                "returned_sequence_includes_prompt": bool(includes_prompt),
                "logit_sanitization_events": list(
                    logits_processors[0].sanitization_events
                ),
            }
        )
    return rows
