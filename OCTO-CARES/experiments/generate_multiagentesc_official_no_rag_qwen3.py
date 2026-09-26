#!/usr/bin/env python3
"""Run the official MultiAgentESC protocol without retrieval, using Qwen3 only.

This is a project adaptation of the authors' public implementation at
https://github.com/MindIntLab-HFUT/MultiAgentESC, pinned to commit
631b7f1961fc7502e547fd9258e847230dbcb973.

Preserved protocol:

* the official early-dialogue / LLM complexity gate;
* sequential emotion -> event/cause -> intention analysis;
* three homogeneous strategy agents in a shared round-robin discussion;
* one candidate response per distinct valid ESConv strategy;
* candidate-bound response debate;
* a separate reflection discussion in which agents may change position;
* majority voting, a dedicated tie judge, and a mandatory final refiner.

Intentional adaptations:

* SBERT retrieval, top-10 ESConv cases, and all retrieved examples are removed;
* the strategy agents select directly from the eight published ESConv strategy
  definitions;
* every role is the same local Qwen3 checkpoint (Qwen3-8B by default);
* the official deterministic round-robin GroupChat semantics are implemented
  directly so the existing local Transformers/four-worker setup needs no
  OpenAI-compatible server or AutoGen runtime.

The input JSONL is only a dialogue container.  Summary, neighbors, posts,
comments, reference responses, labels, and ESCoT reasoning are neither read
nor placed in a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


CONDITION = "multiagent_esc_official_no_rag"
PROTOCOL_VERSION = "multiagentesc-official-protocol-no-rag-qwen3-v1"
UPSTREAM_REPOSITORY = "https://github.com/MindIntLab-HFUT/MultiAgentESC"
UPSTREAM_COMMIT = "631b7f1961fc7502e547fd9258e847230dbcb973"

STRATEGIES: dict[str, str] = {
    "Question": (
        "Asking for information related to the problem to help the user articulate the issues "
        "that they face. Open-ended questions are best, and closed questions can be used to "
        "get specific information."
    ),
    "Restatement or Paraphrasing": (
        "A simple, more concise rephrasing of the user's statements that could help them see "
        "their situation more clearly."
    ),
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": (
        "Divulge similar experiences that you have had or emotions that you share with the "
        "user to express your empathy."
    ),
    "Affirmation and Reassurance": (
        "Affirm the user's strengths, motivation, and capabilities and provide reassurance "
        "and encouragement."
    ),
    "Providing Suggestions": (
        "Provide suggestions about how to change, but be careful to not overstep and tell "
        "them what to do."
    ),
    "Information": (
        "Provide useful information to the user, for example with data, facts, opinions, "
        "resources, or by answering questions."
    ),
    "Others": (
        "Exchange pleasantries and use other support strategies that do not fall into the "
        "above categories."
    ),
}

STRATEGY_DEFINITIONS = "\n".join(
    f"{name}: {description}" for name, description in STRATEGIES.items()
)
COUNSELOR_SYSTEM = "You are a psychological counseling expert."
AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_INPUT_CONTEXT: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="responses")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--model-path", default="models/Qwen3-8B")
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
        "--complexity-gate",
        choices=("official", "always_multiagent"),
        default="official",
        help="official preserves the early-turn and LLM behavior-control gate",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input JSONL: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
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
            query_id = str(row.get("query_id") or "").strip()
            dialogue = str(row.get("dialogue_prefix") or "").strip()
            latest = str(row.get("last_seeker") or "").strip()
            if not query_id or not dialogue or not latest:
                raise ValueError(
                    f"Input row {line_number} requires query_id, dialogue_prefix, and last_seeker"
                )
            if query_id in seen:
                raise ValueError(f"Duplicate query_id at row {line_number}: {query_id}")
            seen.add(query_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cache row at {path}:{line_number}: {exc}") from exc
            query_id = str(row.get("query_id") or "").strip()
            if query_id and row.get("response") and not row.get("error"):
                completed[query_id] = row
    return completed


def worker_path(args: argparse.Namespace, worker: int | None = None) -> Path:
    index = args.worker_index if worker is None else worker
    return args.output_dir / f"{args.output_prefix}.{CONDITION}.worker{index}.jsonl"


def final_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"{args.output_prefix}.{CONDITION}.jsonl"


def normalize_dialogue(dialogue: str) -> str:
    """Use the official User/Assistant labels without altering utterance text."""

    lines: list[str] = []
    for raw in str(dialogue or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("seeker:"):
            line = "User:" + line.split(":", 1)[1]
        elif line.lower().startswith("supporter:"):
            line = "Assistant:" + line.split(":", 1)[1]
        lines.append(line)
    return "\n".join(lines)


def dialogue_turn_count(dialogue: str) -> int:
    matches = re.findall(r"(?im)^\s*(?:seeker|supporter|user|assistant)\s*:", dialogue)
    if not matches:
        raise ValueError("dialogue_prefix contains no recognizable speaker turns")
    return len(matches)


def strip_thinking(text: str) -> str:
    value = str(text or "").strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    return value


def canonical_strategy(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", str(value or "").lower()).strip()
    aliases = {
        re.sub(r"[^a-z]+", " ", strategy.lower()).strip(): strategy
        for strategy in STRATEGIES
    }
    aliases.update(
        {
            "restatement": "Restatement or Paraphrasing",
            "paraphrasing": "Restatement or Paraphrasing",
            "reflection": "Reflection of feelings",
            "affirmation": "Affirmation and Reassurance",
            "reassurance": "Affirmation and Reassurance",
            "suggestion": "Providing Suggestions",
            "suggestions": "Providing Suggestions",
        }
    )
    return aliases.get(normalized)


def parse_strategy(text: str) -> str | None:
    cleaned = strip_thinking(text)
    match = re.search(r"(?im)^\s*Strategy\s*[:：=]\s*\[?([^\]\n]+)\]?", cleaned)
    return canonical_strategy(match.group(1)) if match else None


def parse_response(text: str, fallback_strategy: str | None = None) -> tuple[str | None, str]:
    cleaned = strip_thinking(text)
    match = re.search(
        r"(?is)^.*?Response\s*[:：=]\s*\[([^\]]+)\]\s*(.*?)"
        r"(?:\n\s*Reasoning\s*[:：=]|\Z)",
        cleaned,
    )
    if match:
        strategy = canonical_strategy(match.group(1))
        if strategy:
            response = match.group(2).strip()
            if response.startswith("[") and response.endswith("]"):
                response = response[1:-1].strip()
            return strategy, response
    plain_match = re.search(
        r"(?is)^.*?Response\s*[:：=]\s*(.*?)(?:\n\s*Reasoning\s*[:：=]|\Z)",
        cleaned,
    )
    if plain_match:
        response = plain_match.group(1).strip()
        if response.startswith("[") and response.endswith("]"):
            response = response[1:-1].strip()
        return fallback_strategy, response
    return fallback_strategy, cleaned


def parse_labeled_first_line(text: str, label: str, default: str) -> str:
    cleaned = strip_thinking(text)
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*[:：=]\s*(.+)$", cleaned)
    return match.group(1).strip() if match else default


def behavior_control_prompt(context: str) -> str:
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with an incomplete conversation between an Assistant and a User.
Please analyze whether this conversation reflects the user's current emotional state, the reason the user is seeking emotional support, and how the user plans to cope with the event.
If all three points are reflected, please reply "YES," otherwise reply "NO."

### Conversation
{context}

Your answer must include two parts:
1. "YES" or "NO"
2. If "YES", briefly explain how the conversation reflects these elements; if "NO", explain which elements are missing.

Your answer must follow this format:
1. [YES or NO]
2. [explanation]"""


def zero_shot_prompt(context: str) -> str:
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

### Dialogue context
{context}

Your answer must be fewer than 30 words and must follow this format:
Response: [response]"""


def emotion_prompt(context: str) -> str:
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Please infer the emotional state expressed in the user's last utterance.

### Dialogue context
{context}

Your answer must include the following elements:
Emotion: the emotion user expressed in their last utterance.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Emotion: [emotion]
Reasoning: [reasoning]"""


def cause_prompt(context: str, emotion_analysis: str) -> str:
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Another agent analyzes the conversation and infers the emotional state expressed by the user in their last utterance.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

Please infer the specific event that led to the user's emotional state based on the dialogue context.
Your answer must follow this format:
Event: [event]
Reasoning: [reasoning]"""


def intention_prompt(context: str, emotion_analysis: str, cause_analysis: str) -> str:
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, inferring the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

### Event
{cause_analysis}

Please reasonably infer the user's intention based on the dialogue context, with the goal of addressing the event that led to their emotional state.
Your answer must follow this format:
Intention: [intention]
Reasoning: [reasoning]"""


def strategy_task(
    context: str, emotion_analysis: str, cause_analysis: str, intention_analysis: str
) -> str:
    return f"""### Strategy Deliberation
You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the user's emotional state, the event that led to it, and the user's intention.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

### Event
{cause_analysis}

### Intention
{intention_analysis}

Select exactly one strategy from the following eight published ESConv strategies:
{STRATEGY_DEFINITIONS}

Select a strategy for the Assistant to generate an appropriate response and explain why. Read the preceding group messages when present and choose a different valid strategy whenever another strategy is also reasonable, so the group retains diverse candidates.

Your answer must follow this format exactly:
Strategy: [strategy]
Reasoning: [reasoning]"""


def candidate_response_prompt(
    context: str,
    emotion_analysis: str,
    cause_analysis: str,
    intention_analysis: str,
    strategy: str,
) -> str:
    return f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the user's emotional state, the event that led to it, and the user's intention.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

### Event
{cause_analysis}

### Intention
{intention_analysis}

Please generate a response from the Assistant's perspective using the {strategy} strategy.
Strategy definition: {STRATEGIES[strategy]}

Your answer must be fewer than 30 words and must follow this format:
Response: [{strategy}] [response]"""


def format_candidates(candidates: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{item['strategy']}] {item['response']}" for item in candidates
    )


def debate_task(
    context: str,
    emotion_analysis: str,
    cause_analysis: str,
    intention_analysis: str,
    candidates: list[dict[str, str]],
) -> str:
    return f"""### Response Debate
You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the user's emotional state, the event that led to it, and the user's intention.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

### Event
{cause_analysis}

### Intention
{intention_analysis}

Based on the information and dialogue context, select the most appropriate response from the following options and explain why.

### Responses
{format_candidates(candidates)}

Your answer must follow this format exactly:
Response: [strategy] [response]
Reasoning: [reasoning]"""


def reflection_task(
    context: str,
    emotion_analysis: str,
    cause_analysis: str,
    intention_analysis: str,
    candidates: list[dict[str, str]],
    debate_transcript: list[dict[str, Any]],
) -> str:
    discussion = "\n\n".join(
        f"{turn['agent']}: {turn['output']['output']}" for turn in debate_transcript
    )
    return f"""### Response Reflection
You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the user's emotional state, the event that led to it, and the user's intention.

### Dialogue context
{context}

### Emotional state
{emotion_analysis}

### Event
{cause_analysis}

### Intention
{intention_analysis}

### Candidate responses
{format_candidates(candidates)}

### Previous group discussion
{discussion}

Carefully analyze the different viewpoints above, reflect on your own view, and arrive at a convincing result. You may change your position if another viewpoint is more reasonable.

Your answer must follow this format exactly:
Response: [strategy] [response]
Reasoning: [reasoning]"""


def judge_prompt(context: str, tied_candidates: list[dict[str, str]]) -> str:
    return f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following tied responses were generated using different strategies. Select the most appropriate response and explain why.

### Responses
{format_candidates(tied_candidates)}

Your answer must follow this format exactly:
Response: [strategy] [response]
Reasoning: [reasoning]"""


def refiner_prompt(context: str, strategy: str, response: str) -> str:
    return f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following response was generated using the {strategy} strategy. Analyze whether it is consistent with the ongoing conversation, aligns with the strategy, and effectively helps alleviate the user's emotional stress.

### Response
[{strategy}] {response}

If the response meets the requirements, return it as is; otherwise provide a refined version. The response must be fewer than 30 words.

Your answer must follow this format exactly:
Response: [strategy] [original/refined response]
Reasoning: [reasoning]"""


def prompt_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        attn_implementation="sdpa",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("One visible CUDA GPU is required per worker")
    model.to(torch.device("cuda:0"))
    model.eval()
    return torch, tokenizer, model


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage_args(args: argparse.Namespace, max_new_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def generate_text(
    torch: Any,
    tokenizer: Any,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    args: SimpleNamespace,
    seed: int,
) -> dict[str, Any]:
    set_seed(torch, seed)
    messages = prompt_messages(system_prompt, user_prompt)
    rendered = render_prompt(tokenizer, messages)
    untruncated = tokenizer(rendered, truncation=False, add_special_tokens=True)
    before = len(untruncated["input_ids"])
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_count = int(inputs["input_ids"].shape[-1])
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation: dict[str, Any] = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if args.temperature > 0:
        generation["temperature"] = args.temperature
        generation["top_p"] = args.top_p
    if tokenizer.eos_token_id is not None:
        generation["eos_token_id"] = tokenizer.eos_token_id
        generation["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**generation)
    generated_ids = generated.sequences[0, input_count:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not output:
        raise RuntimeError("Qwen3 generated an empty stage output")
    output_tokens = int(generated_ids.numel())
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages_sha256": sha256_json(messages),
        "output": output,
        "input_tokens": input_count,
        "input_tokens_before_truncation": int(before),
        "input_truncated": bool(before > input_count),
        "output_tokens": output_tokens,
        "finish_reason": "eos" if output_tokens < args.max_new_tokens else "length",
        "seed": seed,
    }


def add_transcript(task: str, transcript: list[dict[str, Any]], agent: str) -> str:
    if not transcript:
        discussion = "(No participant has spoken yet.)"
    else:
        discussion = "\n\n".join(
            f"{turn['agent']}: {turn['output']['output']}" for turn in transcript
        )
    return (
        task
        + "\n\n### Shared group discussion so far\n"
        + discussion
        + f"\n\nYou are {agent}. Give your own current answer now."
    )


def run_round_robin(
    generate: Callable[[str, str, int, int], dict[str, Any]],
    task: str,
    systems: list[str],
    seed_start: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Mirror official GroupChat(max_round=len(agents)+1, round_robin)."""

    transcript: list[dict[str, Any]] = []
    for index, system_prompt in enumerate(systems):
        agent = f"agent_{index}"
        user_prompt = add_transcript(task, transcript, agent)
        output = generate(system_prompt, user_prompt, seed_start + index, max_new_tokens)
        transcript.append({"agent": agent, "output": output})
    return transcript


def merge_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    expected_ids = [str(row["query_id"]) for row in rows]
    expected = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for worker in range(args.worker_count):
        path = worker_path(args, worker)
        if not path.is_file():
            raise FileNotFoundError(f"Missing worker output: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                query_id = str(payload.get("query_id") or "").strip()
                if not query_id or payload.get("condition") != CONDITION:
                    raise ValueError(f"Malformed worker row at {path}:{line_number}")
                by_id[query_id] = payload
    missing = expected - set(by_id)
    extra = set(by_id) - expected
    if missing or extra:
        raise ValueError(f"Coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    failures = [
        query_id
        for query_id in expected_ids
        if by_id[query_id].get("error") or not by_id[query_id].get("response")
    ]
    if failures:
        first = failures[0]
        raise RuntimeError(f"Incomplete generation for {first}: {by_id[first].get('error')}")
    ordered = [by_id[query_id] for query_id in expected_ids]
    atomic_write_jsonl(final_path(args), ordered)
    print(f"[{timestamp()}] merged {len(ordered)} rows -> {final_path(args)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count)")
    if "qwen3" not in Path(args.model_path).name.lower():
        raise ValueError(f"All roles must use a Qwen3 checkpoint, got: {args.model_path}")
    if args.limit < 0 or args.max_input_tokens <= 0:
        raise ValueError("limit and token budgets must be non-negative/positive")
    if args.strategy_agent_count != 3:
        raise ValueError("The official protocol uses exactly three strategy agents")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid temperature/top-p")
    budgets = (
        args.decision_max_new_tokens,
        args.analysis_max_new_tokens,
        args.discussion_max_new_tokens,
        args.response_max_new_tokens,
        args.judge_max_new_tokens,
        args.refine_max_new_tokens,
    )
    if any(value <= 0 for value in budgets):
        raise ValueError("all stage token budgets must be positive")

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.merge:
        merge_outputs(args, rows)
        return

    shard = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = worker_path(args)
    if args.force:
        path.unlink(missing_ok=True)
    path.touch(exist_ok=True)
    completed = load_completed(path)
    print(
        f"[{timestamp()}] official-no-rag worker={args.worker_index}/{args.worker_count} "
        f"queries={len(shard)} model={args.model_path} retrieval=false",
        flush=True,
    )
    if not shard:
        return

    torch, tokenizer, model = load_model(args)

    def generate(
        system_prompt: str, user_prompt: str, seed: int, max_new_tokens: int
    ) -> dict[str, Any]:
        return generate_text(
            torch,
            tokenizer,
            model,
            system_prompt,
            user_prompt,
            stage_args(args, max_new_tokens),
            seed,
        )

    for local_index, row in enumerate(shard):
        query_id = str(row["query_id"])
        dialogue_prefix = str(row["dialogue_prefix"]).strip()
        context = normalize_dialogue(dialogue_prefix)
        turn_count = dialogue_turn_count(dialogue_prefix)
        official_count_after_target = turn_count + 1
        base_seed = args.seed + (
            int(str(row.get("source_id")))
            if str(row.get("source_id") or "").isdigit()
            else local_index * 1009
        )
        config_for_hash = {
            "protocol": PROTOCOL_VERSION,
            "model_path": args.model_path,
            "context": context,
            "complexity_gate": args.complexity_gate,
            "strategy_agent_count": args.strategy_agent_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "bf16": bool(args.bf16),
            "local_files_only": bool(args.local_files_only),
            "trust_remote_code": bool(args.trust_remote_code),
            "max_input_tokens": args.max_input_tokens,
            "budgets": budgets,
        }
        input_fingerprint = sha256_json(config_for_hash)
        previous = completed.get(query_id)
        if (
            previous
            and not args.force
            and previous.get("generation_protocol_version") == PROTOCOL_VERSION
            and previous.get("input_fingerprint") == input_fingerprint
            and previous.get("model_path") == args.model_path
            and previous.get("seed") == base_seed
        ):
            print(f"[{timestamp()}] reuse {query_id}", flush=True)
            continue

        print(
            f"[{timestamp()}] worker={args.worker_index} query={local_index + 1}/{len(shard)} {query_id}",
            flush=True,
        )
        stage_offset = 0

        def next_seed() -> int:
            nonlocal stage_offset
            stage_offset += 1
            return base_seed + stage_offset

        trace: dict[str, Any] = {}
        try:
            pipeline_path = "multiagent"
            gate_reason = "always_multiagent"
            is_complex = True
            if args.complexity_gate == "official":
                if official_count_after_target <= 5:
                    is_complex = False
                    gate_reason = "official_early_turn_gate"
                else:
                    gate = generate(
                        COUNSELOR_SYSTEM,
                        behavior_control_prompt(context),
                        next_seed(),
                        args.decision_max_new_tokens,
                    )
                    trace["complexity_gate"] = gate
                    gate_clean = strip_thinking(gate["output"])
                    # Prefer the first explicit YES/NO token if both occur in explanation.
                    token = re.search(r"(?i)\b(YES|NO)\b", gate_clean)
                    is_complex = bool(token and token.group(1).upper() == "YES")
                    gate_reason = "llm_yes" if is_complex else "llm_no"

            selected_strategy = "None"
            original_response = ""
            final_response = ""

            if not is_complex:
                pipeline_path = "zero_shot"
                direct = generate(
                    COUNSELOR_SYSTEM,
                    zero_shot_prompt(context),
                    next_seed(),
                    args.response_max_new_tokens,
                )
                _, final_response = parse_response(direct["output"])
                if not final_response:
                    raise ValueError("Could not parse zero-shot response")
                trace["zero_shot"] = direct
            else:
                emotion = generate(
                    COUNSELOR_SYSTEM,
                    emotion_prompt(context),
                    next_seed(),
                    args.analysis_max_new_tokens,
                )
                cause = generate(
                    COUNSELOR_SYSTEM,
                    cause_prompt(context, emotion["output"]),
                    next_seed(),
                    args.analysis_max_new_tokens,
                )
                intention = generate(
                    COUNSELOR_SYSTEM,
                    intention_prompt(context, emotion["output"], cause["output"]),
                    next_seed(),
                    args.analysis_max_new_tokens,
                )
                trace["dialogue_analysis"] = {
                    "emotion": emotion,
                    "cause": cause,
                    "intention": intention,
                    "parsed": {
                        "emotion": parse_labeled_first_line(
                            emotion["output"], "Emotion", "Negative"
                        ),
                        "event": parse_labeled_first_line(
                            cause["output"], "Event", "Not mentioned"
                        ),
                        "intention": parse_labeled_first_line(
                            intention["output"], "Intention", "Not mentioned"
                        ),
                    },
                }

                strategy_systems = [COUNSELOR_SYSTEM] * args.strategy_agent_count
                strategy_seed = next_seed()
                strategy_discussion = run_round_robin(
                    generate,
                    strategy_task(
                        context, emotion["output"], cause["output"], intention["output"]
                    ),
                    strategy_systems,
                    strategy_seed,
                    args.discussion_max_new_tokens,
                )
                # Reserve every seed consumed inside the group after its first one.
                stage_offset += args.strategy_agent_count - 1
                chosen_strategies: list[str] = []
                for turn in strategy_discussion:
                    strategy = parse_strategy(turn["output"]["output"])
                    turn["parsed_strategy"] = strategy
                    if strategy and strategy not in chosen_strategies:
                        chosen_strategies.append(strategy)
                trace["strategy_deliberation"] = {
                    "round_robin": True,
                    "max_round_equivalent": args.strategy_agent_count + 1,
                    "discussion": strategy_discussion,
                    "selected_strategies": chosen_strategies,
                }

                if not chosen_strategies:
                    pipeline_path = "strategy_parse_fallback_zero_shot"
                    direct = generate(
                        COUNSELOR_SYSTEM,
                        zero_shot_prompt(context),
                        next_seed(),
                        args.response_max_new_tokens,
                    )
                    _, final_response = parse_response(direct["output"])
                    trace["zero_shot_fallback"] = direct
                    if not final_response:
                        raise ValueError("Strategy parse failed and fallback response is empty")
                else:
                    candidates: list[dict[str, Any]] = []
                    for strategy in chosen_strategies:
                        candidate = generate(
                            COUNSELOR_SYSTEM,
                            candidate_response_prompt(
                                context,
                                emotion["output"],
                                cause["output"],
                                intention["output"],
                                strategy,
                            ),
                            next_seed(),
                            args.response_max_new_tokens,
                        )
                        parsed_strategy, response = parse_response(
                            candidate["output"], fallback_strategy=strategy
                        )
                        if not response:
                            raise ValueError(f"Empty candidate for strategy {strategy}")
                        candidates.append(
                            {
                                "strategy": parsed_strategy or strategy,
                                "response": response,
                                "generation": candidate,
                            }
                        )
                    trace["candidate_generation"] = candidates

                    if len(candidates) == 1:
                        selected_strategy = str(candidates[0]["strategy"])
                        original_response = str(candidates[0]["response"])
                        trace["selection"] = {
                            "method": "single_strategy",
                            "strategy": selected_strategy,
                            "response": original_response,
                        }
                    else:
                        candidate_view = [
                            {"strategy": str(item["strategy"]), "response": str(item["response"])}
                            for item in candidates
                        ]
                        debate_systems = [
                            (
                                "You are a psychologist who listens to others and reflects on "
                                "your own view. You are participating in a discussion about the "
                                "most appropriate response and initially support this response: "
                                f"[{item['strategy']}] {item['response']}. Carefully consider "
                                "other perspectives and ultimately reach a reliable answer."
                            )
                            for item in candidate_view
                        ]
                        debate_seed = next_seed()
                        debate_history = run_round_robin(
                            generate,
                            debate_task(
                                context,
                                emotion["output"],
                                cause["output"],
                                intention["output"],
                                candidate_view,
                            ),
                            debate_systems,
                            debate_seed,
                            args.discussion_max_new_tokens,
                        )
                        stage_offset += len(debate_systems) - 1

                        reflection_seed = next_seed()
                        reflection_history = run_round_robin(
                            generate,
                            reflection_task(
                                context,
                                emotion["output"],
                                cause["output"],
                                intention["output"],
                                candidate_view,
                                debate_history,
                            ),
                            debate_systems,
                            reflection_seed,
                            args.discussion_max_new_tokens,
                        )
                        stage_offset += len(debate_systems) - 1

                        valid_strategy_set = {item["strategy"] for item in candidate_view}
                        parsed_votes: list[dict[str, str]] = []
                        for turn in reflection_history:
                            strategy, response = parse_response(turn["output"]["output"])
                            if strategy in valid_strategy_set and response:
                                parsed_votes.append(
                                    {
                                        "agent": turn["agent"],
                                        "strategy": str(strategy),
                                        "response": response,
                                    }
                                )
                        counts = Counter(vote["strategy"] for vote in parsed_votes)
                        tied: list[dict[str, str]] = []
                        if counts:
                            maximum = max(counts.values())
                            tied_strategies = [
                                strategy for strategy, count in counts.items() if count == maximum
                            ]
                            for strategy in tied_strategies:
                                vote = next(
                                    item for item in parsed_votes if item["strategy"] == strategy
                                )
                                tied.append(
                                    {"strategy": strategy, "response": vote["response"]}
                                )
                        else:
                            tied = candidate_view

                        judge_meta: dict[str, Any] | None = None
                        if len(tied) == 1:
                            selected_strategy = tied[0]["strategy"]
                            original_response = tied[0]["response"]
                            selection_method = "reflection_majority_vote"
                        else:
                            judge_meta = generate(
                                COUNSELOR_SYSTEM,
                                judge_prompt(context, tied),
                                next_seed(),
                                args.judge_max_new_tokens,
                            )
                            judge_strategy, judge_response = parse_response(judge_meta["output"])
                            if judge_strategy not in {item["strategy"] for item in tied}:
                                judge_strategy = tied[0]["strategy"]
                            if not judge_response:
                                judge_response = next(
                                    item["response"]
                                    for item in tied
                                    if item["strategy"] == judge_strategy
                                )
                            selected_strategy = str(judge_strategy)
                            original_response = judge_response
                            selection_method = (
                                "dedicated_judge_no_valid_vote"
                                if not counts
                                else "dedicated_judge_tie"
                            )
                        trace["response_debate"] = debate_history
                        trace["response_reflection"] = reflection_history
                        trace["reflection_votes"] = {
                            "parsed_votes": parsed_votes,
                            "counts": dict(counts),
                            "tied_candidates": tied,
                        }
                        trace["selection"] = {
                            "method": selection_method,
                            "strategy": selected_strategy,
                            "response": original_response,
                            "judge": judge_meta,
                        }

                    # The official implementation always runs its final self-reflection
                    # after at least one valid strategy was generated.
                    refiner = generate(
                        COUNSELOR_SYSTEM,
                        refiner_prompt(context, selected_strategy, original_response),
                        next_seed(),
                        args.refine_max_new_tokens,
                    )
                    refined_strategy, final_response = parse_response(
                        refiner["output"], fallback_strategy=selected_strategy
                    )
                    if refined_strategy:
                        selected_strategy = refined_strategy
                    if not final_response:
                        final_response = original_response
                    trace["final_refiner"] = refiner

            if not final_response:
                raise ValueError("Pipeline produced no final response")

            payload: dict[str, Any] = {
                "query_id": query_id,
                "source_id": row.get("source_id"),
                "condition": CONDITION,
                "generation_protocol_version": PROTOCOL_VERSION,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "protocol_fidelity": "official_control_flow_without_retrieval",
                "intentional_adaptations": [
                    "removed_sbert_and_top10_experience_retrieval",
                    "removed_retrieved_strategy_response_examples",
                    "all_roles_use_same_local_qwen3_checkpoint",
                    "deterministic_round_robin_group_manager_without_autogen_runtime",
                    AUDIT_INPUT_CONTEXT or "input_is_escot_dialogue_prefix",
                ],
                "model_path": args.model_path,
                "all_roles_model_path": args.model_path,
                "homogeneous_agents": True,
                "dialogue_prefix": row.get("dialogue_prefix"),
                "last_seeker": row.get("last_seeker"),
                "dialogue_turn_count": turn_count,
                "official_count_after_target": official_count_after_target,
                "complexity_gate_mode": args.complexity_gate,
                "complexity_gate_result": is_complex,
                "complexity_gate_reason": gate_reason,
                "pipeline_path": pipeline_path,
                "retrieval_used_in_prompt": False,
                "retrieval_used_anywhere": False,
                "top_k": None,
                "top_k_applicability": "not_applicable",
                "summary_used_in_generation_prompt": (
                    AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT
                    if AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT is not None
                    else False
                ),
                "dialogue_used_in_generation_prompt": (
                    AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT
                    if AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT is not None
                    else True
                ),
                "last_seeker_used_in_generation_prompt": (
                    AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT
                    if AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT is not None
                    else True
                ),
                "neighbors_read": False,
                "reference_response_used": False,
                "selected_strategy": selected_strategy,
                "original_selected_response": original_response or None,
                "response": final_response,
                "trace": trace,
                "input_fingerprint": input_fingerprint,
                "seed": base_seed,
                "stage_call_count": stage_offset,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "bf16": bool(args.bf16),
                "max_input_tokens": args.max_input_tokens,
                "worker_index": args.worker_index,
                "created_at": timestamp(),
            }
            completed[query_id] = payload
        except Exception as exc:
            payload = {
                "query_id": query_id,
                "source_id": row.get("source_id"),
                "condition": CONDITION,
                "generation_protocol_version": PROTOCOL_VERSION,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "model_path": args.model_path,
                "retrieval_used_anywhere": False,
                "top_k": None,
                "input_fingerprint": input_fingerprint,
                "seed": base_seed,
                "trace": trace,
                "error": f"{type(exc).__name__}: {exc}",
                "created_at": timestamp(),
            }
            print(f"[{timestamp()}] ERROR {query_id}: {payload['error']}", flush=True)
        append_jsonl(path, payload)
    print(f"[{timestamp()}] official-no-rag worker={args.worker_index} complete", flush=True)


if __name__ == "__main__":
    main()
