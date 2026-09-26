#!/usr/bin/env python3
"""Judge five Reddit methods against pure_qwen3 through a local vLLM API."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import judge_humanlikeness_qwen37 as four_way
import judge_pairwise_humanlikeness as pairwise
from judge_local_humanlikeness_vs_pure import (
    BASELINE,
    CHALLENGERS,
    CONDITIONS,
    SYSTEM_PROMPT,
    prompt_args,
    prompt_hash,
    read_items,
)


PROMPT_VERSION = "vllm-ab-liveliness-v2-single-user-no-length-heuristic"
MESSAGE_LAYOUT = "single_user_combined_system_and_pair_prompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", action="append", required=True, metavar="CONDITION=JSONL")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-dialogue-chars", type=int, default=60000)
    parser.add_argument("--max-response-chars", type=int, default=12000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        condition, separator, raw_path = value.partition("=")
        if not separator or not condition.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --generation value: {value!r}")
        condition = condition.strip()
        if condition in result:
            raise ValueError(f"Duplicate condition: {condition}")
        result[condition] = Path(raw_path).expanduser()
    missing = sorted(set(CONDITIONS) - set(result))
    extra = sorted(set(result) - set(CONDITIONS))
    if missing or extra:
        raise ValueError(f"Generation conditions mismatch: missing={missing}, extra={extra}")
    return result


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            match_id = str(row.get("match_id") or "")
            if not match_id:
                raise ValueError(f"Missing match_id at {path}:{line_number}")
            result[match_id] = row
    return result


def cache_valid(
    row: dict[str, Any] | None,
    digest: str,
    mapping: dict[str, str],
    args: argparse.Namespace,
) -> bool:
    return bool(
        row
        and not row.get("error")
        and row.get("selected_letter") in ("A", "B")
        and row.get("selected_condition") == mapping.get(row.get("selected_letter"))
        and str(row.get("selection_reason") or "").strip()
        and row.get("prompt_sha256") == digest
        and row.get("option_map") == mapping
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("model") == args.model
        and row.get("base_url") == args.base_url.rstrip("/")
        and row.get("seed") == args.seed
        and row.get("max_tokens") == args.max_tokens
        and row.get("message_layout") == MESSAGE_LAYOUT
    )


def visible_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text") or ""))
        return "\n".join(pieces)
    return str(content or "")


def call_judge(client: Any, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    combined = SYSTEM_PROMPT + "\n\n" + prompt
    traces: list[dict[str, Any]] = []
    last_error = "vLLM judge request failed"
    for attempt in range(1, args.max_attempts + 1):
        attempt_prompt = combined
        if attempt > 1:
            attempt_prompt += "\n\nOutput exactly two lines: Choice: A or B, then Reason:."
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": attempt_prompt}],
                temperature=0,
                max_tokens=args.max_tokens,
                seed=args.seed,
                timeout=args.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            elapsed = time.monotonic() - started
            raw = visible_content(response.choices[0].message)
            choice, reason = four_way.parse_judgment(raw)
            trace = {
                "attempt": attempt,
                "seconds": elapsed,
                "raw": raw,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "usage": (
                    response.usage.model_dump() if getattr(response, "usage", None) else None
                ),
                "parse_ok": choice in ("A", "B") and bool(reason),
            }
            traces.append(trace)
            if choice in ("A", "B") and reason:
                return {"choice": choice, "reason": reason, "raw": raw, "error": None, "traces": traces}
            last_error = f"Unparseable vLLM response: {raw[:1000]!r}"
        except Exception as exc:
            elapsed = time.monotonic() - started
            last_error = f"{type(exc).__name__}: {exc}"
            traces.append({"attempt": attempt, "seconds": elapsed, "error": last_error, "parse_ok": False})
    return {"choice": None, "reason": None, "raw": traces[-1].get("raw", "") if traces else "", "error": last_error, "traces": traces}


def main() -> None:
    args = parse_args()
    if args.limit < 0 or min(args.max_tokens, args.max_attempts) <= 0 or args.timeout <= 0:
        raise ValueError("Invalid limit/token/attempt/timeout settings")
    paths = parse_specs(args.generation)
    items = read_items(paths, args.limit)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai Python package in the vLLM client environment") from exc
    client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/"), timeout=args.timeout)
    models = client.models.list()
    available_models = [model.id for model in models.data]
    if args.model not in available_models:
        raise ValueError(f"Served model {args.model!r} not in {available_models}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "judgments.jsonl"
    cached = {} if args.force else load_cache(output_path)
    query_ids = [item["query_id"] for item in items]
    option_maps = {
        challenger: pairwise.balanced_option_maps(
            query_ids, BASELINE, challenger, args.seed + challenger_index
        )
        for challenger_index, challenger in enumerate(CHALLENGERS)
    }
    results = dict(cached)
    ordered_ids = [
        f"{item['query_id']}::{challenger}_vs_{BASELINE}"
        for challenger in CHALLENGERS
        for item in items
    ]
    total = len(ordered_ids)
    completed = 0
    for challenger in CHALLENGERS:
        for item in items:
            mapping = option_maps[challenger][item["query_id"]]
            prompt = pairwise.make_pair_prompt(item, mapping, prompt_args(args, challenger))
            digest = prompt_hash(prompt)
            match_id = f"{item['query_id']}::{challenger}_vs_{BASELINE}"
            previous = cached.get(match_id)
            if not args.force and cache_valid(previous, digest, mapping, args):
                results[match_id] = previous
                completed += 1
                print(f"[{four_way.timestamp()}] judge={args.judge_name} cached={completed}/{total} match={match_id}", flush=True)
                continue
            print(f"[{four_way.timestamp()}] judge={args.judge_name} START {completed + 1}/{total} match={match_id}", flush=True)
            judged = call_judge(client, prompt, args)
            selected = mapping.get(judged["choice"]) if judged["choice"] else None
            results[match_id] = {
                "query_id": item["query_id"],
                "source_id": item.get("source_id"),
                "match_id": match_id,
                "mode": "vllm_pairwise_lived_in_human_presence_vs_pure",
                "pair": [BASELINE, challenger],
                "challenger": challenger,
                "option_map": mapping,
                "option_texts": {letter: item["responses"][condition] for letter, condition in mapping.items()},
                "selected_letter": judged["choice"],
                "selected_condition": selected,
                "challenger_won": selected == challenger if selected else None,
                "selection_reason": judged["reason"],
                "raw_judge_response": judged["raw"],
                "error": judged["error"],
                "attempt_traces": judged["traces"],
                "prompt_sha256": digest,
                "prompt_version": PROMPT_VERSION,
                "message_layout": MESSAGE_LAYOUT,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompt,
                "judge_name": args.judge_name,
                "model": args.model,
                "base_url": args.base_url.rstrip("/"),
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "created_at": four_way.timestamp(),
            }
            completed += 1
            atomic_jsonl(output_path, [results[key] for key in ordered_ids if key in results])
            status = "OK" if not judged["error"] else f"ERROR {judged['error'][:300]}"
            print(f"[{four_way.timestamp()}] judge={args.judge_name} completed={completed}/{total} match={match_id} {status}", flush=True)

    ordered_rows = [results[key] for key in ordered_ids if key in results]
    atomic_jsonl(output_path, ordered_rows)
    by_method: dict[str, Any] = {}
    invalid = 0
    for challenger in CHALLENGERS:
        relevant = [row for row in ordered_rows if row["challenger"] == challenger]
        valid = [row for row in relevant if not row.get("error") and row.get("selected_condition")]
        invalid += len(relevant) - len(valid)
        wins = sum(row["selected_condition"] == challenger for row in valid)
        pure_wins = sum(row["selected_condition"] == BASELINE for row in valid)
        by_method[challenger] = {
            "total": len(relevant), "valid": len(valid), "invalid_or_failed": len(relevant) - len(valid),
            "challenger_wins": wins, "pure_qwen3_wins": pure_wins,
            "challenger_win_rate": wins / len(valid) if valid else None,
            "pure_qwen3_win_rate": pure_wins / len(valid) if valid else None,
            "challenger_option_A_count": sum(row["option_map"]["A"] == challenger for row in relevant),
            "challenger_option_B_count": sum(row["option_map"]["B"] == challenger for row in relevant),
        }
    summary = {
        "mode": "vllm_pairwise_lived_in_human_presence_vs_pure",
        "judge_name": args.judge_name, "model": args.model, "base_url": args.base_url.rstrip("/"),
        "prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
        "baseline": BASELINE, "challengers": list(CHALLENGERS), "query_count": len(items),
        "expected_comparisons": total, "recorded_comparisons": len(ordered_rows), "by_method": by_method,
        "interpretation_note": (
            "Relative lived-in human presence versus pure_qwen3; not response quality, "
            "and reply length is not a judging criterion."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", {
        "created_at": four_way.timestamp(), "generation_files": {key: str(value.resolve()) for key, value in paths.items()},
        "judge_name": args.judge_name, "model": args.model, "base_url": args.base_url.rstrip("/"),
        "query_ids": query_ids, "seed": args.seed, "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts, "prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
    })
    if invalid:
        raise SystemExit(f"vLLM judge failed: invalid_or_failed={invalid}/{total}")
    print(f"[{four_way.timestamp()}] vLLM judge complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
