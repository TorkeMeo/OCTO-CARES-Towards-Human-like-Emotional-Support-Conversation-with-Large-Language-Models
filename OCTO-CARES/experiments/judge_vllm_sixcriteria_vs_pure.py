#!/usr/bin/env python3
"""Pairwise A/B judgments on six non-Comforting criteria, each method vs pure Qwen.

Independent of the old humanlike and eight-dimension judges. Each request judges
exactly one criterion, with its own cache row. The old balanced A/B mapping is
reused, with Comforting intentionally excluded from this protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import judge_vllm_humanlikeness_vs_pure as legacy

BASELINE = legacy.BASELINE
CHALLENGERS = legacy.CHALLENGERS
CONDITIONS = legacy.CONDITIONS
PROMPT_VERSION = "user-six-criteria-no-comforting-single-dimension-pairwise-v1"
MESSAGE_LAYOUT = "single_user_one_criterion_context_candidates_format"
CRITERIA = {
    "Identification": "Which of the responses offers a deeper understanding of the seeker's situation and better summarizes his issues, rather than providing generic, standard responses?",
    "Suggestion": "Which response provided more helpful suggestions or mentioned examples that can be drawn upon?",
    "Diversity": "Which response is expressed in a more diverse and comprehensive manner rather than a monotonous and rigid style?",
    "Informativeness": "Which reply is more specific rather than just making empty promises?",
    "Coherence": "whether the conversation is on-topic and in-depth and whether the topic transition is natural",
    "Stability": "Which reply provides more persuasive examples or arguments when offering suggestions or expressing understanding?",
}
DIMENSIONS = tuple(CRITERIA)
BIAS_INSTRUCTION = "Do not make judgments merely based on option order or response length."
OUTPUT_CONTRACT = (
    "Output exactly two lines in the following format. Choose A or B for this criterion "
    "and give a brief single-line reason. Do not output any other text:\n"
    "Choice: <A or B>\nReason: <brief reason>"
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rubric_text(dimension: str) -> str:
    if dimension not in CRITERIA:
        raise ValueError(f"Unknown criterion: {dimension}")
    return f"{dimension}: {CRITERIA[dimension]}"


def make_prompt(item: dict, mapping: dict, dimension: str) -> str:
    # Preserve full generation context and responses; never clip or rewrite them.
    return "\n\n".join([
        BIAS_INSTRUCTION, rubric_text(dimension),
        "<context>\n" + item["dialogue"] + "\n</context>",
        "<option_A>\n" + item["responses"][mapping["A"]] + "\n</option_A>",
        "<option_B>\n" + item["responses"][mapping["B"]] + "\n</option_B>",
        OUTPUT_CONTRACT,
    ])


def parse_judgment(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("Judge output must be text")
    normalized = unicodedata.normalize("NFKC", raw.strip())
    if normalized.startswith("```") and normalized.endswith("```"):
        normalized = "\n".join(normalized.splitlines()[1:-1]).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"Expected one Choice line and one Reason line, got {len(lines)}")
    choice = re.fullmatch(r"Choice\s*:\s*([AB])", lines[0])
    reason = re.fullmatch(r"Reason\s*:\s*(.+)", lines[1])
    if choice is None or reason is None or not reason.group(1).strip():
        raise ValueError("Invalid single-criterion Choice/Reason output")
    return {"choice": choice.group(1), "reason": reason.group(1).strip()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--judge-name", default="")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    if args.limit < 0 or min(args.max_tokens, args.max_attempts, args.request_workers) <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid limit/token/retry/concurrency/timeout")
    if not args.check_only and not all((args.base_url, args.model, args.judge_name)):
        parser.error("Evaluation requires --base-url, --model and --judge-name")
    if args.base_url:
        from urllib.parse import urlsplit
        url = urlsplit(args.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            parser.error("base-url must not contain embedded credentials, query or fragment")
    return args


def read_items(paths: dict[str, Path], limit: int) -> list[dict]:
    orders, rows = {}, {}
    for method in CONDITIONS:
        order, data = [], {}
        with paths[method].open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object generation: {paths[method]}:{line_no}")
                query = row.get("query_id")
                if not isinstance(query, str) or not query.strip() or query in data:
                    raise ValueError(f"Invalid/duplicate query ID: {paths[method]}:{line_no}")
                for field in ("response", "dialogue_prefix", "last_seeker"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"Missing text {field}: {method}/{query}")
                if row.get("condition") != method or row.get("error"):
                    raise ValueError(f"Wrong condition or failed generation: {method}/{query}")
                order.append(query)
                data[query] = row
        if not order:
            raise ValueError(f"Empty generation file: {paths[method]}")
        orders[method], rows[method] = order, data
    reference = orders[BASELINE]
    for method in CHALLENGERS:
        if orders[method] != reference:
            raise ValueError(f"Query coverage/order mismatch: {method}")
    if limit < 0 or limit > len(reference):
        raise ValueError("limit exceeds available queries")
    items = []
    for query in reference:
        base = rows[BASELINE][query]
        for method in CHALLENGERS:
            other = rows[method][query]
            for field in ("dialogue_prefix", "last_seeker", "source_id"):
                if other.get(field) != base.get(field):
                    raise ValueError(f"{field} mismatch: {method}/{query}")
        # The official multiagent writer stores summary+latest in dialogue_prefix
        # without duplicating either summary field. That shared context is what
        # the judge sees; missing optional metadata is not a context mismatch.
        # When a summary field IS supplied, still reject malformed/conflicting
        # metadata, including when the baseline itself omits the duplicate.
        reference_summary = None
        for method in CONDITIONS:
            current = rows[method][query]
            field = next((name for name in ("input_summary", "summary") if name in current), None)
            if field is None:
                continue
            summary = current[field]
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"Invalid optional {field}: {method}/{query}")
            if reference_summary is not None and summary != reference_summary:
                raise ValueError(f"summary mismatch: {method}/{query}")
            reference_summary = summary
        items.append({"query_id": query, "source_id": base.get("source_id"),
                      "dialogue": base["dialogue_prefix"],
                      "responses": {method: rows[method][query]["response"] for method in CONDITIONS}})
    return items[:limit] if limit else items


def build_tasks(items: list[dict], args) -> list[dict]:
    maps = {method: legacy.pairwise.balanced_option_maps(
        [item["query_id"] for item in items], BASELINE, method, args.seed + index)
        for index, method in enumerate(CHALLENGERS)}
    tasks = []
    for method in CHALLENGERS:
        for item in items:
            mapping = maps[method][item["query_id"]]
            pair_id = f"{item['query_id']}::{method}_vs_{BASELINE}"
            for dimension in DIMENSIONS:
                prompt = make_prompt(item, mapping, dimension)
                tasks.append({"match_id": f"{pair_id}::{dimension}",
                              "pair_id": pair_id, "dimension": dimension,
                              "query_id": item["query_id"], "source_id": item["source_id"],
                              "challenger": method, "pair": [BASELINE, method], "option_map": mapping,
                              "option_texts": {letter: item["responses"][name] for letter, name in mapping.items()},
                              "context_sha256": sha256_text(item["dialogue"]),
                              "prompt_sha256": sha256_text(prompt), "user_prompt": prompt})
    return tasks


def settings_for(args) -> dict:
    return {name: getattr(args, name) for name in (
        "model", "judge_name", "base_url", "seed", "limit", "max_tokens", "max_attempts",
        "request_workers", "timeout", "api_key_env")}


def verify_inputs(paths, hashes):
    if {name: file_hash(path) for name, path in paths.items()} != hashes:
        raise ValueError("Generation files changed; use a new evaluation output directory")


def prepare_run(args):
    paths = {name: path.resolve() for name, path in legacy.parse_specs(args.generation).items()}
    output = args.output_dir.resolve()
    for path in paths.values():
        if output == path.parent or output in path.parents or path.parent in output.parents:
            raise ValueError("Output directory must be separate from generation directories")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    items = read_items(paths, args.limit)
    tasks = build_tasks(items, args)
    verify_inputs(paths, hashes)
    signature = {
        "prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
        "criteria": CRITERIA, "bias_instruction": BIAS_INSTRUCTION, "output_contract": OUTPUT_CONTRACT,
        "generation_files": {name: str(path) for name, path in paths.items()}, "input_sha256": hashes,
        "query_ids": [item["query_id"] for item in items], "match_ids": [task["match_id"] for task in tasks],
        "evaluation_unit": "pair_dimension", "criteria_per_request": 1,
        "pair_count": len(items) * len(CHALLENGERS), "criterion_task_count": len(tasks),
        "query_count": len(items), "total": len(tasks), "settings": settings_for(args),
        "judge_name": args.judge_name, "model": args.model, "base_url": args.base_url,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "prompt_set_sha256": sha256_text(canonical([task["prompt_sha256"] for task in tasks])),
        "code_sha256": {name: file_hash(Path(__file__).parent / name) for name in (
            Path(__file__).name, "judge_pairwise_humanlikeness.py", "judge_vllm_humanlikeness_vs_pure.py",
            "judge_local_humanlikeness_vs_pure.py", "judge_humanlikeness_qwen37.py")},
    }
    fingerprint = sha256_text(canonical(signature))
    manifest = {**signature, "fingerprint": fingerprint, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_signature = {k: v for k, v in saved.items() if k not in ("fingerprint", "created_at")}
        if saved.get("fingerprint") != sha256_text(canonical(saved_signature)) or saved_signature != signature:
            raise ValueError("Frozen evaluation inputs/settings/code changed; use a new output directory")
        manifest = saved
    elif output.exists() and (not output.is_dir() or any(p.name not in {"judge.log", "run.lock"} for p in output.iterdir())):
        raise ValueError("Refusing unknown nonempty evaluation output directory")
    if args.check_only:
        return paths, hashes, items, tasks, manifest
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return paths, hashes, items, tasks, manifest


def cache_valid(row, task, args) -> bool:
    if not row or row.get("error") or row.get("finish_reason") != "stop":
        return False
    for name in ("prompt_sha256", "context_sha256", "option_map", "option_texts", "match_id", "pair_id", "dimension", "query_id", "source_id", "challenger", "pair", "user_prompt"):
        if row.get(name) != task[name]:
            return False
    if row.get("prompt_version") != PROMPT_VERSION or row.get("message_layout") != MESSAGE_LAYOUT:
        return False
    for name in ("model", "judge_name", "base_url", "seed", "max_tokens"):
        if row.get(name) != getattr(args, name):
            return False
    try:
        parsed = parse_judgment(row.get("raw_judge_response"))
        return (row.get("choice") == parsed["choice"]
                and row.get("reason") == parsed["reason"]
                and row.get("selected_condition") == task["option_map"][parsed["choice"]]
                and "dimensions" not in row)
    except (ValueError, TypeError):
        return False


def load_cache(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("match_id"), str):
                    raise ValueError("missing match_id")
            except ValueError as exc:
                raise ValueError(f"Invalid evaluation journal {path}:{line_no}: {exc}") from exc
            records[row["match_id"]] = row
    return records


def call_judge(client, task: dict, args) -> dict:
    record = {**task, "prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
              "model": args.model, "judge_name": args.judge_name, "base_url": args.base_url,
              "seed": args.seed, "max_tokens": args.max_tokens, "temperature": 0,
              "choice": None, "reason": None, "selected_condition": None,
              "raw_judge_response": "", "finish_reason": None,
              "error": "Judge not called", "attempt_traces": [], "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    api_key = os.getenv(getattr(args, "api_key_env", "JUDGE_API_KEY"))
    for attempt in range(1, args.max_attempts + 1):
        prompt = task["user_prompt"]
        if attempt > 1:
            prompt += "\n\n" + OUTPUT_CONTRACT  # output-format reminder only
        started = time.monotonic()
        trace = {"attempt": attempt, "request_prompt_sha256": sha256_text(prompt)}
        try:
            response = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}],
                temperature=0, seed=args.seed, max_tokens=args.max_tokens, timeout=args.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = response.choices[0]
            raw = legacy.visible_content(choice.message)
            finish_reason = getattr(choice, "finish_reason", None)
            trace.update(raw=raw, finish_reason=finish_reason,
                         usage=response.usage.model_dump() if getattr(response, "usage", None) else None)
            record["raw_judge_response"], record["finish_reason"] = raw, finish_reason
            if finish_reason != "stop":
                raise ValueError(f"Judge output did not finish normally: {finish_reason}")
            parsed = parse_judgment(raw)
            record.update(parsed, selected_condition=task["option_map"][parsed["choice"]])
            record["error"] = None
            trace["parse_ok"] = True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
            record["error"] = message
            trace.update(error=message, parse_ok=False)
        trace["seconds"] = time.monotonic() - started
        record["attempt_traces"].append(trace)
        if not record["error"]:
            break
    return record


def summarize(tasks: list[dict], records: dict, args) -> dict:
    by_method = {}
    for method in CHALLENGERS:
        relevant = [task for task in tasks if task["challenger"] == method]
        stats = {}
        for dimension in DIMENSIONS:
            dimension_tasks = [task for task in relevant if task["dimension"] == dimension]
            valid = [records[task["match_id"]] for task in dimension_tasks
                     if cache_valid(records.get(task["match_id"]), task, args)]
            wins = sum(row["selected_condition"] == method for row in valid)
            stats[dimension] = {
                "total": len(dimension_tasks), "valid": len(valid), "invalid_or_failed": len(dimension_tasks) - len(valid),
                "challenger_wins": wins, "pure_qwen3_wins": len(valid) - wins,
                "challenger_win_rate": wins / len(valid) if valid else None,
                "pure_qwen3_win_rate": (len(valid) - wins) / len(valid) if valid else None,
            }
        pair_tasks = {task["pair_id"]: task for task in relevant}.values()
        by_method[method] = {"dimensions": stats,
                             "challenger_option_A_count": sum(t["option_map"]["A"] == method for t in pair_tasks),
                             "challenger_option_B_count": sum(t["option_map"]["B"] == method for t in pair_tasks)}
    valid_count = sum(cache_valid(records.get(task["match_id"]), task, args) for task in tasks)
    return {"prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
            "judge_name": args.judge_name, "model": args.model, "baseline": BASELINE,
            "criteria": CRITERIA, "evaluation_unit": "pair_dimension", "criteria_per_request": 1,
            "query_count": len({task["query_id"] for task in tasks}),
            "pair_count": len({task["pair_id"] for task in tasks}),
            "expected_comparisons": len(tasks), "valid_comparisons": valid_count,
            "invalid_or_failed": len(tasks) - valid_count, "complete": valid_count == len(tasks),
            "by_method": by_method,
            "note": "Each criterion is judged in an independent request and has its own A/B win rate. No macro score or overall winner is computed. Failed comparisons are reported, not counted as losses."}


def write_summary(output: Path, summary: dict):
    legacy.write_json(output / "summary.json", summary)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["judge", "challenger", "dimension", "total", "valid", "invalid_or_failed",
                     "challenger_wins", "pure_qwen3_wins", "challenger_win_rate", "pure_qwen3_win_rate"])
    for method, values in summary["by_method"].items():
        for name, counts in values["dimensions"].items():
            writer.writerow([summary["judge_name"], method, name] + [counts[key] for key in (
                "total", "valid", "invalid_or_failed", "challenger_wins", "pure_qwen3_wins", "challenger_win_rate", "pure_qwen3_win_rate")])
    (output / "win_rates.tsv").write_text(stream.getvalue(), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    paths, hashes, items, tasks, manifest = prepare_run(args)
    if args.print_prompt:
        print(tasks[0]["user_prompt"], flush=True)
    if args.check_only:
        print(f"SIXCRITERIA_CHECK_OK queries={len(items)} pairs={len(items) * len(CHALLENGERS)} comparisons={len(tasks)} criteria=6 criteria_per_request=1", flush=True)
        return 0
    # flock is released even if the client exits unexpectedly; no stale PID lock.
    import fcntl
    output = args.output_dir.resolve()
    with (output / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another client is using this judge output directory") from exc
        journal = output / "judgments.jsonl"
        records = load_cache(journal)
        known = {task["match_id"] for task in tasks}
        if set(records) - known:
            raise ValueError("Cache contains comparisons outside the selected input; use a new output directory")
        pending = [task for task in tasks if not cache_valid(records.get(task["match_id"]), task, args)]
        done = len(tasks) - len(pending)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] judge={args.judge_name} queries={len(items)} pairs={len(items) * len(CHALLENGERS)} comparisons={len(tasks)} criteria=6 criteria_per_request=1 cached={done} pending={len(pending)}", flush=True)
        write_summary(output, summarize(tasks, records, args))
        client = None
        if pending:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv(args.api_key_env) or "local-vllm", base_url=args.base_url, timeout=args.timeout, max_retries=0)
            available = [model.id for model in client.models.list().data]
            if args.model not in available:
                raise ValueError(f"Requested model {args.model!r} not served: {available}")
        try:
            with journal.open("a", encoding="utf-8") as handle:
                with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
                    futures = {pool.submit(call_judge, client, task, args): task for task in pending}
                    for future in as_completed(futures):
                        row = future.result()
                        records[row["match_id"]] = row
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        done += 1
                        status = "OK" if not row["error"] else "ERROR " + row["error"][:250]
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] judge={args.judge_name} completed={done}/{len(tasks)} match={row['match_id']} {status}", flush=True)
                        if done % 25 == 0:
                            write_summary(output, summarize(tasks, records, args))
        finally:
            if client is not None:
                client.close()
        verify_inputs(paths, hashes)
        summary = summarize(tasks, records, args)
        write_summary(output, summary)
        if not summary["complete"]:
            print(f"SIXCRITERIA_INCOMPLETE invalid_or_failed={summary['invalid_or_failed']}; rerun to retry", flush=True)
            return 1
        print(f"SIXCRITERIA_COMPLETE output={output}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
