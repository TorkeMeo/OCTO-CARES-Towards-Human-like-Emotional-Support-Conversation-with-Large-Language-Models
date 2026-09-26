#!/usr/bin/env python3
"""Aggregate three judges' independent per-pair, per-criterion A/B calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from judge_vllm_sixcriteria_vs_pure import parse_judgment


BASELINE = "pure_qwen3"
CHALLENGERS = (
    "attention_post", "attention_post_comment", "attention_post_random_comment",
    "simple_persona", "reddit_multiagent",
)
DIMENSIONS = (
    "Identification", "Suggestion", "Diversity", "Informativeness",
    "Coherence", "Stability",
)
PROMPT_VERSION = "user-six-criteria-no-comforting-single-dimension-pairwise-v1"
MESSAGE_LAYOUT = "single_user_one_criterion_context_candidates_format"
INPUT_FIELDS = (
    "match_id", "pair_id", "dimension", "query_id", "source_id", "challenger", "pair", "option_map", "option_texts",
    "context_sha256", "prompt_sha256", "user_prompt", "prompt_version", "message_layout", "seed",
)
ARTIFACTS = {
    "aggregation.manifest.json", "majority.records.jsonl", "majority.summary.json",
    "majority.win_rates.tsv", "majority.win_rates.matrix.tsv",
}


def parse_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip() or name in specs:
            raise ValueError(f"Invalid or duplicate --judge: {value!r}")
        specs[name] = Path(raw_path).expanduser()
    if len(specs) != 3 or len({path.resolve() for path in specs.values()}) != 3:
        raise ValueError("Exactly three unique judges and distinct input files are required")
    return specs


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in INPUT_FIELDS)


def validate_input(row: dict[str, Any], judge: str, location: str) -> None:
    if "dimensions" in row:
        raise ValueError(f"Old multi-criterion dimensions payload is not accepted at {location}")
    query_id, challenger = row.get("query_id"), row.get("challenger")
    if not isinstance(query_id, str) or not query_id.strip() or challenger not in CHALLENGERS:
        raise ValueError(f"Invalid query_id/challenger at {location}")
    pair_id = f"{query_id}::{challenger}_vs_{BASELINE}"
    dimension = row.get("dimension")
    if dimension not in DIMENSIONS or row.get("pair_id") != pair_id:
        raise ValueError(f"Invalid pair_id/dimension at {location}")
    if row.get("match_id") != f"{pair_id}::{dimension}":
        raise ValueError(f"Invalid match_id at {location}")
    if row.get("judge_name") != judge or not isinstance(row.get("model"), str) or not row["model"].strip():
        raise ValueError(f"Judge/model identity mismatch at {location}")
    if row.get("prompt_version") != PROMPT_VERSION or row.get("message_layout") != MESSAGE_LAYOUT:
        raise ValueError(f"Protocol mismatch at {location}")
    mapping = row.get("option_map")
    texts = row.get("option_texts")
    if not isinstance(mapping, dict) or set(mapping) != {"A", "B"} or any(not isinstance(value, str) for value in mapping.values()) or set(mapping.values()) != {BASELINE, challenger}:
        raise ValueError(f"Invalid option_map at {location}")
    if row.get("pair") != [BASELINE, challenger]:
        raise ValueError(f"Invalid pair at {location}")
    if not isinstance(texts, dict) or set(texts) != {"A", "B"} or any(not isinstance(value, str) or not value.strip() for value in texts.values()):
        raise ValueError(f"Missing option_texts at {location}")
    for field in ("context_sha256", "prompt_sha256"):
        if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
            raise ValueError(f"Invalid {field} at {location}")
    prompt = row.get("user_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row["prompt_sha256"]:
        raise ValueError(f"Prompt hash mismatch at {location}")
    if type(row.get("seed")) is not int:
        raise ValueError(f"Missing seed at {location}")
    if type(row.get("max_tokens")) is not int or row["max_tokens"] <= 0:
        raise ValueError(f"Invalid max_tokens at {location}")


def load_judge(path: Path, judge: str, expected_count: int | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing completion manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("prompt_version") != PROMPT_VERSION or manifest.get("message_layout") != MESSAGE_LAYOUT:
        raise ValueError(f"Invalid protocol in {manifest_path}")
    if manifest.get("evaluation_unit") != "pair_dimension" or type(manifest.get("criteria_per_request")) is not int or manifest["criteria_per_request"] != 1:
        raise ValueError(f"Expected independent single-criterion requests in {manifest_path}")
    if manifest.get("judge_name", judge) != judge:
        raise ValueError(f"Judge identity mismatch in {manifest_path}")
    query_ids = manifest.get("query_ids")
    if not isinstance(query_ids, list) or not query_ids or any(not isinstance(value, str) or not value.strip() for value in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError(f"Invalid query_ids in {manifest_path}")
    if expected_count is not None and len(query_ids) != expected_count:
        raise ValueError(f"Expected {expected_count} queries, found {len(query_ids)} in {manifest_path}")
    pair_ids = {f"{query}::{method}_vs_{BASELINE}" for query in query_ids for method in CHALLENGERS}
    expected_ids = {f"{pair_id}::{dimension}" for pair_id in pair_ids for dimension in DIMENSIONS}
    if "match_ids" in manifest and (len(manifest["match_ids"]) != len(expected_ids) or set(manifest["match_ids"]) != expected_ids):
        raise ValueError(f"Manifest match_ids mismatch: {manifest_path}")
    if "query_count" in manifest and manifest["query_count"] != len(query_ids):
        raise ValueError(f"Manifest query_count mismatch: {manifest_path}")
    if "pair_count" in manifest and manifest["pair_count"] != len(pair_ids):
        raise ValueError(f"Manifest pair_count mismatch: {manifest_path}")
    if "pair_ids" in manifest and (len(manifest["pair_ids"]) != len(pair_ids) or set(manifest["pair_ids"]) != pair_ids):
        raise ValueError(f"Manifest pair_ids mismatch: {manifest_path}")
    for count_field in ("total", "comparisons", "match_count", "judgment_count", "criterion_task_count"):
        if count_field in manifest and manifest[count_field] != len(expected_ids):
            raise ValueError(f"Manifest {count_field} mismatch: {manifest_path}")
    rows: dict[str, dict[str, Any]] = {}
    settings = manifest.get("settings") or {}
    judge_settings: tuple[Any, ...] | None = None
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            location = f"{path}:{number}"
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row at {location}")
            validate_input(row, judge, location)
            current_settings = tuple(row.get(key) for key in ("model", "seed", "max_tokens"))
            if judge_settings is not None and current_settings != judge_settings:
                raise ValueError(f"Mixed judge settings at {location}")
            judge_settings = current_settings
            for key in ("model", "seed", "max_tokens"):
                if key in manifest or key in settings:
                    if row.get(key) != manifest.get(key, settings.get(key)):
                        raise ValueError(f"Manifest {key} mismatch at {location}")
            old = rows.get(row["match_id"])
            if old is not None and identity(old) != identity(row):
                raise ValueError(f"Mixed cached inputs at {location}")
            rows[row["match_id"]] = row
    if set(rows) != expected_ids:
        raise ValueError(f"Incomplete five-challenger/six-dimension coverage for {judge}: missing={sorted(expected_ids - set(rows))[:5]}, extra={sorted(set(rows) - expected_ids)[:5]}")
    contexts: dict[str, tuple[Any, Any]] = {}
    pairs: dict[str, tuple[Any, Any]] = {}
    for match_id, row in rows.items():
        context = (row.get("source_id"), row["context_sha256"])
        if row["query_id"] in contexts and contexts[row["query_id"]] != context:
            raise ValueError(f"Inconsistent query context for {judge}/{match_id}")
        contexts[row["query_id"]] = context
        pair_input = (row["option_map"], row["option_texts"])
        if row["pair_id"] in pairs and pairs[row["pair_id"]] != pair_input:
            raise ValueError(f"Inconsistent pair inputs across dimensions for {judge}/{match_id}")
        pairs[row["pair_id"]] = pair_input
        if row.get("error") or row.get("finish_reason") != "stop":
            raise ValueError(f"Failed/incomplete dimension for {judge}/{match_id}")
        if row.get("choice") not in ("A", "B") or row.get("selected_condition") != row["option_map"][row["choice"]] or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"Invalid {row['dimension']} result for {judge}/{match_id}")
        try:
            parsed = parse_judgment(row.get("raw_judge_response"))
        except ValueError as exc:
            raise ValueError(f"Unparseable raw response for {judge}/{match_id}: {exc}") from exc
        if parsed != {"choice": row["choice"], "reason": row["reason"]}:
            raise ValueError(f"Raw response differs from parsed fields for {judge}/{match_id}")
    return rows, manifest


def summarize(winners: list[str], method: str, agreement: list[int] | None = None) -> dict[str, Any]:
    wins = winners.count(method)
    total = len(winners)
    result = {"total": total, "wins": wins, "losses": total - wins, "unresolved": 0,
              "challenger_win_rate": wins / total, "pure_qwen3_win_rate": (total - wins) / total}
    if agreement is not None:
        result.update({"unanimous_3_0": agreement.count(3), "split_2_1": agreement.count(2)})
    return result


def atomic_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate(specs: dict[str, Path], output_dir: Path, expected_count: int | None = None, check_only: bool = False) -> dict[str, Any]:
    if expected_count is not None and expected_count <= 0:
        raise ValueError("expected-count must be positive")
    specs = parse_specs([f"{name}={path}" for name, path in specs.items()])
    loaded = {name: load_judge(path, name, expected_count) for name, path in specs.items()}
    judges = {name: pair[0] for name, pair in loaded.items()}
    reference = next(iter(judges.values()))
    query_ids = next(iter(loaded.values()))[1]["query_ids"]
    if len({next(iter(rows.values()))["model"] for rows in judges.values()}) != 3:
        raise ValueError("Three distinct judge models are required")
    for name, rows in judges.items():
        if set(rows) != set(reference):
            raise ValueError(f"Judge coverage mismatch: {name}")
        for match_id, row in rows.items():
            if identity(row) != identity(reference[match_id]):
                raise ValueError(f"Judge input mismatch: {name}/{match_id}")
    records = []
    for method in CHALLENGERS:
        for query_id in query_ids:
            for criterion in DIMENSIONS:
                match_id = f"{query_id}::{method}_vs_{BASELINE}::{criterion}"
                row = reference[match_id]
                votes = {name: rows[match_id]["selected_condition"] for name, rows in judges.items()}
                winner, count = Counter(votes.values()).most_common(1)[0]
                records.append({
                    **{key: row.get(key) for key in INPUT_FIELDS},
                    "votes": votes, "winner": winner, "challenger_won": winner == method,
                    "vote_counts": {method: list(votes.values()).count(method), BASELINE: list(votes.values()).count(BASELINE)},
                    "agreement": "3-0" if count == 3 else "2-1",
                    "judge_details": {name: {field: rows[match_id].get(field) for field in (
                        "choice", "selected_condition", "reason", "raw_judge_response", "finish_reason", "attempt_traces",
                    )} for name, rows in judges.items()},
                })
    by_method, by_judge = {}, {name: {} for name in judges}
    for method in CHALLENGERS:
        selected = [row for row in records if row["challenger"] == method]
        by_method[method] = {}
        for name in judges:
            by_judge[name][method] = {}
        for criterion in DIMENSIONS:
            results = [row for row in selected if row["dimension"] == criterion]
            by_method[method][criterion] = summarize([row["winner"] for row in results], method, [3 if row["agreement"] == "3-0" else 2 for row in results])
            for name in judges:
                by_judge[name][method][criterion] = summarize([row["votes"][name] for row in results], method)
    provenance = {"prompt_version": PROMPT_VERSION, "message_layout": MESSAGE_LAYOUT,
                  "judges": {name: str(path.resolve()) for name, path in specs.items()}, "query_ids": query_ids,
                  "comparison_inputs_sha256": hashlib.sha256(json.dumps(
                      {key: identity(reference[key]) for key in sorted(reference)},
                      ensure_ascii=False, sort_keys=True,
                  ).encode("utf-8")).hexdigest()}
    summary = {**provenance, "baseline": BASELINE, "criteria": list(DIMENSIONS), "query_count": len(query_ids),
               "evaluation_unit": "pair_dimension", "criteria_per_request": 1,
               "comparisons": len(records), "pair_count": len(query_ids) * len(CHALLENGERS),
               "judgment_count": len(records), "judgments_per_pair": len(DIMENSIONS),
               "by_method": by_method, "by_judge": by_judge,
               "judge_models": {name: next(iter(rows.values()))["model"] for name, rows in judges.items()},
               "interpretation_note": "Each criterion was judged in a separate LLM call. A challenger wins with at least two of three votes for that criterion; all three complete votes are required. No cross-criterion composite is calculated."}
    if check_only:
        return summary
    for path in specs.values():
        if output_dir.resolve() == path.parent.resolve():
            raise ValueError("Aggregation output must be separate from judge inputs")
    if output_dir.exists() and any(output_dir.iterdir()):
        marker = output_dir / "aggregation.manifest.json"
        if any(path.name not in ARTIFACTS or not path.is_file() for path in output_dir.iterdir()) or not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != provenance:
            raise ValueError(f"Refusing unknown or incompatible nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(output_dir / "aggregation.manifest.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    atomic_text(output_dir / "majority.records.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    table = io.StringIO()
    writer = csv.writer(table, delimiter="\t", lineterminator="\n")
    writer.writerow(["challenger", "dimension", "total", "wins", "losses", "unresolved", "challenger_win_rate", "pure_qwen3_win_rate", "unanimous_3_0", "split_2_1"])
    for method in CHALLENGERS:
        for criterion in DIMENSIONS:
            writer.writerow([method, criterion, *by_method[method][criterion].values()])
    atomic_text(output_dir / "majority.win_rates.tsv", table.getvalue())
    matrix = io.StringIO()
    writer = csv.writer(matrix, delimiter="\t", lineterminator="\n")
    writer.writerow(["challenger", *DIMENSIONS])
    for method in CHALLENGERS:
        writer.writerow([method, *(by_method[method][criterion]["challenger_win_rate"] for criterion in DIMENSIONS)])
    atomic_text(output_dir / "majority.win_rates.matrix.tsv", matrix.getvalue())
    atomic_text(output_dir / "majority.summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = aggregate(parse_specs(args.judge), args.output_dir, args.expected_count, args.check_only)
    print(json.dumps({"check_only": args.check_only, "queries": result["query_count"], "comparisons": result["comparisons"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
