#!/usr/bin/env python3
"""Summarize the optional post-hoc report for the 1053 classifier run.

This is intentionally stdlib-only so it can run in any shell after the eval
finishes. It reads the CSV/JSON files written by eight_classifier_1053_attention_eval.py
and prints the retrieval, positive-Jaccard, label-overlap, rank, score, vector,
and pairwise diagnostics in one report.  It is not used by the default
no-gold extraction run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METHOD_ALIAS = {
    "qwen_base_hidden_post_mean": "base_mean",
    "qwen_base_hidden_adapter_attention_mean_label": "base+attn",
    "qwen_base_hidden_delta_ratio_attention_mean_label": "base+delta",
    "qwen_hidden_attention_mean_label": "adapter+attn",
    "qwen_delta_ratio_attention_mean_label": "adapter+delta",
    "qwen_hidden_attention_change_rate_mean_label": "adapter+rate",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 4) -> str:
    parsed = as_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def fmt_bool(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "TRUE"
    if text in {"false", "0", "no"}:
        return "FALSE"
    return str(value)


def alias(method: str) -> str:
    return METHOD_ALIAS.get(method, method)


def discover_top_ks(rows: list[dict[str, str]]) -> list[int]:
    values: set[int] = set()
    for row in rows:
        for key in row:
            match = re.fullmatch(r"top(\d+)_accuracy", key)
            if match:
                values.add(int(match.group(1)))
    return sorted(values)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = []
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    for row in string_rows:
        lines.append("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))
    return "\n".join(lines)


def find_one(directory: Path, suffix: str) -> Path | None:
    matches = sorted(directory.glob(f"*{suffix}"))
    return matches[0] if matches else None


def layer_dirs(run_dir: Path) -> list[Path]:
    candidates = [item for item in sorted(run_dir.iterdir()) if item.is_dir() and item.name.startswith("last")]
    return [item for item in candidates if find_one(item, "_method_summary.csv")]


def method_summary_section(rows: list[dict[str, str]], top_ks: list[int]) -> str:
    headers = ["method"]
    headers.extend(f"top{k}" for k in top_ks)
    headers.extend(f"pj_mean@{k}" for k in top_ks)
    headers.extend(f"pj_max@{k}" for k in top_ks)
    headers.extend(f"label_mean@{k}" for k in top_ks)
    headers.extend(f"exact@{k}" for k in top_ks)
    headers.append("total")

    out_rows = []
    for row in rows:
        method = row.get("method", "")
        values: list[Any] = [alias(method)]
        values.extend(fmt_num(row.get(f"top{k}_accuracy")) for k in top_ks)
        values.extend(fmt_num(row.get(f"top{k}_positive_jaccard_mean_query_mean")) for k in top_ks)
        values.extend(fmt_num(row.get(f"top{k}_positive_jaccard_max_query_mean")) for k in top_ks)
        values.extend(fmt_num(row.get(f"top{k}_label_match_mean_query_mean")) for k in top_ks)
        values.extend(fmt_num(row.get(f"top{k}_exact_label_hit_rate")) for k in top_ks)
        values.append(row.get("total", ""))
        out_rows.append(values)
    return table(headers, out_rows)


def by_rank_section(rows: list[dict[str, str]]) -> str:
    headers = [
        "method",
        "rank",
        "same_sub",
        "pos_jaccard",
        "label_match",
        "exact_label",
        "any_pos",
        "shared_pos",
        "missing_pos",
        "extra_pos",
        "count",
    ]
    out_rows = []
    for row in rows:
        out_rows.append(
            [
                alias(row.get("method", "")),
                row.get("rank", ""),
                fmt_num(row.get("same_subreddit_rate")),
                fmt_num(row.get("positive_jaccard_mean")),
                fmt_num(row.get("label_match_fraction_mean")),
                fmt_num(row.get("exact_label_match_rate")),
                fmt_num(row.get("any_shared_positive_rate")),
                fmt_num(row.get("shared_positive_count_mean")),
                fmt_num(row.get("missing_query_positive_count_mean")),
                fmt_num(row.get("extra_neighbor_positive_count_mean")),
                row.get("count", ""),
            ]
        )
    return table(headers, out_rows)


def score_diag_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "missing score diagnostics"
    tie_key = next((key for key in rows[0] if key.startswith("top_score_tie_count_ge_")), "")
    headers = ["method", "finite", "nonfinite", "std", "min", "max", "unique8", "all_equal", "tie_mean"]
    if tie_key:
        headers.append(tie_key.replace("top_score_tie_count_", ""))
    out_rows = []
    for row in rows:
        values = [
            alias(row.get("method", "")),
            fmt_num(row.get("finite_score_rate")),
            row.get("nonfinite_score_count", ""),
            fmt_num(row.get("finite_score_std")),
            fmt_num(row.get("finite_score_min")),
            fmt_num(row.get("finite_score_max")),
            row.get("finite_score_unique_rounded_8dp", ""),
            fmt_bool(row.get("all_finite_scores_equal_8dp", "")),
            fmt_num(row.get("top_score_tie_count_mean")),
        ]
        if tie_key:
            values.append(fmt_num(row.get(tie_key)))
        out_rows.append(values)
    return table(headers, out_rows)


def pair_check_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "missing method pair checks"
    ordered_key = next((key for key in rows[0] if key.startswith("top") and key.endswith("_ordered_same_rate")), "")
    set_key = next((key for key in rows[0] if key.startswith("top") and key.endswith("_set_jaccard_mean")), "")
    headers = ["left", "right", "same_top1", "ordered_same", "set_jaccard", "mean_abs_diff", "max_abs_diff", "equal"]
    out_rows = []
    for row in rows:
        out_rows.append(
            [
                alias(row.get("left_method", "")),
                alias(row.get("right_method", "")),
                fmt_num(row.get("top1_same_rate")),
                fmt_num(row.get(ordered_key)) if ordered_key else "",
                fmt_num(row.get(set_key)) if set_key else "",
                fmt_num(row.get("score_mean_abs_diff")),
                fmt_num(row.get("score_max_abs_diff")),
                fmt_bool(row.get("scores_exactly_equal", "")),
            ]
        )
    return table(headers, out_rows)


def vector_diag_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "missing vector diagnostics"
    headers = ["method", "shape", "finite", "nan", "inf", "norm_mean", "norm_std", "zero_vec"]
    out_rows = []
    for row in rows:
        out_rows.append(
            [
                alias(row.get("method", "")),
                row.get("shape", ""),
                fmt_num(row.get("finite_value_rate")),
                row.get("nan_value_count", ""),
                row.get("inf_value_count", ""),
                fmt_num(row.get("l2_norm_mean")),
                fmt_num(row.get("l2_norm_std")),
                row.get("zero_vector_count", ""),
            ]
        )
    return table(headers, out_rows)


def summarize_layer(layer_dir: Path) -> str:
    method_path = find_one(layer_dir, "_method_summary.csv")
    if method_path is None:
        return f"## {layer_dir.name}\nmissing method summary\n"
    suffix = "_method_summary.csv"
    prefix = method_path.name[:-len(suffix)] if method_path.name.endswith(suffix) else method_path.stem
    by_rank_path = layer_dir / f"{prefix}_by_rank.csv"
    score_path = layer_dir / f"{prefix}_score_diagnostics.csv"
    vector_path = layer_dir / f"{prefix}_vector_diagnostics.csv"
    pair_path = layer_dir / f"{prefix}_method_pair_checks.csv"
    summary_path = layer_dir / f"{prefix}_summary.json"

    method_rows = read_csv(method_path)
    top_ks = discover_top_ks(method_rows)
    summary = read_json(summary_path)
    shard_info = summary.get("shard_info", {}) if isinstance(summary, dict) else {}

    parts = [f"## {layer_dir.name}"]
    if shard_info:
        parts.append(
            "meta: "
            f"query_count={summary.get('query_count', '')}, "
            f"attention_layers={shard_info.get('attention_layers', '')}, "
            f"answer_mode={shard_info.get('attention_answer_mode', '')}"
        )
    parts.append("\n### Method Summary: retrieval + positive Jaccard + label overlap")
    parts.append(method_summary_section(method_rows, top_ks))
    parts.append("\n### By Rank: rank-level overlap statistics")
    parts.append(by_rank_section(read_csv(by_rank_path)))
    parts.append("\n### Score Diagnostics: check NaN/constant/tied score matrices")
    parts.append(score_diag_section(read_csv(score_path)))
    parts.append("\n### Method Pair Checks: whether methods collapse to same ranking")
    parts.append(pair_check_section(read_csv(pair_path)))
    parts.append("\n### Vector Diagnostics: vector finite/norm checks")
    parts.append(vector_diag_section(read_csv(vector_path)))
    parts.append("\nsource files:")
    for path in [method_path, by_rank_path, score_path, pair_path, vector_path, summary_path]:
        parts.append(f"  {path}")
    return "\n".join(parts) + "\n"


def build_report(run_dir: Path) -> str:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    layers = layer_dirs(run_dir)
    if not layers:
        raise FileNotFoundError(f"No layer result directories with *_method_summary.csv under {run_dir}")
    parts = ["# No-gold Eight-Classifier 1053 Attention Report", f"run_dir: {run_dir}"]
    for layer_dir in layers:
        parts.append(summarize_layer(layer_dir))
    return "\n\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Fresh eval run directory, e.g. outputs/retrieval_test")
    parser.add_argument("--output", default="", help="Optional report path. Default: RUN_DIR/fresh_results_report.txt")
    parser.add_argument("--no-write", action="store_true", help="Only print to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    print(report, end="")
    if not args.no_write:
        output = Path(args.output) if args.output else run_dir / "fresh_results_report.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"\nWrote report: {output}")


if __name__ == "__main__":
    main()
