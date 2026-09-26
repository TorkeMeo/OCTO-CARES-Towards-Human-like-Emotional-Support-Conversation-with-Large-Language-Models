#!/usr/bin/env python3
"""Aggregate three local-judge votes into per-method majority win rates."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from judge_local_humanlikeness_vs_pure import BASELINE, CHALLENGERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --judge: {value!r}")
        result[name.strip()] = Path(raw_path).expanduser()
    if len(result) != 3:
        raise ValueError(f"Exactly three unique judges are required, got {len(result)}")
    return result


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["match_id"])] = row
    return rows


def main() -> None:
    args = parse_args()
    specs = read_specs(args.judge)
    judges = {name: load(path) for name, path in specs.items()}
    reference_ids = set(next(iter(judges.values())))
    for name, rows in judges.items():
        if set(rows) != reference_ids:
            raise ValueError(f"Judge coverage mismatch for {name}: {len(rows)} vs {len(reference_ids)}")
    records = []
    by_method: dict[str, Any] = {}
    for challenger in CHALLENGERS:
        match_ids = sorted(match_id for match_id in reference_ids if f"::{challenger}_vs_{BASELINE}" in match_id)
        wins = losses = unresolved = unanimous = split = 0
        for match_id in match_ids:
            votes = {
                name: row[match_id].get("selected_condition")
                for name, row in judges.items()
                if not row[match_id].get("error")
                and row[match_id].get("selected_condition") in (BASELINE, challenger)
            }
            counts = Counter(votes.values())
            majority = None
            if counts[challenger] >= 2:
                majority = challenger
                wins += 1
            elif counts[BASELINE] >= 2:
                majority = BASELINE
                losses += 1
            else:
                unresolved += 1
            if len(votes) == 3 and len(counts) == 1:
                unanimous += 1
            elif len(votes) == 3 and len(counts) == 2:
                split += 1
            records.append(
                {
                    "match_id": match_id,
                    "query_id": match_id.split("::", 1)[0],
                    "challenger": challenger,
                    "votes": votes,
                    "vote_counts": dict(counts),
                    "majority_selected_condition": majority,
                    "challenger_won": majority == challenger if majority else None,
                }
            )
        valid = wins + losses
        by_method[challenger] = {
            "total": len(match_ids),
            "majority_valid": valid,
            "majority_unresolved": unresolved,
            "challenger_majority_wins": wins,
            "pure_qwen3_majority_wins": losses,
            "challenger_majority_win_rate": wins / valid if valid else None,
            "pure_qwen3_majority_win_rate": losses / valid if valid else None,
            "unanimous_cases": unanimous,
            "two_to_one_cases": split,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "majority.records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    summary = {
        "mode": "three_local_judges_majority_vs_pure",
        "judges": {name: str(path.resolve()) for name, path in specs.items()},
        "baseline": BASELINE,
        "by_method": by_method,
        "interpretation_note": (
            "Majority win rate is the share of resolved 2-of-3 local-judge votes selecting "
            "the challenger as more human-like than pure_qwen3."
        ),
    }
    (args.output_dir / "majority.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "majority.win_rates.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "challenger", "total", "majority_valid", "majority_unresolved",
            "challenger_wins", "pure_qwen3_wins", "challenger_win_rate",
            "pure_qwen3_win_rate", "unanimous", "two_to_one",
        ])
        for challenger in CHALLENGERS:
            row = by_method[challenger]
            writer.writerow([
                challenger, row["total"], row["majority_valid"], row["majority_unresolved"],
                row["challenger_majority_wins"], row["pure_qwen3_majority_wins"],
                row["challenger_majority_win_rate"], row["pure_qwen3_majority_win_rate"],
                row["unanimous_cases"], row["two_to_one_cases"],
            ])
    print(f"Wrote local-judge majority results: {args.output_dir}")


if __name__ == "__main__":
    main()
