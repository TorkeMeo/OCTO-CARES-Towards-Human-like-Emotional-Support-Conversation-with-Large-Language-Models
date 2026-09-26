from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import aggregate_local_humanlikeness_vs_pure as aggregate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_stage import strict_humanlike_coverage


class HumanlikeMajorityTests(unittest.TestCase):
    def test_three_judge_five_method_majority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("glm4_32b", "gemma3_27b", "qwen35_27b")
            for judge_index, name in enumerate(names):
                folder = root / name
                folder.mkdir()
                with (folder / "judgments.jsonl").open("w", encoding="utf-8") as handle:
                    for challenger in aggregate.CHALLENGERS:
                        winner = challenger if judge_index < 2 else aggregate.BASELINE
                        row = {
                            "match_id": f"synthetic_1::{challenger}_vs_{aggregate.BASELINE}",
                            "challenger": challenger,
                            "selected_condition": winner,
                        }
                        handle.write(json.dumps(row) + "\n")
            strict_humanlike_coverage(root, [{"name": name} for name in names], 1)
            command = [sys.executable, "-B", str(Path(aggregate.__file__)),
                       *[arg for name in names for arg in (
                           "--judge", f"{name}={root / name / 'judgments.jsonl'}")],
                       "--output-dir", str(root / "majority")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            with (root / "majority/majority.win_rates.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["challenger"] for row in rows}, set(aggregate.CHALLENGERS))
            self.assertTrue(all(row["challenger_win_rate"] == "1.0" for row in rows))

    def test_missing_vote_blocks_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "glm4_32b"
            folder.mkdir()
            (folder / "judgments.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                strict_humanlike_coverage(Path(directory), [{"name": "glm4_32b"}], 1)


if __name__ == "__main__":
    unittest.main()
