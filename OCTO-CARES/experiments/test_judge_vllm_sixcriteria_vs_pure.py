"""CPU checks for the six-criterion pairwise judge."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import judge_vllm_sixcriteria_vs_pure as judge


class SixCriteriaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = {}
        for method in judge.CONDITIONS:
            path = self.root / f"{method}.jsonl"
            path.write_text(
                json.dumps({
                    "query_id": "q1", "source_id": 1, "condition": method,
                    "dialogue_prefix": "Seeker: context", "last_seeker": "latest",
                    "input_summary": "background", "response": f"reply {method}",
                }) + "\n", encoding="utf-8"
            )
            self.paths[method] = path

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_six_requested_criteria_are_defined(self):
        self.assertEqual(
            judge.DIMENSIONS,
            ("Identification", "Suggestion", "Diversity", "Informativeness", "Coherence", "Stability"),
        )
        self.assertNotIn("Comforting", judge.CRITERIA)
        self.assertIn("no-comforting", judge.PROMPT_VERSION)

    def test_task_count_is_six_per_pair_and_prompt_is_single_dimension(self):
        values = []
        for name, path in self.paths.items():
            values.extend(["--generation", f"{name}={path}"])
        args = judge.parse_args([
            *values, "--output-dir", str(self.root / "out"),
            "--base-url", "http://localhost:8000/v1", "--model", "m", "--judge-name", "j",
        ])
        tasks = judge.build_tasks(judge.read_items(self.paths, 0), args)
        self.assertEqual(len(tasks), 5 * 6)
        self.assertEqual({task["dimension"] for task in tasks}, set(judge.DIMENSIONS))
        self.assertTrue(all("Comforting" not in task["user_prompt"] for task in tasks))

    def test_parse_requires_one_choice_and_reason(self):
        self.assertEqual(judge.parse_judgment("Choice: A\nReason: specific"), {"choice": "A", "reason": "specific"})
        with self.assertRaises(ValueError):
            judge.parse_judgment("Choice: A\nReason: one\nChoice: B\nReason: two")


if __name__ == "__main__":
    unittest.main()
