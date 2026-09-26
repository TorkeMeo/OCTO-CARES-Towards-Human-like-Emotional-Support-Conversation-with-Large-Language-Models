"""CPU checks for six-criterion majority aggregation."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import aggregate_three_judge_sixcriteria as aggregate


class SixCriteriaAggregateTests(unittest.TestCase):
    def test_dimension_set_excludes_comforting(self):
        self.assertEqual(
            aggregate.DIMENSIONS,
            ("Identification", "Suggestion", "Diversity", "Informativeness", "Coherence", "Stability"),
        )
        self.assertNotIn("Comforting", aggregate.DIMENSIONS)
        self.assertIn("no-comforting", aggregate.PROMPT_VERSION)

    def test_expected_match_ids_are_six_per_pair(self):
        ids = {
            f"q1::{method}_vs_{aggregate.BASELINE}::{dimension}"
            for method in aggregate.CHALLENGERS
            for dimension in aggregate.DIMENSIONS
        }
        self.assertEqual(len(ids), 5 * 6)
        self.assertTrue(all("Comforting" not in value for value in ids))


if __name__ == "__main__":
    unittest.main()
