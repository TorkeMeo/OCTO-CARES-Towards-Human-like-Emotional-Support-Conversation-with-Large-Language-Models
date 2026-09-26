from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import audit_release
import run_stage
import runtime


class PortablePipelineTests(unittest.TestCase):
    def test_gpu_validation_and_owned_server_command(self):
        self.assertEqual(runtime.gpu_ids([0, 1]), [0, 1])
        for invalid in ([0, 0], [True], [], [-1]):
            with self.assertRaises(ValueError):
                runtime.gpu_ids(invalid)
        command = runtime.server_command(Path("/tmp/model"), "judge", 8101, [0, 1],
                                         {"reasoning_parser": "openai_gptoss"}, "owned-tag")
        self.assertEqual(command[:4], ["docker", "run", "--detach", "--rm"])
        self.assertIn("openai_gptoss", command)
        self.assertIn("127.0.0.1:8101:8000", command)

    def test_humanlike_aggregator_uses_five_challengers(self):
        experiment_dir = run_stage.ROOT / "experiments"
        sys.path.insert(0, str(experiment_dir))
        try:
            import aggregate_local_humanlikeness_vs_pure as aggregate
            import judge_vllm_humanlikeness_vs_pure as judge
            self.assertEqual(len(aggregate.CHALLENGERS), 5)
            self.assertEqual(judge.PROMPT_VERSION, "vllm-ab-liveliness-v2-single-user-no-length-heuristic")
        finally:
            sys.path.remove(str(experiment_dir))

    def test_plan_is_read_only_and_mentions_each_stage(self):
        command = [sys.executable, "-B", str(run_stage.ROOT / "scripts/run_stage.py"),
                   "--stage", "escot-all"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for stage in ("summary", "escot-retrieve", "generate", "evaluate"):
            self.assertIn("STAGE " + stage, result.stdout)
        self.assertIn("aggregate_local_humanlikeness_vs_pure.py", result.stdout)
        self.assertIn("judge_vllm_sixcriteria_vs_pure.py", result.stdout)
        self.assertIn("aggregate_three_judge_sixcriteria.py", result.stdout)

    def test_release_audit_rejects_runtime_and_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("Synthetic package")
            (root / "THIRD_PARTY.md").write_text("Synthetic notice")
            inventory, problems = audit_release.inspect(root)
            self.assertEqual(len(inventory), 2)
            self.assertEqual(problems, [])
            (root / "data/private").mkdir(parents=True)
            (root / "data/private/posts.json").write_text("[]")
            self.assertTrue(audit_release.inspect(root)[1])


if __name__ == "__main__":
    unittest.main()
