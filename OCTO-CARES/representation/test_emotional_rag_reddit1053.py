"""Offline tests: official retrieval rules, no-label inputs, old metric parity."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import emotional_rag_retrieval as retrieval
import eight_classifier_1053_attention_eval as legacy
import evaluate_emotional_rag_reddit1053 as evaluate
import prepare_emotional_rag_reddit1053 as prepare


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.semantic = rng.normal(size=(35, 12))
        self.emotion = rng.uniform(1, 10, size=(35, 8))
        self.ids = [f"p{i}" for i in range(35)]

    def test_all_five_rules_match_literal_upstream_calculations(self):
        sd, ed = retrieval.distance_matrices(self.semantic, self.emotion)
        actual = retrieval.rankings_from_distances(sd, ed, self.ids)
        for i in range(len(self.ids)):
            eligible = np.array([j for j in range(len(self.ids)) if j != i])
            s = np.array([np.linalg.norm(self.semantic[i] - v) for v in self.semantic])
            e = np.array([1 - np.dot(self.emotion[i], v) / (np.linalg.norm(self.emotion[i]) * np.linalg.norm(v)) for v in self.emotion])
            expected = {
                "OriginalRAG": eligible[np.argsort(s[eligible])[:10]],
                "C-A": eligible[np.argsort((s + e)[eligible])[:10]],
                "C-M": eligible[np.argsort((s * e)[eligible])[:10]],
            }
            first_s = eligible[np.argsort(s[eligible])[:20]]
            first_e = eligible[np.argsort(e[eligible])[:20]]
            expected["S-C"] = first_s[np.argsort(e[first_s])[:10]]
            expected["S-S"] = first_e[np.argsort(s[first_e])[:10]]
            for method in retrieval.METHODS:
                np.testing.assert_array_equal(actual[method][i], expected[method])

    def test_raw_semantic_scale_is_not_silently_normalized(self):
        sd, ed = retrieval.distance_matrices([[0, 0], [3, 4]], [[1, 1], [1, 2]])
        self.assertEqual(sd[0, 1], 5)
        self.assertAlmostEqual(ed[0, 1], 1 - 3 / (np.sqrt(2) * np.sqrt(5)))

    def test_semantic_shortlist_is_not_global_emotion_sort(self):
        sd = np.tile(np.arange(35), (35, 1)).astype(float)
        ed = sd[:, ::-1].copy()
        ranks = retrieval.rankings_from_distances(sd, ed, self.ids)
        self.assertEqual(ranks["S-C"][0, 0], 20)
        self.assertNotIn(34, ranks["S-C"][0])
        self.assertEqual(ranks["S-S"][0, 0], 15)

    def test_same_id_exclusion_and_stable_ties_precede_shortlisting(self):
        ids = ["same", "same"] + self.ids[2:]
        ranks = retrieval.rankings_from_distances(np.zeros((35, 35)), np.zeros((35, 35)), ids)
        for values in ranks.values():
            np.testing.assert_array_equal(values[0], np.arange(2, 12))
            self.assertNotIn(0, values[1])
            self.assertNotIn(1, values[1])

    def test_rank_score_adapter_preserves_two_stage_order(self):
        ranks = retrieval.rankings_from_distances(*retrieval.distance_matrices(self.semantic, self.emotion), self.ids)
        for method, score in retrieval.rank_scores(ranks, 35).items():
            actual = legacy.ranked_indices_excluding_same_post_id(np, score, self.ids, 10)
            np.testing.assert_array_equal(actual, ranks[method.removeprefix("EmotionalRAG_")])

    def test_nonfinite_zero_and_coverage_errors_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Zero emotion"):
            retrieval.distance_matrices(self.semantic, np.zeros((35, 8)))
        bad = self.semantic.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            retrieval.distance_matrices(bad, self.emotion)
        with self.assertRaises(ValueError):
            retrieval.rankings_from_distances(np.zeros((2, 2)), np.zeros((2, 2)), ["a", "b"])


class PreparationTests(unittest.TestCase):
    def test_official_acceptance_axis_and_one_to_ten_scale(self):
        self.assertIn("acceptance", prepare.EMOTIONS)
        self.assertNotIn("trust", prepare.EMOTIONS)
        values = {k: float(i + 1) for i, k in enumerate(prepare.EMOTIONS)}
        self.assertEqual(prepare.parse_emotion(json.dumps(values)), list(values.values()))
        for invalid in (0, 11, True, "1", float("nan")):
            bad = dict(values, joy=invalid)
            with self.assertRaises(ValueError):
                prepare.parse_emotion(json.dumps(bad))
        with self.assertRaises(ValueError):
            prepare.parse_emotion('{"joy":1,"joy":2}')
        with self.assertRaises(ValueError):
            prepare.parse_emotion('```json\n' + json.dumps(values) + '\n```')

    def test_input_prompt_ignores_support_labels_and_subreddit(self):
        a = legacy.PostExample("a", "secret_subreddit", "title", "body", "same text", {"gold_secret": 1})
        b = legacy.PostExample("a", "another_subreddit", "title", "body", "same text", {})
        self.assertEqual(prepare.messages(a), prepare.messages(b))
        text = prepare.canonical(prepare.messages(a))
        self.assertNotIn("gold_secret", text)
        self.assertNotIn("secret_subreddit", text)

    def test_unknown_output_and_changed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            marker = out / "manifest.json"
            prepare.claim_manifest(marker, {"a": 1})
            prepare.claim_manifest(marker, {"a": 1})
            with self.assertRaises(ValueError):
                prepare.claim_manifest(marker, {"a": 2})
            other = out / "other"
            other.mkdir()
            (other / "do_not_overwrite.txt").write_text("user data")
            with self.assertRaises(ValueError):
                prepare.claim_manifest(other / "manifest.json", {})


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "inputs"
        self.source.mkdir()
        self.data = self.source / "posts.json"
        self.n = 32
        records = [{"post_id": f"id{i}", "status": "complete", "subreddit": f"sub{i % 4}",
                    "title": f"Post {i}", "body": f"Text about feelings {i}.",
                    "labels": {key: (i + j) % 2 for j, key in enumerate(legacy.LABEL_KEYS)}} for i in range(self.n)]
        self.data.write_text(json.dumps(records))
        self.rows = prepare.corpus(self.data, self.n)
        rng = np.random.default_rng(7)
        self.arrays = {key: rng.normal(size=(self.n, 6) if key == "base_hidden_post_mean" else (self.n, 8, 6)).astype(np.float32)
                       for key in legacy.VECTOR_KEYS}
        self.cache = self.source / "legacy.npz"
        np.savez(self.cache, post_ids=np.array([r.post_id for r in self.rows]),
                 subreddits=np.array([r.subreddit for r in self.rows]), label_keys=np.array(legacy.LABEL_KEYS),
                 attention_answer_mode=np.array(["predicted"]), gold_labels_used_for_prompt=np.array([0]),
                 **{key + "_vectors": array for key, array in self.arrays.items()})
        self.emotions = self.source / "emotion"
        self.emotions.mkdir()
        self.manifest = {"labels_used_in_prompt": False, "subreddit_used_in_prompt": False,
                         "data_sha256": prepare.file_hash(self.data), "emotion_order": list(prepare.EMOTIONS)}
        (self.emotions / "emotion_manifest.json").write_text(json.dumps(self.manifest))
        self.emotion_records = [{"post_id": r.post_id, "text_sha256": prepare.digest(r.text),
            "prompt_sha256": prepare.digest(prepare.canonical(prepare.messages(r))),
            "signature_sha256": prepare.digest(prepare.canonical(self.manifest)),
            "emotion_protocol": prepare.EMOTION_PROTOCOL, "emotion_order": list(prepare.EMOTIONS),
            "labels_used_in_prompt": False, "input_truncated": False, "error": None,
            "emotion_embedding": rng.uniform(1, 10, 8).tolist()} for r in self.rows]
        self.emotion_file = self.emotions / "emotions.jsonl"
        self.write_emotions(self.emotion_records)
        self.args = argparse.Namespace(data_file=self.data, expected_count=self.n, attention_cache=self.cache,
                semantic_source="qwen-base-cache", semantic_cache=None, emotions=self.emotion_file,
                skip_attention_baselines=False, output_dir=root / "results", check_only=False)

    def write_emotions(self, records):
        self.emotion_file.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_full_report_contains_eleven_methods_and_old_metrics_are_identical(self):
        original = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.evaluate(self.args)
        path = self.args.output_dir / "last1/emotional_rag_reddit1053_method_summary.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(int(r["total"]) == self.n for r in rows))
        examples, _ = legacy.load_examples(self.data, None, legacy.LABEL_KEYS)
        label_matrix = legacy.label_matrix(np, examples, legacy.LABEL_KEYS)
        for method, matrix in legacy.method_scores(np, self.arrays).items():
            expected, _, _ = legacy.evaluate_method(np, method, matrix, examples, label_matrix, [1, 3, 5], 10, 5)
            actual = next(r for r in rows if r["method"] == method)
            for k, v in expected.items():
                if type(v) in (int, float):
                    self.assertAlmostEqual(float(actual[k]), v)
        with (self.args.output_dir / "last1/emotional_rag_reddit1053_query_results.jsonl").open() as f:
            self.assertEqual(sum(1 for _ in f), 11 * self.n)
        report = (self.args.output_dir / "fresh_results_report.txt").read_text()
        self.assertIn("EmotionalRAG_C-A", report)
        self.assertIn("positive Jaccard", report)
        self.assertEqual(original, {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        self.assertNotIn("EmotionalRAG_C-A", legacy.METHOD_DEFINITIONS)

    def test_check_only_has_no_writes(self):
        self.args.check_only = True
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.evaluate(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_failed_missing_duplicate_or_stale_emotions_are_not_scored(self):
        variants = [self.emotion_records[:-1], self.emotion_records + [self.emotion_records[0]]]
        for fields in ({"error": "bad format"}, {"text_sha256": "wrong"}, {"input_truncated": True},
                       {"labels_used_in_prompt": True}, {"emotion_embedding": [0] * 8}):
            records = [dict(row) for row in self.emotion_records]
            records[0].update(fields)
            variants.append(records)
        for records in variants:
            self.write_emotions(records)
            with self.assertRaises(ValueError):
                evaluate.load_emotions(self.emotion_file, self.rows, self.data)

    def test_output_cannot_overwrite_original_cache_directory(self):
        self.args.output_dir = self.source / "new"
        with self.assertRaises(ValueError):
            evaluate.check_paths(self.args)

    def test_semantic_cache_alignment_and_text_provenance(self):
        out = self.source / "bge"
        out.mkdir()
        manifest = {"labels_used_for_embedding": False, "model_path": "synthetic_test_only"}
        (out / "semantic_manifest.json").write_text(json.dumps(manifest))
        path = out / "semantic_vectors.npz"
        order = list(reversed(range(self.n)))
        np.savez(path, post_ids=np.array([self.rows[i].post_id for i in order]),
                 text_sha256=np.array([prepare.digest(self.rows[i].text) for i in order]),
                 signature_sha256=np.array([prepare.digest(prepare.canonical(manifest))]),
                 semantic_vectors=self.arrays["base_hidden_post_mean"][order])
        actual, _ = evaluate.load_semantics(path, self.rows)
        np.testing.assert_array_equal(actual, self.arrays["base_hidden_post_mean"])
        self.args.semantic_source = "bge"
        self.args.semantic_cache = path
        self.args.skip_attention_baselines = True
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.evaluate(self.args)
        summary = json.loads((self.args.output_dir / "last1/emotional_rag_reddit1053_summary.json").read_text())
        self.assertEqual(len(summary["methods"]), 5)

    def test_semantic_duplicate_and_missing_ids_fail(self):
        with self.assertRaises(ValueError):
            evaluate.align_ids(["a", "a"], ["a", "b"])
        with self.assertRaises(ValueError):
            evaluate.align_ids(["a"], ["a", "b"])

    def launcher_env(self):
        return dict(os.environ, PYTHON_BIN=sys.executable, DATA_FILE=str(self.data),
            ATTENTION_CACHE=str(self.cache), EXPECTED_COUNT=str(self.n),
            SEMANTIC_SOURCE="qwen-base-cache", OUTPUT_ROOT=str(self.args.output_dir),
            DRY_RUN="1", PRINT_PROMPT="0", STAGE="all", GPU_LIST="",
            MAX_INPUT_TOKENS="32768", SEMANTIC_MAX_LENGTH="512")

    def test_launcher_dry_run_without_gpu_or_model_has_no_writes(self):
        before = {p: p.read_bytes() for p in Path(self.temp.name).rglob("*") if p.is_file()}
        launcher = Path(prepare.__file__).with_name("run_emotional_rag_reddit1053.sh")
        result = subprocess.run(["bash", str(launcher)], env=self.launcher_env(), text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PREFLIGHT_OK", result.stdout)
        self.assertIn("GPU/model/API not started", result.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in Path(self.temp.name).rglob("*") if p.is_file()})

    def test_launcher_three_gpu_dry_run_sets_three_workers_without_writes(self):
        before = {p: p.read_bytes() for p in Path(self.temp.name).rglob("*") if p.is_file()}
        launcher = Path(prepare.__file__).with_name("run_emotional_rag_reddit1053.sh")
        env = self.launcher_env()
        env["GPU_LIST"] = "5,6,7"
        result = subprocess.run(["bash", str(launcher)], env=env, text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"worker_count": 3', result.stdout)
        self.assertIn("DRY_RUN", result.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in Path(self.temp.name).rglob("*") if p.is_file()})

    def test_launcher_rejects_original_or_unknown_output_root(self):
        launcher = Path(prepare.__file__).with_name("run_emotional_rag_reddit1053.sh")
        env = self.launcher_env()
        env["OUTPUT_ROOT"] = str(self.source)
        result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.args.output_dir.mkdir()
        (self.args.output_dir / "old.txt").write_text("preserve")
        env["OUTPUT_ROOT"] = str(self.args.output_dir)
        result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.args.output_dir / "old.txt").read_text(), "preserve")

    def test_no_gold_cache_is_required_even_for_control_semantics(self):
        with np.load(self.cache, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        arrays["gold_labels_used_for_prompt"] = np.array([1])
        np.savez(self.cache, **arrays)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "no-gold"):
            evaluate.evaluate(self.args)

    def test_emotion_merge_requires_complete_worker_coverage(self):
        for worker_count in (3, 4):
            with self.subTest(worker_count=worker_count):
                args = argparse.Namespace(output_dir=self.emotions, worker_count=worker_count)
                # Matching deterministic synthetic signature; no model loading.
                for worker in range(worker_count):
                    records = self.emotion_records[worker::worker_count]
                    (self.emotions / f"emotions.worker{worker}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in records))
                with patch.object(prepare, "emotion_signature", return_value=self.manifest):
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepare.merge(args, self.rows)
                    merged = prepare.read_cache(self.emotion_file)
                    self.assertEqual(list(merged), [r.post_id for r in self.rows])
                    partial = self.emotions / "emotions.worker2.jsonl"
                    partial.write_text("")
                    with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                        prepare.merge(args, self.rows)

    def test_longer_report_metrics_preserve_empty_positive_jaccard_convention(self):
        result = legacy.positive_jaccard(np, np.zeros(8), np.zeros((1, 8)))
        self.assertEqual(result[0], 1)


if __name__ == "__main__":
    unittest.main()
