from __future__ import annotations

import unittest

import join_qwen_labels as join
from post_labels import POST_LABEL_KEYS


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"source_post": {"id": name, "title": "A title", "selftext": "Body", "subreddit": "synthetic"}}
                       for name in ("a", "b")]
        self.labels = [{"post_id": name, "status": "complete", "model": "qwen3.7-max",
                        "labels": {key: int(name == "b") for key in POST_LABEL_KEYS}}
                       for name in ("a", "b")]
        self.split = {"test_post_ids": ["b"]}

    def test_full_join_uses_supplied_ids(self):
        rows = join.build(self.source, self.labels, self.split, 2, "qwen3.7-max")
        self.assertEqual([row["fixed_split"] for row in rows], ["train", "test"])
        self.assertEqual(rows[1]["labels"][POST_LABEL_KEYS[0]], 1)
        self.assertEqual(rows[1]["label_vector"], [1] * len(POST_LABEL_KEYS))

    def test_missing_or_wrong_annotation_fails(self):
        with self.assertRaises(ValueError):
            join.build(self.source, self.labels[:1], self.split, 2, "qwen3.7-max")
        self.labels[0]["labels"].pop(POST_LABEL_KEYS[0])
        with self.assertRaises(ValueError):
            join.build(self.source, self.labels, self.split, 2, "qwen3.7-max")

    def test_missing_split_and_duplicate_id_fail(self):
        with self.assertRaises(ValueError):
            join.build(self.source, self.labels, {}, 2, "qwen3.7-max")
        with self.assertRaises(ValueError):
            join.build(self.source, [*self.labels, self.labels[0]], self.split, 2, "qwen3.7-max")

    def test_manual_annotation_template_is_not_exported_or_used(self):
        self.source[0]["human_annotation"] = {
            "labels": {key: 1 for key in POST_LABEL_KEYS}, "status": "unlabeled"
        }
        rows = join.build(self.source, self.labels, self.split, 2, "qwen3.7-max")
        self.assertNotIn("human_annotation", rows[0])
        self.assertEqual(rows[0]["label_vector"], [0] * len(POST_LABEL_KEYS))


if __name__ == "__main__":
    unittest.main()
