#!/usr/bin/env python3
"""Evaluate EmotionalRAG on Reddit using the existing 1053-post metric functions.

Five retrieval rules, no reply generation/judge. Predictions are post-only;
subreddit and eight support labels are read only for post-hoc scoring. BGE
semantic embeddings and existing Qwen-base embeddings are separate experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eight_classifier_1053_attention_eval as legacy
import summarize_eight_classifier_1053_attention as reporting
from emotional_rag_retrieval import (METHODS, PROTOCOL, UPSTREAM_COMMIT, UPSTREAM_URL,
                                    distance_matrices, rankings_from_distances, rank_scores)
from prepare_emotional_rag_reddit1053 import (DEFAULT_DATA, EMOTIONS, atomic_text, canonical,
    claim_manifest, corpus, digest, exclusive_lock, file_hash, read_cache, validate_emotion_record)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = Path("artifacts/classifiers/attention_eval_supplement1053/predicted_20260901_010901/eight_classifier_1053_attention_vectors.npz")
METHOD_DETAILS = {
    "OriginalRAG": "Rank by semantic Euclidean distance only.",
    "C-A": "Rank by semantic Euclidean distance plus emotion cosine distance; no rescaling.",
    "C-M": "Rank by the product of semantic Euclidean distance and emotion cosine distance.",
    "S-C": "Keep 20 nearest semantic neighbors, then rerank by emotion distance and return 10.",
    "S-S": "Keep 20 nearest emotion neighbors, then rerank by semantic distance and return 10.",
}


def align_ids(source_ids, target_ids):
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate post_ids in vector cache")
    if set(source_ids) != set(target_ids):
        raise ValueError("Vector cache and corpus post_id coverage differ")
    positions = {pid: index for index, pid in enumerate(source_ids)}
    return [positions[pid] for pid in target_ids]


def load_emotions(path, rows, data_file):
    manifest_path = path.parent / "emotion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("labels_used_in_prompt") is not False or manifest.get("subreddit_used_in_prompt") is not False:
        raise ValueError("Emotion manifest does not establish post-only inference")
    if manifest.get("data_sha256") != file_hash(data_file):
        raise ValueError("Emotion cache belongs to a different data file")
    if manifest.get("emotion_order") != list(EMOTIONS):
        raise ValueError("Emotion schema differs from the official eight dimensions")
    records = read_cache(path)
    if set(records) != {row.post_id for row in rows}:
        raise ValueError("Incomplete emotion cache; cannot silently drop failed posts")
    signature_hash = digest(canonical(manifest))
    values = [validate_emotion_record(records[row.post_id], row, signature_hash) for row in rows]
    return np.array(values, dtype=float), manifest


def load_semantics(path, rows):
    manifest_path = path.parent / "semantic_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("labels_used_for_embedding") is not False:
        raise ValueError("Semantic cache lacks no-label provenance")
    with np.load(path, allow_pickle=False) as cache:
        order = align_ids(cache["post_ids"].astype(str).tolist(), [r.post_id for r in rows])
        if cache["text_sha256"].astype(str)[order].tolist() != [digest(r.text) for r in rows]:
            raise ValueError("Semantic cache text hashes differ from current posts")
        if str(cache["signature_sha256"][0]) != digest(canonical(manifest)):
            raise ValueError("Semantic cache signature mismatch")
        values = cache["semantic_vectors"][order].astype(np.float64)
    return values, manifest


def new_definitions():
    return {"EmotionalRAG_" + name: {"family": "emotional_rag_retrieval_adaptation",
                "core": description,
                "score": "Ordinal -rank score represents the exact returned top-10; not a distance or calibrated similarity."}
            for name, description in METHOD_DETAILS.items()}


def metric_tables(scores, examples, labels, top_ks):
    """Reuse old metric implementation, restoring the imported definitions after use."""
    previous = legacy.METHOD_DEFINITIONS.copy()
    legacy.METHOD_DEFINITIONS.update(new_definitions())
    summaries, ranks, queries = [], [], []
    try:
        for method, matrix in scores.items():
            summary, by_rank, rows = legacy.evaluate_method(np, method, matrix, examples, labels,
                                                           top_ks, 10, max(top_ks))
            summaries.append(summary)
            ranks.extend(by_rank)
            queries.extend(rows)
    finally:
        legacy.METHOD_DEFINITIONS.clear()
        legacy.METHOD_DEFINITIONS.update(previous)
    return summaries, ranks, queries


def check_paths(args):
    out = args.output_dir.resolve()
    for source in (args.data_file, args.attention_cache, args.emotions, args.semantic_cache):
        if source is None:
            continue
        source = source.resolve()
        if out == source.parent or out in source.parents or source.parent in out.parents:
            raise ValueError(f"Output must be separate from every source directory: {out} / {source}")


def evaluate(args):
    check_paths(args)
    watched = [args.data_file, args.emotions, args.emotions.parent / "emotion_manifest.json"]
    if args.semantic_source == "bge" and args.semantic_cache is not None:
        watched += [args.semantic_cache, args.semantic_cache.parent / "semantic_manifest.json"]
    if args.semantic_source == "qwen-base-cache" or not args.skip_attention_baselines:
        watched.append(args.attention_cache)
    before = {str(path): file_hash(path) for path in watched}
    rows = corpus(args.data_file, args.expected_count)
    label_keys = legacy.parse_label_keys("all")
    post_ids = [row.post_id for row in rows]
    arrays = None
    legacy_info = None
    if args.semantic_source == "qwen-base-cache" or not args.skip_attention_baselines:
        # This loader validates no-gold predicted-mode metadata and ID coverage.
        arrays, legacy_info = legacy.merge_shards_for_examples(np, [args.attention_cache], rows, label_keys)
        for key, value in arrays.items():
            if len(value) != len(rows) or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite or incomplete legacy vectors: {key}")
    if args.semantic_source == "bge":
        if args.semantic_cache is None:
            raise ValueError("--semantic-cache is required for --semantic-source bge")
        semantic, semantic_info = load_semantics(args.semantic_cache, rows)
    else:
        semantic = arrays["base_hidden_post_mean"]
        semantic_info = {"encoder": "existing Qwen base post-only hidden mean", "cache": str(args.attention_cache),
                         "normalization": "preserve stored values; no additional normalization",
                         "adaptation": "Different semantic encoder from upstream BGE; controlled Qwen-feature variant",
                         "legacy_cache_text_hashes_available": False,
                         "coverage_check": "exact post_id set, aligned by post_id; old cache has no per-post text hashes"}
    emotion, emotion_info = load_emotions(args.emotions, rows, args.data_file)
    sd, ed = distance_matrices(semantic, emotion)
    rankings = rankings_from_distances(sd, ed, post_ids, pool_size=20, output_k=10)
    scores = rank_scores(rankings, len(rows))
    if not args.skip_attention_baselines:
        scores = {**legacy.method_scores(np, arrays), **scores}

    # Only now load evaluation labels. They cannot affect features or rankings.
    examples, label_info = legacy.load_examples(args.data_file, None, label_keys, 0, 0)
    if label_info["skipped_count"] or [e.post_id for e in examples] != post_ids:
        raise ValueError("Evaluation labels missing/misaligned; no query dropping is permitted")
    if [e.text for e in examples] != [e.text for e in rows]:
        raise ValueError("Labeled loader and inference loader returned different post text")
    labels = legacy.label_matrix(np, examples, label_keys)
    # Ensure the metric adapter cannot accidentally undo a two-stage ranking.
    for method, expected in rankings.items():
        actual = legacy.ranked_indices_excluding_same_post_id(np, scores["EmotionalRAG_" + method], post_ids, 10)
        if not np.array_equal(np.stack(actual), expected):
            raise AssertionError(f"Rank-to-metric mismatch: {method}")
    summaries, by_rank, queries = metric_tables(scores, examples, labels, [1, 3, 5])
    if before != {str(path): file_hash(path) for path in watched}:
        raise ValueError("Inputs changed during evaluation; no results written")
    signature = {"protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT, "upstream_source": UPSTREAM_URL,
        "input_sha256": before,
        "data_file": str(args.data_file.resolve()), "data_sha256": file_hash(args.data_file),
        "emotion_cache": str(args.emotions.resolve()), "emotion_sha256": file_hash(args.emotions),
        "semantic_source": args.semantic_source, "semantic_info": semantic_info,
        "semantic_sha256": file_hash(args.semantic_cache) if args.semantic_source == "bge" else None,
        "attention_cache_sha256": file_hash(args.attention_cache) if arrays is not None else None,
        "include_attention_baselines": not args.skip_attention_baselines,
        "post_ids": post_ids, "query_count": len(rows), "top_ks": [1, 3, 5],
        "stage1_pool": 20, "retrieval_k": 10, "tie_breaking": "stable input corpus order",
        "self_exclusion": "all same-post-ID candidates excluded before first-stage shortlist",
        "subreddit_filter_used": False, "labels_used_in_retrieval": False,
        "code_sha256": {name: file_hash(SCRIPT_DIR / name) for name in (
            Path(__file__).name, "emotional_rag_retrieval.py", "prepare_emotional_rag_reddit1053.py",
            "eight_classifier_1053_attention_eval.py", "summarize_eight_classifier_1053_attention.py")}}
    if args.check_only:
        print(json.dumps({"status": "CHECK_OK_NO_WRITES", "query_count": len(rows), "methods": list(scores),
                          "semantic_source": args.semantic_source, "labels_used_in_retrieval": False}, indent=2))
        return
    out = args.output_dir
    with exclusive_lock(out / "evaluation.lock"):
        claim_manifest(out / "evaluation_manifest.json", signature)
        layer = out / "last1"
        layer.mkdir(exist_ok=True)
        prefix = "emotional_rag_reddit1053"
        definitions = {**legacy.METHOD_DEFINITIONS, **new_definitions()}
        # Use the same names expected by the historical report renderer.
        fields = list(summaries[0])
        legacy.write_csv(layer / f"{prefix}_method_summary.csv", summaries, fields)
        legacy.write_csv(layer / f"{prefix}_by_rank.csv", by_rank)
        legacy.write_jsonl(layer / f"{prefix}_query_results.jsonl", queries)
        vector_rows = legacy.vector_diagnostics(np, arrays) if arrays is not None else []
        for name, value in (("EmotionalRAG_semantic", semantic), ("EmotionalRAG_emotion", emotion)):
            norms = np.linalg.norm(value, axis=1)
            vector_rows.append({"method": name, "shape": "x".join(map(str, value.shape)),
                "finite_value_rate": float(np.isfinite(value).mean()), "nan_value_count": int(np.isnan(value).sum()),
                "inf_value_count": int(np.isinf(value).sum()), "l2_norm_mean": float(norms.mean()),
                "l2_norm_std": float(norms.std()), "zero_vector_count": int((norms <= 1e-12).sum())})
        legacy.write_csv(layer / f"{prefix}_vector_diagnostics.csv", vector_rows)
        diagnostics = legacy.score_diagnostics(np, scores, post_ids, 5)
        for row in diagnostics:
            row["score_kind"] = "ordinal_rank" if row["method"].startswith("EmotionalRAG_") else "legacy_cosine"
        legacy.write_csv(layer / f"{prefix}_score_diagnostics.csv", diagnostics)
        pair_checks = legacy.pairwise_method_checks(np, scores, post_ids, 5)
        for row in pair_checks:
            if any(row[k].startswith("EmotionalRAG_") for k in ("left_method", "right_method")):
                # Ordinal ranks and cosine similarities do not share score units.
                for key in ("score_mean_abs_diff", "score_max_abs_diff", "scores_exactly_equal"):
                    row[key] = "not_comparable_score_units"
        legacy.write_csv(layer / f"{prefix}_method_pair_checks.csv", pair_checks)
        legacy.write_json(layer / f"{prefix}_method_definitions.json", {name: definitions[name] for name in scores})
        legacy.write_json(layer / f"{prefix}_summary.json", {"query_count": len(rows), "methods": summaries,
            "label_keys": label_keys, "top_ks": [1, 3, 5], "semantic_info": semantic_info, "emotion_info": emotion_info,
            "legacy_cache_info": legacy_info, "label_load_info": label_info,
            "shard_info": {"attention_layers": "inherited only for old baselines", "attention_answer_mode": "post-only emotion prediction"}})
        report = ("# EmotionalRAG Reddit1053 Retrieval Adaptation Report\n\n"
                  f"semantic_source: {args.semantic_source}\nupstream_commit: {UPSTREAM_COMMIT}\n"
                  "Evaluation: subreddit hit@K, support-label agreement/Jaccard; not reply quality.\n"
                  "Emotion axes are independent 1-10 predictions, NOT the eight evaluation labels.\n"
                  "Official distance fusion and 20->10 rules; leave-one-post-ID-out and stable ties.\n"
                  "Model/domain/annotation prompt are adapted; this is not the original role-play experiment.\n"
                  "EmotionalRAG score diagnostics use ordinal ranks, not the raw distance scale.\n\n"
                  + reporting.summarize_layer(layer))
        atomic_text(out / "fresh_results_report.txt", report)
    print(f"COMPLETE queries={len(rows)} methods={len(scores)} report={out / 'fresh_results_report.txt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--expected-count", type=int, default=1053)
    p.add_argument("--emotions", type=Path, required=True)
    p.add_argument("--semantic-source", required=True, choices=("bge", "qwen-base-cache"))
    p.add_argument("--semantic-cache", type=Path)
    p.add_argument("--attention-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--skip-attention-baselines", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.expected_count < 11:
        p.error("Need at least 11 posts for leave-one-out top-10 retrieval")
    evaluate(args)


if __name__ == "__main__":
    main()
