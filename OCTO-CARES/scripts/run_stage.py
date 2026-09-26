#!/usr/bin/env python3
"""Portable stage launcher. Plans by default; --execute authorizes the listed jobs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

from runtime import ROOT, Job, gpu_ids, managed_server, run, run_lock, show, wait_resources

sys.path.insert(0, str(ROOT / "annotation"))
from post_labels import POST_LABEL_KEYS

STAGES = ("annotate1600", "annotate1053", "train", "vectors",
          "summary", "escot-retrieve", "generate", "evaluate")
GENERATION_FILES = {
    "pure_qwen3": "four_methods/top1/responses.pure_qwen3.jsonl",
    "attention_post": "four_methods/top1/responses.attention_post.jsonl",
    "attention_post_comment": "four_methods/top1/responses.attention_post_comment.jsonl",
    "attention_post_random_comment": "four_methods/top1/responses.attention_post_random_comment.jsonl",
    "simple_persona": "simple_persona/generation/responses.simple_persona.jsonl",
    "reddit_multiagent": "reddit_multiagent/responses.reddit_multiagent.jsonl",
}


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class Pipeline:
    def __init__(self, config, execute=False, serve=False, wait=False):
        self.config, self.execute, self.serve, self.wait = config, execute, serve, wait
        self.output = resolve(config["output_root"])
        if self.output == ROOT or self.output in ROOT.parents:
            raise ValueError("Output must not be the package directory or an ancestor")
        self.gpus = gpu_ids(config["gpu_ids"])
        self.data = {key: resolve(value) for key, value in config["data"].items()}
        self.model = resolve(config["qwen_model"])
        self.adapters = resolve(config["adapter_root"])
        self.limit = config.get("limit", 0)
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        self.annotation_limit = config.get("annotation_limit", 0)
        if type(self.annotation_limit) is not int or self.annotation_limit < 0:
            raise ValueError("annotation_limit must be a nonnegative integer")
        for key in ("vector_count", "escot_count"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if self.limit > config["escot_count"]:
            raise ValueError("limit exceeds escot_count")
        self.count = self.limit or config["escot_count"]
        self.summary = self.output / "summary/escot_seeker_only_summaries.jsonl"
        self.vectors = self.output / "vectors/reddit1053.npz"
        self.retrieval = self.output / "retrieval/retrieval.jsonl"
        self.mirror = self.output / "input/seeker_only_summary_latest_retrieval.jsonl"

    def job(self, group, script, *args, env=None, name=None):
        return Job([sys.executable, str(ROOT / group / script), *map(str, args)],
                   self.output / "logs" / ((name or Path(script).stem) + ".log"), env or {})

    def wave(self, *jobs):
        show(list(jobs))
        if self.execute:
            run(list(jobs))

    def need(self, *paths):
        if self.execute:
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"Required prerequisite is missing: {path}")

    def gpu_check(self):
        if self.execute:
            wait_resources(self.gpus, self.config.get("server", {}), self.wait)

    def annotate1600(self):
        self.need(self.data["reddit1600_source"])
        extra = ["--limit", str(self.annotation_limit)] if self.annotation_limit else []
        self.wave(self.job("annotation", "label_posts_bailian.py", "--input", self.data["reddit1600_source"],
                           "--output", self.output / "annotation/reddit1600_qwen37.json",
                           "--model", "qwen3.7-max", "--max-retries", 3, *extra))

    def annotate1053(self):
        self.need(self.data["supplement_source"])
        destination = self.output / "annotation/supplement"
        extra = ["--limit", str(self.annotation_limit)] if self.annotation_limit else []
        self.wave(self.job("annotation", "label_supplement_three_models_bailian.py",
                           "--input", self.data["supplement_source"], "--output-dir", destination,
                           "--max-retries", 3, *extra))
        # A smoke subset must not replace the full 1053 corpus supplied for later stages.
        self.wave(self.job("annotation", "export_supplement_majority_eval_schema.py",
                           "--input", destination / "supplement_post_labels_majority_vote.json",
                           "--output", destination / "reddit_labeled.json",
                           "--summary", destination / "export_summary.json"))

    def train(self):
        self.need(self.data["reddit1600_labeled"], self.model)
        training = self.config.get("training", {})
        if training.get("variant", "stable") != "stable":
            raise ValueError("Historical v2 training uses a different CLI; run it explicitly (see docs)")
        self.gpu_check()
        for offset in range(0, len(POST_LABEL_KEYS), len(self.gpus)):
            jobs = []
            for gpu, label in zip(self.gpus, POST_LABEL_KEYS[offset:offset + len(self.gpus)]):
                jobs.append(self.job("representation", "train_stable_binary_sft.py",
                                     "--data-file", self.data["reddit1600_labeled"], "--label-key", label,
                                     "--model-name-or-path", self.model, "--output-dir", self.adapters / label,
                                     "--data-split", training.get("data_split", "all"),
                                     "--validation-ratio", 0, "--evaluate-fixed-test", 0,
                                     env={"CUDA_VISIBLE_DEVICES": str(gpu)}, name="train_" + label))
            self.wave(*jobs)

    def vectors_stage(self):
        self.need(self.data["reddit1053_labeled"], self.model, self.adapters)
        self.gpu_check()
        extractor = "eight_classifier_1053_attention_eval.py"
        count, workers = self.config["vector_count"], len(self.gpus)
        if workers > count:
            raise ValueError("More vector workers than records")
        shards, jobs = [], []
        for worker, gpu in enumerate(self.gpus):
            start, end = count * worker // workers, count * (worker + 1) // workers
            shard = self.output / f"vectors/worker{worker}.npz"
            shards.extend(["--shard-file", str(shard)])
            jobs.append(self.job("representation", extractor, "extract",
                "--data-file", self.data["reddit1053_labeled"], "--output-file", shard,
                "--record-start", start, "--max-records", end - start,
                "--qwen-model-name-or-path", self.model, "--adapter-root", self.adapters,
                "--attention-answer-mode", "predicted", "--attention-layers", "last:1",
                "--max-qwen-length", 40960, "--qwen-prefill-chunk-size", 2048,
                env={"CUDA_VISIBLE_DEVICES": str(gpu)}, name=f"vectors_worker{worker}"))
        self.wave(*jobs)
        self.wave(self.job("representation", extractor, "merge", "--data-file",
                           self.data["reddit1053_labeled"], *shards, "--output-file", self.vectors))

    def summary_stage(self):
        self.need(self.data["escot"])
        spec = self.config["summary"]
        model_path = resolve(spec["model_path"])
        base_url = spec["base_url"]
        context = nullcontext(base_url)
        if self.serve:
            ids = gpu_ids(spec["gpu_ids"])
            print(f"MANAGED SUMMARY: {spec['model']} model={model_path} GPUs={ids}")
            if self.execute:
                settings = {**self.config["server"], "reasoning_parser": "openai_gptoss"}
                context = managed_server(model_path, spec["model"], spec["port"], ids,
                                         settings, self.output / "summary/server", self.wait)
        with context as base_url:
            workers = spec.get("client_workers", 2)
            if type(workers) is not int or workers <= 0:
                raise ValueError("summary.client_workers must be positive")
            shards, jobs = [], []
            for worker in range(workers):
                shard = self.output / f"summary/worker{worker}.jsonl"
                shards.extend(["--shard", str(shard)])
                jobs.append(self.job("experiments", "summarize_escot_local_seeker_only.py",
                    "--input", self.data["escot"], "--output", shard, "--manifest", shard.with_suffix(".manifest.json"),
                    "--model", spec["model"], "--model-path-provenance", model_path,
                    "--base-url", base_url, "--api-key-env", "SUMMARY_API_KEY",
                    "--worker-index", worker, "--worker-count", workers, "--limit", self.limit,
                    "--max-tokens", 1024, "--timeout", 600,
                    env={"SUMMARY_API_KEY": "local-vllm"} if self.serve else {}, name=f"summary_worker{worker}"))
            self.wave(*jobs)
        self.wave(self.job("experiments", "merge_escot_seeker_only_summary_shards.py",
            "--input", self.data["escot"], "--output", self.summary,
            "--manifest", self.output / "summary/summary.manifest.json", *shards,
            "--worker-count", workers, "--limit", self.limit, "--model", spec["model"], "--model-path", model_path))

    def escot_retrieve(self):
        self.need(self.summary, self.vectors, self.data["reddit1053_labeled"], self.adapters, self.model)
        self.gpu_check()
        self.wave(self.job("experiments", "retrieve_escot_attention.py",
            "--summaries", self.summary, "--corpus-file", self.data["reddit1053_labeled"],
            "--output-dir", self.output / "retrieval", "--output-file", self.retrieval,
            "--source-corpus-vectors", self.vectors, "--method", "base+attn", "--attention-layers", "last:1",
            "--top-k", 5, "--injectable-comment-required", 1,
            "--qwen-model-path", self.model, "--adapter-root", self.adapters,
            "--gpu-ids", ",".join(map(str, self.gpus))))

    def generate(self):
        if len(self.gpus) not in (3, 4):
            raise ValueError("ESCoT generation requires three or four GPU workers")
        self.need(self.retrieval, self.model)
        self.gpu_check()
        common = ["--model-path", self.model, "--gpu-ids", ",".join(map(str, self.gpus)),
                  "--worker-count", len(self.gpus), "--limit", self.limit, "--max-input-tokens", 32768]
        self.wave(self.job("experiments", "prepare_escot_seeker_only_reddit_mirror_v2.py",
                           "--input", self.retrieval, "--output", self.mirror, "--limit", self.limit))
        self.wave(self.job("experiments", "run_escot_seeker_only_reddit_four_methods_v2.py",
            "--retrieval", self.mirror, "--output-dir", self.output / "four_methods/top1", *common,
            "--max-new-tokens", 768, "--temperature", 0.7, "--top-p", 0.9,
            "--seed", 20260906, "--top-k", 1, "--context-source", "dialogue"))
        persona_dir = self.output / "simple_persona"
        personas = persona_dir / "escot_seeker_only_persona.jsonl"
        self.wave(self.job("experiments", "extract_escot_seeker_only_persona_local_v2.py",
            "--input", self.mirror, "--output-dir", persona_dir, "--output-prefix", "escot_seeker_only_persona",
            *common, "--max-new-tokens", 900, "--seed", 20260906))
        persona_args = ["--retrieval", self.mirror, "--personas", personas,
                        "--output-dir", persona_dir / "generation", "--worker-count", len(self.gpus),
                        "--limit", self.limit, "--conditions", "simple_persona"]
        self.wave(*[self.job("experiments", "generate_escot_seeker_only_persona_replies_v2.py",
            *persona_args, "--worker-index", worker, "--model-path", self.model,
            "--max-new-tokens", 768, "--temperature", 0.7, "--top-p", 0.9, "--seed", 20260906,
            env={"CUDA_VISIBLE_DEVICES": str(gpu)}, name=f"persona_worker{worker}")
            for worker, gpu in enumerate(self.gpus)])
        self.wave(self.job("experiments", "generate_escot_seeker_only_persona_replies_v2.py",
                           *persona_args, "--merge"))
        self.wave(self.job("experiments", "run_escot_seeker_only_reddit_multiagent_v2.py",
            "--input", self.mirror, "--output-dir", self.output / "reddit_multiagent", *common,
            "--decision-max-new-tokens", 100, "--analysis-max-new-tokens", 400,
            "--discussion-max-new-tokens", 400, "--response-max-new-tokens", 768,
            "--judge-max-new-tokens", 400, "--refine-max-new-tokens", 768,
            "--temperature", 0, "--seed", 42, "--strategy-agent-count", 3, "--complexity-gate", "always_multiagent"))
        self.wave(self.job("experiments", "validate_escot_seeker_only_six_method_v2.py",
                           "--root", self.output, "--expected-count", self.count))

    def evaluate(self):
        self.need(*[self.output / value for value in GENERATION_FILES.values()])
        specs = self.config["judges"]
        if len(specs) != 3 or len({spec["name"] for spec in specs}) != 3:
            raise ValueError("Exactly three unique judges are required")
        self.wave(self.job("experiments", "validate_escot_seeker_only_six_method_v2.py",
                           "--root", self.output, "--expected-count", self.count))
        generation = [arg for name, value in GENERATION_FILES.items()
                      for arg in ("--generation", f"{name}={self.output / value}")]
        folder = "humanlike"
        for spec in specs:
            context = nullcontext(spec["base_url"])
            if self.serve:
                ids = gpu_ids(self.config["server"]["judge_gpu_ids"])
                print(f"MANAGED JUDGE serial: {spec['name']} GPUs={ids}")
                if self.execute:
                    context = managed_server(resolve(spec["model_path"]), spec["name"], spec["port"], ids,
                                             self.config["server"], self.output / folder / spec["name"], self.wait)
            with context as base_url:
                self.wave(self.job("experiments", "judge_vllm_humanlikeness_vs_pure.py", *generation,
                    "--output-dir", self.output / folder / spec["name"], "--base-url", base_url,
                    "--model", spec["name"], "--judge-name", spec["name"], "--limit", self.limit,
                    "--max-tokens", 96, "--max-attempts", 2,
                    name=folder + "_" + spec["name"]))
        if self.execute:
            strict_humanlike_coverage(self.output / folder, specs, self.count)
        judges = [arg for spec in specs for arg in ("--judge", f"{spec['name']}={self.output / folder / spec['name'] / 'judgments.jsonl'}")]
        self.wave(self.job("experiments", "aggregate_local_humanlikeness_vs_pure.py",
                           *judges, "--output-dir", self.output / folder / "majority"))

        # The six criteria are a separate one-criterion-per-request protocol.
        # Comforting is intentionally absent; never reuse seven-criteria caches.
        criteria_folder = "sixcriteria"
        for spec in specs:
            context = nullcontext(spec["base_url"])
            if self.serve:
                ids = gpu_ids(self.config["server"]["judge_gpu_ids"])
                print(f"MANAGED SIX-CRITERIA JUDGE serial: {spec['name']} GPUs={ids}")
                if self.execute:
                    context = managed_server(resolve(spec["model_path"]), spec["name"], spec["port"], ids,
                                             self.config["server"], self.output / criteria_folder / spec["name"], self.wait)
            with context as base_url:
                self.wave(self.job("experiments", "judge_vllm_sixcriteria_vs_pure.py", *generation,
                    "--output-dir", self.output / criteria_folder / spec["name"], "--base-url", base_url,
                    "--model", spec["name"], "--judge-name", spec["name"], "--limit", self.limit,
                    "--max-tokens", 768, "--max-attempts", 3,
                    name=criteria_folder + "_" + spec["name"]))
        criteria_judges = [arg for spec in specs for arg in (
            "--judge", f"{spec['name']}={self.output / criteria_folder / spec['name'] / 'judgments.jsonl'}")]
        self.wave(self.job("experiments", "aggregate_three_judge_sixcriteria.py",
                           *criteria_judges, "--expected-count", self.count,
                           "--output-dir", self.output / criteria_folder / "majority"))


def strict_humanlike_coverage(folder, specs, count):
    reference = None
    for spec in specs:
        latest = {}
        with (folder / spec["name"] / "judgments.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["match_id"]] = row
        ids = set(latest)
        if len(ids) != count * 5 or (reference is not None and ids != reference):
            raise ValueError("Incomplete humanlike coverage; majority publication is blocked")
        for row in latest.values():
            if row.get("error") or row.get("selected_condition") not in ("pure_qwen3", row.get("challenger")):
                raise ValueError("Invalid humanlike vote; rerun the same stage to retry before aggregation")
        reference = ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/example.json")
    parser.add_argument("--stage", choices=(*STAGES, "escot-all"), required=True)
    parser.add_argument("--execute", action="store_true", help="Run the plan (otherwise no files/models/API calls)")
    parser.add_argument("--serve", action="store_true", help="Start own Docker servers for summary/judges; otherwise use configured endpoints")
    parser.add_argument("--wait-for-gpus", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pipeline = Pipeline(config, args.execute, args.serve, args.wait_for_gpus)
    stages = ("summary", "escot-retrieve", "generate", "evaluate") if args.stage == "escot-all" else (args.stage,)
    methods = {"vectors": pipeline.vectors_stage, "summary": pipeline.summary_stage,
               "escot-retrieve": pipeline.escot_retrieve}
    if "generate" in stages and len(pipeline.gpus) not in (3, 4):
        raise ValueError("ESCoT generation requires three or four GPU workers")
    with run_lock(pipeline.output) if args.execute else nullcontext():
        if args.execute:
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            marker = pipeline.output / "config.sha256"
            if marker.exists() and marker.read_text().strip() != fingerprint:
                raise ValueError("Configuration changed; use a fresh output_root")
            marker.write_text(fingerprint + "\n")
        for stage in stages:
            print(f"STAGE {stage}: {'EXECUTE' if args.execute else 'PLAN ONLY'}", flush=True)
            action = methods.get(stage) or getattr(pipeline, stage)
            action()
    print("COMPLETE" if args.execute else "PLAN ONLY: no files written or models/API/GPU started")


if __name__ == "__main__":
    main()
