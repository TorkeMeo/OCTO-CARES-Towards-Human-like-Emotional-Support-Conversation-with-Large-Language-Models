# Reproducing the Two Reports

Run commands from the root of `OCTO-CARES`. Inputs, checkpoints, API responses,
and prior outputs are not shipped. Use fresh output directories and record the
actual dataset/model/adapters and file hashes for every run.

## ESCoT Humanlikeness Majority

`scripts/run_stage.py --stage escot-all` plans or runs these stages in order:

1. `summarize_escot_local_seeker_only.py` asks a local gpt-oss-120b endpoint
   for first-person Seeker background summaries. Earlier Supporter turns are
   present as input context, but prompt instructions forbid copying their
   proposed advice into Seeker facts. This is prompt-level leakage control.
2. `retrieve_escot_attention.py` uses the summary query and eight
   attention-weighted label-perspective vectors to rank Reddit memories.
   The `base+attn` score is the mean of eight cosine similarities. A valid
   paired comment may come from below the requested Top-1 cutoff.
3. Six local Qwen3-8B response methods use the same seeker background and
   latest Seeker turn: `pure_qwen3`, `attention_post`,
   `attention_post_comment`, `attention_post_random_comment`,
   `simple_persona`, and `reddit_multiagent`. Only the three `attention_*`
   methods inject retrieved post/comment material. The multiagent condition
   is a no-retrieval adaptation of MultiAgentESC, not its entire upstream
   runtime. The methods have different full prompts by design.
4. `judge_vllm_humanlikeness_vs_pure.py` runs a separate A/B humanlikeness
   request for each of five challengers versus `pure_qwen3`, with balanced
   seeded order. GLM-4-32B, Gemma-3-27B, and Qwen3.5-27B judge separately.
   The retained protocol is `vllm-ab-liveliness-v2-single-user-no-length-heuristic`.
5. `aggregate_local_humanlikeness_vs_pure.py` writes
   `majority/majority.win_rates.tsv`. The launcher requires 1705 valid votes
   from **each** judge for a full 341-query run before aggregation. Majority
   means at least two of three valid judges select the same side.
6. `judge_vllm_sixcriteria_vs_pure.py` then evaluates the same five challengers
   against `pure_qwen3`, one criterion per request, for six criteria:
   Identification, Suggestion, Diversity, Informativeness, Coherence, and
   Stability. Comforting is deliberately excluded. For 341 queries this is
   10,230 independent judgments per judge (341 x 5 x 6), followed by
   `aggregate_three_judge_sixcriteria.py` in `sixcriteria/majority`.

The six-criterion cache has a distinct prompt version and output directory; it
must never be mixed with the historical seven-criterion cache.

The recorded evaluation result is the humanlikeness TSV under the user's
historical `vllm_docker_liveliness_vs_pure_full_4gpu_v2/majority` directory.
The package defaults to a new `<output_root>/humanlike/majority` path to avoid
overwriting it. To aggregate existing three compatible judgment files only:

```bash
python experiments/aggregate_local_humanlikeness_vs_pure.py \
  --judge glm4_32b=/path/to/glm4_32b/judgments.jsonl \
  --judge gemma3_27b=/path/to/gemma3_27b/judgments.jsonl \
  --judge qwen35_27b=/path/to/qwen35_27b/judgments.jsonl \
  --output-dir outputs/humanlike/majority
```

Check that each judge has all five comparisons per query, no invalid rows,
and matching response-file provenance before treating the table as complete.
The original aggregator itself can report unresolved cases; the configured
stage launcher adds the strict coverage/validity gate.

## Reddit1053 Emotional RAG Report

The reported variant sets `SEMANTIC_SOURCE=qwen-base-cache`. It uses the
plain base-mean vectors from the 1053 Qwen attention cache as semantic
features, rather than the upstream BGE encoder. A local Qwen3-8B model
estimates eight emotion intensities, one post at a time, without label or
subreddit input. The retrieval evaluator applies all five EmotionalRAG rules
(`OriginalRAG`, `C-A`, `C-M`, `S-C`, `S-S`) plus six existing Qwen vector
baselines to the same 1053 leave-one-post-out corpus. The output includes
same-subreddit rates and support-label overlap/Jaccard at Top-1/3/5. Gold
labels are used only **after retrieval** to score the results.

```bash
PYTHON_BIN=python DATA_FILE=data/private/reddit1053_labeled.json \
ATTENTION_CACHE=outputs/paper_run/vectors/reddit1053.npz \
SEMANTIC_SOURCE=qwen-base-cache MODEL_PATH=models/Qwen3-8B \
GPU_LIST=5,6,7 OUTPUT_ROOT=outputs/emotional_rag_qwen_cache \
DRY_RUN=1 PRINT_PROMPT=1 \
bash representation/run_emotional_rag_reddit1053.sh
```

After inspecting the corpus count and prompt, rerun without `DRY_RUN` and
`PRINT_PROMPT`. The launcher uses three Qwen workers, merges emotion scores,
then runs `evaluate_emotional_rag_reddit1053.py`. The report is
`<output_root>/evaluation/fresh_results_report.txt`. Valid caches resume;
changed data/model signatures require a fresh output root. `STAGE=evaluate`
recomputes CPU metrics from existing compatible caches.

This is a Reddit retrieval adaptation of EmotionalRAG, **not** a reproduction
of its role-play response generation or its original BGE setup. Keep all five
rules in raw reports, including `C-A`, even if a paper emphasizes a justified
subset. No result from the historical private output directory is copied
into this source package.

## Interpretation

The humanlikeness TSV is a three-LLM preference result, not a human study or
proof of clinical benefit. Same-subreddit and label-Jaccard metrics are
retrieval proxies, not response-quality judgments. Changed checkpoints,
sampling, prompt versions, adapters, or source posts can change both reports.
