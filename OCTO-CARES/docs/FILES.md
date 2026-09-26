# File Roles

All paths below are relative to `OCTO-CARES`. Files labeled "support" are
imported or invoked by an entry point; they are not separate experiments.

## Entry Points

| File | Purpose |
| --- | --- |
| `scripts/run_stage.py` | Plan/run Qwen annotations, eight-adapter training, 1053 vectors, and the ESCoT summary-to-humanlikeness pipeline. |
| `representation/run_emotional_rag_reddit1053.sh` | Recreate the Qwen-base-cache Reddit1053 retrieval report. |
| `scripts/check_package.py` | CPU syntax and regression checks. |
| `scripts/audit_release.py` | Detect private/runtime files before anonymous upload. |

## Annotation and Training

| File | Purpose |
| --- | --- |
| `annotation/post_labels.py` | Canonical eight label definitions and decision rule. |
| `annotation/label_posts_bailian.py` | Qwen3.7 single-model annotation cache for the original posts. |
| `annotation/join_qwen_labels.py` | Strictly join Qwen cache, source posts, and supplied split IDs into full training records. |
| `annotation/label_supplement_three_models_bailian.py` | Three independent API annotations and per-label majority votes. |
| `annotation/export_supplement_majority_eval_schema.py` | Export complete 1053-style labeled records for vectors and retrieval metrics. |
| `representation/train_stable_binary_sft.py` | Train one numerically checked Qwen3-8B LoRA binary adapter; repeat for eight labels. |
| `representation/train_8_type_binary_sft_v2.py` | Preserve the distinct historical adapter-training settings. |
| `representation/eight_classifier_1053_attention_eval.py` | Extract/merge base and attention-weighted vectors; supply original retrieval metric functions. |

## Emotional RAG Report

| File | Purpose |
| --- | --- |
| `representation/prepare_emotional_rag_reddit1053.py` | Infer and cache eight post-only emotion intensities. |
| `representation/emotional_rag_retrieval.py` | Five retrieval ranking rules and distance calculations. |
| `representation/evaluate_emotional_rag_reddit1053.py` | Score retrieved posts and write `fresh_results_report.txt`. |
| `representation/summarize_eight_classifier_1053_attention.py` | Shared formatting for retrieval metrics. |

## ESCoT Generation and Evaluation

| File | Purpose |
| --- | --- |
| `experiments/summarize_escot_local_seeker_only.py`, `summarize_escot_bailian_rolecard.py`, `merge_escot_seeker_only_summary_shards.py` | Build and merge gpt-oss-120b Seeker summaries. |
| `experiments/retrieve_escot_attention.py`, `prepare_escot_seeker_only_reddit_mirror_v2.py` | Rank Reddit memories and prepare seeker-background/latest-turn input. |
| `experiments/run_escot_seeker_only_reddit_four_methods_v2.py`, `generate_escot_seeker_only_reddit_four_methods_v2.py`, `run_lively_topk_generation.py`, `generate_support_replies_lively_topk.py` | Four baseline/post/comment Qwen generation conditions and worker merge. |
| `experiments/extract_escot_seeker_only_persona_local_v2.py`, `extract_simple_seeker_persona_local.py`, `extract_simple_seeker_persona.py`, `generate_escot_seeker_only_persona_replies_v2.py`, `generate_simple_persona_replies.py` | Local persona extraction and response condition. |
| `experiments/run_escot_seeker_only_reddit_multiagent_v2.py`, `generate_escot_seeker_only_reddit_multiagent_v2.py`, `run_multiagentesc_official_no_rag_qwen3.py`, `generate_multiagentesc_official_no_rag_qwen3_refiner_guard.py`, `generate_multiagentesc_official_no_rag_qwen3.py` | No-retrieval MultiAgentESC adaptation. |
| `experiments/validate_escot_seeker_only_six_method_v2.py` | Check six final response files and aligned query coverage. |
| `experiments/judge_vllm_humanlikeness_vs_pure.py` | Make one A/B humanlikeness judgment per method/query using a local vLLM API. |
| `experiments/judge_local_humanlikeness_vs_pure.py`, `judge_humanlikeness_qwen37.py`, `judge_pairwise_humanlikeness.py`, `local_judge_native_adapters.py` | Retained prompt, input-alignment, parser and imported support code for the judge/aggregator. |
| `experiments/aggregate_local_humanlikeness_vs_pure.py` | Three-judge majority table and JSONL vote audit. |
| `experiments/judge_vllm_sixcriteria_vs_pure.py` | Make one independent A/B judgment per pair and criterion for Identification, Suggestion, Diversity, Informativeness, Coherence, and Stability. Comforting is excluded. |
| `experiments/aggregate_three_judge_sixcriteria.py` | Validate three six-criterion judge caches and write majority records, summary, and win-rate tables. |
| `experiments/test_judge_vllm_sixcriteria_vs_pure.py`, `test_aggregate_three_judge_sixcriteria.py` | CPU checks that the six-criterion contract excludes Comforting and produces six tasks per pair. |

`scripts/runtime.py` manages only subprocesses/containers started by the
package. `configs/example.json`, `.env.example`, `requirements.txt`, and
`data/README.md` specify the environment and input contracts. `test_*.py`
files are synthetic CPU regression tests. `THIRD_PARTY.md` records pinned
upstream attribution and unresolved redistribution checks.
