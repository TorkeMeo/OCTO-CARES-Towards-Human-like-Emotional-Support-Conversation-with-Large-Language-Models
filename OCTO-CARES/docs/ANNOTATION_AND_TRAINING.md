# Annotation and Eight Binary Classifiers

The eight independent labels and prompts are defined in
`annotation/post_labels.py`. They are multi-label binary targets, not one
eight-way softmax. Each Qwen API annotation call evaluates one post-label pair
and expects exactly `Yes` or `No`; invalid outputs remain incomplete after
retries and are never silently mapped to zero.

## Original 1600 Posts

The original 1600-post corpus was labeled with `qwen3.7-max`. Historical
construction used 1520 labeled records plus a separate 80-post source pool,
also labeled by Qwen, and marked a 1440/160 partition. No manual label values
were used as classifier targets. Historical intermediate files and exact split
IDs are not in this code package.

For a fresh run on authorized source posts:

```bash
python annotation/label_posts_bailian.py \
  --input data/private/reddit1600_source.json \
  --output outputs/annotation/reddit1600_qwen37.json \
  --model qwen3.7-max --max-retries 3 --limit 3
```

Remove `--limit 3` and use a fresh output file for the full run. The output
is a label cache. To attach complete Qwen labels to the source posts, supply
the **original explicit** `test_post_ids` in a split manifest:

```bash
python annotation/join_qwen_labels.py \
  --source data/private/reddit1600_source.json \
  --labels outputs/annotation/reddit1600_qwen37.json \
  --split-manifest data/private/reddit1600_split.json \
  --output data/private/reddit1600_labeled.json --expected-count 1600
```

The joiner refuses missing/duplicate IDs, wrong models, incomplete labels and
changed counts. It drops unused `human_annotation` templates from raw source
records. Supplying different split IDs changes the experiment; the script does
not reconstruct missing historical IDs.

## Three-Model 1053 Supplement

The recorded source had 1060 Reddit posts. `qwen3.7-max`, `kimi-k3`, and
`deepseek-v4-pro-0813` each judged every post for each of eight labels.
Per-label majority requires three valid votes by default; the recorded
complete export contained 1053 records. A new API run may have a different
incomplete set. Do not force the count by relabeling missing votes as zero.

```bash
python annotation/label_supplement_three_models_bailian.py \
  --input data/private/reddit1060_source.json \
  --output-dir outputs/annotation/supplement \
  --models qwen37_max=qwen3.7-max,kimi_k3=kimi-k3,deepseek_v4_pro_0813=deepseek-v4-pro-0813 \
  --max-retries 3
python annotation/export_supplement_majority_eval_schema.py \
  --input outputs/annotation/supplement/supplement_post_labels_majority_vote.json \
  --output data/private/reddit1053_labeled.json \
  --summary outputs/annotation/supplement/export_summary.json
```

These API annotators are **not** the GLM/Gemma/Qwen local response judges in
the humanlikeness experiment.

## Train Eight Adapters

`representation/train_stable_binary_sft.py` trains one LoRA adapter per
label. The configured stage launcher cycles through all eight labels on the
specified GPUs and saves each successful adapter under `<adapter_root>/<label>/final`:

```bash
python scripts/run_stage.py --config configs/local.json --stage train \
  --execute --wait-for-gpus
```

This stable training entry point defaults to fitting all 1600 posts and does
not provide a held-out score on the historical 160. Set `training.data_split`
to `train` and run the low-level evaluation options separately when a true
1440/160 classifier experiment is intended.

`representation/train_8_type_binary_sft_v2.py` retains the historical trainer
whose epochs, learning rate and LoRA settings differ. Existing attention
caches/adapters must be attributed to the trainer and checkpoints that actually
produced them; rerunning the stable trainer does not make an identical cache.

The vector extractor is
`representation/eight_classifier_1053_attention_eval.py`. It uses eight
adapter-predicted answers to select attention rows, pools base-model hidden
states with those weights, and writes a post vector for each perspective.
The configured `vectors` stage shards the 1053 corpus across GPUs and merges
the output:

```bash
python scripts/run_stage.py --config configs/local.json --stage vectors \
  --execute --wait-for-gpus
```

`attention-answer-mode=predicted` keeps dataset gold labels out of vector
extraction. Labels are read later only for post-hoc retrieval metrics. Check
the cache manifest, model/adapters, token limits, and post IDs before comparing
new vectors with the recorded report.
