# OCTO-CARES

Standalone code for the two reported artifacts below and the eight binary
classifiers used by their representation pipeline:

1. ESCoT seeker-only v2, three local judges, humanlikeness majority:
   `majority/majority.win_rates.tsv`.
2. ESCoT seeker-only v2, the same three local judges on six independent
   pairwise criteria (Identification, Suggestion, Diversity, Informativeness,
   Coherence, Stability), explicitly excluding Comforting:
   `sixcriteria/majority/majority.win_rates.tsv`.
3. Reddit1053, EmotionalRAG with Qwen base-cache semantics:
   `evaluation/fresh_results_report.txt`.

The package also contains the Qwen3.7 1600-post annotation and three-model
supplement annotation needed to prepare the labeled inputs. It does **not**
contain manual-label data or a manual-label evaluation. Comforting is not part
of the retained six-criterion protocol. Reddit320 generation and unrelated
exploration code are intentionally omitted.
The 80 historically held-out source posts were assigned Qwen labels; the
historical `human_gold` name does not mean human labels trained the classifiers.

See [file roles](docs/FILES.md), [data contracts](data/README.md),
[annotation and training](docs/ANNOTATION_AND_TRAINING.md), and
[experiment steps](docs/EXPERIMENTS.md). The package imports nothing from its
parent repository. It includes code, not posts, cached labels, response files,
model weights, or historical results.

## Setup

Run from this directory on Linux with Python 3.10+ and an appropriate CUDA
PyTorch installation. Install the remaining dependencies with
`python -m pip install -r requirements.txt`. Copy `configs/example.json` to
`configs/local.json` and supply authorized dataset paths, local checkpoints,
adapter paths, and GPU IDs. Relative paths resolve from this directory.
Keep credentials in environment variables, not config or Git:

```bash
export BAILIAN_API_KEY='YOUR_KEY'
export BAILIAN_BASE_URL='YOUR_OPENAI_COMPATIBLE_URL'
export SUMMARY_API_KEY=local-vllm
export JUDGE_API_KEY=local-vllm
```

`BAILIAN_API_KEY` is needed only when reannotating. Local serving with `--serve`
requires Docker, the NVIDIA container runtime, and the configured vLLM image.
No launcher kills unrelated GPU jobs. `--wait-for-gpus` polls availability but
does not reserve GPUs atomically.

## Run

Inspect the ESCoT execution plan without writing files or starting a model:

```bash
python scripts/run_stage.py --config configs/local.json --stage escot-all
```

Prepare labels and vectors with the commands in the linked docs. Once the
complete 1053-record labeled corpus and Qwen attention-vector cache exist,
the Reddit report is produced by:

```bash
PYTHON_BIN=python DATA_FILE=data/private/reddit1053_labeled.json \
ATTENTION_CACHE=outputs/paper_run/vectors/reddit1053.npz \
SEMANTIC_SOURCE=qwen-base-cache MODEL_PATH=models/Qwen3-8B \
GPU_LIST=5,6,7 OUTPUT_ROOT=outputs/emotional_rag_qwen_cache \
bash representation/run_emotional_rag_reddit1053.sh
```

After the ESCoT summary, retrieval and six response conditions are prepared,
generate and judge them using the configured stage launcher:

```bash
python scripts/run_stage.py --config configs/local.json --stage escot-all \
  --execute --serve --wait-for-gpus
```

`escot-all` is one possible fresh end-to-end run; rerunning the exact historical
numbers requires the original input files, checkpoint revisions, adapters,
prompts/settings and valid caches. Historical private artifacts are not
bundled or readable from a copied source directory. Do not describe a
different checkpoint or annotation run as a numerical reproduction of them.

## Verify and Release

```bash
python scripts/check_package.py
python scripts/audit_release.py
```

These are syntax and CPU regression checks, not evidence that GPU training,
paid annotation, vLLM serving, or full generation completed. Before an
anonymous upload, inspect the exact directory being uploaded, omit local
configs/results/data, and use a new anonymous Git history. See
[third-party sources](THIRD_PARTY.md). Upstream redistribution terms and this
project's license still require author review before public release.
