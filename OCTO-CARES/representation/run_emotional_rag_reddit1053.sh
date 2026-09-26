#!/usr/bin/env bash
# Isolated retrieval evaluation. Never starts response generation or a judge.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_FILE="${DATA_FILE:-${PACKAGE_ROOT}/data/private/reddit1053_labeled.json}"
ATTENTION_CACHE="${ATTENTION_CACHE:-${PACKAGE_ROOT}/outputs/paper_run/vectors/reddit1053.npz}"
SEMANTIC_SOURCE="${SEMANTIC_SOURCE:-bge}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${PACKAGE_ROOT}/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/emotional_rag_${SEMANTIC_SOURCE}_v1}"
GPU_LIST="${GPU_LIST:-}"
WORKER_COUNT=4
if [[ -n "$GPU_LIST" ]]; then
  [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+){2,3}$ ]] || {
    echo 'Set GPU_LIST to three or four comma-separated GPU IDs, e.g. 5,6,7.' >&2
    exit 2
  }
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  WORKER_COUNT="${#gpus[@]}"
fi
EXPECTED_COUNT="${EXPECTED_COUNT:-1053}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"
SEMANTIC_MAX_LENGTH="${SEMANTIC_MAX_LENGTH:-512}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PRINT_PROMPT="${PRINT_PROMPT:-0}"
PREP="${SCRIPT_DIR}/prepare_emotional_rag_reddit1053.py"
EVAL="${SCRIPT_DIR}/evaluate_emotional_rag_reddit1053.py"
[[ $# == 0 ]] || { echo 'Use environment settings; no positional arguments.' >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ "$SEMANTIC_SOURCE" == bge || "$SEMANTIC_SOURCE" == qwen-base-cache ]] || exit 2
[[ "$STAGE" == all || "$STAGE" == evaluate ]] || exit 2
[[ "$DRY_RUN" =~ ^[01]$ && "$PRINT_PROMPT" =~ ^[01]$ ]] || exit 2
if [[ "$STAGE" == evaluate && -z "$GPU_LIST" && -f "$OUTPUT_ROOT/run_manifest.json" ]]; then
  WORKER_COUNT="$("$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["worker_count"])' "$OUTPUT_ROOT/run_manifest.json")"
  [[ "$WORKER_COUNT" == 3 || "$WORKER_COUNT" == 4 ]] || {
    echo "Invalid worker_count in $OUTPUT_ROOT/run_manifest.json" >&2; exit 2;
  }
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SCRIPT_DIR DATA_FILE ATTENTION_CACHE SEMANTIC_SOURCE BGE_MODEL_PATH MODEL_PATH OUTPUT_ROOT GPU_LIST
export EXPECTED_COUNT MAX_INPUT_TOKENS SEMANTIC_MAX_LENGTH WORKER_COUNT

# CPU validation only: exact corpus coverage and no-gold cache provenance.
"$PYTHON_BIN" - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ['SCRIPT_DIR'])
import numpy as np
import eight_classifier_1053_attention_eval as old
from prepare_emotional_rag_reddit1053 import corpus
count = int(os.environ['EXPECTED_COUNT'])
ids = [int(value) for value in os.environ['GPU_LIST'].split(',')] if os.environ['GPU_LIST'] else []
if ids and (len(ids) != int(os.environ['WORKER_COUNT']) or len(set(ids)) != len(ids)):
    raise ValueError('GPU IDs must be distinct and match the worker count')
if count < 11 or min(int(os.environ['MAX_INPUT_TOKENS']), int(os.environ['SEMANTIC_MAX_LENGTH'])) <= 0:
    raise ValueError('Invalid count or token limit')
rows = corpus(Path(os.environ['DATA_FILE']), count)
cache = Path(os.environ['ATTENTION_CACHE']).resolve()
out = Path(os.environ['OUTPUT_ROOT']).resolve()
for source in (cache, Path(os.environ['DATA_FILE']).resolve()):
    if out == source.parent or out in source.parents or source.parent in out.parents:
        raise ValueError('Use a new output root outside every source directory')
if out.exists() and not (out / 'run_manifest.json').is_file() and any(p.suffix != '.lock' for p in out.iterdir()):
    raise ValueError('Unrecognized nonempty output root; choose a new directory')
old.merge_shards_for_examples(np, [cache], rows, old.LABEL_KEYS)
print(json.dumps({'status':'PREFLIGHT_OK', 'posts':len(rows), 'methods':11,
    'semantic_source':os.environ['SEMANTIC_SOURCE'], 'output':str(out),
    'worker_count':int(os.environ['WORKER_COUNT']),
    'emotion_dimensions':['joy','acceptance','fear','surprise','sadness','disgust','anger','anticipation'],
    'top_ks':[1,3,5], 'candidate_pool':20, 'retrieval_k':10,
    'labels_used_in_retrieval':False}, ensure_ascii=False, indent=2))
PY
common=(--data-file "$DATA_FILE" --expected-count "$EXPECTED_COUNT" --model-path "$MODEL_PATH"
        --worker-count "$WORKER_COUNT" --max-input-tokens "$MAX_INPUT_TOKENS")
if [[ "$PRINT_PROMPT" == 1 ]]; then
  "$PYTHON_BIN" "$PREP" preview "${common[@]}" --output-dir "$OUTPUT_ROOT/emotions"
fi
if [[ "$DRY_RUN" == 1 ]]; then
  echo 'DRY_RUN: no files written, GPU/model/API not started; runtime dependencies and GPU availability not checked.'
  [[ "$SEMANTIC_SOURCE" != bge || -n "$BGE_MODEL_PATH" ]] || echo 'Before the real BGE run, set BGE_MODEL_PATH to an existing local bge-base-zh-v1.5 directory.'
  exit 0
fi

if [[ "$STAGE" == all ]]; then
  [[ -d "$MODEL_PATH" ]] || { echo "Missing local emotion model: $MODEL_PATH" >&2; exit 2; }
  [[ "$SEMANTIC_SOURCE" != bge || -d "$BGE_MODEL_PATH" ]] || { echo 'Set BGE_MODEL_PATH to an existing local BGE directory.' >&2; exit 2; }
  "$PYTHON_BIN" -c 'import torch, transformers, numpy'
  [[ "$SEMANTIC_SOURCE" != bge ]] || "$PYTHON_BIN" -c 'from FlagEmbedding import FlagModel'
  [[ -n "$GPU_LIST" ]] || { echo 'Set GPU_LIST explicitly to three or four idle GPU IDs, e.g. 5,6,7.' >&2; exit 2; }
  export GPU_LIST
  "$PYTHON_BIN" - <<'PY'
import os, subprocess
ids = [int(x) for x in os.environ['GPU_LIST'].split(',')]
if len(ids) != int(os.environ['WORKER_COUNT']) or len(set(ids)) != len(ids):
    raise ValueError('GPU IDs must be distinct')
raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True)
usage = {int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in raw.splitlines() if line.strip())}
if any(i not in usage or usage[i] > 1024 for i in ids):
    raise RuntimeError(f'Selected GPUs must exist and be idle (<=1024 MiB): {usage}. No jobs were stopped.')
PY
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>>"$OUTPUT_ROOT/run.lock"
flock -n 9 || { echo 'Another run owns this output directory.' >&2; exit 2; }
"$PYTHON_BIN" - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ['SCRIPT_DIR'])
from prepare_emotional_rag_reddit1053 import claim_manifest, file_hash
root = Path(os.environ['OUTPUT_ROOT'])
signature = {'protocol':'emotional-rag-reddit1053-pipeline-v1',
    'data_file':str(Path(os.environ['DATA_FILE']).resolve()), 'data_sha256':file_hash(os.environ['DATA_FILE']),
    'attention_cache':str(Path(os.environ['ATTENTION_CACHE']).resolve()),
    'semantic_source':os.environ['SEMANTIC_SOURCE'], 'expected_count':int(os.environ['EXPECTED_COUNT']),
    'worker_count':int(os.environ['WORKER_COUNT']), 'max_input_tokens':int(os.environ['MAX_INPUT_TOKENS']),
    'model_path':str(Path(os.environ['MODEL_PATH']).resolve()),
    'semantic_max_length':int(os.environ['SEMANTIC_MAX_LENGTH'])}
# Per-stage manifests additionally freeze encoder, prompt, code and cache hashes.
claim_manifest(root / 'run_manifest.json', signature)
PY
mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/run.log") 2>&1
pids=()
cleanup() {
  for pid in "${pids[@]}"; do [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$STAGE" == all ]]; then
  if [[ "$SEMANTIC_SOURCE" == bge ]]; then
    echo 'START semantic embeddings (one GPU; original FlagModel.encode interface)'
    CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PYTHON_BIN" "$PREP" semantic "${common[@]}" \
      --semantic-model-path "$BGE_MODEL_PATH" --semantic-max-length "$SEMANTIC_MAX_LENGTH" \
      --output-dir "$OUTPUT_ROOT/semantic" >>"$OUTPUT_ROOT/logs/semantic.log" 2>&1
  fi
  echo "START ${WORKER_COUNT} emotion workers; one local Qwen3-8B per assigned GPU"
  for ((worker=0; worker<WORKER_COUNT; worker++)); do
    CUDA_VISIBLE_DEVICES="${gpus[$worker]}" "$PYTHON_BIN" "$PREP" extract "${common[@]}" \
      --worker-index "$worker" --output-dir "$OUTPUT_ROOT/emotions" \
      >>"$OUTPUT_ROOT/logs/emotion.worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for ((worker=0; worker<WORKER_COUNT; worker++)); do
    wait "${pids[$worker]}" || failed=1
    pids[$worker]=""
  done
  (( failed == 0 )) || { echo 'Emotion extraction failed; inspect worker logs. Valid cached posts are preserved.' >&2; exit 1; }
  "$PYTHON_BIN" "$PREP" merge "${common[@]}" --output-dir "$OUTPUT_ROOT/emotions"
fi
echo 'START CPU retrieval metrics; original attention report stays untouched'
args=(--data-file "$DATA_FILE" --expected-count "$EXPECTED_COUNT" --attention-cache "$ATTENTION_CACHE"
      --emotions "$OUTPUT_ROOT/emotions/emotions.jsonl" --semantic-source "$SEMANTIC_SOURCE"
      --output-dir "$OUTPUT_ROOT/evaluation")
[[ "$SEMANTIC_SOURCE" != bge ]] || args+=(--semantic-cache "$OUTPUT_ROOT/semantic/semantic_vectors.npz")
"$PYTHON_BIN" "$EVAL" "${args[@]}"
echo "ALL COMPLETE: $OUTPUT_ROOT/evaluation/fresh_results_report.txt"
