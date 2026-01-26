#!/usr/bin/env bash
set -euo pipefail

# Run text-only inference on outputs/abus_b_g.csv.
# Usage:
#   bash demo/run_abus_csv_text_infer.sh
#   CSV_PATH=outputs/abus_b_g.csv MAX_ITEMS=50 bash demo/run_abus_csv_text_infer.sh

log() {
  echo "[INFO] $(date '+%F %T') $*"
}

MODEL_PATH="${MODEL_PATH:-/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/}"
CSV_PATH="${CSV_PATH:-outputs/abus_b_g.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/abus_b_g_infer}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_THINK="${USE_THINK:-0}"
MAX_ITEMS="${MAX_ITEMS:-}"
OVERWRITE="${OVERWRITE:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n1 || true)"
    if [[ -n "${gpu_line}" ]]; then
      gpu_index="$(echo "${gpu_line}" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
      export CUDA_VISIBLE_DEVICES="${gpu_index}"
    fi
  fi
fi

log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
log "MODEL_PATH=${MODEL_PATH}"
log "CSV_PATH=${CSV_PATH}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DTYPE=${DTYPE} ATTN_IMPL=${ATTN_IMPL}"
log "USE_THINK=${USE_THINK} MAX_ITEMS=${MAX_ITEMS:-<all>} OVERWRITE=${OVERWRITE}"

ARGS=(
  "--model-path" "${MODEL_PATH}"
  "--csv-path" "${CSV_PATH}"
  "--output-dir" "${OUTPUT_DIR}"
  "--max-new-tokens" "${MAX_NEW_TOKENS}"
  "--dtype" "${DTYPE}"
  "--attn-impl" "${ATTN_IMPL}"
)

if [[ -n "${PROMPT_TEMPLATE}" ]]; then
  ARGS+=("--prompt-template" "${PROMPT_TEMPLATE}")
fi

if [[ "${USE_THINK}" == "1" ]]; then
  ARGS+=("--use-think")
fi

if [[ -n "${MAX_ITEMS}" ]]; then
  ARGS+=("--max-items" "${MAX_ITEMS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=("--overwrite")
fi

log "Launching inference..."
PYTHONUNBUFFERED=1 python demo/abus_csv_text_infer.py "${ARGS[@]}"
