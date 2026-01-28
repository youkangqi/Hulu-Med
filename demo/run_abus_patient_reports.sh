#!/usr/bin/env bash
set -euo pipefail

# Batch inference for ABUS dataset using Hulu-Med-7B.
# Usage:
#   bash demo/run_abus_patient_reports.sh
#   DATA_ROOT=... OUTPUT_DIR=... MAX_PATIENTS=... RANDOM_SELECT=1 SEED=42 bash demo/run_abus_patient_reports.sh

log() {
  echo "[INFO] $(date '+%F %T') $*"
}

DATA_ROOT="${DATA_ROOT:-data/abus_data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/abus_reports_raw_report_cn}"
MODEL_PATH="${MODEL_PATH:-/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
PATIENT_ID="${PATIENT_ID:-}"
MAX_PATIENTS="${MAX_PATIENTS:-}"
OVERWRITE="${OVERWRITE:-1}"
USE_THINK="${USE_THINK:-0}"
RANDOM_SELECT="${RANDOM_SELECT:-1}"
SEED="${SEED:-2026}"
CN_LANGUAGE="${CN_LANGUAGE:-1}"
KEYWORDS_PATH="${KEYWORDS_PATH:-data/kVME_data/data/key_technical_description_words_v1.txt}"
NUM_PER_CLASS="${NUM_PER_CLASS:-5}"
NO_SINGLE_FILES="${NO_SINGLE_FILES:-1}"
STRUCTURAL_OUTPUT="${STRUCTURAL_OUTPUT:-0}"

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
log "DATA_ROOT=${DATA_ROOT}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "MODEL_PATH=${MODEL_PATH}"
log "MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DTYPE=${DTYPE} ATTN_IMPL=${ATTN_IMPL}"
log "PATIENT_ID=${PATIENT_ID:-<all>} MAX_PATIENTS=${MAX_PATIENTS:-<all>}"
log "RANDOM_SELECT=${RANDOM_SELECT} SEED=${SEED:-<auto>} OVERWRITE=${OVERWRITE} USE_THINK=${USE_THINK}"
log "KEYWORDS_PATH=${KEYWORDS_PATH}"
log "CN_LANGUAGE=${CN_LANGUAGE}"
log "NUM_PER_CLASS=${NUM_PER_CLASS} STRUCTURAL_OUTPUT=${STRUCTURAL_OUTPUT}"
log "NO_SINGLE_FILES=${NO_SINGLE_FILES}"

ARGS=(
  "--data-root" "${DATA_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--model-path" "${MODEL_PATH}"
  "--max-new-tokens" "${MAX_NEW_TOKENS}"
  "--dtype" "${DTYPE}"
  "--attn-impl" "${ATTN_IMPL}"
  "--keywords-path" "${KEYWORDS_PATH}"
  "--num-per-class" "${NUM_PER_CLASS}"
  "--structural-output" "${STRUCTURAL_OUTPUT}"
)

if [[ -n "${PATIENT_ID}" ]]; then
  ARGS+=("--patient-id" "${PATIENT_ID}")
fi

if [[ -n "${MAX_PATIENTS}" ]]; then
  ARGS+=("--max-patients" "${MAX_PATIENTS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=("--overwrite")
fi

if [[ "${USE_THINK}" == "1" ]]; then
  ARGS+=("--use-think")
fi

if [[ "${RANDOM_SELECT}" == "1" ]]; then
  ARGS+=("--random-select")
  if [[ -n "${SEED}" ]]; then
    ARGS+=("--seed" "${SEED}")
  fi
fi

if [[ "${NO_SINGLE_FILES}" == "1" ]]; then
  ARGS+=("--no-single-files")
fi

if [[ "${CN_LANGUAGE}" == "1" ]]; then
  ARGS+=("--language" "zh")
else
  ARGS+=("--language" "en")
fi

log "Launching inference..."
PYTHONUNBUFFERED=1 python demo/abus_patient_reports.py "${ARGS[@]}"
log "Inference done"
