#!/usr/bin/env bash
set -euo pipefail

# Run multi-image inference on ABUS patient sequences.
# Usage:
#   PATIENT_ID=33062 bash demo/run_abus_multi_images_patient.sh
#   ALL_PATIENTS=1 STRIDE=2 MAX_IMAGES=128 bash demo/run_abus_multi_images_patient.sh

log() {
  echo "[INFO] $(date '+%F %T') $*"
}

DATA_ROOT="${DATA_ROOT:-data/abus_data}"
MODEL_PATH="${MODEL_PATH:-/homeB/youkangqi/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-7B/snapshots/258594714a0d3835eb2c9e4cc165a4242e606d71/}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/abus_reports_raw_report_cn_with_keys}"
PATIENT_ID="${PATIENT_ID:-33062}"
ALL_PATIENTS="${ALL_PATIENTS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_THINK="${USE_THINK:-0}"
STRIDE="${STRIDE:-1}"
MAX_IMAGES="${MAX_IMAGES:-}"
LABELS_ONLY="${LABELS_ONLY:-1}"
KEYWORDS_PATH="${KEYWORDS_PATH:-data/kVME_data/data/key_technical_description_words.txt}"
OVERWRITE="${OVERWRITE:-1}"
PRINT_INPUT_LENGTH="${PRINT_INPUT_LENGTH:-0}"
PRINT_PROMPT="${PRINT_PROMPT:-0}"
CN_LANGUAGE="${CN_LANGUAGE:-1}"

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
log "MODEL_PATH=${MODEL_PATH}"
log "PATIENT_ID=${PATIENT_ID:-<none>} ALL_PATIENTS=${ALL_PATIENTS}"
log "OUTPUT_PATH=${OUTPUT_PATH:-<auto>}"
log "MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DTYPE=${DTYPE} ATTN_IMPL=${ATTN_IMPL}"
log "STRIDE=${STRIDE} MAX_IMAGES=${MAX_IMAGES:-<all>} LABELS_ONLY=${LABELS_ONLY} USE_THINK=${USE_THINK} OVERWRITE=${OVERWRITE}"
log "KEYWORDS_PATH=${KEYWORDS_PATH}"

ARGS=(
  "--data-root" "${DATA_ROOT}"
  "--model-path" "${MODEL_PATH}"
  "--max-new-tokens" "${MAX_NEW_TOKENS}"
  "--dtype" "${DTYPE}"
  "--attn-impl" "${ATTN_IMPL}"
  "--stride" "${STRIDE}"
  "--keywords-path" "${KEYWORDS_PATH}"
)

if [[ -n "${OUTPUT_PATH}" ]]; then
  ARGS+=("--output-path" "${OUTPUT_PATH}")
fi

if [[ "${USE_THINK}" == "1" ]]; then
  ARGS+=("--use-think")
fi

if [[ -n "${MAX_IMAGES}" ]]; then
  ARGS+=("--max-images" "${MAX_IMAGES}")
fi

if [[ "${LABELS_ONLY}" == "1" ]]; then
  ARGS+=("--labels-only")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=("--overwrite")
fi

if [[ "${ALL_PATIENTS}" == "1" ]]; then
  ARGS+=("--all-patients")
else
  if [[ -z "${PATIENT_ID}" ]]; then
    echo "ERROR: set PATIENT_ID or ALL_PATIENTS=1" >&2
    exit 1
  fi
  ARGS+=("--patient-id" "${PATIENT_ID}")
fi

if [[ "${PRINT_INPUT_LENGTH}" == "1" ]]; then
  ARGS+=("--print-input-length")
fi

if [[ "${PRINT_PROMPT}" == "1" ]]; then
  ARGS+=("--print-prompt")
fi

if [[ "${CN_LANGUAGE}" == "1" ]]; then
  ARGS+=("--language" "zh")
else
  ARGS+=("--language" "en")
fi

log "Launching inference..."
PYTHONUNBUFFERED=1 python demo/abus_multi_images_patient.py "${ARGS[@]}"
