#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cpfs01/hy/LlamaFactory/.venv/bin/python"
MERGER="${SCRIPT_DIR}/stream_merge_lora_safetensors.py"
BASE_DIR="/cpfs01/models/Macaron-V1-Venti"
ADAPTER_DIR="${BASE_DIR}/loras/L3"
OUTPUT_DIR="/cpfs01/models/Macaron-V1-Venti-merged-L3"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/stream_merge_l3_$(date +%Y%m%d_%H%M%S).log"

MODE="${1:-}"
if [[ $# -gt 1 || ( -n "${MODE}" && "${MODE}" != "--check" ) ]]; then
  printf 'Usage: %s [--check]\n' "$0" >&2
  exit 2
fi

[[ -x "${PYTHON}" ]] || { printf 'Python not found: %s\n' "${PYTHON}" >&2; exit 1; }
[[ -f "${MERGER}" ]] || { printf 'Merger not found: %s\n' "${MERGER}" >&2; exit 1; }

COMMON_ARGS=(
  --base-dir "${BASE_DIR}"
  --adapter-dir "${ADAPTER_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --row-chunk 2048
  --threads 64
)

if [[ "${MODE}" == "--check" ]]; then
  exec env CUDA_VISIBLE_DEVICES="" "${PYTHON}" "${MERGER}" "${COMMON_ARGS[@]}" --check-only
fi

mkdir -p -- "${LOG_DIR}"
printf 'CPU streaming merge; GPUs are intentionally disabled.\n'
printf '  base:    %s\n' "${BASE_DIR}"
printf '  adapter: %s\n' "${ADAPTER_DIR}"
printf '  output:  %s\n' "${OUTPUT_DIR}"
printf '  log:     %s\n' "${LOG_FILE}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export MALLOC_ARENA_MAX=4
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" "${MERGER}" "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
