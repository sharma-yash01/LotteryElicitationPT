#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_grpo_carc.sbatch"
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Validates config, prints resolved run summary, and submits run_grpo_carc.sbatch."
    echo ""
    echo "Required exports:"
    echo "  LEPT_ROOT       absolute path to LotteryElicitationPT on CARC"
    echo "  LEPT_VENV       absolute path to Python venv"
    echo "  ENV_BASE_URL    OpenEnv space URL"
    echo ""
    echo "Options:"
    echo "  --dry-run       Print config and exit without submitting"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

# ------------------------------------------------------------------ validate
ERRORS=0
for var in LEPT_ROOT LEPT_VENV ENV_BASE_URL; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] $var is not set."
        ERRORS=$((ERRORS + 1))
    fi
done

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "[ERROR] Missing sbatch file: ${SBATCH_FILE}"
    ERRORS=$((ERRORS + 1))
fi

if [[ -n "${LEPT_ROOT:-}" && ! -d "${LEPT_ROOT}" ]]; then
    echo "[ERROR] LEPT_ROOT does not exist: ${LEPT_ROOT}"
    ERRORS=$((ERRORS + 1))
fi

if [[ -n "${LEPT_VENV:-}" && ! -f "${LEPT_VENV}/bin/activate" ]]; then
    echo "[ERROR] LEPT_VENV has no bin/activate: ${LEPT_VENV} (run bootstrap first)"
    ERRORS=$((ERRORS + 1))
fi

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "$ERRORS error(s). Fix before submitting."
    exit 1
fi

# ------------------------------------------------------------------ resolved config
LEPT_MODEL="${LEPT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LEPT_OUTPUT_DIR="${LEPT_OUTPUT_DIR:-/scratch1/$USER/lept/runs/grpo_train_carc}"
LEPT_NUM_EPOCHS="${LEPT_NUM_EPOCHS:-1}"
LEPT_NUM_GENERATIONS="${LEPT_NUM_GENERATIONS:-8}"
LEPT_BATCH_SIZE="${LEPT_BATCH_SIZE:-2}"
LEPT_GRAD_ACCUM="${LEPT_GRAD_ACCUM:-8}"
LEPT_VLLM_MODE="${LEPT_VLLM_MODE:-colocate}"
LEPT_ALPHA="${LEPT_ALPHA:-1.0}"
LEPT_LOG_EVERY="${LEPT_LOG_EVERY:-1}"

echo "=== Resolved Run Config ==="
echo "  LEPT_ROOT        = ${LEPT_ROOT}"
echo "  LEPT_VENV        = ${LEPT_VENV}"
echo "  ENV_BASE_URL     = ${ENV_BASE_URL}"
echo "  Model            = ${LEPT_MODEL}"
echo "  Output dir       = ${LEPT_OUTPUT_DIR}"
echo "  Epochs           = ${LEPT_NUM_EPOCHS}"
echo "  Generations      = ${LEPT_NUM_GENERATIONS}"
echo "  Batch size       = ${LEPT_BATCH_SIZE}"
echo "  Grad accum       = ${LEPT_GRAD_ACCUM}"
echo "  vLLM mode        = ${LEPT_VLLM_MODE}"
echo "  Alpha            = ${LEPT_ALPHA}"
echo "  Log every        = ${LEPT_LOG_EVERY}"
echo "  Partition/GPU    = gpu / a40:1"
echo "==========================="

if [[ "$LEPT_OUTPUT_DIR" != /scratch1/* ]]; then
    echo "[WARN] LEPT_OUTPUT_DIR is not on /scratch1. Large checkpoints may exhaust home quota."
fi

# ------------------------------------------------------------------ submit or dry-run
if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Would submit: sbatch ${SBATCH_FILE}"
    echo "Re-run without --dry-run to submit."
    exit 0
fi

echo ""
sbatch "${SBATCH_FILE}"
