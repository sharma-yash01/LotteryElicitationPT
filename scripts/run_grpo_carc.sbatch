#!/bin/bash
#SBATCH --job-name=lept-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lept-grpo-%j.out

set -euo pipefail

# Required exports (set in shell before sbatch, or in submit_grpo_carc.sh):
#   LEPT_ROOT                absolute path to LotteryElicitationPT on CARC
#   LEPT_VENV                absolute path to Python venv with dependencies installed
#   ENV_BASE_URL             OpenEnv space URL (https://owner-space.hf.space)
#
# Optional exports:
#   LEPT_MODEL               default: Qwen/Qwen2.5-0.5B-Instruct
#   LEPT_OUTPUT_DIR          default: /scratch1/$USER/lept/runs/grpo_train_carc
#   LEPT_NUM_EPOCHS          default: 1
#   LEPT_NUM_GENERATIONS     default: 8
#   LEPT_BATCH_SIZE          default: 2
#   LEPT_GRAD_ACCUM          default: 8
#   LEPT_VLLM_MODE           default: colocate
#   LEPT_ALPHA               default: 1.0
#   LEPT_LOG_EVERY           default: 1

: "${LEPT_ROOT:?LEPT_ROOT is required}"
: "${LEPT_VENV:?LEPT_VENV is required}"
: "${ENV_BASE_URL:?ENV_BASE_URL is required}"

LEPT_MODEL="${LEPT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LEPT_OUTPUT_DIR="${LEPT_OUTPUT_DIR:-/scratch1/$USER/lept/runs/grpo_train_carc}"
LEPT_NUM_EPOCHS="${LEPT_NUM_EPOCHS:-1}"
LEPT_NUM_GENERATIONS="${LEPT_NUM_GENERATIONS:-8}"
LEPT_BATCH_SIZE="${LEPT_BATCH_SIZE:-2}"
LEPT_GRAD_ACCUM="${LEPT_GRAD_ACCUM:-8}"
LEPT_VLLM_MODE="${LEPT_VLLM_MODE:-colocate}"
LEPT_ALPHA="${LEPT_ALPHA:-1.0}"
LEPT_LOG_EVERY="${LEPT_LOG_EVERY:-1}"
LEPT_MAX_COMPLETION_LENGTH="${LEPT_MAX_COMPLETION_LENGTH:-2048}"
LEPT_MAX_TOKENS_PER_STEP="${LEPT_MAX_TOKENS_PER_STEP:-512}"
LEPT_CURRICULUM_STAGE="${LEPT_CURRICULUM_STAGE:-1}"
LEPT_FORMAT_WEIGHT="${LEPT_FORMAT_WEIGHT:-0.1}"

SCRATCH_CACHE="/scratch1/$USER/cache"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu121}"
MODULE_GCC="${MODULE_GCC:-gcc/13.3.0}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.11.9}"
MODULE_CUDA="${MODULE_CUDA:-cuda/12.6.3}"

# ------------------------------------------------------------------ modules
# gcc/12.3.0: loadable on Discovery2 without legacy/CentOS7 (gcc/11.3.0 requires that stack).
echo ">>> Loading modules..."
module purge
module load "$MODULE_GCC"
module load "$MODULE_PYTHON"
module load "$MODULE_CUDA"
echo "    Loaded: $MODULE_GCC, $MODULE_PYTHON, $MODULE_CUDA"
# ------------------------------------------------------------------ venv
cd "$LEPT_ROOT"
mkdir -p logs
if [[ -d "$LEPT_VENV" ]]; then
  echo ">>> Reusing venv at $LEPT_VENV"
else
  echo ">>> Creating venv at $LEPT_VENV"
  python3 -m venv "$LEPT_VENV"
fi
source "$LEPT_VENV/bin/activate"

# ------------------------------------------------------------------ runtime caches on scratch
mkdir -p "$SCRATCH_CACHE/pip" "$SCRATCH_CACHE/huggingface" "$SCRATCH_CACHE/tmp" "$LEPT_OUTPUT_DIR"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH_CACHE/huggingface/transformers"
export TMPDIR="$SCRATCH_CACHE/tmp"

# ------------------------------------------------------------------ in-job dependency bootstrap
echo ">>> Upgrading pip..."
pip install --quiet --upgrade pip
echo ">>> Installing pinned requirements..."
pip install -r "$LEPT_ROOT/requirements.carc-cu121.txt" \
  --extra-index-url "$PYTORCH_WHEEL_INDEX"

# ------------------------------------------------------------------ diagnostics
echo "=== LEPT GRPO Training ==="
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Python:      $(python --version)"
echo "  Torch:       $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Wheel index: $PYTORCH_WHEEL_INDEX"
echo "  Model:       $LEPT_MODEL"
echo "  Output dir:  $LEPT_OUTPUT_DIR"
echo "  Env URL:     $ENV_BASE_URL"
echo "=========================="

# ------------------------------------------------------------------ training
python -m training.grpo_train \
  --model "$LEPT_MODEL" \
  --env_base_url "$ENV_BASE_URL" \
  --alpha "$LEPT_ALPHA" \
  --format_weight "$LEPT_FORMAT_WEIGHT" \
  --curriculum_stage "$LEPT_CURRICULUM_STAGE" \
  --max_tokens_per_step "$LEPT_MAX_TOKENS_PER_STEP" \
  --max_completion_length "$LEPT_MAX_COMPLETION_LENGTH" \
  --log_every_n_steps "$LEPT_LOG_EVERY" \
  --num_train_epochs "$LEPT_NUM_EPOCHS" \
  --num_generations "$LEPT_NUM_GENERATIONS" \
  --per_device_train_batch_size "$LEPT_BATCH_SIZE" \
  --gradient_accumulation_steps "$LEPT_GRAD_ACCUM" \
  --vllm_mode "$LEPT_VLLM_MODE" \
  --output_dir "$LEPT_OUTPUT_DIR"

echo "=== Training complete. Artifacts at: $LEPT_OUTPUT_DIR ==="
