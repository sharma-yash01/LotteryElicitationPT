#!/bin/bash
# bootstrap_lambda.sh -- One-time setup for GRPO training on Lambda Cloud VMs.
#
# Usage:
#   export LEPT_ROOT=/home/ubuntu/LotteryElicitationPT
#   export LEPT_VENV=/home/ubuntu/.venvs/lept-lambda
#   export LEPT_FS_NAME=<lambda-filesystem-name>   # optional if LEPT_DATA_ROOT set
#   export LEPT_DATA_ROOT=/lambda/nfs/<fs-name>/lept  # optional override
#   export PYTORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu121  # recommended for torch wheels
#   export LEPT_REQUIREMENTS_FILE="$LEPT_ROOT/requirements.txt"  # override default pin file if needed
#   bash scripts/bootstrap_lambda.sh

set -euo pipefail

: "${LEPT_ROOT:?Set LEPT_ROOT to the absolute path of LotteryElicitationPT}"
: "${LEPT_VENV:?Set LEPT_VENV to the absolute path for the Python venv}"

if [[ -n "${LEPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$LEPT_DATA_ROOT"
elif [[ -n "${LEPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${LEPT_FS_NAME}/lept"
else
    DATA_ROOT="/lambda/nfs/lept"
fi

LEPT_REQUIREMENTS_FILE="${LEPT_REQUIREMENTS_FILE:-$LEPT_ROOT/requirements.lambda.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"

CACHE_ROOT="${DATA_ROOT}/cache"
RUNS_ROOT="${DATA_ROOT}/runs"
LEPT_OUTPUT_DIR="${LEPT_OUTPUT_DIR:-${RUNS_ROOT}/grpo_train_lambda}"

echo "=== Lambda Bootstrap ==="
echo "  LEPT_ROOT             = $LEPT_ROOT"
echo "  LEPT_VENV             = $LEPT_VENV"
echo "  DATA_ROOT             = $DATA_ROOT"
echo "  Requirements          = $LEPT_REQUIREMENTS_FILE"
if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
    echo "  PyTorch extra index   = $PYTORCH_WHEEL_INDEX"
else
    echo "  PyTorch extra index   = <default pip indexes>"
fi
echo ""

if [[ ! -d "$LEPT_ROOT" ]]; then
    echo "[ERROR] LEPT_ROOT does not exist: $LEPT_ROOT"
    exit 1
fi

if [[ ! -f "$LEPT_REQUIREMENTS_FILE" ]]; then
    echo "[ERROR] requirements file not found: $LEPT_REQUIREMENTS_FILE"
    exit 1
fi

echo ">>> Preparing filesystem directories..."
mkdir -p "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$LEPT_OUTPUT_DIR"

if [[ -d "$LEPT_VENV" ]]; then
    echo ">>> Reusing existing venv at $LEPT_VENV"
else
    echo ">>> Creating venv at $LEPT_VENV"
    python3 -m venv "$LEPT_VENV"
fi

# shellcheck source=/dev/null
source "$LEPT_VENV/bin/activate"
echo "    Python: $(python --version) ($(which python))"

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TMPDIR="$CACHE_ROOT/tmp"

echo ">>> Cache config"
echo "    PIP_CACHE_DIR       = $PIP_CACHE_DIR"
echo "    HF_HOME             = $HF_HOME"
echo "    TRANSFORMERS_CACHE  = $TRANSFORMERS_CACHE"
echo "    TMPDIR              = $TMPDIR"

echo ">>> Upgrading pip..."
pip install --quiet --upgrade pip

echo ">>> Installing dependencies..."
if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
    pip install -r "$LEPT_REQUIREMENTS_FILE" --extra-index-url "$PYTORCH_WHEEL_INDEX"
else
    pip install -r "$LEPT_REQUIREMENTS_FILE"
fi

# echo ">>> TEMORARY TRANSFORMERS PIN..."
# pip install transformers==5.3.0 --force-reinstall --no-deps

echo ""
echo ">>> GPU visibility check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not found in PATH"
fi

echo ""
echo ">>> Smoke-testing critical imports..."
python - <<'PY'
import torch
import vllm
import trl
import transformers
import openenv
import jmespath

print(f"  torch        {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  vllm         {vllm.__version__}")
print(f"  trl          {trl.__version__}")
print(f"  transformers {transformers.__version__}")
print("  openenv-core OK")
print(f"  jmespath     {jmespath.__version__}")
PY

echo ""
echo "=== Bootstrap complete ==="
echo "Activate with: source \"$LEPT_VENV/bin/activate\""
echo "Suggested run dir: $LEPT_OUTPUT_DIR"
