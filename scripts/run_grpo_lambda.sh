#!/bin/bash
# run_grpo_lambda.sh -- Direct GRPO launcher for Lambda Cloud VMs (no Slurm).
#
# Usage:
#   export LEPT_ROOT=/home/ubuntu/LotteryElicitationPT
#   export LEPT_VENV=/home/ubuntu/.venvs/lept-lambda
#   export ENV_BASE_URL=http://127.0.0.1:9000
#   export LEPT_FS_NAME=<lambda-filesystem-name>   # optional
#   bash scripts/run_grpo_lambda.sh [--dry-run]
#
# Multi-GPU: LEPT_VLLM_MODE=auto picks server if >=2 visible GPUs else colocate.
# Server mode: starts trl vllm-serve on LEPT_VLLM_PORT (default 8001), then accelerate launch
# on the remaining GPUs (LotteryElicitationEnv uses 9000 — do not collide).

set -euo pipefail

DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Runs GRPO training directly on a Lambda VM (no sbatch)."
    echo ""
    echo "Required exports:"
    echo "  LEPT_ROOT       absolute path to LotteryElicitationPT"
    echo "  LEPT_VENV       absolute path to Python venv"
    echo "  ENV_BASE_URL    OpenEnv endpoint base URL"
    echo ""
    echo "Optional exports:"
    echo "  LEPT_FS_NAME          Lambda filesystem name (used when LEPT_DATA_ROOT unset)"
    echo "  LEPT_DATA_ROOT        Base data path (default: /lambda/nfs/<fs>/lept or /lambda/nfs/lept)"
    echo "  LEPT_MODEL            default: Qwen/Qwen3-8B (Hub id → prefetched to \$DATA_ROOT/models/...; same id → OpenEnv tokenizer)"
    echo "  LEPT_OUTPUT_DIR       default: <DATA_ROOT>/runs/grpo_train_lambda"
    echo "  LEPT_NUM_EPOCHS       default: 1"
    echo "  LEPT_NUM_GENERATIONS  default: 8"
    echo "  LEPT_BATCH_SIZE       default: 8 (must divide evenly by LEPT_NUM_GENERATIONS)"
    echo "  LEPT_GRAD_ACCUM       default: 8 (auto-tuned unless LEPT_GRAD_ACCUM_OVERRIDE set)"
    echo "  LEPT_GRAD_ACCUM_OVERRIDE  set to any value to skip grad-accum auto-tune"
    echo "  LEPT_NUM_GPUS         default: auto (nvidia-smi count)"
    echo "  LEPT_VLLM_MODE        default: auto (server if >=2 GPUs, else colocate)"
    echo "  LEPT_VLLM_TP          default: 1 (vLLM tensor parallel GPUs in server mode)"
    echo "  LEPT_VLLM_PORT        default: 8001 (trl vllm-serve HTTP; must differ from OpenEnv, LotteryElicitationEnv uses 9000)"
    echo "  LEPT_VLLM_GROUP_PORT  default: 51216 (TRL weight-sync TCP; match training --vllm_group_port)"
    echo "  LEPT_VLLM_SERVER_HOST default: 127.0.0.1 (passed to grpo_train --vllm_server_host)"
    echo "  LEPT_NCCL_P2P_DISABLE optional: if set, overrides NCCL_P2P_DISABLE for multi-GPU (else 1; use 0 on NVLink A100)"
    echo "  LEPT_VLLM_GPU_UTIL    default: 0.9"
    echo "  LEPT_GRADIENT_CHECKPOINTING  default: 1"
    echo "  LEPT_MAX_COMPLETION_LENGTH   default: 2048"
    echo "  LEPT_MAX_TOKENS_PER_STEP   default: 512"
    echo "  LEPT_CURRICULUM_STAGE      default: 1"
    echo "  LEPT_FORMAT_WEIGHT         default: 0.1 (set LEPT_NO_FORMAT_REWARD=1 for --no_format_reward)"
    echo "  LEPT_NO_BF16          default: 0 (set to 1 for --no_bf16)"
    echo "  LEPT_ACCELERATE_MAIN_PORT  optional (default: 29500) for accelerate launch"
    echo "  LEPT_ALPHA            default: 1.0"
    echo "  LEPT_LOG_EVERY        default: 1"
    echo "  LEPT_INSTALL_DEPS_ON_RUN  default: 0 (set to 1 to pip install before run)"
    echo "  LEPT_REQUIREMENTS_FILE    default: <LEPT_ROOT>/requirements.lambda.txt"
    echo "  PYTORCH_WHEEL_INDEX       optional pip extra index URL"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

: "${LEPT_ROOT:?LEPT_ROOT is required}"
: "${LEPT_VENV:?LEPT_VENV is required}"
: "${ENV_BASE_URL:?ENV_BASE_URL is required}"

if [[ -n "${LEPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$LEPT_DATA_ROOT"
elif [[ -n "${LEPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${LEPT_FS_NAME}/lept"
else
    DATA_ROOT="/lambda/nfs/lept"
fi

LEPT_MODEL="${LEPT_MODEL:-Qwen/Qwen3-4B}"
LEPT_OUTPUT_DIR="${LEPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_train_lambda}"
LEPT_NUM_EPOCHS="${LEPT_NUM_EPOCHS:-1}"
LEPT_NUM_GENERATIONS="${LEPT_NUM_GENERATIONS:-8}"
LEPT_BATCH_SIZE="${LEPT_BATCH_SIZE:-8}"
LEPT_GRAD_ACCUM="${LEPT_GRAD_ACCUM:-8}"
LEPT_NUM_GPUS="${LEPT_NUM_GPUS:-auto}"
LEPT_VLLM_TP="${LEPT_VLLM_TP:-1}"
if ! [[ "$LEPT_VLLM_TP" =~ ^[0-9]+$ ]] || [[ "$LEPT_VLLM_TP" -lt 1 ]]; then
    echo "[ERROR] LEPT_VLLM_TP must be a positive integer (got: $LEPT_VLLM_TP)"
    exit 1
fi
LEPT_VLLM_GPU_UTIL="${LEPT_VLLM_GPU_UTIL:-0.9}"
LEPT_VLLM_PORT="${LEPT_VLLM_PORT:-8001}"
LEPT_VLLM_GROUP_PORT="${LEPT_VLLM_GROUP_PORT:-51216}"
LEPT_VLLM_SERVER_HOST="${LEPT_VLLM_SERVER_HOST:-127.0.0.1}"
LEPT_VLLM_MODE="${LEPT_VLLM_MODE:-auto}"
LEPT_GRADIENT_CHECKPOINTING="${LEPT_GRADIENT_CHECKPOINTING:-1}"
LEPT_MAX_COMPLETION_LENGTH="${LEPT_MAX_COMPLETION_LENGTH:-2048}"
LEPT_MAX_TOKENS_PER_STEP="${LEPT_MAX_TOKENS_PER_STEP:-512}"
LEPT_CURRICULUM_STAGE="${LEPT_CURRICULUM_STAGE:-1}"
LEPT_FORMAT_WEIGHT="${LEPT_FORMAT_WEIGHT:-0.1}"
LEPT_NO_FORMAT_REWARD="${LEPT_NO_FORMAT_REWARD:-0}"
LEPT_NO_BF16="${LEPT_NO_BF16:-0}"
LEPT_ALPHA="${LEPT_ALPHA:-1.0}"
LEPT_LOG_EVERY="${LEPT_LOG_EVERY:-1}"
LEPT_INSTALL_DEPS_ON_RUN="${LEPT_INSTALL_DEPS_ON_RUN:-0}"
LEPT_REQUIREMENTS_FILE="${LEPT_REQUIREMENTS_FILE:-$LEPT_ROOT/requirements.lambda.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"

# ---- GPU fleet & vLLM mode (respects CUDA_VISIBLE_DEVICES via nvidia-smi) ----
if [[ "$LEPT_VLLM_MODE" != "auto" && "$LEPT_VLLM_MODE" != "server" && "$LEPT_VLLM_MODE" != "colocate" ]]; then
    echo "[ERROR] LEPT_VLLM_MODE must be auto, server, or colocate (got: $LEPT_VLLM_MODE)"
    exit 1
fi

if [[ "$LEPT_NUM_GPUS" == "auto" ]]; then
    LEPT_NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if ! [[ "$LEPT_NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$LEPT_NUM_GPUS" -lt 1 ]]; then
        LEPT_NUM_GPUS=1
    fi
    echo "  Auto-detected GPUs: $LEPT_NUM_GPUS"
elif ! [[ "$LEPT_NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$LEPT_NUM_GPUS" -lt 1 ]]; then
    echo "[ERROR] LEPT_NUM_GPUS must be auto or a positive integer (got: $LEPT_NUM_GPUS)"
    exit 1
fi

if [[ "$LEPT_VLLM_MODE" == "auto" ]]; then
    if [[ "$LEPT_NUM_GPUS" -ge 2 ]]; then
        LEPT_VLLM_MODE="server"
    else
        LEPT_VLLM_MODE="colocate"
    fi
    echo "  Auto-selected vLLM mode: $LEPT_VLLM_MODE"
fi

if [[ "$LEPT_VLLM_MODE" == "server" ]]; then
    TRAIN_PROCS=$((LEPT_NUM_GPUS - LEPT_VLLM_TP))
    if [[ "$TRAIN_PROCS" -lt 1 ]]; then
        echo "[ERROR] Not enough GPUs for server mode: $LEPT_NUM_GPUS total, TP=$LEPT_VLLM_TP"
        echo "Need at least (TP + 1) GPUs. Use colocate mode or reduce LEPT_VLLM_TP."
        exit 1
    fi
else
    TRAIN_PROCS=1
fi

TARGET_EFFECTIVE_PROMPTS=16
EFFECTIVE_PER_STEP=$((LEPT_BATCH_SIZE * TRAIN_PROCS))
if [[ -z "${LEPT_GRAD_ACCUM_OVERRIDE:-}" ]] && [[ "$EFFECTIVE_PER_STEP" -gt 0 ]]; then
    AUTO_GRAD_ACCUM=$(( (TARGET_EFFECTIVE_PROMPTS + EFFECTIVE_PER_STEP - 1) / EFFECTIVE_PER_STEP ))
    if [[ "$AUTO_GRAD_ACCUM" -lt 1 ]]; then
        AUTO_GRAD_ACCUM=1
    fi
    LEPT_GRAD_ACCUM="$AUTO_GRAD_ACCUM"
    echo "  Auto-adjusted grad_accum to $LEPT_GRAD_ACCUM (target $TARGET_EFFECTIVE_PROMPTS effective prompts/step)"
fi

if [[ "$TRAIN_PROCS" -gt 1 ]]; then
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
    # Prefer LEPT_NCCL_P2P_DISABLE when set; else inherit NCCL_P2P_DISABLE; default 1 (PCIe/V100-safe).
    # On NVLink A100 SXM4, set LEPT_NCCL_P2P_DISABLE=0 or export NCCL_P2P_DISABLE=0 before the script.
    export NCCL_P2P_DISABLE="${LEPT_NCCL_P2P_DISABLE:-${NCCL_P2P_DISABLE:-1}}"
    echo "  NCCL: NCCL_TIMEOUT=${NCCL_TIMEOUT} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} (multi-process training)"
fi

CACHE_ROOT="${DATA_ROOT}/cache"
mkdir -p "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$LEPT_OUTPUT_DIR"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TMPDIR="$CACHE_ROOT/tmp"

if [[ ! -d "$LEPT_ROOT" ]]; then
    echo "[ERROR] LEPT_ROOT does not exist: $LEPT_ROOT"
    exit 1
fi

if [[ ! -f "$LEPT_VENV/bin/activate" ]]; then
    echo "[ERROR] LEPT_VENV has no bin/activate: $LEPT_VENV"
    echo "Run scripts/bootstrap_lambda.sh first."
    exit 1
fi

if [[ "$LEPT_OUTPUT_DIR" != /lambda/nfs/* ]]; then
    echo "[WARN] LEPT_OUTPUT_DIR is not under /lambda/nfs: $LEPT_OUTPUT_DIR"
fi

cd "$LEPT_ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source "$LEPT_VENV/bin/activate"

if [[ "$LEPT_INSTALL_DEPS_ON_RUN" == "1" ]]; then
    if [[ ! -f "$LEPT_REQUIREMENTS_FILE" ]]; then
        echo "[ERROR] requirements file not found: $LEPT_REQUIREMENTS_FILE"
        exit 1
    fi
    echo ">>> Installing dependencies before run..."
    pip install --quiet --upgrade pip
    if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
        pip install -r "$LEPT_REQUIREMENTS_FILE" --extra-index-url "$PYTORCH_WHEEL_INDEX"
    else
        pip install -r "$LEPT_REQUIREMENTS_FILE"
    fi
fi

GPU_LINE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
echo "=== LEPT GRPO Training (Lambda) ==="
echo "  Host:            $(hostname)"
echo "  GPUs:            ${LEPT_NUM_GPUS} x ${GPU_LINE}"
echo "  vLLM mode:       $LEPT_VLLM_MODE"
echo "  vLLM TP:         ${LEPT_VLLM_TP} GPU(s)"
echo "  Training procs:  ${TRAIN_PROCS}"
echo "  Python:          $(python --version)"
echo "  Torch:           $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Model:           $LEPT_MODEL"
echo "  Batch/device:    $LEPT_BATCH_SIZE"
echo "  Num generations: $LEPT_NUM_GENERATIONS"
echo "  Grad accum:      $LEPT_GRAD_ACCUM"
echo "  Output dir:      $LEPT_OUTPUT_DIR"
echo "  Env URL:         $ENV_BASE_URL"
echo "==============================="

# ------------------------------------------------------------------ dependency precheck
echo ">>> Verifying critical Python imports..."
for mod in torch vllm trl transformers datasets huggingface_hub openenv jmespath; do
    if python -c "import $mod" >/dev/null 2>&1; then
        echo "  [PASS] import $mod"
    else
        echo "  [FAIL] import $mod"
        echo "Install/update dependencies first (bootstrap_lambda.sh or LEPT_INSTALL_DEPS_ON_RUN=1)."
        exit 1
    fi
done

if [[ "$LEPT_VLLM_MODE" == "server" ]]; then
    if ! command -v accelerate >/dev/null 2>&1; then
        echo "  [FAIL] accelerate CLI not on PATH (required for vllm_mode=server)"
        exit 1
    fi
    echo "  [PASS] accelerate CLI available"
fi

# ---- Prefetch Hub models to a plain directory (avoids NFS + concurrent Hub cache races) ----
# If LEPT_MODEL is already a path (absolute or ./...), use it as-is.
if [[ $DRY_RUN -eq 0 ]] && [[ "$LEPT_MODEL" != /* ]] && [[ "$LEPT_MODEL" != ./* ]]; then
    MODEL_LOCAL_DIR="${DATA_ROOT}/models/${LEPT_MODEL//\//_}"
    mkdir -p "${DATA_ROOT}/models"
    if [[ ! -f "$MODEL_LOCAL_DIR/model.safetensors.index.json" ]] && \
       [[ ! -f "$MODEL_LOCAL_DIR/model.safetensors" ]]; then
        echo ">>> Prefetching model to plain directory: $MODEL_LOCAL_DIR"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${LEPT_MODEL}',
    local_dir='${MODEL_LOCAL_DIR}',
    local_dir_use_symlinks=False,
)
print('Prefetch complete.')
"
    else
        echo ">>> Model already present at: $MODEL_LOCAL_DIR"
    fi
    LEPT_MODEL="$MODEL_LOCAL_DIR"
    echo ">>> Training will load model from: $LEPT_MODEL"
fi

VLLM_PID=""
if [[ "$LEPT_VLLM_MODE" == "server" ]]; then
    VLLM_GPU_END=$((LEPT_NUM_GPUS - 1))
    VLLM_GPU_START=$((LEPT_NUM_GPUS - LEPT_VLLM_TP))
    VLLM_CUDA_DEVS=$(seq -s, "$VLLM_GPU_START" "$VLLM_GPU_END")
    TRAIN_CUDA_DEVS=$(seq -s, 0 $((VLLM_GPU_START - 1)))

    VLLM_CURL_HOST="$LEPT_VLLM_SERVER_HOST"
    if [[ "$VLLM_CURL_HOST" == "0.0.0.0" ]]; then
        VLLM_CURL_HOST="127.0.0.1"
    fi
    VLLM_URL="http://${VLLM_CURL_HOST}:${LEPT_VLLM_PORT}"

    if [[ $DRY_RUN -eq 0 ]]; then
        if ! command -v trl >/dev/null 2>&1; then
            echo "[ERROR] trl CLI not on PATH (required to start trl vllm-serve in server mode)"
            exit 1
        fi
        echo ">>> Starting trl vllm-serve on GPU(s) [$VLLM_CUDA_DEVS] port $LEPT_VLLM_PORT ..."
        CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVS" \
            trl vllm-serve \
                --model "$LEPT_MODEL" \
                --tensor-parallel-size "$LEPT_VLLM_TP" \
                --gpu-memory-utilization "$LEPT_VLLM_GPU_UTIL" \
                --port "$LEPT_VLLM_PORT" \
            >> "$LEPT_OUTPUT_DIR/vllm_serve.log" 2>&1 &
        VLLM_PID=$!
        echo "  vllm-serve PID: $VLLM_PID  (log: $LEPT_OUTPUT_DIR/vllm_serve.log)"

        echo ">>> Waiting for trl vllm-serve at $VLLM_URL ..."
        VLLM_READY=0
        for i in $(seq 1 120); do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "[ERROR] vllm-serve process exited early (PID $VLLM_PID). Check $LEPT_OUTPUT_DIR/vllm_serve.log"
                exit 1
            fi
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                "${VLLM_URL}/get_world_size/" 2>/dev/null || echo "000")
            if [[ "$HTTP_CODE" == "200" ]]; then
                VLLM_READY=1
                echo "  [PASS] trl vllm-serve ready at ${VLLM_URL} (attempt $i)"
                break
            fi
            sleep 5
        done

        if [[ "$VLLM_READY" -ne 1 ]]; then
            echo "[ERROR] trl vllm-serve did not become ready within 600s. Check $LEPT_OUTPUT_DIR/vllm_serve.log"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi

        # shellcheck disable=SC2064
        trap "echo '>>> Stopping vllm-serve (PID $VLLM_PID)...'; kill '$VLLM_PID' 2>/dev/null || true" EXIT
    fi
fi

COMMON_ARGS=(
    -m training.grpo_train
    --model "$LEPT_MODEL"
    --env_base_url "$ENV_BASE_URL"
    --alpha "$LEPT_ALPHA"
    --format_weight "$LEPT_FORMAT_WEIGHT"
    --curriculum_stage "$LEPT_CURRICULUM_STAGE"
    --max_tokens_per_step "$LEPT_MAX_TOKENS_PER_STEP"
    --log_every_n_steps "$LEPT_LOG_EVERY"
    --num_train_epochs "$LEPT_NUM_EPOCHS"
    --num_generations "$LEPT_NUM_GENERATIONS"
    --per_device_train_batch_size "$LEPT_BATCH_SIZE"
    --gradient_accumulation_steps "$LEPT_GRAD_ACCUM"
    --vllm_mode "$LEPT_VLLM_MODE"
    --vllm_tensor_parallel_size "$LEPT_VLLM_TP"
    --vllm_gpu_memory_utilization "$LEPT_VLLM_GPU_UTIL"
    --vllm_server_host "$LEPT_VLLM_SERVER_HOST"
    --vllm_server_port "$LEPT_VLLM_PORT"
    --vllm_group_port "$LEPT_VLLM_GROUP_PORT"
    --max_completion_length "$LEPT_MAX_COMPLETION_LENGTH"
    --output_dir "$LEPT_OUTPUT_DIR"
)
if [[ "${LEPT_NO_FORMAT_REWARD}" == "1" ]]; then
    COMMON_ARGS+=(--no_format_reward)
fi

if [[ "${LEPT_GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
    COMMON_ARGS+=(--gradient_checkpointing)
fi

if [[ "${LEPT_NO_BF16:-0}" == "1" ]]; then
    COMMON_ARGS+=(--no_bf16)
fi

if [[ "$LEPT_VLLM_MODE" == "server" ]]; then
    TRAIN_CMD=(
        env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_DEVS"
        accelerate launch
        --num_processes "$TRAIN_PROCS"
        --main_process_port "${LEPT_ACCELERATE_MAIN_PORT:-29500}"
        "${COMMON_ARGS[@]}"
    )
else
    TRAIN_CMD=(python "${COMMON_ARGS[@]}")
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Command:"
    printf '  %q' "${TRAIN_CMD[@]}"
    echo ""
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "=== Training complete. Artifacts at: $LEPT_OUTPUT_DIR ==="
