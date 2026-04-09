#!/bin/bash
# preflight_lambda.sh -- Sanity checks before GRPO training on Lambda Cloud VMs.
#
# Usage:
#   export LEPT_ROOT=/home/ubuntu/LotteryElicitationPT
#   export LEPT_VENV=/home/ubuntu/.venvs/lept-lambda
#   export ENV_BASE_URL=http://127.0.0.1:9000
#   export LEPT_FS_NAME=<lambda-filesystem-name>   # optional
#   export LEPT_DATA_ROOT=/lambda/nfs/<fs-name>/lept  # optional override
#   bash scripts/preflight_lambda.sh

set -euo pipefail

FAIL=0
WARN=0

pass()  { echo "  [PASS] $*"; }
fail()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
warn()  { echo "  [WARN] $*"; WARN=$((WARN + 1)); }

if [[ -n "${LEPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$LEPT_DATA_ROOT"
elif [[ -n "${LEPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${LEPT_FS_NAME}/lept"
else
    DATA_ROOT="/lambda/nfs/lept"
fi

LEPT_OUTPUT_DIR="${LEPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_train_lambda}"

echo "=== Lambda Preflight Checks ==="
echo ""

echo "--- Required environment variables ---"
for var in LEPT_ROOT LEPT_VENV ENV_BASE_URL; do
    if [[ -z "${!var:-}" ]]; then
        fail "$var is not set"
    else
        pass "$var = ${!var}"
    fi
done

echo ""
echo "--- Paths ---"
if [[ -n "${LEPT_ROOT:-}" && -d "$LEPT_ROOT" ]]; then
    pass "LEPT_ROOT directory exists"
elif [[ -n "${LEPT_ROOT:-}" ]]; then
    fail "LEPT_ROOT directory does not exist: $LEPT_ROOT"
fi

REQ_FILE="${LEPT_REQUIREMENTS_FILE:-${LEPT_ROOT:-}/requirements.lambda.txt}"
if [[ -n "${LEPT_ROOT:-}" && -d "$LEPT_ROOT" ]]; then
    echo ""
    echo "--- Requirements file ---"
    if [[ -f "$REQ_FILE" ]]; then
        pass "requirements file exists: $REQ_FILE"
    else
        warn "requirements file not found: $REQ_FILE"
        warn "  bootstrap_lambda.sh defaults to requirements.lambda.txt; set LEPT_REQUIREMENTS_FILE to override."
    fi
fi

if [[ -n "${LEPT_VENV:-}" && -f "$LEPT_VENV/bin/activate" ]]; then
    pass "LEPT_VENV has bin/activate"
elif [[ -n "${LEPT_VENV:-}" ]]; then
    fail "LEPT_VENV missing or no bin/activate: $LEPT_VENV (run bootstrap first)"
fi

if [[ -d "$DATA_ROOT" ]]; then
    pass "Data root exists: $DATA_ROOT"
else
    warn "Data root does not exist yet: $DATA_ROOT"
    warn "  Create it or set LEPT_DATA_ROOT to your mounted Lambda filesystem path."
fi

echo ""
echo "--- Output directory ---"
if [[ "$LEPT_OUTPUT_DIR" == /lambda/nfs/* ]]; then
    pass "LEPT_OUTPUT_DIR is on Lambda filesystem mount: $LEPT_OUTPUT_DIR"
else
    warn "LEPT_OUTPUT_DIR is not under /lambda/nfs: $LEPT_OUTPUT_DIR"
    warn "  Large checkpoints can exhaust root volume; consider /lambda/nfs/<fs>/lept/runs/..."
fi

echo ""
echo "--- GPU checks ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tr '\n' '; ')
        pass "nvidia-smi OK ($GPU_INFO)"
    else
        fail "nvidia-smi command failed"
    fi
else
    fail "nvidia-smi not found in PATH"
fi

echo ""
echo "--- Python from venv ---"
if [[ -n "${LEPT_VENV:-}" && -f "$LEPT_VENV/bin/python" ]]; then
    VENV_PY="$LEPT_VENV/bin/python"
    PY_VER=$("$VENV_PY" --version 2>&1 || echo "unknown")
    pass "Python: $PY_VER"

    echo ""
    echo "--- Critical imports ---"
    for mod in torch vllm trl transformers datasets openenv jmespath; do
        if "$VENV_PY" -c "import $mod" >/dev/null 2>&1; then
            pass "import $mod"
        else
            fail "import $mod failed (run bootstrap or reinstall deps)"
        fi
    done

    CUDA_OK=$("$VENV_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo "0")
    if [[ "$CUDA_OK" == "1" ]]; then
        pass "torch.cuda.is_available() is True"
    else
        fail "torch.cuda.is_available() is False"
    fi

    echo ""
    echo "--- Multi-GPU / training layout ---"
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
        GPU_COUNT=0
    fi
    echo "  Visible GPUs: $GPU_COUNT"

    LEPT_VLLM_MODE="${LEPT_VLLM_MODE:-auto}"
    if [[ "$LEPT_VLLM_MODE" != "auto" && "$LEPT_VLLM_MODE" != "server" && "$LEPT_VLLM_MODE" != "colocate" ]]; then
        fail "LEPT_VLLM_MODE must be auto, server, or colocate (got: $LEPT_VLLM_MODE)"
    fi
    LEPT_VLLM_TP="${LEPT_VLLM_TP:-1}"
    RESOLVED_MODE="$LEPT_VLLM_MODE"
    if [[ "$RESOLVED_MODE" == "auto" ]]; then
        if [[ "$GPU_COUNT" -ge 2 ]]; then
            RESOLVED_MODE="server"
        else
            RESOLVED_MODE="colocate"
        fi
    fi
    echo "  Resolved vLLM mode: $RESOLVED_MODE (LEPT_VLLM_TP=$LEPT_VLLM_TP)"

    if [[ "$RESOLVED_MODE" == "server" ]]; then
        if [[ "$GPU_COUNT" -lt 2 ]]; then
            fail "vllm_mode=server requires >= 2 visible GPUs, found $GPU_COUNT"
        fi
        if [[ "$GPU_COUNT" -le "$LEPT_VLLM_TP" ]]; then
            fail "GPU count ($GPU_COUNT) must be > LEPT_VLLM_TP ($LEPT_VLLM_TP) for server mode"
        fi
    fi

    LEPT_BATCH_SIZE="${LEPT_BATCH_SIZE:-8}"
    LEPT_NUM_GENERATIONS="${LEPT_NUM_GENERATIONS:-8}"
    rem=$((LEPT_BATCH_SIZE % LEPT_NUM_GENERATIONS))
    if [[ "$rem" -ne 0 ]]; then
        fail "LEPT_BATCH_SIZE ($LEPT_BATCH_SIZE) must be divisible by LEPT_NUM_GENERATIONS ($LEPT_NUM_GENERATIONS) for GRPO"
    else
        pass "LEPT_BATCH_SIZE / LEPT_NUM_GENERATIONS divisibility OK ($LEPT_BATCH_SIZE / $LEPT_NUM_GENERATIONS)"
    fi

    if [[ "$GPU_COUNT" -ge 2 ]]; then
        if "$VENV_PY" -c "import torch.distributed" >/dev/null 2>&1; then
            pass "torch.distributed available"
        else
            fail "torch.distributed not importable (needed for multi-GPU training)"
        fi
    fi

    if [[ "$RESOLVED_MODE" == "server" ]]; then
        if [[ -x "${LEPT_VENV}/bin/accelerate" ]]; then
            pass "accelerate CLI in venv (server mode)"
        else
            fail "accelerate not found at ${LEPT_VENV}/bin/accelerate (required for vllm_mode=server)"
        fi
    fi
else
    fail "Cannot run Python checks because LEPT_VENV python is unavailable"
fi

echo ""
echo "--- Env endpoint ---"
if [[ -n "${ENV_BASE_URL:-}" ]]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ENV_BASE_URL/health" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass "ENV_BASE_URL /health returned 200"
    elif [[ "$HTTP_CODE" == "000" ]]; then
        fail "ENV_BASE_URL unreachable (timeout or DNS failure): $ENV_BASE_URL"
    else
        fail "ENV_BASE_URL /health returned HTTP $HTTP_CODE (expected 200)"
    fi
fi

echo ""
echo "--- Disk usage ---"
echo "  Home:"
df -h "$HOME" | sed -n '1,2p' || true
if [[ -d "$DATA_ROOT" ]]; then
    echo "  Data root:"
    df -h "$DATA_ROOT" | sed -n '1,2p' || true
fi

echo ""
echo "=== Summary: $FAIL failure(s), $WARN warning(s) ==="
if [[ $FAIL -gt 0 ]]; then
    echo "Fix failures before launching training."
    exit 1
fi
if [[ $WARN -gt 0 ]]; then
    echo "Warnings present -- review before launching."
fi
echo "Preflight complete."
