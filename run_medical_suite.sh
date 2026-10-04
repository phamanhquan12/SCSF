#!/usr/bin/env bash
# =============================================================================
# run_medical_suite.sh
# Single-command launcher for the medical selective classification suite.
#
# Usage:
#   bash run_medical_suite.sh [options]
#
# SSH background examples:
#   nohup bash run_medical_suite.sh --gpu 0 > logs/run.log 2>&1 &
#   screen -S scsf bash run_medical_suite.sh --gpu 0
#   tmux new-session -d -s scsf "bash run_medical_suite.sh --gpu 0"
#
# Default: thesis datasets (datasets_thesis.md) x {sr, ccl_sc (official), sat, dg,
# selectivenet, scsf, dualaug} x training seeds {0, 1, 2}, each evaluated on the full
# test set; tables/ reports mean/std across training seeds.
#
# Resume after interruption (reuse the run id printed at launch):
#   bash run_medical_suite.sh --skip-existing --run-id 20261005_030000 --gpu 0
#
# Seeds / CCL-SC variants (mean and std are taken across training seeds):
#   bash run_medical_suite.sh --seeds 42 0 1 --ccl-variants official paper
#
# Smoke test:
#   bash run_medical_suite.sh --smoke-train-samples 64 --smoke-eval-samples 32 \
#       --epochs 3 --pretrain 1 --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$LOG_DIR/launch_${RUN_TS}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LAUNCH_LOG"
}

log "=== Medical suite launcher started (PID $$) ==="
log "Working directory: $SCRIPT_DIR"
log "Launch log: $LAUNCH_LOG"

# --------------------------------------------------------------------------- #
# Python / virtualenv detection
# --------------------------------------------------------------------------- #
setup_and_find_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "$VIRTUAL_ENV/bin/python"; return; fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then echo "$CONDA_PREFIX/bin/python"; return; fi
    
    local VENV_DIR="$SCRIPT_DIR/.venv"
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log "Virtual environment not found at $VENV_DIR. Creating..." >&2
        if ! command -v python3 &>/dev/null; then
            log "ERROR: python3 is not installed on the system." >&2
            exit 1
        fi
        python3 -m venv "$VENV_DIR" || {
            log "Failed to create venv. Attempting to install python3-venv..." >&2
            if [[ $EUID -eq 0 ]] && command -v apt-get &>/dev/null; then
                apt-get update -yqq && apt-get install -y python3-venv >&2
                python3 -m venv "$VENV_DIR" || {
                    log "ERROR: Still failed to create venv." >&2
                    exit 1
                }
            else
                log "ERROR: Failed to create venv. Is python3-venv installed?" >&2
                exit 1
            fi
        }
    fi

    # Check if dependencies are actually installed
    if ! "$VENV_DIR/bin/python" -c "import torch, kaggle" &>/dev/null; then
        log "Dependencies missing in venv. Installing requirements..." >&2
        "$VENV_DIR/bin/pip" install --upgrade pip >&2
        log "Installing PyTorch with CUDA support..." >&2
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 >&2
        "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" kaggle >&2
    fi

    echo "$VENV_DIR/bin/python"
}

PYTHON="$(setup_and_find_python)"
if [[ -z "$PYTHON" ]]; then
    log "ERROR: No Python interpreter found."
    exit 1
fi
log "Python: $PYTHON ($($PYTHON --version 2>&1))"

# --------------------------------------------------------------------------- #
# Kaggle credentials check
# --------------------------------------------------------------------------- #
KAGGLE_JSON="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    log "WARNING: kaggle.json not found at $KAGGLE_JSON"
    log "         Datasets will fail to download unless credentials are present."
    log "         Place your kaggle.json at $KAGGLE_JSON and re-run."
else
    chmod 600 "$KAGGLE_JSON"
    log "Kaggle credentials: $KAGGLE_JSON (OK)"
fi

# --------------------------------------------------------------------------- #
# Defaults (override via CLI args)
# --------------------------------------------------------------------------- #
DATASETS_FILE="$SCRIPT_DIR/datasets_thesis.md"
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_ROOT="$SCRIPT_DIR/results/paper/medical_suite"
METHODS=(sr ccl_sc sat dg selectivenet scsf dualaug)
ARCH="resnet50"
EPOCHS=100
PRETRAIN=20
BATCH_SIZE=64
EVAL_BATCH_SIZE=128
WORKERS=4
LR=0.01
LR_GAMMA=0.1
SEEDS=(0 1 2)
CCL_VARIANTS=(official)
RUN_ID="$RUN_TS"
GPU=""
INPUT_SIZE=224
DOWNLOAD_FLAG="--download"
EXTRA_ARGS=()

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)            GPU="$2"; shift 2 ;;
        --gpu=*)          GPU="${1#--gpu=}"; shift ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --pretrain)       PRETRAIN="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --workers)        WORKERS="$2"; shift 2 ;;
        --seeds)
            shift; SEEDS=()
            while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
                SEEDS+=("$1"); shift
            done
            ;;
        --ccl-variants)
            shift; CCL_VARIANTS=()
            while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
                CCL_VARIANTS+=("$1"); shift
            done
            ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --results-root)   RESULTS_ROOT="$2"; shift 2 ;;
        --run-id)         RUN_ID="$2"; shift 2 ;;
        --datasets-file)  DATASETS_FILE="$2"; shift 2 ;;
        --no-download)    DOWNLOAD_FLAG=""; shift ;;
        --methods)
            shift; METHODS=()
            while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
                METHODS+=("$1"); shift
            done
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --------------------------------------------------------------------------- #
# Build command
# --------------------------------------------------------------------------- #
CMD=(
    "$PYTHON" "$SCRIPT_DIR/run_paper_medical_suite.py"
    "--datasets-file"   "$DATASETS_FILE"
    "--data-dir"        "$DATA_DIR"
    "--results-root"    "$RESULTS_ROOT"
    "--run-id"          "$RUN_ID"
    "--methods"         "${METHODS[@]}"
    "--arch"            "$ARCH"
    "--input-size"      "$INPUT_SIZE"
    "--epochs"          "$EPOCHS"
    "--pretrain"        "$PRETRAIN"
    "--batch-size"      "$BATCH_SIZE"
    "--eval-batch-size" "$EVAL_BATCH_SIZE"
    "--workers"         "$WORKERS"
    "--lr"              "$LR"
    "--milestones"      40 70 90
    "--lr-gamma"        "$LR_GAMMA"
    "--seeds"           "${SEEDS[@]}"
    "--ccl-variants"    "${CCL_VARIANTS[@]}"
    "--pretrained"
)
[[ -n "$GPU" ]]           && CMD+=("--gpu" "$GPU")
[[ -n "$DOWNLOAD_FLAG" ]] && CMD+=("$DOWNLOAD_FLAG")
CMD+=("${EXTRA_ARGS[@]}")

# --------------------------------------------------------------------------- #
# Print and run
# --------------------------------------------------------------------------- #
log "Methods : ${METHODS[*]}"
log "Seeds   : ${SEEDS[*]}"
log "CCL-SC  : ${CCL_VARIANTS[*]}"
log "Datasets: $DATASETS_FILE"
log "Results : $RESULTS_ROOT/$RUN_ID"
log "Command : ${CMD[*]}"
log ""

exec > >(tee -a "$LAUNCH_LOG") 2>&1

"${CMD[@]}"

log ""
log "=== Medical suite COMPLETED ==="
log "Results at: $RESULTS_ROOT/$RUN_ID"
