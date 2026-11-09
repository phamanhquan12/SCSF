#!/usr/bin/env bash
# Fresh SC methods (no acccon). See train_fresh_sc.py for variant list.
set -euo pipefail
cd "$(dirname "$0")"
VARIANT="${1:?variant required}"
SEED="${2:-42}"
GPU="${3:-0}"
if [[ -x /workspace/SCSF/.venv/bin/python ]]; then
  PY=/workspace/SCSF/.venv/bin/python
elif [[ -x /venv/main/bin/python ]]; then
  PY=/venv/main/bin/python
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  echo "no project venv found" >&2
  exit 1
fi

OUT="./save/fresh_${VARIANT}_cifar100_seed${SEED}"
LOG="./save/fresh_${VARIANT}_cifar100_seed${SEED}.log"
mkdir -p ./save

echo "starting fresh ${VARIANT} seed=${SEED} gpu=${GPU} -> ${OUT}"
PYTHONUNBUFFERED=1 "$PY" train_fresh_sc.py \
  --variant "${VARIANT}" \
  -d cifar100 \
  --epochs 300 \
  --pretrain 100 \
  --ramp-epochs 20 \
  --batch-size 128 \
  --seed "${SEED}" \
  --workers 4 \
  --gpu "${GPU}" \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG}"
