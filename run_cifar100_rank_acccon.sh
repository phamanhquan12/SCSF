#!/usr/bin/env bash
# HardRank-ACCon: qw acccon + hard-mined pairwise ranking (recipe §5).
# CIFAR-100, official 10k, unique output dir (does not overwrite baselines).
set -euo pipefail
cd "$(dirname "$0")"
SEED="${1:-42}"
GPU="${2:-0}"
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

OUT="./save/next_rank_acccon_cifar100_seed${SEED}"
LOG="./save/rank_acccon_cifar100_seed${SEED}.log"
mkdir -p ./save

echo "starting rank_acccon seed=${SEED} gpu=${GPU} -> ${OUT}"
PYTHONUNBUFFERED=1 "$PY" train_next_scsf.py \
  --variant rank_acccon \
  -d cifar100 \
  --epochs 300 \
  --pretrain 100 \
  --ramp-epochs 20 \
  --batch-size 128 \
  --seed "${SEED}" \
  --workers 4 \
  --gpu "${GPU}" \
  --con-weight 0.5 \
  --con-temperature 0.1 \
  --soft-temperature 0.2 \
  --queue-size 3000 \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG}"
