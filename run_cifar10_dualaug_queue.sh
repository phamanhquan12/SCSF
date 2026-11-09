#!/usr/bin/env bash
# dualaug on CIFAR-10, same protocol as CIFAR-100 (VGG16-BN, 300 ep, last.pth, official 10k).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
GPU="${1:-0}"
QUEUE_LOG="./save/dualaug_cifar10_queue.log"
mkdir -p ./save
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "===== dualaug cifar10 queue $(date -Is) ====="
SEEDS=(42 0 1 2 3)
for seed in "${SEEDS[@]}"; do
  out="./save/fresh_dualaug_cifar10_seed${seed}"
  if [[ -f "${out}/results.json" ]]; then
    echo "[skip] dualaug cifar10 seed=${seed}"
    continue
  fi
  echo "----- $(date -Is) START dualaug cifar10 seed=${seed} -----"
  PYTHONUNBUFFERED=1 "$PY" train_fresh_sc.py \
    --variant dualaug \
    -d cifar10 \
    --epochs 300 \
    --pretrain 100 \
    --ramp-epochs 20 \
    --batch-size 128 \
    --seed "${seed}" \
    --workers 4 \
    --gpu "${GPU}" \
    --output-dir "${out}" \
    > "./save/fresh_dualaug_cifar10_seed${seed}.log" 2>&1 || echo "seed=${seed} FAILED"
  grep "Evaluated" "./save/fresh_dualaug_cifar10_seed${seed}.log" || true
  echo "----- $(date -Is) DONE dualaug cifar10 seed=${seed} -----"
done
echo "===== dualaug cifar10 queue finished $(date -Is) ====="
