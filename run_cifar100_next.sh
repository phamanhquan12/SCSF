#!/usr/bin/env bash
# Sequential CIFAR-100 next-wave search. One GPU; last.pth is official.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

run() {
  echo "===== $* ====="
  PYTHONUNBUFFERED=1 "$@"
}

# Strongest combined hypothesis first: keep micro, add RC-weighted features.
run "$PY" train_next_scsf.py -d cifar100 --variant micro_acccon \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --con-weight 0.5 --queue-size 3000 --seed 42 --workers 4 \
  --output-dir ./save/next_micro_acccon_cifar100_seed42

run "$PY" train_next_scsf.py -d cifar100 --variant micro_leftcon \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --con-weight 0.5 --queue-size 3000 --seed 42 --workers 4 \
  --output-dir ./save/next_micro_leftcon_cifar100_seed42

run "$PY" train_next_scsf.py -d cifar100 --variant acccon \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --con-weight 0.5 --queue-size 3000 --seed 42 --workers 4 \
  --output-dir ./save/next_acccon_cifar100_seed42

run "$PY" train_next_scsf.py -d cifar100 --variant micro_ace \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --acceptce-weight 0.5 --seed 42 --workers 4 \
  --output-dir ./save/next_micro_ace_cifar100_seed42

run "$PY" train_next_scsf.py -d cifar100 --variant micro_hi \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --seed 42 --workers 4 \
  --output-dir ./save/next_micro_hi_cifar100_seed42
