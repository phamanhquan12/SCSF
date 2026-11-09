#!/usr/bin/env bash
# Sequential CIFAR-100 search against CCL-SC. One GPU; last.pth is official.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

run() {
  echo "===== $* ====="
  PYTHONUNBUFFERED=1 "$@"
}

run "$PY" train_search_scsf.py -d cifar100 --variant rank_csc \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --rank-weight 0.5 --csc-weight 0.5 --queue-size 3000 --seed 42 --workers 4 \
  --output-dir ./save/search_rank_csc_cifar100_seed42

run "$PY" train_search_scsf.py -d cifar100 --variant csc \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --csc-weight 0.5 --queue-size 3000 --seed 42 --workers 4 \
  --output-dir ./save/search_csc_cifar100_seed42

run "$PY" train_search_scsf.py -d cifar100 --variant rank \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --rank-weight 0.5 --seed 42 --workers 4 \
  --output-dir ./save/search_rank_cifar100_seed42

run "$PY" train_search_scsf.py -d cifar100 --variant scsf \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --seed 42 --workers 4 \
  --output-dir ./save/search_scsf_cifar100_seed42

run "$PY" train_cbr_scsf.py -d cifar100 --confusion-groups coarse \
  --epochs 300 --pretrain 100 --cbr-ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --confusion-weight 0.5 --coverage-floor-ratio 0.8 \
  --seed 42 --workers 4 \
  --output-dir ./save/cbr_coarse_cifar100_seed42

run "$PY" train_cbr_scsf.py -d cifar100 \
  --epochs 300 --pretrain 100 --cbr-ramp-epochs 20 --batch-size 128 \
  --micro-weight 0.5 --confusion-weight 0.0 --dual-max 0.0 \
  --seed 42 --workers 4 \
  --output-dir ./save/cbr_micro_cifar100_seed42
