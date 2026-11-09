#!/usr/bin/env bash
# Resume the CIFAR-100 dynamics queue from the unfinished variants only.
set -euo pipefail
cd "$(dirname "$0")"
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

run() {
  echo "===== $* ====="
  PYTHONUNBUFFERED=1 "$@"
}

common=(
  -d cifar100
  --epochs 300
  --pretrain 100
  --ramp-epochs 20
  --batch-size 128
  --seed 42
  --workers 4
  --dyn-window 20
  --dyn-start-epoch 101
  --dynamics-mode eval
)

# Restarted: previous local run was killed at epoch 8.
run "$PY" train_next_scsf.py --variant carto_acccon "${common[@]}" \
  --ambiguity-beta 0.10 --stability-gamma 2.0 --queue-size 3000 \
  --output-dir ./save/dyn_carto_acccon_cifar100_seed42

run "$PY" train_next_scsf.py --variant el2n_acccon "${common[@]}" \
  --el2n-start 10 --el2n-end 20 --el2n-beta 0.10 --queue-size 3000 \
  --output-dir ./save/dyn_el2n_acccon_cifar100_seed42

run "$PY" train_next_scsf.py --variant depth_diag "${common[@]}" \
  --output-dir ./save/dyn_depth_diag_cifar100_seed42

run "$PY" train_next_scsf.py --variant depth_score "${common[@]}" \
  --output-dir ./save/dyn_depth_score_cifar100_seed42

run "$PY" train_next_scsf.py --variant temporal_depth "${common[@]}" \
  --output-dir ./save/dyn_temporal_depth_cifar100_seed42
