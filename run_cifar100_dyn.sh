#!/usr/bin/env bash
# Sequential CIFAR-100 runs for recipe sections 7-15. One GPU; last.pth is official.
# Isolated variants only. Combinations wait until a module shows a complementary gain.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

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

# §8 T1 / §15 T1
run "$PY" train_next_scsf.py --variant temporal_target "${common[@]}" \
  --output-dir ./save/dyn_temporal_target_cifar100_seed42

# §8 hybrid
run "$PY" train_next_scsf.py --variant temporal_hybrid "${common[@]}" \
  --output-dir ./save/dyn_temporal_hybrid_cifar100_seed42

# §9A forgetting-modified temporal target
run "$PY" train_next_scsf.py --variant forget_target "${common[@]}" \
  --forget-gamma 0.5 \
  --output-dir ./save/dyn_forget_target_cifar100_seed42

# §12A / §15 T4 margin reliability target
run "$PY" train_next_scsf.py --variant margin_target "${common[@]}" \
  --margin-temperature 1.0 \
  --output-dir ./save/dyn_margin_target_cifar100_seed42

# §11 cartography-aware asymmetric ACCon
run "$PY" train_next_scsf.py --variant carto_acccon "${common[@]}" \
  --ambiguity-beta 0.10 --stability-gamma 2.0 --queue-size 3000 \
  --output-dir ./save/dyn_carto_acccon_cifar100_seed42

# §13 EL2N-weighted ACCon
run "$PY" train_next_scsf.py --variant el2n_acccon "${common[@]}" \
  --el2n-start 10 --el2n-end 20 --el2n-beta 0.10 --queue-size 3000 \
  --output-dir ./save/dyn_el2n_acccon_cifar100_seed42

# §14 diagnostic probes
run "$PY" train_next_scsf.py --variant depth_diag "${common[@]}" \
  --output-dir ./save/dyn_depth_diag_cifar100_seed42

# §14.3C append discrete disagreement
run "$PY" train_next_scsf.py --variant depth_score "${common[@]}" \
  --output-dir ./save/dyn_depth_score_cifar100_seed42

# §15 temporal + depth distillation
run "$PY" train_next_scsf.py --variant temporal_depth "${common[@]}" \
  --output-dir ./save/dyn_temporal_depth_cifar100_seed42
