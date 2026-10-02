#!/usr/bin/env bash
# Isolated query-weighted acccon follow-ups on CIFAR-100 (seed 42).
# Wait for any in-flight trainer, then run floor / small-β bound / tail.
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

echo "waiting for any train_next_scsf.py process to exit..."
while pgrep -f '[t]rain_next_scsf.py' >/dev/null; do
  sleep 60
done
echo "no trainer running; starting acccon sweep"

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
  --con-weight 0.5
  --con-temperature 0.1
  --soft-temperature 0.2
  --queue-size 3000
)

run "$PY" train_next_scsf.py --variant acccon_floor "${common[@]}" \
  --query-floor 0.20 \
  --output-dir ./save/next_acccon_floor_cifar100_seed42

run "$PY" train_next_scsf.py --variant acccon_bound "${common[@]}" \
  --boundary-beta 0.10 \
  --output-dir ./save/next_acccon_bound_b010_cifar100_seed42

run "$PY" train_next_scsf.py --variant acccon_tail "${common[@]}" \
  --tail-weight 0.10 \
  --output-dir ./save/next_acccon_tail_cifar100_seed42
