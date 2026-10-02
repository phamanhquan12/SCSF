#!/usr/bin/env bash
# Isolated architectural acccon variant: DS-SCSF fused score + query-weighted acccon.
# Wait for any in-flight trainer (including CelebA), then train CIFAR-100 seed 42.
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

echo "waiting for any train_next_scsf.py / train_acccon_large.py process to exit..."
while pgrep -f '[t]rain_next_scsf.py|[t]rain_acccon_large.py' >/dev/null; do
  sleep 60
done
echo "no trainer running; starting acccon_ds"

PYTHONUNBUFFERED=1 "$PY" train_next_scsf.py \
  --variant acccon_ds \
  -d cifar100 \
  --epochs 300 \
  --pretrain 100 \
  --ramp-epochs 20 \
  --batch-size 128 \
  --seed 42 \
  --workers 4 \
  --con-weight 0.5 \
  --con-temperature 0.1 \
  --soft-temperature 0.2 \
  --queue-size 3000 \
  --output-dir ./save/next_acccon_ds_cifar100_seed42
