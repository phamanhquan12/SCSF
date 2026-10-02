#!/usr/bin/env bash
# DSS-ACCon: live DSN aux CE + SCSF + detached depth disagreement + qw acccon.
# CIFAR-100 seed 42, official 10k, unique output dir (does not overwrite baselines).
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
echo "no trainer running; starting dss_acccon"

PYTHONUNBUFFERED=1 "$PY" train_next_scsf.py \
  --variant dss_acccon \
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
  --output-dir ./save/next_dss_acccon_cifar100_seed42
