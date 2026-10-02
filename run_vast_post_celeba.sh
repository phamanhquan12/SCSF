#!/usr/bin/env bash
# After the in-flight CelebA qw acccon job, keep the 3090 busy with the
# remaining paper-comparison runs. Does not retrain local acccon_ds (CIFAR-100)
# and does not overwrite existing seed-42 CelebA / acccon baselines.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x /workspace/SCSF/.venv/bin/python ]]; then
  PY=/workspace/SCSF/.venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  echo "no project venv found" >&2
  exit 1
fi

echo "waiting for any train_next_scsf.py / train_acccon_large.py / run_acccon_cclsc_remaining.sh to exit..."
while pgrep -f '[t]rain_next_scsf.py|[t]rain_acccon_large.py|[r]un_acccon_cclsc_remaining.sh' >/dev/null; do
  sleep 60
done
echo "no trainer running; starting post-CelebA queue"

run() {
  local out=$1
  shift
  if [[ -f "${out}/results.json" ]]; then
    echo "SKIP already done: $out"
    return 0
  fi
  echo "===== $* -> $out ====="
  PYTHONUNBUFFERED=1 "$@" --output-dir "$out"
}

# 1) CelebA qw acccon extra seeds (seed 42 is already running / done).
for seed in 0 1 2 3; do
  run "./save/next_acccon_celeba_seed${seed}_qw" \
    "$PY" train_acccon_large.py -d celeba --seed "$seed" --workers 4
done

# 2) CIFAR-100 qw acccon extra seeds on official 10k (seed 42 already exists).
cifar100_common=(
  train_next_scsf.py --variant acccon -d cifar100
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128
  --workers 4 --con-weight 0.5 --con-temperature 0.1
  --soft-temperature 0.2 --queue-size 3000
)
for seed in 0 1 2 3; do
  run "./save/next_acccon_cifar100_seed${seed}_official" \
    "$PY" "${cifar100_common[@]}" --seed "$seed"
done

# 3) CIFAR-10 official 10k for the winning method and the DS architecture.
#    Local CIFAR-10 acccon was split-8k; local acccon_ds is CIFAR-100 only.
run "./save/next_acccon_cifar10_seed42_official" \
  "$PY" train_next_scsf.py --variant acccon -d cifar10 \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --seed 42 --workers 4 --con-weight 0.5 --con-temperature 0.1 \
  --soft-temperature 0.2 --queue-size 3000

run "./save/next_acccon_ds_cifar10_seed42" \
  "$PY" train_next_scsf.py --variant acccon_ds -d cifar10 \
  --epochs 300 --pretrain 100 --ramp-epochs 20 --batch-size 128 \
  --seed 42 --workers 4 --con-weight 0.5 --con-temperature 0.1 \
  --soft-temperature 0.2 --queue-size 3000

echo "post-CelebA queue finished"
