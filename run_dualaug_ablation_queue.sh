#!/usr/bin/env bash
# dualaug ablation on CIFAR-100 (VGG16-BN, 300 ep, last.pth, official 10k).
# Runs up to PARALLEL jobs at once on one GPU; skips jobs that have results.json.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x /workspace/SCSF/.venv/bin/python ]]; then
  PY=/workspace/SCSF/.venv/bin/python
else
  PY=.venv/bin/python
fi
GPU="${1:-0}"
PARALLEL="${2:-3}"
QUEUE_LOG="./save/dualaug_ablation_queue.log"
mkdir -p ./save
exec > >(tee -a "${QUEUE_LOG}") 2>&1

# Seed 42 of every ablation first, then seed 0.
JOBS=(
  "dualce_msp 42"
  "dualce_scsf 42"
  "dualaug_detach 42"
  "agree_only 42"
  "msp 42"
  "dualaug 42"
  "dualce_msp 0"
  "dualce_scsf 0"
  "dualaug_detach 0"
  "agree_only 0"
  "msp 0"
  "scsf 0"
)

run_one() {
  local variant="$1" seed="$2"
  local out="./save/fresh_${variant}_cifar100_seed${seed}"
  local log="./save/fresh_${variant}_cifar100_seed${seed}.log"
  if [[ -f "${out}/results.json" ]]; then
    echo "[skip] ${variant} seed=${seed}"
    return 0
  fi
  echo "----- $(date -Is) START ${variant} seed=${seed} -----"
  PYTHONUNBUFFERED=1 "$PY" train_fresh_sc.py \
    --variant "${variant}" \
    -d cifar100 \
    --epochs 300 \
    --pretrain 100 \
    --ramp-epochs 20 \
    --batch-size 128 \
    --seed "${seed}" \
    --workers 4 \
    --gpu "${GPU}" \
    --output-dir "${out}" \
    > "${log}" 2>&1 || echo "${variant} seed=${seed} FAILED"
  echo "----- $(date -Is) DONE ${variant} seed=${seed} -----"
  grep -E "Evaluated|scored by MSP" "${log}" | sed "s/^/  ${variant} seed=${seed}: /" || true
}

echo "===== dualaug ablation queue $(date -Is) parallel=${PARALLEL} ====="
for job in "${JOBS[@]}"; do
  while (( $(jobs -rp | wc -l) >= PARALLEL )); do
    wait -n || true
  done
  # shellcheck disable=SC2086
  run_one ${job} &
  sleep 5
done
wait
echo "===== dualaug ablation queue finished $(date -Is) ====="
