#!/usr/bin/env bash
# Trained-only overnight queue (no post-hoc logit scores).
# Waits for any current train_fresh_sc.py to finish, then runs sequential jobs.
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

GPU="${1:-0}"
QUEUE_LOG="./save/vast_trained_queue.log"
mkdir -p ./save
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "===== trained queue launcher $(date -Is) ====="
echo "waiting for any train_fresh_sc.py to exit..."
while pgrep -f '[t]rain_fresh_sc.py' >/dev/null; do
  sleep 60
done
echo "GPU free; starting trained-only jobs $(date -Is)"

# Only methods that TRAIN a score / objective (not post-hoc MSP/energy/...).
JOBS=(
  "softauc 42"    # CE + BCE + SoftAUC ranking of correct vs wrong
  "selnet 42"     # SelectiveNet coverage-constrained selection head
  "temptrain 42"  # learned temperature; CE(logits/T), score=MSP(logits/T)
  "dualaug 42"    # two-view CE; head predicts view agreement
  "focal 42"      # focal correctness BCE
  "tcp 42"        # soft TCP regression head
  "scsf 42"       # plain SCSF head baseline (trained BCE)
  "softauc 0"
  "selnet 0"
  "temptrain 0"
)

run_one() {
  local variant="$1" seed="$2"
  local out="./save/fresh_${variant}_cifar100_seed${seed}"
  if [[ -f "${out}/results.json" ]]; then
    echo "[skip] ${variant} seed=${seed} already has results.json"
    return 0
  fi
  echo "----- $(date -Is) START ${variant} seed=${seed} -> ${out} -----"
  pkill -f '[t]rain_fresh_sc.py' 2>/dev/null || true
  sleep 2
  bash ./run_cifar100_fresh_sc.sh "${variant}" "${seed}" "${GPU}"
  echo "----- $(date -Is) DONE  ${variant} seed=${seed} -----"
  if [[ -f "${out}/results.json" ]]; then
    "${PY}" - <<PY
import json
from pathlib import Path
r=json.loads(Path("${out}/results.json").read_text())
t=r["test"]; ce=t["coverage_errors"]
def g(k): return float(ce.get(k, ce.get(str(k))))
extra=""
if "temperature" in r: extra=f" T={r['temperature']:.3f}"
print(f"SUMMARY ${variant} seed=${seed}: acc={t['accuracy']:.2f} aurc={t['aurc']:.4f} @95={g(95):.2f} @90={g(90):.2f} @80={g(80):.2f} @10={g(10):.2f}{extra}")
PY
  fi
}

for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run_one ${job}
done

echo "===== trained queue finished $(date -Is) ====="
