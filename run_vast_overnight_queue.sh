#!/usr/bin/env bash
# Sequential overnight queue for VAST (one GPU). Skips jobs that already have results.json.
# ~1.2–1.5h per 300-epoch CIFAR-100 run on RTX 3090 → 7 jobs ≈ 9–11h.
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
QUEUE_LOG="./save/vast_overnight_queue.log"
mkdir -p ./save
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "===== overnight queue start $(date -Is) gpu=${GPU} ====="

# Order was post-hoc-heavy; kept for reference. Prefer run_vast_trained_queue.sh.
# Format: VARIANT SEED
JOBS=(
  "valblend 42"   # semi post-hoc: train SCSF + val-tuned λ (already started)
  "softauc 42"
  "selnet 42"
  "temptrain 42"
  "dualaug 42"
  "focal 42"
  "tcp 42"
  "scsf 42"
)

run_one() {
  local variant="$1" seed="$2"
  local out="./save/fresh_${variant}_cifar100_seed${seed}"
  if [[ -f "${out}/results.json" ]]; then
    echo "[skip] ${variant} seed=${seed} already has results.json"
    return 0
  fi
  echo "----- $(date -Is) START ${variant} seed=${seed} -> ${out} -----"
  # Kill any leftover trainer before starting (safety).
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
if "valblend" in r: extra=f" λ={r['valblend']['lambda']}"
print(f"SUMMARY ${variant} seed=${seed}: acc={t['accuracy']:.2f} aurc={t['aurc']:.4f} @95={g(95):.2f} @90={g(90):.2f} @80={g(80):.2f} @10={g(10):.2f}{extra}")
PY
  else
    echo "[warn] missing results.json for ${variant} seed=${seed}"
  fi
}

for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run_one ${job}
done

echo "===== overnight queue finished $(date -Is) ====="
