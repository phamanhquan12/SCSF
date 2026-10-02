#!/usr/bin/env bash
# dualaug CelebA seeds after CIFAR dualaug seed3 finishes.
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
LOG="./save/dualaug_celeba_queue.log"
mkdir -p ./save
exec > >(tee -a "${LOG}") 2>&1

echo "===== dualaug celeba queue $(date -Is) ====="
echo "waiting for train_fresh_sc.py (CIFAR dualaug seed3) to finish..."
while pgrep -f '[t]rain_fresh_sc.py' >/dev/null; do
  sleep 60
done
echo "CIFAR free $(date -Is); starting CelebA dualaug"

# Confirm CelebA is present (VAST uses /workspace/data via train_acccon_large.data_root)
ATTR="$("${PY}" - <<'PY'
import os
from train_acccon_large import data_root
p=os.path.join(data_root(),"celeba","list_attr_celeba.txt")
print(p)
print("ok" if os.path.isfile(p) else "missing")
PY
)"
echo "celeba check: ${ATTR}"
echo "${ATTR}" | tail -1 | grep -q ok

SEEDS=(42 0 1 2 3)
for seed in "${SEEDS[@]}"; do
  out="./save/fresh_dualaug_celeba_seed${seed}"
  if [[ -f "${out}/results.json" ]]; then
    echo "[skip] dualaug celeba seed=${seed}"
    continue
  fi
  echo "----- $(date -Is) START dualaug celeba seed=${seed} -----"
  PYTHONUNBUFFERED=1 "$PY" train_dualaug_celeba.py \
    --seed "${seed}" \
    --gpu "${GPU}" \
    --workers 4 \
    --output-dir "${out}" \
    2>&1 | tee "./save/fresh_dualaug_celeba_seed${seed}.log"
  echo "----- $(date -Is) DONE dualaug celeba seed=${seed} -----"
  if [[ -f "${out}/results.json" ]]; then
    "${PY}" - <<PY
import json
from pathlib import Path
r=json.loads(Path("${out}/results.json").read_text())
t=r["test"]; ce=t["coverage_errors"]
def g(k): return float(ce.get(k, ce.get(str(k))))
print(f"SUMMARY celeba dualaug seed=${seed} ep={r.get('selected_epoch')}: acc={t['accuracy']:.2f} aurc={t['aurc']:.4f} @95={g(95):.2f} @90={g(90):.2f} @80={g(80):.2f} @10={g(10):.2f}")
PY
  fi
done

echo "===== dualaug celeba queue finished $(date -Is) ====="
