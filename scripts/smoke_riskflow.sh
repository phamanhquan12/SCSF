#!/usr/bin/env bash
# Run the RiskFlow smoke (tiny overfit ResNet-18 + DeiT-S) against the server
# training repo. Produces trace npz, redundancy report, trajectory plots and an
# overhead JSON for each backbone under SCSF_RISKFLOW_SMOKE_ROOT.
#
#   bash /mnt/d/OPD/scsf/scripts/smoke_riskflow.sh
#
set -euo pipefail
cd "$(dirname "$0")"
cd /root/scsf

PY=/root/scsf_venv/bin/python
DATA="${SCSF_DATA_ROOT:-/root/scsf/data}"
ROOT="${SCSF_RISKFLOW_SMOKE_ROOT:-/root/scsf_scratch/riskflow_smoke}"
mkdir -p "$ROOT"

for bb in resnet18 deit_s; do
  "$PY" /mnt/d/OPD/scsf/scripts/smoke_riskflow.py \
    --backbone "$bb" --results-root "$ROOT" --data-root "$DATA" \
    --device cuda --epochs 1 --overfit 64 --trace-subset 128
done

echo
echo "RiskFlow smoke OK. Artifacts under $ROOT"
