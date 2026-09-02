#!/usr/bin/env bash
#
# DepthFrag smoke: tiny overfit ResNet-18 AND DeiT-S runs + frozen-checkpoint
# extraction (raw-profile npz, correlation ladder, distilled-score artifacts).
# Run on the GPU host (server). Safe to re-run; scratch goes to
# SCSF_SMOKE_ROOT (default /root/scsf_scratch/depthfrag_smoke), NOT into git.
#
#   bash scripts/smoke_depthfrag.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
export SCSF_DATA_ROOT="${SCSF_DATA_ROOT:-/root/scsf/data}"
PY="${PY:-./.venv/bin/python}"
ROOT="${SCSF_SMOKE_ROOT:-/root/scsf_scratch/depthfrag_smoke}"
mkdir -p "$ROOT"

"$PY" scripts/smoke_depthfrag.py --backbone resnet18 --results-root "$ROOT" \
  --data-root "$SCSF_DATA_ROOT" --device cuda --epochs 1 --overfit 16
"$PY" scripts/smoke_depthfrag.py --backbone deit_s --results-root "$ROOT" \
  --data-root "$SCSF_DATA_ROOT" --device cuda --epochs 1 --overfit 16

echo
echo "DepthFrag smoke OK. Artifacts under $ROOT"
echo "  raw profiles:  <run>/depthfrag/profiles_{val,test}.npz"
echo "  ladder :       <run>/depthfrag/metrics.json + scores_{val,test}.csv"