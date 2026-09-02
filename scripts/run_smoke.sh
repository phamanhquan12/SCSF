#!/usr/bin/env bash
#
# Full-pipeline smoke: builds every method, trains a 2-epoch overfit run,
# evaluates the selected checkpoint, and aggregates the registry. Passing is
# a prerequisite for any real experiment.
#
set -euo pipefail

cd "$(dirname "$0")/.."
export SCSF_DATA_ROOT="${SCSF_DATA_ROOT:-/mnt/c/Users/ADMIN/data}"
PY=./.venv/bin/python

smoke_root=/tmp/scsf_smoke_$$
rm -rf "$smoke_root"
trap 'rm -rf "$smoke_root"' EXIT

for m in ce dg selectivenet sat scsf ccl_sc; do
  SCSF_DATA_ROOT="$SCSF_DATA_ROOT" "$PY" -m scsf.train \
    dataset=cifar10 backbone=resnet18 method_name="$m" seed=13 recipe=singlerun \
    results_root="$smoke_root" train.epochs=2 train.overfit=512 \
    train.batch_size=32 train.eval_every=1 data.num_workers=0 \
    train.lr=0.05 train.scheduler=cosine
done

for run in "$smoke_root"/*/; do
  "$PY" -m scsf.evaluate run_dir="${run%/}" split=val
done

"$PY" -m scsf.aggregate "$smoke_root/registry.csv" "$smoke_root/registry_agg.csv"
echo "smoke OK: registry + aggregate under $smoke_root"