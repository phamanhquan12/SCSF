#!/usr/bin/env bash
#
# Seed sweep: train <method> over SEEDS seeds, evaluate, aggregate.
# Usage: scripts/run_sweep.sh <method> [seeds="7 13 21"] [results_root=results]
#
set -euo pipefail

cd "$(dirname "$0")/.."
export SCSF_DATA_ROOT="${SCSF_DATA_ROOT:-/mnt/c/Users/ADMIN/data}"
PY=./.venv/bin/python

method="${1:?usage: run_sweep.sh <method> [seeds] [results_root]}"
seeds="${2:-7 13 21}"
results_root="${3:-results}"

for seed in $seeds; do
  "$PY" -m scsf.train dataset=cifar10 backbone=resnet18 method_name="$method" \
    seed="$seed" recipe=singlerun results_root="$results_root"
  "$PY" -m scsf.evaluate \
    run_dir="$results_root/cifar10-resnet18-$method-rsinglerun-s$seed" split=val
done

"$PY" -m scsf.aggregate "$results_root/registry.csv"
echo "sweep done: $method seeds=[$seeds]"