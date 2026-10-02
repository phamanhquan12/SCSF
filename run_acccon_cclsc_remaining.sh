#!/usr/bin/env bash
# Query-weighted acccon on the two remaining CCL-SC paper datasets.
# CelebA fits this instance. ImageNet (~140GB) is skipped unless the data is already there.
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

run() {
  echo "===== $* ====="
  PYTHONUNBUFFERED=1 "$@"
}

echo "waiting for any train_next_scsf.py / train_acccon_large.py process to exit..."
while pgrep -f '[t]rain_next_scsf.py|[t]rain_acccon_large.py' >/dev/null; do
  sleep 60
done

run "$PY" prepare_celeba.py
run "$PY" train_acccon_large.py -d celeba --seed 42 --workers 4 \
  --output-dir ./save/next_acccon_celeba_seed42_qw

if [[ -d /workspace/data/imagenet/train && -d /workspace/data/imagenet/val ]]; then
  run "$PY" train_acccon_large.py -d imagenet --seed 42 --workers 4 \
    --output-dir ./save/next_acccon_imagenet_seed42_qw
else
  echo "SKIP imagenet: /workspace/data/imagenet/{train,val} not present"
  echo "This overlay has ~89GB free; ImageNet needs ~140GB. Use a persistent volume."
fi
