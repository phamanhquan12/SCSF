# Reproduce SAGE-TopK Dynamic-K

This branch contains the complete round-3 launcher. It has no dependency on
the author's server paths. The launcher creates a portable manifest from the
paths supplied by the caller and runs the four registered seed-13 jobs:

- CIFAR-10 and CIFAR-100
- `pool` and `pool+conv` candidate sets
- dynamic K with `|mean| > 2.0 * SEM`

Clone the branch anywhere:

```bash
git clone --branch quan --single-branch \
  https://github.com/phamanhquan12/SCSF.git scsf
cd scsf
```

## Requirements

- Linux with a CUDA-capable GPU and `nvidia-smi`
- Python environment with the repository's existing PyTorch, torchvision, and
  Python dependencies installed
- CIFAR-10 and CIFAR-100 files available below one writable data directory
- enough disk for four training runs and their checkpoints

The protocol keeps `data.download=false` by default. Use `--download` only when
the data directory is writable and parallel torchvision downloads are safe.

## Run

From the repository root:

```bash
python scripts/run_sage_topk_dynamic.py \
  --data-root /path/to/cifar-data \
  --results-root /path/to/new/sage-topk-round3 \
  --python /path/to/venv/bin/python \
  --max-jobs 2
```

The launcher writes the manifest to
`<results-root>/manifests/sage_topk_round3_dynamic.tsv`, progress to
`<results-root>/progress.tsv`, and per-run logs to `<results-root>/logs/`.
`--max-jobs 1` is safer on smaller GPUs. Before the long run, validate paths
and resolved run names without starting training:

```bash
python scripts/run_sage_topk_dynamic.py \
  --data-root /path/to/cifar-data \
  --results-root /path/to/new/sage-topk-round3 \
  --python /path/to/venv/bin/python \
  --dry-run
```

The launcher sets `SCSF_SOURCE_COMMIT` to the current checkout commit and
exports the repository on `PYTHONPATH`. It does not delete existing run
artifacts, but the scheduler resumes incomplete run directories and skips
registry-complete runs by design. Use a fresh results root for a clean pilot.
