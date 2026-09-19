# Experiment suites

Portable, torch-free planning uses `scsf.engine.config.resolve(..., resolve_device=False)`
via `scripts/run_experiments.py`. Identity (`run_name`, `scientific_hash`,
`comparison_signature`) is computed by that resolver — there is no drifting
planning-hash replica. `--execute` is required to launch; the default is plan-only.

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # pyproject; stdlib + yaml
pip install torch torchvision numpy timm pytest   # execute / tests
```

## Suites

| suite        | training-row ids |
|--------------|------------------|
| review       | 5: ce, scsf_correctness, r3_scsf, dtr_scsf, cbr_scsf |
| sage         | 4: ce, sage_ds_v2, sage_topk_v2_fixedk2_pool, sage_topk_v2_fixedk2_pool_conv |
| all          | 8: dedup union of review ∪ sage |
| next5_pilot  | 11 ids × datasets × seeds; DAG expands fold teachers, OOF, memory, eval |
| r3/dtr/cbr_ablations | ablation ladders, never folded into `all` |

`next5_pilot` default matrix (no `--methods` filter): **9 main methods + 2 fold teachers = 11 training ids × 2 datasets × 1 seed = 22 training jobs**, plus artifact/eval tasks. See `docs/NEXT5_PROTOCOL.md`.

The two `sage_topk_v2_fixedk2_pool*` ids fold to distinct run names and scientific hashes.

## Commands

```bash
python scripts/run_experiments.py --suite next5_pilot \
  --datasets cifar10 cifar100 --backbones vgg16_bn --seeds 13 \
  --recipe ccl_sc_reference \
  --data-root /path/to/data --results-root /path/to/results

python scripts/run_experiments.py --suite next5_pilot \
  --datasets cifar10 cifar100 --backbones vgg16_bn --seeds 13 \
  --recipe ccl_sc_reference \
  --data-root /path/to/data --results-root /path/to/results \
  --devices cuda:0 --max-jobs 1 --execute

python scripts/run_experiments.py --status --results-root /path/to/results

python scripts/run_experiments.py --suite next5_pilot ... --resume --execute
```

Unsupported methods, devices, or backbones fail before any training. Resume
accepts a matching scientific hash and rejects conflicting configs. No automatic
`_v2` directories.
