# Experiment Suites (spec §10; commit 6/7)

Portable, torch-free planning lives in the planning replica
`scsf/engine/planning.py`. All counts below come from that replica and are
locked by `tests/test_planning_variant.py` (py_compile static gate, never
executed locally — torch/numpy unavailable in this environment).

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # pyproject; stdlib + yaml
pip install torch numpy     # required only for --execute / aggregation
```

## Exact suites (§10.2)

| suite        | cells | ids |
|--------------|-------|-----|
| review       | 5     | ce, scsf_correctness, r3_scsf, dtr_scsf, cbr_scsf |
| sage         | 4     | ce, sage_ds_v2, sage_topk_v2_fixedk2_pool, sage_topk_v2_fixedk2_pool_conv |
| all          | 8     | dedup union of review ∪ sage (never doctored) |
| r3_ablations | n-1   | r3 ablation ladder — never folded into `all` |
| dtr_ablations| n-1   | dtr ablation ladder — never in `all` |
| cbr_ablations| n-1   | cbr ablation ladder — never in `all` |

The two `sage_topk_v2_fixedk2_pool*` ids fold to **distinct** run names and
config hashes — never merged (locked by variant tests).

## Run (torch-free plan; no torch/numpy/CUDA/download needed)

```bash
python3 scripts/run_experiments.py --suite all --dry-run --data-root data \
    --results-root results
python3 scripts/run_experiments.py --suite review --dry-run --data-root data \
    --results-root results
```

`--dry-run` and `--execute` are mutually exclusive; `--execute` is required to
launch anything)Skip enclosure garbage. Read the file directly at completion of this cell.

