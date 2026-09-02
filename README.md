# scsf

Reusable selective-classification harness and the SCSF method (post-hoc
MetaCalibrator predicting True Class Probability from supervised features),
with reproducible, test-locked scientific definitions.

## What's in the package

* Engine: config resolution (YAML layers + CLI overrides), deterministic
  training with checkpointing / resume, and an evaluation pipeline that writes
  a run registry.
* Methods (all behind one factory — `scsf.methods.build_method`):
  * `ce` — plain cross-entropy baseline.
  * `dg` — Deep Gamblers (C+1 reservation neuron; reward 2.2).
  * `selectivenet` — SelectiveNet (closed-form loss with coverage penalty).
  * `sat` — Self-Adaptive Training with selective motivation.
  * `scsf` — our method: post-hoc MetaCalibrator + detached TCP target, with
    explicit gradient semantics (`posthoc` / `e2e` / `legacy_partial_detach`).
  * `ccl_sc` — Confidence-aware Contrastive Learning for Selective
    Classification (class-conditioned MoCo queues).
  * `sage_ds` — SAGE-DS: depth-wise selective supervision with a controller
    and per-site utility-EMA routing; ablated aliases `sage_ds_*`.
  * `depthfrag` — DepthFrag: distill depth-wise decision fragility into a
    terminal score. Signed relative fragility radii are extracted from a
    frozen checkpoint (`python -m scsf.extract_depthfg run_dir=...`) and an
    end-to-end method (`depthfrag.yaml`) regresses the detached targets via
    per-site probes + a terminal head. Ablation/control aliases
    `depthfrag_terminal_margin`, `depthfrag_terminal`,
    `depthfrag_intermediate`, `depthfrag_raw`, `depthfrag_frozen`,
    `depthfrag_clip`. See `docs/depthfrag.md`.
* Backbones: ResNet-18, VGG16-BN, WideResNet-28-10, ConvNeXt-Tiny, DeiT-Small,
  with tap roles (`top_l2`, `top_l1`) for SCSF and config-driven recipes.
* Metrics: exact selective-classification metrics (numpy only), including the
  hard-coverage grid, prefix-AURC, failure AUROC/AUPR, per-class/worst-class
  AURC, and excess AURC.

## Getting started

```bash
# from the repo root of this package
export SCSF_DATA_ROOT=/path/to/torchvision/data   # CIFAR-10/100 root
./.venv/bin/python -m scsf.train \
    dataset=cifar10 backbone=resnet18 method_name=ce \
    recipe=singlerun seed=13 train.epochs=200 train.overfit=0

# evaluate the selected checkpoint on the val split
./.venv/bin/python -m scsf.evaluate \
    run_dir=results/cifar10-resnet18-ce-rsinglerun-s13 split=val

# aggregate seeds from the registry
./.venv/bin/python -m scsf.aggregate results/registry.csv
```

Quick smoke (tiny overfit) so no full training is required:

```bash
./scripts/run_smoke.sh
```

## Configuration

`configs/` — datasets, backbones, methods, recipes — is merged in order
defaults < dataset < backbone < method < recipe < CLI overrides. Any
`a.b=value` CLI argument lands as a dotted key (see `resolve` /
`overrides_from_cli` in `scsf/engine/config.py`); nested values must be
passed as nested dicts when calling `resolve` directly.

The resolved config is dumped per run as `cfg.json`. Recipes express the
per-backbone schedule (CNNs → SGD + cosine, transformers → AdamW) and
per-method paper defaults (e.g. `paper` recipe sets SCSF `pretrain: 100`).

## Scientific definitions

Read `docs/EMPIRICAL_CONTRACT.md` — splits (seed 20260902, stratified
45k/5k), metric definitions (confidence-higher-is-keep, `u = -confidence`,
prefix AURC, locked coverage grid, NaN degenerates), the checkpoint
selection rule (`guard_delta_acc` + min val AURC), registry columns, and
aggregation keying are all locked by tests.

`docs/scsf-gradient-semantics.md` documents the SCSF gradient-routing audit
(v1's `end_to_end=False` detached logits only; that bug is now the explicit,
deprecated `legacy_partial_detach` mode).
`docs/depthfrag.md` documents the DepthFrag fragility geometry (margin-derived
signed relative radii), the BatchNorm fast/exact treatment, the limits of the
linear approximation, and the deployment path (terminal head only).

## Tests

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/
```

Covered: score primitives, exact selective metrics, split determinism and
statification, config/registry contracts, and per-method scientific
invariants (SelectiveNet loss formula, SAT history buffers, MoCo queues,
SCSF gradient routing and meta-weight schedule, SAGE-DS utility sign + gate
controller, DepthFrag fragility geometry and BatchNorm coupling).

## External sources & licenses

Method ports carry their provenance in `external_sources.yaml`; license texts
live in `external_licenses/` (Deep Gamblers, SelectiveNet, SAT, CCL-SC).
Run/results artifacts are recorded in the registry with `commit`/`dirty`
snapshots.

## Layout

```
scsf/
  backbones/   # models + tap probing
  data/        # cifar + deterministic serialized splits
  engine/      # config, trainer, evaluator, registry, checkpointing
  methods/     # factory + the methods + score primitives
  metrics/     # roc helpers + selective-classification metrics
  train.py evaluate.py aggregate.py extract_depthfg.py   # CLI entrypoints
tests/         # pytest suite (contract + methods + engine)
docs/          # EMPIRICAL_CONTRACT.md, scsf-gradient-semantics.md, depthfrag.md
configs/       # datasets / backbones / methods / recipes
scripts/       # runners and verification
```

## License

MIT (as declared by the legacy README of this repository). Any third-party
method port carries its own provenance; see `external_sources.yaml` and
`external_licenses/`.