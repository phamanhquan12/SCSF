# Empirical Contract

This file pins the reproducibility-critical definitions of the harness. Every
claim here is enforced by unit tests (`tests/`) or the CLI smoke suite; change
any of the following only through a deliberate, reviewable commit.

## 1. Datasets and splits

* Datasets: CIFAR-10 and CIFAR-100, 32×32, loaded via `torchvision` without
  automatic download (`data.download: false`; the user is expected to provide
  the data under `SCSF_DATA_ROOT`).
* The **official training fold** is used as the raw pool (50000 samples for
  CIFAR). Our own locked split carves it into a train split (45000) and an
  evaluation-on-training-data val split (5000). The official test set is
  **never** used for our val split; structure is guaranteed by
  `assert_no_official_test_leakage`.
* Split seed: **20260902**. Splits are class-stratified: exactly 4500 train and
  500 val samples per class (CIFAR-10), so no class resampling artifacts.
* Split files are serialized to `<results_root>/splits/` as
  `cifar10_train_seed20260902.txt` / `cifar10_val_seed20260902.txt` (one
  global official-fold index per line) and are loaded back by default
  (`data.use_serialized_splits: true`). Hashing director bytes includes the
  split seed, the train list and the val list; stored in `split_hash`.
* Using global official-fold indices means every method's *purported* official
  per-example accounting (e.g. SAT's history buffer keyed by example id) is
  reproducible regardless of shuffling.

## 2. Selective-classification metrics

Conventions (locked by `tests/test_metrics.py` and `tests/test_scores.py`):

* `confidence` — higher = more confident = **keep**. Uncertainty is `u = -confidence`.
* `error = 1[prediction != label]` is the positive class for detection metrics.
* Ties in `confidence` are broken by ascending sample *id* (deterministic
  secondary key `ids`); `stable_confidence_order` is lexicographic
  `(ids, -confidence)`.
* `selective_risk_at_coverages` accepts the `k = max(1, floor(q·N/100))`
  most-confident samples for the locked grid
  `q ∈ {100,99,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,1}`
  and reports `{coverage, k, n, accepted_frac, risk}`.
* `aurc` is the empirical mean of prefix risks over **all** accepted prefixes
  `k = 1..N` (not a trap over the hard-coverage grid).
* `auroc_error` / `aupr_error`: failure-detection AUROC / AUPR with
  `u = -confidence` as the risk score, error positive. Average-rank ties make
  both fully determined for tied scores (numpy only, no sklearn dependency).
* `excess_aurc = aurc - optimal_aurc(·)`, where `optimal_aurc(e, n)` is the
  AURC of a perfect selector at the same empirical error rate `e`.
* `per_class_aurc` / `worst_class_aurc`: class-restricted AURC over each
  class's own samples; worst class = max over classes.
* Degenerate cases (zero errors, zero negatives, empty class) return
  `NaN` — never a fabricated number.

## 3. Run selection rule

Each training run saves per-epoch checkpoints and the per-epoch val metrics.
The **selected** checkpoint is the one minimizing val `aurc` among all epochs
whose val accuracy is within `train.guard_delta_acc` (absolute percentage
points, default 1.0) of the best val accuracy. Selection is deterministic on
the metrics; the winners are recorded in `checkpoint_epoch` and `selection`.
`evaluate` defaults to the `selected` checkpoint (`python -m scsf.evaluate
checkpoint=best|last|selected|<epoch>`).

## 4. Run directory and registry

A run lives in `<results_root>/<run_name>/` with:
`best.pt`, `last.pt`, `selected.pt`, `cfg.json`, `manifest.json`, plus
`eval_val.json` (and anything written by `evaluate_run`). `run_name` is
`<dataset>-<backbone>-<method_name>[.<score>]-r<recipe>-s<seed>`.

The registry is append-only CSV at `<results_root>/registry.csv`. Column set is
locked in `scsf/engine/registry.py` (`BASE_COLUMNS`) and asserted by
`tests/test_config_registry.py`. Rows are keyed by `(run_dir, split)`: writing
a row for an existing key **replaces** it; a new split appends. Columns include
`run_dir, dataset, backbone, method_name, score, seed, recipe, split, style,
split_hash, commit, dirty`, the scalar metrics, the grid of
`risk_at_cov_<q>`, `checkpoint_epoch, selection, params_total, created_at,
complete`.

## 5. Aggregation

`python -m scsf.aggregate <registry.csv> [<out.csv>]` groups by
`(dataset, backbone, method_name, score, recipe, split)` and averages only
`complete == 1` rows, writing mean over seeds per metric column plus `std_*`
columns and a `runs` counter. Output naming defaults to
`registry_aggregate.csv`.

## 6. Reproducibility guarantees

* Seeded deterministic split (global, not per-folder).
* `torch_threads`, GPU/CPU device selection, `data_order_seed` all come from
  the resolved cfg; the config is dumped to `cfg.json` per run.
* No hidden global RNG at module import; method construction is deterministic
  given the config (probe-driven from a detached backbone forward).
* Checkpoint format is a flat dict of module state, optimizer state, and
  method buffers (SAT history, SCSF calibrator, MoCo queues), so runs can be
  resumed identically.

## 7. Invariants locked by tests

* Every method factory member produces a `MethodPrediction` with finite
  confidence and its declared `default_scores`.
* SCSF gradient modes (posthoc / e2e / legacy_partial_detach) behave per
  `docs/scsf-gradient-semantics.md` — see `tests/test_methods.py`.
* SelectiveNet loss equals the closed-form
  `α · (EMR + λ·Pcov(lm)) + (1-α) · L_CE_aux`.
* CCL-SC queues are cyclic and the loss builds a graph into the online encoder
  while never back-propagating into the momentum encoder.
* SAT momentum history is a registered buffer mixing one-hot prior and model
  soft-label, keyed by global official-fold index.