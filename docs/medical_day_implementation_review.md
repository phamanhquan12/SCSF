# Current Implementation Problems: Medical Day Experiments

Reviewed: 2026-10-15. Status: all seven findings confirmed and fixed on 2026-10-15 (see "Resolution" below).

## Resolution

1. `_iter_images()` skips files whose stem ends in `_mask`/`_mask_N` (`MASK_STEM_PATTERN`). The inflated Breast Ultrasound count matches the defect: Table 1 lists 1,576 images, whereas the dataset has 780 ultrasound images. `CACHE_FORMAT_VERSION` 3 rebuilds caches that contain masks; mask-free v2 caches are only relabelled, so their splits are unchanged. Breast Ultrasound runs must be retrained.
2. `run_experiment.py` saves `last.pt` atomically. It removes any stale `complete.json` when a run starts and writes the marker only after all evaluation outputs exist. `scsf/run_status.py` is the single completion check, used by the resume step, `aggregate_tables()` and `run_medical_day.sh`. Older runs count as complete when `test_curve.csv` (their last output) exists. With `--skip-existing`, an incomplete run is launched with `--reuse-last`, which re-evaluates a finished `last.pt` of the same configuration instead of retraining.
3. `expected_cumulative_errors()` gives the expected error count under a uniformly random order within tied groups. AURC and risk@coverage use it, so results no longer depend on input order. AUROC and FPR@95TPR already grouped ties.
4. NAURC is no longer clipped above 1. It is NaN when it is undefined (no errors, or no correct predictions).
5. AUROC and FPR@95TPR are NaN when undefined. Aggregation averages only valid seeds and writes `<metric>_n_valid` when some seeds are undefined.
6. Each invocation is recorded in `config/invocations/<timestamp>/`, and a line is appended to `config/invocations.jsonl`.
7. `run_medical_day.sh` reads `--run-id`/`--results-root` (including the `=` forms) before computing paths and progress.

`recompute_metrics.py --run-root <run>` applies fixes 3–5 to finished runs from their `per_sample.csv` files, without retraining.

## Scope

This report covers the execution path:

```text
run_medical_day.sh
  -> run_medical_suite.sh
  -> run_paper_medical_suite.py
  -> run_experiment.py
  -> scsf/{medical_registry,datasets,methods,metrics,reporting}.py
```

It concerns implementation defects reachable through this launcher, not changes to research ideas, baseline tuning, or experimental design. A trigger listed below does not imply that existing experiment results encountered it.

## Settings Already Consistent Through This Launcher

The default daily workflow selects seven methods on seven thesis datasets, with training seeds `0, 1, 2`, shared split seed `42`, pretrained ResNet-50, 224-pixel inputs, 100 epochs, 20 warm-up epochs, and batch size 64. All methods reach the same full-test evaluation of the last checkpoint and the same metric implementation.

The older evaluators and their inconsistent E-AURC definitions are not used by this path. Changing split seeds or mixing configurations is not an automatic issue in a fresh run with unchanged defaults.

Sources: [daily plan](../run_medical_day.sh), [launcher defaults](../run_medical_suite.sh), [shared experiment runner](../run_experiment.py).

## Findings

### 1. High: Breast Ultrasound Masks Enter Classification Discovery

- **Trigger:** source images and annotation masks share a class directory, for example `benign (1).png` and `benign (1)_mask.png`.
- **Implementation:** `_iter_images()` excludes certain directory names but does not exclude mask filename patterns. `_discover_samples()` accepts both files as examples of the same class.
- **Effect:** masks can become classifier inputs and alter dataset counts and splits. The launcher does not correct this.
- **Evidence:** a synthetic fixture passed through the original discovery helpers returned both filenames with class `benign`.
- **Required fix:** exclude annotation filenames before discovery/splitting, add a regression check, and rebuild affected caches. Existing affected results require reevaluation after retraining on the corrected data.

Source: [medical_registry.py](../scsf/medical_registry.py), `_iter_images()` and `_discover_samples()`.

### 2. High: Incomplete Evaluation Can Be Counted as Complete

- **Trigger:** interruption after writing `summary.csv` but before writing the remaining evaluation files; a truncated summary file can also remain after interruption.
- **Implementation:** `write_rich_evaluation()` writes the summary first. `day_done_count()` checks only whether a matching summary exists. The Python resume check requires only `last.pt` and the summary.
- **Effect:** `next` can skip unfinished outputs. Even explicitly rerunning the day can skip the affected run rather than regenerate missing risk curves, ROC curves, or per-sample predictions. A truncated summary can later break aggregation.
- **Evidence:** with 21 summary-only fixtures and no checkpoints or evaluation sidecars, the original wrapper reported day 1 as `21/21` and selected day 2.
- **Required fix:** write an atomic completion marker only after all required outputs succeed; use the same completion validation in the wrapper and Python runner. Permit evaluation-only recovery from an existing checkpoint.

Sources: [run_medical_day.sh](../run_medical_day.sh), `day_done_count()`; [run_paper_medical_suite.py](../run_paper_medical_suite.py), resume check; [run_experiment.py](../run_experiment.py), `write_rich_evaluation()`.

### 3. Medium: Tied Scores Make Risk Metrics Depend on Sample Order

- **Trigger:** equal confidence scores, including softmax scores that saturate to the same floating-point value.
- **Implementation:** risk-at-coverage and exact AURC use `np.argsort()` and select prefixes without a defined policy for tied groups.
- **Effect:** arbitrary order within a tie can change AURC and coverage errors. Class-sorted input order can influence results.
- **Evidence:** arithmetic using the implemented prefix-risk formula gives AURC from `0.2083` to `0.7917` for four equally scored samples with two correct predictions, depending on their order. This is an illustrative extreme, not a measured effect on thesis runs.
- **Required fix:** implement and document tie-aware risk calculations, such as expected risk under random ordering within each tied group. Stable sorting alone does not remove input-order dependence.

Source: [metrics.py](../scsf/metrics.py), `risk_coverage_curve()` and `exact_aurc()`.

### 4. Medium: NAURC Is Artificially Capped at One

- **Trigger:** the normalized excess-AURC ratio exceeds one for a poor confidence ranking.
- **Implementation:** both NAURC implementations clip the ratio to `[0, 1]`.
- **Effect:** distinct poor rankings can receive the same reported value.
- **Evidence:** for ranked correctness `[0, 0, 1, 1]`, the formula yields approximately `2.0`; the current clipping reports `1.0`.
- **Required fix:** remove upper clipping in both implementations and define handling for a zero normalization denominator. Existing saved predictions can be used to recompute this metric without retraining.

Source: [metrics.py](../scsf/metrics.py), `naurc()` and `summarize_selective_metrics()`.

### 5. Medium: Undefined ROC Metrics Receive Ordinary Numeric Values

- **Trigger:** every evaluated prediction is correct, or every prediction is incorrect.
- **Implementation:** correctness-detection AUROC returns `0.5` and FPR@95TPR returns `1.0` when correctness has only one unique value.
- **Effect:** undefined statistics enter mean/std tables as valid observations; a perfect classifier can appear to have chance AUROC and worst-case FPR.
- **Required fix:** explicitly mark these cases undefined, record the reason, and make aggregation report valid observation counts rather than average sentinel values.

Source: [metrics.py](../scsf/metrics.py), `auroc()` and `fpr_at_tpr()`.

### 6. Low: Each Day Overwrites Shared Configuration Metadata

- **Trigger:** normal consecutive daily invocations using the shared default run ID `thesis`.
- **Implementation:** `write_run_config()` rewrites `config/args.json`, `environment.txt`, `git_status.txt`, and other shared configuration files on every invocation.
- **Effect:** shared metadata describes the latest invocation rather than the full seven-day history. Individual checkpoint arguments and the appended command log remain available.
- **Required fix:** preserve invocation-specific configuration records and maintain a separate run-level manifest.

Source: [run_paper_medical_suite.py](../run_paper_medical_suite.py), `write_run_config()`.

### 7. Conditional: CLI Run-ID Overrides Disagree with Progress Checks

- **Trigger:** `bash run_medical_day.sh next --run-id alternate`.
- **Implementation:** the daily wrapper computes its metrics directory from the environment/default run ID. It forwards `--run-id alternate` to the child launcher without updating its own progress target.
- **Effect:** `next` chooses a day based on `thesis` but trains under `alternate`. Forwarded result-root overrides can cause the analogous mismatch.
- **Evidence:** shell-dispatch testing reproduced selection from the default run while the child command received the alternate run ID.
- **Required fix:** parse supported identity overrides in the daily wrapper before computing paths and progress.
- **Current workaround:** use environment variables consistently:

```bash
RUN_ID=alternate bash run_medical_day.sh next --gpu 0
RUN_ID=alternate bash run_medical_day.sh status
RUN_ID=alternate bash run_medical_day.sh aggregate
```

Source: [run_medical_day.sh](../run_medical_day.sh), initialization, `run_day()`, and `next` dispatch.

## Verification and Limits

Shell syntax checks passed for both launchers. The original daily wrapper was exercised against temporary result fixtures while intercepting the child launcher to avoid installing dependencies or starting training. Dataset discovery was checked using selected original helpers; metric examples used direct arithmetic for the formulas in the active implementation.

No full GPU training run was performed. Raw medical datasets and completed experiment outputs were unavailable locally, so the audit does not establish which existing results were affected. Repository implementation files were not modified during this review.

## Fix Priority

1. Exclude Breast Ultrasound masks and validate the corrected dataset cache.
2. Make completion checks atomic and recover interrupted evaluation outputs.
3. Correct tied-score, NAURC, and undefined-ROC metric handling.
4. Preserve daily configuration history and align override handling with progress checks.
