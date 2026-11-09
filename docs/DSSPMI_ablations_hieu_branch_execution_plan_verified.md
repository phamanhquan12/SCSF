# DSSPMI — New-Only Ablations — FAIR-Comparison Verified Execution Plan (v4)

> **Purpose:** An executable engineering plan **only for experiments or measurements that still need to be run**. This replaces the previous plan. It deliberately excludes dataset/backbone analysis, reconstruction of existing results, thesis prose, figures from old artifacts, and any re-running of already-completed comparisons.
>
> **Repository:** [phamanhquan12/SCSF, branch `hieu`](https://github.com/phamanhquan12/SCSF/tree/hieu)  
> **Code snapshot RE-VERIFIED:** branch `hieu` HEAD = `e9366ae87f2b838104c4c2fae4e0af88c160f663` (2026-11-09; confirmed via GitHub branch API + source reads)  
> **Execution environment:** H100 80 GB; ResNet-50 ImageNet-pretrained for all new full-training runs  
> **PROTOCOL LOCK:** 224 px, 100 epochs, confidence-head warm-up 20, SGD LR 0.01/momentum 0.9/weight decay 5e-4, milestones 40/70/90/gamma 0.1, batch 64, eval batch 128, 4 workers, split seed 42, train seeds 0/1/2, ImageNet pretrained, AMP **OFF**, validation each epoch, `last.pt` as official checkpoint. All effective settings must also be verified against each historic reference checkpoint/config; defaults alone are not proof.  
> **Execution mode:** staged; 87 new full R50 runs; 108 probe fits; standalone compute profiling; test-hidden screen/confirmation and hard compatibility gates; no automatic expansion based on test metrics.  
> **Metric:** existing `scsf/metrics.py::exact_aurc` tie-aware AURC using the **raw SCSF confidence-head logit**; keep accuracy/AUROC/NAURC/RC unchanged. Test data are *never* used for model/variant selection.

## 0. Exact scope and hard exclusions

### In scope — only NEW work that must execute

| ID | New execution required | Type | Primary backbone |
|---|---|---|---|
| **E02** | Extract frozen features from existing DS+ checkpoints; fit three independent probe families | Feature extraction + probe training | R50 |
| **E05** | Train **layer2-only** variant | Full model training | R50 |
| **E06** | Train arithmetic-mean target variant | Full model training | R50 |
| **E07B** | Train new β = 0.25, 0.75, 1.00 variants | Full model training | R50 |
| **E07W** | Train new layer2 target weights = 0.25, 0.75 | Full model training | R50 |
| **E09** | Train auxiliary-gradient **step** and **early** schedule controls | Full model training | R50 |
| **E16** | Run controlled H100 runtime/VRAM/inference benchmark, including DS+ primary-only inference | Profiling/benchmark; no 100-epoch training | R50; optional R101 |

**Explicitly excluded from this file:** aggregation of existing R50/R101 summaries or diagnostics; completed full DS+, layer3-only, final-TCP, temperature-matched, detached-gradient, or cosine-vs-linear runs; old PCam results analysis; R101 full sweep; PCam data-scaling experiment; drafting thesis text/claims or producing final research figures from already available files.

**Reference-only rule:** existing configurations appear below only to identify **what not to run** and which checkpoint/result is needed to **execute** a new controlled comparison. New-run scripts must **never enqueue** reference configurations. Cross-run results interpretation is a separate later task.

### 0.1. Count of NEW runs

| Experiment | New configurations | Datasets in primary confirmation | Seeds | **New 100-epoch R50 runs** |
|---|---:|---|---|---:|
| E05: layer2-only | 1 | COVID, HAM, APTOS, Chest CT | 0,1,2 | **12** |
| E06: arithmetic target | 1 | COVID, HAM, APTOS, Chest CT | 0,1,2 | **12** |
| E07B: β sensitivity | 3 | COVID, HAM, Chest CT | 0,1,2 | **27** |
| E07W: depth-weight sensitivity | 2 | COVID, HAM, Chest CT | 0,1,2 | **18** |
| E09: gradient schedules | 2 | COVID, HAM, Chest CT | 0,1,2 | **18** |
| **Total full-training confirmation** | **9 variants** | | | **87** |

Other **new executions**, not counted as 100-epoch model runs:

- **E02:** 4 datasets × 3 frozen DS+ checkpoint seeds × 3 probe families × 3 probe initialization seeds = **108 lightweight probe fits**, with 12 cached feature-extraction jobs (one per dataset/checkpoint). No backbone retraining.
- **E16:** benchmark SR, DS, DS+ with repeated instrumented forward, backward, optimizer and inference passes. No full training. Exact repetition count is specified in §7.
- **Stage 1:** 9 configurations × 2 datasets (COVID, Chest CT) × seed 0 = **18 actual 100-epoch runs**, **included** in the 87 total (not additional).
- **Stage 2:** complete the remaining **69** runs if Stage 1 passes correctness and validation/health gates.
- Smoke tests may perform short, non-reportable runs; they are additional development executions, **not** part of 87 research runs.

### 0.2. Canonical dataset slugs

```text
COVID      covid_qu_ex
HAM        ham10000
APTOS      aptos2019
Chest CT   chest_ct_scan
Malaria    malaria_microscopic
Brain MRI  brain_tumor_mri_masoud
BUSI       breast_ultrasound
PCam       pcam
```

Primary sets (predeclared before looking at new test results):

```bash
SCREEN=(covid_qu_ex chest_ct_scan)
CORE3=(covid_qu_ex ham10000 chest_ct_scan)
CORE4=(covid_qu_ex ham10000 aptos2019 chest_ct_scan)
ALL7=(covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan)
SEEDS=(0 1 2)
```

**Reason for these choices:** COVID represents an effective case, Chest CT a known failure case, HAM a class-imbalanced multiclass case, and APTOS an additional difficult multiclass domain. The set is fixed *before* new-run validation/test results are examined. PCam is restricted to cost profiling, not new training ablations.

---


### 0.3. Repository audit: exact source-of-truth settings and traps (rechecked 2026-11-09)

**This section overrides any merely illustrative command below.** All production commands must be generated from a locked manifest and must pass `--dry-run` assertions against the exact historical reference configuration. Sources reviewed in `hieu@e9366ae`: `run_medical_suite.sh`, `run_paper_medical_suite.py`, `run_experiment.py`, `scsf/methods.py`, `scsf/models.py`, `scsf/datasets.py`, `scsf/medical_registry.py`, `scsf/metrics.py`, and round-3/PD-ablation scripts.

| Contract | Grounded `hieu` behavior | Fairness rule for NEW runs |
|---|---|---|
| Main launcher | `run_medical_suite.sh`: R50, 224, 100 epochs, `pretrain=20`, **batch 64**, **eval batch 128**, **workers 4**, seeds **0 1 2**, pretrained, LR 0.01, milestones 40/70/90 | Invoke main wrapper or pass every field explicitly to the Python runner. **Do not inherit Python runner defaults.** |
| Dangerous Python defaults | `run_paper_medical_suite.py` defaults to **batch 128, eval batch 200, workers 8, seeds 42 0 1**; `run_experiment.py` also has its own defaults | Assert *resolved* command is 64/128/4, seeds 0/1/2, and contains `--pretrained`. Fail if not. |
| Optimizers | `run_experiment.py`: for DS+/DS, **SGD** on backbone plus depth heads, **Adam LR=0.001** on `calibrator.*`; SGD momentum **0.9**, weight decay **0.0005** | Do not accidentally move heads to Adam, change decay, or mix optimizer groups between arms. |
| Precision | Main script never supplies `--amp` (AMP **OFF**) | New runs FP32, unchanged TF32/cuDNN settings if recoverable; log exact torch/torchvision/CUDA and hardware. Do not silently add AMP merely because H100 supports it. |
| Transform | `scsf/datasets.py::_transforms`: train `Resize(224)→RandomCrop(224,padding=22)→RandomHorizontalFlip→ToTensor→Normalize`; eval `Resize(224)→CenterCrop(224)→ToTensor→Normalize` | Cache+transform parity is mandatory; do not "improve" preprocessing for only new runs. |
| Data splits | `medical-split-seed=42`; prepared cache under `data/preprocessed/<slug>_224`, via `split_manifest.csv` and `preprocess_manifest.json`; training seed only governs run RNG when cache is fixed | Match manifests **and class-index meaning** to old runs; never call `--force-preprocess` or rebuild a partial cache as a convenience. |
| Warm-up/aux CE | `PDSCSFMethod.training_loss`: depth-head CE trains from **epoch 1**, but its gradient into backbone is scaled via `_ScaleGrad` by `ds_aux_coefficient` (0 until epoch 20, then .02/.04/.06/.08/.10); confidence-head meta loss starts epoch 21 | Keep both independently identifiable. E09 changes **only aux CE→backbone gradient scale**, not head CE or confidence warm-up. |
| Default DS+ target | `p_L(y)^(1-β) * mean(q_layer2(y),q_layer3(y))^β`, `β=.5`, cosine heads (logit scale=10), depth CE is unweighted **mean** | Preserve bit-for-bit source expression for default; only nominated dimension changes. |
| Primary confidence | `PDSCSFMethod` inherits SCSF scoring; default `scsf_scorer=meta` ranks by **raw confidence-head logit**, not sigmoid/MSP/depth score | Same primary score for every ablation; auxiliary confidence is diagnostic only. |
| Checkpoint/metric | `last.pt` saved at epoch 100; evaluator uses `exact_aurc` with expected error allocation across score ties; `test_full` and `val_last` exported | Use the same `last.pt` and evaluator; never switch to `best.pt`, another tie policy, or partial test. |
| Test leakage hazard | `run_experiment.py` currently **evaluates test after every completed run** | Implement a **deferred official test** mode for screen/confirmation; only unlock final test after run matrix and validation-based decisions are frozen. |
| Unsafe reuse | `load_last_checkpoint()` currently checks only `dataset,method,arch,variant,seed,epochs`; suite `--skip-existing` checks completion, not all effective hyperparameters | For new execution, implement manifest+checkpoint **full effective-config hash** validation before reuse/skip/eval; do not assume variant name ensures correctness. |
| Timing | `complete.json::train_seconds` sums **train_one_epoch only**, not validation, saving and test | Profile training-step cost separately; add **end-to-end wall-clock** instrumentation in new 100-epoch runs. |

**Reference evidence constraint:** `SCSF_files.zip` contains summary and diagnostics copies, but not all `last.pt`, original command lines or complete provenance. The old variant directory name and summary are **not enough** to certify a matched historical run. Read existing original `last.pt`/`complete.json`/invocation config on the training server. If absent or incompatible, mark that comparison **BLOCKED / NOT STRICTLY MATCHED**; do not silently re-run old variants or misstate them as strictly paired controls.

**Chest CT hard check:** source cache has 7 distinct class-name strings across splits, but 4 semantic classes per split: train/val use e.g. `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`, test uses `adenocarcinoma`. Sorted 4-class numeric indices appear aligned, but **verify each split's `ImageFolder.class_to_idx` and explicit semantic mappings**. Do not change physical class folders/labels in new runs without matching historical reprocessing; if mapping is not semantically aligned, **block Chest CT comparisons and document the existing data issue**. The count `metadata.num_classes=7` is *not* the classifier-output dimension; `_imagefolder` returns train class count (4).

**What "fair" can and cannot guarantee:** exactly matching dataset membership, labels, code defaults, checkpoint policy and seeds permits a controlled comparison within the original protocol. GPU/driver/library differences can still affect numerical training trajectories; log them and report them as an environment limitation. Do not claim hardware-independent bitwise equality unless demonstrated. Do not run old baselines again automatically to hide this limitation.

## 1. Engineering baseline: implementation entry points and invariants

### 1.1. Verified code entry points at the audited branch

| Existing file / symbol | Relevant current behavior | Required NEW modification |
|---|---|---|
| `scsf/methods.py::PD_DEPTHS` | `layer2+layer3`, `layer3` | Add `layer2` for E05 |
| `scsf/methods.py::PDSCSFMethod` | Geometric target; mean depth probability; mean depth CE | Add aggregation operator and weight for E06/E07W |
| `scsf/methods.py::ds_aux_coefficient` | Auxiliary gradient scale tied to `pretrain` | Add **PD-specific** independent schedule for E09 |
| `scsf/methods.py::PDSCSFMethod.forward` | Calculates auxiliary depth score under `eval()` | Add optional **primary-only** path for E16 |
| `run_experiment.py::parse_args` | Single `--pd-beta`, limited depth choices, `--ds-aux-ramp` | Add CLI flags for new variants; preserve current defaults |
| `run_paper_medical_suite.py::method_variants` | Existing β sweep, legacy variant IDs | Thread new flags and stable unique new IDs |
| `run_medical_suite.sh` | Generic R50 medical launcher | Reuse; do **not** modify historical scripts as primary launcher |
| `scsf/models.py::FeatureModel.forward_with_maps` | Single ResNet forward returns layer maps and logits | Reuse for E02 feature export |
| `run_experiment.py::write_rich_evaluation` | Exports AURC, AUROC, per-sample, risk–coverage | Keep same metrics for each **new** run |
| `scsf/metrics.py` | Expected-error treatment of tied scores | **No change** in AURC definition |

**Aliases:** ResNet `layer2→early`, `layer3→mid`, `layer4→late`. DS+ primary confidence head reads `mid+late+logits`; depth classifiers read `early+mid` in full DS+.

### 1.2. Shared invariants for all new R50 training

1. Same source training data, **same original prepared cache**, normalization, RGB conversion, sample order/IDs, class index semantics and train/val/test membership as historical matched R50 references. Verify medical split seed `42`, source manifest checksum, train/val/test intersection=empty and per-split class-index maps. Do not invoke `--force-preprocess`; if cache provenance mismatches or split overlaps, **stop** rather than silently rebuilding.
2. Same final classifier/backbone/pretrained setting (including **pretrained weight identity/version**), epochs, optimizer(s), LR and scheduler, confidence head, primary score, batch size, augmentation, validation frequency, precision and seed. **Do not use Python-runner defaults:** explicitly lock batch 64, eval batch 128, workers 4, train seeds 0/1/2, AMP off. Change **only** the explicitly named ablation dimension.
3. Keep DS+ primary score as **raw meta/confidence logit**, not the auxiliary depth mean.
4. Use `last.pt`, **not validation-best checkpoint**, for official new test metrics. Log test only at the predeclared final stage where possible.
5. Save `complete.json`, full command line, source commit, `git status`, package lock, training/evaluation seed, exact pretrained-weight source/hash when available, dataset cache identity, variant ID, independent wall-clock timings, epoch-level logs and metrics for all new runs.
6. Preserve `NaN` for undefined AUROC; do not replace it by 0.5. Units for AURC: raw in CSV, with optional ×10⁻³ only in external displays.
7. Each run should use a **new output root** and run ID. Existing `round3`, `r101`, `R50_pcam` and prior ablation directories must remain untouched.
8. Stage 1 is **full 100 epochs**, not a shortcut model. Its result counts toward the 87 runs **only if** config/split/provenance integrity and deferred-test policy are satisfied.

### 1.3. Experiment lock and no test-driven hyperparameter selection

Before starting Stage 1, create `configs/dsspmi_new_runs_lock.yaml` recording:

- Source commit and uncommitted diff (must be empty for production runs).
- Exact nine new configurations and their stable IDs.
- Primary dataset/seed lists, data-cache manifest checksums, H100 device/config, training protocol.
- The rule that **validation and run-health checks** determine technical continuation; **test AURC must not determine which β/weight becomes the headline configuration**.
- The official thesis DS+ baseline remains unchanged at β=0.5, equal depth weights, geometric target and existing auxiliary gradient schedule.

For production, **implement and require** `--defer-test-evaluation` (the audited code does not have it): save `last.pt`, training/validation diagnostics, and `trained_complete` status, but **no test pass, per-sample test scores or test summaries** during Stage 1/validation-based decisions. Implement `--evaluate-last-only` to finalize tests from frozen checkpoints **without retraining** after the full matrix/config selection are locked. Track `trained_complete` vs `evaluated_complete` separately. If deferral is not implemented, test outputs must be access-controlled and never consulted during stage decisions, but this is weaker and is **not the preferred production gate**.

### 1.4. Historical-reference compatibility gate (mandatory before **any** 100-epoch R50 run)

Build `tools/verify_dsspmi_reference_contract.py` (new) plus `configs/dsspmi_reference_contract.json`. For **each dataset, seed and legacy comparison arm used by an E05–E09 comparison**, read the original source-server checkpoint and invocation configuration when available; do **not** infer equality solely from `summary.csv` or variant names. At minimum compare:

- Data: dataset slug, data root, cache `split_manifest.csv` hash, `preprocess_manifest.json` hash, train/val/test counts and class-index semantics, image input size and train/eval transforms. **Historical cache files may not have hashes recorded; when impossible to establish old cache identity, label the comparison `UNVERIFIED_SPLIT`, not `MATCHED`.**
- Model: `arch=resnet50`, `pretrained=True`, torchvision weight identifier or recovered initial checkpoint identity, num classes, feature spec `mid+late+logits`, head dimensions (`hidden_dim=256`, depth head dim `128`), `pd_head_type=cosine`, calibrator type and primary score `meta`.
- Training: `epochs=100`, `pretrain=20`, training batch 64, validation batch 128, workers 4 (affects speed/data loading), seed 0/1/2, `medical_split_seed=42`, SGD LR 0.01/momentum 0.9/weight decay 5e-4, LR milestones 40/70/90/gamma 0.1, Adam calibrator LR 0.001, loss `weighted_nll`, target `tcp`, confidence-weight scheduler cosine init 1/min 1e-4, error weight 1, default auxiliary weight 0.1/ramp 5 and detach `none`.
- Evaluation: `last.pt` epoch 100, no smoke subsets, full test loader, raw meta confidence logit, same tie-aware `exact_aurc` implementation; AMP off unless historic checkpoint/config explicitly proves otherwise.

**Comparison rule:** hold all fields equal except the one registered `allowed_diff_keys` per experiment: E05 only `pd_depths`; E06 only target aggregation operator; E07B only beta; E07W only depth target mixing weight; E09 only PD auxiliary-gradient schedule. New flags with default values must normalize to the legacy effective behavior. If another meaningful field differs, fail preflight, do not train it as a strict apples-to-apples control. Keep a `comparison_contract.csv` with `matched | unverified | blocked` and reason; do not silently repair old data.

**Paired seeds are common random-seed identifiers, not guaranteed identical minibatch augmentations across hardware/worker/PyTorch changes.** Verify train indices/first-batch order on a deterministic fixture and preserve isolated-init RNG; report seed-wise paired differences later only when reference integrity is demonstrated.

### 1.5. Provenance, immutable outputs, test lock and resume semantics

- `run_experiment.py::load_last_checkpoint` currently compares only **six keys**. It may accept `last.pt` with a different beta/aggregation/schedule if variant IDs or paths are reused. Implement a **normalized effective-config fingerprint**, compare all model/training/split/score parameters and dataset-manifest hashes, and strict state-dict loading before any resume/eval. Do not trust `--reuse-last` without this new guard.
- `run_paper_medical_suite.py::is_complete`/`--skip-existing` must be made aware of `trained_complete` vs `evaluated_complete` and expected config fingerprint; never silently skip missing official metrics, or destroy/overwrite an old run.
- Current `run_experiment.py` automatically runs `collect_outputs(model, loaders["test_full"])` and writes test metrics after every training run. The new wrapper must route **training-only/val-only** execution through the proposed deferred-test flag, and **final evaluation-only** through another flag that does not call `train_one_epoch` or overwrite `last.pt`.
- Freeze the experiment matrix **before** test-score exposure. No test-guided β/w2/schedule selection; repeated test-based selection across 87 runs would invalidate the held-out evaluation claim. Runtime/NaN/config integrity are legitimate reasons to stop; negative validation performance alone is not permission to quietly drop prespecified conditions.
- Store all new runs in a **distinct results root**, unique variant names and immutable config hashes; atomic writing for checkpoint and completion markers; verify actual source commit and clean tree for each run. Log any environment changes between stages instead of pretending the old results came from H100.

### 1.6. Default-path numerical parity test (run locally/briefly on H100; **not** an old 100-epoch rerun)

1. Checkout audited `hieu` in an untouched source worktree and implementation branch in another. Use the same fixed synthetic batch, same frozen test seeds, and the same original `pd_scsf` config `layer2+layer3`, beta 0.5, geometric, equal depth weights, `legacy` schedule.
2. Compare `forward()` primary logits/meta logit, `_current_tcp`, depth CE, SCSF meta BCE, total loss, optimizer parameter membership, and first 1–3 update steps **after epoch 20** with the same initial states. Verify equality within pinned-FP32 tolerance, ideally exact on same hardware/software. The new default must not require changes to historical checkpoints.
3. Verify `_ScaleGrad` at warm-up (0), first post-warmup epoch (.02) and ramp completion (.10), including **unscaled depth-head parameter gradients**.
4. Recheck `last.pt`/`best.pt` policy, tie-aware AURC and scoring using the same fixed predictions. Verify primary-only DS+ evaluation produces equal classes/confidence and AURC.
5. If numerical parity fails, stop; fix the implementation before any official training. **Do not use a fresh 100-epoch baseline rerun to disguise a changed default.**

---

## 2. E02 — NEW frozen-feature extraction and confidence-probe fitting

### 2.1. Research question and controlled design

Test whether intermediate/late backbone features provide usable confidence-ranking information beyond the same backbone's terminal logits. **Freeze a pre-existing DS+ trained model** so its class predictions, accuracy and features are identical across probes. Do not train another backbone.

| New probe | Input | Output | Supervision |
|---|---|---|---|
| `logits_only` | Terminal logits `z_L(x)` | Correctness-confidence scalar | BCE with `1[argmax(z_L)==y]` |
| `features_only` | GAP(layer3) + GAP(layer4) | Same | Same |
| `combined` | Both pooled features and terminal logits | Same | Same |

All three use a common, capacity-controlled probe design. Because input dimensions differ, record trainable parameter count and use a common hidden-width strategy; do not confuse a larger input with evidence of richer information. **Primary comparison to be analyzed later:** combined versus logits-only on the same fixed final classifier. Train all three probe families now; do not include analysis/plots of results from older runs in this execution plan.

### 2.2. Inputs that new execution actually needs

For each of COVID, HAM, APTOS and Chest CT × existing DS+ seeds `0,1,2`:

```text
<SERVER_RUN_ROOT>/checkpoints/<dataset>/resnet50/pd_scsf/
  l23-tgtdepth-b0.5-d128__w0.1r5__detach-none/<seed>/last.pt
```

Also need original prepared dataset cache + transformations and the saved checkpoint configuration. **Do not expect `last.pt` inside the compact summary ZIP**. If checkpoint is missing, mark that E02 dataset/seed **BLOCKED**, never fabricate a replacement checkpoint or retrain DS+ as part of E02.

### 2.3. New files to implement

`analysis/export_pd_frozen_features.py`:

1. Resolve `last.pt` by explicit path (do not infer from summary CSV).
2. Reconstruct `PDSCSFMethod` using checkpoint `args` and medical dataset class count; load `model` state dict strictly.
3. Set model to `eval()` and `torch.no_grad()`; use deterministic transforms and `forward_with_maps`.
4. Export terminal logits, GAP(layer3), GAP(layer4), ground-truth label, correctness, dataset split and stable sample identifier.
5. Export in **chunks** (`.npz`, `.pt` or parquet); avoid retaining all PCam activations in RAM (PCam is not in this E02 matrix anyway).
6. Fingerprint checkpoint SHA256, split/cache manifests, sample IDs and feature shapes. Verify exact order and label mapping between batches.

`analysis/train_pd_frozen_probes.py`:

1. Load features from a specified dataset/checkpoint/split; never modify backbone weights.
2. Use **validation-only calibration fitting** by default: split the existing validation set into deterministic stratified internal folds (if both correctness groups are large enough), select probe regularization via internal CV, then refit only on allowed validation samples. Never fit probe, scaler or search hyperparameters on test. This avoids optimistic correctness probes fitted on the backbone’s own near-saturated original training examples. **Chest CT validation may be too small to support honest probe training; explicitly mark insufficient rather than train on test or borrow test data.**
3. Fix feature normalization using **probe-fitting observations only**; apply unchanged statistics to held-out folds/test.
4. Match optimizer, regularization/search budget, number of updates and architecture family across probes; **explicitly report and, where practical, match trainable parameter budget**, since concatenating feature vectors increases input dimension. Keep prediction labels/ordering identical across all probes.
5. Run three probe initialization seeds `0,1,2` for each backbone seed; record fit failures if a fold lacks positive/negative correctness labels. Do not invent AUROC in degenerate folds.
6. Export probe checkpoint, internal-validation score, configuration and per-sample test scores only when test evaluation is unlocked.
7. Verify all probe families use exactly the same terminal classifier predictions and `test_N` for each checkpoint.

### 2.4. NEW run matrix and acceptance

```text
Datasets                 4 : COVID, HAM, APTOS, Chest CT
Existing frozen checkpoints 3 per dataset : DS+ R50 seeds 0,1,2
Probe families           3 : logits-only, features-only, combined
Probe seeds              3 : 0,1,2
--------------------------
Feature-export jobs     12
Probe training jobs    108
New backbone runs        0
```

**Important warning:** these are 108 probe fits, **not 108 independent backbone replications**. Probe-seed variability and backbone-seed variability must stay separately identifiable.

**Exit checks:** no state-dict changes to the backbone; no test labels in feature scaling/probe fitting/selection; no missing or duplicated sample IDs; fit/test positive-negative counts reported. Where Chest CT validation has too few errors, record `insufficient_calibration_errors` instead of silently reallocating official test examples into training.

**New outputs:** `E02_feature_exports/`, `E02_feature_manifest.csv`, `E02_probe_config.json`, `E02_probe_runs.csv`, trained probe weights, and per-sample **new probe** scores. No required thesis interpretation in this phase.

---

## 3. E05 — NEW layer2-only training (depth selection)

### 3.1. One and only change

Train `pd_scsf` using **only a layer2 (`early`) auxiliary cosine head** with β=0.5 and geometric target. Retain the existing confidence head reading layer3+layer4+detached final logits, and retain head CE/gradient scheduling. Only the set of depth heads used in confidence-target construction and depth CE changes.

**Already existing and therefore NOT to enqueue:** `layer2+layer3` full DS+, `layer3-only`.

### 3.2. Required code edits

```python
# scsf/methods.py — additive change
PD_DEPTHS = {
    "layer2+layer3": ("early", "mid"),
    "layer3": ("mid",),
    "layer2": ("early",),                  # NEW
}
```

- `run_experiment.py`: extend `--pd-depths` choices with `layer2`.
- `run_paper_medical_suite.py`: extend suite-level choices; add `site_tags["layer2"] = "l2"`.
- `PDSCSFMethod`: single-depth `torch.stack(...).mean()` and depth CE must remain valid; no unnecessary new architecture.
- Keep **legacy head-initialization behavior** (`init_seed+i`) for this primary comparison, because old controls were trained that way. Full model layer2 receives offset +0, so E05 layer2-only receives the same head init offset; in the existing layer3-only baseline, the layer3 head receives +0 rather than +1 (full model), a pre-existing initialization difference to document for later interpretation. **Do not silently switch stage init seeds in one arm.** Removing a head also changes the per-head CE share (mean over active heads), so the comparison isolates the *head-set design*, not a pure identical-gradient feature-information counterfactual. Record gradient norms.

### 3.3. Run specification

| Setting | Value |
|---|---|
| `--method` | `pd_scsf` |
| `--arch` | `resnet50` |
| `--pd-depths` | `layer2` **(new CLI choice)** |
| `--pd-target` | `depth` |
| `--pd-beta` | `0.5` |
| `--pd-head-type` | `cosine` |
| `--ds-detach` | `none` |
| Aux weight/ramp | 0.1 / 5 (legacy) |
| New variant ID | `l2-tgtdepth-b0.5-d128__w0.1r5__detach-none` |
| Dataset/seed | Core-4 × seeds 0,1,2 = **12 new runs** |

**Outputs:** normal `last.pt`, `complete.json`, `summary.csv`, `risk_coverage.csv`, `per_sample.csv`, `aux_diagnostics.csv`, grad logs, actual `pd_depths` in checkpoint args. No R50/R101 historical-results analysis here.

**Gate:** layer2-only forward/backward succeeds for 2- and multi-class datasets; `early_head_acc` is present; no unexpected `mid_head_acc` from a non-existent layer3 depth head; confidence-head input dimension and final inference rule are unchanged.

---

## 4. E06 — NEW arithmetic confidence-target variant

### 4.1. Precisely controlled intervention

Let `p=softmax(z_L)_y`, `qbar=0.5*q2_y+0.5*q3_y`, β=0.5.

**Existing geometric (reference, NOT to rerun):**

```text
t_geo = p^(1-beta) * qbar^beta
```

**New arithmetic (train):**

```text
t_arith = (1-beta)*p + beta*qbar
```

Keep heads, logits, supervision, optimizers, auxiliary gradients, confidence loss, beta, and inference identical. Both targets use labels **only during training** and are constructed inside the existing `torch.no_grad()` path for the confidence target. The actual SCSF meta loss is `weighted_nll` (`binary_cross_entropy_with_logits` by default), **not an ordinary hard-label correctness BCE**; do not change this while adding the arithmetic variant.

### 4.2. Code and CLI additions

- `scsf/methods.py::PDSCSFMethod.__init__`: optional `target_aggregation="geometric"`, choices `geometric|arithmetic`.
- `_current_tcp`: preserve original geometric expression **exactly** for legacy default; compute arithmetic only when explicitly requested.
- `run_experiment.py`: `--pd-target-aggregation`.
- `run_paper_medical_suite.py`: `--pd-target-aggregations`; include `agg-arithmetic` in new variant ID, but **do not rename the legacy geometric variant**.
- Record aggregation operator in checkpoint args and `complete.json`; add precision tests at beta 0 and 1, and gradient detachment.

### 4.3. Run specification

```text
method=pd_scsf; arch=resnet50
pd_depths=layer2+layer3
pd_target=depth; pd_target_aggregation=arithmetic
pd_beta=0.5; pd_head_type=cosine
pd_depth_weight_layer2=0.5
aux_gradient_schedule=legacy; ds_detach=none
```

**Variant:** `l23-tgtdepth-b0.5-d128-agg-arithmetic__w0.1r5__detach-none`.

**Datasets:** Core-4 × seeds 0,1,2 = **12 new training runs**.

**Gate:** arithmetic target equals expected closed-form values; label probabilities in `[0,1]`; `target.detach()` blocks confidence-loss gradients from target logits; default geometric path bitwise/near-bitwise matches audited code for fixed test tensors. No extra runs for geometric.

---

## 5. E07 — NEW β and depth-weight sensitivity sweeps

### 5.1. E07B — new β levels only

Current target:

```text
t_beta = p^(1-beta) * qbar^beta
qbar = (q2_y + q3_y)/2
```

**New levels to train:** `β ∈ {0.25, 0.75, 1.00}`.

**Do not enqueue:** β=0.5 full DS+; β=0 with unchanged auxiliary-head training (already available as the terminal-target control). If the old terminal-target control has a protocol mismatch, treat it as a later interpretation constraint, not a reason to automatically retrain anything under this execution-only plan.

E07B is already partially supported by `run_paper_medical_suite.py` through `--pd-betas`; `run_experiment.py` already supports `--pd-beta`. **No new model code is needed for β itself.** Ensure new command uses only the three new values and does not include `0.5`.

**Datasets:** Core-3 × seeds 0,1,2 × 3 β values = **27 new R50 runs**.

**New variant IDs:** preserve current pattern `l23-tgtdepth-b0.25-d128__w0.1r5__detach-none`, similarly `b0.75` and `b1`.

### 5.2. E07W — new layer2/layer3 weighting values only

Define the two-head depth mixture as

```text
qbar = w2*q2_y + (1-w2)*q3_y
beta = 0.5
```

**New levels:** `w2 ∈ {0.25, 0.75}`.

**Do not enqueue:** `w2=0.5` (already full DS+). Do not substitute layer2-only or layer3-only for `w2=1` / `w2=0`: they differ because the unused depth head can still contribute CE/backbone gradients in weighting experiments.

**Implementation:**

- `PDSCSFMethod.__init__(..., depth_weight_layer2=0.5)`; use weighted depth TCP only in the **two-head** case.
- Preserve legacy `torch.stack(probs).mean(dim=0)` for weight `0.5` to guarantee compatible default numerical behavior.
- **Do not reweight depth CE.** Keep its `mean` exactly as before; otherwise this would change two factors.
- Validate weight finite and within [0,1]; for one-head E05, ignore or reject irrelevant two-head weighting flags explicitly.
- `run_experiment.py`: `--pd-depth-weight-layer2`.
- `run_paper_medical_suite.py`: `--pd-depth-weight-layer2-values` (list) with unique `dw2-<value>` variant suffix for non-default; thread the selected scalar through the launched command.

**Datasets:** Core-3 × seeds 0,1,2 × 2 weights = **18 new R50 runs**.

**New variant IDs:** `l23-tgtdepth-b0.5-d128-dw2-0.25__w0.1r5__detach-none` and `...-dw2-0.75...`.

### 5.3. Required diagnostic output for both E07 arms

- Save `beta`, `w2`, actual depth head list, target aggregation mode, checkpoint commit and seed for each new run.
- Save per-epoch `final_tcp_mean`, `depth_tcp_mean`, `target_mean`, `target_saturated_frac`, `early_head_acc`, `mid_head_acc`; reuse `aux_diagnostics.csv` and add new structured fields only as needed.
- Save official validation/test AURC, accuracy, AUROC, NAURC, and risk@coverage under the unchanged evaluator.
- Do not implement automatic **test-based winner selection** in the launcher.

### 5.4. Regression gates

- `β=0` produces terminal TCP target; `β=1` produces selected depth TCP target (with epsilon handling matching original implementation).
- `w2=0.5` preserves the original depth mean.
- Changing `w2` changes **target aggregation only**, not the depth CE weighting.
- All nine new variant IDs across E05/E06/E07/E09 are unique and cannot overwrite an old run.

---

## 6. E09 — NEW auxiliary-gradient warm-up/ramp controls

### 6.1. Critical code constraint

Current helper `scsf/methods.py::ds_aux_coefficient(epoch,args)` reads `args.pretrain`; `args.pretrain` also determines confidence-head warm-up. **Never use `--pretrain 0` for this experiment.** That changes two mechanisms at once.

In all E09 variants: `--pretrain 20`; depth heads **still receive their own full CE gradient from epoch 1**. Only the *gradient scale from depth CE into the shared backbone* varies.

### 6.2. Exact schedules, epoch index 1–100

| Epoch | Legacy/default (DO NOT RUN) | NEW `step` | NEW `early` |
|---|---:|---:|---:|
| 1–20 | 0.0 | 0.0 | **0.10** |
| 21 | 0.02 | **0.10** | 0.10 |
| 22 | 0.04 | 0.10 | 0.10 |
| 23 | 0.06 | 0.10 | 0.10 |
| 24 | 0.08 | 0.10 | 0.10 |
| 25–100 | 0.10 | 0.10 | 0.10 |

The scale is applied via existing `_ScaleGrad` to depth features, while **head parameter gradients remain unscaled**. Confidence-head warm-up and cosine meta-weight schedule stay identical.

### 6.3. Engineering

Prefer a PD-specific option:

```text
--pd-aux-gradient-schedule {legacy,step,early}  # default legacy
```

Implement `pd_aux_coefficient(epoch,args)`:

```python
if schedule == "legacy":
    return ds_aux_coefficient(epoch, args)  # exact old path
if schedule == "step":
    return 0.0 if epoch <= args.pretrain else args.ds_aux_weight
if schedule == "early":
    return args.ds_aux_weight
```

Use it **only** in `PDSCSFMethod.training_loss` to compute `grad_scale`; leave other DS classes unchanged. Do not alter `SCSFMethod.training_loss`, the meta optimizer or the warm-up gate.

- CLI: `run_experiment.py` and suite-level `run_paper_medical_suite.py` add schedule choices.
- Variant names: `...__sched-step-w0.1__detach-none` and `...__sched-early-w0.1__detach-none`; original legacy name unchanged.
- Log exact epoch `aux_backbone_grad_scale` every epoch in structured `train_history.csv`; log meta-loss coefficient, train CE, depth CE, confidence loss, validation AURC/accuracy, gradient norms and NaN/Inf flags.
- Add tests at epochs `1, 20, 21, 22, 24, 25, 26, 100`; verify depth-head gradients nonzero but depth branch-to-backbone gradients zero in legacy warm-up and step warm-up.

### 6.4. NEW runs

Two variants × Core-3 datasets × seeds 0,1,2 = **18 new R50 full-training runs**. No re-run of legacy warm-up/ramp reference.

**Operational gate:** Verify new early schedule actually starts depth gradients at epoch 1 and that confidence-head loss remains gated off through epoch 20. If these cannot be independently verified, block E09 instead of reporting a confounded comparison.

---

## 7. E16 — NEW controlled computational-cost measurements (NO ADDITIONAL full training)

### 7.1. Measurements to execute

Use existing pretrained/finished checkpoints as architecture/weight sources; **do not** re-run 100 epochs merely for timing. Execute a controlled H100 profiler for SR, DS (`scsf`) and DS+ (`pd_scsf`), **plus instrument the 87 already-planned new full runs** for real full-epoch and end-to-end wall-clock cost. A few timed training steps estimate throughput but **cannot establish exact 100-epoch training wall-clock** or faithfully reconstruct an old run on a different GPU.

| Measurement | Execution to run | Unit | Important distinction |
|---|---|---|---|
| Parameter counts | Instantiate and load matched architectures; count trainable, total, backbone, confidence-head, auxiliary head params | parameters / M | DS+ training-only head params must be reported separately |
| Inference latency, primary score | 224×224 input; actual terminal class + official learned confidence only | ms/image or ms/batch | Depth heads not needed for DS+ deployed primary scoring |
| Inference latency, diagnostics | Same input plus auxiliary depth scores enabled | ms/image or ms/batch | Reports upper bound if diagnostic heads retained |
| Peak inference GPU memory | Reset peak before timed inference; measure allocation | MiB | Use consistent CUDA memory policy |
| Train-step latency | `forward + classification/meta/aux losses + backward + optimizer step` at representative joint-training epoch | ms/step; img/s | Distinguish training overhead from evaluation latency |
| Peak train-step GPU memory | Peak allocated and optionally reserved, under matching optimizer state | MiB | Include gradients and optimizer states |
| Pipeline throughput (new) | Real cached loader on HAM **and PCam** (if cache exists), same loader setup | img/s; s/epoch | Separate data/PNG decode/I/O and GPU compute; PCam projection is an **estimate** |
| Full-run wall clock (new) | Time 100-epoch R50 new runs already in Stage 1/2 (training + validation + checkpoints) | hours/run; total GPU-hours | Adds **zero extra 100-epoch runs**; baseline `train_seconds` excludes nontraining time |

### 7.2. Benchmark inputs and hardware

- **GPU:** H100 80 GB, record exact GPU model, CUDA driver, PyTorch, torchvision, GPU clock/power environment when available.
- **Backbone:** R50 primary; R101 only if checkpoint/config available and cost permits (not required in the 87-run matrix).
- **Models:** SR, DS, DS+ weights/configs from the **same dataset and seed** (prefer HAM, seed 0); checkpoint-realistic benchmarking requires their `last.pt` files. **For PCam-specific model latency**, use the actual PCam-matched SR/DS/DS+ R50 checkpoints only if they exist; do not relabel a HAM checkpoint as PCam-trained.
- **Input:** FP32, RGB 224×224. Use synthetic on-device tensor for pure GPU inference/step compute; benchmark a real HAM loader separately when feasible. PCam **does not require full retraining**; synthetic compute captures fixed-shape network FLOPs, but run a **separate real-PCam dataloader throughput** measurement and clearly label a projected 100-epoch runtime as an extrapolation, not an actual PCam training observation.
- **Batch sizes:** inference `1, 8, 32, 64`; training `64` (reduce only if OOM, then report the deviation explicitly).
- **Replications:** 3 independent measurement rounds; each with 20 warm-up iterations and 100 timed inference repetitions. For train steps, 10 warm-up steps and 30 timed steps × 3 rounds; record median, p90 and per-round variation. If memory/time budget requires adjustment, save actual counts in machine-readable config.
- Synchronize CUDA before/after host-wall timing (`torch.cuda.synchronize`) or use CUDA events, document which. Disable per-step logging/checkpoint saving from the timed region. Avoid sharing the H100 with other jobs while profiling.

### 7.3. Fix the DS+ evaluation path BEFORE measuring latency

Current `PDSCSFMethod.forward()` calculates auxiliary depth probabilities in `.eval()` to provide `aux_confidence`, even though the official primary score comes from the SCSF confidence head. This would artificially inflate the measured *deployed* DS+ latency.

**Required code change:**

- New `model.emit_aux_confidence` or explicit `forward(..., diagnostics=False)` hook; **default must preserve existing evaluation behavior** to keep historical metrics intact.
- When `diagnostics=False`, return exact same `eval_logits` and `confidence` without calculating depth head logits or auxiliary score (and skip optional pooling if not otherwise needed for the primary head).
- Benchmark both `primary_only` and `with_aux_diagnostics` modes on the same checkpoint and inputs.
- Unit test: `max|old_primary_confidence - new_primary_only_confidence|` ≤ expected FP32 tolerance; same class prediction and AURC. No change in training forward/backward.

### 7.4. Implementation files

`analysis/benchmark_pd_compute.py`:

1. Load SR, DS, DS+ checkpoints + exact configs; validate model param shape and strict state-dict loading.
2. Set consistent FP32 (AMP off), deterministic seed, batch sizes, gradients and optimizer state; log cuDNN/TF32 settings and keep per-method benchmark policy identical.
3. Separate inference from train-step benchmark; avoid validation/evaluation time being included in train-step measurement.
4. For training, mimic **actual** method-specific optimizer(s): SR SGD only, SCSF/DS and DS+ SGD backbone (+ DS+ depth heads) plus Adam calibrator; representative epoch `30` (after 20-epoch warm-up), aux gradients active. Put model in `.train()` with proper zero_grad/backward/optimizer steps, fixed identical input-label batches, and warm optimizer states. Record this as **steady-state per-step cost**, not time-to-convergence.
5. Call `torch.cuda.reset_peak_memory_stats` after setup/warmup at consistent boundary; distinguish allocated vs reserved. Recreate/reset weights/optimizer state between independent rounds as needed, and avoid drifting BN/optimizer states contaminating an otherwise identical timing comparison.
6. Record exceptions/OOM and skipped metrics explicitly rather than reporting zero.
7. Export raw per-iteration times and summary statistics, plus hardware/software config. Add instrumentation in new `run_experiment.py` training-only mode for `epoch_train_seconds`, `epoch_validation_seconds`, `checkpoint_seconds`, `total_wall_seconds` and per-run work; do not use historic `complete.json::train_seconds` as if it includes validation.

`tests/test_pd_primary_only.py`: same predictions/confidence, expected head not executed under primary-only mode; gradient/training path unaffected.

**Important:** This provides **controlled cost comparisons** at a representative H100 setting; PCam end-to-end cost is only an estimate unless an actual completed PCam training log from the same environment is available. Do **not** claim it establishes PCam data-scaling efficiency or clinical deployment latency. Do not silently substitute random weights for a missing checkpoint and label the result as checkpoint-realistic.

### 7.5. New outputs

```text
benchmarks/E16/
  benchmark_config.json
  checkpoint_manifest.csv
  parameter_counts.csv
  inference_raw_times.csv
  train_step_raw_times.csv
  inference_summary.csv
  train_step_summary.csv
  gpu_memory_summary.csv
  run_environment.json
  pcam_loader_throughput.csv
  projected_pcam_cost.csv         # explicitly estimated; NOT observed 100-epoch run
  new_runs_full_wall_clock.csv    # measured during E05/E06/E07/E09, no extra training
```

---

## 8. Precise staged execution workflow

### Stage 0 — Implement and verify new-only features (no 100-epoch run)

1. Create an isolated experiment work branch from audited `hieu` (for example `dsspmi/new-only-ablations`); record new commit before production runs.
2. Implement E05/E06/E07W/E09 CLI, variant IDs, model logic and diagnostics; keep baseline defaults unchanged. E07B uses existing β flags.
3. Implement E02 feature export/probe training separately; E16 profiler and primary-only mode separately.
4. Create `configs/dsspmi_new_runs_lock.yaml`, `configs/dsspmi_reference_contract.json`, `tools/verify_dsspmi_reference_contract.py`, and `run_medical_dsspmi_new.sh`, with subcommands `status`, `dry-run`, `smoke`, `screen`, `confirm`, `profile`, `probe`, `aggregate-new`.
5. Smoke tests (fake tensors and tiny real-data smoke subsets), unit tests, reference-config/split verification, default-path numerical parity and full deferral/resume checks; no official metrics from smoke.
6. Validate split/cache identities (including Chest CT class-index semantics), historical reference configs, checkpoint presence for E02/E16, GPU availability and output directory permissions.

**Stage 0 pass criteria:** all newly introduced CLI flags correctly reach `run_experiment.py`, exactly 9 new variants are enumerated and old variants excluded, **historical reference contract is matched or flagged**, default loss and optimizer parity holds, deferred-test and strict resume work, smoke shows finite losses/gradients, output metadata is complete, and primary-only inference matches the standard score.

### Stage 1 — Small full-protocol screen (18 of 87)

```text
New variants: E05 (1), E06 (1), E07B (3), E07W (2), E09 (2) = 9
Datasets: covid_qu_ex, chest_ct_scan
Seed: 0
Training: 100 epochs each, unchanged input and optimizer settings
Total: 18 NEW production runs
```

Run health checks: valid train/val results, no NaN, correct head gradients/target parameters, expected checkpoint state and metadata, stable GPU usage. Use **validation-only** scores and run-health checks for screening decisions. **Defer all test evaluation** until after the prespecified matrix and any validation-based selection are locked; do **not** inspect test to pick a new β/target as if unbiased.

**Important:** Stage 1 counts toward Stage 2 if the exact configuration and artifact integrity pass; no re-running the same variant/seed.

### Stage 2 — Complete the predeclared 87-run matrix

- E05/E06: add HAM/APTOS seeds 0,1,2 and COVID/Chest seeds 1,2 until 12 per E05/E06.
- E07B/E07W/E09: add HAM seeds 0,1,2 and COVID/Chest seeds 1,2 until 27/18/18 respectively.
- Run with `--skip-existing`, **but only within the new run root**. If a run has `last.pt` but missing metrics, use **new strictly-validated `--evaluate-last-only`** with full config-hash verification. **Do not use unpatched `--reuse-last`**, which compares only six fields.
- After all required **training** runs pass and selection is locked, call `--evaluate-last-only` for the full held-out test once per completed checkpoint. Then produce **new-run-only** CSV aggregates. Do **not** run any script that rebuilds main thesis tables or analyzes old ZIP artifacts in this phase.
- If resource limits require stopping early, preserve the partial run matrix and `status.csv`; never claim the entire 87 completed.

### Stage 3 — Optional NEW training expansion (separate approval)

| Expansion | Added NEW runs beyond primary 87 | Condition |
|---|---:|---|
| E05 layer2-only on remaining Original-7 datasets | +9 | Validation/scientific need for broader layer-selection evidence |
| E06 arithmetic on remaining Original-7 datasets | +9 | Need for broader target-operator robustness |
| E07B add APTOS | +9 | Strong sensitivity claim needs second multiclass check |
| E07W add APTOS | +6 | Depth weight effect unstable across Core-3 |
| E09 add APTOS | +6 | Schedule findings need another task |
| R101 replicate ONE predeclared new variant on 2 datasets × 3 seeds | +6 per variant | Only if an important conclusion depends on architecture robustness |

None of these optional runs should be submitted automatically. No extra PCam training or data-scaling runs.

---

## 9. New-only command and orchestration contract

### 9.1. Proposed run launcher (must be implemented)

Create `run_medical_dsspmi_new.sh` that **only enumerates NEW variants**:

```bash
# Intended interface — new wrapper is NOT present in audited hieu.
bash run_medical_dsspmi_new.sh dry-run
bash run_medical_dsspmi_new.sh smoke
bash run_medical_dsspmi_new.sh screen --gpu 0
bash run_medical_dsspmi_new.sh confirm --gpu 0
bash run_medical_dsspmi_new.sh status
bash run_medical_dsspmi_new.sh aggregate-new
bash run_medical_dsspmi_new.sh probe --gpu 0
bash run_medical_dsspmi_new.sh profile --gpu 0
```

**Never enqueue these old variants:** `l23-tgtdepth-b0.5-d128__w0.1r5__detach-none`, `l3-tgtdepth-b0.5-d128__w0.1r5__detach-none`, `l23-tgtfinal-b0-d128__w0.1r5__detach-none`, `ts_scsf`, detached, or linear-head controls.

### 9.2. One runnable example on current `hieu`: new β values

The existing runner already accepts `--pd-betas` for β sweeps. This command runs **only values that have not yet been run**:

```bash
# Run from repo root after setting a NEW output root; inspect dry-run first.
export NEW_RESULTS_ROOT=/absolute/path/to/NEW/results/medical_suite
export NEW_RUN_ID=dsspmi_new_ablations_v4
export ORIGINAL_DATA_DIR=/absolute/path/to/ORIGINAL/prepared/data

bash run_medical_suite.sh \
  --results-root "$NEW_RESULTS_ROOT" \
  --run-id "$NEW_RUN_ID" \
  --skip-existing --no-download \
  --data-dir "$ORIGINAL_DATA_DIR" \
  --datasets-file datasets_thesis.md \
  --arch resnet50 --methods pd_scsf \
  --datasets covid_qu_ex ham10000 chest_ct_scan \
  --seeds 0 1 2 \
  --epochs 100 --pretrain 20 --batch-size 64 \
  --workers 4 --eval-batch-size 128 --eval-every 1 \
  --medical-split-seed 42 \
  --meta-lr 0.001 --meta-loss weighted_nll \
  --scsf-feature-spec mid+late+logits --scsf-meta-target tcp \
  --scsf-scorer meta --ds-aux-weight 0.1 --ds-aux-ramp 5 \
  --pd-depths layer2+layer3 \
  --pd-betas 0.25 0.75 1.0 \
  --pd-targets depth \
  --pd-head-types cosine \
  --ds-detach-modes none \
  --dry-run
```

**This is DRY-RUN ONLY.** The audited script still writes test metrics if `--dry-run` is removed, so **do not launch the 87-run campaign via this raw command** until the new wrapper implements test deferral and full effective-config checks.

Caveat: the shell wrapper forwards unfamiliar flags to `run_paper_medical_suite.py`; verify generated commands with suite-level `--dry-run` before spending GPU hours. The script above is runnable after setting real paths and checking the medical cache; the new variants other than β still require code changes.

### 9.3. Intended calls after implementing new flags

```bash
# All examples below describe the NEW wrapper interface, not commands currently shipped.
bash run_medical_dsspmi_new.sh confirm --experiment E05 --gpu 0  # layer2-only
bash run_medical_dsspmi_new.sh confirm --experiment E06 --gpu 0  # arithmetic
bash run_medical_dsspmi_new.sh confirm --experiment E07B --gpu 0 # beta sweep
bash run_medical_dsspmi_new.sh confirm --experiment E07W --gpu 0 # depth weights
bash run_medical_dsspmi_new.sh confirm --experiment E09 --gpu 0  # step/early gradient
python analysis/export_pd_frozen_features.py --config configs/E02.yaml
python analysis/train_pd_frozen_probes.py --config configs/E02.yaml
python analysis/benchmark_pd_compute.py --config configs/E16.yaml
```

The standalone Python commands above are **design specifications** until their corresponding parsers are implemented and documented. They are not claims that files already exist.

### 9.4. Unique variant strings for exactly nine new configurations

```text
E05   l2-tgtdepth-b0.5-d128__w0.1r5__detach-none
E06   l23-tgtdepth-b0.5-d128-agg-arithmetic__w0.1r5__detach-none
E07B  l23-tgtdepth-b0.25-d128__w0.1r5__detach-none
E07B  l23-tgtdepth-b0.75-d128__w0.1r5__detach-none
E07B  l23-tgtdepth-b1-d128__w0.1r5__detach-none
E07W  l23-tgtdepth-b0.5-d128-dw2-0.25__w0.1r5__detach-none
E07W  l23-tgtdepth-b0.5-d128-dw2-0.75__w0.1r5__detach-none
E09   l23-tgtdepth-b0.5-d128__sched-step-w0.1__detach-none
E09   l23-tgtdepth-b0.5-d128__sched-early-w0.1__detach-none
```

Variant IDs must be derived from *effective* config values, not manually pasted into output paths. A test must assert **no collision** with any historical full/ablated variants; never truncate a differing suffix.

---

## 10. Testing: fail fast BEFORE expensive H100 runs

### 10.1. New unit tests

Create `tests/test_pd_remaining_new.py` and `tests/test_pd_primary_only.py` with at minimum:

- E05: layer2-only constructs exactly one `early` head; output shapes and gradients valid for C=2,4,5,7; no phantom `mid` head.
- E06: geometric default reproduces unmodified target values; arithmetic closed-form; β=0/1 endpoints; no target-path gradient from confidence BCE into depth classifier logits.
- E07B: β guards `[0,1]`; three beta values produce unique variants; default does not change.
- E07W: weights finite and in `[0,1]`; default w2=.5 matches original mean; CE per-head losses remain equally weighted; target changes correctly.
- E09: exact gradient scale at epochs 1,20,21,22,24,25,26,100; confidence BCE remains gated by `pretrain=20` for both new schedules; head CE gradients remain active at epoch 1.
- E16: `primary_only` produces same confidence logits/predictions as normal evaluation but does not execute depth-head inference; no accidental change to training output.
- E02: backbone parameter hashes identical before/after export/probe; no test labels in fit; sample IDs unique; impossible AUROC handles correctly.
- CLI: all flags round-trip through `run_medical_suite.sh` → `run_paper_medical_suite.py` → `run_experiment.py`; checkpoint `args` and `complete.json` preserve effective values.
- New IDs: no collisions with legacy IDs, no accidental generation of old configurations in the `confirm` matrix.
- Runner-default trap: invoking `run_paper_medical_suite.py` directly with defaults must be rejected by preflight (128/200/8, seed 42/0/1 != locked 64/128/4, seed 0/1/2).
- Full reference contract: every expected historical baseline/config/data fingerprint validated or explicitly `UNVERIFIED`; a swapped split manifest or Chest CT class-index permutation **must hard-fail**.
- Strict checkpoint reuse: change `pd_beta`, aggregation, depth weight, gradient schedule or data hash while leaving path/seed fixed; verify evaluation/skip rejects the stale checkpoint.
- Test isolation: after Stage-1 `trained_complete`, there must be **no** test `summary.csv`, `per_sample.csv` or other held-out-test evaluation artifact; evaluation-only after lock must never call training or mutate model checkpoint.
- Precision and augmentation: exact old random crop padding=22, horizontal flip, normalized eval center crop, AMP off, same `exact_aurc` for fixtures with confidence ties.
- Benchmark parity: DS+ `primary_only` confidence logits/labels match diagnostic mode; measured train-step uses actual SGD+Adam optimizer groups, `model.train()`, warm-up/existing scheduler settings and epoch-30 auxiliary gradient. 

### 10.2. Integration smoke matrix

| Smoke | Inputs | Expected outcome |
|---|---|---|
| E05 | R50, sample C=2, depth `layer2`, 2 epochs | One depth head; valid loss/metrics |
| E06 | R50, arithmetic, 2 epochs | New target + unchanged primary score |
| E07B | Two β values on tiny sample | Different target values; correct IDs |
| E07W | Two weight values | Only target mixture changes, not CE mean |
| E09 | epoch schedule unit test + tiny 2-epoch/20-pretrain override smoke | Gradient path matches declared variant; **do not** mistake 2-epoch smoke for schedule-efficacy result |
| E02 | One existing checkpoint with small subset | Three probe fits, frozen backbone, no label leakage |
| E16 | One actual SR/DS/DS+ checkpoint, batch=1 | Latency/VRAM outputs and primary-only parity |

**Warning for E09 smoke:** with `pretrain=20`, a two-epoch smoke does not exercise the legacy/step transition at epoch 21. Test transition by direct schedule-unit tests or a tiny ≥26-epoch controlled smoke with a tiny data subset. Do not shorten confidence warm-up when validating E09 mechanics.

### 10.3. Hard-stop conditions

- Data cache/split/class-index mapping differs from historical R50 reference; historical config not verifiably matched; missing or corrupt checkpoints needed for E02/E16 (block only dependent probe/profile jobs, not unrelated trainings).
- New code changes default full DS+ target or inference score at the same weights/inputs.
- Unexpected head/gradient activation, NaN/Inf, bad labels or variant name collision.
- Stage-1 and Stage-2 test outputs are generated before the matrix/selection lock or used to search for best β/w2.
- `--skip-existing` / `--reuse-last` silently skips an incomplete/incompatible run, or overwrites historical artifacts; a full config hash mismatch must stop.
- H100 profiling polluted by concurrent GPU jobs, or unsupported precision/AMP drift.

---

## 11. Required new-run output structure and completeness checks

Keep all **new** outputs separated from previously completed run packages:

```text
<NEW_RESULTS_ROOT>/<NEW_RUN_ID>/
  config/
    invocations.jsonl
    commands.txt
    new_experiment_matrix.yaml
  checkpoints/
    <dataset>/resnet50/pd_scsf/<NEW_VARIANT>/<seed>/
      last.pt
      complete.json
      aux_diagnostics.csv
      aux_grad_norms.csv                # when logging applicable
      train_history.csv                # NEW for schedule experiments
      trained_complete.json            # NEW: before any held-out test evaluation
  metrics/
    <dataset>/resnet50/pd_scsf/<NEW_VARIANT>/seed_<seed>/
      last/                             # created only by locked evaluation-only step
        summary.csv
        risk_coverage.csv
        per_sample.csv
      val_last/
        summary.csv
        per_sample.csv
  tables/
    new_runs_per_seed.csv
    new_runs_mean_std.csv
    new_runs_status.csv
    comparison_contract.csv              # historical-run compatibility: matched/unverified/blocked

probes/E02/
  frozen_features/...
  probe_checkpoints/...
  probe_runs.csv
  probe_scores/...

benchmarks/E16/
  benchmark_config.json
  checkpoint_manifest.csv
  parameter_counts.csv
  inference_raw_times.csv
  train_step_raw_times.csv
  gpu_memory_summary.csv
  run_environment.json
  pcam_loader_throughput.csv
  projected_pcam_cost.csv         # explicitly estimated; NOT observed 100-epoch run
  new_runs_full_wall_clock.csv    # measured during E05/E06/E07/E09, no extra training
```

For **each newly trained model**, save a machine-readable row with:

```text
experiment_id, source_commit, run_id, dataset, arch, train_seed,
variant_id, input_size, epochs, data_split_seed, checkpoint_epoch,
pd_depths, pd_head_type, pd_target, pd_target_aggregation,
pd_beta, pd_depth_weight_layer2, pd_aux_gradient_schedule,
ds_aux_weight, ds_aux_ramp, ds_detach, validation_aurc,
test_aurc, test_accuracy, test_auroc, test_eaurc, test_naurc,
train_seconds, validation_seconds, checkpoint_seconds, total_wall_seconds,
reference_contract_status, effective_config_sha256, data_manifest_sha256,
checkpoint_path, trained_complete, evaluated_complete, status
```

**Run completeness must not be inferred solely from file presence.** Check `trained_complete.json` and eventual `complete.json` match the exact effective config and data hashes; `last.pt` has epoch 100 and checkpoint type `last`; after the explicit evaluation-only stage, non-empty official `last/summary.csv` exists. If tests are deferred, **never** treat `trained` as `evaluated`, nor trigger retraining when evaluation is pending.

No script in this phase should combine the new outputs with pre-existing thesis tables or perform explanatory analysis of existing artifacts. The purpose here is to **generate complete and reproducible new evidence** for later analysis.

---

## 12. H100 execution controls and handoff

### 12.1. Safety and reproducibility

- On H100, measure memory footprint on one real new R50 experiment before scheduling parallel jobs. Do not assume two simultaneous experiments fit or that they will yield comparable runtime on a shared accelerator.
- Use `CUDA_VISIBLE_DEVICES` / `--gpu` consistently. Record NFS/cache paths; stage data locally if I/O becomes bottleneck and cache identity is preserved.
- Save stdout/stderr for each command and the exit code. Interrupted runs are *not complete*; if checkpoint resume beyond last-epoch is not implemented, restart the failed configuration in a fresh temporary directory, not by corrupting completed artifacts.
- Use `--skip-existing` only for verified complete runs under the **new** root.
- Include commit SHA, normalized config digest, historical reference contract status and a clean git status for production. If code changes between stages, record a new version and rerun numerical compatibility checks; do not silently mix incomparable algorithms.

### 12.2. Operational checklist: only work that still must be executed

- [ ] Create code branch, immutable version lock, **historical reference contract validator**, verified split/class mapping and **new** output root.
- [ ] Implement E05 `layer2-only` and propagate CLI/variant ID.
- [ ] Implement E06 arithmetic target; default geometric parity test.
- [ ] Configure E07B existing β sweep with **only** three new β values.
- [ ] Implement E07W custom layer2 weight; preserve unweighted depth CE.
- [ ] Implement E09 decoupled auxiliary-gradient schedules and per-epoch logs.
- [ ] Implement E02 frozen feature export + correctness probes; locate real `last.pt` checkpoints.
- [ ] Implement E16 H100 profiler and primary-only inference path; locate real SR/DS/DS+ `last.pt` checkpoints.
- [ ] Run unit tests, numerical old-default parity, strict config-reuse/skip and deferred-test integration tests; resolve blockers.
- [ ] Freeze exactly-87 run matrix and conduct Stage 1: **18 of 87** 100-epoch runs, **validation-only until locked test evaluation**.
- [ ] Conduct Stage 2: **69** more 100-epoch runs if quality gates pass.
- [ ] Run E02: 12 exports + 108 lightweight probe fits.
- [ ] Run E16: SR/DS/DS+ profiled, 3 rounds, measured inference/training/memory; real PCam loader throughput and real wall-clock telemetry from newly trained runs.
- [ ] Lock all validation decisions, perform deferred final test evaluation, validate new artifacts against effective-config/data hashes, hand off machine-readable outputs for **later** analysis.

### 12.3. Final handoff definition

The run owner delivers **new-only** checkpoint trees, per-run config/provenance, exported probes, cost benchmark traces, compact new-run summaries, and a run-status report identifying completed/failed/blocked runs. **No analyses of the previously available artifacts are required by this plan.** Negative results and failed validation gates are retained, not erased.

**End of execution-only plan.**

---

## 13. Final compatibility decision matrix (pre-execution, no result analysis)

| Job | New GPU execution | Requires historical checkpoint? | Hard fairness gate | If unavailable |
|---|---|---|---|---|
| E02 12 exports + 108 probes | Frozen forward and probe fitting | DS+ R50 `last.pt` by dataset/seed | Same frozen classifier; validation-only probe fit; no test training | Mark dependent job blocked; no DS+ retrain |
| E05 layer2-only (12) | 100-epoch R50 | Historical full and layer3 reference **metadata/results**; original checkpoint preferred for audit | Same original cache and training protocol; only depth-head set varies | Do not claim strict paired comparison |
| E06 arithmetic (12) | 100-epoch R50 | Historical full reference metadata/results | Only target aggregation changes; `weighted_nll` unchanged | Block matched claim |
| E07B β (27) | 100-epoch R50 | Historical β=0.5/full and β=0/final reference metadata/results | Same head training and schedule; β only | Block matched claim |
| E07W depth weights (18) | 100-epoch R50 | Historical full reference metadata/results | Only depth target q-mix weight changes, depth CE unchanged | Block matched claim |
| E09 schedules (18) | 100-epoch R50 | Historical full reference metadata/results | Only `_ScaleGrad` coefficient changes; meta warm-up fixed | Block matched claim |
| E16 cost profiling | Timed infer/train steps and data loading, plus telemetry of new runs | Real SR, DS, DS+ checkpoints for **checkpoint-realistic** result | Same GPU/precision/batch/input/cold-warm policy, correct optimizer groups; primary-only vs diagnostics distinct | Label architecture-only/synthetic estimate separately or block unsupported PCam-specific claim |

**Execution-only objective:** deliver **new** training/probe/profile outputs with a machine-checkable, honest fairness contract. **No #15 analysis, no old ablation retraining, no selection/analysis of existing ZIP or past data in this plan.**
