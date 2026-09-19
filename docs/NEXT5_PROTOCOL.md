# NEXT5 protocol — five independent selective-prediction hypotheses

Date locked: 2026-09-19. This file is the preregistered scientific protocol
for the `next5_pilot` suite. Defaults below are **initial engineering
choices**, not empirically optimized values. They are not retuned on
validation or test outcomes of this pilot. One-seed findings (seed 13)
remain exploratory.

Related-work anchors (precedent, not novelty proofs):

- SCROSS / cross-fitting: https://ojs.aaai.org/index.php/AAAI/article/view/26133
- Correctness Ranking Loss: https://proceedings.mlr.press/v119/moon20a.html
- Augmentation-based confidence: https://arxiv.org/abs/2006.16705
- Trust Score: https://proceedings.neurips.cc/paper/2018/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html
- CCL-SC: https://arxiv.org/abs/2406.04745
- FMFP: https://github.com/Impression2805/FMFP
- Normalized-logit confidence: https://proceedings.mlr.press/v244/cattelan24a.html

## 1. Shared recipe and data

- Datasets: CIFAR-10 and CIFAR-100.
- Backbone: VGG16-BN using this repository's CIFAR adapter (not torchvision's
  ImageNet VGG head).
- Recipe: `ccl_sc_reference` — 300 epochs, batch 64, SGD lr 0.1, momentum 0.9,
  weight decay 5e-4, LR × 0.5 at epochs 25, 50, …, 275.
- Official training pool 50k → stratified 45k train / 5k validation, split
  seed `20260902`. Official 10k test is untouched during training.
- Main seed: **13**. Backbone initialization and the primary
  data-order/augmentation stream use this seed. Extra heads, class embeddings,
  pair sampling, and neighborhood RNGs use dedicated streams derived from it
  (see each method). Warmup, if any, is inside the 300-epoch budget.
- Checkpoint selection (validation only): minimum validation AURC among
  checkpoints whose val accuracy is within `guard_delta_acc = 1.0` percentage
  points of the best val accuracy. One preregistered primary score per method;
  other scores are evaluated on **that same** selected checkpoint.
- Gradients, buffers, fold teachers, retrieval memories, replay/reference
  batches and fitted probes use TRAIN data only. Validation is only for
  declared checkpoint selection and diagnostics.
- Baseline `scsf_correctness` is correctness-BCE, attached feature taps,
  detached logits into the confidence head, decreasing cosine meta-weight.
  Legacy `scsf` (TCP-MSE) is not substituted.
- `sage_ds_v2` is the matched historical SAGE-v2 anchor, not a redesigned
  variant.

Identity:

- `run_name` follows the engine (`dataset-backbone-method[.score][.mode][.variant]-r<recipe>-s<seed>`).
- `scientific_hash` hashes the resolved scientific config with runtime
  keys removed (`device`, `results_root`, `run_name`, `data.root`,
  `data.num_workers`, `data.split_index_dir`, `method.oof_path`,
  `method.memory_path`). A run hash is `scientific_hash` (includes seed).
  Artifact file paths are locators, not scientific identity.
- `comparison_signature` groups seeds: same dataset, backbone, method,
  variant, recipe, method hyperparameters, source SHA; **seed excluded**.
  Variants are never collapsed because they share a factory class.

## 2. Primary score and extra controls

Always reported on the selected checkpoint (label-free):

- primary method score
- MSP, logit margin, negative entropy
- `normalized_logit`: max entry of \(\ell_2\)-normalized logits
  \(\max_c z_c / \|z\|_2\) (Cattelan-style control; not a claim of
  ImageNet transfer)

Score changes on a fixed checkpoint cannot change accuracy@100.

## 3. Method A — `crossfit_failure`

Hypothesis: in-sample correctness targets become uninformative as the
classifier memorizes train samples; excluded-fold teacher difficulty may
provide better representation supervision.

Construction:

1. Split the 45k TRAIN IDs into two deterministic stratified folds
   (seed `20260902`, class identity `index // per_class` matching the locked
   split convention). Store IDs, seed, and SHA-256 of each fold.
2. Train two CE teachers from fresh initialization (seed 13), each on the
   complementary 22.5k examples (`data.exclude_fold ∈ {0,1}`). Original 5k
   validation is used for checkpoint selection, **not** the excluded fold.
   Each teacher therefore receives fewer parameter updates than a full-data
   model at the same epoch count (disclosed).
3. Evaluate each selected teacher only on its excluded TRAIN fold with the
   **eval transform** (one fixed deterministic view). Cache sample ID,
   predicted class, correctness, logits, confidence, and teacher/checkpoint
   hashes. This is a training-artifact extraction role, not a test evaluation.
4. Fresh full-data student = `scsf_correctness` plus a training-only
   difficulty head \(d(\cdot)\) on `final_embedding`:

```
L = L_scsf_correctness + λ_difficulty * BCEWithLogits(d(x), stopgrad(teacher_error(x)))
```

Locked constants:

- `lambda_difficulty = 0.3`
- difficulty head: `Linear(D, 128) → ReLU → Linear(128, 1)`, extra RNG stream
  offset `+ 10_001`
- D = backbone `final_dim` (512 on VGG16-BN / ResNet-18)
- deployed primary score: student's current-correctness head `scsf_corr`
  (not `1 - teacher_error`, not an ensemble, not an oracle)
- difficulty module is excluded from `inference_modules`

Hard errors: missing IDs, duplicates, fold overlap, or mismatched
teacher checkpoint/source/split hashes.

Opt-in controls (not in the automatic primary pilot): `insample`,
`shuffled` (permute teacher errors within class), `distill` (CE to teacher
softmax). A teacher error is never claimed to be the student's error.

## 4. Method B — `candidate_verify`

Hypothesis: even a correctly classified training image supplies negative
class claims.

- Retain the standard C-way CE classifier.
- Shared class-conditioned verifier
  `q(h(x), embedding(c))` on `final_embedding`.
- Architecture (locked):
  - feature projection `Linear(D, 128)`
  - class embedding `Embedding(C, 128)`
  - bilinear-style concat `[h, e_c, |h-e_c|, h ⊙ e_c]` → `Linear(512, 128)`
    → ReLU → `Linear(128, 1)`
  - extra RNG stream offset `+ 10_002`
- Queries per example (selection detached): true class `y`; highest-logit
  incorrect class; `n_random_negatives = 3` uniform other classes.
  Deduplicate; if `C` is too small, drop extras rather than repeating.
  The verifier never receives a ground-truth indicator as an input feature.
- Loss: mean BCE-with-logits over the queried classes, with weights
  `{positive: 1.0, hard: 1.0, random: 0.5}` then divided by the sum of
  weights actually used on that example (no sampling-bias correction).
  `q` is therefore a **discriminative confidence score**, not a calibrated
  posterior. `lambda_verify = 1.0`.
- Inference: prediction = `argmax z`; primary score =
  `sigmoid(q(h(x), argmax z))`. No true labels, no full class sweep.

Opt-in controls: `ovr`, `scalar_correctness`, `second_ce`.

## 5. Method C — `intervention_rank`

Hypothesis: within-image comparisons reduce between-image difficulty
confounding.

- Two independently seeded views from the existing CIFAR training
  augmentation family (`RandomCrop(32, padding=4)` + `RandomHorizontalFlip`
  + Normalize). View seeds = `view_seed(data_order_seed, sample_id, {0,1})`.
  This family is **not** licensed for medical data without a separate
  label-preservation review.
- CE and current-correctness BCE on both views, then **mean over views**
  so adding a view does not double the base loss scale.
- Where one view is correct and one is wrong:

```
L_pair = mean_over_valid_pairs softplus(s_wrong - s_correct + margin)
L = mean_view(L_scsf_correctness) + λ_pair * L_pair
```

Locked: `lambda_pair = 0.5`, `margin = 0.1`. Pair identities and
correctness masks are detached and recomputed each step. Pairless batches
contribute a finite zero extra loss. Raw correctness-head logits are used
for the pair term. Inference: one ordinary view, same `scsf_corr` score.

Opt-in controls: `two_view_only` (no ranking), `cross_image`, `shuffled`,
`consistency`.

## 6. Method D — `neighbor_distill`

Hypothesis: unsupported high-confidence extrapolation can be addressed by
transferring local training-data support into a single-pass representation.

- Depend on the matched full-data CE anchor's **selected** checkpoint.
  Freeze it as the reference feature extractor (no extra teacher training).
- Train-only memory: L2-normalized `final_embedding`, labels, stable IDs.
  Default group id = sample id (CIFAR has no patient groups). Exclude self,
  known groups, and duplicate IDs from neighbors. Query and memory features
  share the frozen coordinate system (eval transform).
- Neighborhood sizes `k ∈ {8, 32}`. Distance = cosine distance
  `1 - <q, m>`. Weights = `softmax(-dist / τ)` with `τ = 0.1`. Class support
  = weighted histogram + uniform smoothing `ε = 1e-3`. Retrieval is chunked;
  an `N×N` matrix is never allocated unconditionally.
- Student: `scsf_correctness` plus two training-only linear support heads
  (`Linear(D, C)` each) predicting the detached k=8 and k=32 class-support
  distributions. Auxiliary = mean of two batchmean KLs
  (`log_softmax(head)` vs detached target). `lambda_support = 0.3`.
- Deployed primary score remains `scsf_corr`. Memory and support heads are
  not required at inference.
- Disclosure: self is excluded from retrieval, but the reference encoder
  **was trained on the query image**. This is not cross-fitted representation
  evidence.

Opt-in controls: `shuffled`, `frozen_features`, `contrastive`,
`direct_trust`.

## 7. Method E — `rank_sharpness`

Hypothesis: good correct/error ordering can be fragile to parameter
perturbations even when CE is relatively stable.

- Start from CE + correctness BCE plus a bounded sampled correct/error
  ranking loss on TRAIN only.
- Operating-window weights from empirical prefix AURC on coverages
  **[0.8, 1.0]**. For a batch of size B, ranks `r = 1..B` (1 = most
  confident, id tie-break). Let `k_lo = max(1, ceil(0.8 B))`, `k_hi = B`.
  An error at rank `r` receives

```
w(r) = |{k : k_lo ≤ k ≤ k_hi and k ≥ r}| / (k_hi - k_lo + 1)
```

  Pairs are incorrect×correct; pair weight = `w(r_wrong)`. Empty pair set
  → finite zero ranking loss and an **unperturbed** CE+BCE update with an
  explicit diagnostic. Never manufacture failure labels.
- Two passes of **one** update:
  1. Freeze pair identities and `w(r)` from pass 1. Backward ranking loss
     only (`create_graph=False`). Perturbation
     `ε = ρ ∇L_rank / (‖∇L_rank‖₂ + 10⁻¹²)` with `ρ = 0.05`, **detached**.
  2. Evaluate `L_CE + L_BCE + λ_rank L_rank` at `θ+ε` (same frozen pairs).
     Restore `θ`. One optimizer step on that gradient.
- Locked: `lambda_rank = 0.5`, `margin = 0.0`, `ρ = 0.05`, at most 32 errors
  and 32 corrects sampled per batch (deterministic top-rank subset).
- Both passes run in train mode (BN updates and dropout resample). Pair
  masks do not. Exception paths restore weights in `finally`.
- Inference: single-model correctness head, no weight perturbations.

## 8. `fmfp_reference` (mandatory comparator)

Port of official FMFP (MIT, https://github.com/Impression2805/FMFP):

- SAM two-pass on **CE only** (`rho=0.05`, non-adaptive), matching
  `utils/sam.py` + `train_fmfp.py` (`first_step` / second CE backward /
  `second_step`). Ranking criterion in the upstream `main_fmfp.py` is
  constructed but **not applied** in `train_fmfp.py`; this port does not
  invent a CRL term.
- SWA: running average of parameters after `swa_start = 180`
  (300-epoch analogue of official 120/201). Official code used
  `CosineAnnealingLR(T_max=100)` + `SWALR(swa_lr=0.05)` after epoch 120.
  **Deviation (disclosed):** this pilot keeps the shared `ccl_sc_reference`
  step schedule for the live SAM optimizer and still averages weights after
  epoch 180. This is FMFP's SAM+SWA mechanism under the shared recipe, not a
  bitwise paper reproduction.
- After `swa_start`, validation and checkpoint selection use SWA weights.
  Intermediate val does **not** run full `update_bn`. At training end, one
  `torch.optim.swa_utils.update_bn` over the train loader is applied to the
  selected SWA snapshot before deployment (official BN treatment).
- Primary score: MSP. Deployment uses the BN-updated SWA weights.

If this port were only generic SAM without averaging, it would be named
`sam_reference` and must not be presented as FMFP.

## 9. Pilot matrix

Per dataset `{cifar10, cifar100}`, seed 13, VGG16-BN:

Training jobs (11 × 2 = **22**):

```
ce
ce.fold_teacher_a          # exclude fold 0
ce.fold_teacher_b          # exclude fold 1
scsf_correctness
sage_ds_v2
fmfp_reference
candidate_verify
intervention_rank
rank_sharpness
crossfit_failure           # after OOF targets
neighbor_distill           # after CE memory
```

DAG:

```
fold teacher A + fold teacher B → validate OOF targets → crossfit_failure
matched CE → validate reference memory/targets → neighbor_distill
all training jobs → selected-checkpoint evaluation (val, then test)
complete compatible results → paired reports
```

Default concurrency: **one** live training job per GPU. Evaluation,
extraction, and memory construction are scheduler tasks, not extra hidden
training processes. Additional seeds, datasets, architectures, and tuning
grids are implemented as selectable configs but **not** launched.

## 10. Metrics

Locked full-prefix metrics (not a trapezoid over the sparse coverage grid).
Partial AURC / excess-AURC on coverages `[0.8, 1.0]` is the mean of prefix
risks for accepted sizes `k` with `k/N ∈ [0.8, 1.0]` (include endpoints;
empty window → NaN). Per-class coverage/risk uses the **same global**
confidence threshold as the micro coverage point. Single-seed SD is NA.

Between-method accuracy guard for later interpretation: 1 pp vs the matched
CE or `scsf_correctness` anchor (not the within-run checkpoint guard).

This benchmark is exploratory for new hypotheses. Confirmation seeds and an
untouched evaluation source are required before any superiority claim.
The old five-seed gate is not claimed here.

## 11. Smoke vs science

GPU/CPU smoke uses `recipe=next5_smoke` (tiny overfit, 2 epochs, forced
phase thresholds). Artifacts are marked `SMOKE_ONLY` and never aggregated
as scientific runs. Smoke AURC is not used to choose hyperparameters.
