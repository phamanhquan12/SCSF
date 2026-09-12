# Review-aligned RC-Training Methods (R3 / DTR / CBR) — Implementation Specification

Self-contained English specification for implementing three new selective-
classification training methods on top of the existing `scsf` harness, plus a
review-aligned correctness baseline and a portable multi-method suite launcher.

The author's review document is **not redistributed**; this file is the
canonical specification. Collaborators must be able to implement and run
everything from this repository and file alone.

## 1. Provenance

### 1.1 Source review

* File: `SCSF_Risk_Coverage_In_Training_Review.pdf`
* Date: 2026-09-10
* SHA-256: `00188e1055d94d2d5b1c3885d705ff244a2807a1e669717c5170103f819bb3ba`
* Language: Vietnamese; 16 pages; page + section mapping in §14.
* Status: a research proposal ("đề xuất nghiên cứu"), not experimental
  results. It analyzes the SCSF baseline PDF
  (`ECCV_2026_SCSF (1).pdf`, 17 pages, correctness–NLL formulation — note that
  the *in-repo* README/`train_scsf.py` describe a TCP–MSE formulation; those
  differ, see §2.4).
* All numbers quoted from the review are the author's references, **not**
  evidence for the new methods.

### 1.2 Repository baseline at write time

* Git branch `quan`, HEAD `1c80ad4e3101f03ff3e011b7215d610fb5620d2d`.
* Existing methods preserved: `ce`, `dg`, `selectivenet`, `sat`, `scsf`,
  `ccl_sc`, `sage_ds` (+ aliases), `depthfrag` (+ aliases),
  `riskflow_v2`, `sage_ds_v2`, `sage_topk` (+ fixed-K variants),
  `depthfrag_v2`.
* Implementations added by this work: §3 lays out the shared package,
  §4–§8 the methods, §9 the launcher.

### 1.3 Code-only delivery

This handoff is **code-only**. Explicitly **not** attempted, and all such
runtime checks are labelled **NOT RUN** in `docs/VALIDATION_STATUS.md`:

* training / evaluation / benchmarks / GPU smoke tests / dataset downloads /
  schedulers / monitoring loops / remote jobs; no SSH to the experiment server
  is performed and nothing on `/root/scsf` or any registry/checkpoint is
  changed;
* executing the unit/integration test suite (tests are authored; commands are
  in §12);
* any claim of measured speedups, novelty guarantees, calibrated safety, or
  estimated completion times.

### 1.4 Scientific qualifications to preserve (from the handoff)

1. DTR is the review's most direct mechanism-led proposal, but a four-state
   head alone does not establish a useful repair mechanism. The
   correct-to-incorrect (1,0) subgroup must eventually be measured on
   held-out data and compared with ordinary auxiliary CE and reverse KD.
2. R3's `rho` is a local, two-view training response, **not** an estimate of
   irreducible error or a generalization guarantee. Its value over cheap
   confidence/CE proxies is untested.
3. CBR may improve per-class access while worsening micro-risk. Soft class
   floors are not guarantees on hard test coverage or under distribution
   shift. Sparse per-class support (CIFAR-100) is a central implementation
   concern.
4. The TopK pilot changed several components at once; it does not cleanly
   identify pruning as a degradation cause. Historical reports/provenance are
   preserved with an audit annotation (`docs/audits/sage_v2/`), never
   rewritten.
5. No method is a guaranteed accuracy or AURC improvement. Training objectives
   are surrogates; hard selective risk is evaluated separately.

## 2. Locked scientific definitions

### 2.1 RC curve and AURC

For classifier `f_θ` with per-sample predictions `ŷ_i = argmax_k f_θ(x_i)_k`,
error indicator `e_i = 1[ŷ_i != y_i]`, raw confidence score `s_i` and hard
acceptance `a_i(h) = 1[s_i >= h]`:

```
coverage(h) = (1/n) Σ_i a_i(h)
R(h)        = Σ_i a_i(h) e_i / Σ_i a_i(h)      (undefined when no sample kept)
```

Risk is evaluated at coverage `c > 0` by top-`k`, `k = max(1, floor(c·n))`,
sorted by descending confidence. Ties are broken by ascending sample id
(deterministic, independent of correctness — see §2.3).

### 2.2 Finite-sample AURC, weights W_n, swap identity (§5, p. 6)

Sort samples by descending score with stable-ID tie-breaking; `r_i = 1` is the
most confident rank. The locked full-prefix convention is:

```
AURC_n = (1/n) Σ_{k=1..n} (1/k) Σ_{j=1..k} e_(j)
       = Σ_i e_i W_n(r_i),     W_n(r) = (1/n) Σ_{k=r..n} 1/k
```

This is a summation identity (order swap), not a novel theorem. Two useful
consequences used in training:

* Repairing an error at rank `r` (fixing its label) while holding other ranks
  fixes AURC by exactly `W_n(r)`.
* Swapping an error at rank `r_i` with a correct sample at `r_j > r_i` (all
  others fixed) fixes AURC by exactly
  `Δ_ij = W_n(r_i) − W_n(r_j) = (1/n) Σ_{k=r_i..r_j−1} 1/k`.

For a partial RC target or a weighted coverage grid, `W` is replaced by a sum
over only the `k` of interest with matching integration weights. Optimizing a
partial RC is not the same as optimizing full AURC.

**Training-law**: in training, `n` is the batch (or reference pool) size;
ranks and `W` are **stop-gradient** weights, and the useful gradient always
flows through a differentiable score/confidence or classification objective —
never through the hard 0–1 error.

### 2.3 Metric vocabulary (§10.4, p. 15)

* **Risk fraction vs risk %**: `AURC 0.00059 == 0.059%` (Table 3 style). Both
  spellings must appear with their units.
* **Oracle / E-AURC**: the empirical oracle keeps the *same classifier's*
  error count and sorts every correct sample before every error;
  `E-AURC_n = AURC_n − AURC_oracle(n)`. Never use a shared oracle across
  classifiers with different error counts. E-AURC adds ranking information; it
  does not replace accuracy and AURC. If the old trapezoid convention must be
  compared with the paper, export a separate column computed by the **same**
  evaluator for **all** methods; never mix conventions.
* **Partial AURC [0.8, 1]**: report the normalized `(1/0.2) ∫_{0.8}^{1} R(c) dc`
  (divided by width 0.2), with a defined finite-sample step-curve integration
  convention.
* **Tie-breaking** is fixed and label-independent.
* **Operating thresholds**: full curves may use test scores top-k for ranking
  comparisons. A *deployable* threshold is chosen on **validation** only, at
  which point the achieved test coverage is reported separately. `coverage@risk≤r`
  read from a test curve is a descriptive metric that used test labels, not a
  learned threshold; the two reporting styles are kept separate.
* **Score vs mechanism gains**: with each trained backbone, evaluate both the
  new score and a selection-rank oracle (SR). A frozen-feature confidence-head
  control further isolates representation gains (§9.4 of the review). Claims
  distinguish: better backbone (acc@100 ↑, E-AURC ↓), better ranking only
  (acc@100 ≈, E-AURC ↓), ECE-only (not a win), better score on train only
  (overfit), or a fixed wrong surrogate (an implementation fix, not a
  contribution).

### 2.4 The review-aligned correctness baseline vs the current `scsf`

Current in-repo `scsf` (module `scsf/methods/scsf.py`) predicts detached
**TCP** with **MSE** and exposes `posthoc` / `e2e` / `legacy_partial_detach`
gradient modes. The review's locked reference instead uses **correctness-NLL**
(target `t = 1[argmax(z) == y]`, BCE with attached features and detached
logits, `error_weight = 1`, cosine decay to `1e-4`, logit-detach kept, no
end-to-end). The new method `scsf_correctness` (§4) implements the review
baseline. The existing `scsf` class, its config, its state dict and its
checkpoint compatibility are **preserved unchanged**.
The review's §3 maps the old `train_scsf.py` mismatches
(`--brier --nll` combos, `MetaCalibrator.forward` default keeping feature
gradients contrary to a "detach all" comment, unweighted NLL branch,
trapezoid-vs-sparse AURC inconsistencies, smaller eval MLP). None of these are
treated as new contributions; the new implementation is written cleanly on the
modular harness.

## 3. Shared architecture and data contracts

### 3.1 Files to add / change

```text
scsf/methods/scsf_correctness.py
scsf/methods/r3_scsf.py
scsf/methods/dtr_scsf.py
scsf/methods/cbr_scsf.py
scsf/methods/rc_training/            # shared utilities (below)
configs/methods/<method_id>.yaml
configs/suites/*.yaml
scripts/run_experiments.py
scsf/aggregate_v.py                  # variant-aware aggregation
scsf/data/two_view.py                # two-view train batch path
docs/RC_TRAINING_IMPLEMENTATION.md   # this file
docs/EXPERIMENT_SUITES.md
docs/VALIDATION_STATUS.md
tests/*                              # tests authored; see §12
tests/smoke_rc_training_gpu.py       # opt-in GPU smoke (NOT RUN)
```

Every exact method name is registered in `scsf/methods/factory.py` and
`__init__.py`; its config resolves through the real factory
(`tests` enforce it). Existing classes/state dicts are preserved.

### 3.2 Method interface contract

Methods subclass `scsf/methods/base.py:Method` (`nn.Module`):

* `train_loss(batch, state) -> dict` — every key scalar; additional scalar
  outputs that are **diagnostics only** carry a `diag_`/`_raw` suffix and are
  **never** accumulated into the optimized sum (no detached diagnostic must be
  optimized by accident). No extra optimizer steps may be hidden inside
  `train_loss`; all optimizer steps happen in the trainer, once per declared
  successful step.
* `predict_batch(x) -> MethodPrediction(logits, pred, confidence, scores)` —
  label-free inference by construction.
* `optimizer_specs()`, `scheduler_spec()` — parameter groups and LR schedule
  (see §3.3).
* `needs_indices: bool` — batch includes sample ids: `(x, y, idx)`
  when `True` (3-tuple). Used by R3/DTR for caches and two-view seeds.
* `needs_two_views: bool` — trainer routes to the two-view batch builder
  (4-tuple `(x, v, y, idx)`); see §3.5. Existing batch layouts are preserved
  for all existing methods.
* `inference_modules` — modules needed at deployment. For DTR this excludes
  the probe; for the family this is backbone + calibrator/head. Must return
  `nn.Module`s, never a bare `Parameter`.

### 3.3 Optimizer / schedule discipline

* One optimizer spec handles the backbone (and any extra trainable heads that
  should share CE updates, e.g. DTR probe) with the recipe LR; a second, Adam,
  spec with `meta_lr` handles the calibrator/confidence head. New family
  settings must **not** accidentally inherit TCP-specific or historical method
  overrides from recipes (method-layer defaults are evaluated against
  `ccl_sc_reference`).
* R3's CBR-style or DTR's dual variables / temperatures are **not** model
  parameters and are **not** in any model optimizer (see §7/§8 for the exact
  mechanisms).
* No incremental (un-sequentialized) changes to pretrain/warmup: warmups are
  explicit and *inside* the epoch budget, saved exactly by phase (see §5.5,
  §6.4).
* Determinism: primary backbone initialization, augmentation stream and
  data-order RNG are preserved when extra heads/views are added
  (`init_rng_key`, per-component RNG streams; see engine `seeding.py`).

### 3.4 Config system

* Layer merge order (locked by `scsf/engine/config.py`):
  `defaults < dataset < backbone < method < recipe < CLI`. Method-layer YAML
  files may use `extends: <other_method_id>` — a deep merge of the parent
  layer first, then the child overrides — so ablation aliases stay small.
  `engine/resolve` must accept the alias through the same factory path; a test
  locks alias resolution and `extends` semantics.
* `method.variant` field: a short stable string (e.g. `pool`, `pool_conv`,
  `full`, `alloc`) used to disambiguate runs whose *class and id* would
  otherwise collide (SAGE pool vs pool+conv TopK, R3/DTR/CBR ablations). It is
  represented in `run_name` (§9.4) and in aggregation (§10.4). A run's full
  identity = `(dataset, backbone, method_id, variant, recipe, seed)`.
* Config hashes: `config_hash` (full, includes seed) and
  `comparison_signature` (seed-independent) are recorded per run. Hashes are
  stable across seeds only where scientifically intended.

### 3.5 Two-view stable-ID training path (R3/DTR)

New `scsf/data/two_view.py`:

* `build_two_view_dataloader(...)` yields `(x, v, y, idx)` — an augmented
  primary image `x`, a second label-preserving view `v`, label `y`, and the
  stable global sample id `idx` (official-fold index).
* Views are generated **from the original images through the documented
  training transform** (`scsf/data/cifar.py:get_train_transform` made public
  and used as the single source of truth), never by improvising transforms on
  already-normalized tensors. The two views use the same transform pipeline
  with **deterministic per-sample/per-view seeds** derived from
  `(data_order_seed, sample_id, view_index)`; the RNG stream is captured and
  restored around each sample so the primary view's randomness is
  **identical** whether or not a second view is requested ("enabling extra
  views alone does not alter the primary forward" test in §12).
* Two-view determinism is checkpointed: the exact RNG state / open orders of
  both streams are reconstructed on resume (see §3.7).
* `needs_indices` and `needs_two_views` never change the *primary* `(x, y)`
  tuple ordering for existing methods.
* Classes/batches never load datasets or download weights during `import` or
  launcher planning.

### 3.6 Shared `rc_training` package

All new-family logic lives in `scsf/methods/rc_training/` (registered as a
subpackage of the methods package so it is importable by the factory):

* `schedules.py`
  * `meta_weight_cosine_decay(epoch, pretrain, total_epochs, start_weight,
    min_weight)` — the **new, correctly decreasing** schedule
    `lambda = min_w + 0.5*(start_w − min_w)*(1 + cos(pi*progress))`:
    starts at `start_weight`, ends at `min_weight`; returns `0.0` before
    `pretrain`; clamps `progress` to `[0,1]`. Endpoints and one-epoch
    definitions are locked by tests. **Does not modify** the legacy
    `scsf.meta_weight_cosine` (audit: §3.9).
  * `temperature_schedule(...)` — explicit, unvalidated default for CBR `Ts`
    (positive, annealed) and any KD temperature; endpoints locked.
  * `phase_fn(start_epoch, warmup, ...)` — explicit phase bookkeeping
    (warmup / joint / frozen), saved exactly.
* `calibration.py`
  * `RawScoreCalibrator(nn.Module)` — the **shared** review-aligned confidence
    head: takes tapped features (attached), detached logits, hidden MLP with
    the v1 architecture family, and a single sigmoid output (raw score
    `s = logit`; primary confidence `sigmoid(s)`). Used by `scsf_correctness`,
    R3, CBR. Features flow, logits are detached (`stopgrad(z)`) — the review
    rule.
  * `TransitionCalibrator(nn.Module)` — same inputs, 4-output head for DTR
    (`softmax` over fixed state order `[00,01,10,11]`; index `2*tS + tL`);
    primary deployed confidence = `softmax[01] + softmax[11]` (§6). The
    head's shared layers match `RawScoreCalibrator` capacity; final output
    width differs.
  * Stable raw logits for ranking, not `inverse_sigmoid` of saturated
    probabilities.
* `rc_weights.py`
  * `harmonic_weights(batch_sorted_desc_conf, retains)` = exact `W_n(r)`
    (§2.2), computed on detached confidence from a stable-ordered batch.
  * `coverage_window_weights(...)` — explicit coverage-window variant
    (sum over the `k` of interest with integration weights).
  * Normalization: distinguish *raw* harmonic weights, weights
    `normalized to mean 1`, sums and means in telemetry names
    (`w_raw_sum`, `w_bar_mean`, ...). Document normalization sets and
    clipping order (§5.2, §6.6, §7.4). Empty masks / empty pair sets → finite
    zero losses. Avoid `O(n^2)` tensor materialization unless bounded by
    explicit config.
  * `pairwise_d(W_ranked_w)` — `|W(r_i) − W(r_j)|` detached (R3) using
    structured kernels bounded by config, not full `(B,B)` unless configured.
* `losses.py`
  * `weighted_bce_correctness(s, t, w)` — `BCEWithLogits` mean over `w`
    (incorrect-only weighting).
  * `rank_pair_loss(...)` — `d_ij*(ε_r + 1 − ρ_i)*T*softplus((margin +
    s_i − s_j)/T)` normalized by `Σ d_ij + eps` (R3; empty set → 0).
  * `conditional_reverse_kd(...)` — detached teacher `softmax(probe/T)`,
    student `log_softmax(final/T)`, `d_i * W̄_i * T² * KL`/`Σ d_i + eps`
    (DTR).
  * `confusion_loss(logits, pred_probs, scores, h, Ts, edges…)` —
    `τ log((1/|E|) Σ exp(U_ab/τ))` (CBR), per-coverage with normalized
    weights; `U_ab` soft confusion per class pair over accepted mass.
  * `soft_quantile_derivative_thresholds(...)` — bisection solve + analytic
    implicit derivative `a(1−a)/Σ a(1−a)` as a **custom autograd function**
    (§7.3).
* `views.py` — seed derivation and two-view helpers abstracted from §3.5
  (primary-view identity tests).
* `state.py` — buffer-backed `nn.Module` state containers so checkpoint/
  `state_dict` carries every method state:
  * `RhoCache` (sample-id → `(rho, epoch_measured)`, keyed rho = valid when
    age ≤ 1 epoch),
  * `DualState` (CBR `nu_ac`, support/edge masks, residual EMA),
  * `EdgeSupport` (periodic train-only edge-support statistics),
  * `PhaseState` (warmup/joint counters, refresh counters, sample-id history
    for "future corrected / newly wrong" logging).

### 3.7 Checkpoint / resume

Every method state needed for continuation is in `state_dict` (buffers in
§3.6, RNG snapshots, trainer phase counters, sample/view RNG streams). Resume
is exact on tiny synthetic fixtures at phase/refresh/cache boundaries
(§12). Selection and registry mechanics are unchanged
(`SelectionTracker` guard, append-only registry locked `BASE_COLUMNS`).

### 3.8 Capability map and backbone tap roles

Static, validated at **launch-plan and factory** levels; unsupported combos
fail at plan validation, never silently re-backbone/site-select/method-switch.

| method family                         | supported backbones           |
|---------------------------------------|-------------------------------|
| `ce`                                  | all 5                          |
| `dg`, `sat`, `selectivenet`, `ccl_sc` | vgg16_bn, resnet18             |
| `scsf`                                | vgg16_bn, resnet18 (and legacy)|
| `scsf_correctness`, `r3_scsf`, `dtr_scsf`, `cbr_scsf` | vgg16_bn, resnet18 |
| `sage_ds`, `sage_ds_v2`, `sage_topk` fixed-K, `depthfrag_v2`, `riskflow_v2` / v3 | vgg16_bn |
| historical `depthfrag`/`riskflow`/`sage_*` aliases | as documented historically |

Tap roles for the family (adapter `roles`):
SCSF pair order is `[top_l2, top_l1]` with the older/shallower tap first.

* **VGG16-BN**: `top_l1 = pool5`, `top_l2 = pool4`. Native block order is
  **Conv2d → ReLU → BatchNorm2d** (taps read after BN), with dropout in the
  feature stack — the adapter is correct as-is; do **not** reorder it to match
  a generic Conv-BN-ReLU description. Documented real endpoints live in
  `scsf/backbones/vgg.py` docstring and §3.8 table.
* **ResNet-18**: `top_l1 = layer4`, `top_l2 = layer3`.
* **DTR probe role** (`dtr.probe_role = top_l2`): VGG `pool4`, ResNet `layer3`
  — one probe only (`GAP → Linear(C)`).
* **R3 omega subset** (`r3.omega_prefixes`, per-backbone named-parameter
  prefixes on the **final block + classifier** only): VGG16-BN = classifier
  plus feature-block parameters strictly after `pool4` up to `pool5`; ResNet-18
  = `layer4` + `fc`. Validated by name against the actual state dict; the
  virtual update never touches earlier blocks.

### 3.9 `meta_weight_cosine` audit (summary)

`scsf/methods/scsf.py:meta_weight_cosine` computes
`min_weight + 0.5*(start_weight − min_weight)*(1 − cos(π·progress))`. At
`progress = 0` this equals `min_weight` and at `progress = 1` it equals
`start_weight`: it **starts at the minimum** and climbs to the maximum
inverted relative to its "decayed" description. The existing behavior,
callers (`scsf` and `train_scsf.py`) and the locked test
(`tests/test_methods.py::test_scsf_meta_weight_cosine_schedule`) are
**unchanged**. Full audit: `docs/audits/meta_weight_cosine_audit.md`.

## 4. `scsf_correctness` — review-aligned correctness baseline (§3, 5; pp. 3–4, 14)

Classifier logits `z`, tapped features `h` (attached), raw confidence logit
`s = Q(h, stopgrad(z))`, detached correctness target `t = 1[argmax(z) == y]`:

```text
L_base = CE(z, y) + lambda(epoch) · mean( w · BCEWithLogits(s, t) )
w      = error_weight when t = 0, else 1
```

* Default `error_weight = 1`; if configured otherwise, weighting applies to
  **incorrect examples only**, never to the mean's denominator wholesale.
* Features attached; logits detached; targets detached.
* `lambda(epoch)` = `meta_weight_cosine_decay` (starts `init_meta_weight`,
  reaches `min_meta_weight`; `pretrain` epochs train CE only and are inside
  the budget). Config in `configs/methods/scsf_correctness.yaml`
  (`mode` unused; this is correctness-BCE, **distinct** from `scsf`'s TCP-MSE).
* Primary confidence = `sigmoid(s)`; raw `s` saved too. Exposed secondary
  scores on the same classifier: MSP, margin, negative entropy, consistently
  oriented energy (`scores.py` names reused).
* Named "review-aligned", not a verified paper reproduction — unspecified
  original-paper/supplement details are documented assumptions.
* `predict_batch` etc. identical in shape to `scsf`.

## 5. Shared RC weights

Exact harmonic `W_n` and coverage-window variant (§2.2); `rc_weights.py`
implements both, plus normalization/clipping. Semantics shared across the
family:

* ranks, hard errors and weights are **detached training weights**; the useful
  gradient flows through CE, ranking or soft-acceptance losses;
* their fixed-rank RC identities describe one intervention in isolation, not a
  finite-step SGD guarantee;
* normalization sets and clipping order are explicit:
  1. compute `w_raw = W_n(r)` on the stable-ordered detached confidence;
  2. per-error-normalize to `w_bar` (mean 1 over the incorrect set) where a
     method requires it; clip large values (`rc.clip_max`), never silently;
  3. log both raw sums and normalized means separately.

## 6. DTR-SCSF — Depth-Transition Repair (§7; pp. 8–11)

Configured `dtr_scsf`, probe at `top_l2` (VGG pool4 / ResNet layer3) through
adapter roles. SCSF tap pair unchanged; **not** coupled to TopK.

### 6.1 Four-state transition confidence

`tS` = probe correctness, `tL` = final correctness. Fixed order `[00,01,10,11]`,
state index `2·tS + tL`. `Q` produces four logits from attached features and
detached final logits:

```text
confidence_final = softmax(Q)[01] + softmax(Q)[11]
```

Never use "either classifier correct" probability as confidence for the final
prediction. Deployment needs backbone + Q only (probe excluded from
`inference_modules`). Probe diagnostics are reported separately and
optionally.

### 6.2 Conditional shallow→final distillation

Two label-preserving views `u, v`; select samples where the probe is correct
on **both** views and the final classifier is wrong on the primary:

```text
d_i = stopgrad( probe_correct(u_i) · probe_correct(v_i) · (1 − final_correct(u_i)) )
L_KD = Σ_i d_i · W̄_i · T² · KL( stopgrad(softmax(probe_logits_u / T))
                              ‖ softmax(final_logits_u / T) ) / (Σ d_i + eps)
```

* W̄ from the **DTR primary score's** current stable batch ranks (not TCP).
* Teacher distribution, labels, mask and RC weights detached; the student
  path updates final classifier + shared backbone. The probe path updates the
  probe + shared features via its own CE. Shared params moving through the
  student path are **not** a teacher-gradient leak.
* KD applied on (1,0) only; no KD on (0,1) — protects correct-and-repairable
  cases from shallow-distillation harm. Zero mask → `L_KD = 0` (finite).

### 6.3 Objective

```text
L = CE_final + beta · CE_probe + lambda(epoch) · CE_four_state + gamma · L_KD
```

`CE_four_state = −mean log π_{tS_i, tL_i}`. `lambda` uses the new cosine decay.
Warmup (final + probe CE only) is explicit, inside the epoch budget, and
**saved/resumed exactly by phase** (adopted unvalidated default: document in
the config as `*_warmup_epochs`, `0` for the ablation set unless stated).

### 6.4 Variants (explicit named configs, `dtr_ablations.yaml`)

1. `dtr_aux_ce` — auxiliary probe CE only (no transition head, no KD).
2. `dtr_rev_kd_all` — reverse KD on all samples.
3. `dtr_rev_kd_correct_probe` — KD only where probe correct, without the final
   wrong condition.
4. `dtr_state_norepair` — four-state head, no KD.
5. `dtr_cond_norc` — (1,0)-conditioned KD without RC weights (W̄ = 1).
6. `dtr_full` — full method.
7. `dtr_rand_states` — seeded random auxiliary-state labels, same head
   capacity, as a plain-regularization control.

Head capacity and views matched across variants; all differences disclosed.
MSP and disagreement diagnostics are exposed without replacing the
preregistered primary score.

### 6.5 Logging / diagnostics

Four-state counts (train; optional probe-frozen diagnostics separately),
accepted-error state composition at 90%/95%, repair-mask frequency, future
corrected / newly wrong sample IDs, probe confidence and `d_i` coverage. Before/after
transitions are **not** interpreted as causal proof of repair.

### 6.6 Frozen-backbone probe diagnostic workflow (optional, NOT fit now)

Author `dtr_probe_diag.py` runner: trains `GAP → Linear(C)` on frozen backbone
features only, on training data, to measure the amount of class signal
available at the probe site. It must not select taps/hyperparameters using
test labels; it must not be used to fabricate claims. Not executed in this
delivery.

## 7. R3-SCSF — Repair-or-Rank (§6; pp. 7–8)

Configured `r3_scsf`. For sampled **incorrect** training examples, two views
`u, v`; on named final-block+classifier parameter subset `omega` (§3.8):

### 7.1 Virtual repairability measurement

```text
omega_plus_i = omega − eta_virtual · grad_omega CE(f(u_i), y_i)
rho_i        = stopgrad( clip( (CE_before(v_i) − CE_after_virtual(v_i))
                             / (CE_before(v_i) + eps), 0, 1 ) )
```

* One virtual step **per measured example** (`functional_call` on detached
  param clones); a single pooled batch step is a different algorithm and is
  **not** substituted silently.
* Isolation: never touches live model `state_dict`, live `.grad`, optimizer or
  momentum state, BN running statistics, or RNG. Module modes (train/eval) are
  defined and restored; before/after measurement has identical stochasticity.
  BN training on singleton inputs is avoided (examples measured under eval-BN
  with running stats that are never mutated).
* `rho` detached; no retained second-order graph in this first version.
* Sampling: explicit config `rho.refresh_interval` (default 10 steps),
  `rho.max_errors` (default 8), review-suggested 4–8 × per 10 steps is an
  unvalidated bounded default; unmeasured examples get `rho = 0.5` or a valid
  stable-ID cache value (age ≤ 1 epoch, checkpointed). Sampling is train-only.

### 7.2 Ranking + repair objectives

For `i` incorrect, `j` correct:

```text
d_ij    = stopgrad( |W(r_i) − W(r_j)| )
pair    = T · softplus((margin + s_i − s_j)/T)
L_rank  = Σ_{i,j} d_ij · (ε_rank + 1 − ρ_i) · pair / (Σ d_ij + eps)
L_repair= Σ_{incorrect i} W̄_i · ρ_i · CE_i / (number_incorrect + eps)
L       = L_base + beta · L_rank + gamma · L_repair
```

* **Not** normalized by the sum of `rho` (or its complement) — that cancels
  routing strength.
* Pair weighting for already-correctly-ordered pairs is margin protection, not
  a realized RC gain (documented).
* Missing one of the two classes in a batch → the ranking term is skipped
  (`0`), never NaN.

### 7.3 Gradient routing semantics

Repair updates classifier + backbone; ranking updates calibrator + intermediate
features (logits into the calibrator remain stop-grad like the baseline);
`rho`, hard labels/ranks/weights are detached; no meta-gradient through the
virtual step.

### 7.4 R3 variants (explicit configs, `r3_ablations.yaml`)

`r3_uniform_rank` (d_ij=1), `r3_rc_rank` (RC ranking, ρ=0.5), `r3_hard_ce`
(extra CE on hard examples, same budget), `r3_fixed` (ρ≡0.5), `r3_proxy_*`
(ρ replaced by confidence / current CE), `r3_shuffled` (distribution-matched
shuffled ρ control), `r3_detach_feat` (detached confidence features), and
`r3_full`. Proxy mappings and loss scales are explicit config, never
results-based invented settings. Logged: measurement time, cache age/hit rate,
corrected vs reranked error transitions, proxy values.

## 8. CBR-SCSF — Confusion-Budgeted Risk (§8; pp. 11–13)

Configured `cbr_scsf`. One raw confidence score `s`, one shared positive
temperature `Ts`, coverage grid `C = {0.70, 0.80, 0.90, 0.95}`.

### 8.1 Soft coverage masks with implicit threshold

```text
a_ic = sigmoid((s_i − h_c)/Ts),     mean_i(a_ic) = c
```

`h_c` is solved numerically (bisection with explicit tolerance/bracketing) so
the mask constraint holds; the **derivative of the solution** is implemented
(not a detached threshold):

```text
∂h_c/∂s_j = a_jc(1−a_jc) / Σ_i a_ic(1−a_ic)
```

Implemented as a custom autograd function so every mask's backward includes
the threshold's contribution. Tests: constant-score shift invariance, finite
differences, fixed-coverage Jacobian row/column sums, small-temperature /
tie handling, and a documented saturation fallback diagnostic (never an
unexplained epsilon that voids the constraint gradient). `Ts` is a positive
configured schedule initially, not a learned parameter needing an omitted
derivative.

### 8.2 Objectives

For true class `a` and `a != b`:

```text
phi_a(c)     = Σ_{y_i=a} a_ic / n_a
U_ab(c)      = Σ_{y_i=a} a_ic·p_θ(b|x_i) / (Σ_{y_i=a} a_ic + eps)
L_conf(c)    = τ · [ logsumexp( U_ab(c)/τ over supported edges )
                     − log(number_supported_edges) ]
L_micro(c)   = Σ_i a_ic·(1 − p_θ(y_i|x_i)) / (B·c)
L            = L_base + eta·Σ_c w_c·L_micro(c) + beta·Σ_c w_c·L_conf(c)
               + Σ_{supported a,c} nu_ac·(kappa·c − phi_a(c))
nu_ac        ← clamp( nu_ac + lr_dual · detached_residual_ac, 0, dual_max )
```

* Aggregate **per coverage** with explicit weights `w_c ≥ 0, Σ w_c = 1`
  (equal by default); never one max over all coverages.
* Edge support: only sufficient-support train edges (periodic train-only
  `EdgeSupport` update, checkpointed); source-class/edge support criteria are
  explicit; missing/unsupported classes or empty edge sets are **not**
  zero-risk observations — skipped contributions and soft accepted
  denominators are recorded.
* Descent on model parameters, **ascent** on duals; `nu` never goes into any
  model optimizer. Dual updates occur once per declared successful optimizer
  step (after a finite `train_loss`), never during repeated
  forward/evaluation passes. Residual EMA optional; absent classes are skipped
  without decaying/updating their duals from fabricated residuals.
* Floor applies to true classes during training (not error groups); labels
  never enter deployed thresholding.

### 8.3 Sampling law

Default **natural-prior** sampling. Class-balanced sampling is **rejected for
CBR explicitly** (config validation error) unless and until it importance-
weights all relevant global sums plus the threshold equation/implicit
derivative consistently (§12 weighted-derivative tests). A biased partially
weighted implementation is never advertised as prior-correct.

### 8.4 Diagnostics

Score variance/ties, soft residuals, hard per-class coverage, hard accepted
confusion at 90/95%, full accuracy, absolute error counts. Soft feasibility
coexisting with poor hard ranking is reported, not hidden.

### 8.5 CBR variants (`cbr_ablations.yaml`)

`cbr_global_multicov` (no confusion / no floor), `cbr_conf_nfloor`,
`cbr_floor_nconf`, `cbr_full`, `cbr_true_class_group` (true-class group risk
instead of edges), `cbr_single_cov`, and controls `ce_weighted`
(class-weighted CE) and `cbr_groupdro` (full-coverage group-DRO by true class,
noninferiority discipline). No invented clinical cost matrix or hardcoded
problematic class.

## 9. SAGE / TopK preservation (§2, §10 of handoff)

* Selectable existing experiments: `sage_ds_v2`; pool-only fixed-K TopK;
  expanded pool+conv fixed-K TopK; `dynamic_k` stays a **separately named
  opt-in** experiment (`sage_topk_dynamic`), never the hidden default of the
  fixed-K comparison. For the candidate-set comparison, force
  `dynamic_k=false, K=2`, otherwise-identical settings, and distinct
  `variant` strings (`pool`, `pool_conv`).
* Cosine-correction check: after normalizing the selective direction to unit
  norm, do **not** divide by the original selective-gradient norm again;
  bound/reconstruction tests are authored in §12; historical run statistics
  are preserved, not rewritten.
* Deterministic candidate names/order for `sage_topk`; extra taps alone must
  not alter final forward outputs (forward-identity tests).
* Historical `depthfrag`, `riskflow`, `sage_v3` remain selectable by explicit
  `--methods`; no new versions or experiments for them in this handoff.

## 10. One portable launcher (§11 of handoff)

### 10.1 Interface

```text
scripts/run_experiments.py
  --suite all|review|sage|<family-suite>          default: all
  --methods m1 m2 ...                             explicit subset / historical
  --datasets cifar10 cifar100
  --backbones vgg16_bn
  --seeds 13 [17 23]
  --data-root PATH --results-root PATH
  --recipe ccl_sc_reference
  --devices cuda:0 [cuda:1 ...]
  --max-jobs N                                    default = #devices, start 1/GPU
  --num-workers N  --python /path/to/python
  --dry-run | --execute                           mutually exclusive; --execute required
  --resume  --continue-on-error
  --download                                       opt-in; no effect in planning mode
  --overrides k=v [k=v ...]                        safe, quoted
  --list-suites  --list-methods
  --plan-artifact PATH                            plan-only output
```

* `--dry-run` / plan mode: pure-Python matrix resolution — no dataset/model
  construction, no torch import, no CUDA init, no download, no process launch.
  The plan replicates the engine's layer merge for `run_name`/`config_hash`
  parity (locked by a parity test against `engine.resolve`).
* `--execute` is mandatory to launch; mutually exclusive with `--dry-run`.
* Planning with unsupported combinations fails loudly (capability map §3.8),
*before* any job is scheduled.

### 10.2 Suites (exact IDs; `configs/suites/*.yaml`)

* `review`: `ce`, `scsf_correctness`, `r3_scsf`, `dtr_scsf`, `cbr_scsf` (5).
* `sage`: `ce`, `sage_ds_v2`, `sage_topk_v2_fixedk2_pool`,
  `sage_topk_v2_fixedk2_pool_conv` (4); the two TopK ids share a class with
  distinct variants, `dynamic_k=false`, `K=2`, otherwise identical.
* `all`: deduplicated union of `review` and `sage` = **8** primary ids; at the
  documented example (2 datasets × 1 backbone × 1 seed) that is exactly **16**
  planned runs, not 16 concurrent workers.
* `r3_ablations`, `dtr_ablations`, `cbr_ablations`: the variant configs of
  §6.4/§7.4/§8.5 (never silently included in `all`).
* Historical methods are selectable via `--methods` and discoverable with
  `--list-methods`; `all` means the documented primary suite, not every
  historical alias/frozen-checkpoint workflow.

### 10.3 Execution contract (authors of §11 served to collaborators)

1. Resolve/validate the full matrix; print methods, variants, seeds, recipe,
   prerequisites, data path, source commit, and job count before executing.
2. Capability, required-checkpoint, and data-presence checks. Frozen-method
   prerequisites are never silently replaced by fresh training.
3. Training → validation-selected checkpoint → final val/test evaluation →
   aggregation under one command. Val-only workflows are allowed but never
   pretend test evaluation ran.
4. Reproducible run IDs carry variant identity + config hash; pool vs pool+conv
   and same-class ablations never collide. Full run hash and seed-independent
   comparison signature both recorded.
5. Overwrite refusal on conflicting provenance; resume requires matching
   source/config/splits and restores method state; changed experiments get a
   fresh directory. `SCSF_SOURCE_COMMIT` is accepted only if it equals the
   real checkout HEAD.
6. Full git SHA, dirty state, config, split IDs/hashes, software/device info,
   requested concurrency, phases, completion states recorded. Clean source by
   default; allowed dirty runs are visibly marked and record a recoverable
   diff.
7. Portable GPU mapping — one job/GPU by default; no one-copy-per-GPU-per-rank
   duplication.
8. Args are built with `shlex.quote` where whitespace is present; scheduler
   parsing upgraded to `shlex.split` (backward compatible). No shell
   interpolation of user overrides.
9. Atomic per-run outputs; interruptible, persistent logs, resumable
   scheduling; failed outputs never deleted; no auto cloud rental / package
   install / background SSH.
10. Fresh-clone install instructions with explicit torch/CUDA guidance; no
   invented exact CUDA compatibility claims.

### 10.4 Run naming

`run_name = "<dataset>-<backbone>-<method_id>[.<variant>]-r<recipe>-s<seed>"`,
where `method.variant` (pool vs pool_conv, ablation tag) is folded in so
same-class ablations never collide. Example:
`cifar10-vgg16_bn-dtr_scsf.dtr_full-rccl_sc_reference-s13`.

## 11. Evaluation and aggregation (§12 of handoff)

* Reuse the locked full-prefix evaluator; extend additively/versionedly; AURC
  is never replaced by a sparse-grid trapezoid, and historical registry
  meanings never change.
* Per primary score/selected checkpoint: accuracy/error; full-prefix AURC;
  same-error-count oracle AURC and excess-AURC; failure AUROC/AUPR (error
  positive, `u = −confidence`); the locked grid
  `100,99,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,1` with risk
  fractions, percentages, retained counts, cumulative errors; pointwise
  oracle-adjusted risk with the same empirical error count / actual k; nested
  stable-ID tie sets, `risk@100 == error` verified; normalized partial AURC and
  partial excess-AURC on [0.8,1] with defined step-curve convention;
  class-conditioned AURC and per-class coverage/risk at the **same global
  thresholds** (with class sizes, error counts, accepted confusion, ties,
  top-confident error IDs/ranks); method score + MSP/margin/negative-entropy/
  oriented-energy all on the **identical selected checkpoint** (no per-score or
  per-coverage re-selection); diagnostic coverage@risk from labeled test curves
  kept distinct from deployable val thresholds; train time, phase/component
  timing, peak VRAM, number of views/updates, deployment-only parameter counts.
* `scsf/aggregate_v.py`: aggregation keyed by `(method, variant, recipe,
  backbone, dataset, score, split, comparison_signature, intended seed set)`.
  The old aggregator groups only by method/score/recipe and can pool
  same-class ablations; the family paths never feed a mixed suite to it
  without the variant-aware layer.
* Seeding discipline: pair seeds/IDs, list missing/failed cells, never pick
  the best 3 seeds; with one seed show single-run values + `std=NA`. Multi-seed
  rows use sample SD and correctly labelled paired differences/intervals.
  Bootstrap uncertainty stays distinct from training-seed variation.
* No historical five-seed gate passes automatically; review-family results are
  exploratory until the planned seeds/controls complete. The within-run
  validation checkpoint guard stays distinct from a between-method accuracy
  noninferiority criterion chosen before evaluation.

## 12. Tests to author (NOT RUN in this delivery)

Commands are recorded in `docs/VALIDATION_STATUS.md`. CPU unit tests use tiny
synthetic inputs; launcher tests mock subprocess/model/data creation; no real
CIFAR downloads required; GPU-marked tests are opt-in.

* Factory/method registration, config `extends` resolution, loss composition,
  label-free inference, `inference_modules` returns `nn.Module`s.
* `scsf_correctness`: correctness-vs-TCP target distinction, feature/logit
  detach routes, incorrect-only weighted BCE, schedule endpoints, one-step
  `train_loss` (no hidden optimizer steps).
* `rc_weights`: harmonic identity (`Σ_i e_i W_n(r_i)` equals prefix-mean
  formula), swap identity under fixed ordering, ties, empty error/correct
  pairs, normalization/clipping order, finite outputs.
* R3: virtual-step param/buffer/optimizer/RNG isolation, no retained higher-
  order graph, two-view use, rho bounds/default/cache expiry, routing-strength
  preservation under constant rho (no normalization by Σrho).
* DTR: all four state labels and marginals, no KD on (0,1), correct KL
  orientation with detached teacher, zero-mask behavior, warmup/resume phase
  exactness, label-free inference without probe.
* CBR: threshold residual/nesting, analytic backward vs finite differences,
  shift invariance, small-temperature/tie handling, ascent sign, absent-class
  behavior, checkpointable dual/support state; weighted-derivative tests if
  balanced sampling is ever enabled.
* Identity: extra taps/views do not change primary init/forward or primary
  augmentation; exact continuation on tiny synthetic fixtures at
  phase/refresh/cache boundaries.
* TopK: cosine boundedness, raw-to-normalized reconstruction, fixed-vs-dynamic
  separation, forward identity with extra taps.
* Launcher: suite Cartesian counts/dedup (16-run example), unsupported combos,
  spaces in paths, dry-run launch prohibition, resume conflicts, failure exit
  statuses, portable GPU mapping, variant-aware aggregation, non-overwriting
  score outputs, plan/engine `config_hash` parity.

## 13. GNU-style notes

* No method is claimed novel in the standalone sense. Antecedents named by the
  review: SDN (overthinking, ICML 2019), CRL (confidence ranking, ICML 2020),
  SoftRank-AURC (Zhou et al. v1, 2025), Learning to Reweight (virtual
  updates, ICML 2018), ReSIDe 2026 (multi-depth confidence + preference
  optimization, post-hoc), TULIP (ICML 2024), Student Self-reflection (IJCV
  2023), Fair SC via Sufficiency (ICML 2021), SelectiveNet (ICML 2019),
  ConfidNet (NeurIPS 2019), CCL-SC (ICML 2024), group-magnification of SC
  disparities (Jones et al., ICLR 2021).
* No requirement to add ResNet-50 / ViT; generality discussion alone does not
  mandate new backbones.

## 14. Page → implementation mapping

| Review pages | Content | Implemented in |
| --- | --- | --- |
| 2–5 | Scope, "increase accuracy" reading, baseline analysis, strengths/weaknesses | §2, §4, §3.9, `docs/audits/meta_weight_cosine_audit.md` |
| 5–6 | Shared AURC/W_n foundations | §2.2, §5 (`rc_training/rc_weights.py`) |
| 6–8 | Methods overview + R3 (pp. 7–8) | §7 (`r3_scsf.py`, `rc_training/losses.py::rank_pair_loss`) |
| 8–11 | DTR (pp. 8–11) | §6 (`dtr_scsf.py`, `TransitionCalibrator`, `conditional_reverse_kd`) |
| 11–13 | CBR (pp. 11–13) | §8 (`cbr_scsf.py`, `soft_quantile_derivative_thresholds`, `confusion_loss`, state) |
| 13 | Novelty matrix / controls | §13 |
| 14–15 | Protocol: splits, budget, datasets, metrics | §2.3, §10–§11 |
| 16 | Source-change map, decision rules | §3–§9 file map, §12 |
| handoff §4–§13 | Shared/data/launcher/test contracts | §3 (shared), §9 (launcher), §11 (eval), §12 (tests) |