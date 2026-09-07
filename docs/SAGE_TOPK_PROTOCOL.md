# SAGE-TopK: budgeted top-k selective deep supervision

Status: **single-seed exploratory pilot** (preregistration before any implementation).

Companion docs: `docs/EMPIRICAL_CONTRACT.md` (shared empirical contract),
`docs/SAGE_V2_PROTOCOL.md` (the selective surrogate SAGE-TopK inherits).

## 1. Identity

`method_name = "sage_topk"`. Backbone: VGG16-BN. Datasets: CIFAR-10 and
CIFAR-100. One seed: **13**. Recipe: `ccl_sc_reference` (300 epochs, batch 64,
SGD lr 0.1 / momentum 0.9 / wd 5e-4, multiplicative LR decay 0.5 every 25
epochs, per-epoch validation, checkpoint-selection rule from the shared
contract). **Total: two runs, one per dataset.**

DepthFrag, RiskFlow, and the SAGE-v3 certified allocation are **no longer part
of the active queue**; their code, checkpoints, and results are preserved
untouched. This pilot does **not** modify any existing method.

## 2. Locked settings (not claims of conventional defaults)

These are pilot constants. They are frozen before implementation and are **not**
to be tuned after inspecting pilot results (no tuning of K, profiling duration,
loss coefficients, or the selective target).

| Setting | Value |
|---|---|
| `K` (top candidates kept) | 2 |
| `seed` | 13 (per dataset) |
| profiling epochs | 0–4 (five epochs) |
| profiling utility interval | every 50th training batch |
| selection statistic | **mean cosine utility** |
| variance penalty | none (descriptive variance is logged only) |
| `B` (allocation budget) | 1 |
| `rho` (aux strength) | 1 |
| companion classifiers | linear (`native pooling -> Linear(d, C)`) |
| selective target | SAGE-v2 global selective surrogate (sec. 7) |
| meta refresh interval | 50 training batches + first post-profiling batch |
| backbone training | `ccl_sc_reference` (unchanged) |

The same settings apply to CIFAR-10 and CIFAR-100.

## 3. Method shape

SAGE-TopK has:
- automatic enumeration of registered backbone block-boundary candidates;
- a short profiling stage (epochs 0–4);
- fixed Top-K selection *after* profiling;
- normalized, classification-compatible auxiliary directions;
- a small convex allocation problem (`K x K` Gram).

It has **no** learned controller, no stochastic gates, no hard-concrete
mechanism, no auxiliary confidence MLP, no robust/class-weighted selective
objective, and no amortized solver. Class robustness is kept separate from this
allocation experiment.

## 4. Candidate sites and heads

For VGG16-BN the natural pooling-stage taps are the five registered taps
`pool1..pool5`. Candidates are obtained **through the backbone adapter**
(`backbone.taps`), never as a VGG-specific list inside `sage_topk.py`; aliased
representations are not duplicated.

> Limitation (documented): the method automatically selects among
> **adapter-exposed** candidates. It does not discover arbitrary computational
> boundaries in an unsupported network.

For each candidate:

```
native pooling -> Linear(feature_dim, num_classes)
```

CNN features use global average pooling. Transformer adapters may use their
documented native token readout (the framework's `_pool_tap` convention). No
LayerNorm, hidden layer, or nonlinear MLP.

The auxiliary objective is ordinary ground-truth classification CE:

```
L_aux_l = CE(head_l(feature_l), y)
```

## 5. Profiling: epochs 0–4

- Train the backbone with ordinary final-head CE only.
- Train **every** companion head on **detached** backbone features (so each
  head's own-CE gradient updates only that head's parameters).
- At every 50th training batch, on the *same* training batch, compute the
  candidate auxiliary gradients `g_l = grad L_aux_l` **with features attached**
  (measurement only) and the selective gradient `g_sel` on a **disjoint
  meta-batch** (eval mode, deterministic sampling from the held-out validation
  split). Gradient measurements are never applied to the backbone during
  profiling.
- Compute classification-compatible projected auxiliary gradients
  `tilde_g_l = proj(g_l)` against this batch's CE gradient (sec. 6 projection),
  then the **cosine utility** `U_l = <g_sel, tilde_g_l> / (||g_sel|| ||tilde|| + eps)`.
- Deterministic data (fixed meta-batch order) and tie handling (registration
  order). Accumulate mean utility and descriptive variance per candidate.
- At the end of epoch 4: select the **two** candidates with highest mean
  utility; **break exact ties by stable candidate order**; freeze the selection
  for the remainder of training; save selected sites and profiling statistics
  in checkpoints and manifests.

Top-K selects the candidate *subspace*. The later QP may assign zero weight to
either or both selected sites.

## 6. Training after profiling (epoch 5 onward)

- Compute auxiliary objectives and gradients **only** for selected sites.
- Unselected companion heads stop training; their gradients are not computed;
  unnecessary unselected feature tensors are not retained.
- The optimizer and learning-rate schedule stay continuous (same single
  optimizer and scheduler spec as `ccl_sc_reference`; no extra epochs).
- For each selected site, compute its auxiliary CE gradient, project away any
  component opposing the current main CE gradient, and normalize:

```
v_l = projected_gradient_l / (norm(projected_gradient_l) + eps)
```

Zero-norm directions are handled explicitly (zero vector -> allocation
coordinate contributes nothing).
- Selected companion-head parameters continue training under their own CE
  gradients.

## 7. Selective target and refresh policy

The **actual SAGE-v2 global selective surrogate** is reused verbatim:
`soft_aurc_surrogate(logits, targets)` (default MSP confidence, default
error_mode `"proxy"`, i.e. a *detached-constant-free smooth* error likelihood
`1 - softmax_true`), smoothing temperature `tau = 0.3` (the effective SAGE-v2
setting inherited from `hard_concrete_tau`), and the framework's locked coverage
grid (`COVERAGE_GRID_PERCENT`). It is **not** exact empirical AURC, and its
temperature/coverage weighting is **not** changed here.

- The meta batch is a deterministic validation batch (independent of the
  training data by the locked 45k/5k split; identical split hashes as the
  SAGE-V2 reference runs).
- The selective gradient is refreshed every 50 training batches and on the
  **first post-profiling training batch** (epoch 5, batch 0).
- Between refreshes the **detached, normalized** selective target is cached:
  `s = g_sel / ||g_sel||`. Selected auxiliary directions are recomputed from
  the current training batch; the allocation is solved with the cached target.
  Target age (steps since last refresh) is logged.

> Certificate caveat (state this in the report): the descent certificate is
> relative to the gradient used by the solver. Between refreshes it is a
> certificate against the **cached** target, not a guarantee about the current
> selective gradient.

## 8. Convex allocation

Let `V = [v_1, ..., v_K]` be the selected normalized directions, `s` the cached
normalized selective target, `G = V^T V`, `b = V^T s`. Solve:

```
min_lambda   0.5 * lambda^T G lambda - b^T lambda
subj. to     lambda >= 0
             sum(lambda) <= B                         (B = 1)
```

Solver: a **deterministic exact enumeration for K = 2** that checks the
interior (unconstrained minimizer, if feasible) and every boundary face
(`lambda_i = 0` (1-D clamped solution), and the `sum(lambda) = B` segment
(1-D quadratic minimization)) and takes the feasible candidate with the smallest
objective. Singular/collinear directions are handled **without** silently adding
a ridge penalty that changes the objective (zero rows/columns simply remove a
coordinate). A trusted small reference (fine grid + direct enumeration) is used
in unit tests to verify optimality.

Every individual direction is already CE-projected, so a separate CE constraint
is mathematically redundant in exact arithmetic. The final mixture is
**still verified numerically**. Checks (documented tolerance `TOL = 1e-6`):
- feasibility (`lambda >= -TOL`, `sum(lambda) <= B + TOL`);
- all values finite;
- objective no worse than `lambda = 0` within tolerance (objective(0) = 0);
- selective alignment `b^T lambda >= -TOL`;
- final mixture CE compatibility `<V lambda, g0>` normalized `>= -TOL`.

On any failed check (or a numerically zero allocation) the method applies the
**zero-update fallback** (`lambda = 0`) and records the event (zero-update
frequency is reported).

Apply, per step:

```
g_update = g_CE + rho * norm(g_CE) * V lambda            (rho = 1)
```

This is **gradient routing**, not an ordinary scalar weighted-loss sum: the
auxiliary contribution enters through normalized, normalized-vector-magnitude
directions after explicit normalization and projection. The theoretical claim
is **local and applies to the auxiliary contribution**; the pilot does **not**
claim guaranteed improvement in accuracy, actual AURC, every class, or the
total momentum-SGD optimizer step.

## 9. Inference

Only the final backbone classifier is used at inference; primary confidence is
**MSP**. Companion heads and all allocation machinery are excluded. Actual
training overhead (backbone CE / selected aux gradients / meta refresh / Gram
construction / QP + certification) is measured and reported. `K` limits the
number of selected auxiliary gradients; it does **not** guarantee identical
runtime across different depths.

## 10. Verification (before launch)

Tests lock: candidate enumeration and stable selection; linear companion-head
construction; profiling leaves backbone auxiliary gradients unapplied;
unselected heads/directions are not computed after profiling; Top-K boundary
and exact resume; QP feasibility and optimality against a trusted small
reference; collinear and zero-gradient cases; normalization and mixture CE
compatibility; cached-target refresh and age; label-free inference; preservation
of existing methods. The integrated suite must stay green.

Before the long runs an isolated GPU smoke crosses the profiling boundary,
exercises checkpoint save/resume and the final evaluation, and measures real
time per part (see sec. 8/9). Do not assume the QP is negligible merely because
its matrices are small: Python loops, CUDA launches, and synchronization can
dominate small solves.

## 11. Launch

After tests and smoke pass: forward commits to `origin/quan`; a fresh mirror
pinned to the final commit; a **new** results root; **exactly the two seed-13
runs**; batch size, optimizer, LR schedule, augmentations, dataset splits, and
checkpoint-selection rule identical to `ccl_sc_reference`; concurrency chosen
from smoke throughput/memory. Record source commit, dirty state, resolved
configuration, split hashes, selected sites, profiling statistics, and
environment. No additional seeds, backbones, methods, or ablations are launched
automatically.

## 12. Report

Paired against the existing matched SAGE-V2 seed-13 checkpoints on both
datasets: accuracy; AURC and exact excess-AURC; failure AUROC/AUPR; full
risk-coverage table; mean- and worst-class AURC; selected sites and profiling
utilities; allocation weights over training; zero-update frequency; profiling
cost and post-profiling epoch time; measured total training time and inference
overhead. All negative results are reported. A single seed supports an
exploratory decision only; it cannot establish statistical superiority or the
full baseline gate.