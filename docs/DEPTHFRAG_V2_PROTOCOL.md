# DepthFrag-V2 protocol: stable whitened depth fragility

Status: **preregistered protocol**. Implemented as the `depthfrag_v2` method
alias; `depthfrag` and `depthfrag_warm25` are preserved unchanged. This file
must not be edited after the implementation commit without a new protocol
revision.

## 1. Why v2 exists (the v1 critique)

DepthFrag-v1 distills a *geometry* target computed on the **student/online**
backbone: the terminal true-class margin `m`, its representation gradient
`g_l = dm/dh_l`, and the fragility radius `rho_l = positive_part(m) /
(||g_l||_q + eps)` regressed by per-site probes and a terminal head.

V1 has structural weaknesses that the SAGE-v2-confirmation era surfaced:

1. **No teacher.** Targets are computed from the same parameters being
   trained, so the pairing signal is *self-predictive*: the student can chase
   its own current decision geometry, and the target distribution drifts as
   quickly as the model updates. Nothing stabilizes the geometry being
   distilled.
2. **Unnormalized geometry.** `rho_l = m / (||g_l||_q + eps)` divides by the
   raw gradient norm without conditioning on the *feature scale*: a site whose
   activations are large (deep sites) produces tiny `rho` spuriously, and the
   same damaged site in a new architecture scales differently. The
   denominator mixes representation scale into the fragility estimate and the
   transform `log1p` cannot repair a mis-scaled ratio.
3. **Tiny/zero-gradient blowup.** With `m > 0` and `||g_l|| ~ 0`, `rho_l`
   explodes to infinity; with `m = 0` and `||g_l|| = 0` it is `0/0`. V1 only
   padded the denominator; it did not clip the numerator or the ratio.
4. **Warmup is config-gated but not principled.** `depthfrag_warm25` detaches
   features during warm-up, but v1's default had no warmup and its warmup
   variant did not freeze the *teacher* geometry (there is none).

DepthFrag-V2 fixes (1)-(4): a true EMA teacher yields stable targets;
diagonal feature-whitening normalizes geometry per site; bounded stable
clipped transforms prevent blowups; a fixed 25-epoch warmup decoupled from any
teacher reset stabilizes early training.

## 2. Research thesis

DepthFrag-v2 distills stable, representation-conditioned decision geometry
from an EMA teacher into a label-free final confidence score.

## 3. EMA teacher

Parameter copy

    theta_T <- nu * theta_T + (1 - nu) * theta_student

maintained from the **first optimizer step** (not from initialization), with

    nu = 0.999   (locked)

The teacher:

- receives **no gradient** (all teacher params are pure buffers — never
  registered as trainable, wrapped in `torch.no_grad()` during updates);
- supplies geometry targets only;
- is saved/restored exactly in checkpoints (teacher params part of
  `state_dict`; resume restores `theta_T` bit-exact);
- is built by `AveragedModel`-style deep copy of the student backbone at
  construction, with the same initialization seed so that student and teacher
  begin identical (teacher update starts at step 1, not 0, so step-0 teacher
  == student's initial state).

Teacher warmup: teacher parameters begin as a copy of the student *at
construction*; because the student is freshly initialized with the run seed,
teacher and student agree at step 0. During epochs 0–24 the teacher continues
to EMA-update (it is always in lockstep with the student's early training);
only the *student loss* changes routing at the epoch-25 boundary.

## 4. Whitened fragility

At each site `l`, maintain an EMA diagonal covariance `Sigma_l` computed from
**teacher features**:

    Sigma_l <- (1 - nu_v) Sigma_l + nu_v * h_T,l^2      (elementwise square)

with `nu_v = 0.01` (locked) and entrywise mean over the batch at each step
(no gradient flows into `Sigma_l`; the EMA buffers are `persistent=True`).

Using the **true-class teacher margin only during training**:

    m_T(x, y) = z_T,y - max_{j != y} z_T,j

define

    g_l = gradient of m_T w.r.t. teacher representation h_T,l
          (first-order, no create_graph)

    rho_l = positive_part(m_T) / (sqrt(g_l^T Sigma_l g_l) + eps)

where `positive_part(x) = relu(x)` keeps only margin-loss-relevant examples
(mirrors v1 for AMT compatibility but now on teacher geometry), and the
denominator is the **Mahalanobis-whitened gradient norm**: dividing by feature
covariance means `g_l` is measured in units of feature variation rather than
in raw activation scale — sites with large, high-variance activations no
longer produce spuriously tiny `rho` (critique 2).

### 4.1 Finite stable targets

Use a transformed target of `rho_l` with hard clipping. Clipping statistics
ARE LEARNED FROM TRAINING DATA ONLY, never validation or test:

- Maintain a running training-set estimate of `rho_l` quantiles per site
  (`rho_p1`, `rho_p99` EMA buffers over training batches).
- Clamp: `rho_clip = clamp(rho_l, rho_p1, rho_p99)`.
- Transform: `target_l = sign(rho_clip) * log1p(|rho_clip|)` (v1's
  `signed_log1p`, now applied to the clipped pre-transform value, not to an
  unclipped ratio).
- If `g_l == 0` or `Sigma_l` is all-zero, the ratio `0/0` is guarded by
  `eps`; additionally the clip to `[rho_p1, rho_p99]` bounds the value —
  properly, if `rho_p1 == rho_p99 == 0` (degenerate site), `target = 0`, and
  the probe sees a constant zero target; this is finite and logged as
  `degenerate_site_l`. The **target is never infinite** even for tiny
  gradients: `positive_part(m_T) <= max margin` (bounded by `logits`), the
  denominator is `>= eps`, and the clip caps the transform.

### 4.2 Diagonal whitening across architectures

The per-site `Sigma_l` (diagonal) makes the whitening architecture-neutral:
it only depends on `h_T,l` shape (`pool_tap` output: `GAP` for 4-D CNN feature
maps, `CLS`/mean pooling for ViT), giving the same geometry formula for every
backbone. Tests reproduce the invariance under channel rescaling:
`h -> alpha * h` with `Sigma -> alpha^2 Sigma` leaves `rho` invariant
(denominator scales under `sqrt(alpha^2)` exactly compensating numerator; assert
same `rho` before/after with numerical tolerance).

## 5. Warmup

- Epochs 0–24 **inclusive**: the backbone receives CE gradients only.
  Probes/head may train using **detached teacher features** and teacher
  targets (the per-site probe inputs are `detach()` from the backbone path;
  targets come from the frozen/EMA teacher; early steps cannot send the aux
  head gradient into the backbone).
- Epoch 25 onward: enable end-to-end DepthFrag-v2 supervision (probe/head
  gradients flow into the backbone as in v1's nontrivial path).
- **No optimizer, scheduler, model or EMA reset at the boundary.**
- Same warmup for both datasets (locked: `warmup_epochs: 25`).

## 6. Oracle/deployment separation

- True-label radii (`m_T`, `rho_l`) are **privileged training targets** only.
- True-label radii on val/test are labeled **ORACLE_DIAGNOSTIC** only: they may
  be computed and logged into a separate oracle-diagnostic channel, never into
  the primary metrics/registry score, and never fed to `predict`/`evaluate`
  for the primary score.
- Primary test confidence is the **label-free final student prediction**.
- `predict_batch` / evaluator path **must not access test labels** (the
  evaluation path never passes `y`; the existing `stripped_predict_batch`
  discipline is retained).

## 7. Primary loss

    L = (1/|S|) sum_l Huber( q_l(pool(h_S,l)), target_l )   (probes, detached at warmup)
      + Huber( head(final_embedding), agg_target )          (terminal head)

with `delta = 1.0` (Huber delta locked), on the **signed-log1p transformed
geometry target** of section 4.1. `agg` = `soft_min` with `tau = 2.0` (v1
default locked), `use_probes: true`, `use_head: true` (config keys from v1).

No pairwise ranking loss in the primary method. A deterministic small
pairwise ranking term is **not adopted** — deferred to a future iteration so
that the primary distance measurements stay decoupled from a learnable
ordering; this keeps the method a geometry distiller, not a ConfidNet-style
correctness predictor. (This decision is locked in the protocol, before any
result.)

## 8. Mandatory decomposition

Evaluate and log separately (all on the **teacher** geometry, per site):

- `terminal_margin` — `m_T` at terminal site;
- `grad_denom` — `sqrt(g_l^T Sigma_l g_l) + eps` per site;
- `margin_grad_ratio` — `rho_l` per site;
- per-site individual score `q_l(pool(h_S,l))` and per-site target `target_l`
  (fixing the known v1 per-site logging bug: every site logs its own prediction
  and its own target);
- complete depth profile — full vector `{site: rho_l}`;
- distilled final head score — `head(final_embedding)`.

This determines whether geometry contributes information beyond final margin.

## 9. Fixed diagnostic (preregistered, on a subset)

On a preregistered subset (**50 validation examples per class**, CIFAR-10 and
CIFAR-100, i.e. the 5k val fold split further by class; total 50k examples
CIFAR-100 — the val fold is 5,000; use up to the full val fold; the count is
"as many as the locked val fold provides per class"), compare **analytic rho**
(`rho_l = pos(m_T)/sqrt(g^T Sigma g)`) against an **iterative local
boundary-distance approximation** using Spearman correlation. The iterative
approximation walks the teacher output logits in the normalized gradient
direction in small steps until the predicted class flips, and records the
displacement bound; it is computed on `feats.requires_grad_(True)` input with
`torch.no_grad()` on the teacher and a manually stepped logit perturbation.
Evaluation runs only after training, once; logged to
`<run>/depthfrag_v2_oracle_diag.json`. If a site is degenerate (target
identically zero), record `nan` for that site's correlation and keep the
per-site flag.

## 10. Implementation corrections (carried from v1)

- Per-site logging bug fixed: every site logs its own prediction and target.
- Tiny gradients cannot produce infinite targets (clipping, bounded numerator,
  epsilon denominator, degenerate-site guard in 4.1).
- Representation dependence documented: geometry is defined on teacher
  features, so it depends on the teacher representation and the site pool.
- Diagonal whitening consistent across architectures (4.2).
- Inference stays label-free (6).

## 11. Tests (`tests/test_depthfrag_v2.py`)

1. EMA update formula: after one optimizer step `theta_T = nu*theta_T_0 +
   (1-nu)*theta_S_1`; exact resume restores `theta_T` bit-exact.
2. Diagonal whitening invariance under channel rescaling (4.2).
3. Finite targets for tiny/zero gradients (construct `g_l -> 0`, `Sigma -> 0`
   and assert target is finite, equals a bounded clipped value, and the
   degenerate-site flag is raised).
4. Epoch-24 vs epoch-25 gradient routing: at epoch 24, `autograd.grad(loss,
   backbone.params)` is zero for probe/head loss; at epoch 25 it is nonzero.
5. No test-label access: `predict_batch` never requires `y`; oracle results
   segregated from primary metrics (assert oracle channel absent from
   registry `metrics` fields).
6. Correct per-site logging (each site logs own prediction + target).
7. Margin/denominator/ratio decomposition values match hand-computed
   `m_T`, `sqrt(g^T Sigma g)`, `rho`.
8. Architecture-neutral site handling (parametrized backbones, mirroring v1).
9. Existing DepthFrag tests remain green.

## 12. Main runs

After the current SAGE-v2 queue is idle:

    - depthfrag_v2
    - CIFAR-10 and CIFAR-100
    - seeds 13, 17, 23
    - six total runs

Recipe `ccl_sc_reference` (300 epochs, batch 64, SGD lr 0.1/0.9/wd 5e-4, step
0.5/25), existing val/test split, selection rule
`min_val_aurc_among_acc>=best_acc-1.0pp`, unchanged metrics. Compare against
`depthfrag` and `depthfrag_warm25` using matched seeds. Negative results are
preserved.

## 13. Locked constants (single values for both datasets)

| constant | value |
|---|---|
| nu (EMA teacher) | 0.999 |
| nu_v (feature-cov EMA) | 0.01 |
| warmup_epochs | 25 |
| eps (denominator) | 1e-12 |
| huber_delta | 1.0 |
| agg | soft_min, tau 2.0 |
| clip source | training-only EMA quantiles (p1, p99) |
| token | cls |

No dataset-specific hyperparameters anywhere in the primary method.