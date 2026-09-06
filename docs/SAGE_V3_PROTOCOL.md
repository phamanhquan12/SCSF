# SAGE-V3 protocol: robust training-aware gradient allocation

Status: **preregistered protocol**. Implemented as the `sage_ds_v3` method alias;
`sage_ds` (v1) and `sage_ds_v2` are preserved unchanged. SAGE-V3 must never be
launched and this file must not be edited after the implementation commit
without a new protocol revision. A secondary efficiency variant,
`sage_ds_v3_amortized`, is registered as a separate alias and is not the
primary scientific method.

## 1. Why v3 exists (the v1/v2 critique)

SAGE-v1 and SAGE-v2 both measure *coverage-agnostic* selective utility: their
differentiable surrogate is a scalar mean of soft selective risk over the
coverage grid, and per-site supervision is allocated to improve that **global**
objective.

This optimizes average risk over all accepted-confidence thresholds but does
not protect **worst classes**: a class that is systematically hard to calibrate
contributes little to the mean, so allocation can silently ignore it. The
SAGE-v2 multi-seed confirmation documented exactly this failure mode on
CIFAR-100 (class-35 regression, Audit B; reproduces in 4/5 seeds). v3 therefore
targets a **class-conditioned robust** selective objective instead of the
global mean.

A second critique is that v1/v2 allocate supervision via a learned controller
acting on one utility number per site (EMA of cosine similarity). That is a
learned, partially-interpretable rule, not an exact constrained allocation.
v3 replaces the learned controller with a **certified convex QP** whose
solution provably maximizes robustness gain under feasibility constraints,
removing a free learned component and giving exact, verifiable behavior.

## 2. Research thesis

SAGE-v3 approximates robust selective descent using only update directions
realizable through intermediate deep supervision. The target protects
class-conditioned selective performance instead of optimizing only global AURC.

## 3. Robust selective target

For each class `c`, define the same differentiable selective-risk surrogate on
a class-balanced meta-batch:

    J_c(theta) = soft_aurc_surrogate(logits^c, targets^c)

where `soft_aurc_surrogate` is the existing differentiable surrogate
(`scsf/metrics/surrogate.py`) with the locked defaults `error_mode="proxy"` and
the coverage grid `COVERAGE_GRID_PERCENT`.

Aggregate with a log-sum-exp robust mean:

    J_rob = (logsumexp(tau * [J_1, ..., J_C]) - log(C)) / tau

with **single fixed tau = 10 for both datasets** (locked in this protocol; the
implementation asserts this value). No per-dataset tau is permitted.

Small-tau regimes approach mean-class risk; large-tau regimes approach
worst-class risk; tau=10 is a compromise that up-weights the worst classes
without collapsing onto a single class. Exactly `-log(C)/tau` is subtracted so
`J_rob` is a generalized mean that ranges between the mean (`tau -> 0`) and the
max (`tau -> inf`) class surrogate value.

### 3.1 Class-balanced meta-batch

`J_c` must be estimated on a **deterministic class-balanced meta-batch** so that
every class is represented. For CIFAR-100, an ordinary random batch omits most
classes; sampling uniformly at random would make `J_c` undefined or zero-mean
for the majority of classes. We therefore:

- build per-class index masks of the current validation fold (memorized once at
  construction time, using the training-fold indices as in v2);
- sample exactly `k_meta` examples per class per utility estimation, with
  `k_meta` fixed across datasets (`k_meta = 8`); if a class has fewer than
  `k_meta` examples in the val fold it contributes all of its examples (this
  cannot happen: CIFAR-10/100 val folds have >=50 and >=50 balanced? — the
  SCSF split is 45k/5k single folds per dataset and CIFAR-100 has 50 examples
  per class in each class-balanced 50k split, so exactly 50 per class in val;
  assert >= 4 per class in the implementation);
- require the meta batch to be **disjoint from the current training batch**
  (reuse the v2 disjointness machinery): `needs_indices = True`, and raise a
  `RuntimeError` on overlap, exactly as SAGE-v2 does;
- if the meta batch has < `C` classes represented (should not happen), raise.

This is a deviation from v2, which used a leftover single meta batch from the
val loader without per-class balancing. The per-class structure is the
scientific point of v3, so class balancing is fixed and asserted.

### 3.2 Single backward pass

`g_r = grad_theta J_rob` must be obtained with a **single backward pass** from
`J_rob` to the backbone parameters (`create_graph=False`), reusing v2's
`_utility_params`. Test asserts one backward.

## 4. Deep-supervision directions

For each candidate site `l` (resolved from `backbone.taps`, architecture
neutral):

    g_l  = gradient of auxiliary CE on the training batch
    g_0  = main CE gradient on the same training batch

**Project each `g_l` to be classification-compatible** exactly as in v2
(per-site, same-batch projection against `g_0`):

    dot_l      = <g_l, g_0>
    g_l^proj   = g_l - min(0, dot_l) / (||g_0||^2 + eps) * g_0
    a_l        = g_l^proj / (||g_l^proj|| + eps)

The projection makes every `a_l` non-hostile to classification (non-negative
alignment with the CE direction). Normalization gives unit `l2` norm regardless
of site scale. `eps = 1e-8` (config `projection_eps`, locked).

Let

    A = [a_1, ..., a_L]            (columns, each in parameter space)
    r = g_r / (||g_r|| + eps)      (normalized robust gradient)
    c = g_0 / (||g_0|| + eps)      (normalized CE gradient)

Sufficient statistics:

    G = A^T A     (L x L Gram, cheap: number of sites is small)
    b = A^T r     (per-site alignment with robust gradient)
    q = A^T c     (per-site CE compatibility after projection)

All computed in the flattened parameter space of `_utility_params` (same
flattening as v2's `_cat`/`_flatten`).

## 5. Exact allocation

Solve the convex QP over per-site weights `lambda`:

    minimize_lambda   0.5 * lambda^T (G + ridge * I) lambda - b^T lambda
    subject to        lambda >= 0
                      sum(lambda) <= B
                      q^T lambda >= 0

with **fixed B = 1** and **fixed ridge = 1e-4** for both datasets (locked in this
protocol).

- `lambda >= 0` — no site is anti-trained.
- `sum(lambda) <= B` — total auxiliary-gradient budget is bounded by the
  primary CE gradient scale (B=1 means at most one unit of normalized aux mass);
  together with unit-normalized sites this bounds the applied gradient.
- `q^T lambda >= 0` — the *post-projection weighted* direction still has
  non-negative overall alignment with CE (CE-safety at the aggregate level;
  note each `a_l` is already individually non-negative aligned by projection,
  so `q >= 0` componentwise and this constraint is technically implied but is
  kept explicit and rechecked in the certificate).

### 5.1 Exact solver

Because the number of sites is small (`L <= 5` for VGG16-BN taps), implement a
**deterministic exact active-set solver** that:

- solves the unconstrained QP on each candidate active/non-free set by direct
  linear solve of `(G_sub + ridge I) lambda_sub = b_sub` (guarded by `chol` /
  `lu_solve`), 
- enumerates/derives the active set from the KKT sign structure so the
  returned point is verified against KKT conditions,
- returns `lambda` satisfying all constraints to `1e-9` tolerance.

Do **not** call ordinary projected-gradient iterations an exact solver. The
solver's output must be checked against the KKT conditions in the certificate
and in a test that compares against a trusted brute-force/faithful reference
(`scipy.optimize` is not a dependency; use the test-only reference below).

### 5.2 Certificate

After solving and before applying:

    b^T lambda  >  0
    q^T lambda  >= -tolerance
    lambda      >= -tolerance
    sum(lambda) <= B + tolerance

with `tolerance = 1e-6`. If the certificate **fails**, set `lambda = 0`
(zero auxiliary-gradient mass applied this step). The zero-gradient fallback
must be logged and tested.

## 6. Backbone update

Apply, per backbone parameter `p`:

    g_update = g_CE + rho * ||g_CE|| * A lambda

i.e. the applied auxiliary gradient direction is `sum_l lambda_l * a_l`,
scaled by `rho * ||g_CE||`, added to the CE gradient. **Fixed rho = 1 for both
datasets** (locked in this protocol).

Implementation reuses v2's routing trick: materialize `g_desired = g_CE +
rho*||g_CE|| * sum_l (lambda_l * a_l)` and route it through
`routed = sum_p <p, g_desired.detach()>` — a dot-product loss — so
`total.backward()` produces the applied gradient exactly (matches v2 and the
existing applied-gradient identity test).

**Aux heads** keep their own unweighted CE gradient exactly as v2 (their
parameters are never part of `A lambda`; only backbone params enter `A`).

## 7. Inference

- MSP is the **primary** inference confidence (score `msp`, as v1/v2).
- Auxiliary heads and allocation machinery are removed from the deployment
  graph (`inference_modules` returns `[backbone]` only).
- **Zero additional inference overhead** over a plain backbone classifier.

## 8. Required logging (per utility estimation)

- Per-class `J_c` (all `C` values).
- Robust weights `w_c = softmax(tau * J)` (the class-conditioned robust
  weights) — diagnostic only, never used to constrain the QP.
- `G`, `b`, `q` (per-site).
- `lambda` (per-site, post-solve).
- QP objective value `0.5 l^T(G+ridge I)l - b^T l` and duality/primal gap
  (computed against the KKT dual bound, logged as `qp_gap`).
- Every feasibility/certificate value (`cert_b_lambda`, `cert_q_lambda`,
  `cert_nonneg`, `cert_sum`, `cert_ok`).
- Global and per-class selective gradient norms (`gJ_norm`, `per_class_grad_norm_<c>`).
- Per-site auxiliary-gradient norms (`gl_norm_<l>`).
- Total applied auxiliary-gradient mass `sum(lambda)` and
  `rho * ||g_CE|| * sum(lambda)`.
- CE compatibility before and after allocation (`align_before_l`, `align_after_l`).
- Frequency of `lambda = 0` fallback (`lambda_zero` count + running fraction).
- Topology over epoch (`gatep_<l>` / `topology_<l>` — same semantics as v2 for
  the gate strengths, but v3 does not learn gates; log per-site `lambda` norm
  profile as the traced topology).
- Generic class-conditioned diagnostics (per-class risk summary); **never
  special-case class 35**.

Telemetry file: `<run>/sage_ds_v3_utility.jsonl` (JSON-per-line, mirroring v2).

## 9. Strict per-class constraints (diagnostic only)

Do **not** impose per-class gradient constraints in the primary method. As a
pure diagnostic, record how often a direction satisfying every per-class
constraint `(A^T grad_theta J_c, lambda)_l >= -tol` exists (solved feasibility
check), logged as `per_class_feasible` and `per_class_feasible_frac`. This is
informational for future amortization; it never gates application.

## 10. Amortized solver (secondary variant only)

`sage_ds_v3_amortized` is a separate alias registered as an efficiency
extension. It is **not** the primary scientific method and the three-seed main
matrix runs `sage_ds_v3` only, never the amortized variant (unless v3 first
validates the principle).

- Inputs: `G`, `b`, `q` only (the same sufficient statistics).
- 3–5 unrolled projected optimization steps; learning the **step sizes**
  (scalar per iteration) and optionally a momentum coefficient. Exact QP
  solution used as regression target / train loss = objective gap.
- Final-project and certify every predicted allocation (same certificate as
  primary).
- Report objective gap and allocation error relative to the exact solver.
- The learned step-size module is a tiny extra set of `<= 10` trainable scalars
  and is a training-time-only artifact (not part of inference).

## 11. Tests (targeted suite, `tests/test_sage_ds_v3.py`)

1. Small-tau approaches mean-class risk; large-tau approaches worst-class risk
   (compute on a synthetic 2-class population with known per-class surrogates;
   assert ordering and monotone convergence).
2. Class-balanced meta-batches cover every class (assert each of `C` classes
   present in built meta batch; on both CIFAR configs).
3. Meta batches are disjoint from training batches, enforced (`RuntimeError`
   on overlap; mirror the v2 disjointness test).
4. Robust gradient uses a single backward pass (patch `autograd.grad` in
   `scsf.metrics.surrogate` scope or count via `torch.autograd.grad` calls /
   `torch.profiler`-free counter in the method under test).
5. QP solution agrees with a trusted brute-force reference over a
   representative grid of small `(G, b, q)` cases (enumeration over flattened
   active sets with dense linear solve is itself exact; use a reference active
   set that the test verifies by independent KKT residual).
6. KKT, feasibility and certificate tests (primal feasibility, stationarity
   residual, constraint satisfaction; certificate `True` on feasible
   instances and `False`/fallback on constructed infeasible instances).
7. CE-safety and zero-gradient fallback (`lambda=0` when certificate fails or
   when `||g_r||` is ~0).
8. Resume exactly preserves all states (mirror v2 resume test).
9. Architecture-neutral site discovery (mirror v2 parametrized backbones).
10. MSP-only inference, zero extra inference modules.
11. No regression in SAGE-v1/v2 tests (`test_sage_ds.py`, `test_sage_ds_v2.py`
    stay green).

## 12. Main runs

After the current SAGE-v2 queue is idle:

    - sage_ds_v3
    - CIFAR-10 and CIFAR-100
    - seeds 13, 17, 23
    - six total main runs

Recipe `ccl_sc_reference` (300 epochs, batch 64, SGD lr 0.1, wd 5e-4, step
0.5/25), existing val/test split, selection rule
`min_val_aurc_among_acc>=best_acc-1.0pp`, unchanged metric implementation.
Compare against SAGE-v1 and SAGE-v2 using matched seeds. Negative results are
preserved.

## 13. Locked constants (single values for both datasets)

| constant | value |
|---|---|
| tau (robust aggregation) | 10 |
| B (lambda budget) | 1 |
| rho (application scale) | 1 |
| ridge (QP regularization) | 1e-4 |
| projection_eps | 1e-8 |
| k_meta (examples per class) | 8 |
| certificate tolerance | 1e-6 |

No dataset-specific hyperparameters are permitted anywhere in the primary
method.