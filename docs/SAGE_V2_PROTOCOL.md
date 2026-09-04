# SAGE-V2 protocol: bilevel-utility gated selective supervision

Status: **preregistered protocol**. Implemented as the `sage_ds_v2` method alias;
`sage_ds` (v1) is preserved unchanged. SAGE-V2 must never be launched directly
from `f76c4a2` — the protocol commit and the implementation commit are the
minimum runnable state.

## 1. Why v2 exists (the v1 critique)

SAGE-DS v1 estimates the per-site selective utility as the dot product

    U_l = <g_sel, g_l>

where `g_sel` is the gradient of the differentiable AURC surrogate and `g_l` is
the site's auxiliary-supervision gradient **of the same validation batch**. The
practical step-outcome it predicts is "pulling the backbone along `g_l` improves
held-out selective competence", but both gradients being on the *same* batch
mixes two separate questions:

1. **Direction of the supervision pull** is a *training-side* quantity: it is
   the gradient of the site's auxiliary cross-entropy at the training
   distribution.
2. **Selective competence** of that direction is a *held-out* quantity: the
   selective objective must be estimated out-of-sample.

v1 also (a) projected the *combined* gated-auxiliary gradient against a running
EMA of past CE gradients (`g0_ema`) rather than per-site against the current
step's CE gradient, so the safety invariant is only approximate and per-step
alignment is not auditable; and (b) scaled every projected auxiliary
contribution by the *sampled* hard-concrete gate, so the "utility of the site"
was entangled with the current gate strength.

SAGE-V2 fixes the four structural issues:

- **True bilevel structure.** The per-site supervision direction is the train
  gradient `g_l_train = nabla L_l_aux(B_train)`; the selective objective is
  graded by `g_J_meta = nabla AURC~(B_meta)` from a **disjoint held-out meta
  batch**. No batch is ever reused across the two sides.
- **Per-site, same-batch CE-safety.** Each site's auxiliary gradient is
  projected against **this training batch's** CE gradient `g0_train = nabla
  L_CE(B_train)`, giving an exact per-step, per-site safety inequality that
  tests can lock.
- **Unweighted utility.** The utility measures the site's *unweighted*
  supervision direction (`g_l_train`, projected); gate weighting is applied only
  at the application step, so a weak gate does not corrupt the utility estimate.
- **Cosine-controlled gates.** The controller consumes the cosine utility
  `U_cos` (scale-free, in `[-1, 1]`); the signed magnitude `U_raw` is retained
  in the per-estimate telemetry (required logging) but never feeds the EMA.

Plain terminal MSP remains the only primary inference score; the auxiliary
heads and the controller are training-only instruments absent from deployment
(unchanged from v1).

## 2. Algorithm (locked)

Notation: `theta` = backbone parameters (the sites' supervision gradient set;
`_utility_params`, default: every backbone parameter). `s` = global supervision
scale (`method.supervision_scale`, one fixed value across sites).
`z_l ~ HardConcrete(log_alpha_l)` sampled each step. `eps = 1e-8`
(`method.projection_eps`).

At every training step, on the training batch `B_train`:

    g0_train    = nabla_theta L_CE(B_train)
    g_l_train   = nabla_theta L_l_aux(B_train)                 (unweighted)
    tilde_g_l   = g_l_train - (min(0, <g_l_train, g0_train>)
                               / (||g0_train||^2 + eps)) g0_train
    g_applied   = g0_train + sum_l (z_l * s) * tilde_g_l       (backbone params)
    aux heads   = own unweighted CE gradients (nabla L_l_aux)

`g_applied` is routed directly onto the backbone parameters (the v1 "routed"
mechanism: `sum_p p * g_desired.detach()`, so the backward of the routed scalar
reproduces `g_applied` exactly — locked by a test).

Every `method.utility_interval` steps (step `> 0`, `step % interval == 0`), on
a **fresh disjoint meta batch** `B_meta` drawn from the held-out validation
split, in eval mode (fixed BN statistics):

    g_J_meta    = nabla_theta AURC~(B_meta)                     (soft surrogate,
                 = nabla_theta soft_aurc_surrogate(logits(B_meta), y_meta))
    U_unproj_l  = <g_J_meta, g_l_train>                          (retained, logged)
    U_raw_l     = <g_J_meta, tilde_g_l>                          (retained, logged)
    U_cos_l     = U_raw_l / (||g_J_meta|| * ||tilde_g_l|| + eps) (controller input)

`B_train` and `B_meta` are guaranteed disjoint: the harness 45 000/5 000
stratified split never overlaps, the method requests sample indices
(`needs_indices = True`, new for v2), and the estimate **raises** if the two
batch index sets intersect. Gate control is the v1 controller consuming
`U_cos`:

    utility_ema <- beta * utility_ema + (1 - beta) * U_cos   (bias-corrected read)
    log_alpha_l  += lr * clamp(u_norm_l - sparsity_cost, -cap, +cap)

`u_norm_l` normalises the per-site EMA `U_cos` values to `[-1, 1]` (cosine
values already lie there). `U_raw` is never written back into the EMA; it is
recorded per estimate.

## 3. Deviations from v1 (deliberate, preregistered)

| Aspect | v1 | v2 |
|---|---|---|
| Utility side pairing | `g_sel` and `g_l` on one val batch | `g_l` on train batch, `g_J_meta` on disjoint meta (val) batch |
| CE safety | combined aux gradient vs `g0_ema` (EMA) | per site vs same-batch `g0_train` (exact per-step inequality) |
| Utility gradient | weighted by nothing (val side) | unweighted train-side `g_l`, projected |
| Controller input | EMA-normalised raw `U_l` | cosine `U_cos` (`U_raw` retained and logged) |
| Batch indices | not needed | `needs_indices = True` + runtime disjointness guard |
| Log files | `sage_ds.jsonl` (per epoch) | `sage_ds_v2.jsonl` (per epoch) + `sage_ds_v2_utility.jsonl` (per estimate) |

Everything else (sparse hard-concrete gates, backbone taps/roles registry,
MSP-only inference, training-only aux heads, `ccl_sc_reference` recipe, val-only
checkpoint selection) is unchanged from v1.

## 4. Required per-estimate logging

One append-only row per estimate in `<run_dir>/sage_ds_v2_utility.jsonl`:

Shared: `step`, `epoch`, `train_ids`, `meta_ids`, `meta_bs`, `gJ_norm`,
`g0_norm`.

Per site `l` (`..._{l}` suffixes): `raw_unprojected_utility`, `raw_utility`
(= `U_raw_l`), `cos_utility` (= `U_cos_l`), `gl_norm`, `tilde_gl_norm`,
`support_frac` (fraction of nonzero entries of `tilde_g_l` over the utility
parameter set), `align_before` (`<g_l_train, g0_train>`), `align_after`
(`<tilde_g_l, g0_train>`), `gatep` (expected gate probability), `sampled_gate`
(the step's realised `z_l`), `eff_aux_w` (`z_l * s`), `uema` (post-update EMA of
`U_cos`).

Per-epoch aggregates continue to land in `sage_ds_v2.jsonl` (gate probabilities,
utility EMA, aux accuracy/loss, L0, sparsity penalty, mean alignments) in the
v1 row shape so existing plotting/heatmap tooling works unchanged.

`train_ids`/`meta_ids` are the actual global official-fold indices used on each
side; the row is therefore self-certifying for the disjointness guarantee.

## 5. Required tests (`tests/test_sage_ds_v2.py`)

1. **Finite-difference utility sign (bilevel).** Pulling `theta` along
   `-eta * tilde_g_l` moves the meta AURC surrogate by `-eta * U_raw_l`
   (sign lock + approximate magnitude).
2. **Cosine invariance to positive rescaling.** `U_cos` is unchanged when a
   site gradient (equivalently `tilde_g_l`) is scaled by any `c > 0`; `U_raw`
   scales by `c`; `project_aux(c * g_l, g0) = c * project_aux(g_l, g0)`.
3. **Per-site CE-safety inequality.** For every site `l`:
   `<tilde_g_l, g0_train> >= -tol`, where the tolerance is the exact
   epsilon-blocking residue `align_after = align_before * eps /
   (||g0_train||^2 + eps)` (locked exactly from the logged `g0_norm2`;
   numerically `<tilde_g_l, g0_train> >= -1e-2` absolute).
4. **Applied gradient identity.** The gradient of the routed scalar equals
   `g0_train + sum_l (z_l * s) * tilde_g_l` on backbone parameters and the raw
   CE gradients on auxiliary heads (audit dict compared against
   `torch.autograd.grad` of `routed`).
5. **Train/meta batch discipline.** Train indices ⊆ train split and meta
   indices ⊆ val split, the two sets disjoint; an overlapping meta batch raises
   `RuntimeError` inside the estimate.
6. **Zero-gradient finiteness.** Zero `g_l`, `g0`, or `g_J` yield finite
   utilities (0 or epsilon-guarded), finite EMA, and finite controller steps.
7. **Exact checkpoint/resume of controller state.** Gates (`log_alpha`),
   `utility_ema`, and `ui_step` round-trip through `state_dict`.
8. **Inference is plain MSP.** `predict_batch` confidence == `scores["msp"]`;
   aux heads and the controller are never deployment modules; mutating them
   leaves predictions bit-identical.
9. **No deployment overhead.** `inference_modules()` parameter count equals the
   backbone-only count (aux heads/controller excluded).

Plus the v1-pattern smoke tests (one engine-style step routes gradients onto
backbone + aux heads while controller gates stay manual; utility-interval=1
exercises the estimate branch; config loads across all registered backbones).

## 6. Experimental plan and labelling

Initial run(s): CIFAR-10 and CIFAR-100, VGG16-BN, the identical
`ccl_sc_reference` recipe, seed 13, `sage_ds_v2`. The comparison target is the
SAGE-DS v1 seed-13 run already in the reference registry (CIFAR-10 AURC
0.007045 mean-5-seed; direct v1 s13 compare at run level).

**Exploratory label.** The meta batch is drawn from the 5 000-example
validation split, and checkpoint selection also uses that split
(contract §3). This is a documented limitation, not a fix: exploratory SAGE-V2
runs are labelled `exploratory` in the manifest/registry note until either (a) a
matched selection scheme that reads no meta information, or (b) an
out-of-meta validation protocol is introduced. A superiority claim against v1
requires the matched `reference_metric_anchors` CE/CCL-SC manifest per the
empirical contract §8 and a separate decision run.

## 7. Execution constraints

- Run only from a checkout pinned to this protocol + implementation commit
  (mirror or clone), with `SCSF_SOURCE_COMMIT` exported so the manifest carries
  the real commit (f76c4a2 added the reproducibility fix; the blank-commit
  failure mode is a scheduling-environment mistake, not this method's).
- CPU test run in a *separate* mirror before any GPU launch; full `pytest -q`
  must be 0 failures.
- `sage_ds` v1 artifacts, configs, tests, and logs are untouched.