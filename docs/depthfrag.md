# DepthFrag: depth-wise decision fragility

DepthFrag turns the *distance to the decision boundary at every network
depth* into a per-example fragility score and distills it into a small
terminal head. Two artifacts exist:

1. **Frozen-checkpoint extractor** — `python -m scsf.extract_depthfg
   run_dir=... split=val` loads an ordinary checkpoint and computes
   sample-level signed radius profiles plus a validation-fitted score ladder
   evaluated on the untouched test split.
2. **Distillation method** — `method_name=depthfrag` trains per-site probes
   that regress the **detached** fragility targets, with a terminal head that
   learns a depth-aggregated fragility score. Inference keeps only the
   terminal logits + the small head (probes are training instruments).

## Geometry (locked by tests)

For an example with true class `y` and final logits `z`, the terminal
true-class margin is

    m = z_y - max_{c != y} z_c .

At a tapped site `l` with representation `h_l` (a registry tap), the
first-order fragility geometry is

    g_l             = d m / d h_l          (vector-Jacobian product, first order)
    rho_l           = m / (||g_l||_q + eps)
    relative_rho_l  = rho_l / (||h_l||_p + eps)
    target_l        = sign(relative_rho_l) * log1p(|relative_rho_l|)

Defaults are `p = q = 2` (Euclidean); `p = inf, q = 1` are supported for
every primitive and audited by tests.

* **Signed radii.** Incorrect examples (negative margin) keep a negative
  radius, so a correct/incorrect distinction is preserved instead of being
  clipped away. `target_l` preserves the sign.
* **Scale invariance.** For positive re-scaling of logits and gradients the
  radius is invariant (exact with `eps = 0`, and the division is homogeneous
  in the model's output layer) — tested.
* **Finite everywhere.** The `eps` floors keep the formula finite when
  `||g_l|| = 0` or `||h_l|| = 0`; a site that is not on the margin's path
  (`allow_unused`) yields a zero radius — tested.
* **First order only.** `site_gradients` uses `autograd.grad` with
  `create_graph=False` and `retain_graph` only across the site columns within
  one batch; each returned tensor is detached and the graph is freed per
  batch. A dedicated test proves no second-order graph is retained and the
  live-tensor count stays bounded across repeated batches.

The exact linear case (a linear classifier with `h = x`) is tested against
the closed form `g = W[:, y] - W[:, c*]`.

## BatchNorm treatment

`autograd.grad(m.sum(), h_l)` is the *batch* VJP: it equals the stack of
per-example gradients exactly when the forward graph is block-diagonal across
examples. Training-mode BatchNorm couples the examples (batch statistics) and
invalidates that reading. Two analytic modes are exposed and tested:

* `mode=fast` — one batched forward with BatchNorm pinned to **eval
  statistics** (the parameters stay differentiable) plus a single batch VJP.
  Valid for any network whose only cross-example operator is BatchNorm.
* `mode=exact` — per-example Jacobians via `torch.func.functional_call` on a
  configurable microbatch; no example is ever coupled to another.

The BatchNorm coupling is **demonstrated, not hidden**: a dedicated test
compares `fast` with the BatchNorm in *training* role against the exact
per-example reading and asserts the divergence is well above float noise,
while the eval-role readings agree. Always use the eval-role target forward
in training and extraction.

## Limits of the linear approximation

The radius is a *local* linearized distance. Two documented caveats (both
tested):

* The iterative DeepFool-style audit (`scsf.depthfrag.iterative`) walks
  toward the **current nearest competitor class** only — a documented
  equivalent of DeepFool, not the full min-over-classes step, and not an
  adversarial-distance or robustness guarantee.
* The analytic radius lives at a tap in **feature space**; the iterative walk
  accumulates distance in **input-pixel space**. They are compared as rank
  correlation plus a scale-normalized relative error — never as equality.

## Distillation method

`scsf.methods.depthfrag.DepthFragMethod`:

* Probes `q_l(h_l)` (LayerNorm + small MLP → scalar) at every tapped site
  regress the **detached** per-site targets with Huber loss.
* The predicted per-example depth profile is aggregated by a configurable
  `soft_min` / lower-tail CVaR / `mean` / `min` / `terminal` operator.
* A small terminal fragility head on `final_embedding` regresses the
  **detached** aggregate of the true targets.
* Targets are computed by a **separate target forward** whose BatchNorm runs
  in eval role while the parameters remain differentiable `nn.Parameters`.
  `bn_mode="train"` exists only to benchmark the coupling (see the BatchNorm
  test); it is never the deployed default.
* Inference = terminal logits + terminal head only; confidence is the head
  output (robustness-style: higher = keep). Probes are absent from the
  inference graph (mutating them changes nothing) — tested.
* `target_interval=K` (documented): every K-th step recomputes the eval-plan
  targets and runs the probe/head supervision; the in-between steps train CE
  only. Targets are per-example, so stale targets are never reused.
* `freeze_backbone: true` reproduces the identical probe/head capacity with a
  frozen backbone (the control). Gradient-reach tests assert the end-to-end
  run reaches the backbone and the frozen control does not.

## Required configurations (all served by `configs/methods/depthfrag*.yaml`)

| config                    | role                                        |
|---------------------------|---------------------------------------------|
| `depthfrag_terminal_margin` | terminal true-class margin only (CE baseline, no probes/head) |
| `depthfrag_terminal`        | terminal normalized radius only (top_l1)   |
| `depthfrag_intermediate`    | one intermediate radius (top_l2)           |
| `depthfrag_raw`             | raw full-depth profile (all sites, mean agg) |
| `depthfrag`                 | distilled terminal score (default)         |
| `depthfrag_frozen`          | distilled score, frozen backbone (control) |
| `depthfrag_clip`            | absolute/clipped radius sensitivity control |

## Deployment path

1. Train a `depthfrag` run.
2. Extract: `python -m scsf.extract_depthfg run_dir=... split=val
   out_dir=...`.
3. The deployment module is `backbone + head` (probes discarded); the head's
   output is the fragility confidence. See `scripts/smoke_depthfrag.sh` for a
   full overfit→extract→audit run.