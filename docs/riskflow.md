# RiskFlow: accumulate conditional failure innovations

RiskFlow frames selective risk as a **persistent sequential failure-evidence
state** over the ordered backbone taps. Instead of reading one confidence
estimate off a concatenation of features (SCSF) or averaging independent
intermediate confidence heads (TULIP-style aggregation), it maintains a small
per-example state vector that is updated recursively across depth, where each
depth contributes only the *new* failure evidence not already explained by the
cumulative state.

## Why this is not TULIP / deep exits

The nearest published boundary is TULIP (Transitional Uncertainty with Layered
Intermediate Predictions, ICML 2024) and the deep-exit family. The distinction
is the **object being supervised**:

* TULIP / deep exits attach an independent predictor (or head) per layer and
  combine/average the per-layer outputs at the end. Each layer is supervised to
  predict the *same* final label or confidence directly, so adjacent layers
  learn highly correlated, redundant snapshots of the same signal. The
  aggregation mechanism is a static weighted combination **after** the fact,
  not a state that evolves **through** the network.
* RiskFlow supervises the **conditional residual** — the *change* in failure
  evidence the current depth contributes relative to the previous cumulative
  state — not an independent confidence snapshot. The state is persisted
  through depth (no end-concatenation), and per-example gates modulate how
  much of each proposed innovation is admitted. The novelty claim is supported
  **only if** residual updates are less redundant across depth than
  independent-head outputs while the cumulative risk is more predictive; both
  are measured explicitly (see Diagnostics).

In short: TULIP combines independent predictions; RiskFlow accumulates
conditional innovations into one persistent state. We do not claim
early-exit/classification at intermediate depths; the intermediate quantities
are training-time evidence only, and the final cumulative risk is the
deployment score.

## Algorithm

Let `h_1..h_L` be the ordered candidate features from the backbone registry
(`backbone.taps`), pooled per architecture (GAP for CNN stages; CLS or token
mean for ViT) and projected by an architecture-specific **input adapter** into
a shared state space of dimension `D` (`state_dim`, default 64):

```text
adapter_l(h_l) = Linear_l(LayerNorm(pool_tap(h_l)))      # (B, D)
```

The shared update cell `psi` and the shared readout heads are
architecture-agnostic: only the input adapters differ per backbone.

```text
r_0 = base_state                            # learned (B, D) broadcast vector
for l in 1..L:
    delta_r_l, gate_logit_l = psi(adapter_l(h_l), r_{l-1})
    gate_l = sigmoid(gate_logit_l)                      # per-example gate
    r_l = r_{l-1} + gate_l * delta_r_l
    s_hard_l = readout_hard(r_l)        # error-risk channel (inference)
    s_soft_l = readout_soft(r_l)        # difficulty channel (auxiliary)
    q_l = sigmoid(s_hard_l)
```

`L` is the number of taps (4 for ResNet-18/WideResNet groups/stages, 5 for VGG
pools, 12 for DeiT-S blocks). The state `r_l` is `(B, D)`, `D` default 64.
`psi` is `RiskCell`: `Linear(2D, H) -> ReLU -> Linear(H, D)` for `delta_r_l`
and a `Linear(2D, 1)` for the gate logit when gates are sample-dependent; for
the fixed-gate ablation the gate comes from per-depth learnable scalars.

## State size

Per example the persistent state is `D = state_dim` (default 64) floats — a
single shared vector updated through depth, plus the base state. No per-depth
accumulation of the full feature space; only the pooled, projected
innovations touch the state.

## Loss terms

For a batch with prediction `pred = argmax logits` and labels `y`:

* **Hard pseudo-residual** (detached error target):
  `e = stopgrad(1[pred != y])`,
  `eps_hard_l = stopgrad(e - sigmoid(s_hard_{l-1}))`.
* **Hard innovation loss**: `Huber(s_hard_l - s_hard_{l-1}, eps_hard_l)`.
  The scalar contribution induced by `delta_r_l` is exactly
  `s_l - s_{l-1} = readout_hard(gate_l * delta_r_l)`. The target is detached so
  later stages never backprop through it; the current innovation loss may shape
  its feature adapter/backbone.
* **Soft channel**: detached final difficulty target
  `d = stopgrad(NLL(logits, y) / log(C))` (normalized true-label
  cross-entropy, in `[0, 1]` roughly), with its own pseudo-residual
  `eps_soft_l = stopgrad(d - sigmoid(s_soft_{l-1}))` and an analogous Huber
  innovation loss. The soft channel is **auxiliary**: it is never the inference
  score, and `readout_soft` is excluded from deployment parameter counts.
  Both channels are logged.
* **Terminal proper-scoring loss**: `BCE(sigmoid(s_hard_L), e)` keeps the
  cumulative state meaningful; the soft terminal `BCE(sigmoid(s_soft_L), d)`
  is auxiliary. (Also `cross_entropy(logits, y)` for the classifier.)
* **Innovation decorrelation penalty** (small, configurable `decorr_scale`,
  default 0.01): the mean absolute off-diagonal entry of the per-example Gram
  of L2-normalized innovation vectors, averaged over the batch. It penalizes
  redundant updates across depth and is robust to zero-variance columns (an
  `eps` floor, never a fragile scalar correlation).

Inference score: `confidence = -s_hard_L` (higher cumulative risk -> lower
confidence). The soft channel and all per-depth quantities are exported for
analysis but are not part of the deployment head.

## Modes (ablation ladder / required comparisons)

| config | comparison | residual targets | gates | soft |
|--------|-----------|------------------|-------|------|
| `riskflow_concat` | 1: concat head | n/a | n/a | off |
| `riskflow_heads` | 2: independent heads + weight sum | n/a (per-head BCE + weighted sum) | n/a | off |
| `riskflow_cum` | 3: cum state, no residuals | off | fixed | off |
| `riskflow_resid` | 4: cum state + residuals | on | fixed | off |
| `riskflow` (default) | 5: residual + sample gates | on | sample | on |
| `riskflow_frozen` | 6: frozen backbone | on | sample | on |
| `riskflow_hard` | 7: hard only | on | sample | off |

All share the `RiskFlowMethod` class driven by `mode` / `use_soft` /
`freeze_backbone` / `gate_mode`. This directly implements the seven required
comparisons in the empirical contract.

## Diagnostics

* **Per-sample, per-depth exports** (`export_trace`): gate values, innovation
  logits/vectors, cumulative risk logits, and hard/soft pseudo-residual
  targets (numpy arrays; per-sample row-major).
* **Redundancy report** (`redundancy_report`): cross-depth Pearson correlation
  and centered linear CKA (`pairwise_linear_cka`) for (a) independent-head
  outputs and (b) RiskFlow innovations, and the cumulative-risk correlation
  with the final score. Novelty is supported only if innovations are less
  redundant while cumulative risk is more predictive.
* **Trajectory plots**: per-example gate / innovation / risk trajectories for
  the four fixed-rule categories (easy-correct, ambiguous-correct,
  high-confidence-wrong, corrupted), assigned by the config thresholds
  `cat_lo` / `cat_hi` (default 0.3 / 0.7) — never hand-selected after seeing
  plots.

## Deployment overhead

RiskFlow carries a nonzero inference cost beyond the backbone: `L` input
adapters (LayerNorm + Linear each), a shared update cell applied `L` times,
two small readout heads and the base-state vector. Parameters, MACs, latency,
and memory are reported directly in the run ledger (`report_overhead`) for the
default RiskFlow versus the SCSF concatenation baseline; the overhead is
quantified rather than assumed negligible. See `riskflow_report.md` for current
measured numbers.

## Notes

* State size is `state_dim` (default 64).
* Only the **input adapters** are architecture-specific; the update cell,
  readouts, and base state are shared, so nothing special-cases a backbone.
* We do **not** run or claim the final gate at deployment; the gate is a
  training-time sample-dependent modulator whose trajectory is diagnostic.
