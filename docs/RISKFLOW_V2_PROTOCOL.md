# RiskFlow-V2 protocol: stage-wise risk refinement

Status: **preregistered protocol**. Implemented as the `riskflow_v2` method
alias; RiskFlow-v1 (`riskflow`, `riskflow_concat`, `riskflow_heads`,
`riskflow_cum`, `riskflow_resid`, `riskflow_frozen`, `riskflow_hard`) is
preserved unchanged. This file must not be edited after the implementation
commit without a new protocol revision.

## 1. Why v2 exists (the v1 critique)

RiskFlow-v1 treats network depth as a *sequence of risk updates* but models
each update with a **multiplicative gate**: `r = r + gate * delta(LSTM-style)`
with `gate = sigmoid(gate_logit)`, and imposes a correlation/decorrelation
penalty on the innovation vectors. The gate forces every innovation to be
*widening* (positive or zero mass on `delta`); a gate is a soft switch, not an
identifiable signed correction. The penalty regularizes a correlation that is
not part of the primary generative model. RiskFlow-v1 also has **no fixed
teacher**: its targets are detached hard errors `e = (pred != y)` from the
same live model, which drift with every step.

RiskFlow-v2 removes the gate, makes each stage's innovation an identifiable
**signed, bounded** increment, separates a live-teacher hard target from a
dense soft target, and supervises every stage's risk logit with proper BCE
rather than Huber-on-residual furniture.

## 2. Research thesis

Network depth provides sequentially decodable predictive refinement — not new
Shannon information. Each layer updates the unresolved failure-risk estimate.

## 3. Identifiable stage-wise state

    s_l = stop_gradient(s_{l-1}) + Delta_max * tanh(a_l)

- `s_l` is the stage-l cumulative **failure-risk logit** (scalar per example).
- `a_l` is an unbounded adapter output (the *pre-bound* innovation), mapped to
  `[-Delta_max, +Delta_max]` by `tanh` so innovations are **signed and
  bounded**.
- Later layers may increase **or decrease** risk (signed innovation).
- The multiplicative gate of v1 is **removed** from the primary method.
  Gate-based behavior may exist only as a secondary ablation
  (`riskflow_v2_gate` alias — a config flag `remove_gate: false` that restores
  v1's gate, for ablation only, never part of the reported primary method).

Locked: `Delta_max = 2.0` (logit magnitude), `state_dim = 64`, `cell_hidden =
64`, `token = cls` (from v1 defaults, kept).

The adapter `a_l` is `InputAdapter(pool_tap(h_l))` (LayerNorm → Linear→state).
`tanh` forces the innovation into `[-Delta_max, Delta_max]`; no unbounded
innovations in the primary method.

### 3.1 Stop-gradient routing

`stop_gradient(s_{l-1})` means stage `l`'s loss cannot rewrite stage `l-1`'s
risk module (or any earlier module): the risk modules for previous stages are
kept as *sequential readout points*, not entangled optimizers. The current
stage's BCE loss may still shape `h_l` and the appropriate backbone prefix
(via the adapter→cell→backbone path). Tests explicitly verify:
`autograd.grad(L_l, cell_{<l}.params) == 0` and
`autograd.grad(L_l, adapter_{<l}.params) == 0`, and
`autograd.grad(L_l, backbone.params for prefix at site l) != 0`.

## 4. EMA teacher targets

Teacher is an EMA copy of the student backbone (the same mechanism as
DepthFrag-v2, shared helper or equivalent inline implementation):

    theta_T <- nu*theta_T + (1-nu)*theta_student       nu=0.999

- Teacher receives no gradient; saved/restored in checkpoints exactly.
- Hard target: `e_T = 1[argmax f_teacher(x) != y]`

- Dense stabilization target: `d_T = -log p_teacher(y|x)`

Use the **hard channel for failure prediction** and the **soft channel only
for stabilization**.

## 5. Loss

At every stage `l` (base + each site):

    L_l = BCEWithLogits(s_l, e_T)

i.e. every cumulative risk logit is supervised by the teacher's hard-error
indicator. The terminal stage is included in this stage-wise BCE (it is `s_L`).

A separate bounded soft channel regresses `s_soft_l` against `d_T` using Huber
loss (bounded: `s_soft_l` readouts also bounded via the same `tanh`
innovations; normalize `d_T` to `[0, 1]` by dividing by `log(C)` so soft target
and logits share scale).

**Do not** use

    Huber( innovation, e - q_previous )

as though a probability residual were an exact desired logit increment.
Innovations are taught by the teacher's *target* at the aggregate level
(stage-wise BCE on `s_l` against `e_T`), not by arithmetic subtraction of
probabilities.

## 6. Gradient behavior

- Stop-gradient through `s_{l-1}` prevents later stages from rewriting previous
  risk modules (3.1).
- Current-stage risk loss may still shape `h_l` and the appropriate backbone
  prefix — this is the intended mechanism and it is tested.
- Standard optimization: backbone + adapters + cell + readouts all update via
  the summed stage-wise BCE (+ soft Huber), with per-stage routing enforced by
  stop-gradients.

## 7. Inference

- No true labels at inference.
- Define confidence consistently:

      confidence = 1 - sigmoid(s_L)

  (higher score must mean more trusted; assert in tests that higher
  `1 - sigmoid(s_L)` correlates with higher correctness).
- Report sequential-state inference overhead (per-example extra MACs =
  sum over sites of `adapter_l` + `cell` + `readout`; compute at build time
  and log).
- A final-head distilled variant may be implemented as a secondary
  zero-overhead option (`riskflow_v2_head` alias), not the primary method; it
  is excluded from the three-seed main matrix unless the sequential case
  validates the principle.

## 8. Excluded from the primary method

- multiplicative gate (ablation-only `riskflow_v2_gate`);
- correlation/decorrelation penalty;
- monotonic-risk constraint;
- unbounded innovations.

## 9. Required logging

- Per-stage state logits `s_l` and probabilities `sigmoid(s_l)`.
- Signed innovation values `Delta_max * tanh(a_l)`.
- Positive vs negative correction frequency (fraction of stages where
  `innovation > 0` vs `< 0`).
- Stage-wise BCE (`bce_l` per stage + total).
- Teacher error prevalence `mean(e_T)` per batch/step.
- Soft-target distribution summary (`mean/std d_T`).
- Per-stage gradient mass (`||grad wrt cell_l||`, `||grad wrt adapter_l||`,
  `||backbone-prefix grad||`).
- Final trajectory for correct and erroneous examples (store `s_l` trace for
  sampled examples, logged as JSON arrays in the method log).

## 10. Tests (`tests/test_riskflow_v2.py`)

1. Previous-state stop-gradient (`autograd.grad(L_l, cell_{<l}) == 0`).
2. Intended current-feature/backbone gradients flow (`!= 0`).
3. Positive and negative corrections both reachable (construct teacher that
  demands a decrease and assert innovation negative).
4. Innovation bounds (`abs(innovation) <= Delta_max` for random inputs, all
  stages).
5. EMA target stability and exact resume.
6. Behavior with zero errors in a minibatch (all `e_T = 0`; BCE finite;
  states relax toward `-inf`-ish low logits; no NaN).
7. Label-free inference (`predict_batch` without `y`).
8. Confidence direction (higher `1 - sigmoid(s_L)` ⇔ lower error rate on a
  synthetic ordered population).
9. Architecture-neutral taps (parametrized backbones).
10. Existing RiskFlow tests remain green.

## 11. Main runs

After the current SAGE-v2 queue is idle:

    - riskflow_v2
    - CIFAR-10 and CIFAR-100
    - seeds 13, 17, 23
    - six total runs

Recipe `ccl_sc_reference` (300 epochs, batch 64, SGD lr 0.1/0.9/wd 5e-4, step
0.5/25), existing val/test split, selection rule
`min_val_aurc_among_acc>=best_acc-1.0pp`, unchanged metrics. Compare against
RiskFlow-v1 (`riskflow`) using matched seeds. Negative results are preserved.

## 12. Locked constants (single values for both datasets)

| constant | value |
|---|---|
| nu (EMA teacher) | 0.999 |
| Delta_max | 2.0 |
| state_dim | 64 |
| cell_hidden | 64 |
| huber_delta (soft channel) | 1.0 |
| token | cls |
| soft normalization | d_T / log(C) ∈ [0, 1] |

No dataset-specific hyperparameters anywhere in the primary method.