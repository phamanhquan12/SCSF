# DSSPMI / ACCon Experimental Recipe
## Training-Dynamics, Coverage-Aware, and Representation-Shaping Variants for Selective Classification

**Purpose.** This document is an implementation recipe for systematically testing the ideas discussed around DSSPMI/SCSF + ACCon. The goal is not to merge everything into one large method immediately. Implement each idea as an isolated variant, compare them under the same training/evaluation protocol, then combine only the variants that show complementary gains.

**Current code context (from `acccon_methodology.md`).**
- Main training entry: `train_next_scsf.py`
- Existing variant: `--variant acccon`
- Helper: `soft_coverage_masks` in `train_cbr_scsf.py`
- Existing modules: `ProjectionHead`, `FeatureQueue` in `train_search_scsf.py`
- Backbone on CIFAR: VGG16-BN
- Confidence head input: `[pool4, pool5, stopgrad(logits)]`
- Confidence head target: hard correctness
- Contrastive embedding: projected `pool5`
- Queue: one online FIFO, no momentum encoder
- Official evaluation: **last-epoch checkpoint (`last.pth`) on the official 10k test set**
- Current coverage set: `{0.80, 0.90, 0.95}`
- Current ACCon temperature: `tau = 0.1`
- Soft-acceptance temperature: `T = 0.2`
- Contrastive coefficient: `eta = 0.5`
- Training: 300 epochs; CE warmup 1–100; joint ramp 101–120; joint 121–300

---

# 0. Experimental rules: do this before implementing variants

## 0.1 Do not overwrite the current baselines

Keep at least these checkpoints/results untouched:

- `baseline_scsf/last.pth`
- `acccon_old_cancelled_wi/last.pth`
- `acccon_query_weighted/last.pth`

Every new experiment should have a unique output directory containing:

```text
config.json
last.pth
best.pth                # optional; do NOT use for official comparison unless explicitly planned
train_log.csv
test_metrics.json
band_analysis.json
dynamics_summary.npz    # only for dynamics variants
```

Do not silently change preprocessing, optimizer, LR schedule, data order, augmentation, or evaluation code between variants.

## 0.2 Primary metrics

Always report:

1. Classification accuracy.
2. AURC.
3. E-AURC if already supported.
4. AUROC for correct-vs-wrong ranking.
5. Selective risk at 100%, 95%, 90%, 85%, 80%, 75%, 70%; optionally 50%, 25%, 10% for diagnosing the high-confidence head.
6. Error distribution by rank band:

```text
0–10
10–50
50–70
70–75
75–80
80–85
85–90
90–95
95–100
```

For each band save `n_samples`, `n_errors`, and `slice_error_rate`.

7. Median/mean rank of all errors.
8. Median/mean rank of shared errors when comparing two checkpoints.
9. Number of unique mistakes per model when comparing two checkpoints.

## 0.3 Primary objective

The target is not merely higher accuracy. For selective classification, the desired shape is:

```text
top-confidence region: very few errors
middle shoulder: clean enough to preserve risk at 75–90% coverage
bottom tail: strongly enriched with errors
```

A method can improve AURC without improving classification accuracy by moving errors down the score ranking.

## 0.4 Hyperparameter discipline

Do not tune hyperparameters on the official test set.

Recommended workflow:

```text
train -> validation selection/analysis -> freeze hyperparameters -> official 10k test
```

For the first screening run, seed 42 is acceptable. For any promising method, run at least 3 seeds and report mean ± std.

---

# 1. Reproduce the two ACCon forms first

This is necessary because later variants build on their observed trade-off.

## 1.1 Old ACCon: cancelled query weight

For query `i`, positives `P(i)`:

\[
L_i^{old}
=
-
\frac{
\sum_{j\in P(i)} w_i w_j \log p_{ij}
}{
\sum_{j\in P(i)} w_i w_j + \epsilon
}.
\]

For any `w_i > 0`, `w_i` approximately cancels:

\[
L_i^{old}
\approx
-
\frac{
\sum_j w_j\log p_{ij}
}{
\sum_jw_j+\epsilon
}.
\]

**Interpretation:** almost every query receives full-strength contrastive training; only key/positive reliability is meaningfully weighted.

Expected behavior from the existing run:
- comparatively better 75–85% shoulder;
- weaker extreme top purification than the corrected form.

Keep this as a deliberate baseline named `acccon_key_only` rather than treating it only as a bug. It tests whether query weighting is beneficial.

## 1.2 Corrected ACCon: query-weighted mean

Define per-query positive loss:

\[
\ell_i^{con}
=
-
\frac{
\sum_{j\in P(i)} w_j\log p_{ij}
}{
\sum_{j\in P(i)}w_j+\epsilon
}.
\]

Then:

\[
L_{acccon}
=
\frac{
\sum_i w_i \ell_i^{con}
}{
\sum_iw_i+\epsilon
}.
\]

**Interpretation:** high-acceptance queries dominate; low-acceptance queries contribute little.

Expected behavior from the existing run:
- purer top-confidence core;
- more errors concentrated at the extreme bottom;
- possible deterioration around 75–85% because medium-acceptance queries train less.

Use this as `acccon_query_weighted`.

---

# 2. Boundary-aware ACCon

## 2.1 Motivation

The corrected query weighting improves the trusted core but can starve samples near the acceptance/rejection transition.

Define a boundary score:

\[
b_i = 4 w_i(1-w_i).
\]

Properties:

```text
w = 0.0 -> b = 0.00
w = 0.1 -> b = 0.36
w = 0.25 -> b = 0.75
w = 0.5 -> b = 1.00
w = 0.75 -> b = 0.75
w = 0.9 -> b = 0.36
w = 1.0 -> b = 0.00
```

Then use query weight:

\[
q_i = w_i + \beta b_i.
\]

Contrastive loss:

\[
L_{boundary}
=
\frac{
\sum_i q_i \ell_i^{con}
}{
\sum_i q_i+\epsilon
}.
\]

Key weighting stays `w_j` initially.

## 2.2 First sweep

Do not change the coverage set yet.

```text
beta = 0.00
beta = 0.05
beta = 0.10
beta = 0.15
beta = 0.25
```

`beta=0` must exactly reproduce `acccon_query_weighted`. Recommended first serious candidate: `beta=0.10`.

## 2.3 Why beta <= 0.25 is a useful first range

\[
q(w)=w+4\beta w(1-w)
\]

has derivative:

\[
q'(w)=1+4\beta-8\beta w.
\]

For `beta <= 0.25`, `q(w)` remains non-decreasing over `[0,1]`. Thus a more accepted sample never receives less query weight than a less accepted sample, while mid-acceptance examples receive a bonus.

## 2.4 Success criterion

Prefer a variant that:
- keeps 0–10% errors nearly unchanged relative to corrected ACCon;
- reduces errors in 75–80 and/or 80–85;
- keeps or increases error concentration in 95–100;
- lowers AURC.

Do not call it a success merely because 80% risk improves if the top-confidence region becomes noticeably dirtier and AURC worsens.

## 2.5 Optional follow-up: denser target coverage set

Only after selecting a promising beta, compare:

```text
C1 = {0.80, 0.90, 0.95}
C2 = {0.80, 0.85, 0.90, 0.95}
C3 = {0.75, 0.80, 0.85, 0.90, 0.95}
```

Do not change beta and coverage grid simultaneously in the first ablation.

---

# 3. Query-weight floor baseline

This is a simpler competitor to boundary-aware weighting.

Define:

\[
q_i=\alpha+(1-\alpha)w_i.
\]

Test:

```text
alpha = 0.10
alpha = 0.20
alpha = 0.30
```

Use the same positive-key loss and weighted mean over queries.

**Purpose:** determine whether boundary-aware weighting adds value beyond simply giving every sample a minimum amount of contrastive training.

---

# 4. Soft risk-coverage / tail-ranking loss

## 4.1 Motivation

Correctness BCE says “wrong examples should have lower scores,” but it does not directly optimize which examples survive at a target coverage.

For each example:

\[
e_i=\mathbf 1[\hat y_i\neq y_i].
\]

For each target coverage `c`, compute threshold `h_c` from **detached scores** so that mean soft acceptance approximates `c`.

Then compute the differentiable acceptance mask using the **non-detached** score:

\[
a_i^{(c)}
=
\sigma\left(
\frac{s_i-\operatorname{stopgrad}(h_c)}{T}
\right).
\]

Define soft accepted risk:

\[
L_{risk}^{(c)}
=
\frac{
\sum_i e_i a_i^{(c)}
}{
\sum_i a_i^{(c)}+\epsilon
}.
\]

Aggregate:

\[
L_{tail}
=
\frac1{|\mathcal C|}
\sum_{c\in\mathcal C}
L_{risk}^{(c)}.
\]

Recommended initial coverage grid:

```text
C_tail = {0.70, 0.80, 0.90, 0.95}
```

## 4.2 Total loss

First test without altering ACCon:

\[
L
=
L_{CE}
+
\lambda_t L_{BCE}
+
\eta L_{acccon}
+
\mu L_{tail}.
\]

Sweep:

```text
mu = 0.00
mu = 0.05
mu = 0.10
mu = 0.20
```

## 4.3 Important gradient rule

For `L_tail`:

```text
scores.detach() -> threshold solver
scores          -> sigmoid acceptance -> L_tail -> gradient
threshold       -> detached
wrong/correct label -> detached
```

Do **not** detach the score inside the final soft acceptance or the loss cannot reshape the ranking.

## 4.4 Expected effect

A wrong sample with a score high enough to remain in many accepted sets is repeatedly penalized. A wrong sample already pushed into the bottom tail receives little additional penalty.

This is the most direct variant for the goal: **concentrate wrong predictions at low-confidence extremes.**

---

# 5. Hard-error pairwise ranking loss

## 5.1 Objective

For correct query `i` and wrong query `j`, enforce:

\[
s_i > s_j.
\]

Use:

\[
L_{rank}(i,j)
=
\operatorname{softplus}(s_j-s_i+m).
\]

Recommended initial margin: `m = 0.2` (adjust to score scale).

## 5.2 Do not pair randomly

Hard-mining strategy per batch:
1. collect wrong samples;
2. sort them by confidence descending;
3. choose top-K high-confidence errors;
4. for each such wrong sample, pair with one or more correct examples ranked near/above it or with top-confidence correct examples.

Suggested:

```text
K_wrong = min(num_wrong, 16)
K_correct_per_wrong = 4
```

## 5.3 Sweep

```text
gamma_rank = {0.02, 0.05, 0.10}
```

Test this separately from `L_tail` first.

---

# 6. Focal correctness loss for high-confidence errors

This is a cheap baseline.

If `p_i = sigmoid(s_i)` and `t_i` is current correctness, for wrong samples (`t_i=0`):

\[
L_{wrong}
=
-p_i^\gamma \log(1-p_i).
\]

Recommended:

```text
gamma = {1, 2, 3}
```

Purpose: penalize high-confidence mistakes much more strongly than errors already assigned low confidence.

---

# 7. Training-dynamics recorder

All later dynamics variants require stable per-example IDs.

## 7.1 Dataset requirement

Every training example must return:

```python
image, label, sample_id
```

`sample_id` must be deterministic across epochs, independent of shuffle order.

## 7.2 What to record per sample

At each selected epoch after a configurable start epoch, record:

```text
correct[e, i]       # 0/1
p_true[e, i]        # softmax probability of ground-truth class
margin[e, i]        # true-class logit - max other logit
pred[e, i]          # predicted class, optional
loss[e, i]          # CE per example, optional
selector[e, i]      # learned confidence score, optional
```

Do not store full feature tensors for every epoch by default.

Recommended storage:

```text
float16: p_true, margin, selector
uint8: correct
int16: pred
```

## 7.3 Recording window

Implement configurable strategies:

```text
all epochs
last K epochs
joint phase only (101–300)
late joint phase only (e.g. 151–300)
```

For initial experiments use:

```text
dynamics_start_epoch = 101
window = rolling last 20 epochs
```

---

# 8. Temporal correctness target

## 8.1 Motivation

Current target:

\[
t_i^{(e)}
=
\mathbf1[\hat y_i^{(e)}=y_i].
\]

This discards whether an example has been stable or unstable over time.

Define rolling temporal reliability:

\[
r_i^{temp}
=
\frac1K
\sum_{u=e-K}^{e-1}
c_i^{(u)}.
\]

Examples:

```text
C C C C C -> reliability 1.0
C W C W C -> reliability 0.6
W W W C W -> reliability 0.2
```

Use this soft target for the confidence head:

\[
L_{temp}
=
-r_i^{temp}\log p_i
-(1-r_i^{temp})\log(1-p_i).
\]

## 8.2 Avoid immediate target leakage

Prefer using previous epochs only. Update current-epoch dynamics **after** computing the current loss.

## 8.3 Sweep

```text
K = {5, 10, 20, 50}
```

Start with `K=20`.

Compare:

```text
hard current correctness BCE
temporal correctness BCE
0.5 * hard + 0.5 * temporal
```

Hybrid target:

\[
r_i = \alpha t_i^{current}+(1-\alpha)r_i^{temp}
\]

with `alpha = {0.0, 0.25, 0.5}`.

## 8.4 Main hypothesis

Temporal reliability should be a better supervision target if current training errors are noisy/transient and test-time failures resemble persistent/unstable training cases more than one-epoch mistakes.

---

# 9. Example forgetting

Define a forgetting event:

\[
F_i^{(e)}
=
\mathbf1[
c_i^{(e-1)}=1
\land
c_i^{(e)}=0
].
\]

Cumulative count:

\[
N_i^{forget}
=
\sum_e F_i^{(e)}.
\]

Also record learning events:

\[
L_i^{(e)}
=
\mathbf1[
c_i^{(e-1)}=0
\land
c_i^{(e)}=1
].
\]

## 9.1 Derived features

Normalize:

\[
f_i =
\frac{N_i^{forget}}{\max_j N_j^{forget}+\epsilon}.
\]

Possible temporal stability:

\[
S_i^{forget}=e^{-\gamma_f N_i^{forget}}.
\]

Try `gamma_f = {0.25, 0.5, 1.0}`.

## 9.2 Uses to test

### Variant A: confidence target modifier

\[
r_i = r_i^{temp} \cdot S_i^{forget}.
\]

Be cautious: this may over-penalize genuinely ambiguous-but-useful samples.

### Variant B: auxiliary training-only prediction target

Predict forgetting/stability from final intermediate features. Discard this teacher at inference.

### Variant C: analysis first

Before incorporating it into the loss, test whether `N_forget` correlates with final error, low selector score, intermediate-layer disagreement, or low margin.

---

# 10. Dataset Cartography statistics

For each training example over a dynamics window:

\[
\mu_i = E_t[p_t(y_i|x_i)]
\]

\[
\sigma_i = Std_t[p_t(y_i|x_i)]
\]

\[
c_i = E_t[\mathbf1(\hat y_t=y_i)].
\]

Interpretation:

```text
easy/stable:     high mu, low sigma
ambiguous:       moderate mu, high sigma
hard-to-learn:   low mu, often low/moderate sigma
```

## 10.1 Important design principle

Do **not** simply downweight all ambiguous examples. Established dataset-cartography results suggest ambiguous examples can be especially useful for generalization.

Therefore separate:
- how much an example should learn as a query;
- how much it should act as a prototype/key.

---

# 11. Cartography-aware asymmetric ACCon

This is one of the highest-priority experiments.

## 11.1 Define temporal ambiguity and stability

Simple normalized ambiguity:

\[
A_i =
\operatorname{clip}
\left(
\frac{\sigma_i}{\sigma_{ref}+\epsilon},
0,1
\right).
\]

Prefer `sigma_ref = 95th percentile` over max to reduce sensitivity to outliers.

Stability:

\[
S_i=e^{-\gamma_A A_i}.
\]

Try `gamma_A = {1, 2, 4}`.

## 11.2 Asymmetric roles

Define query importance:

\[
q_i=w_i+\beta A_i
\]

or safer:

\[
q_i=w_i+\beta w_iA_i.
\]

The second form prevents highly rejected samples from becoming strong queries solely because they are unstable.

Define key reliability:

\[
k_i=w_iS_i.
\]

Then:

\[
\ell_i
=
-
\frac{
\sum_{j\in P(i)} k_j\log p_{ij}
}{
\sum_{j\in P(i)}k_j+\epsilon
}
\]

and:

\[
L_{carto-con}
=
\frac{
\sum_i q_i\ell_i
}{
\sum_iq_i+\epsilon
}.
\]

## 11.3 Interpretation

```text
easy/stable accepted:
  q high, k high
  -> learns and acts as a strong class prototype

ambiguous but accepted/near-boundary:
  q boosted, k reduced
  -> receives training but does not strongly pull other samples toward itself

hard/rejected:
  q low, k very low
  -> limited effect
```

Goal:

```text
stable class-core formation
+ decision-boundary refinement
- prototype contamination
```

## 11.4 First sweep

Use only one dynamics statistic initially: normalized variability of `p_true`.

Try:

```text
beta = {0.05, 0.10, 0.20}
gamma_A = {1, 2}
```

Do not combine forgetting, AUM, EL2N, and cartography in the first run.

---

# 12. Margin dynamics / AUM-inspired signals

## 12.1 Per-example margin

At epoch `t`:

\[
m_i^{(t)}
=
z_{y_i}^{(t)}
-
\max_{k\ne y_i}z_k^{(t)}.
\]

Define rolling average margin:

\[
\overline m_i
=
\frac1K\sum_t m_i^{(t)}.
\]

And margin variability:

\[
\sigma_{m,i}=Std_t[m_i^{(t)}].
\]

## 12.2 Uses in DSSPMI

### A. Reliability target

\[
r_i^{margin}
=
\sigma\left(\frac{\overline m_i}{T_m}\right).
\]

Sweep `T_m = {0.5, 1.0, 2.0}`.

### B. Hardness filter for contrastive keys

If margin is persistently strongly negative, reduce its key weight:

\[
k_i
=
w_i\cdot
\sigma\left(\frac{\overline m_i-\delta}{T_m}\right).
\]

### C. Ambiguity detector

Examples with average margin near zero and high margin variability are likely boundary cases:

\[
A_i^{margin}
=
\exp(-|\overline m_i|/\tau_m)
\cdot
\operatorname{norm}(\sigma_{m,i}).
\]

Use this only after simpler cartography variability succeeds.

---

# 13. EL2N early-difficulty prior

EL2N score:

\[
EL2N_i
=
\|p(x_i)-onehot(y_i)\|_2.
\]

## 13.1 Recording

At an early epoch or averaged over a small early window:

```text
epochs 5–10
or
epochs 10–20
```

compute EL2N per example and normalize to `[0,1]`.

## 13.2 Experiments

Do not directly assume high EL2N equals low reliability. First test whether early EL2N predicts:
- late temporal variability;
- forgetting count;
- final confidence score;
- final error;
- low ranking position.

Possible simple use:

\[
q_i = w_i + \beta_E \cdot EL2N_i \cdot w_i
\]

for query training, while:

\[
k_i=w_i(1-EL2N_i)
\]

for keys.

Use only if correlations support the interpretation.

---

# 14. Cross-depth disagreement / prediction-depth-inspired signals

This fits DSSPMI particularly well.

## 14.1 Add lightweight auxiliary probes

Attach training-only linear classifiers to selected intermediate features:

```text
pool3 -> probe3
pool4 -> probe4
pool5 -> probe5
final logits -> final prediction
```

Two experimental modes:

### Diagnostic-only probes

Features detached before probe. They measure depth disagreement without affecting backbone.

### Deep-supervision probes

Features not detached; use a tiny supervision coefficient (`lambda_probe = {0.01, 0.05}`). Start with diagnostic-only.

## 14.2 Cross-depth disagreement

Let predictions be:

\[
\hat y_i^{(3)},
\hat y_i^{(4)},
\hat y_i^{(5)},
\hat y_i^{final}.
\]

Define disagreement:

\[
D_i
=
\frac1L
\sum_{\ell}
\mathbf1[
\hat y_i^{(\ell)}
\ne
\hat y_i^{final}
].
\]

Soft alternative: average JS divergence between intermediate and final predictive distributions.

Start with simple discrete disagreement.

## 14.3 Test-time score augmentation

Because depth features are available in one final forward pass, this signal can be used at inference if desired.

Options:

```text
A. confidence head already learns from multi-depth features; no explicit D input
B. append probe predictions/probabilities to confidence head
C. append scalar D or soft-D to confidence head
```

Test B/C only after diagnostic analysis confirms D predicts errors.

---

# 15. Temporal + depth reliability distillation

This is the most research-oriented variant.

## 15.1 Core idea

Training-time temporal statistics are unavailable for a new test sample. Use them as **teacher targets** so the final single-pass model learns to infer reliability from current intermediate representations.

Training:

```text
historical training trajectory
        |
        v
temporal reliability / ambiguity target
        |
        v
pool4 + pool5 + logits -> selector/student
```

Inference:

```text
single final model
x -> pool4 + pool5 + logits -> selector score
```

No historical checkpoint ensemble at inference.

## 15.2 Teacher target candidates

Implement separately:

```text
T1 = rolling correctness rate
T2 = dataset-cartography mean confidence
T3 = 1 - normalized confidence variability
T4 = sigmoid(average margin / T)
T5 = learned composite only after T1–T4 are evaluated
```

Recommended first target: rolling correctness rate.

## 15.3 Multi-task version

Instead of collapsing all dynamics into one scalar, add small training-only heads for:
- selector correctness score;
- temporal stability prediction;
- ambiguity prediction.

Start simple; do not build a multi-head system before scalar temporal-target experiments work.

---

# 16. Out-of-fold correctness supervision

This directly addresses whether training errors resemble inference errors.

## 16.1 Rationale

Ordinary training correctness is in-sample:

```text
x_i -> model trained on x_i -> correct/wrong target
```

Out-of-fold (OOF) correctness is closer to inference:

```text
x_i -> model NOT trained on x_i -> correct/wrong target
```

## 16.2 K-fold recipe

Use e.g. 5 folds:

```text
D = D1 U D2 U D3 U D4 U D5
```

For each fold `k`:
1. train a base classifier on `D \ Dk`;
2. predict on `Dk`;
3. save prediction, logits, correctness, confidence;
4. combine predictions for all folds.

This gives one out-of-sample target per training example. Then train final DSSPMI on all training data using OOF reliability targets.

## 16.3 Cost-aware version

If 5 full trainings are too expensive:
- 2-fold or 3-fold cross-fitting;
- shorter teacher training;
- cached teachers.

OOF is expensive, so treat it as a high-value ablation, not the first implementation.

---

# 17. Augmentation-based robustness supervision

Cheaper alternative to cross-fitting.

For each training sample generate label-preserving transformed views:

\[
\tilde x=A(x).
\]

Record whether the current model remains correct.

Define augmentation robustness:

\[
r_i^{aug}
=
\frac1M
\sum_{m=1}^M
\mathbf1[
\hat y(A_m(x_i))=y_i
].
\]

Use this as:
- auxiliary temporal/reliability target;
- query weight;
- key reliability.

Start with `M=2` extra views if compute allows. For medical data, use only clinically label-preserving transformations.

---

# 18. Direct test of train-error -> test-error generalization

Before adding more losses, quantify the problem.

## 18.1 Frozen feature probe experiment

Train the base/backbone.

Create feature datasets:

```text
train: pool4, pool5, logits, correct/wrong
validation: same
test: same (evaluation only)
```

For each representation, fit a simple logistic/linear correctness probe on TRAIN and evaluate correct-vs-wrong AUROC on TRAIN/VAL/TEST.

Report:

| Source | Train AUROC | Val AUROC | Test AUROC |
|---|---:|---:|---:|
| pool4 | | | |
| pool5 | | | |
| logits | | | |
| pool4+pool5 | | | |

Large train-to-test drop indicates that “what training errors look like” does not fully generalize.

---

# 19. Suggested variant registry

Implement a registry so each idea is independently callable.

Example CLI:

```bash
python train_next_scsf.py --variant scsf
python train_next_scsf.py --variant acccon_key_only
python train_next_scsf.py --variant acccon_query
python train_next_scsf.py --variant acccon_boundary --boundary-beta 0.10
python train_next_scsf.py --variant acccon_floor --query-floor 0.20
python train_next_scsf.py --variant acccon_tail --tail-weight 0.10
python train_next_scsf.py --variant acccon_rank --rank-weight 0.05
python train_next_scsf.py --variant temporal_target --dyn-window 20
python train_next_scsf.py --variant carto_acccon --ambiguity-beta 0.10 --stability-gamma 2
python train_next_scsf.py --variant margin_target --dyn-window 20
python train_next_scsf.py --variant depth_disagreement
python train_next_scsf.py --variant temporal_depth
```

Avoid a giant set of nested `if` statements. Prefer a configuration/dataclass:

```python
@dataclass
class VariantConfig:
    use_acccon: bool = False
    query_weight_mode: str = "none"   # none|acceptance|boundary|floor|cartography
    use_tail_loss: bool = False
    use_rank_loss: bool = False
    confidence_target: str = "hard"   # hard|temporal|margin|hybrid
    record_dynamics: bool = False
    use_depth_probes: bool = False
```

---

# 20. TrainingDynamicsTracker implementation sketch

```python
class TrainingDynamicsTracker:
    def __init__(self, n_samples, window=20):
        self.n = n_samples
        self.window = window

        self.correct_hist = ...
        self.ptrue_hist = ...
        self.margin_hist = ...

        self.prev_correct = torch.zeros(n_samples, dtype=torch.bool)
        self.seen_prev = torch.zeros(n_samples, dtype=torch.bool)
        self.forget_count = torch.zeros(n_samples, dtype=torch.int32)
        self.learn_count = torch.zeros(n_samples, dtype=torch.int32)

    @torch.no_grad()
    def update(self, ids, logits, labels):
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        correct = pred.eq(labels)
        ptrue = probs.gather(1, labels[:, None]).squeeze(1)

        masked = logits.clone()
        masked[torch.arange(len(labels)), labels] = -torch.inf
        max_other = masked.max(dim=1).values
        true_logit = logits.gather(1, labels[:, None]).squeeze(1)
        margin = true_logit - max_other

        prev = self.prev_correct[ids]
        seen = self.seen_prev[ids]

        forgetting = seen & prev & (~correct.cpu())
        learning = seen & (~prev) & correct.cpu()

        self.forget_count[ids] += forgetting.int()
        self.learn_count[ids] += learning.int()
        self.prev_correct[ids] = correct.cpu()
        self.seen_prev[ids] = True

        # append rolling statistics keyed by stable sample id
        ...
```

Important:
- update history **after** computing current loss if previous-epoch-only targets are desired;
- if updating per mini-batch, distinguish “previous visit” from “previous epoch”;
- for strict epoch-level dynamics, aggregate/save exactly once per sample per epoch.

---

# 21. Two options for collecting dynamics

## Option A: dynamics from normal stochastic training pass

Pros:
- almost free.

Cons:
- augmentation noise becomes part of the trajectory;
- each epoch observes a transformed version of the sample.

This may be useful if you want robustness dynamics, but it is not identical to classic dataset cartography.

## Option B: deterministic dynamics pass

At the end of each epoch (or every N epochs):
- set model to eval;
- use training dataset with validation/test transforms (no random augmentation);
- run no-grad inference;
- update dynamics.

Pros:
- clean, comparable trajectories;
- closer to dataset-cartography methodology.

Cons:
- additional forward pass.

Recommended for thesis-quality analysis if compute allows: every 2 or 5 epochs after epoch 100.

---

# 22. Experiment ladder: do not run the combinatorial product

## Stage A — establish ranking-shape baselines

A0. SCSF  
A1. ACCon key-only / cancelled query weight  
A2. ACCon corrected query weight

Confirm previous error-band observations.

## Stage B — fix shoulder vs core trade-off

B1. Boundary beta sweep  
B2. Query-floor sweep

Promote only the best one.

## Stage C — directly bottom-load errors

C1. Tail loss sweep  
C2. Hard-pair rank loss  
C3. Focal correctness baseline

Compare to B-best.

## Stage D — training-dynamics target

D1. Temporal correctness target (`K=20`)  
D2. Hard + temporal hybrid  
D3. Margin soft target

Do these first without changing contrastive weighting.

## Stage E — cartography-aware representation learning

E1. Cartography statistics only + analysis  
E2. Asymmetric query/key ACCon  
E3. Best B variant + cartography keys if justified

## Stage F — depth/temporal mechanisms

F1. Depth disagreement diagnostic  
F2. Append depth disagreement to selector  
F3. Temporal target + multi-depth selector  
F4. Temporal + depth reliability distillation

## Stage G — expensive generalization study

G1. OOF correctness labels  
G2. Augmentation robustness labels  
G3. Compare in-sample vs OOF temporal supervision

---

# 23. Recommended first 12 runs

If compute is limited, prioritize:

```text
1.  scsf
2.  acccon_key_only
3.  acccon_query
4.  acccon_boundary beta=0.05
5.  acccon_boundary beta=0.10
6.  acccon_boundary beta=0.15
7.  acccon_query + tail mu=0.05
8.  acccon_query + tail mu=0.10
9.  temporal_target K=20
10. temporal_target K=20 + acccon_query
11. carto_acccon beta=0.10 gamma=2
12. temporal_target + carto_acccon
```

If #9 clearly beats hard correctness, prioritize dynamics-related work over further ACCon beta tuning.

---

# 24. Diagnostic plots to generate automatically

Every run should generate:

## 24.1 Risk-coverage curve

- x: coverage
- y: selective risk
- include AURC in legend.

## 24.2 Error density by confidence rank

Histogram or line plot:

```text
rank percentile -> fraction wrong
```

Use bins of 1%, 5%, or both.

## 24.3 Cumulative captured errors from bottom

For rejection fraction `r`:

\[
CapturedError(r)
=
\frac{
\#\text{ errors in bottom }r
}{
\#\text{ all errors}
}.
\]

Report reject bottom 5%, 10%, 20% and how many total errors are removed.

## 24.4 Training dynamics map

Scatter:
- x = mean true-class confidence `mu`;
- y = variability `sigma`;
- color = final correctness or selector score.

Also optionally color by forgetting count, acceptance weight, or final error.

## 24.5 Temporal reliability vs final score

Scatter/correlation:

```text
rolling correctness -> final selector score
```

## 24.6 Depth disagreement vs error

Bar plot:

```text
D=0, 0.33, 0.67, 1.0
vs
final error rate
```

---

# 25. Tail-concentration metrics

AURC is primary, but add explicit diagnostics.

## 25.1 Bottom-k error enrichment

\[
E_k
=
P(\text{wrong}\mid\text{bottom }k\%).
\]

Report `k = 1, 5, 10, 20`. Higher is better for rejection.

## 25.2 Top-k purity

\[
P_k
=
1-P(\text{wrong}\mid\text{top }k\%).
\]

Report `k = 1, 5, 10, 20`.

## 25.3 Error capture

\[
C_k
=
\frac{
\#\text{wrong in bottom }k\%
}{
\#\text{wrong total}
}.
\]

This distinguishes “the bottom region is dirty because accuracy is bad” from “the selector successfully moves a large fraction of all errors into the bottom region.”

---

# 26. Generalization safeguards

## 26.1 No test-derived dynamics

Never compute training weights, thresholds, temporal targets, or hyperparameters using official test labels.

## 26.2 Beware memorization of sample IDs

The selector should only consume model-derived features, never sample IDs or stored per-example dynamics at test time.

Training dynamics may be used as targets/weights but not as unavailable inference inputs unless explicitly distilled.

## 26.3 Confidence-head capacity

Because the confidence head receives thousands of dimensions, compare at least one smaller head:

```text
current: 1024 -> 512 -> 256 -> 128 -> 1
small:   512 -> 128 -> 1
```

If train correctness AUROC is near 1.0 but validation/test AUROC drops strongly, memorization may be occurring.

## 26.4 Analyze persistent vs transient mistakes

For train samples define:
- persistent wrong;
- transient wrong;
- stable correct;
- frequently forgotten.

Compare representation statistics and selector scores.

---

# 27. Ablation table structure

Eventually produce a table like:

| Variant | Query weighting | Key weighting | Confidence target | Ranking loss | AURC ↓ | AUROC ↑ | Err@80 ↓ | Err@90 ↓ | Bottom5 error% ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| SCSF | – | – | hard | – | | | | | |
| ACCon-key | uniform | acceptance | hard | – | | | | | |
| ACCon-query | acceptance | acceptance | hard | – | | | | | |
| Boundary | acceptance+boundary | acceptance | hard | – | | | | | |
| Tail | acceptance | acceptance | hard | tail | | | | | |
| Temporal | acceptance | acceptance | temporal | – | | | | | |
| Cartography | ambiguity-aware | stability-aware | hard | – | | | | | |
| Temporal+Carto | ambiguity-aware | stability-aware | temporal | – | | | | | |

This table makes each contribution identifiable.

---

# 28. Novelty guardrails relative to CCL-SC

Do not frame any of these as:

> “We introduce confidence-aware contrastive learning for selective classification.”

That space is already directly occupied by CCL-SC.

The potentially distinctive directions are instead:

1. **coverage-conditioned representation learning** — weights derived from soft acceptance across deployment coverages rather than raw softmax confidence;
2. **core-vs-boundary contrastive allocation** — balancing trusted-core purification and acceptance-boundary refinement;
3. **asymmetric query/key roles** — ambiguous examples may be useful queries but poor prototypes;
4. **training-dynamics supervision distilled into a single-pass selector** — temporal information is used only during training, while inference uses one final model;
5. **temporal + depth reliability** — combine how predictions evolve over training time with how evidence evolves across network depth.

Any publication claim must be based on ablations showing these mechanisms matter.

---

# 29. Relationship to established literature

These papers should guide implementation and interpretation.

## Dataset Cartography
Swayamdipta et al., EMNLP 2020, **“Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.”**

Relevant insight:
- mean true-class confidence and its variability characterize easy, ambiguous, and hard-to-learn examples;
- ambiguous examples can be particularly useful for OOD generalization;
- hard-to-learn examples often include labeling issues.

Use here:
- do not equate “uncertain” with “useless”;
- separate query importance from prototype/key reliability.

## Example Forgetting
Toneva et al., ICLR 2019, **“An Empirical Study of Example Forgetting during Deep Neural Network Learning.”**

Relevant insight:
- correct -> incorrect transitions identify unstable examples;
- forgetting behavior is structured and can be relatively stable across models.

Use here:
- temporal instability target/feature;
- distinguish stable vs repeatedly forgotten samples.

## AUM
Pleiss et al., NeurIPS 2020, **“Identifying Mislabeled Data using the Area Under the Margin Ranking.”**

Relevant insight:
- margin trajectories contain information about persistent difficulty and possible label problems.

Use here:
- average margin and margin variability as reliability/hardness signals;
- avoid treating persistently wrong samples as good class prototypes.

## EL2N / GraNd
Paul et al., NeurIPS 2021, **“Deep Learning on a Data Diet: Finding Important Examples Early in Training.”**

Relevant insight:
- example difficulty/importance can be estimated early;
- some scores transfer across architectures/settings.

Use here:
- early difficulty prior;
- analyze whether early difficulty predicts later selective failures.

## Prediction Depth
Baldock et al., NeurIPS 2021, **“Deep Learning Through the Lens of Example Difficulty.”**

Relevant insight:
- difficult examples tend to require deeper computation;
- prediction depth relates to uncertainty, confidence, accuracy, and learning speed.

Use here:
- cross-depth prediction disagreement;
- test whether internal disagreement predicts failures beyond final logits.

## Selective Classification via Training Dynamics
Rabanser et al., 2022, **“Selective Classification Via Neural Network Training Dynamics.”**

Relevant insight:
- disagreement with the final prediction across historical checkpoints is useful for rejection.

Important distinction for DSSPMI direction:
- their selective score uses test-input behavior across intermediate training checkpoints;
- the proposed DSSPMI direction should test **distilling training-dynamics reliability into a final single-pass selector**, avoiding multi-checkpoint inference.

## CCL-SC
Wu et al., ICML 2024, **“Confidence-aware Contrastive Learning for Selective Classification.”**

Relevant insight:
- feature-level contrastive optimization can improve selective classification.

Novelty caution:
- generic “confidence-weighted contrastive learning for SC” is not new;
- ACCon variants should be justified through coverage conditioning, temporal dynamics, asymmetric roles, or other mechanisms that are genuinely distinct.

---

# 30. Recommended implementation priority

If the agent must choose an order, use:

```text
P0  reproduce current metrics exactly
P1  boundary-aware ACCon
P2  soft tail/risk-coverage loss
P3  dynamics recorder
P4  temporal correctness target
P5  cartography-aware asymmetric query/key weights
P6  depth-disagreement diagnostic
P7  temporal + depth distillation
P8  out-of-fold correctness supervision
P9  EL2N/AUM extras
```

Reason:
- P1/P2 are cheap and directly target the observed rank-shape problem;
- P3/P4 test the strongest generalization hypothesis;
- P5 is the most promising contrastive extension if temporal statistics are predictive;
- P6/P7 align strongly with DSSPMI’s intermediate-feature motivation;
- P8 is principled but expensive;
- P9 are useful supporting signals but should not make the method unnecessarily complex.

---

# 31. Decision rules after experiments

## Keep boundary weighting if
- AURC improves consistently;
- top 10% purity is preserved;
- 75–85% shoulder improves;
- improvement survives >=3 seeds.

## Keep tail loss if
- bottom 5/10% captures more total errors;
- AURC improves;
- accuracy is not materially damaged;
- scores do not collapse to a degenerate distribution.

## Keep temporal target if
- test/validation correctness AUROC improves;
- AURC improves over hard correctness;
- improvement is not merely due to accuracy.

## Keep cartography-aware ACCon if
- it beats both ordinary ACCon and temporal-target-only;
- ambiguous queries help while unstable keys are safely downweighted;
- effect survives seed changes.

## Keep depth disagreement if
- disagreement strongly correlates with errors;
- adding it to the selector improves test AURC beyond already using pool4/pool5/logits.

## Combine modules only if
Each module gives an independently measurable gain or complementary error-rank effect.

Do not combine four weak ideas into one large method simply because the joint number is marginally better.

---

# 32. Final target research story if results support it

A coherent end-state could be:

> **Selective prediction is treated as a representation-learning problem across both network depth and training time. DSSPMI uses intermediate representations to estimate reliability in a single forward pass. Training dynamics provide richer supervision than instantaneous correctness, while coverage-aware contrastive learning allocates representation learning differently to stable class-core examples and ambiguous boundary examples. This encourages trustworthy predictions to form a stable accepted core and pushes likely failures toward the rejection tail.**

This story is only valid if the ablations demonstrate:
1. temporal supervision > instantaneous correctness;
2. multi-depth features > terminal-only features;
3. coverage/dynamics-aware representation learning > generic SupCon;
4. final inference remains single-pass.

---

# 33. Minimal acceptance checklist for the coding agent

Before declaring a variant complete:

- [ ] Reproduces baseline when its new coefficient is zero.
- [ ] Does not use test labels during training or tuning.
- [ ] Uses stable sample IDs for dynamics.
- [ ] Correctly controls detach/gradient routing.
- [ ] Saves config and random seed.
- [ ] Saves `last.pth`.
- [ ] Evaluates on official 10k using unchanged evaluator.
- [ ] Reports accuracy, AURC, AUROC, E-AURC if available.
- [ ] Reports risk by coverage.
- [ ] Reports error-rank bands.
- [ ] Reports top-k purity and bottom-k error enrichment/capture.
- [ ] Produces risk-coverage curve.
- [ ] Produces dynamics diagnostics for temporal variants.
- [ ] Has at least one ablation isolating the new mechanism.
- [ ] Promising result is repeated across >=3 seeds.

---

# 34. First implementation task to hand to the agent

Implement these three items without changing any other training behavior.

### Task 1 — make ACCon query weighting modular

Support:

```text
uniform
acceptance
boundary
floor
```

with a common function:

```python
def make_query_weight(
    w,
    mode,
    beta=0.1,
    floor=0.2,
):
    ...
```

### Task 2 — add tail loss

Implement:

```python
soft_selective_risk_loss(
    scores,
    wrong_mask,
    coverages=(0.70, 0.80, 0.90, 0.95),
    temperature=0.2,
)
```

Thresholds are solved from detached scores; final sigmoid masks use non-detached scores.

### Task 3 — add reusable dynamics tracker

Implement stable per-example tracking for:

```text
correctness
true-class probability
margin
forget count
```

and expose:

```python
tracker.temporal_correctness(ids, window=20)
tracker.mean_confidence(ids, window=20)
tracker.variability(ids, window=20)
tracker.mean_margin(ids, window=20)
tracker.forgetting_count(ids)
```

Do not yet change the training objective with the dynamics tracker. First verify the stored statistics and generate diagnostics.

Once Tasks 1–3 are validated, run the experiment ladder above.
