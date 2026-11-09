# SCSF: Current Direction and Proposed Methods

**Project:** Thesis — selective classification through intermediate reliability supervision  
**Prepared:** 2 October 2026, Asia/Bangkok  
**Status:** Research plan; proposed methods have not been implemented or validated in this discussion.

This document records the current direction after the latest ablation and specifies three candidate methods. The goal is to establish a defensible role for an auxiliary head attached to internal layers, while retaining the current classification and agreement objectives initially. Method names below are descriptive working labels, not claims of established novelty.

## 1. Current finding and research direction

The latest user-reported ablation suggests that the improvement comes primarily from the positive-view design rather than from auxiliary supervision. In the current documented `dualaug` implementation, the positive view is the horizontally flipped image, and both views receive classification supervision.

This weakens the causal explanation that deep supervision drives the current gain. It does not establish that intermediate supervision is inherently ineffective. Exact effect sizes, uncertainty, and the identities of the latest ablation variants were not supplied in this turn, so the conclusion remains qualitative.

The next work should pursue two tracks:

| Track | Purpose | Required outcome |
|---|---|---|
| Medical evaluation | Run CCL-SC and other relevant baselines under a controlled medical-image protocol | Establish comparative performance and identify conditions where the method helps |
| Mechanism and design | Separate view augmentation, confidence readout, and auxiliary backbone gradients; then test the candidate methods | Establish whether internal supervision contributes and explain how |

Medical datasets should test a stated hypothesis rather than be selected because they produce a preferred ablation. A positive medical result would support a domain-specific mechanism only if the relevant controls are positive too.

### Working research question

> Can an internal auxiliary head develop reliability information beyond the terminal decision, and can its supervision improve selective prediction beyond the gains from the two-view classification recipe?

### Constraints carried forward

- Keep the existing two-view CE plus agreement-BCE objective for the initial candidate methods.
- Do not add contrastive, reconstruction, ranking, or distillation objectives in the initial implementation.
- Preserve single-view, single-backbone-pass inference.
- Distinguish architectural feature readout from supervision that changes shared backbone representations.
- Aim for a mechanism and evidence suitable for a ranked conference or a strong journal; adding heads alone is not a sufficient novelty claim.
- Do not assume that the thesis title establishes the source of the improvement.

## 2. Current documented method

Let a backbone produce intermediate feature maps H_l(x) and final class logits z(x). Let T(x) be a horizontal flip. The confidence head is a five-layer MLP over the two latest pooled feature maps and detached final logits:

$$
s(x)=h_\phi\big(H_4(x),H_5(x),\operatorname{sg}[z(x)]\big),
\qquad g(x)=\sigma(s(x)).
$$

Here, sg denotes stop-gradient. The notation H_4 and H_5 follows the documented VGG pooling sites; equivalent ResNet sites must be specified explicitly.

The classification loss is

$$
\mathcal L_{\mathrm{CE}}=
\frac12\left[\mathrm{CE}(z(x),y)+\mathrm{CE}(z(Tx),y)\right].
$$

The confidence target is prediction agreement:

$$
A(x)=\mathbf1\left[\arg\max z(x)=\arg\max z(Tx)\right].
$$

The combined objective is

$$
\mathcal L=\mathcal L_{\mathrm{CE}}
+\lambda_t\,\mathrm{BCE}(g(x),A(x)).
$$

The agreement target is computed without gradients. Auxiliary gradients reach the backbone through the feature inputs, but not through the detached logit input. The documented schedule uses CE warm-up followed by cosine decay of the auxiliary weight toward 10^-4.

At inference, the model predicts from z(x) and ranks examples using the confidence head, without running the flipped image. The auxiliary head remains an inference component.

**Evidence boundary:** This description follows the README examined during the preceding discussion. It is not a fresh audit of the training scripts. The latest ablation finding takes precedence over earlier interpretations of the README's results.

## 3. Why the auxiliary supervision may be redundant

| Hypothesis | Explanation | Diagnostic |
|---|---|---|
| Two-view CE explains most of the gain | Classification training already learns useful view invariance | One-view versus two-view CE with no auxiliary backbone gradients |
| Terminal information dominates the head | Logits and late features may suffice to predict agreement | Logits-only, internal-only, and combined heads with matched capacity |
| The agreement target adds little information | Easy examples and large margins may already explain agreement | Held-out agreement-BCE reduction from adding intermediate features |
| The target is misaligned with error detection | Two wrong predictions of the same class receive a positive target | Agreement and confidence statistics conditional on correctness |
| The head absorbs the task | A large MLP can improve its own prediction without making useful backbone changes | Active versus detached features with identical head structure |
| Auxiliary gradients are weak or redundant | The coefficient decays, or auxiliary directions duplicate CE directions | Gradient norms and cosine alignment by block and training phase |
| The chosen depth loses the useful evidence | Two very late sites may be nearly redundant | One internal site at a time, including an earlier site |

These explanations are not yet established facts. Measure them before choosing a complicated intervention.

### Agreement is not correctness

Let C(x) = 1[prediction equals ground truth], and let V be the information available to the confidence head. At the population optimum of ordinary BCE:

$$
g^*(V)=P(A=1\mid V),
$$

whereas selective classification needs a score related to P(C=1 | V). Define

$$
\alpha(V)=P(A=1\mid C=1,V),\qquad
\beta(V)=P(A=1\mid C=0,V).
$$

Then

$$
P(A=1\mid V)=\beta(V)+[\alpha(V)-\beta(V)]P(C=1\mid V).
$$

Constant alpha greater than beta gives a monotone relationship. Having alpha(V) greater than beta(V) separately at each input is not sufficient for global ranking when these quantities vary across inputs.

An architecture-only change can improve agreement prediction without improving correctness ranking. None of the three proposed methods generally eliminates consistent wrong positives while the agreement target is retained.

Changing BCE to Bernoulli KL does not resolve this issue: their difference is target entropy, which is independent of the prediction. The target semantics are the underlying concern.

## 4. Overview of the three proposed methods

| Method | Main intervention | Priority | Main limitation |
|---|---|---|---|
| 1. Residual intermediate reliability head | Learn an internal-feature correction around a frozen logits-only agreement predictor | First prototype | Residual structure does not guarantee complementarity or novelty |
| 2. Decision-preserving auxiliary updates | Restrict auxiliary shared-parameter updates to directions that preserve current logits to first order | Main research candidate after Method 1 | Training cost, approximation, and overlap with existing gradient methods |
| 3. Decision-conditioned spatial reliability head | Replace coarse feature fusion with a class-conditioned spatial evidence readout at an internal layer | Optional medical architecture experiment | Attention and spatial pooling are established components |

Methods 1 and 3 are architectural candidates. Method 2 is a supervision and optimization design, paired with an internal head. Test them separately before combining them.

## 5. Method 1 — Residual intermediate reliability head

### Motivation

The existing concatenated MLP can solve much of its task from terminal information. Make the final-logit baseline explicit, then measure whether an internal feature representation supplies useful corrections.

### Architecture

Fit a small logits-only agreement predictor b_psi on detached logits using training data. Freeze its parameters for the subsequent joint phase. It outputs an agreement logit, not a class logit.

Attach a small residual head at one internal layer:

$$
g(x)=\sigma\left(
\operatorname{sg}[b_\psi(z(x))]
+r_\phi\big(P(H_l(x)),\operatorname{sg}[z(x)]\big)
\right).
$$

P is initially a simple spatial pooling operator, such as adaptive 2-by-2 pooling. Use a small one- or two-hidden-layer residual MLP. Initialize its final output layer to zero so the starting score equals the baseline score.

Use one internal site first, such as ResNet layer3. Treat this as a starting hypothesis; a depth sweep determines whether it is appropriate.

### Training

1. Warm up the backbone with the same two-view CE recipe as the matched baseline.
2. Fit b_psi with the existing agreement BCE on detached logits, using training data only.
3. Freeze b_psi and initialize the residual head.
4. Train the backbone and residual head with the original combined objective.
5. Allow auxiliary backbone gradients through H_l only; terminal logits remain detached in the confidence branch.

The preliminary baseline fit reuses agreement BCE; it introduces a training phase, not a new objective family. Match that extra head-fitting budget in comparison variants.

**Important:** Freezing b_psi does not freeze the overall baseline score when the backbone changes, because z(x) changes. The residual may compensate for baseline drift. Record this drift, and include a frozen-backbone diagnostic to isolate readout information.

### Expected mechanism

The residual branch may exploit internal structure that terminal confidence does not explain. Its small capacity may also make representation adaptation more relevant than with the current large MLP.

Neither effect is guaranteed. Intermediate features can reproduce terminal information, and the residual can simply correct a poorly fitted baseline.

### Required ablations

| Variant | What it tests |
|---|---|
| b_psi only | Agreement information explained by logits |
| Residual head with detached H_l | Additional feature readout without backbone supervision |
| Residual head with active H_l | Incremental contribution of auxiliary representation training |
| Ordinary concatenated head, matched capacity | Residual structure versus head size |
| Residual head on final features | Internal placement versus terminal placement |
| Several individual internal sites | Whether useful information depends on depth |

For fixed representations and unrestricted predictors, the reduction in optimal BCE from adding H_l to Z is

$$
H(A\mid Z)-H(A\mid Z,H_l)=I(A;H_l\mid Z).
$$

Held-out BCE differences with finite fitted heads are operational diagnostics, not exact mutual-information estimates. They concern agreement, not correctness.

### Evidence required to retain the method

- The internal feature branch improves held-out error ranking, not merely agreement accuracy.
- The active variant improves over its detached counterpart across repeated runs.
- The improvement survives matched two-view CE and matched-capacity controls.
- Internal placement has a measurable advantage over terminal placement.

**Novelty status:** A low-cost, interpretable prototype. A residual head alone is unlikely to sustain the strongest publication claim.

## 6. Method 2 — Decision-preserving auxiliary updates

### Motivation

If auxiliary gradients mostly reinforce classification directions, ordinary deep supervision may be redundant. Test whether reliability supervision is useful when its shared-parameter update is constrained to preserve the current terminal decision locally.

The hypothesis is that representations can support additional reliability information without every auxiliary update modifying the same class logits.

### Architecture and update rule

Use an internal confidence head, initially Method 1 or a capacity-matched ordinary internal head. Let theta_l denote a selected shared parameter block upstream of the attachment site.

Define the auxiliary gradient and protected-logit Jacobian:

$$
u=\nabla_{\theta_l}\mathcal L_{\mathrm{BCE}},\qquad
J=\frac{\partial z_{\mathcal B}}{\partial\theta_l}.
$$

z_B stacks final logits for a specified protected set of minibatch inputs. Protect both views when feasible; if only original views or a subset are protected, state that explicitly.

Let

$$
P=I-J^\dagger J,
\qquad d_{\mathrm{aux}}=-Pu,
$$

where the dagger denotes the Moore-Penrose pseudoinverse. P is the orthogonal projector onto the null space of J. Apply the projected auxiliary displacement alongside the normal CE update. Train the confidence-head parameters normally with BCE.

The objective remains the existing CE plus agreement BCE. The shared-parameter auxiliary update is constrained; it is not ordinary gradient descent on the unconstrained combined loss.

### Local theoretical statement

For the isolated projected auxiliary displacement:

$$
Jd_{\mathrm{aux}}=0,
\qquad
u^\top d_{\mathrm{aux}}=-\|Pu\|^2\leq0.
$$

For a sufficiently small step, smooth logits change only at second order along this displacement, while the auxiliary loss decreases to first order unless Pu is zero.

This is a local statement about protected examples and the isolated auxiliary displacement. It is not a generalization guarantee, an AURC guarantee, or a guarantee that the full CE-plus-auxiliary step leaves predictions unchanged.

### Practical implementation scope

- Start with one shared block and a small protected subset.
- Use Jacobian-vector and vector-Jacobian products or a low-rank approximation rather than materializing a full-network Jacobian.
- Protect centered class logits if eliminating the common logit-shift direction is desired; specify this choice because it changes the protected operator.
- A damped approximation using J-transpose times the inverse of JJ-transpose plus epsilon-I is not an exact null-space projector. Report the residual constraint violation.
- Feature-space projection can be cheaper, but its guarantee concerns a feature perturbation. It does not automatically imply that the actual shared-parameter update preserves logits.
- With momentum or Adam, projecting raw gradients does not ensure that the resulting optimizer displacement is projected. Begin with a separately applied projected auxiliary displacement or explicitly project the actual auxiliary update.
- Avoid rescaling an almost-zero projected gradient to a large norm.
- Jacobian and projection computations are training-only. Inference retains one backbone pass and the confidence head.

### Required measurements and ablations

| Measurement or control | Purpose |
|---|---|
| Surviving norm ||Pu|| / ||u|| | Determine whether usable auxiliary directions remain |
| Logit change from an isolated auxiliary step | Verify the intended local constraint |
| Unprojected auxiliary update | Establish whether projection matters |
| Detached confidence head | Establish whether shared representation training matters |
| Matched-magnitude unprojected update | Separate direction from gradient strength |
| CE-conflict projection such as PCGrad | Compare against a generic gradient intervention |
| Internal versus terminal attachment | Establish the role of intermediate supervision |
| Full versus approximate projection | Quantify the approximation and compute tradeoff |

### Failure conditions

The method may fail because useful reliability information requires changing class-sensitive directions, because the agreement proxy is unhelpful, because almost no auxiliary gradient survives, or because minibatch preservation does not transfer to unseen examples.

### Novelty boundary

Gradient surgery, auxiliary update decomposition, and null-space projection are existing ideas [7–9]. The potential contribution is the selective-classification-specific constraint, its relation to internal reliability supervision, and evidence explaining when it helps. A targeted prior-art search remains necessary before claiming novelty of the exact formulation.

**Priority:** This is the main research candidate if Method 1 and the diagnostics establish useful intermediate information and a plausible gradient-redundancy problem.

## 7. Method 3 — Decision-conditioned spatial reliability head

### Motivation

In medical classification, coarse pooled features may omit local morphology or evidence heterogeneity. Test whether a spatial internal head can evaluate support for the final decision while retaining information about inconsistent local evidence.

This is a domain hypothesis, not an established medical explanation for the current improvement.

### Architecture

Take spatial tokens h_(l,p) from one internal feature map. Build a query from detached terminal probabilities:

$$
q(x)=Q\big(\operatorname{sg}[\operatorname{softmax}(z(x))]\big).
$$

Compute decision-conditioned spatial weights and two summary statistics:

$$
w_p=\operatorname{softmax}_p\left(
\frac{q(x)^\top K(h_{l,p})}{\sqrt d}
\right),
$$

$$
m(x)=\sum_p w_p V(h_{l,p}),
$$

$$
v(x)=\sum_p w_p\left(V(h_{l,p})-m(x)\right)^{\odot2}.
$$

The confidence head is

$$
g(x)=\sigma\left(h_\phi\left[m(x),v(x),\operatorname{sg}[z(x)]\right]\right).
$$

The vector v is a feature-dispersion statistic, not a calibrated uncertainty estimate. It can reflect anatomy, acquisition conditions, or other variation unrelated to correctness.

Use the same CE plus agreement BCE. Gradients enter the internal feature map through the key/value and pooling paths; logits remain detached. Detaching the logits does not freeze Q's parameters.

### Expected mechanism

The weighted mean captures decision-conditioned evidence; dispersion captures heterogeneity among the attended features. Their combination may be more informative for reliability than late feature concatenation alone.

Attention can still focus on shortcuts or ignore ambiguous regions. It is not inherently lesion-localizing and does not establish clinical explanation validity.

### Required ablations

- Global average pooling versus a small spatial grid versus learned attention.
- Mean only versus mean plus dispersion.
- Decision-conditioned query versus a learned constant query.
- Active versus detached internal features.
- Internal versus terminal feature maps, where comparable spatial maps exist.
- Matched parameter count and comparable inference cost.

Start with ordinary 2-by-2 pooling. Add attention only if spatial diagnostics suggest that coarse pooling is the bottleneck.

**Novelty status:** A useful medical architecture experiment, but attention, class conditioning, and feature moments are established tools. Strong novelty would require a demonstrated reliability-specific mechanism beyond assembling these components.

## 8. Core attribution experiment

Run the following factorial design before a broad architecture search:

| Classification recipe | No auxiliary backbone gradients | Active auxiliary backbone gradients |
|---|---|---|
| One-view CE | A | B |
| Two-view CE | C | D |

Add a trained detached-head control to both classification recipes. Computing a second view for an agreement target is distinct from applying CE to that second view; record both computational choices.

Interpret the comparisons as follows:

- C versus A measures the view-recipe effect under the same score procedure.
- Active versus detached versions measures the incremental effect of auxiliary representation training.
- Native head versus standard scores on the same backbone measures the confidence-readout effect.
- The difference between supervision effects under one-view and two-view CE measures interaction.

If supervision helps only with two-view training, attribute the gain to their interaction. Do not require the contribution to be exclusively supervision: require evidence that supervision contributes beyond the matched view baseline.

For every trained backbone, evaluate native confidence scores where applicable, standard terminal scores, and an identical frozen-backbone probe-fitting procedure. Fit probes separately using the same protocol; do not assume that one head's weights transfer across different representations.

Control view generation, initialization, random-number use where feasible, batch-normalization behavior, optimizer settings, and checkpoint rules. A detached head should not accidentally alter the classifier's training recipe through different forward passes or random augmentation draws.

## 9. Medical dataset and baseline protocol

### Dataset principles

- Use patient-disjoint partitions when patients contribute multiple images or slices.
- Include at least two modalities with different visual demands.
- Include an external institution or acquisition shift when available.
- Verify task validity of horizontal flips and other transformations; do not assume every medical label is flip-invariant.
- Use a common augmentation recipe across methods in the controlled comparison.
- Move from one CNN prototype to a CNN and a transformer once the mechanism is promising.
- Select datasets and operating points before inspecting their test results.

### Baseline priorities

| Group | Methods | Purpose |
|---|---|---|
| Standard scores | MSP/SR, logit margin, validation-tuned pNorm [5] | Test whether the head beats inexpensive score choices |
| Learned confidence | ConfidNet [10], matched detached internal head | Separate learned readout from backbone supervision |
| Selective training | CCL-SC [1], SAT+EM, Deep Gamblers, SelectiveNet | Establish comparative selective performance |
| Shift-focused methods | Delta-MDS and Delta-KNN [6] | Include a recent comparison under acquisition or domain shifts |
| Deep supervision comparator | Standard auxiliary CE; relevant DTS implementation [3] if feasible | Compare reliability supervision with established internal classification supervision |

Run CCL-SC under a reproduced native recipe and a matched two-view classification recipe. A modified-recipe run complements rather than replaces the native reproduction. Give all methods reasonable, comparable validation tuning budgets.

The previously examined README lists a CIFAR batch size different from the CCL-SC paper's documented setup. Earlier published-number comparisons are encouraging but are not sufficient evidence of a fully matched reproduction.

## 10. Metrics and mechanism analysis

Report classification accuracy, macro-F1 for imbalanced tasks, AURC, NAURC [5], failure-detection AUROC, and risk at fixed coverages. Consider AUGRC [11] as an additional aggregate evaluation.

Specify the positive label for failure-detection AUROC and the direction of the confidence score. Predefine tie handling and metric scaling. NAURC is undefined when the ideal and random references coincide, including a zero-error classifier; report this rather than dividing by zero.

Accuracy and ranking must be assessed together. Fewer overall errors can improve AURC, but higher accuracy alone does not guarantee lower risk at every coverage. NAURC and AUROC do not replace the active-versus-detached experiment.

Inspect these four groups:

| Final prediction | View agreement | Main question |
|---|---|---|
| Correct | Agree | Does the head retain reliable examples? |
| Correct | Disagree | Does the head reject correct but view-sensitive examples? |
| Incorrect | Agree | Does it remain confident on consistent errors? |
| Incorrect | Disagree | Does it identify unstable errors? |

For each group, record sample count, score distribution, acceptance rate at prespecified operating points, and relevant class or acquisition breakdowns. For clinical interpretation, also report accepted errors and coverage by class; low average risk can hide disproportionate referral of difficult classes.

Use repeated seeds and patient-level bootstrap uncertainty where appropriate. Repeated seeds and patient bootstraps address different sources of variation; report them distinctly. Fix validation-based checkpoint and threshold rules before test evaluation.

## 11. Execution order and decision criteria

1. **Consolidate the latest ablation:** Record variant definitions, mean effects, uncertainty, accuracy, and ranking metrics. Audit whether active and detached models truly share the same view recipe.
2. **Measure the current mechanism:** Compare logits-only and internal readouts; measure agreement among correct and incorrect examples and auxiliary gradient strength by block.
3. **Run matched medical baselines:** Start with the two-view CE control, CCL-SC, inexpensive confidence scores, and a detached confidence head.
4. **Prototype Method 1:** Use one internal site and a small residual head. Expand seeds only after the initial mechanism comparison is informative.
5. **Choose the next intervention from evidence:** Test Method 2 if gradient redundancy or interference is supported; test Method 3 if spatial pooling is the plausible bottleneck.
6. **Confirm the winning mechanism:** Validate across another modality and backbone, with internal-versus-terminal and active-versus-detached controls.
7. **Complete novelty review:** Compare the final formulation against the closest selective-classification, confidence-learning, deep-supervision, and gradient-method papers.

| Outcome | Defensible direction |
|---|---|
| Active supervision consistently beats detached controls | Develop the internal reliability-supervision contribution |
| Detached internal features help, active gradients do not | Frame the contribution around intermediate confidence readout |
| Two-view CE explains the improvement | Frame the contribution around view design and its selective-prediction effects |
| Benefits occur only in specific medical settings | State the domain conditions and support them with mechanism analysis |
| Improvements are unstable or disappear against tuned simple scores | Simplify the design or revise the research direction |

A credible eventual claim would be: internal reliability supervision is effective under identifiable conditions, and a specific architectural or update design makes that contribution measurable. Do not claim a generalization bound from superior benchmark results alone, or infer auxiliary causality from the presence of gradients.

## 12. References and prior-art boundaries

The references below were consulted in the preceding research discussion. This document does not represent an exhaustive novelty review. Each paper serves a specific comparison or conceptual purpose; not every one is a required runnable baseline.

1. **Confidence-aware Contrastive Learning for Selective Classification.** ICML 2024. Correctness-aware feature-level selective learning; official baseline referred to here as CCL-SC. [Paper](https://proceedings.mlr.press/v235/wu24s.html) · [Code](https://github.com/lamda-bbo/CCL-SC)
2. **Contrastive Deep Supervision.** ECCV 2022. Intermediate supervision using augmentation-based contrastive learning. [Paper](https://arxiv.org/abs/2207.05306)
3. **Deep Trajectory Supervision: Deep Supervision Strikes Back.** ICML 2026. Auxiliary supervision aligned with semantic-evidence progression across depth. [Paper](https://proceedings.mlr.press/v306/wang26ib.html)
4. **Post-hoc Selective Classification for Reliable Synthetic Image Detection.** 2026 preprint; ReSIDe. Intermediate-layer confidence extraction and aggregation in synthetic-image detection. [Paper](https://arxiv.org/abs/2605.08574)
5. **How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks.** UAI 2024. Logit normalization, pNorm, and NAURC. [Paper](https://proceedings.mlr.press/v244/cattelan24a.html)
6. **Know When to Abstain: Optimal Selective Classification with Likelihood Ratios.** ICLR 2026. Likelihood-ratio-based selectors, including Delta-MDS and Delta-KNN, evaluated under shifts. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/3fe2a777282299ecb4f9e7ebb531f0ab-Paper-Conference.pdf) · [Code](https://github.com/clear-nus/sc-likelihood-ratios)
7. **Gradient Surgery for Multi-Task Learning.** PCGrad. Prior art for projecting conflicting task gradients. [Paper](https://arxiv.org/abs/2001.06782)
8. **Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral.** Prior art for decomposing auxiliary updates using the primary task's gradient geometry. [Paper](https://arxiv.org/abs/2108.11346)
9. **GNSP: Gradient Null Space Projection for Preserving Cross-Modal Alignment in VLMs Continual Learning.** 2025 preprint. Prior art for null-space constraints in a different application. [Paper](https://arxiv.org/abs/2507.19839)
10. **Addressing Failure Prediction by Learning Model Confidence.** NeurIPS 2019; ConfidNet. Learned confidence prediction using true-class probability as a target. [Paper](https://papers.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf) · [Code](https://github.com/valeoai/ConfidNet)
11. **Overcoming Common Flaws in the Evaluation of Selective Classification Systems.** NeurIPS 2024. AUGRC and evaluation requirements. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf)

## 13. Items to fill in after the next experiments

- Exact latest ablation variants, numerical results, and uncertainty.
- Current code confirmation of feature detachment, target computation, and gradient paths.
- Medical dataset names, patient identifiers, partition definitions, and transformation validity.
- Selected internal attachment sites and matched head capacities.
- Baseline reproduction settings and validation search budgets.
- Whether agreement provides additional information about correctness in each dataset.
- Which proposed mechanism is supported, rejected, or still inconclusive.

