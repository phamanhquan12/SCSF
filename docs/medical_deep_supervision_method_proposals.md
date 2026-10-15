# Deep Supervision for Medical Selective Classification: Method Proposals

Date: 2026-10-15  
Status: implemented on 2026-10-15 as `verify_global`, `verify_spatial` and `dualaug_correctness`, with the controls of Section 6. Not yet experimentally validated. Run with `DAY_PLAN=proposals bash run_medical_day.sh next`.

## 1. Objective and Constraints

Develop a jointly trained selective classifier in which intermediate supervision contributes to performance on medical images. The classifier predicts a diagnosis and a confidence score; low-confidence predictions can be rejected.

Requirements:

- Train classification and intermediate auxiliary objectives together.
- Allow auxiliary gradients to update shared backbone layers.
- Preserve single-image, single-backbone-pass inference.
- Avoid separate post-hoc fitting, external teachers, and expensive gradient projection.
- Demonstrate the contribution of deep supervision with a matched detached-feature ablation.

The two verification proposals below are related, inexpensive hypotheses. The third experiment is a target diagnostic, not a standalone novelty claim. No proposal is currently established as state of the art.

## 2. Shared Classification Recipe

Use the current pretrained ResNet-50 backbone, input preprocessing, optimizer, learning-rate schedule, training seeds, data splits, and last-checkpoint evaluation policy. Preserve dualaug's classification recipe:

\[
L_{\mathrm{cls}}=\tfrac12\left[
\mathrm{CE}(z(x),y)+\mathrm{CE}(z(Tx),y)
\right],
\]

where \(T\) is the existing horizontal flip. Reuse these two training forwards for auxiliary features; do not add another backbone pass. Keep the final prediction unchanged:

\[
\hat y(x)=\arg\max_c z_c(x).
\]

Each method replaces the confidence branch or its target as specified below. Do not initially combine all proposals into one model.

## 3. Proposal A: Class-Conditioned Intermediate Verification

### Motivation

Agreement supervision can label two matching wrong predictions as reliable. A scalar correctness target can also provide few negative examples once training accuracy becomes high.

Instead, supervise whether intermediate image features support a particular candidate diagnosis. Ground-truth labels supply both positive and negative candidate pairs, regardless of the final classifier's current training accuracy.

### Architecture

Attach one verifier to ResNet `layer3`. Let

\[
h(x)=\mathrm{GAP}(H_3(x)),\qquad
u(x,c)=v_\phi([h(x),e_c]),
\]

where \(e_c\) is a learned class embedding and \(v_\phi\) is a small shared MLP producing one raw verification logit.

Initial architecture: embedding dimension 16; MLP with one hidden layer of width 128 and a ReLU. Do not feed final logits to this first prototype, so the verifier must obtain its evidence from intermediate features. These dimensions are starting settings, not validated optima.

### Supervision and Joint Loss

For each training image, the true class is a positive candidate and every other class is a negative candidate:

\[
L_{\mathrm{ver}}(x,y)=
\tfrac12\operatorname{softplus}[-u(x,y)]
+\frac{1}{2(C-1)}\sum_{c\ne y}
\operatorname{softplus}[u(x,c)].
\]

This balances the positive candidate against the aggregate negative candidates. Evaluate all classes initially; avoid hard-negative sampling and another tuning parameter.

Apply the same objective to both training views:

\[
L=L_{\mathrm{cls}}+
\frac{\alpha_t}{2}
\left[L_{\mathrm{ver}}(x,y)+L_{\mathrm{ver}}(Tx,y)\right].
\]

Proposed initial auxiliary schedule: \(\alpha_t=0\) during the existing 20-epoch warm-up, then \(\alpha_t=0.1\) for the remaining epochs. This fixed verification coefficient is a deliberate starting choice; it differs from dualaug's cosine-decayed agreement coefficient. Keep it identical in the active and detached controls.

### Inference

Run the backbone once and return the existing final prediction with

\[
s(x)=u(x,\hat y(x)).
\]

Use the raw logit for ranking. Do not take the verifier's own maximum across classes: confidence must assess the diagnosis actually returned by the final classifier. Ground-truth labels are used only during training and evaluation, never to construct an inference score.

### Deep-Supervision Mechanism and Risks

The verification loss updates the verifier and the backbone through `layer3` and its upstream blocks. It does not directly supervise `layer4` or the final classification head. Classification loss continues to train the complete backbone.

The hypothesis is that intermediate evidence can challenge a wrong terminal diagnosis. This is closely related to auxiliary classification and may simply learn a redundant classifier. Its value and novelty require evidence beyond adding another head.

Required controls:

- The same verifier receiving `h.detach()`.
- The same candidate-score architecture trained with ordinary auxiliary softmax CE, scored by its probability for the final predicted class.
- Main-classifier softmax response from the same trained checkpoint.

## 4. Proposal B: Spatial Class-Conditioned Verification

### Motivation

Global pooling may dilute localized diagnostic evidence. Test whether spatial class supervision makes intermediate features more useful for checking the final diagnosis.

This differs from the existing `SpatialHeadMethod`, which learns attention and spatial statistics to predict scalar view agreement. The new branch receives diagnosis-specific supervision from ground-truth labels.

### Architecture and Score

Apply a lightweight `1x1` convolution to the `layer3` feature map:

\[
A(x)=\operatorname{Conv}_{1\times1}(H_3(x))
\in\mathbb R^{C\times H\times W}.
\]

For candidate class \(c\), aggregate its strongest spatial responses:

\[
u(x,c)=\frac1k\sum_{p\in\operatorname{TopK}(A_c(x))}A_c(x)_p,
\qquad k=\max(1,\lceil0.2HW\rceil).
\]

Use Proposal A's verification loss, two-view classification recipe, and auxiliary schedule. At inference, retain the final classifier's prediction and use \(u(x,\hat y(x))\) as confidence. No annotation masks or localization labels are required.

### Deep Supervision, Controls, and Risks

Gradients from diagnosis-specific evidence scores update the spatial feature map and upstream backbone layers. The selected locations are differentiable with respect to their selected values; the selection indices are discrete.

Required controls:

- The identical spatial branch receiving `H3.detach()`.
- The identical convolution using global average pooling instead of top-k pooling, isolating spatial aggregation without changing parameter count.
- Same-checkpoint final-classifier softmax response.

The main risks are reliance on artifacts, insufficient evidence for diffuse diagnoses, and correlated errors between the verifier and final classifier. Activation maps are diagnostic visualizations, not proof of lesion localization. Existing work on image-label-supervised medical localization is relevant prior art, not evidence that this proposal improves selective classification.

## 5. Diagnostic C: Correctness-Aware Agreement Supervision

### Purpose and Target

Keep dualaug's backbone, confidence head, and two-view CE. Change only the target for the original-view confidence head:

\[
a_0=\mathbf1[\arg\max z(x)=y],\qquad
a_1=\mathbf1[\arg\max z(Tx)=y],
\]

\[
t(x)=a_0\left[0.5+0.5a_1\right].
\]

Compute these targets without gradients.

| Original prediction | Flipped prediction | Target |
|---|---|---:|
| Correct | Correct | 1.0 |
| Correct | Wrong | 0.5 |
| Wrong | Either | 0.0 |

Train the original-view confidence logit \(s(x)\) with

\[
L=L_{\mathrm{cls}}+
\lambda_t\mathrm{BCEWithLogits}(s(x),t(x)),
\]

using dualaug's existing warm-up and cosine coefficient schedule. Preserve its feature-gradient path and detached final-logit input. Inference remains unchanged.

### Interpretation and Controls

This diagnostic removes consistent-wrong positive targets while distinguishing stable from fragile correct predictions. It introduces no extra backbone forwards or confidence-head parameters.

Compare original agreement, plain original-view correctness, and the proposed target with identical architecture and classification training. For each relevant target, compare active versus detached intermediate features.

Improved robustness does not guarantee improved clean-image error ranking. This target deliberately downweights some correct predictions and could worsen AURC. Retain it only if validation evidence supports it; do not present a target substitution alone as a strong novelty claim.

## 6. Establishing the Contribution of Deep Supervision

The architecture guarantees an auxiliary gradient path, not a performance benefit. For every retained candidate, compare:

| Variant | Auxiliary head trained? | Auxiliary gradient enters backbone? |
|---|---|---|
| Active | Yes | Yes |
| Detached | Yes | No |

Keep head architecture, inputs, classification loss, optimizer settings, auxiliary coefficient, training budget, and inference scoring identical. Detach only the features entering the auxiliary branch. Do not detach the classification path or freeze the backbone globally.

For a clear claim, the active variant should improve validation and confirmed held-out selective performance over its detached counterpart across repeated seeds. If only the detached variant helps, the result supports intermediate-feature readout rather than a benefit from auxiliary backbone supervision.

Record gradient norms from the auxiliary objective at upstream backbone blocks and confirm that these vanish in the detached control. This verifies implementation; the performance ablation establishes contribution.

## 7. Evaluation and Implementation Requirements

- Preserve the same data splits and final-checkpoint policy across compared variants.
- Select configurations using validation data; reserve the test split for final comparisons.
- Report accuracy, AURC, corrected NAURC, risk at fixed coverages, and correctness-detection AUROC. Record undefined metrics explicitly.
- Report confidence-head and final-classifier scoring on the same checkpoint to separate classifier changes from confidence-readout changes.
- Correct relevant dataset-discovery, confidence-tie, and completion-marker defects before relying on final comparisons; see the [implementation review](medical_day_implementation_review.md).
- Check classification and auxiliary gradients separately; ensure inference requires no labels, flipped images, optimizer state, or additional fitted model.
- Use raw logits where specified to avoid sigmoid saturation ties.
- Save the resolved architecture, auxiliary schedule, pooling rule, feature-detachment setting, and complete run configuration.

The methods are available as `verify_global`, `verify_spatial` and `dualaug_correctness`; the default daily suite keeps its established baselines. The controls map to the following options:

| Control | `run_experiment.py` | `run_paper_medical_suite.py` (variant directory) |
|---|---|---|
| Detached features (all three) | `--aux-detach` | `--aux-feature-modes active detached` (`*__active`, `*__detached`) |
| A: auxiliary softmax-CE verifier | `--verify-loss softmax` | `--verify-losses binary softmax` |
| B: GAP instead of top-k | `--verify-pool gap` | `--verify-pools topk gap` |
| C: original agreement / plain correctness | `--method dualaug --dualaug-target {agreement,correctness}` | `--methods dualaug --dualaug-targets agreement correctness` |
| Same-checkpoint softmax response | automatic | `msp_aurc`, `msp_naurc`, `msp_auroc` in `summary.csv` |
| Auxiliary gradient norms at layer2/3/4 | automatic, once per epoch after warm-up | `checkpoints/.../aux_grad_norms.csv` |

The verifier heads (and the dualaug calibrator) use Adam with `--meta-lr`, as the existing confidence heads do; the backbone and classifier keep SGD.

## 8. Research Positioning

The intended contribution is learning intermediate diagnostic evidence through deep supervision to improve medical selective classification. Success requires both competitive selective performance and evidence that active intermediate supervision contributes beyond a matched readout.

These proposals reuse established ingredients, including auxiliary classification, class-conditioned prediction, and spatial pooling. Their exact novelty has not been established. A state-of-the-art claim requires matched medical protocols and strong relevant training-based comparators, not merely improvement over one existing implementation.

Relevant primary references:

- [Confidence-Aware Learning for Deep Neural Networks, ICML 2020](https://arxiv.org/abs/2007.01458): training confidence ordering is established prior work.
- [Deep Interpretable Classification and Weakly-Supervised Segmentation of Histology Images via Max-Min Uncertainty](https://arxiv.org/abs/2011.07221): related image-label-supervised spatial evidence learning.
- [Towards Better Selective Classification, ICLR 2023](https://arxiv.org/abs/2206.09034): relevant training regularization and classifier-based selection baselines.
- [Confidence-aware Contrastive Learning for Selective Classification, ICML 2024](https://arxiv.org/abs/2406.04745): an established training-based selective-classification comparator.
