# Round 2: Three Training-Based Deep-Supervision Proposals

Date: 2026-10-19. Status: implemented as `ro_ds`, `ag_ds`, `cm_ds` (site default `layer3+layer4`; MSP primary with aux margin logged). Not yet experimentally validated. Launch with `bash run_medical_round2.sh next --gpu 0`. DC-DS, the previous fourth candidate, is retired.

## 1. Research decision and evidence

The central thesis question is whether supervision entering an intermediate backbone layer improves medical selective classification. A better auxiliary confidence head alone does not establish that contribution.

The source is the local six-page report `docs/Ket_qua_3_phuong_phap_DS_mg_mp_se.pdf`, reporting 63 runs at experimental commit `52162a1`, and the accompanying analysis `docs/analysis_mg_mp_se_results.md`. The following evidence is reproduced here so this plan remains understandable without those local attachments. All AURCs use the report's ×10^-3 scale.

| Observation | Implication for this round |
|---|---|
| MG-SCSF beats SCSF on 4/7 reported means, but BUSI worsens from 29.26 to 51.69 | Investigate how auxiliary supervision affects the classifier; do not equate a better mean rank with a uniformly useful method |
| MP-DS on COVID: auxiliary score 11.12, same-checkpoint MSP 5.46; risk rises again as coverage falls | Address confidently ranked errors during training without substituting an unreliable prototype score at inference |
| SE-DS on Chest CT: spatial score 18.29, same-checkpoint MSP 15.72 | Auxiliary spatial learning may be useful even when its score is not |
| SE-DS on Masoud: AURC 15.33 ± 12.97, accuracy 95.73 ± 0.13% | Track confidence tails and seed variation, not accuracy alone |
| All six newer methods have BUSI AURC between 50.25 and 72.00 | Audit shared protocol differences before interpreting this as six independent architectural failures |
| No detached DS controls are reported | The contribution of auxiliary backbone gradients remains unmeasured |

The PDF uses a new tie-aware evaluator for the three DS methods, while the seven original methods have not been recomputed. It also describes MP/SE using `layer3 + layer4`, whereas their original proposals specified `layer3` only. Consequently, old/new rankings are provisional, and failures do not establish that any particular loss, pooling choice, or gradient conflict caused the result.

The three new hypotheses are:

| Candidate | Proposed identifier | Question |
|---|---|---|
| Risk-Ordered Deep Supervision (RO-DS) | `ro_ds` | Can intermediate supervision explicitly separate the final classifier's correct and incorrect decisions? |
| Aligned-Gradient Deep Supervision (AG-DS) | `ag_ds` | Does controlling harmful auxiliary gradient directions and magnitudes preserve useful DS while reducing classifier regressions? |
| Complement-Masked Deep Supervision (CM-DS) | `cm_ds` | Can intermediate supervision reduce dependence on a few highly activated locations while retaining the final classifier's confidence score? |

These are new experiment specifications for this repository, not claims that their ingredients are novel. Deep supervision, confidence ranking, gradient projection, and complementary feature erasing all have established prior art. The contributions must be established against those ingredients and matched controls, not inferred from new method names.

## 2. Shared contract: make the DS contribution identifiable

### Training and inference

All three methods use one augmented image, one backbone forward, and exactly one terminal classification loss:

`L_main = mean CE(z(x), y)`.

Auxiliary branches attach to ResNet **layer3 + layer4** in the implementation (pooled concat for RO/AG; spatially upsampled concat for CM), with no attached terminal logits. They receive class-label supervision during the same training run. Aux gradients can update those stages and earlier blocks; they have no direct auxiliary path into the terminal classifier. The original written layer3-only site remains available via `--ds-site layer3`.

**All three remove their auxiliary branches at inference.** The retained ResNet includes its final classification head and returns:

`prediction = argmax softmax(z)`; `confidence = max softmax(z)`.

Accept a prediction when confidence exceeds the chosen threshold. This remains selective classification: the classifier supplies both the diagnosis and an MSP rejection score. Using this fixed score does not introduce a fitted post-hoc method. All proposed changes happen during joint training; there is no later confidence fitting, temperature fitting, reference bank, or validation-label training.

Fix MSP as the primary score before screening. Auxiliary scores can be logged for diagnosis, but must not replace it retrospectively on favorable test datasets. SCSF remains an important comparator with its original designated score, and its MSP can be reported as a secondary comparison.

### Matched settings

Preserve the resolved settings of the medical day/suite launchers:

- ImageNet-pretrained ResNet-50; existing 224×224 preprocessing and single-view augmentation.
- Identical dataset roots, class mapping, split/sample manifests, training sampler, and split seed.
- Training seeds 0/1/2 for confirmation; batch size 64; evaluation batch size 128.
- 100 epochs; backbone SGD with learning rate 0.01, existing momentum and weight decay; ×0.1 decay at epochs 40/70/90. Record actual resolved momentum, weight decay, AMP, and library versions rather than silently substituting defaults.
- Epochs 1-20: terminal CE only; new head parameters receive no optimizer step, including weight decay.
- Epochs 21-25: `alpha(e) = 0.1 * min(1, max(0, (e - 20) / 5))`; retain 0.1 thereafter.
- New heads use the same SGD settings. No auxiliary BatchNorm or dropout. Isolate initialization RNG and any method-specific randomness from data loading and augmentation.
- Final-epoch checkpoints for the primary comparison; validation for method/configuration selection. Preserve existing validation frequency and evaluate every method with the same metric implementation and tie policy.

Additional backward passes for AG-DS are part of its computational cost. Report GPU time, peak memory, and parameter count rather than claiming identical compute from identical epoch counts. Equal loss weights also do not imply equal gradient norms.

All new margins, temperatures, gates, and norm caps below are initial screening settings, not validated optima. Give each candidate the same declared validation-tuning budget; record unsuccessful settings and freeze choices before final comparisons.

### Required controls

| Condition | Auxiliary head learns | Auxiliary gradient reaches backbone | Inference |
|---|---|---|---|
| CE reference | No auxiliary objective | No | Final classifier + MSP |
| Full candidate | Yes | Yes | Final classifier + MSP |
| Detached candidate | Yes, from `layer3.detach()` | No | Final classifier + MSP |
| Ordinary DS control | Yes, ordinary auxiliary CE | Yes | Final classifier + MSP |

Detached must sever **every** auxiliary path into the backbone, including manual gradient injection. Main CE stays attached. Masks, correctness labels, and gating variables use stop-gradient values. The auxiliary branch must not change the main forward, running statistics, RNG stream, or optimizer state through a side effect.

Under this contract, Detached should reproduce the CE reference's backbone trajectory to numerical tolerance. Verify that property on a deterministic miniature run. If it fails materially, fix the control before interpreting DS attribution.

This design removes an inference-head explanation for the gain. It does not guarantee a gain. A near-100% sequential attribution relative to a matching detached CE reference would largely follow from the experimental construction; it is not automatically evidence of novelty or success. A useful thesis contribution still requires improvements over **ordinary DS and strong existing methods**. Do not manipulate a denominator to meet the previous 50-70% aspiration.

## 3. Method 1: Risk-Ordered Deep Supervision (RO-DS)

### Hypothesis

Intermediate class separation alone may not teach the backbone which terminal decisions should receive low confidence. Explicitly order intermediate support for the final classifier's correct versus incorrect predictions, then test whether that training improves the terminal classifier's own MSP ranking.

This is motivated by MP's high-confidence COVID errors. It does not assume that every such error is caused by poor intermediate separation.

### Architecture and loss

Let `h = GAP(layer3)`. Use `LayerNorm(D) -> Linear(D,128) -> ReLU`, then L2-normalize its output to `v`. Learn one normalized class vector `w_c` per class. Normalize in float32 with epsilon `1e-6`. Define bounded similarities `a_c = dot(v, normalize(w_c))` and auxiliary logits `u = 10*a`. No learned logit scale, class bias, prototypes, or terminal-logit input.

From detached terminal logits, construct `k_i = argmax(z_i)` and `c_i = 1[k_i = y_i]`. Let:

`s_i = a_i[k_i] - max_{c != k_i} a_i[c]`.

`P = {i: c_i=1}` and `N = {j: c_j=0}`. Use all correct/incorrect pairs within the minibatch, without a queue or cross-batch history:

`L_rank = mean_{i in P, j in N} tau * softplus((m + s_j - s_i) / tau)`,

with `m=0.2`, `tau=0.1`. If either set is empty, set `L_rank=0` without division by zero. Average over pairs so changes in error count do not multiply the loss magnitude. Scores lie in [-2,2], limiting arbitrary scale inflation.

`L_aux = mean CE(10*a, y) + 0.5 * L_rank`.

`L = L_main + alpha(e) * L_aux`.

Terminal labels `k_i` and correctness `c_i` are supervision signals only; no auxiliary gradient flows through them. The true label still anchors the auxiliary classifier. The ranking term encourages lower support for an incorrect terminal diagnosis, including cases where both branches initially agree on that wrong diagnosis.

### Why the DS path is essential

The auxiliary head is discarded. RO-DS can help the evaluated model only by changing shared features through its training gradients. If only the auxiliary margin improves while final MSP AURC does not, the proposed transfer mechanism has failed.

### Distinction and controls

This differs from MP-DS by using one class vector and a training-only score; from MG-SCSF by ranking final decision outcomes rather than applying a fixed true-class hinge; and from the existing DS-SCSF-v2 design by omitting confidence fusion, per-depth TCP heads, and terminal-score ranking at inference.

Confidence ordering itself is established by [Moon et al., Confidence-Aware Learning, ICML 2020](https://proceedings.mlr.press/v119/moon20a.html). The proposed experiment applies a specified current-decision pair objective at an intermediate branch; it is not a reproduction or a novelty claim for Correctness Ranking Loss.

Required mechanism control: set the ranking coefficient to zero, preserving the head, CE term, schedule, and score. Also run Full versus Detached. For a promising result, compare with the same rank term applied directly at the terminal classifier as a separate training comparator; a terminal-only win would weaken the need for the intermediate placement.

### Risks and falsification

Training mistakes may become scarce, making the new term inactive. Current mistakes can be noisy and class-imbalanced; correcting a mistake also changes its supervision group. The head may learn the ordering without improving shared representations. Increasing a correct example's score can preserve overconfidence on superficially similar future errors.

Log the fraction of batches with valid pairs, error counts by class, rank loss and gradient norms, MSP risk at 10/20/50/80/90% coverage, and top-confidence wrong predictions. Pause this candidate if the pair term is mostly inactive, if its improvement disappears relative to ordinary DS, or if low-coverage risk worsens despite better auxiliary ranking.

## 4. Method 2: Aligned-Gradient Deep Supervision (AG-DS)

### Hypothesis

An auxiliary classification objective may contain both useful and harmful updates for the main classifier. Preserve aligned updates and limit the conflicting component and relative size of the auxiliary update. Test whether this reduces regressions while retaining measurable DS benefit.

The common BUSI regression motivates checking this hypothesis; the PDF does not prove gradient conflict. Protocol mismatch must be ruled out independently.

### Architecture and update rule

Use exactly the same pooled layer3 cosine-classification head as RO-DS, with `L_aux = mean CE(10*a, y)`. There is no ranking term. Compute main and auxiliary gradients separately from the same forward graph.

Let `theta_s` contain all trainable backbone parameters in the stem and layers 1-3, flattened conceptually into one shared parameter group. Exclude layer4, the final classifier, and the auxiliary head. Let:

`g = grad_theta_s L_main`; `b = alpha(e) * grad_theta_s L_aux`.

In float32, if `||g|| < 1e-12`, set the auxiliary update to zero. Otherwise:

`b_projected = b - min(dot(b,g), 0) / ||g||^2 * g`.

`b_safe = min(1, rho * ||g|| / max(||b_projected||, 1e-12)) * b_projected`, with `rho=0.25`.

Assign shared data gradients `g + b_safe`. Assign main-CE gradients only to layer4 and the final classifier. Assign `alpha(e) * grad_phi L_aux` to auxiliary-head parameters. Perform **one** normal SGD optimizer step, with the shared momentum and weight decay settings.

Compute dot products and norms across the whole shared parameter group, not independently per tensor. Treat unused auxiliary gradients as zero. Do not differentiate through the projection. Weight decay is applied once by the optimizer, outside the projected data-gradient calculation. With AMP, unscale both constituent gradients before projection, check finiteness, and handle skipped steps consistently. These details require tests before any full run.

For ordinary SGD without momentum, the projected auxiliary direction has nonnegative dot product with the current main-loss gradient, up to numerical error. This is only a first-order local property. The actual training uses momentum, finite steps, weight decay, and changing batches; neither non-increasing CE nor preserved test accuracy is guaranteed.

### Why this is not simply the previous DP-Head

The current local DP-Head supervises view agreement through a learned scalar confidence head, uses a second flipped backbone pass, and removes the CE-aligned component at the layer3 activation tensor even when that component is helpful. AG-DS uses true-label auxiliary classification, retains positively aligned gradients, operates on shared parameter gradients with a norm cap, and evaluates only final MSP. This distinction is based on the local implementation; confirm the experimental DP revision before claiming an exact historical comparison.

Gradient surgery is established prior art: [Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020](https://arxiv.org/abs/2001.06782). The asymmetric CE-prioritized rule and explicit DS controls here are an experimental specialization, not a new claim for projection itself.

There is also closer auxiliary-learning precedent: [Du et al., Adapting Auxiliary Losses Using Gradient Similarity, 2018](https://arxiv.org/abs/1812.02224) adapts auxiliary influence using gradient similarity. AG-DS must not claim that protecting a main task from harmful auxiliary learning is a new idea. If pursuing an optimizer contribution, include a simple cosine-gated auxiliary-weight control as well.

### Controls and risks

Compare with ordinary DS using the identical auxiliary CE/head/schedule, then Full versus Detached. If promising, isolate projection-only and norm-cap-only variants. A gain fully reproduced by a smaller fixed auxiliary weight would favor a simpler explanation over the proposed projection mechanism.

Log conflict frequency, gradient cosine, auxiliary/main norm ratio before and after the rule, cap activation, and fraction of updates effectively zeroed. Also report extra training time and peak memory. The procedure may suppress useful hard-example learning, follow a noisy main gradient, or reduce to nearly CE-only training. If it avoids regressions merely by eliminating DS and does not outperform CE/ordinary DS, it has not established a useful method.

## 5. Method 3: Complement-Masked Deep Supervision (CM-DS)

### Hypothesis

An intermediate classifier may rely on a few dominant locations that do not generalize. A modest, gated removal of those locations within the auxiliary branch can require evidence elsewhere, while the terminal classifier continues to see its original features. Evaluate whether this training improves final MSP rather than exposing a spatial margin as a new rejection score.

SE's useful Chest CT MSP and poor spatial readout motivate separating spatial training from spatial scoring. Dependence on shortcuts remains a hypothesis, not a result established by the PDF.

### Architecture and loss

Use only the `layer3` feature map `H` with `S=H_height*H_width` locations. At each location, apply the same channel-wise `LayerNorm(D) -> Linear(D,128) -> ReLU`, L2-normalize the vector, and compute cosine similarities against one normalized class vector per class. This shared pointwise head produces `A[c,s]` in [-1,1]. No spatial BatchNorm, attention network, masks supplied by annotations, or layer4 features are used.

Full auxiliary logits are `u_full[c] = 10 * mean_s A[c,s]`.

From **detached** `A[y,:]`, identify the top `K = min(S-1, max(1, ceil(0.1*S)))` locations. Exclude those locations from the complementary pooling for every class, using the same mask:

`u_comp[c] = 10 * mean_{s outside top-K} A[c,s]`.

If `S < 2`, disable complement masking. Use a deterministic spatial-index rule for ties. The mask is discrete and detached; gradients flow through retained evidence values. Do not modify `H` in place or pass its masked version into layer4. Both summaries reuse the same backbone forward.

Activate the complementary objective only when the main and full auxiliary predictions are both correct and both assign the true class probability at least 0.7. Define this detached binary gate as `q_i`:

`ell_aux_i = (CE(u_full_i,y_i) + q_i * CE(u_comp_i,y_i)) / (1 + q_i)`.

`L = L_main + alpha(e) * mean_i ell_aux_i`.

The gate avoids demanding complementary evidence while either classifier is still uncertain about the original image. Per-image normalization keeps the total auxiliary weighting from doubling on gated examples. Gate decisions and top-K indices receive no gradients. No second image classification CE or second backbone/BatchNorm pass is introduced.

### Why the DS path is essential

The full and masked auxiliary outputs are both removed at inference. The only intended mechanism is supervision entering the unmodified intermediate backbone. Detaching `H` before the complete auxiliary branch must restore the CE backbone trajectory even though the auxiliary head continues learning.

Complementary erasing has established precedent in [Zhang et al., Adversarial Complementary Learning, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.html). CM-DS is a gated, shared-head training proposal for this selective-classification experiment, not a claim to invent feature erasing or validated lesion localization.

### Controls and risks

The ordinary DS control uses the identical pointwise head and full average-pooled CE with no masking. Run Full versus Detached. If promising, compare with a random mask of the same size and gate, generated from an isolated auxiliary RNG. This tests whether targeted complement learning matters beyond generic feature regularization.

The most serious risk is medical: a genuine diagnostic finding may occupy exactly the erased region. An easy, correctly classified image does not prove that sufficient diagnostic information remains outside it. The model may learn background correlations instead. The gate may also preferentially train easy or majority classes and be almost inactive on difficult diagnoses.

Record gate rates by class and epoch, the increase in auxiliary CE after masking, MSP risk by class/coverage, and full versus masked evidence on validation examples. Do not interpret an evidence map as segmentation. If the gate seldom activates, or masked training damages focal-disease performance or increases confident errors, stop rather than escalating the erased fraction.

## 6. Comparison and experimental priority

| Candidate | Main intended advantage | Main uncertainty | Decisive mechanism comparison |
|---|---|---|---|
| RO-DS | Directly train error-related separation at an intermediate layer | Will ordering transfer to the final MSP, and are enough training errors available? | Rank term on versus off, same auxiliary classifier |
| AG-DS | Control interference while allowing useful auxiliary supervision | Is there harmful conflict, or does the rule simply turn DS off? | Projected/capped versus ordinary auxiliary gradients |
| CM-DS | Learn from more than the strongest spatial evidence | Does complementary evidence exist, or are lesions erased? | Targeted complement versus full pooling and matched random masking |

My priority is **RO-DS first**, because it targets the observed confident-error problem; **AG-DS second**, because it examines the accuracy/regression concern; **CM-DS third**, because its spatial assumption is more medically fragile. This ordering is a research judgment, not a forecast of performance. Do not bundle all three into one model before identifying which mechanism works.

All three are more conservative at inference than MP/SE: they preserve the same classifier/MSP contract. That is an attribution advantage, not an accuracy or SOTA guarantee. If MSP still deteriorates, the training objective itself must be reconsidered.

## 7. Screening, fairness, and a thesis decision

### Before new training

Recover the reported experimental commit, resolved configurations, and seed-level predictions. Recompute the old/new methods with one tie-aware evaluator and identical test manifests. This is evaluation harmonization, not fitting a post-hoc method. Investigate the common BUSI regression. Preserve original source results and log recomputation versions.

CLI options: `ro_ds`, `ag_ds`, `cm_ds`. Launcher: `run_medical_round2.sh` (`ROUND2_PLAN=full|controls|screen`). Both MSP and aux-margin AURC are written to `summary.csv` / `per_sample.csv`.

### Bounded sequence

1. Implement numerical/gradient tests and a deterministic small-data smoke test before expensive runs. Confirm a single backbone forward, no auxiliary warm-up updates, unchanged terminal predictions when removing the training-only head, and CE-equivalent detached controls.
2. Screen the **three Full candidates on COVID, seed 0**, with the complete fixed schedule. Compare last-checkpoint validation AURC with matched CE and available ordinary DS references. This is three candidate runs, not sufficient causal evidence. Reuse baselines only when configuration and evaluation match.
3. For a promising candidate, run its detached and ordinary-DS controls and confirm with seeds 1 and 2. RO/AG share an ordinary-DS architecture; CM needs its own matching spatial-head control. Add missing reference runs explicitly to the compute budget.
4. Confirm on HAM as a larger multiclass dataset, and inspect BUSI as an early regression check. BUSI is a small, noisy check rather than the sole tuning dataset. Masoud is useful for investigating the previously observed confidence instability.
5. Expand surviving candidates to all seven datasets and all three seeds. Run only the mechanism controls justified by earlier results. Keep configurations selected on validation fixed for final reporting.

Budget by measured run counts and actual runtime. The earlier 4-6-hour estimate per run and the report's aggregate completion interval do not establish current per-run cost; profile before scheduling the full suite. No training is launched by this plan.

### Required outputs

Save final checkpoints and resolved configuration, split hashes and sample identifiers, per-seed MSP predictions, tie-aware AURC, E-AURC, NAURC with its definition, accuracy, AUROC, and risk at predeclared coverages. Add class-level retention/error counts and the method-specific diagnostics above. Report mean rank within one fixed competitor set alongside absolute paired differences and uncertainty; do not compare ranks calculated with different method pools.

Use validation for configuration choices. These datasets' test results have already influenced research, so describe them as part of development history. Training-seed variability does not replace patient-sampling uncertainty; use patient-aware paired analysis where identifiers permit it.

### Promotion criteria

Aim for clear gains over matched CE and original SCSF on preferably five of seven datasets; four can support a carefully qualified thesis if losses elsewhere are modest and the mechanism is convincing. Be competitive with stronger existing training comparators such as SAT and DualAug. Do not count tiny changes or a severe BUSI regression as a success merely because a mean rank improves.

Promote a candidate only if Full repeatedly beats its Detached control, improves beyond the ordinary-DS mechanism control, and avoids a recurring high-confidence error failure. For these training-only proposals, reject a result that benefits only the auxiliary head. A candidate can be a defensible thesis method without establishing global SOTA; the latter requires a broader, current, protocol-matched comparison.

## 8. Prior-art boundaries

The literature check supports the following ingredients, not these candidates' expected performance:

- [Lee et al., Deeply-Supervised Nets, AISTATS 2015](https://proceedings.mlr.press/v38/lee15a.html): supervised companion objectives at hidden layers.
- [Moon et al., Confidence-Aware Learning, ICML 2020](https://proceedings.mlr.press/v119/moon20a.html): training for confidence ordering.
- [Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020](https://arxiv.org/abs/2001.06782): handling conflicting task gradients by projection.
- [Du et al., Adapting Auxiliary Losses Using Gradient Similarity, 2018](https://arxiv.org/abs/1812.02224): adapting auxiliary learning to its alignment with a main objective.
- [Zhang et al., Adversarial Complementary Learning, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.html): complementary evidence discovered through feature erasing.

The proposed research contribution would be an effective, demonstrably causal use of intermediate supervision for medical selective classification under matched controls. A limited literature check and a new combination of familiar ingredients do not establish algorithmic novelty.
