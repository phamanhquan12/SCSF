# Three Deep-Supervision Methods for Medical Selective Classification

Revised: 2026-10-15. Status: implemented as `mg_scsf`, `mp_ds`, `se_ds` (see Section 10); not yet validated on medical data.

## 1. Research Decision

The required core is useful supervision of intermediate backbone features during classification training. Reusing SCSF is allowed; neither its architecture nor its confidence target is mandatory. This plan replaces the previous requirement to preserve DualAug's two-view classification loss.

| Candidate | Hypothesis | Relationship to SCSF | Inference confidence |
|---|---|---|---|
| **MG-SCSF: Margin-Gated Deep Supervision** | Intermediate class separation needs a stopping condition for easy examples | Keeps its TCP confidence head; adds a training branch | Original SCSF raw confidence logit |
| **MP-DS: Multi-Prototype Deep Supervision** | Several class modes yield better intermediate representations and evidence | Replaces confidence architecture and target | Prototype support margin for the final prediction |
| **SE-DS: Spatial Evidence Deep Supervision** | Spatial class supervision preserves evidence diluted by global pooling | Replaces confidence architecture and target | Spatial evidence margin for the final prediction |

These are new experiment specifications for this repository, not established novel algorithms or promised SOTA methods. Their ingredients have substantial prior art. The research claim must emerge from medical selective-classification results and a demonstrated DS mechanism. Test them separately before considering combinations.

## 2. Training-Based Versus Post-Hoc

Here, a training-based method changes classifier learning through an auxiliary objective and learns selector parameters within the same training run. Post-hoc methods operate on an already-trained predictor, for example by subsequently fitting temperature scaling, a confidence model, or a feature reference bank. Some post-hoc scores require no fitting; the distinction is not simply whether parameters are fitted.

All three candidates jointly train intermediate branches and backbone after the shared warm-up. They require no separate teacher, calibration stage, test-time adaptation, or nearest-neighbor database. Validation-based configuration selection remains allowed; validation labels must not train model parameters.

A selective classifier needs a diagnosis and a selection score. **MG-SCSF retains the original SCSF confidence head at inference. MP-DS and SE-DS retain their new intermediate heads.** Every method retains the final classification head. Accept predictions above a confidence threshold; choosing a deployment threshold is separate from learning the model.

Temperature scaling is an established post-processing example [1]. Research distinguishing classifier improvements from selector improvements motivates evaluating both here [2]. Calibration or OOD-detection results alone do not establish good selective classification [3].

## 3. Shared Training Contract

Preserve the medical launcher's resolved pretrained ResNet-50, input size 224, preprocessing, split manifests, batch size 64, 100 epochs, 20-epoch warm-up, and backbone SGD settings. Keep ordinary single-view augmentation unchanged. Each minibatch has **one backbone forward and one final-classification CE**:

\[
L_{\mathrm{cls}}=\mathrm{CE}(z(x),y),\qquad
\hat y=\arg\max_c z_c(x).
\]

There is no additional `CE(z(Tx), y)`. Intermediate label losses below are the proposed deep supervision, rather than extra classification training on a second image.

Let \(H_3\) be ResNet `layer3`, \(h=\mathrm{GAP}(H_3)\), and \(C\ge2\) the class count. New auxiliary losses update `layer3` and upstream blocks. They have no direct gradient through `layer4` or the final classifier, which still receive final CE. Method 1's retained SCSF branch has its existing additional gradient paths.

Starting coefficient for each new objective, with epochs numbered from one:

\[
\alpha_e=0.1\min\left(1,\max\left(0,\frac{e-20}{5}\right)\right).
\]

New auxiliary learning starts at epoch 21 and reaches weight 0.1 at epoch 25. Leave new auxiliary parameters unstepped during warm-up, including weight decay. These are screening settings, not validated optima or evidence of matched gradient strength across losses. Log actual gradient norms. Method 1 preserves SCSF's separate TCP-loss schedule.

Use backbone SGD settings for new auxiliary parameters; Method 1 keeps Adam for the original calibrator. New heads use no BatchNorm or dropout. Normalize vectors in float32 with epsilon \(10^{-6}\) and compute losses in float32. Isolate auxiliary initialization RNG so controls preserve backbone initialization, data ordering, and augmentation randomness.

## 4. Method 1: MG-SCSF

### Hypothesis and objective

SCSF estimates the terminal model's TCP without explicitly requiring intermediate features to separate diagnoses. Add a class-supervised branch whose per-example pressure stops after sufficient separation.

Normalize pooled intermediate features and one trainable vector per class:

\[
v=\frac{h}{\max(\|h\|_2,\epsilon)},\quad
\bar w_c=\frac{w_c}{\max(\|w_c\|_2,\epsilon)},\quad
u_c=v^\top\bar w_c.
\]

Use independent weight initialization, no bias, and no trainable scale. Define:

\[
d=u_y-\max_{c\ne y}u_c,\qquad
L_{\mathrm{MG}}=[\max(0,0.2-d)]^2.
\]

Average over the entire minibatch, not just active examples. A sufficiently separated example has zero auxiliary loss; the branch does not keep increasing its margin indefinitely. This does not guarantee calibrated or correctly ranked final predictions.

The full objective is:

\[
L=L_{\mathrm{cls}}+
\lambda_e\mathrm{BCEWithLogits}(s_{\mathrm{SCSF}},\operatorname{sg}[p_y])
+\alpha_eL_{\mathrm{MG}}.
\]

Preserve SCSF's TCP target, head architecture, feature inputs, and detached final-logit input. At inference, keep the final classifier and original SCSF confidence head. Only the new margin branch is training-only.

### Difference from previous attempts

Unlike `ds_scsf`, this adds one normalized, margin-limited branch without replacing SCSF by fused per-depth calibrators. Unlike `residual_head` and `dp_head`, it uses neither agreement supervision nor gradient projection. Auxiliary discriminative objectives are established DS ingredients [4]; a hinge loss itself is not a novelty claim.

**Risks:** the margin is reached too early, difficult classes remain permanently active, or mislabeled examples dominate. TCP and margin gradients may conflict. Log active-loss fraction by class, margins, and separate CE/TCP/margin gradient norms.

**Mechanism control:** replace the hinge with `CE(10 * u, y)` using the identical branch and coefficient schedule. This tests the stopping condition against ordinary auxiliary classification.

## 5. Method 2: MP-DS

### Hypothesis and objective

A diagnosis can have several appearances. Test whether intermediate supervision preserves multiple modes rather than representing a class with one direction. This does not assume that every dataset contains exactly three biological subtypes.

Project \(h\) to 128 dimensions with a bias-free linear layer, then normalize:

\[
v=\operatorname{normalize}(Wh).
\]

Learn \(K=3\) independently initialized, normalized prototypes per class:

\[
a_c=\max_{1\le k\le K}v^\top\bar p_{c,k},\qquad
L_{\mathrm{MP}}=\mathrm{CE}(a/0.1,y),
\]

\[
L=L_{\mathrm{cls}}+\alpha_eL_{\mathrm{MP}}.
\]

Prototypes and projection are jointly optimized parameters. Do not fit them with post-training clustering, use a memory bank, or force equal occupancy. The maximum updates selected centers; record unused centers rather than silently restarting them.

### Selective inference

Retain the prototype branch. The final classifier supplies \(\hat y\); score support for that actual prediction:

\[
s(x)=a_{\hat y}-\max_{c\ne\hat y}a_c.
\]

Do not substitute the prototype branch's own predicted label. A negative score indicates disagreement with the final diagnosis. This is a ranking statistic, not a calibrated correctness probability. No SCSF TCP head is required.

### Difference from previous attempts

Unlike the repository's acceptance-weighted sample contrastive objectives, this uses learned class representatives without queues, sample-pair mining, or acceptance weights. Proxy learning and class sub-centers are established ideas [5,6]; this simplified cosine-prototype objective is not a reproduction of either cited loss.

**Risks:** collapse to one center, unused centers, acquisition artifacts, fitting noise, and confident shared errors. Log center occupancy per class, center similarity, and final/auxiliary disagreement conditional on correctness.

**Mechanism control:** set \(K=1\), keeping other settings fixed. If it matches or wins, the multiple-mode explanation is unsupported.

## 6. Method 3: SE-DS

### Hypothesis and objective

Global pooling may dilute localized diagnostic evidence. Attach a bias-enabled `1x1` convolution to `layer3`, producing one spatial evidence map per class:

\[
A=\operatorname{Conv}_{1\times1}(H_3)\in\mathbb R^{C\times H\times W}.
\]

Pool the strongest 20% of spatial values per class:

\[
k=\max(1,\lceil0.2HW\rceil),\qquad
a_c=\frac1k\sum_{j\in\operatorname{TopK}(A_c)}A_{c,j}.
\]

Train from the image diagnosis label:

\[
L_{\mathrm{SE}}=\mathrm{CE}(a,y),\qquad
L=L_{\mathrm{cls}}+\alpha_eL_{\mathrm{SE}}.
\]

No masks, anatomical annotations, crops predicted by the model, flipped image, or additional backbone pass are required. Selected values receive gradients; top-k indices are discrete.

### Selective inference

Retain the spatial branch and use:

\[
s(x)=a_{\hat y}-\max_{c\ne\hat y}a_c.
\]

The final classifier supplies the diagnosis; intermediate spatial evidence supplies its selection score. Use raw scores, without sigmoid. No SCSF TCP head is required.

### Difference from previous attempts

The existing `spatial_head` predicts scalar view agreement using attention and pooled statistics. This branch instead receives class-label supervision at spatial features. It also differs from the previous proposal's candidate-verification BCE and two-view CE. Image-label spatial supervision is established prior art [7,8], not a new invention here.

**Risks:** shortcut regions, missed diffuse evidence, unstable top-k selection, and confident wrong predictions. Evidence maps are not validated lesion segmentations. Inspect validation examples and class-specific errors.

**Mechanism control:** use global average pooling with the identical convolution. This tests spatial selection against ordinary pooled auxiliary classification.

## 7. Attribution: Does Deep Supervision Explain the Gain?

Distinguish a useful confidence readout from useful supervision entering the backbone:

| Condition | Auxiliary parameters learn? | Auxiliary gradients enter backbone? | Inference architecture |
|---|---|---|---|
| Full | Yes | Yes | Candidate's specified architecture |
| All-DS-detached | Yes | No | Identical to Full |

Detach only the feature inputs to auxiliary branches; preserve final CE gradients. In MG-SCSF, All-DS-detached detaches both the original TCP branch and the new margin branch. In MP-DS/SE-DS, detach the new branch input. Existing detached targets and logit inputs remain detached.

For MG-SCSF, also compare against ordinary active SCSF to isolate the new margin supervision. If it wins, run a control detaching only the new margin branch while original SCSF gradients stay active. Since the margin branch is unused at inference, this should match ordinary SCSF under matched randomness and optimizer behavior; verify instead of assuming exact numerical equivalence.

Report final-classifier MSP and designated confidence scores on both Full and All-DS-detached checkpoints. This gives a two-by-two comparison of backbone supervision and score choice without fitting additional models. It exposes representation gains, readout gains, and their interaction. A gain visible with MSP also supports useful training.

### The proposed 50–70% contribution target

For one matched dataset/seed and a fixed primary AURC protocol, let:

- \(B\): single-view CE baseline, scored by MSP;
- \(D\): All-DS-detached candidate, scored by its designated head;
- \(F\): Full candidate, scored by the same head.

If \(B-F\) is meaningfully positive, report the sequential attribution:

\[
\text{DS fraction}=\frac{D-F}{B-F},\qquad
B-F=(B-D)+(D-F).
\]

Illustration only: AURC values 0.050, 0.046, and 0.040 yield 60%. These are not experimental results.

This percentage depends on baseline and decomposition order, not a unique allocation of causality. It can exceed 100%, be negative, or become unstable near zero total improvement. Do not clip it or choose denominators to obtain a preferred percentage. Report absolute paired improvements and paired-seed uncertainty first. A substantial reproducible contribution is the objective; 50–70% is an aspiration, not a guaranteed threshold.

For MG-SCSF, this fraction credits **all** DS, including original SCSF supervision. Separately report `AURC(SCSF) - AURC(MG-SCSF)` so the new branch receives no credit for inherited gains.

## 8. Compute-Bounded Screening

A full run currently costs approximately 4–6 hours. New-head overhead is unmeasured; profile it. Three methods do not mean three runs can establish their mechanisms.

1. **Execution checks:** synthetic gradient/shape tests, a small-data smoke test, and timing. These are not performance evidence.
2. **First screen:** all three Full candidates, one medical dataset, seed 0, complete 100-epoch schedule: approximately **12–18 hours**. Reuse CE and SCSF baselines only when manifests, settings, seeds, scoring, and evaluation match; otherwise add 4–6 hours per missing baseline. Compare last-epoch validation AURC.
3. **Mechanism screen:** All-DS-detached for promising candidates. One survivor adds **4–6 hours**; all three add **12–18 hours**. A candidate beating CE but not its detached counterpart does not support the intended DS claim.
4. **Confirmation:** the leading Full/Detached pair on a second medical dataset, then repeat promising comparisons with seeds 1 and 2. Each pair costs approximately **8–12 hours**. Run its mechanism control and, for MG-SCSF, the extra margin-path control before making a mechanism claim.
5. **Full benchmark:** expand confirmed candidates to the full medical suite and relevant training-based comparators. Report unsuccessful candidates too.

First screening plus one detached survivor is approximately **16–24 hours** excluding missing baselines and overhead. All three active/detached pairs cost **24–36 hours** under the same approximation. Use validation for selection; a single-seed result is a screening signal, not significance evidence.

For a failure, check that the auxiliary gradient is active and finite before interpreting it. One failed setting does not prove DS ineffective, but does not justify unlimited tuning either. Allow at most one documented diagnostic revision before reconsidering the hypothesis.

## 9. Evaluation and Claim Boundaries

Primary endpoint: full risk–coverage AURC using the designated score and a declared tie policy. Also report accuracy, risk at predeclared coverages, correctness-detection AUROC, and same-checkpoint MSP AURC. Record per-class retention and errors; aggregate gains can hide rejection of difficult classes.

Apply relevant fixes from the [implementation review](medical_day_implementation_review.md) before trusted comparisons. Ensure correct image discovery, complete artifacts, and valid metric edge cases. Recompute or rerun baselines affected by preprocessing or scoring changes. Do not use clipped NAURC for attribution.

Promote candidates whose validation gains repeat and whose active DS improves over detachment. Report accuracy changes alongside AURC. Freeze configurations before final test reporting. Historical test results have already informed development; do not describe them as wholly untouched evidence.

Existing benchmark wins do not establish global SOTA. Reassess strong relevant training-based comparators, including classifier-training approaches [2,3] and CCL-SC [9], under matched protocols. Post-hoc methods are outside the proposed method scope; qualify comparison claims accordingly. This focused literature check does not establish exhaustive novelty or current SOTA coverage.

## 10. Implementation Handoff

Proposed identifiers: `mg_scsf`, `mp_ds`, `se_ds`. **Implemented; no experiment was launched by this document.**

Implementation status:

| Specification | CLI (`run_experiment.py`) | Suite flag (`run_paper_medical_suite.py`) |
|---|---|---|
| \(\alpha_e\) weight / ramp | `--ds-aux-weight 0.1 --ds-aux-ramp 5` | same |
| Full / All-DS-detached / margin-only detached | `--ds-detach none\|all\|new` | `--ds-detach-modes` (`new` is skipped for `mp_ds`/`se_ds`) |
| MG hinge / CE control | `--mg-loss hinge\|ce --mg-margin 0.2 --mg-ce-scale 10` | `--mg-losses` |
| MP prototypes / K=1 control | `--mp-prototypes 3 --mp-dim 128 --mp-temperature 0.1` | `--mp-prototypes` (list) |
| SE top-k / GAP control | `--se-pool topk\|gap --se-topk-fraction 0.2` | `--se-pools` |
| MP/SE branch site (extension beyond this spec) | `--ds-site layer3\|layer4\|layer3+layer4` (default `layer3`) | `--ds-sites`; directories `l3-`, `l4-`, `l34-` |

The `layer3+layer4` site mirrors SCSF's mid+late features: MP concatenates GAP(layer3) and GAP(layer4) before the projection; SE concatenates layer3 with nearest-upsampled layer4 before the 1x1 convolution. With a layer4 site the auxiliary loss also updates the classifier's input features, so it is no longer purely intermediate supervision; report it as a separate variant with its own All-DS-detached control (`DAY_PLAN=ds_l34`).

Variant directories encode site, head, schedule and gradient mode, e.g. `l3-hinge-m0.2__w0.1r5__detach-none`. `mg_scsf` receives the suite's SCSF flags (TCP target, feature spec, meta lr, lambda) so it matches the `scsf` baseline. Each run writes `aux_grad_norms.csv` (per-term CE/TCP/margin or CE/aux gradient norms at layer2/3/4, first post-warm-up batch of every epoch), `aux_diagnostics.csv` (per-epoch MG active fraction and margin per class; MP centre occupancy, unused centres and centre cosine per class; final/auxiliary disagreement given correctness for MP/SE), `train_seconds` in `complete.json`, and same-checkpoint `msp_*` metrics in `summary.csv`. Auxiliary parameters are initialised from an isolated RNG stream (`mg_scsf` and `scsf` start from identical weights; MP/SE controls share the backbone initialisation and data order).

Launchers: `DAY_PLAN=ds bash run_medical_day.sh next --gpu 0` (three Full candidates) and `DAY_PLAN=ds_controls ...` (All-DS-detached, mechanism controls, margin-only detach). For the first screen use `DAY_DATASETS="<slug>" DAY_SEEDS="0"`.

- `scsf/models.py`: expose pooled `layer3` features and, when needed, its spatial map from the same forward. Preserve existing outputs; avoid repeated feature extraction and duplicate BatchNorm updates.
- `scsf/methods.py`: implement objectives, scores, diagnostics, and explicit detachment modes. Reuse SCSF behavior for MG-SCSF without changing baseline defaults.
- `run_experiment.py`: register methods and optimizer grouping. New parameters use SGD; MG-SCSF's calibrator keeps Adam. Avoid silently falling through to the generic optimizer branch.
- `run_paper_medical_suite.py`: forward arguments and include branch site, coefficient, head settings, gradient mode, and score in resolved configurations and unique output identities.
- Launch pilots through the medical suite with explicit method/dataset/seed subsets. `run_medical_day.sh` currently hardcodes seven methods and 21 runs per method; new candidates are not automatically supported. Its day numbering is not a pilot command.
- Save separate head and MSP metrics, exact primary score, and last checkpoints.

Implementation tests must cover active upstream gradients on constructed examples; absence of auxiliary gradients under detachment; intact final CE gradients; zero MG loss beyond its margin; exclusion of the true class from competitors; MP normalization and selected-center gradients; SE pooling on a known tensor; confidence indexed by the final predicted class; label-free inference; CLI/configuration round-trip; checkpoint round-trip. Run the repository suite when implementation is performed.

## 11. Primary References

1. [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html): post-processing calibration.
2. [Feng et al., Towards Better Selective Classification, ICLR 2023](https://arxiv.org/abs/2206.09034): classifier training versus selection mechanisms.
3. [Zhu et al., Revisiting Confidence Estimation: Towards Reliable Failure Prediction, TPAMI](https://arxiv.org/abs/2403.02886): confidence separation and failure-prediction training.
4. [Lee et al., Deeply-Supervised Nets, AISTATS 2015](https://proceedings.mlr.press/v38/lee15a.html): intermediate companion objectives.
5. [Kim et al., Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html): learned proxies.
6. [Deng et al., Sub-center ArcFace, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1445_ECCV_2020_paper.php): multiple class centers; evidence comes from face recognition, not these medical tasks.
7. [Durand et al., WILDCAT, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.html): spatial image-label supervision.
8. [Belharbi et al., Deep Interpretable Classification and Weakly-Supervised Segmentation of Histology Images via Max-Min Uncertainty](https://arxiv.org/abs/2011.07221): related medical spatial supervision, not proof of selective gains.
9. [Wu et al., Confidence-aware Contrastive Learning for Selective Classification, ICML 2024](https://arxiv.org/abs/2406.04745): training-based comparator.
