# 3.3. Results and analysis

This section reports the results of the medical-suite study described in Sections 3.1 and 3.2. Two experimental records are used.

- **Main suite.** Seven methods (SR, SAT, Deep Gamblers, SelectiveNet, CCL-SC, SCSF, and the Original Baseline, labelled DualAug in Table 2) were trained on the seven datasets of Table 1 with training seeds 0, 1, and 2. This gives 147 runs, produced with commit 06a8e2c. All conclusions about the Original Baseline and the external baselines rest on this record.
- **Variant runs.** The three designs of Section 2.6 (residual head, decision-preserving head, and spatial head) were trained in an earlier run with commit ea8e5fb and training seeds 42, 0, and 1. The backbone, input resolution, optimizer, schedule, and final-epoch checkpoint were the same as in the main suite. The training and test images were also identical (verified file by file) on six of the seven datasets. On HAM10000, the earlier code split the data by image rather than by lesion, so only 414 of its 2,004 test images coincide with the lesion-grouped test set of the main suite. HAM10000 results of the variant runs are therefore not reported.

Unless stated otherwise, every value is the mean ± sample standard deviation (ddof = 1) over three training seeds. Each run is evaluated once on the full test set using the final-epoch checkpoint. AURC and E-AURC are reported ×10⁻³. Following Section 2.2, accuracy and failure-ranking metrics are reported alongside AURC. The goal is to separate gains in confidence ranking from gains in classification accuracy.

Three seeds do not support formal significance tests. As a simple screening rule, a difference between two methods is called *separated* when it exceeds the combined standard deviation \(\sqrt{s_1^2+s_2^2}\). Otherwise, it is described as being within seed variation.

## 3.3.1. Main comparison on the medical suite

Table 3 reports AURC for the seven methods of the main suite. The Original Baseline obtains the best average rank (2.29). It has the lowest AURC on four of the seven datasets: Malaria, APTOS 2019, Brain Tumor MRI, and Breast Ultrasound. Across datasets, its AURC is on average 17% lower than that of SR (geometric mean of the per-dataset ratios, 0.83). SAT and SCSF share second place (2.86). Each obtains the lowest AURC on one dataset: SAT on COVID-QU-Ex and SCSF on HAM10000. SR obtains the lowest AURC on Chest CT and ranks fourth overall (3.29).

The official reproduction of CCL-SC [42] does not improve on SR on average (rank 4.29; geometric-mean AURC ratio to SR 1.03). It is nevertheless the second-best method on APTOS 2019 and Breast Ultrasound. Deep Gamblers is worse than SR on all seven datasets.

SelectiveNet is the weakest method by a large margin. Its selection head saturates: on HAM10000 and Brain Tumor MRI, every test image receives a selector logit of at least 5. Its AUROC falls below 0.5 on four datasets (Table 4), although its accuracy matches the other methods (Table 5). This collapse reflects the selector under the default configuration of Table 2 (target coverage 0.8, λ = 32) rather than the quality of the underlying classifier. The SelectiveNet results should therefore be read as a configuration failure, not as evidence about the method in general.

| Method | COVID-QU-Ex | HAM10000 | Malaria | APTOS 2019 | Brain Tumor MRI | Breast US | Chest CT | Avg. rank |
|---|---|---|---|---|---|---|---|---|
| SR [5] | 5.43 ± 0.20 | <u>36.27 ± 1.53</u> | 4.28 ± 1.20 | 60.16 ± 6.24 | 12.62 ± 4.96 | 24.69 ± 7.15 | **18.43 ± 3.46** | 3.29 |
| SAT [8] | **4.31 ± 0.01** | 37.30 ± 2.55 | 3.37 ± 0.19 | 43.45 ± 1.88 | 8.81 ± 2.11 | 28.08 ± 2.54 | 22.63 ± 1.31 | 2.86 |
| Deep Gamblers [7] | 7.81 ± 0.29 | 44.50 ± 2.06 | 4.81 ± 0.36 | 67.32 ± 1.95 | 13.48 ± 4.39 | 41.90 ± 24.97 | 23.12 ± 3.54 | 5.71 |
| SelectiveNet [6] | 8.70 ± 2.13 | 139.92 ± 30.28 | 4.76 ± 1.12 | 185.42 ± 39.41 | 51.89 ± 17.38 | 101.22 ± 21.27 | 85.16 ± 23.37 | 6.71 |
| CCL-SC [42] | 6.08 ± 0.78 | 41.45 ± 7.26 | 5.49 ± 1.23 | <u>43.33 ± 6.30</u> | 9.99 ± 4.61 | <u>23.61 ± 1.06</u> | 25.49 ± 8.15 | 4.29 |
| SCSF | 5.87 ± 0.83 | **34.39 ± 3.64** | <u>3.34 ± 0.19</u> | 49.82 ± 3.85 | <u>7.09 ± 3.44</u> | 29.26 ± 2.87 | <u>19.65 ± 6.58</u> | 2.86 |
| Original Baseline (DualAug) | <u>4.64 ± 0.84</u> | 39.32 ± 2.62 | **3.25 ± 0.24** | **41.76 ± 0.75** | **5.56 ± 1.19** | **20.41 ± 4.51** | 26.99 ± 6.53 | **2.29** |

Table 3. AURC (×10⁻³, lower is better) on the medical suite, mean ± std over three training seeds. Bold: best; underlined: second best. Average rank is computed over the seven datasets.

| Method | COVID-QU-Ex | HAM10000 | Malaria | APTOS 2019 | Brain Tumor MRI | Breast US | Chest CT |
|---|---|---|---|---|---|---|---|
| SR | 0.898 ± 0.004 | 0.862 ± 0.008 | 0.900 ± 0.018 | 0.816 ± 0.040 | 0.869 ± 0.048 | 0.794 ± 0.053 | 0.838 ± 0.017 |
| SAT | **0.906 ± 0.009** | 0.854 ± 0.009 | 0.909 ± 0.002 | 0.848 ± 0.016 | 0.886 ± 0.030 | 0.763 ± 0.013 | 0.845 ± 0.040 |
| Deep Gamblers | 0.845 ± 0.003 | 0.841 ± 0.013 | 0.892 ± 0.005 | 0.740 ± 0.011 | 0.836 ± 0.036 | 0.681 ± 0.090 | 0.814 ± 0.043 |
| SelectiveNet | 0.848 ± 0.037 | 0.539 ± 0.082 | 0.889 ± 0.012 | 0.468 ± 0.083 | 0.394 ± 0.104 | 0.408 ± 0.092 | 0.467 ± 0.018 |
| CCL-SC | 0.884 ± 0.006 | 0.858 ± 0.013 | 0.887 ± 0.011 | 0.824 ± 0.008 | 0.835 ± 0.076 | 0.775 ± 0.024 | **0.854 ± 0.013** |
| SCSF | 0.893 ± 0.010 | **0.868 ± 0.013** | 0.912 ± 0.004 | 0.829 ± 0.015 | 0.896 ± 0.052 | 0.770 ± 0.032 | 0.820 ± 0.022 |
| Original Baseline | 0.899 ± 0.013 | 0.838 ± 0.013 | **0.918 ± 0.002** | **0.850 ± 0.005** | **0.917 ± 0.016** | **0.839 ± 0.012** | 0.802 ± 0.058 |

Table 4. AUROC of the confidence score for separating correct from incorrect predictions (higher is better; 0.5 corresponds to random ranking).

| Method | COVID-QU-Ex | HAM10000 | Malaria | APTOS 2019 | Brain Tumor MRI | Breast US | Chest CT |
|---|---|---|---|---|---|---|---|
| SR | 96.38 ± 0.05 | 85.03 ± 0.52 | 97.25 ± 0.05 | 83.79 ± 0.42 | 95.71 ± 0.17 | 92.38 ± 0.55 | 93.75 ± 0.60 |
| SAT | 96.66 ± 0.19 | 85.28 ± 0.44 | 97.30 ± 0.06 | 83.33 ± 1.19 | 95.77 ± 0.13 | 92.70 ± 1.14 | 92.13 ± 1.97 |
| Deep Gamblers | 96.61 ± 0.11 | 84.45 ± 0.14 | 97.29 ± 0.15 | 84.70 ± 0.47 | 95.73 ± 0.13 | 92.70 ± 1.68 | 92.94 ± 1.71 |
| SelectiveNet | 96.43 ± 0.13 | 84.95 ± 0.34 | **97.38 ± 0.08** | 84.43 ± 0.27 | 95.81 ± 0.20 | 91.85 ± 0.66 | 91.55 ± 1.91 |
| CCL-SC | 96.50 ± 0.19 | 84.35 ± 0.42 | 97.09 ± 0.22 | **85.34 ± 1.14** | **95.96 ± 0.11** | **93.02 ± 0.63** | 91.90 ± 0.80 |
| SCSF | 96.38 ± 0.16 | 84.93 ± 0.40 | 97.26 ± 0.08 | 83.33 ± 0.47 | 95.85 ± 0.20 | 91.96 ± 1.11 | **93.87 ± 0.72** |
| Original Baseline | **96.71 ± 0.11** | **85.45 ± 0.18** | 97.31 ± 0.16 | 83.70 ± 0.42 | 95.62 ± 0.04 | 92.38 ± 0.84 | 91.20 ± 1.22 |

Table 5. Test accuracy (%).

### Accuracy and ranking quality

Classification accuracy varies little across methods (Table 5). On five datasets, the spread between the best and worst method is about 1.2 percentage points or less. The two exceptions are APTOS 2019 (83.33–85.34%) and Chest CT (91.20–93.87%).

The ordering in Table 3 is therefore driven mainly by the quality of the confidence ranking rather than by differences in base error. This is confirmed by AUROC (Table 4), whose average ranks closely follow the AURC ranks:

- Original Baseline: 2.57.
- SAT and SCSF: 2.86.
- SR: 3.29.
- CCL-SC: 4.14.
- Deep Gamblers: 5.57.
- SelectiveNet: 6.71.

The clearest exception is Chest CT. There, the Original Baseline is the least accurate method (91.20% vs. 93.75% for SR), and part of its higher AURC comes from its larger base error.

### Original Baseline versus SR and the strongest baseline

Compared with SR, the Original Baseline reduces AURC on five datasets and increases it on two:

- **Separated reductions:** APTOS 2019 (−31%, 60.16 → 41.76) and Brain Tumor MRI (−56%, 12.62 → 5.56).
- **Reductions within seed variation:** COVID-QU-Ex (−15%), Malaria (−24%), and Breast Ultrasound (−17%).
- **Separated increases:** HAM10000 (+8%) and Chest CT (+46%).

Against the strongest of the other six methods on each dataset, the margin is smaller:

| Dataset | Strongest other method | Original Baseline AURC vs. that method |
|---|---|---|
| Brain Tumor MRI | SCSF | −22% |
| Breast Ultrasound | CCL-SC | −14% |
| APTOS 2019 | CCL-SC | −4% |
| Malaria | SCSF | −3% |
| COVID-QU-Ex | SAT | +8% |
| HAM10000 | SCSF | +14% |
| Chest CT | SR | +46% |

The Original Baseline is therefore the best method overall, but not uniformly. Its advantage is large on Brain Tumor MRI and APTOS 2019, marginal on Malaria, and absent on HAM10000 and Chest CT.

Two features of the losing cases are informative:

- **HAM10000 (7 classes, strongly imbalanced).** The Original Baseline has the highest accuracy, but its AUROC (0.838) is lower than that of SR (0.862) and SCSF (0.868). Its agreement head ranks errors worse than the plain softmax. This is consistent with the limitation noted in Section 2.5: the two views can agree on the same wrong class. In a dataset dominated by one class (nv), many errors are confident, flip-consistent confusions with the majority class.
- **Chest CT (498 training images, 288 test images).** Both accuracy and AUROC drop, and the standard deviation is large. With so few training images, the two-view objective and the agreement head have little data to learn from. A difference of one test image already changes the error rate by about 0.35 percentage points.

### Stability across seeds

SAT is the most stable method. Its mean coefficient of variation of AURC across seeds is 0.08, averaged over the six datasets excluding HAM10000. The corresponding values for the other methods are:

- Original Baseline: 0.16.
- SCSF and Deep Gamblers: 0.20.
- SR and CCL-SC: 0.22.
- SelectiveNet: 0.25.

The Original Baseline has a lower standard deviation than SR on four of the seven datasets. The difference is most visible on APTOS 2019 (0.75 vs. 6.24) and Brain Tumor MRI (1.19 vs. 4.96). It is less stable than SR on COVID-QU-Ex, HAM10000, and Chest CT. SAT thus offers a slightly worse average AURC than the Original Baseline but much more predictable behaviour across seeds.

## 3.3.2. Risk–coverage behaviour of the Original Baseline

AURC summarizes the entire risk–coverage curve. In deployment, however, a selective classifier operates at one coverage level, and in clinical use this level is usually high. Table 6 compares the Original Baseline with SR at fixed coverages. It also lists two threshold-free summaries, E-AURC and FPR@95TPR.

| Dataset | @95% | @90% | @80% | @70% | @50% | E-AURC (×10⁻³) | FPR@95TPR |
|---|---|---|---|---|---|---|---|
| COVID-QU-Ex | 1.97 / 1.97 | 1.31 / 1.25 | 0.81 / 0.71 | 0.60 / 0.48 | 0.34 / **0.18** | 4.77 / 4.09 | **0.425** / 0.489 |
| HAM10000 | 12.83 / 12.85 | 10.77 / 11.36 | **7.01** / 8.01 | **4.63** / 5.23 | 2.00 / 1.76 | **24.41** / 28.14 | **0.671** / 0.761 |
| Malaria | 1.29 / 1.26 | 0.87 / 0.81 | 0.58 / **0.41** | 0.42 / **0.29** | 0.25 / 0.15 | 3.90 / 2.89 | 0.393 / 0.385 |
| APTOS 2019 | **14.27** / 14.85 | **11.96** / 12.87 | 9.10 / 9.90 | 6.51 / 5.73 | 3.83 / **0.73** | 46.01 / **27.45** | 0.709 / 0.805 |
| Brain Tumor MRI | **1.73** / 2.33 | 1.22 / 1.45 | 1.13 / **0.71** | 0.96 / **0.39** | 0.93 / **0.38** | 11.67 / **4.58** | 0.280 / 0.374 |
| Breast US | 5.57 / 6.24 | 4.93 / 4.46 | 4.50 / **2.51** | 3.48 / 2.42 | 1.69 / 1.48 | 21.58 / 17.29 | 0.656 / 0.665 |
| Chest CT | **4.50** / 7.79 | **3.73** / 6.31 | **2.61** / 4.78 | **1.32** / 3.47 | 0.93 / 1.85 | 16.32 / 22.79 | **0.577** / 0.733 |

Table 6. Selective risk (%) at fixed coverage, E-AURC, and FPR@95TPR. Each cell is SR / Original Baseline (lower is better). Bold marks the better method where the difference is separated under the screening rule.

The comparison shows a clear pattern. The advantage of the Original Baseline is concentrated at moderate and low coverage:

- **APTOS 2019:** at 50% coverage, the selective risk falls from 3.83% to 0.73%.
- **Brain Tumor MRI:** at 70% coverage, it falls from 0.96% to 0.39%.
- **Breast Ultrasound:** at 80% coverage, it falls from 4.50% to 2.51%.

At 95% coverage, the Original Baseline is never meaningfully better than SR. It is worse on APTOS 2019 (14.85% vs. 14.27%), Brain Tumor MRI (2.33% vs. 1.73%), and Chest CT (7.79% vs. 4.50%). FPR@95TPR tells the same story. When 95% of correct predictions are accepted, the Original Baseline accepts a larger fraction of errors than SR on six of the seven datasets. The difference is separated on COVID-QU-Ex, HAM10000, and Chest CT.

The curves therefore cross. SR ranks the few most doubtful cases at least as well, whereas the agreement head is better at separating a broader group of uncertain cases from clearly reliable ones.

A plausible explanation follows from the agreement target. At high coverage, the cases that must be rejected are the most confident errors. Their two views usually agree, so the agreement label is 1 and offers no signal to rank them below correct predictions. At moderate coverage, the rejected set includes many flip-inconsistent cases, and these are exactly what the head was trained to detect.

For a clinical workflow that escalates only 5–10% of cases, this weakens the practical value of the gains in Table 3. The improvement is real, but it materializes mainly when a larger share of cases can be deferred.

[TODO] Insert Figure 3: risk–coverage curves of all seven methods on each dataset, generated from `tables/risk_coverage_mean_std.csv` of the main suite. Use a logarithmic risk axis for COVID-QU-Ex and Malaria. Remove the old Figure 3 and Tables 3–4 of the earlier draft.

## 3.3.3. Evaluation of the proposed variants

Table 7 compares the three designs of Section 2.6 with the Original Baseline and SR on the six datasets where the records are comparable. Ranks are computed over the ten methods of Tables 3 and 7. (The earlier CCL-SC implementation, which was superseded by the official reproduction, is excluded.)

| Method | COVID-QU-Ex | Malaria | APTOS 2019 | Brain Tumor MRI | Breast US | Chest CT | Avg. rank (10 methods) |
|---|---|---|---|---|---|---|---|
| SR | 5.43 ± 0.20 | 4.28 ± 1.20 | 60.16 ± 6.24 | 12.62 ± 4.96 | 24.69 ± 7.15 | **18.43 ± 3.46** | 4.50 |
| Original Baseline | 4.64 ± 0.84 | **3.25 ± 0.24** | **41.76 ± 0.75** | **5.56 ± 1.19** | **20.41 ± 4.51** | 26.99 ± 6.53 | **2.17** |
| Residual head † | 4.46 ± 0.37 | 4.14 ± 1.57 | 86.76 ± 35.22 | 39.75 ± 16.69 | 47.40 ± 7.43 | 45.62 ± 28.02 | 7.17 |
| Decision-preserving head † | 4.68 ± 0.63 | 3.37 ± 0.36 | 71.65 ± 16.65 | 52.91 ± 20.18 | 36.62 ± 14.87 | 44.50 ± 12.67 | 6.67 |
| Spatial head † | 12.84 ± 3.16 | 3.99 ± 0.36 | 56.42 ± 3.84 | 29.78 ± 23.57 | 29.82 ± 14.57 | 45.14 ± 23.44 | 6.83 |
| *AUROC* | | | | | | | |
| Residual head † | **0.913 ± 0.007** | 0.916 ± 0.011 | 0.736 ± 0.064 | 0.497 ± 0.142 | 0.698 ± 0.078 | 0.654 ± 0.159 | |
| Decision-preserving head † | 0.912 ± 0.012 | **0.918 ± 0.002** | 0.748 ± 0.044 | 0.430 ± 0.176 | 0.734 ± 0.076 | 0.633 ± 0.112 | |
| Spatial head † | 0.770 ± 0.070 | 0.905 ± 0.005 | 0.807 ± 0.021 | 0.626 ± 0.218 | 0.754 ± 0.096 | 0.613 ± 0.016 | |

Table 7. AURC (×10⁻³) and AUROC of the proposed variants. † Variant runs (commit ea8e5fb, training seeds 42, 0, 1). The ten-method average rank is computed over the six datasets shown; on this basis, the other methods of Table 3 rank SAT 2.83, SCSF 3.50, CCL-SC 5.00, Deep Gamblers 7.00, and SelectiveNet 9.33. HAM10000 is omitted because the test sets differ.

None of the three variants improves on the Original Baseline. Their average ranks (6.67–7.17) place them below SR and the official CCL-SC, and roughly level with Deep Gamblers. The variants are competitive only on the two largest datasets:

- **COVID-QU-Ex.** The residual head has the second-lowest AURC of all ten methods (4.46) and the highest AUROC (0.913).
- **Malaria.** The decision-preserving head matches SAT (3.37).

On the four small or saturated datasets, all three variants fail. Their AUROC on Brain Tumor MRI is at or below chance (0.43–0.63). On Chest CT, it is 0.61–0.65, compared with 0.80–0.85 for all other methods except SelectiveNet. Their seed-to-seed variation is also the largest in the study (mean coefficient of variation 0.26–0.37).

Their accuracies [TODO: add the variants' accuracy table to the appendix] are within about one percentage point of the other methods on these datasets. The failures are therefore failures of confidence ranking, not of classification.

### Possible causes

Several factors are consistent with this pattern.

1. **Training and test accuracy diverge on some datasets.** On Brain Tumor MRI, all methods reach about 99% validation accuracy but only about 95.8% test accuracy. Training accuracy is close to 100%, so the two views almost always agree during training. The agreement target is then nearly constant, and the head learns to output very large logits for almost every image.

2. **Confidence ties at 1.0.** The variants were ranked by sigmoid probability. In single precision, the probability saturates to exactly 1.0, creating ties, as explained in Section 3.1.4.
   - On Brain Tumor MRI (seed 42), 28% of the test images received a confidence of 1.0 under the decision-preserving head. That group contained 31 of its 65 errors.
   - The order inside such a tie is determined by the sorting procedure rather than by the model. This can push AUROC below 0.5 when the test set is stored in class order.
   - The main suite ranks all learned heads by their unsaturated logits and is not affected.

3. **The variants were not trained with two-view classification.** In the evaluated implementation, cross-entropy is computed on the original image only. The flipped view is passed through the network without gradients and used only to build the agreement target.
   - Section 3.3.4 shows that this regime, an agreement head on a single-view backbone, loses most of the benefit of the Original Baseline on the harder benchmark.
   - The comparison in Table 7 therefore confounds the architectural change of each variant with the removal of two-view training.

4. **The two records differ in code version and training seeds.** The variant runs used commit ea8e5fb with seeds 42/0/1. Seed-to-seed variation on the small datasets is large in both records.

Because of these confounds, Table 7 does not establish that residual decomposition, decision-preserving updates, or spatial readout are harmful as such. It shows that the variants, as implemented and trained here, do not deliver the improvement they were designed to provide. The original design hypotheses (Section 2.6) remain untested in a controlled form.

[VERIFY] Section 2.5 states that the Original Baseline and all three variants use two-view classification. In the evaluated variant code, only the Original Baseline does. Either revise Section 2.5 to state that the variants use single-view classification, or retrain the variants with two-view classification before submission.

[VERIFY] Section 2.6.2 describes a logit-preserving projection with the Moore–Penrose pseudoinverse of the logit Jacobian over selected shared parameters. The evaluated decision-preserving head implements a simpler rule. At the layer-3 activations, for each sample, it removes from the auxiliary gradient its component parallel to the cross-entropy gradient (a damped, per-sample orthogonal projection, in the spirit of gradient surgery [44]). The text of Section 2.6.2 must describe the evaluated rule, or the results must be labelled as an approximation of it.

## 3.3.4. Component analysis

The medical suite does not include a controlled ablation of the Original Baseline. Its components were instead isolated on the standard benchmarks. The setting was VGG16-BN on CIFAR-10 and CIFAR-100, 300 epochs, final-epoch checkpoint, and full official test set. Table 8 summarizes the six variants. Each removes or replaces one component:

- **Two-view classification:** compared by `msp` → `dualce_msp`, and by `agree_only` → `dualaug`.
- **Agreement versus correctness target:** `dualce_scsf` versus `dualaug_detach`.
- **Gradient flow from the head into the backbone:** `dualaug` versus `dualaug_detach`.
- **Ranking by the head rather than by the softmax:** each head-based model versus its own MSP.

| Variant | Two-view CE | Head target | Head gradient into backbone | CIFAR-10 Acc. | CIFAR-10 AURC | CIFAR-100 Acc. | CIFAR-100 AURC |
|---|---|---|---|---|---|---|---|
| `msp` (SR) | no | – | – | 93.82 ± 0.01 | 7.41 ± 0.24 | 72.96 ± 0.29 | 81.41 ± 0.12 |
| `agree_only` | no | agreement | yes | 94.05 ± 0.14 | 5.84 ± 0.04 | 72.49 ± 0.16 | 80.46 ± 0.40 |
| `dualce_msp` | yes | – (MSP) | – | 94.35 ± 0.08 | 7.62 ± 0.27 | 74.07 ± 0.18 | 78.56 ± 0.56 |
| `dualce_scsf` | yes | correctness | yes | 94.30 (n = 1) | 5.28 | 74.13 ± 0.01 | 72.97 ± 0.45 |
| `dualaug` (Original Baseline) | yes | agreement | yes | 94.50 ± 0.09 (n = 5) | 5.24 ± 0.18 | 74.27 ± 0.15 (n = 5) | 72.86 ± 0.96 |
| `dualaug_detach` | yes | agreement | no | 94.35 ± 0.08 | **5.11 ± 0.00** | 74.51 ± 0.08 | **72.80 ± 1.24** |

Table 8. Component analysis of the Original Baseline on the standard benchmarks (VGG16-BN). AURC ×10⁻³, using the head score where a head exists. Two seeds per row unless indicated.

Four conclusions follow.

1. **Ranking by the head is the largest single contributor on both datasets.** On the same checkpoint, the head lowers AURC relative to the model's own MSP by 2.45 × 10⁻³ on CIFAR-10 and 4.93 × 10⁻³ on CIFAR-100.

2. **Two-view classification matters on the harder task.**
   - On CIFAR-100, two-view training alone lowers the MSP AURC from 81.41 to 78.56 and raises accuracy by 1.1 points.
   - An agreement head on a single-view backbone (`agree_only`) reaches only 80.46, far from the 72.86 of the full method.
   - With a single-view backbone, the two views of a fine-grained image often disagree even when the prediction is correct. The agreement signal then becomes noisy.
   - On CIFAR-10, `agree_only` already recovers most of the gain (5.84 vs. 5.24).
   - This regime is exactly the one in which the three variants of Section 3.3.3 were trained.

3. **Agreement supervision is as effective as correctness supervision.** Compare `dualce_scsf` with `dualaug_detach`: 5.28 vs. 5.11 and 72.97 vs. 72.80. The difference is within seed variation. The agreement target has the advantage of not requiring labels to define reliability.

4. **Allowing the head's gradient to reach the backbone gives no measurable benefit.** Detaching the pooled intermediate features from the head leaves AURC unchanged or slightly lower (5.11 vs. 5.24; 72.80 vs. 72.86). The head still reads intermediate features. What is removed is only the deep-supervision pathway through which its loss reshapes the backbone.

Taken together, these results indicate that the gains of the Original Baseline come from two-view classification and from a learned agreement head that reads intermediate features. They do not come from the auxiliary gradient reshaping the backbone. Whether reading intermediate features is itself necessary was not isolated: a head on the terminal logits alone was not trained. This ablation was performed on natural-image benchmarks with two seeds per variant. It has not been repeated on the medical suite.

## 3.3.5. Answers to the research questions

**RQ1: Does reliability-oriented supervision at intermediate stages improve selective prediction compared with terminal-confidence methods?**

Partially. The Original Baseline, which supervises an agreement head on intermediate features, has the best average AURC and AUROC ranks on the medical suite. It lowers AURC relative to SR by 17% on average and achieves the lowest AURC on four of the seven datasets. SCSF, which uses the same head with a correctness target, ranks joint second.

The advantage is not uniform:

- SAT, which derives confidence from a terminal abstention output, is almost as good on average and markedly more stable.
- SR remains better on Chest CT, and better at high coverage on several datasets.
- The ablation in Section 3.3.4 shows that the improvement does not depend on the auxiliary gradient reaching the backbone.

The evidence therefore supports learned confidence heads on intermediate features combined with two-view training. It does not support deep supervision of intermediate stages as the source of the gain.

**RQ2: Does intermediate selective supervision reshape the backbone so that its own softmax response ranks errors better?**

Not supported by the available evidence.

- On CIFAR-100, the MSP of the Original Baseline is better than that of plain cross-entropy training (77.79 vs. 81.41 × 10⁻³). However, a model whose head is detached from the backbone shows the same improvement (77.04). The model trained with two-view classification alone, without a head, shows it too (78.56).
- On CIFAR-10, the MSP of the Original Baseline is slightly worse than that of plain training (7.69 vs. 7.41).

The change in the backbone's own ranking is therefore attributable to two-view classification rather than to intermediate supervision. The softmax response of the Original Baseline backbone was not evaluated on the medical suite. [TODO: if the per-sample logits of the main suite are stored, compute the MSP-based AURC of the DualAug and SCSF checkpoints and add it here.]

**RQ3: Which components contribute, and do the results support the original design hypotheses?**

The contributing components are ranking by a learned head and two-view classification. The agreement target can replace the correctness target without loss. The deep-supervision gradient does not contribute.

The three variants were designed to strengthen the intermediate pathway in three ways: by separating its contribution (residual head), by controlling its update (decision-preserving head), or by preserving spatial detail (spatial head). They did not improve on the Original Baseline. Because they were also trained without two-view classification and ranked by saturated probabilities, these results do not cleanly test the individual design hypotheses (Section 3.3.3).

**RQ4: How consistent are the gains across modalities, and what are the trade-offs?**

The gains are consistent on fundus photography, brain MRI, light microscopy, and breast ultrasound. They are within noise on chest X-ray, and they reverse on dermoscopy and chest CT. They also depend strongly on the operating point: large at 50–80% coverage, absent or negative at 95% coverage.

The method's costs are:

- **Optimization stability.** Seed-to-seed variability is lower than that of SR on most datasets but about twice that of SAT.
- **Parameters.** The confidence head adds about 3.8 M parameters to the 23.5 M of ResNet-50 (+16%).
- **Inference compute.** The head adds roughly 3.8 M multiply–accumulates per image, under 0.1% of a ResNet-50 forward pass at 224 × 224. Inference remains a single forward pass.
- **Training compute.** Two-view classification doubles the backbone forward and backward passes per batch, which roughly doubles training cost. DualAug on COVID-QU-Ex was the longest run of the suite, at about two hours.

Layer choice was not varied on the medical suite, so its effect cannot be assessed here.

---

# 4. Conclusion and Future Work

## 4.1. Conclusion

This thesis studied selective classification for medical images and asked whether reliability-oriented supervision applied to intermediate network stages can produce better confidence estimates than methods that read confidence from the terminal layer.

The question was examined with seven methods on seven public medical datasets spanning seven imaging modalities. All methods shared a ResNet-50 backbone, data splits, and training budget. Each configuration was trained with three seeds and evaluated on the full test sets.

The main finding is positive but qualified. The Original Baseline (DualAug) trains the classifier on an image and its horizontal flip, and lets a head on pooled layer-3 and layer-4 features predict whether the two views agree. It achieved:

- the best average AURC rank among all methods (2.29);
- the lowest AURC on four of the seven datasets;
- AURC on average 17% below the softmax-response baseline.

Its largest improvements were on APTOS 2019 (AURC 41.76 vs. 60.16 × 10⁻³ for SR) and Brain Tumor MRI (5.56 vs. 12.62). These came with higher AUROC and nearly unchanged accuracy, showing that the improvement lies in the ranking of errors rather than in the classifier itself.

The advantage was not universal:

- SAT matched it closely on average (rank 2.86) with much lower variability across seeds.
- SR remained better on Chest CT, and SCSF on HAM10000.
- At 95% coverage, the operating point most relevant to clinical triage, the Original Baseline was not better than SR on any dataset, and it was clearly worse on three.

The thesis also identified what does and does not drive the improvement. A component analysis on CIFAR-10 and CIFAR-100 showed that the benefit comes from two-view training and from ranking by the learned agreement head. Detaching the head from the backbone, which removes the deep-supervision gradient, did not reduce performance. Thus the evidence does not support the hypothesis that intermediate supervision improves selective prediction by reshaping the backbone. The useful mechanism is better described as a learned confidence readout of intermediate features, trained on a self-supervised consistency target.

The three refined designs did not outperform the Original Baseline. These were the residual reliability head, decision-preserving auxiliary updates, and the decision-conditioned spatial head. They were competitive on the two largest datasets but failed on small or near-saturated ones, where their AUROC dropped to chance level. Their evaluation was confounded by single-view training and by ranking with saturated probabilities, so their design ideas remain open rather than refuted.

Finally, the study showed how sensitive selective-classification conclusions are to evaluation details:

- Ranking with saturated probabilities created artificial ties.
- An image-level split of HAM10000 leaked lesions between training and test data and inflated accuracy by about four points.
- A single training seed was insufficient: the ranking of the three variants changed once two more seeds were added.

## 4.2. Limitations

1. **Small number of seeds and small test sets.** Three training seeds allow only coarse statements about significance. On Breast Ultrasound (315 test images), Chest CT (288), and APTOS 2019 (366), a single image changes the error rate by roughly 0.3 percentage points, and several rankings are within seed variation.

2. **The variants were not evaluated under the final protocol.** The three designs of Section 2.6 were trained with an earlier code version, different seeds, single-view classification, and probability-based ranking. On HAM10000, their data split also differed. Their comparison with the Original Baseline is therefore indicative only.

3. **Component analysis only on natural images.** The ablation of the Original Baseline was performed with VGG16-BN on CIFAR-10 and CIFAR-100. It used two seeds for most variants, and one variant had only one run. Whether the same attribution holds for ResNet-50 on medical data has not been verified. The effect of reading intermediate rather than terminal features was not isolated, and layer choice was not varied.

4. **Representation-level claims were not tested on the final method.** Two analyses were not repeated for the evaluated models: the linear-probe evidence mentioned in Section 2.3, and the MSP of the Original Baseline backbone on the medical suite.

5. **Configuration sensitivity of some baselines.** SelectiveNet collapsed under its default configuration, and Deep Gamblers was evaluated with a single reward rule. Neither was tuned per dataset. Tuning was deliberately avoided for all methods, but these two baselines may be weaker here than in a tuned comparison.

6. **The agreement target and the augmentation are limited.** Agreement under horizontal flipping cannot flag errors on which both views agree. This is the likely cause of the weak high-coverage behaviour and of the loss on HAM10000. Horizontal flipping is also not label-preserving for all clinical tasks.

7. **Scope.** All datasets are public research benchmarks from single sources. Only one convolutional backbone was used on the medical suite. No experiment addresses distribution shift across sites or scanners, calibration, or use by clinicians. The results describe algorithmic behaviour, not clinical effectiveness.

## 4.3. Future work

1. **Retrain the three variants under the final protocol.** Use two-view classification, logit-based ranking, the lesion-grouped HAM10000 split, and the same seeds as the main suite. Add the controls described in Section 2.6: active versus detached features, fixed pooling, constant query, and projected versus unprojected updates. This is the most direct way to determine whether any of the three design ideas improves on the Original Baseline.

2. **Repeat the component analysis on the medical suite.** Include a head on terminal logits alone and different attachment layers. This would confirm whether the CIFAR attribution transfers to ResNet-50 and medical images, and would test whether intermediate features are needed at all.

3. **Improve high-coverage behaviour.** Possible directions include:
   - combining the agreement head with the softmax response, which was better at 95% coverage;
   - using richer consistency targets than a single horizontal flip, such as several augmentations or agreement in probability rather than in predicted class;
   - training objectives that weight the high-coverage end of the risk–coverage curve.

4. **Strengthen the statistical evidence.** Use more seeds, prioritising the small datasets, and report paired tests or confidence intervals. Evaluate on external test sets from different sources to measure robustness under distribution shift.

5. **Broaden the architectural scope.** Evaluate the approach on other backbones, including Vision Transformers, and on larger clinical datasets. Assess selective predictions in human-in-the-loop settings, where the usefulness of deferrals to clinicians can be measured directly.
