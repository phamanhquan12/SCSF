# Analysis of MG-SCSF, MP-DS, and SE-DS Medical Results

Date: 2026-10-19.

Primary source: [Ket_qua_3_phuong_phap_DS_mg_mp_se.pdf](Ket_qua_3_phuong_phap_DS_mg_mp_se.pdf), all six pages, including Tables 0-8. Design reference: [medical_deep_supervision_method_proposals.md](medical_deep_supervision_method_proposals.md), Sections 3-6 and 8-10.

This is a results analysis, not an implementation audit. No training or evaluation code was changed. All AURC values below use the PDF's **×10^-3 scale**, with lower values better. Differences and percentages are calculated from rounded published means; they are descriptive, not significance tests.

## 1. Main assessment

**MG-SCSF is the strongest candidate to investigate further, but these results do not yet establish a successful deep-supervision contribution.** It has the best AURC among the three new methods on five datasets and the highest reported mean correctness-detection AUROC among all 13 methods. However, its advantage over original SCSF is uneven, its BUSI result is substantially worse, and no detached control is reported.

**MP-DS has a dataset-dependent confidence problem.** Its prototype score substantially improves on its own classifier's MSP on APTOS and Masoud, but seriously damages ranking on COVID. It should not be described as uniformly unsuccessful: the evidence suggests a useful readout on some tasks and dangerous high-confidence mistakes on others.

**SE-DS currently has the weakest case as a selective-classification method.** Its evidence score is worse than its own MSP on five of seven datasets. Its classifier sometimes has useful accuracy or MSP performance, so the principal lesson is that useful supervised features do not automatically produce a good rejection score.

The most useful direction for later brainstorming is to separate three questions: does training improve the classifier, does the auxiliary score improve error ranking, and does the auxiliary gradient into the backbone cause either improvement?

## 2. What was tested, and what is still comparable

The PDF reports 63 completed runs: three methods, seven datasets, and seeds 0/1/2. It specifies pretrained ResNet-50, 224×224 images, batch size 64, 100 epochs, 20 epochs of CE warm-up, SGD learning rate 0.01 with milestones 40/70/90, and final-epoch checkpoints. New auxiliary weights ramp to 0.1 over five epochs. These are the intended shared training settings.

Four qualifications materially affect interpretation:

1. **The AURC implementations are mixed.** Page 1 explicitly says the new results use tie-aware AURC while the seven original methods have not been recomputed. Cross-generation differences and the 13-method ranking are provisional. This does not prove the new results are inflated: the direction and size depend on tied scores and their correctness. The within-checkpoint head/MSP comparisons are stronger evidence, assuming both columns used the same new evaluator. The PDF does not separately establish the evaluation version for all three earlier verification variants.
2. **MP-DS and SE-DS differ from the written proposal.** The proposal attaches their auxiliary losses to `layer3` only. The PDF describes MP-DS using pooled `layer3 || layer4`, and SE-DS combining `layer3` with upsampled `layer4`. With attached features, their auxiliary losses can therefore also update `layer4`. This remains deep supervision, but tests a different design with more terminal information and different gradient paths. It neither validates nor conclusively rejects the original layer3-only versions. The report does not fully specify spatial fusion details.
3. **All three are `detach-none`.** No Full-versus-Detached comparison is available, so the requested 50-70% DS attribution cannot be calculated. Replacing the inference score by MSP is a score ablation, not a gradient ablation.
4. **The experimental revision is unavailable locally.** The PDF identifies commit `52162a1`; this checkout is at `06a8e2c`, and the reported commit is not available as a local Git object. The local metric function still sorts tied scores directly. No raw result directory is available here. Consequently, optimizer, split, score, and gradient correctness at the reported revision cannot be certified from this analysis.

Matching headline hyperparameters is necessary, but actual comparison also requires identical split/sample manifests, preprocessing, label mapping, checkpoint rule, metric version, and seed handling. Changing auxiliary architectures need not make an experiment unfair; it must be documented and isolated when explaining why a method works.

## 3. Comparison with the old methods

### Overall position

These are the reported macro averages from Table 0. Rank is averaged across the seven datasets within the same 13-method comparison pool.

| Method | Mean AURC rank | Mean AUROC | Mean accuracy (%) |
|---|---:|---:|---:|
| DualAug | 4.14 | 0.866 | 91.77 |
| Verify-Spatial | 4.43 | 0.847 | 91.01 |
| MG-SCSF | 5.00 | 0.870 | 91.01 |
| SAT | 5.00 | 0.859 | 91.88 |
| SCSF | 5.29 | 0.855 | 91.94 |
| Verify-Global | 6.14 | 0.850 | 90.81 |
| SR | 6.71 | 0.854 | 92.04 |
| DualAug-Correctness | 6.86 | 0.841 | 90.60 |
| MP-DS | 8.29 | 0.855 | 90.92 |
| CCL-SC | 8.29 | 0.845 | 92.02 |
| SE-DS | 8.43 | 0.842 | 91.15 |
| Deep Gamblers | 10.00 | 0.807 | 92.06 |
| SelectiveNet | 12.43 | 0.573 | 91.77 |

MG-SCSF's rank is competitive, but its accuracy is 0.93 percentage points below original SCSF. MP-DS matches CCL-SC's mean rank while losing 1.10 accuracy points. SE-DS is below both despite having the highest mean accuracy of the three new DS methods. Beating Deep Gamblers or SelectiveNet is insufficient when SR, SCSF, SAT, and DualAug are stronger on this benchmark.

The historical SCSF rank of approximately 3.5 discussed earlier cannot be directly compared with 5.29 here: adding competitors changes ranks even if every existing method's results remain identical.

### Dataset-level comparison

“Best original” is the lowest reported AURC among the original seven methods, excluding the three earlier verification variants. Its identity is selected retrospectively for comparison, not a deployable dataset-specific method.

| Dataset | SR | SCSF | DualAug | Best original | MG-SCSF | MP-DS | SE-DS |
|---|---:|---:|---:|---|---:|---:|---:|
| COVID | 5.43 | 5.87 | 4.64 | SAT: 4.31 | 5.23 | 11.12 | 5.68 |
| HAM | 36.27 | 34.39 | 39.32 | SCSF: 34.39 | 35.57 | 38.45 | 42.82 |
| Malaria | 4.28 | 3.34 | 3.25 | DualAug: 3.25 | 3.38 | 5.09 | 4.13 |
| APTOS | 60.16 | 49.82 | 41.76 | DualAug: 41.76 | 45.38 | 42.84 | 54.99 |
| Masoud MRI | 12.62 | 7.09 | 5.56 | DualAug: 5.56 | 5.25 | 6.08 | 15.33 |
| BUSI | 24.69 | 29.26 | 20.41 | DualAug: 20.41 | 51.69 | 64.60 | 54.21 |
| Chest CT | 18.43 | 19.65 | 26.99 | SR: 18.43 | 18.66 | 21.01 | 18.29 |

The only nominal wins over the best original method are MG-SCSF on Masoud and SE-DS on Chest CT. Their absolute improvements are only 0.31 and 0.14, respectively. These require matched recomputation and paired-seed analysis before being treated as wins. Verify-Spatial already achieves 17.71 on Chest CT, better than SE-DS.

For another view, count datasets on which each new method has a lower reported mean AURC:

| New method | vs SR | vs SCSF | vs DualAug | vs SAT | vs CCL-SC | vs Verify-Spatial |
|---|---:|---:|---:|---:|---:|---:|
| MG-SCSF | 5/7 | 4/7 | 3/7 | 3/7 | 5/7 | 1/7 |
| MP-DS | 2/7 | 2/7 | 2/7 | 3/7 | 5/7 | 1/7 |
| SE-DS | 3/7 | 2/7 | 1/7 | 1/7 | 3/7 | 1/7 |

These are descriptive counts, not significance tests. MG-SCSF and SAT share a mean rank even though SAT beats MG on four datasets. Similarly, Verify-Spatial beats all three DS candidates on six datasets; its major weakness is Masoud, where it scores 16.33 versus MG's 5.25 and MP's 6.08. Mean rank alone conceals these different failure profiles.

### What the earlier verification methods add to the diagnosis

Verify-Global remains strongest on APTOS at 37.17, ahead of MP's 42.84. Verify-Spatial remains strong on HAM, Malaria, and Chest CT. MG-SCSF beats Verify-Global and DualAug-Correctness on five datasets each, but does not generally outperform Verify-Spatial.

Most importantly, **all six newer methods have poor BUSI AURC**: Verify-Spatial 50.25, MG 51.69, SE 54.21, DualAug-Correctness 60.72, MP 64.60, and Verify-Global 72.00. The stronger original methods are between 20.41 and 29.26; even Deep Gamblers is 41.90. This shared regression deserves a common protocol and training investigation before attributing it to any one new DS mechanism. Small sample size, different predictions, preprocessing, or evaluation changes are possibilities; the PDF does not identify the cause.

## 4. Separate the classifier from the confidence score

Table 1 allows a useful controlled comparison: keep a trained checkpoint and its predictions fixed, and change only the confidence score. Define the readout penalty as `AURC(method score) - AURC(same-checkpoint MSP)`. Negative values favor the proposed score.

| Dataset | MG readout penalty | MP readout penalty | SE readout penalty |
|---|---:|---:|---:|
| COVID | -0.50 | +5.66 | +0.41 |
| HAM | +0.07 | +3.40 | +7.86 |
| Malaria | -0.13 | +0.96 | +0.50 |
| APTOS | -0.97 | -7.51 | +6.40 |
| Masoud | -1.23 | -8.56 | -1.75 |
| BUSI | -9.77 | -14.09 | -1.46 |
| Chest CT | +0.39 | +3.66 | +2.57 |

MG's TCP score wins on 5/7 datasets, MP's prototype margin on 3/7, and SE's spatial margin on 2/7. A small difference such as MG's HAM penalty of 0.07 has no established practical or statistical importance.

The unweighted mean AURCs are MG 23.59 versus MSP 25.33, MP 27.03 versus MSP 29.38, and SE 27.92 versus MSP 25.85. MP improves the raw mean despite losing on most datasets because its gains on difficult tasks are large. This illustrates why macro AURC, mean rank, and win counts answer different questions. These averages weight datasets equally, not individual patients.

An algebraic decomposition is useful for later work:

`AURC(new head) - AURC(old CE/SR) = [AURC(new MSP) - AURC(old CE/SR)] + [AURC(new head) - AURC(new MSP)]`.

The second term isolates score choice within a checkpoint. The first combines changed predictions and changed MSP ranking under the training recipe; it is not proof of representation quality or DS causality. Here, comparisons involving old SR also retain the metric-version limitation.

## 5. MG-SCSF: credible candidate, unresolved mechanism

MG improves reported AURC over original SCSF on COVID, APTOS, Masoud, and Chest CT. Its differences from SCSF are -0.64, +1.18, +0.04, -4.44, -1.84, +22.43, and -0.99 in dataset order. The BUSI regression is much larger than any individual gain.

Consequently, MG has a slightly better mean rank than SCSF, but a worse unweighted mean AURC: **23.59 versus 21.35**. Excluding BUSI reverses that average comparison, but excluding it from the claimed benchmark would be inappropriate. The point is to diagnose the dataset that changes the conclusion.

MG's reported AUROC exceeds SCSF's on every dataset, including BUSI: 0.806 versus 0.770. Yet BUSI AURC deteriorates from 29.26 to 51.69. This is not a logical contradiction. AUROC measures pairwise discrimination between correct and incorrect predictions; AURC depends on both the total error rate and where errors appear in the retained confidence prefixes. The PDF lacks original SCSF's dataset-level accuracy here, so the relative contribution of these effects cannot be quantified from this source alone.

The strongest positive evidence is Masoud: AURC 5.25 ± 1.52, AUROC 0.920 ± 0.031, and a 19.0% AURC reduction relative to its own MSP. However, the nominal 0.31 advantage over DualAug is much smaller than MG's seed standard deviation and is not a demonstrated significant improvement.

The central mechanism question remains unanswered: did margin supervision help, or is the retained SCSF TCP head responsible for most useful behavior? MG needs an original-SCSF comparison under identical metrics, a control detaching only the margin branch, and an all-auxiliary-detached control. The first isolates the new margin contribution; the second distinguishes that contribution from inherited TCP supervision.

Potential explanations to inspect, not established causes, are early exhaustion of the hinge signal, conflicting TCP/margin gradients, and persistent pressure on ambiguous or mislabeled examples. The useful diagnostics are active-hinge fraction by class and epoch, margin distributions, and separate auxiliary gradient norms.

## 6. MP-DS: useful evidence on some tasks, a dangerous confidence tail on others

COVID is the clearest failure: prototype AURC is **11.12 versus MSP 5.46**, a 103.7% increase, while accuracy remains 96.47%. Its risk decreases from 3.53% at full coverage to 0.79% at 70%, then rises to 0.87%, 1.05%, 1.09%, and 1.51% at 50%, 30%, 20%, and 10% coverage. Increasing selectivity starts retaining a more error-prone subset.

At 10% of 7,954 test images, 1.51% risk corresponds to roughly 12 errors per seed on average, subject to rounding and the coverage implementation. This is more substantial than a single error in a tiny test set. The repeated pattern and 0.25-point risk standard deviation suggest a recurring high-confidence failure, although raw predictions are needed to identify the cases.

This directly implicates the prototype readout relative to the same checkpoint's MSP. Possible mechanisms include a class-specific score offset, an incorrectly confident prototype, shared classifier/prototype mistakes, or uneven prototype utilization. The report does **not** establish prototype collapse. Examine top-ranked incorrect cases, their classes and selected prototypes, occupancy, and per-class score distributions before changing the method.

Conversely, MP helps considerably on APTOS (50.35 MSP to 42.84, -14.9%) and Masoud (14.64 to 6.08, -58.5%). On APTOS it has worse accuracy than SE, 83.70% versus 84.24%, but much better AURC, 42.84 versus 54.99. Its error ordering has value even when classification accuracy does not win.

MP therefore provides evidence for **task-dependent auxiliary evidence**, not for a universally reliable prototype margin. It still trails DualAug and Verify-Global on APTOS, and MG on Masoud. A matched one-prototype control is required before attributing any advantage to multiple class modes.

## 7. SE-DS: the classifier and its spatial confidence should be judged separately

SE's strongest warning is consistent readout harm: relative to MSP, spatial confidence increases AURC by 22.5% on HAM, 13.2% on APTOS, and 16.3% on Chest CT. Even its apparent Chest CT success, 18.29, is inferior to **15.72 from its own MSP**. Its MSP on that dataset is nominally better than every designated score listed in Table 0, although this is a retrospectively noticed diagnostic result, not a preselected benchmark submission.

SE also has the highest accuracy among the three new methods on COVID, Malaria, APTOS, and Chest CT. These facts suggest that the spatial auxiliary objective may be more useful as a training signal than its raw evidence margin is as an inference score. They do not yet show that the spatial loss caused the classifier improvement: a detached control and matched CE comparison are still missing.

Masoud has AURC **15.33 ± 12.97**, despite accuracy of **95.73 ± 0.13%**. Stable accuracy with highly variable AURC points toward instability in which errors receive high confidence. It does not prove one specific bad seed, because individual seed results are absent. MSP is also poor there at 17.08, so blaming only the spatial inference score would be incomplete.

Possible explanations include selecting spurious regions, unbounded evidence magnitude, class-dependent margin scales, and reliance on the added layer4 features. Image-level supervision does not establish that selected regions are lesions. These explanations need evidence maps and error-conditioned diagnostics. The intended top-k mechanism requires comparison with identical features and global average pooling; changing branch depth simultaneously would confound that test.

## 8. Coverage behavior, sample size, and metric interpretation

Low-coverage risk deserves attention because the intended rejected/retained decision depends on it. For ordinary discrete AURC without ties, an error at confidence rank `j` contributes `(1/N) * sum(1/k, k=j..N)`. An early, highly confident error therefore affects many coverage prefixes. Tie handling changes how equal-score samples are treated, but not the underlying concern about confidently ranked errors.

MG shows a mostly improving coverage profile outside BUSI. MP has the strong COVID reversal and smaller reversals on Malaria and Chest CT. SE has reversals on APTOS, Masoud, and Chest CT. These are descriptive observations; finite-sample risk need not be monotone even for a useful selector.

BUSI has only 156 test images, Chest CT 288, and APTOS 366. At approximately 10% coverage, one error changes risk by roughly 6.25, 3.45, and 2.70 percentage points, respectively. Zero reported errors at low coverage therefore do not establish safety or generalization. Three training seeds reuse the same test set; they do not create three independent clinical cohorts.

Reported standard deviations measure training-seed variation, not uncertainty from patient sampling. For paired method comparisons, retain seed-level differences and shared image identifiers; for dataset uncertainty, use a suitable paired bootstrap, clustered by patient when those identifiers exist. Three seeds alone give limited precision. A displayed AUROC standard deviation of 0.000 is rounded, not proof of zero variation.

AUROC, AURC, and FPR@95TPR need not agree. For example, MP has better BUSI FPR@95TPR than MG (0.562 versus 0.686) but worse AURC. Confirm the positive-label convention before interpreting this operating point; 95% TPR is not 95% coverage. Do not select whichever metric happens to favor a method after seeing results.

## 9. Implications for the next method and for DC-DS

The three experiments sharpen the design problem:

- **Retaining terminal evidence is a reasonable hypothesis.** MP and SE sometimes discard useful MSP ordering. DC-DS starts from terminal logits and learns corrections, which directly addresses that failure mode. Its zero-correction state preserves MSP ordering, but training can still make it worse; no performance advantage follows automatically.
- **Evidence strength and error detection are different objectives.** Auxiliary class CE can improve discrimination while a raw class margin remains unreliable for ranking the final classifier's mistakes. Later methods should be evaluated on their support for the actual returned prediction, especially where heads agree on an incorrect label.
- **Shared errors matter as much as disagreement.** Rejecting disagreement alone misses confident errors made by both branches. Inspect jointly wrong examples and harmful disagreement with correct final predictions.
- **The training contribution must survive a matched ablation.** A powerful head that compensates for a weak backbone is insufficient for the thesis's DS claim. Conversely, a useful training loss can remain valuable even if MSP is the better readout, provided active-versus-detached training demonstrates the contribution.
- **A common BUSI regression should be understood before adding complexity.** A new architecture should not be expected to repair a potentially shared protocol or optimization issue by itself.

For DC-DS, record Full and Detached results with both corrected confidence and MSP, correction norms, wrong-to-correct and correct-to-wrong auxiliary changes, and gradient diagnostics. The local DC-DS implementation currently inherits the older local evaluator, so comparisons against this PDF require harmonizing the evaluation revision first. This report identifies that requirement; it makes no code changes.

Do not choose a different winning score for each test dataset retrospectively. Same-checkpoint MSP is a diagnostic; any revised primary score must be selected using validation and explicitly named as a new experimental configuration.

## 10. Recommended evidence to collect before further brainstorming

Prioritize work that can use existing artifacts before more full training runs:

1. **Recompute all 13 methods with one evaluator.** Preserve predictions and checkpoints; use the same tie policy and sample manifest. Rebuild AURC, E-AURC, NAURC, coverage tables, and ranks. Audit AUROC/FPR conventions as well. Re-evaluation is not a newly fitted post-hoc method and usually does not require retraining.
2. **Recover the exact experimental revision and seed-level artifacts.** Collect resolved arguments, label maps, split hashes, per-sample predictions, validation results, and the actual multi-depth implementations. This permits checks that the PDF alone cannot support.
3. **Inspect four high-information failures.** MP/COVID isolates readout harm; SE/Masoud isolates ranking instability despite stable accuracy; all methods/BUSI investigates the shared regression; SE/Chest CT distinguishes a useful classifier from a harmful auxiliary score.
4. **Prioritize MG for causal DS testing.** First compare it with original SCSF under identical evaluation. Then detach only the added margin branch while preserving TCP training; separately detach all auxiliary feature paths. Keep training settings and inference heads matched.
5. **Keep MP as a targeted mechanism candidate.** APTOS and Masoud show useful auxiliary evidence, but any revision must also explain the COVID failure. Use the existing one-prototype control to test the multiple-mode hypothesis rather than assuming it.
6. **Pause broad promotion of SE's current inference score.** Investigate whether its training objective helps under MSP, then use matched pooling and detachment controls if that signal persists.

The displayed test results can guide diagnosis, but repeated method selection on them makes those test sets part of development history. Use existing validation splits for subsequent decisions and disclose that history in the thesis. None of these results yet establishes a 50-70% DS contribution or SOTA. They do provide a concrete direction: preserve the useful classifier evidence, diagnose confident errors, and isolate whether intermediate supervision improves the backbone rather than merely adding a different score.
