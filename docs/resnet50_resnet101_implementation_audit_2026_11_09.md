**ResNet50 / ResNet101 implementation audit — 2026-11-09**

Reviewed checkout: `e9366ae87f2b838104c4c2fae4e0af88c160f663`. Implementation files were not modified. Pre-existing workspace changes were left intact.

**Follow-up: fairness reassessment for the user's seven methods**

The user subsequently narrowed the question to metric calculations, data loading, and labels for SAT, Deep Gamblers, SelectiveNet, MSP (`sr`), CCL-SC, SCSF, and DS-SCSF (`ds_scsf`). For that question, the findings below must be read with their triggers. My earlier conversational summary gave conditional failure cases too much weight relative to the normal medical-suite path.

The second pass found **no confirmed method-dependent split, label, or metric discrepancy in an unchanged, correctly prepared medical-suite run**. This is a conclusion about the code path, not certification of unavailable historical artifacts. Specifically:

- Executing the original command generator for seven methods × two architectures × three training seeds produced 42 commands with identical data roots, input size, split seed, and evaluation batch size. All supplied `--metrics-dir` and paper layout. Training seeds 0/1/2 do not change the medical split seed, whose default is 42.
- Repeated loads of a complete synthetic cache preserved its exact manifest and introduced no cross-split duplicate sources. Restarting a cache without completion metadata with the **same inputs and split seed** also introduced no overlap. Overlap was reproduced only after changing the split seed while rebuilding into the existing incomplete cache. `--force-preprocess` deleted the old cache and removed that overlap. Image decoding/resizing and numerical statistics were stubbed; original split, cache-orchestration, and filesystem-write functions were executed.
- A complete cache ignoring a new requested split seed is a real configuration/provenance bug. It does **not by itself** make methods using that same cache see different data, and it does not reassign images or create overlap.
- Independent ImageFolder instances do not randomly assign labels. Matching class-folder sets produce matching sorted class indices. The missing guard matters when a class directory is absent or naming differs in a way that changes semantic index alignment. No such mismatch in the actual eight datasets was demonstrated.
- The normal split resolver creates validation from training when train/test exist without validation. A synthetic fixture confirmed that official test identities remain unchanged. For generated splits, all per-class group counts from 6 through 1000 were checked and yielded train/val/test. The test-as-validation fallback therefore requires an abnormal missing validation directory or sufficiently small group counts; it is not the standard full-dataset path. Even if triggered, the published suite table still uses the last checkpoint, not the diagnostic best checkpoint; test monitoring would be compromised, but automatic best-test selection is not the reporting rule.
- Test/validation output overwriting is excluded by the normal suite's separate metric directories. Its aggregator explicitly reads `last/summary.csv`. That defect applies to direct invocations omitting `--metrics-dir`.
- The shared metric path evaluates disease-class logits for all methods; DG/SAT exclude the reservation output from class prediction, and SelectiveNet excludes its selector output. There is no method-specific alternative AURC/AUROC implementation in this path. Numerical formulas were re-inspected, not executed with NumPy/PyTorch in this environment.
- Patient/study overlap remains a dataset-provenance question. Missing patient-ID enforcement is not evidence that official splits overlap, and does not establish that one method was given different data. HAM10000 lesion grouping is implemented; the other archive splits need actual metadata to establish patient independence.

The practical comparison risk to prioritize is **mixing runs across changed data caches, preprocessing versions, or configurations**, because resume/aggregation do not enforce a dataset/configuration fingerprint. A fresh run with fixed settings and one shared intact cache does not trigger that scenario. Existing artifacts were unavailable, so neither its occurrence nor its absence in past runs was established.

The shared training implementation is broadly coherent, but there are confirmed defects in experiment identity, dataset-cache handling, evaluation output, and several ablation paths. These defects do not establish that all existing results are wrong. Their impact depends on the triggers below. Actual datasets, caches, checkpoints, and result directories were unavailable in this workspace, so this review cannot certify the historical result tables or measure their numerical impact.

**Scope and execution path**

The eight datasets are those in `datasets_r101.md`: COVID-QU-Ex, HAM10000, Malaria Microscopic, APTOS 2019, Brain Tumor MRI Masoud, Breast Ultrasound, Chest CT-Scan, and PCam.

The relevant path is `run_medical_*.sh` → `run_paper_medical_suite.py` → `run_experiment.py` → `scsf/{models,methods,datasets,medical_registry,metrics,reporting,run_status}.py`. The R101 comparison contains SR, SAT, DG, SelectiveNet, CCL-SC, SCSF, MG-SCSF, MP-DS, SE-DS, RO-DS, AG-DS, CM-DS, PD-SCSF, and RO-SCSF. I also checked TS-SCSF and the mechanism/detachment controls. The older CIFAR/CelebA training and multi-trial evaluation scripts are separate paths, not the evaluators used by this medical suite.

**Findings, ordered by impact**

1. **High — changed experiments can silently reuse or skip old results.**

   `load_last_checkpoint()` compares only dataset, method, architecture, variant, training seed, and total epochs. It does not compare learning rate, warm-up length, pretrained setting, split seed, preprocessing, smoke-test limits, most loss settings, or dataset contents. Separately, `--skip-existing` accepts completion by path without comparing the requested configuration. Variant names omit consequential settings: changing SCSF `--meta-loss` keeps the same variant; distinct TS reference variants with the same `l23` prefix also collide.

   A dependency-free fixture executing the original checkpoint-checking function accepted a checkpoint with a different learning rate, split seed, and warm-up configuration. Variant generation also reproduced both naming collisions. In the worst case, a model trained on an old split can be evaluated against a new split containing its former training examples, while the new arguments are written to completion metadata.

   Fix: use a canonical configuration fingerprint, including dataset/split/preprocessing fingerprints, for both completion and checkpoint reuse. Treat smoke runs as a distinct identity. Require explicit compatibility for evaluation-only scoring changes.

   Sources: [checkpoint checks](../run_experiment.py#L365), [variant identity](../run_paper_medical_suite.py#L136), [TS variant identity](../run_paper_medical_suite.py#L215), [skip logic](../run_paper_medical_suite.py#L551), [completion checks](../scsf/run_status.py#L30).

2. **High — dataset caches do not enforce the requested split or their own manifest.**

   Cache identity is only dataset slug plus input size. Although metadata records `split_seed`, loading never compares it with the requested seed. Existing metadata and a manifest are sufficient for reuse, without checking that cached images exist, that directory contents match the manifest, or that raw data changed. Training subsequently enumerates ImageFolder directories instead of loading the manifest's exact sample list.

   A fixture requesting split seed 99 accepted a cache marked seed 42, even with its manifest pointing to a missing cached image. There is a more serious recovery case: if preprocessing stops before metadata is written, restarting into that partial cache does not clear old images. If split assignments change, `_copy_and_resize()` writes the new destination and leaves the old split's copy. Executing its actual path/write logic with image-I/O stubs reproduced one source image remaining in both train and test. Concurrent first-time preprocessing from multiple queue workers is also unprotected and writes/unlinks the same files; that race was identified in source, not exercised.

   Fix: validate cache identity and contents; build in a fresh temporary directory and publish atomically under a dataset-level lock. Train from the validated manifest and reject cross-split duplicate identities. A changed split should produce a distinct immutable cache.

   Sources: [cache reuse](../scsf/medical_registry.py#L475), [cache writes](../scsf/medical_registry.py#L422), [ImageFolder loading](../scsf/datasets.py#L231).

3. **High, direct-run path — validation overwrites saved test metrics.**

   When `--metrics-dir` is omitted, `metrics_root_for()` returns the same directory for `last` and `val_last`. The runner evaluates test first and validation second, so validation overwrites `summary.csv`, `risk_coverage.csv`, `roc_curve.csv`, and `per_sample.csv`. Console test metrics and `test_curve.csv` still describe test, creating conflicting artifacts. The default legacy checkpoint path also omits training seed and variant, so direct runs can overwrite one another.

   Both path collisions were reproduced. The medical suite launcher supplies a metrics directory and paper layout, so its normal path avoids this particular overwrite. Direct ResNet experiments using the default arguments do not.

   Fix: always separate test/validation directories, and include seed and experiment identity in checkpoint paths.

   Sources: [path functions](../run_experiment.py#L454), [test followed by validation](../run_experiment.py#L830).

4. **High if triggered — missing validation silently substitutes the test set.**

   Preprocessing requires only train and test. `_stable_split()` can produce no validation examples for small class/group counts. If there is no `val/` directory, medical datasets have `val_size=None`, so `_split_test()` returns `(testset, testset)`. Epoch-by-epoch validation and best-checkpoint selection then use test data. The reported checkpoint remains the last epoch, but monitoring and development decisions have already accessed test labels.

   Reproduced with a small synthetic two-class split and the original fallback function. This does not demonstrate that any full eight-dataset cache currently lacks validation.

   Fix: require an independent nonempty validation split for this suite, or explicitly disable validation when it is unavailable. Never silently substitute test.

   Sources: [split allocation](../scsf/medical_registry.py#L378), [required splits](../scsf/medical_registry.py#L532), [test fallback](../scsf/datasets.py#L223).

5. **Medium — scorer configuration and saved score labels disagree with actual inference.**

   The suite forwards `--scsf-scorer` to MG-SCSF, PD-SCSF, RO-SCSF, and TS-SCSF, but their constructors hard-code `scorer="meta"`; MG/PD/RO forward methods explicitly return meta logits. Requesting an SR score ablation therefore changes plain SCSF while leaving these candidates on meta scoring. Independently, `write_rich_evaluation()` labels every method with `args.ds_primary_score`, whose default is `msp`. Thus ordinary SCSF, DG, SAT, SelectiveNet, MP-DS, and SE-DS can all receive misleading `primary_score=msp` metadata.

   Fix: expose the actual score identity from each method, honor supported scorer options, and reject incompatible options. The separately calculated `msp_*` metrics remain useful for same-checkpoint comparison; they do not make the mislabeled primary column correct.

   Sources: [forwarded scorer](../run_paper_medical_suite.py#L403), [method construction](../scsf/methods.py#L2744), [MG hard-coded scorer](../scsf/methods.py#L1122), [PD inference](../scsf/methods.py#L1918), [score metadata](../run_experiment.py#L481).

6. **Medium — CM-DS's random-mask ablation repeats the same masks every batch.**

   `_complement_scores()` creates a new generator and seeds it only from spatial size and batch size. For fixed shapes, each minibatch receives exactly the same mask at each batch position, across epochs and training seeds. It is a fixed collection of random masks, not fresh random masking. This limits the interpretation of targeted-versus-random erasure.

   Fix: use a persistent local generator seeded per run, advance it between batches, and preserve its state if training resume is implemented. If fixed masks are intentional, describe that control accurately.

   Source: [mask generator](../scsf/methods.py#L1752).

7. **Medium — TS-SCSF's target-matching solver assumes a false monotonicity condition.**

   Bisection assumes the mean true-class target decreases as temperature increases. True-class probability can instead increase for misclassified examples. The code neither verifies a bracket nor checks whether the requested mean is achievable. In a numerical counterexample with logits `[2, 0]`, label 1, and beta 0.5, the implemented target increases from approximately 0.006323 at T=0.25 to 0.244012 at T=1000. For a feasible goal of 0.1, the implemented bisection direction drives it to approximately 0.244012.

   This can invalidate a claim that the TS control matches PD's mean-target softness. It does not prove a large error on mostly correct training batches. The R101 comparison excludes TS; the issue concerns the TS ablation path.

   Fix: check attainability, search/bracket the actual scalar objective, and record achieved-minus-requested target error. Fail or explicitly flag unmatched batches.

   Source: [temperature search](../scsf/methods.py#L2039).

8. **Medium, AMP only — AG-DS bypasses the mixed-precision gradient safeguards.**

   AG-DS uses autocast when `--amp` is enabled but ignores the passed GradScaler. Both its ordinary/warm-up branch and aligned branch call unscaled backward/autograd gradients and directly step the optimizer. Other methods use scaled gradients and scaler-controlled steps. In the warm-up/unaligned branch, `training_loss()` is also invoked outside the autocast context on autocast-produced logits.

   This creates an avoidable precision/overflow difference between AG-DS and its comparators. No CUDA numerical failure was reproduced here. The default medical shell launcher does not enable AMP, so default FP32 runs do not trigger this finding.

   Fix: implement scaled gradient surgery with correct unscaling and finite-gradient checks, or explicitly require FP32 for AG-DS.

   Sources: [AG update](../scsf/methods.py#L1601), [standard scaled update](../run_experiment.py#L281).

9. **Medium — failed R101 queue jobs remain claimed and are skipped on restart.**

   Each task creates a permanent claim directory before training. Failure is logged, but the claim is never released or checked for a dead worker. Restarting the queue skips the task. The queue can print completion and return success despite failed jobs. A shell fixture using the actual queue function and a failing training stub attempted the job only once over two queue invocations.

   Fix: distinguish an active lease from successful completion; release failed claims and recover abandoned claims safely. Return a failing status for unfinished failed work. Use completion artifacts to skip successful runs.

   A related control issue: `count_complete()` discards architecture. If R50 and R101 share a run root, wrapper progress/`next` decisions can count R50 results toward R101 completion. Default distinct run IDs avoid this, but architecture overrides in ablation wrappers are not sufficient to make progress accounting correct.

   Sources: [queue](../run_medical_r101.sh#L168), [architecture-blind count](../scsf/run_status.py#L58).

10. **Medium — metric recomputation deletes auxiliary-score results.**

    `recompute_metrics.py` rebuilds each summary from a restricted set of identity fields plus primary and MSP metrics. It discards `primary_score` and all `aux_*` summary columns even when `per_sample.csv` contains `aux_confidence`. This affects RO/AG/CM/PD/RO-SCSF score comparisons after recomputation. The original function's file-writing flow reproduced the column loss with metric arithmetic stubbed out.

    Fix: recompute auxiliary metrics whenever their per-sample scores exist and retain meaningful metadata. Rebuild related curves/plots consistently when changing metric definitions.

    Source: [recompute](../recompute_metrics.py#L28).

**Dataset and split assessment**

The following describes the code's behavior, not verified counts or contents of the downloaded archives.

| Dataset | Implemented split/loading behavior | What remains to establish |
| --- | --- | --- |
| COVID-QU-Ex | Generic folder discovery; recognized train/val/test folders are retained; known mask directories and mask filenames excluded; same-class byte duplicates removed | Confirm the chosen dataset subtree, exact class names/counts, cross-source duplicates, and patient separation from the real manifest |
| HAM10000 | Root-level metadata supplies diagnosis and `lesion_id`; group-stratified 70/10/20 allocation when no predefined split exists | Lesion grouping is implemented and synthetic grouping passed; lesion identity is not patient identity. A nested metadata CSV would miss the special handler. Audit historical caches against this version |
| Malaria Microscopic | Generic folder discovery; absent official splits lead to image-level stratified allocation | No patient/slide grouping is extracted; related but nonidentical images can cross splits |
| APTOS 2019 | CSVs with `id_code` and `diagnosis` are matched to image stems; named labelled splits are recognized; if train/test are not both available, labelled rows are re-split | Confirm which labelled CSVs the archive supplies, label validity, image-stem uniqueness, and patient/eye grouping |
| Brain Tumor MRI Masoud | Recognized train/test folders preserved; when validation is missing, approximately 10% of training groups per class become validation and the rest remain training | No patient/study grouping is extracted; inspect real identities and duplicate scans |
| Breast Ultrasound | Mask filenames are excluded in current code; unsplit images are stratified by image | Verify that old mask-contaminated caches/results were actually replaced and assess related images across splits |
| Chest CT-Scan | Generic folder labels and recognized splits | Canonicalize labels across splits and inspect patient/study overlap; arbitrary folder names are not validated as the same class ontology |
| PCam | HDF5 image/label lengths checked; exported train/valid/test mapped directly to train/val/test; no content deduplication in this path | Verify archive counts/labels and original split provenance; the code trusts the supplied splits and does not enforce expected split sizes |

Additional shared weaknesses:

- **No shared class mapping is enforced.** Each split constructs its own ImageFolder class-to-index map. If train has classes A/B/C and validation only A/C, C can become index 1 instead of 2. Differently spelled class directories across splits are neither canonicalized nor checked. This is a conditional silent label-corruption risk; real mislabeling was not established without the archives. Sources: [folder loading](../scsf/datasets.py#L237), [class discovery](../scsf/medical_registry.py#L273).
- **A single unassigned sample can discard an official split.** `_resolve_splits()` re-splits all rows if any row has `split=None`. An unexpected image subtree or partially recognized archive can therefore change the protocol without an explicit error. Source: [split resolution](../scsf/medical_registry.py#L400).
- **Deduplication is limited.** Generic discovery hashes encoded file bytes within each class. It does not detect cross-label duplicates, re-encoded copies, or related images from one patient/study. HAM/APTOS/PCam use separate discovery paths. Source: [generic discovery](../scsf/medical_registry.py#L291).
- **Statistics and input geometry differ.** Mean/std calculation resizes cached images to a square; the loader uses aspect-preserving resize followed by center crop for evaluation and random crop for training. This is not test leakage, but the statistics do not describe the actual center-cropped input distribution. Sources: [statistics](../scsf/medical_registry.py#L454), [transforms](../scsf/datasets.py#L194).
- **Run-level provenance is incomplete.** Per-sample outputs store sequential row numbers, not stable source IDs; checkpoints do not bind to a split-manifest hash or class map. Group IDs are dropped from the cached manifest. This makes historical split and paired-result verification harder. Sources: [manifest fields](../scsf/medical_registry.py#L540), [prediction rows](../run_experiment.py#L506).

Patient-independent evaluation cannot be claimed solely from these loaders. That requires dataset-level identity evidence. Conversely, the absence of those checks does not prove patient overlap in an official split.

**Training, model, and metric behavior that appears consistent by inspection**

- R50/R101 both use the bottleneck's `conv3` channel dimensions. The feature stages and final classifier input are derived from the backbone rather than assuming ResNet18 dimensions. No R50/R101 channel-shape error was identified in source.
- The normal medical launchers use pretrained ResNet, 224-pixel inputs, batch 64, 100 epochs, warm-up 20, SGD 0.01 with milestones 40/70/90, and training seeds 0/1/2. Training seeds and split seed are separate.
- Final medical test evaluation uses the last model on `test_full`; the best-validation checkpoint is saved for analysis and is not substituted into the main result table. On valid medical caches with independent validation, `test_full` is the supplied test split.
- SCSF's calibration target and logit input are detached. Its pooled feature path remains live by design. MG/PD/RO-SCSF preserve a separate Adam calibrator optimizer and SGD for backbone/new branches.
- PD trains its depth heads during warm-up but scales their backbone gradient to zero; later it scales that gradient by the auxiliary coefficient. `detach=new` preserves the SCSF feature-gradient path while stopping the new branch's path. `detach=all` stops both auxiliary feature paths. These are different controls and should be named separately.
- MP/SE/RO auxiliary margins use the final classifier's predicted class, rather than substituting the auxiliary head's class prediction. Evaluation does not pass ground-truth labels to the model's forward method.
- Current exact AURC averages prefix risk over all samples and handles ties by expected cumulative errors. E-AURC uses a finite-sample oracle consistently. NAURC is not capped above 1; undefined NAURC/AUROC/FPR cases are NaN. The older defects documented in `docs/medical_day_implementation_review.md` are already addressed here.
- Correctness is the positive class for AUROC/FPR@95TPR. This is a confidence/failure-ranking statistic, not disease-detection AUROC. Accuracy and `error` are percentages; `risk`, AURC, E-AURC, AUROC, and FPR are fractions. The raw CSVs do not apply the thesis's AURC ×1000 display convention.
- Aggregation separates architecture/method/variant and uses sample standard deviation (`ddof=1`) across training seeds. Undefined metrics are excluded with valid-seed counts. These standard deviations measure training-seed variability, not patient-sampling uncertainty.

These are source-level checks, not a substitute for tensor-gradient, checkpoint-round-trip, or GPU tests.

**Ablation coverage and interpretation**

The checked-in default launchers do not define a complete R50/R101 × eight-dataset ablation matrix. R101 explicitly runs 14 full methods on eight datasets and excludes TS. R50 and the PD/round-two/round-three control wrappers default to the seven thesis datasets. Overrides can extend them, but progress counting must also distinguish architecture. No local result artifacts were available to determine which controls have actually finished.

Keep representation and score claims distinct: comparing a candidate's meta/aux score with same-checkpoint MSP tests the readout; full versus `detach=new` tests the new branch's backbone contribution; full versus `detach=all` changes the inherited SCSF gradient as well. PD `target=final` retains depth supervision and changes the target, while beta=0 alone is not a no-depth-supervision baseline. The code mostly reflects these distinctions; the scorer, CM-random, and TS defects above weaken specific controls.

RO/AG/CM compute auxiliary branches during evaluation even when MSP is primary, and PD/RO-SCSF compute diagnostic auxiliary scores. Prediction correctness is unaffected by their presence, but runtime measured from this diagnostic evaluation is not the latency of a stripped deployment model.

**Verification and limits**

Attempted `python3 -m unittest discover -p 'test_*.py'`: failed during discovery with nine import errors because `torch` is absent. The available Python is 3.9; NumPy, Pillow, h5py, a project virtual environment, raw data, checkpoints, and result directories are also absent. No training, real-image loading, CUDA/AMP, or tensor-gradient tests were run. This is an environment limitation, not nine failing numerical tests.

A temporary dependency-free harness extracted original functions with AST, provided explicit I/O/library stubs where needed, and reproduced checkpoint/config acceptance, variant collisions, cache-seed acceptance, missing-file acceptance, partial-cache duplicate destinations, missing-validation fallback, output-path collisions, failed-queue retry behavior, and recomputation column loss. It also checked grouped splitting and calculated the TS counterexample directly. All Python files passed AST parsing and all root shell scripts passed `bash -n`. Stubs did not validate PyTorch, NumPy, or image-decoding behavior.

The existing seven root test modules exercise the older experiment families; none directly imports the shared medical loader/model/method/metric pipeline. Dedicated coverage is needed for this research path. Dependency declarations also omit `h5py`, which PCam imports, and do not state the modern PyTorch/NumPy versions required by APIs such as `torch.amp` and `np.trapezoid`. A fresh install following only `requirements.txt` is not a complete reproducible PCam environment.

**Recommended order of work**

First fix experiment/cache identity and atomic preprocessing; then separate evaluation outputs and require independent validation plus a canonical class map. Correct scorer handling and the CM/TS controls before drawing ablation conclusions. Repair queue recovery and recomputation, and add focused synthetic medical-pipeline tests in a working PyTorch environment. Finally inspect the server's actual manifests, checkpoint arguments, and prediction files to identify affected historical runs. Configuration or training/split changes may require retraining; metadata and metric-only corrections may be recoverable from saved predictions/checkpoints. Do not assume either that every run needs retraining or that recomputation alone repairs data leakage.
