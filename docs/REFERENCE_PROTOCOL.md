# CIFAR reference protocol (CCL-SC, ICML 2024)

## Why this replaces the first live queue

The initial `backbone_transfer` queue used a cosine schedule, batch size 128,
the wrong CIFAR-10 standard deviation for the cited implementations, and a
non-official Deep Gamblers ranking score.  Its artifacts remain available for
engineering diagnostics but are **superseded for scientific comparison**.

The VGG16-BN passing track now uses `recipe=ccl_sc_reference`, grounded in
Appendix D of CCL-SC:

- 300 epochs for CIFAR-10 and CIFAR-100;
- minibatch 64;
- SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4;
- learning rate multiplied by 0.5 every 25 epochs;
- the official compact CIFAR VGG16-BN already implemented in this repository;
- five seeds: 13, 17, 23, 29, 31.

Dataset-specific method constants also follow the paper:

| Dataset | CCL-SC `(q, queue, weight, Es)` | DG `(reward, Es)` | SAT `(momentum, Es)` |
|---|---|---|---|
| CIFAR-10 | `(0.999, 300, 0.5, 150)` | `(2.2, 100)` | `(0.9, 0)` |
| CIFAR-100 | `(0.99, 3000, 1.0, 150)` | `(4.6, 200)` | `(0.9, 200)` |

`Es` is the initial/pretraining epoch count.  The CIFAR-100 DG and SAT values
are explicitly reported in CCL-SC Table 6.  CIFAR-10 DG follows the original
Deep Gamblers implementation and CCL-SC's published comparison.

## What may be reused from the paper

CCL-SC publishes five-seed selective-risk curves for CCL-SC, SAT+EM, SAT, DG,
and SR/MSP on both CIFAR datasets using this VGG protocol.  Those values may be
used as cited external references for the matching hard-coverage points.  The
paper explicitly omits SelectiveNet because prior work had already surpassed
it, so SelectiveNet is not in the new compute queue.

The published tables do **not** provide full-prefix AURC or failure AUROC.
Consequently:

- the proposed methods run first in `reference_ours_first.tsv`;
- no legacy baseline is required before obtaining the first scientific signal;
- `reference_metric_anchors.tsv` contains only CE/SR and CCL-SC and is deferred
  until AURC/AUROC comparison is needed;
- published hard-coverage numbers must never be presented as published AURC or
  AUROC, and cross-protocol numbers must not be used for paired significance.

## Deliberate deviations

We retain the preregistered leakage-free 45k/5k training/validation split and
the validation-only checkpoint selection rule.  CCL-SC tuned on 20% of the
training set and then retrained on all training examples.  Our deviation avoids
test leakage and permits identical early stopping for every method, but means
our runs are a **reference-protocol matched comparison**, not bitwise paper
reproduction.  Both the paper's published numbers and our matched anchors are
reported with provenance labels.

Sources:

- CCL-SC paper and supplement: https://proceedings.mlr.press/v235/wu24s.html
- Official CCL-SC code: https://github.com/lamda-bbo/CCL-SC
- Original Deep Gamblers code: https://github.com/Z-T-WANG/NIPS2019DeepGamblers
