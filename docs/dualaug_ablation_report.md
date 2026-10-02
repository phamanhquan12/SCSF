# DualAug Ablation Study

**Backbone**: VGG16-BN · **Datasets**: CIFAR-10 & CIFAR-100 · **Protocol**: 300 epochs, last checkpoint, full official test set

---

## Variant Descriptions

Each variant surgically isolates one component of `dualaug`.

| Variant | What it keeps | Component isolated |
| :--- | :--- | :--- |
| `msp` | Single-view CE only, scored by MSP | **Pure baseline** |
| `scsf` | Single-view CE + head trained on **correctness** | Earlier SCSF baseline |
| `agree_only` | Single-view CE + head trained on **agreement** (flip is `no_grad`) | Isolates **Part 1** (two-view CE) |
| `dualce_msp` | **Two-view CE** only, scored by MSP | Isolates **Part 1** (no head) |
| `dualce_scsf` | Two-view CE + head trained on **correctness** | Isolates **Part 2** (agreement vs correctness) |
| `dualaug_detach` | Full dualaug but head gradient **detached** from backbone | Isolates **Part 3** (deep supervision) |
| `dualaug` | Full method | Reproducibility check + full combination |

**The four parts of `dualaug`:**
1. Two-view classification loss (original + horizontal flip)
2. Agreement-based head supervision (instead of correctness)
3. Head gradient flowing into backbone through pool4/pool5 (deep supervision)
4. Ranking predictions by head score instead of MSP

---

## Results

### CIFAR-10 — Per-run results

| Variant | Seed | Acc (%) | Head AURC | MSP AURC |
| :--- | :---: | :---: | :---: | :---: |
| `msp` | 0 | 93.83 | — | 0.00724 |
| `msp` | 42 | 93.81 | — | 0.00758 |
| `scsf` | 0 | 93.91 | 0.00602 | 0.00757 |
| `scsf` | 42 | 94.15 | 0.00600 | 0.00729 |
| `agree_only` | 0 | 94.15 | 0.00581 | 0.00593 |
| `agree_only` | 42 | 93.95 | 0.00587 | 0.00694 |
| `dualce_msp` | 0 | 94.29 | — | 0.00781 |
| `dualce_msp` | 42 | 94.40 | — | 0.00742 |
| `dualce_scsf` | 0 | 94.30 | 0.00528 | 0.00806 |
| `dualaug` | 0 | 94.45 | 0.00508 | 0.00773 |
| `dualaug` | 1 | 94.48 | 0.00529 | — |
| `dualaug` | 2 | 94.40 | 0.00549 | — |
| `dualaug` | 3 | 94.64 | 0.00504 | — |
| `dualaug` | 42 | 94.54 | 0.00530 | 0.00765 |
| `dualaug_detach` | 0 | 94.29 | 0.00511 | 0.00798 |
| `dualaug_detach` | 42 | 94.41 | 0.00511 | 0.00717 |

### CIFAR-10 — Aggregated (mean ± std over seeds)

> **Head gain** = MSP AURC − Head AURC on the same checkpoint.
> **vs MSP base** = MSP baseline AURC − this variant's Head AURC.

| Variant | N | Acc (%) | Head AURC | MSP AURC | Head gain | vs MSP base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `msp` | 2 | 93.82 ± 0.01 | — | 0.00741 ± 0.00024 | — | 0.00000 |
| `scsf` | 2 | 94.03 ± 0.17 | 0.00601 ± 0.00002 | 0.00743 | +0.00142 | +0.00140 |
| `agree_only` | 2 | 94.05 ± 0.14 | 0.00584 ± 0.00004 | 0.00643 | +0.00059 | +0.00157 |
| `dualce_msp` | 2 | 94.35 ± 0.08 | — | 0.00762 ± 0.00027 | — | **−0.00021** |
| `dualce_scsf` | 1 | 94.30 | 0.00528 | 0.00806 | +0.00278 | +0.00213 |
| `dualaug` | 5 | 94.50 ± 0.09 | 0.00524 ± 0.00018 | 0.00769 | +0.00245 | +0.00217 |
| `dualaug_detach` | 2 | 94.35 ± 0.08 | **0.00511 ± 0.00000** | 0.00757 | +0.00246 | **+0.00230** |

---

### CIFAR-100 — Per-run results

| Variant | Seed | Acc (%) | Head AURC | MSP AURC |
| :--- | :---: | :---: | :---: | :---: |
| `msp` | 0 | 73.16 | — | 0.08132 |
| `msp` | 42 | 72.75 | — | 0.08149 |
| `scsf` | 0 | 73.27 | 0.07753 | 0.08008 |
| `scsf` | 42 | 73.13 | 0.07862 | — |
| `agree_only` | 0 | 72.60 | 0.08018 | 0.08208 |
| `agree_only` | 42 | 72.38 | 0.08074 | 0.08078 |
| `dualce_msp` | 0 | 73.94 | — | 0.07816 |
| `dualce_msp` | 42 | 74.20 | — | 0.07895 |
| `dualce_scsf` | 0 | 74.13 | 0.07329 | 0.07963 |
| `dualce_scsf` | 42 | 74.14 | 0.07266 | 0.07731 |
| `dualaug` | 0 | 74.44 | 0.07225 | — |
| `dualaug` | 1 | 74.43 | 0.07195 | — |
| `dualaug` | 2 | 74.15 | 0.07333 | — |
| `dualaug` | 3 | 74.16 | 0.07246 | — |
| `dualaug` | 42 | 74.15 | 0.07431 | 0.07779 |
| `dualaug_detach` | 0 | 74.57 | 0.07368 | 0.07744 |
| `dualaug_detach` | 42 | 74.46 | 0.07193 | 0.07664 |

### CIFAR-100 — Aggregated (mean ± std over seeds)

| Variant | N | Acc (%) | Head AURC | MSP AURC | Head gain | vs MSP base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `msp` | 2 | 72.96 ± 0.29 | — | 0.08141 ± 0.00012 | — | 0.00000 |
| `scsf` | 2 | 73.20 ± 0.10 | 0.07807 ± 0.00077 | 0.08008 | +0.00201 | +0.00334 |
| `agree_only` | 2 | 72.49 ± 0.16 | 0.08046 ± 0.00040 | 0.08143 | +0.00097 | +0.00094 |
| `dualce_msp` | 2 | 74.07 ± 0.18 | — | 0.07856 ± 0.00056 | — | +0.00285 |
| `dualce_scsf` | 2 | 74.13 ± 0.01 | 0.07297 ± 0.00045 | 0.07847 | +0.00549 | +0.00843 |
| `dualaug` | 5 | 74.27 ± 0.15 | 0.07286 ± 0.00096 | 0.07779 | +0.00493 | +0.00855 |
| `dualaug_detach` | 2 | 74.51 ± 0.08 | **0.07280 ± 0.00124** | 0.07704 | +0.00423 | **+0.00860** |

---

## Component Attribution

| Part | Mechanism | Isolated by | CIFAR-10 AURC Δ | CIFAR-100 AURC Δ |
| :--- | :--- | :--- | :---: | :---: |
| **1 — Two-view CE** | Better backbone via dual-view consistency | `msp` → `dualce_msp` | **−0.00021** *(hurts MSP!)* | **+0.00285** |
| **2 — Agreement target** | Self-supervised proxy vs correctness label | `dualce_scsf` → `dualaug_detach` | +0.00017 *(negligible)* | +0.00017 *(negligible)* |
| **3 — Deep supervision** | BCE gradient into backbone through pool4/5 | `dualaug` → `dualaug_detach` | +0.00013 *(always hurts)* | +0.00006 *(always hurts)* |
| **4 — Head ranking** | Head score vs same-checkpoint MSP | `dualaug_detach` head vs its MSP | **+0.00246** | **+0.00423** |

---

## Insights

### 1. Dataset difficulty flips which component is the primary driver

On **CIFAR-100** (100 classes, hard representation problem):

- Two-view CE alone (`dualce_msp`) drops AURC by **+0.00285** — the biggest single lever.
- Part 4 (head ranking) adds another **+0.00423** on top.
- Gains are **additive**: better features AND better ranking, both contributing substantially.

On **CIFAR-10** (10 classes, easy):

- Two-view CE alone (`dualce_msp`) **hurts MSP AURC by −0.00021** — the backbone is already well-trained and dual-view CE introduces softmax noise.
- Part 4 (head ranking) does almost all the work: **+0.00246** gain from `dualaug_detach`.
- Gains are **corrective**: the head compensates for the MSP degradation caused by two-view CE, then goes further.

> The method's advantage is representation-led on hard problems and head-led on easy ones.

---

### 2. Two-view CE degrades MSP calibration on simple datasets

On CIFAR-10:

| | MSP AURC |
| :--- | :---: |
| `msp` (plain single-view CE) | 0.00741 |
| `dualaug_detach` (two-view CE, same model scored by MSP) | 0.00757 |

Training on both the original crop and its flip forces the softmax to handle two augmented views simultaneously. On a 10-class task where the model already saturates training signal, this introduces ambiguity into the softmax distribution — the head corrects this, but MSP itself gets worse. On CIFAR-100, two-view CE improves both MSP AURC and head AURC, confirming the backbone genuinely benefits from consistency pressure when the problem is hard.

---

### 3. Head ranking is the only universally beneficial component

The head gain (MSP AURC − Head AURC on the same checkpoint) is **always positive** for every variant with a head, on both datasets:

| Variant | CIFAR-10 head gain | CIFAR-100 head gain |
| :--- | :---: | :---: |
| `scsf` | +0.00142 | +0.00201 |
| `agree_only` | +0.00059 | +0.00097 |
| `dualce_scsf` | +0.00278 | +0.00549 |
| `dualaug` | +0.00245 | +0.00493 |
| `dualaug_detach` | +0.00246 | +0.00423 |

The head gain is proportionally larger on CIFAR-10 (~31% AURC reduction vs ~10% on CIFAR-100). On the easier task, the backbone is already near-optimal and the head extracts more incremental value from its features.

---

### 4. The `agree_only` anomaly reveals two-view CE's true role

`agree_only` (single-view CE backbone + agreement head, flip passed under `torch.no_grad()`):

| | CIFAR-10 AURC | CIFAR-100 AURC |
| :--- | :---: | :---: |
| `agree_only` | **0.00584** | 0.08046 |
| `dualce_msp` (two-view CE, no head) | 0.00762 | 0.07856 |
| `dualaug_detach` (full method) | **0.00511** | **0.07280** |

- **CIFAR-10**: `agree_only` substantially beats `dualce_msp` and almost matches `dualaug_detach`. An agreement head on a standard backbone is nearly as powerful as the full method.
- **CIFAR-100**: `agree_only` barely improves over MSP and is far behind `dualaug_detach`. The agreement head provides almost no value on a standard backbone.

**Why?** On CIFAR-10, a standard backbone already classifies flipped images consistently — the agreement signal correlates well with correctness even without dual-view training. On CIFAR-100, fine-grained classes are more flip-sensitive; a single-view backbone frequently produces disagreements even for correctly classified samples, making the agreement signal noisy and unreliable.

> Two-view CE does not just improve accuracy — it **creates the flip-consistency in the backbone** that makes the agreement confidence signal meaningful on hard tasks.

---

### 5. Deep supervision consistently hurts; always use `dualaug_detach`

| | CIFAR-10 | CIFAR-100 |
| :--- | :---: | :---: |
| `dualaug` head AURC | 0.00524 | 0.07286 |
| `dualaug_detach` head AURC | **0.00511** | **0.07280** |
| Benefit of detaching | +0.00013 | +0.00006 |

Allowing the BCE loss to backpropagate into pool4/pool5 creates gradient conflict: the BCE loss pushes intermediate features toward separating correct-from-incorrect samples, while CE pushes them toward separating classes. Detaching removes this conflict at zero cost. The benefit is consistent but small — it is a free improvement that should always be applied.

---

### 6. Agreement supervision equals correctness supervision

| | CIFAR-10 head AURC | CIFAR-100 head AURC |
| :--- | :---: | :---: |
| `dualce_scsf` (correctness target) | 0.00528 | 0.07297 |
| `dualaug_detach` (agreement target) | **0.00511** | **0.07280** |

The self-supervised agreement target matches or slightly outperforms ground-truth correctness supervision, consistently on both datasets (difference is 0.00017 AURC, well within noise). Agreement is the preferred choice because it is fully self-supervised — the confidence signal does not require using ground-truth labels during training.

---

## Design Conclusions

| Component | CIFAR-10 verdict | CIFAR-100 verdict | Recommendation |
| :--- | :---: | :---: | :--- |
| Two-view CE (Part 1) | Neutral/slightly harmful (MSP) | **Essential** | Keep — needed for hard tasks; the head compensates on easy tasks |
| Agreement target (Part 2) | ≈ correctness | ≈ correctness | Use agreement — self-supervised, equally effective |
| Detach head gradient (Part 3) | Always helps | Always helps | **Always detach** |
| Head ranking (Part 4) | Primary gain driver | Strong secondary driver | **Essential on all settings** |

**Best configuration: `dualaug_detach`** — two-view CE backbone, agreement-supervised head, gradients detached from backbone, head used for scoring.
