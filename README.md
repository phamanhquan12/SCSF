# dualaug: Selective Classification by Learning View Agreement

**dualaug** is a selective classifier: it predicts a class and a confidence
score, and abstains on the lowest-scoring inputs. It trains the classifier on
two views of each image (the original and its horizontal flip) and trains a
confidence head to predict whether the classifier gives the **same answer on
both views**. It uses no contrastive loss, no reservation class, and no
post-hoc score tuning. At test time it needs a single forward pass.

On CIFAR-100 (5 seeds) and CIFAR-10 (5 seeds) it beats CCL-SC on
accuracy and on selective risk at 95%, 90% and 80% coverage.

## Method

For an image \(x\) with label \(y\), let \(\tilde{x}\) be its horizontal flip
and \(f\) the classifier. The confidence head \(h\) is the SCSF head: a 5-layer
MLP over the two last pooled feature maps and the (stop-gradient) logits,

\[
s(x) = h\big(\text{pool4}(x),\ \text{pool5}(x),\ \text{sg}[f(x)]\big) \in \mathbb{R}.
\]

**Two-view classification loss**

\[
\mathcal{L}_{\text{CE}} = \tfrac{1}{2}\Big[\text{CE}\big(f(x), y\big) + \text{CE}\big(f(\tilde{x}), y\big)\Big]
\]

**Agreement loss.** The head target is view agreement, not correctness:

\[
a(x) = \mathbb{1}\big[\arg\max f(x) = \arg\max f(\tilde{x})\big], \qquad
\mathcal{L}_{\text{BCE}} = \text{BCE}\big(\sigma(s(x)),\ a(x)\big)
\]

\(a\) is computed without gradient. Gradients from \(\mathcal{L}_{\text{BCE}}\)
reach the backbone through the pooled features but not through the logits.

**Total loss**

\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_t\, \mathcal{L}_{\text{BCE}}
\]

with \(\lambda_t = 0\) during warm-up, then cosine decay from 1 to
\(10^{-4}\) over the remaining epochs:

\[
\lambda_t = \lambda_{\min} + \tfrac{1}{2}(1-\lambda_{\min})\Big(1+\cos\big(\pi\,\tfrac{t-E_w}{E-E_w}\big)\Big).
\]

**Inference.** One forward pass on the unflipped image. Predict
\(\arg\max f(x)\); accept inputs in decreasing order of \(s(x)\) until the
target coverage is reached.

Implementation: `variant == "dualaug"` in `train_epoch` of
`train_fresh_sc.py` (CIFAR) and `train_dualaug_celeba.py` (CelebA).

## Experimental protocol

The protocol follows CCL-SC so that its published numbers can be used
directly.

| | CIFAR-10 / CIFAR-100 | CelebA (Attractive) |
|---|---|---|
| Backbone | VGG16-BN | ResNet-18 |
| Optimizer | SGD, lr 0.1, momentum 0.9, wd 5e-4, batch 128 | Adam, 1e-5 backbone / 1e-3 head |
| Epochs (warm-up \(E_w\)) | 300 (100) | 50 (1) |
| Checkpoint | last epoch | best validation accuracy |
| Test set | official 10k | official test (19,962) |

Metrics: accuracy, AURC, and selective risk (error rate, %) at 95/90/80/10%
coverage; lower is better for everything except accuracy. CCL-SC numbers are
the 5-seed means reported in the CCL-SC paper.

## Results

### CIFAR-100 (5 seeds: 42, 0, 1, 2, 3)

| Method | Acc | AURC | Risk@95 | Risk@90 | Risk@80 | Risk@10 |
|---|---:|---:|---:|---:|---:|---:|
| CCL-SC | 73.45 | — | 23.54 ± 0.15 | 20.97 ± 0.20 | 16.07 ± 0.15 | 0.36 ± 0.08 |
| **dualaug** | **74.27 ± 0.15** | 0.0726 | **23.15 ± 0.14** | **20.64 ± 0.16** | **15.38 ± 0.17** | **0.30 ± 0.07** |

<details>
<summary>Per-seed results</summary>

| Seed | Acc | AURC | Risk@95 | Risk@90 | Risk@80 | Risk@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 74.15 | 0.0730 | 23.04 | 20.61 | 15.40 | 0.40 |
| 0 | 74.44 | 0.0722 | 23.23 | 20.71 | 15.38 | 0.30 |
| 1 | 74.43 | 0.0719 | 22.96 | 20.38 | 15.15 | 0.30 |
| 2 | 74.15 | 0.0733 | 23.28 | 20.78 | 15.62 | 0.20 |
| 3 | 74.16 | 0.0725 | 23.25 | 20.72 | 15.34 | 0.30 |

</details>

### CIFAR-10 (5 seeds: 42, 0, 1, 2, 3)

| Method | Acc | AURC | Risk@95 | Risk@90 | Risk@80 | Risk@10 |
|---|---:|---:|---:|---:|---:|---:|
| CCL-SC | 94.03 | — | 3.56 ± 0.06 | 2.01 ± 0.07 | 0.69 ± 0.08 | — |
| dualaug seed 42 | 94.61 | 0.00535 | 3.25 | 1.78 | 0.63 | 0.00 |
| dualaug seed 0 | 94.67 | 0.00506 | 3.19 | 1.66 | 0.54 | 0.00 |
| dualaug seed 1 | 94.48 | 0.00529 | 3.34 | 1.73 | 0.55 | 0.00 |
| dualaug seed 2 | 94.40 | 0.00549 | 3.35 | 1.88 | 0.65 | 0.00 |
| dualaug seed 3 | 94.64 | 0.00504 | 3.19 | 1.80 | 0.54 | 0.00 |
| **dualaug mean** | **94.56 ± 0.12** | 0.00525 | **3.26 ± 0.08** | **1.77 ± 0.08** | **0.58 ± 0.05** | 0.00 |

## Discussion and limitations

- **Accuracy vs. ranking confound.** The two-view CE loss raises accuracy by
  itself (+0.8 on CIFAR-100, +0.5 on CIFAR-10 versus CCL-SC). A selective
  classifier with fewer errors has lower risk at every coverage, so part of
  the gain likely comes from accuracy rather than from the agreement target.
  Separating the two needs ablations: two-view CE with a correctness-trained
  head, two-view CE with softmax-response scoring, and single-view CE with the
  agreement head.
- **Agreement is not correctness.** \(a(x)=1\) also when both views agree on
  the same wrong class, so the head can be confidently wrong on consistent
  errors.
- CIFAR-10 is near saturation, so absolute differences are small; CIFAR-100
  is the most informative benchmark here.

## Reproducing

```bash
pip install -r requirements.txt
```

Data is read from `../data` relative to this directory (CIFAR is
auto-downloaded by torchvision; CelebA via `python prepare_celeba.py`).

```bash
# CIFAR-100, one seed
bash run_cifar100_fresh_sc.sh dualaug 42 0      # variant seed gpu

# CIFAR-10, seeds 42,0,1,2,3 sequentially (skips seeds that have results.json)
bash run_cifar10_dualaug_queue.sh 0

# CelebA, seeds 42,0,1,2,3 sequentially
bash run_dualaug_celeba_queue.sh 0
```

Or directly:

```bash
python train_fresh_sc.py --variant dualaug -d cifar100 \
    --epochs 300 --pretrain 100 --batch-size 128 --seed 42 \
    --output-dir ./save/fresh_dualaug_cifar100_seed42

python train_dualaug_celeba.py --seed 42 --gpu 0 \
    --output-dir ./save/fresh_dualaug_celeba_seed42
```

Each run writes `config.json`, `history.jsonl` (per-epoch train/validation
metrics), checkpoints, `results.json` (config plus final test metrics) and
`test_metrics.json` to its output directory.

---

# Legacy: SCSF (Selective Classification with Supervised Features)

The sections below document the original SCSF code and earlier experimental
branches in this repository.

## Overview

SCSF replaces the reservation-class approach (e.g., Deep Gamblers' C+1 neuron) with a post-hoc **MetaCalibrator** that predicts True Class Probability (TCP) from intermediate backbone features. The backbone trains normally with cross-entropy; the MetaCalibrator learns to score confidence using supervised features from two late pooling layers plus the logit vector.

Key design choices:
- **No reservation neuron** — standard C-class output, no architecture modification
- **Post-hoc MetaCalibrator** — gradients are detached from the backbone; only the MLP is trained on the meta-loss
- **Cosine-decay meta-weight** — λ decays from 1.0 → 1e-4 over the joint phase, no RL or learned weighting needed
- **m=2 intermediate layers** — pool4 + pool5 (VGG16-BN) or layer3 + layer4 (ResNet-18)

## Requirements

```
torch >= 1.10
torchvision
numpy
```

## Usage

### DTR-SCSF experiment

`train_dtr_scsf.py` implements the four-state depth-transition prototype:

```bash
python train_dtr_scsf.py -d cifar10 \
    --epochs 300 --pretrain 100 \
    --probe-weight 0.3 --transition-weight 1.0 --repair-weight 0.5 \
    --temperature 2.0 --seed 42
```

For a quick pipeline smoke test, add
`--epochs 2 --pretrain 1 --limit-train-batches 2`. Checkpoints are selected using validation AURC; the
held-out test split is evaluated only after training.

### R3-SCSF experiment

`train_r3_scsf.py` routes high-impact errors between confidence ranking and
additional classifier repair using a detached two-view virtual update:

```bash
python train_r3_scsf.py -d cifar10 \
    --epochs 300 --pretrain 100 --ramp-epochs 20 \
    --meta-weight 1.0 --min-meta-weight 0.0001 \
    --rank-weight 0.5 --repair-weight 0.5 \
    --virtual-lr 0.001 --seed 42
```

### CBR-SCSF experiment

`train_cbr_scsf.py` optimizes soft selective risk and class-pair confusion over
nested coverage levels, with projected per-class coverage constraints:

```bash
python train_cbr_scsf.py -d cifar10 \
    --epochs 300 --pretrain 100 --cbr-ramp-epochs 20 \
    --coverages 0.70,0.80,0.90,0.95 \
    --coverage-floor-ratio 0.8 --min-meta-weight 0.0001 --seed 42
```

Both scripts overwrite `last.pth` every epoch and report the last-epoch
checkpoint, matching the usual selective-classification protocol. `best.pth`
is still written from validation AURC for analysis only.

### Standard benchmarks (CIFAR-10, SVHN, Cats vs Dogs)

Uses VGG16-BN backbone via `train_scsf.py`:

```bash
# CIFAR-10 (paper configuration)
python train_scsf.py -d cifar10 \
    --epochs 300 --pretrain 100 \
    --meta-weight-mode decay --init-meta-weight 1.0 --min-meta-weight 0.001 \
    --error-weight 1.0 --seed 42

# SVHN
python train_scsf.py -d svhn \
    --epochs 300 --pretrain 100 \
    --meta-weight-mode decay --init-meta-weight 1.0 --min-meta-weight 0.001 \
    --seed 42

# Cats vs Dogs (64×64 input)
python train_scsf.py -d catsdogs \
    --epochs 300 --pretrain 100 \
    --meta-weight-mode decay --init-meta-weight 1.0 --min-meta-weight 0.001 \
    --seed 42
```

<!-- ### Medical datasets

Each medical dataset has a self-contained script with ResNet-18 backbone (trained from scratch, no ImageNet pretraining). All use cosine-decay meta-weight and error_weight=10:

```bash
python scsf_brain_tumor.py        # Brain Tumor MRI (4 classes, 200 epochs)
python scsf_chest_xray.py         # Chest X-Ray Pneumonia (2 classes)
python scsf_malaria.py            # Malaria Cell Images (2 classes, 100 epochs)
python scsf_alzheimer_tpu.py      # Alzheimer's MRI (4 classes, 100 epochs)
python scsf_oct.py                # OCT Retinal (4 classes, 200 epochs)
python scsf_idrid.py              # IDRiD Diabetic Retinopathy (5 classes, 200 epochs)
python scsf_busi_tpu.py           # Breast Ultrasound (3 classes, 100 epochs)
python scsf_oasis_alzheimer.py    # OASIS Alzheimer's (4 classes, multi-trial)
python scsf_brain_tumor_tpu.py    # Brain Tumor MRI (TPU variant)
```

### Baselines

Deep Gamblers and SelectiveNet baselines are run from the parent directory:

```bash
cd ..
python main.py -d cifar10 --epochs 300 -o 2.2   # Deep Gamblers
```

SAT baselines for medical datasets:

```bash
python sat_brain_tumor.py
python sat_chest_xray.py
python sat_malaria.py
python sat_alzheimer_tpu.py
python sat_idrid.py
python sat_busi_tpu.py
```

### Ablation (layer selection)

```bash
python ablation_layer_selection.py
```

## Architecture

```
Input Image
    ↓
Backbone (VGG16-BN or ResNet-18, standard C-class output)
    ├── layer_m-1 features ──┐
    ├── layer_m features ────┼──→ [flatten + concat] → MetaCalibrator MLP → ĉ(x) ∈ [0,1]
    └── logits (C-dim) ──────┘
                                        ↓
                              Reject if ĉ(x) < τ
```

**MetaCalibrator**: 5-layer MLP (D → 1024 → 512 → 256 → 128 → 1), ReLU + Dropout(0.3), Sigmoid output, Xavier init.

**Training protocol**:
1. **Phase 1** (warmup): Train backbone with CE only
2. **Phase 2** (joint): Train backbone with CE + λ · MSE(ĉ(x), TCP), where λ follows cosine decay

## File Structure

```
rl_reward/
├── train_scsf.py              # Standard benchmarks (VGG16-BN)
├── scsf_brain_tumor.py        # Medical: Brain Tumor MRI
├── scsf_chest_xray.py         # Medical: Chest X-Ray
├── scsf_malaria.py            # Medical: Malaria
├── scsf_alzheimer_tpu.py      # Medical: Alzheimer's
├── scsf_oct.py                # Medical: OCT Retinal
├── scsf_idrid.py              # Medical: IDRiD DR
├── scsf_busi_tpu.py           # Medical: Breast Ultrasound
├── scsf_oasis_alzheimer.py    # Medical: OASIS Alzheimer's
├── scsf_brain_tumor_tpu.py    # Medical: Brain Tumor (TPU)
├── sat_*.py                   # SAT baselines
├── ablation_layer_selection.py# Layer selection ablation
└── README.md
../
├── main.py                    # Deep Gamblers / SelectiveNet baseline
├── dataset_utils.py           # Cats vs Dogs resizing utility
├── models/cifar/vgg.py        # VGG16-BN architecture
└── utils/                     # Logger, Bar, AverageMeter
``` -->

## License

MIT License
