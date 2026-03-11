# SCSF: Selective Classification with Supervised Features

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
