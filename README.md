# SCSF: Self-Calibrated Selective Framework

A training-aware calibration framework for selective classification that improves upon Deep Gamblers baseline.

## Overview

SCSF (Self-Calibrated Selective Framework) is a novel approach to selective classification that:
- Uses a **Meta-Calibrator** network to predict confidence scores
- Employs **RL-based hyperparameter tuning** for threshold and loss weighting
- Supports **training-aware calibration** where backbone and meta-calibrator evolve together
- Achieves **10-15% improvement** in AURC over Deep Gamblers baseline

## Key Features

| Feature | Description |
|---------|-------------|
| Meta-Calibrator | MLP that takes pool4/pool5 features + logits to predict calibrated confidence |
| RL Controller | PPO agent that dynamically adjusts rejection threshold and meta-loss weight |
| Training-aware | Backbone continues training while meta-calibrator adapts (decoupled gradients) |
| End-to-end mode | Optional full gradient flow for joint optimization |
| Spatial Variance | Alternative uncertainty signal using per-channel feature variance |

## Installation

```bash
pip install torch torchvision numpy
```

## Usage

### Train SCSF (default configuration)
```bash
python train_scsf.py -d cifar10 --epochs 300 --pretrain 100 --seed 42
```

### Train with End-to-End gradient flow
```bash
python train_scsf.py -d cifar10 --epochs 300 --pretrain 100 --end-to-end --seed 42
```

### Train with Spatial Variance features
```bash
python train_scsf.py -d cifar10 --epochs 300 --pretrain 100 --spatial-var --seed 42
```

### Evaluate with multiple trials
```bash
python eval_scsf_multi_trial.py --checkpoint ./save/cifar10/vgg16_bn_scsf/300.pth -d cifar10 --seeds 42 10 300
```

## Results (CIFAR-10)

| Method | AURC (mean±std) | Improvement |
|--------|-----------------|-------------|
| Deep Gamblers Baseline | 0.006903 | - |
| SCSF (TCP + RL) | 0.005875 ± 0.000243 | +14.9% |
| SCSF End-to-End | 0.006191 ± 0.000243 | +10.3% |
| SCSF Spatial Variance | 0.0063 | +8.7% |

## Architecture

```
Input Image
    ↓
VGG16_bn Backbone (with intermediate feature extraction)
    ↓
┌─────────────────────────────────────────────────────┐
│ Pool4 Features (512×2×2) + Pool5 Features (512×1×1) │
│              + Logits (10)                          │
└─────────────────────────────────────────────────────┘
    ↓
Meta-Calibrator MLP
    ↓
Calibrated Confidence ĉ(x) ∈ [0, 1]
    ↓
RL Controller (PPO) adjusts τ and meta_weight
```

## File Structure

```
├── train_scsf.py          # Main training script
├── eval_scsf_multi_trial.py   # Multi-seed evaluation
├── models/                # VGG model definitions
│   └── cifar/
│       └── vgg.py
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{scsf2026,
  title={Self-Calibrated Selective Framework for Selective Classification},
  author={...},
  booktitle={...},
  year={2026}
}
```

## License

MIT License
