#!/usr/bin/env python3
"""Evaluate one R3 checkpoint with SCSF-style shuffle trials.

This is evaluation-split variance, not independent training seeds.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from train_r3_scsf import COVERAGE_POINTS, RawScoreCalibrator, VGG16BN_FeatureExtractor


def load_r3_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    backbone = VGG16BN_FeatureExtractor(num_classes=10, input_size=32).to(device)
    calibrator = RawScoreCalibrator(512 * 4, 512, 10).to(device)
    backbone.load_state_dict(checkpoint["backbone"])
    calibrator.load_state_dict(checkpoint["calibrator"])
    backbone.eval()
    calibrator.eval()
    return backbone, calibrator, checkpoint.get("epoch")


@torch.no_grad()
def collect_scores(backbone, calibrator, loader, device):
    scores = []
    correctness = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, pool4, pool5 = backbone(inputs, return_features=True)
        scores.append(calibrator(pool4, pool5, logits).cpu())
        correctness.append(logits.argmax(dim=1).eq(targets).cpu())
    return torch.cat(scores), torch.cat(correctness)


def coverage_errors(scores, correctness, coverages):
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_correct = correctness[order]
    n = len(scores)
    errors = {}
    for coverage in coverages:
        k = max(1, int(round(n * coverage / 100.0)))
        errors[coverage] = 100.0 * (~sorted_correct[:k]).float().mean().item()
    prefix = (~sorted_correct).float().cumsum(0) / torch.arange(
        1, n + 1, dtype=torch.float32
    )
    return errors, prefix.mean().item()


def sparse_aurc(error_dict):
    coverages = sorted(error_dict, reverse=True)
    aurc = 0.0
    for left, right in zip(coverages[:-1], coverages[1:]):
        width = (left - right) / 100.0
        aurc += 0.5 * (error_dict[left] / 100.0 + error_dict[right] / 100.0) * width
    return aurc


def split_by_shuffle(scores, correctness, seed, val_size=2000):
    rng = random.Random(seed)
    indices = list(range(len(scores)))
    rng.shuffle(indices)
    test_idx = torch.tensor(indices[val_size:], dtype=torch.long)
    return scores[test_idx], correctness[test_idx]


def summarize(trials):
    keys = sorted(trials[0]["coverage_errors"])
    means = {k: float(np.mean([t["coverage_errors"][k] for t in trials])) for k in keys}
    stds = {
        k: float(np.std([t["coverage_errors"][k] for t in trials], ddof=1))
        for k in keys
    }
    aurcs = [t["prefix_aurc"] for t in trials]
    sparse = [t["sparse_aurc"] for t in trials]
    return {
        "n_trials": len(trials),
        "prefix_aurc_mean": float(np.mean(aurcs)),
        "prefix_aurc_std": float(np.std(aurcs, ddof=1)) if len(aurcs) > 1 else 0.0,
        "sparse_aurc_mean": float(np.mean(sparse)),
        "sparse_aurc_std": float(np.std(sparse, ddof=1)) if len(sparse) > 1 else 0.0,
        "coverage_error_mean": means,
        "coverage_error_std": stds,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 42, 123])
    parser.add_argument("--output", default="./save/r3_cifar10_paper_seed42/multi_trial.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )
    loader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)

    backbone, calibrator, best_epoch = load_r3_checkpoint(args.checkpoint, device)
    scores, correctness = collect_scores(backbone, calibrator, loader, device)

    full_errors, full_prefix = coverage_errors(scores, correctness, COVERAGE_POINTS)
    full = {
        "n": int(len(scores)),
        "accuracy": 100.0 * correctness.float().mean().item(),
        "coverage_errors": full_errors,
        "prefix_aurc": full_prefix,
        "sparse_aurc": sparse_aurc(full_errors),
    }

    trials = []
    for seed in args.seeds:
        split_scores, split_correct = split_by_shuffle(scores, correctness, seed)
        errors, prefix = coverage_errors(split_scores, split_correct, COVERAGE_POINTS)
        trials.append(
            {
                "seed": seed,
                "n": int(len(split_scores)),
                "accuracy": 100.0 * split_correct.float().mean().item(),
                "coverage_errors": errors,
                "prefix_aurc": prefix,
                "sparse_aurc": sparse_aurc(errors),
            }
        )

    payload = {
        "checkpoint": args.checkpoint,
        "best_epoch": best_epoch,
        "note": "Shuffle trials reuse one trained model; they are not independent training seeds.",
        "full_test": full,
        "shuffle_split_8k": summarize(trials),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Loaded validation-best epoch {best_epoch}")
    print(
        f"Full 10k test accuracy={full['accuracy']:.3f}% "
        f"prefix_AURC={full['prefix_aurc']:.6f}"
    )
    stats = payload["shuffle_split_8k"]
    print(
        f"8k shuffle trials ({args.seeds}): "
        f"prefix_AURC={stats['prefix_aurc_mean']:.6f}±{stats['prefix_aurc_std']:.6f}"
    )
    for coverage in [100, 95, 90, 85, 80]:
        print(
            f"  error@{coverage}%  "
            f"full={full['coverage_errors'][coverage]:.3f}  "
            f"8k={stats['coverage_error_mean'][coverage]:.3f}"
            f"±{stats['coverage_error_std'][coverage]:.3f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
