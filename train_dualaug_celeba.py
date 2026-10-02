#!/usr/bin/env python3
"""dualaug on CelebA (Attractive), matching CCL-SC protocol.

Same idea as CIFAR dualaug:
  - two views: x and horizontal flip(x)
  - CE on both views
  - confidence head trained to predict view agreement (not hard correctness)
  - no acccon / contrastive

Protocol (CCL-SC / prior acccon CelebA runs):
  ResNet-18, Adam 1e-5 backbone / 1e-3 head, 50 epochs, Es=1, ramp 2,
  select best val accuracy, evaluate official test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dyn_scsf import evaluate_selective, jsonable
from train_acccon_large import (
    ResNetFeatureExtractor,
    build_celeba_loaders,
    data_root,
    dataset_presets,
)
from train_cbr_scsf import RawConfidenceHead, seed_everything
from train_scsf import get_meta_weight_decay


def ramp_weight(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))


def train_epoch_dualaug(
    backbone,
    confidence_head,
    loader,
    backbone_optimizer,
    head_optimizer,
    device,
    epoch,
    args,
):
    backbone.train()
    confidence_head.train()
    lambda_t = (
        0.0
        if epoch <= args.pretrain
        else get_meta_weight_decay(
            epoch,
            args.pretrain,
            args.epochs,
            args.meta_weight,
            args.min_meta_weight,
        )
    )
    totals = defaultdict(float)
    sample_count = 0

    for batch_index, batch in enumerate(loader):
        if args.limit_train_batches and batch_index >= args.limit_train_batches:
            break
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, pool4, pool5 = backbone(inputs, return_features=True)
        logits_b, _, _ = backbone(inputs.flip(-1), return_features=True)
        ce = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits_b, targets)
        )
        with torch.no_grad():
            agree = logits.argmax(dim=1).eq(logits_b.argmax(dim=1))
            correctness = logits.argmax(dim=1).eq(targets)
        scores = confidence_head(pool4, pool5, logits)
        bce = F.binary_cross_entropy_with_logits(scores, agree.to(scores.dtype))
        loss = ce + lambda_t * bce
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite dualaug celeba loss")

        backbone_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(backbone.parameters()) + list(confidence_head.parameters())
        grad_norm = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        backbone_optimizer.step()
        head_optimizer.step()

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += ce.item() * batch_size
        totals["bce"] += bce.item() * batch_size
        totals["correct"] += correctness.sum().item()
        totals["agree"] += agree.float().sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("empty loader")
    return {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "bce": totals["bce"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "agree": 100.0 * totals["agree"] / sample_count,
        "lambda_t": lambda_t,
        "ramp": ramp_weight(epoch, args.pretrain, args.ramp_epochs),
    }


def parse_args():
    preset = dataset_presets("celeba")
    parser = argparse.ArgumentParser(description="dualaug on CelebA Attractive")
    parser.add_argument("--epochs", default=preset["epochs"], type=int)
    parser.add_argument("--pretrain", default=preset["pretrain"], type=int)
    parser.add_argument("--ramp-epochs", default=preset["ramp_epochs"], type=int)
    parser.add_argument("--batch-size", default=preset["batch_size"], type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr", default=preset["lr"], type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--meta-weight", default=1.0, type=float)
    parser.add_argument("--min-meta-weight", default=1e-4, type=float)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--limit-train-batches", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: ./save/fresh_dualaug_celeba_seed{seed}",
    )
    args = parser.parse_args()
    args.dataset = "celeba"
    args.arch = preset["arch"]
    args.num_classes = preset["num_classes"]
    args.select_by = preset["select_by"]
    args.eval_test_set = preset["eval_test_set"]
    if args.output_dir is None:
        args.output_dir = f"./save/fresh_dualaug_celeba_seed{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = data_root()
    celeba_attr = os.path.join(root, "celeba", "list_attr_celeba.txt")
    if not os.path.isfile(celeba_attr):
        raise FileNotFoundError(
            f"CelebA not found at {root}/celeba (missing list_attr_celeba.txt)"
        )

    train_loader, val_loader, test_loader = build_celeba_loaders(args)
    backbone = ResNetFeatureExtractor(args.arch, args.num_classes).to(device)
    confidence_head = RawConfidenceHead(
        backbone.layer3_dim, backbone.layer4_dim, args.num_classes
    ).to(device)

    backbone_optimizer = optim.Adam(backbone.parameters(), lr=args.lr)
    head_optimizer = optim.Adam(confidence_head.parameters(), lr=args.meta_lr)

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2)

    best_val_acc = -math.inf
    best_val_aurc = math.inf
    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch_dualaug(
                backbone,
                confidence_head,
                train_loader,
                backbone_optimizer,
                head_optimizer,
                device,
                epoch,
                args,
            )
            validation = evaluate_selective(
                backbone, confidence_head, val_loader, device
            )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
            print(
                f"Epoch {epoch:03d}/{args.epochs} dualaug celeba "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"agree={train_metrics['agree']:.2f}% "
                f"bce={train_metrics['bce']:.4f} "
                f"val_acc={validation['accuracy']:.2f}% "
                f"val_AURC={validation['aurc']:.6f}",
                flush=True,
            )
            history_file.write(json.dumps(jsonable(record)) + "\n")
            history_file.flush()
            payload = {
                "epoch": epoch,
                "variant": "dualaug",
                "dataset": "celeba",
                "backbone": backbone.state_dict(),
                "confidence_head": confidence_head.state_dict(),
                "args": jsonable(vars(args)),
                "validation": validation,
            }
            torch.save(payload, last_path)
            if validation["accuracy"] > best_val_acc:
                best_val_acc = validation["accuracy"]
                best_val_aurc = validation["aurc"]
                torch.save(payload, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    confidence_head.load_state_dict(checkpoint["confidence_head"])
    test = evaluate_selective(backbone, confidence_head, test_loader, device)

    last_ckpt = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(last_ckpt["backbone"])
    confidence_head.load_state_dict(last_ckpt["confidence_head"])
    test_last = evaluate_selective(backbone, confidence_head, test_loader, device)

    results = {
        "eval_checkpoint": "best_val_acc",
        "eval_test_set": args.eval_test_set,
        "variant": "dualaug",
        "dataset": "celeba",
        "arch": args.arch,
        "selected_epoch": checkpoint["epoch"],
        "best_val_accuracy": best_val_acc,
        "best_val_aurc": best_val_aurc,
        "test": test,
        "test_last": test_last,
        "args": jsonable(vars(args)),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(results), handle, indent=2)
    with open(
        os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(jsonable(test), handle, indent=2)

    ce = test["coverage_errors"]
    print(
        f"Evaluated best_val_acc (epoch {checkpoint['epoch']}) dualaug celeba; "
        f"official_test accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f} "
        f"@95={ce[95]:.2f} @90={ce[90]:.2f} @80={ce[80]:.2f} @10={ce[10]:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
