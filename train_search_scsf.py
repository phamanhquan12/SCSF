#!/usr/bin/env python3
"""CIFAR-100 search variants built on the clean SCSF stack.

Variants
  scsf      CE + correctness BCE (missing CIFAR-100 SCSF baseline)
  rank      SCSF + single-view RC-weighted pairwise ranking
  csc       SCSF + SR-weighted supervised contrastive on pool5 (CCL-SC mechanism)
  rank_csc  rank + CSC together
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train_r3_scsf import (
    RawScoreCalibrator,
    evaluate,
    pairwise_rank_loss,
    seed_everything,
)
from train_scsf import VGG16BN_FeatureExtractor, get_dataset, get_meta_weight_decay


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features):
        return F.normalize(self.network(features), dim=1)


class FeatureQueue:
    """FIFO memory of detached projected features for queue-based SupCon."""

    def __init__(self, dim, size, device):
        self.size = size
        self.z = torch.zeros(size, dim, device=device)
        self.y = torch.full((size,), -1, device=device, dtype=torch.long)
        self.c = torch.zeros(size, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def enqueue(self, features, targets, confidence):
        count = features.size(0)
        if count >= self.size:
            self.z.copy_(features[-self.size:])
            self.y.copy_(targets[-self.size:])
            self.c.copy_(confidence[-self.size:])
            self.ptr = 0
            self.filled = self.size
            return
        end = self.ptr + count
        if end <= self.size:
            self.z[self.ptr:end] = features
            self.y[self.ptr:end] = targets
            self.c[self.ptr:end] = confidence
            self.ptr = end % self.size
        else:
            first = self.size - self.ptr
            self.z[self.ptr:] = features[:first]
            self.y[self.ptr:] = targets[:first]
            self.c[self.ptr:] = confidence[:first]
            self.z[: count - first] = features[first:]
            self.y[: count - first] = targets[first:]
            self.c[: count - first] = confidence[first:]
            self.ptr = count - first
        self.filled = min(self.size, self.filled + count)


def confidence_supcon(query, targets, confidence, queue, temperature):
    """SR-weighted supervised contrastive loss against an in-batch + FIFO queue."""

    if query.size(0) == 0:
        return query.sum() * 0.0
    if queue.filled > 0:
        keys = torch.cat([query, queue.z[: queue.filled]], dim=0)
        key_targets = torch.cat([targets, queue.y[: queue.filled]], dim=0)
        key_confidence = torch.cat([confidence, queue.c[: queue.filled]], dim=0)
    else:
        keys = query
        key_targets = targets
        key_confidence = confidence

    logits = query @ keys.t() / temperature
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    eye = torch.eye(query.size(0), dtype=torch.bool, device=query.device)
    self_mask[:, : query.size(0)] = eye
    positives = key_targets.unsqueeze(0).eq(targets.unsqueeze(1)) & ~self_mask
    valid = positives.any(dim=1)
    if not valid.any():
        return query.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    weights = (
        confidence.detach().unsqueeze(1) * key_confidence.detach().unsqueeze(0)
    ).clamp_min(1e-6)
    positive_weights = weights * positives.to(weights.dtype)
    denom = positive_weights.sum(dim=1).clamp_min(1e-6)
    # 0 * -inf from the self-mask would otherwise become NaN.
    safe_log_prob = log_prob.masked_fill(~positives, 0.0)
    loss = -(positive_weights * safe_log_prob).sum(dim=1) / denom
    return loss[valid].mean()


def ramp_weight(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def train_epoch(
    backbone,
    calibrator,
    projector,
    queue,
    loader,
    backbone_optimizer,
    head_optimizer,
    device,
    epoch,
    args,
):
    backbone.train()
    calibrator.train()
    if projector is not None:
        projector.train()

    joint = epoch > args.pretrain
    lambda_t = (
        0.0
        if not joint
        else get_meta_weight_decay(
            epoch,
            args.pretrain,
            args.epochs,
            args.meta_weight,
            args.min_meta_weight,
        )
    )
    extra_ramp = ramp_weight(epoch, args.pretrain, args.ramp_epochs)
    totals = defaultdict(float)
    sample_count = 0

    for batch_index, (inputs, targets) in enumerate(loader):
        if args.limit_train_batches and batch_index >= args.limit_train_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, pool4, pool5 = backbone(inputs, return_features=True)
        raw_scores = calibrator(pool4, pool5, logits)
        ce = F.cross_entropy(logits, targets)
        with torch.no_grad():
            correctness = logits.argmax(dim=1).eq(targets)
        errors = ~correctness
        correctness_bce = F.binary_cross_entropy_with_logits(
            raw_scores, correctness.to(raw_scores.dtype)
        )

        rank = raw_scores.sum() * 0.0
        csc = raw_scores.sum() * 0.0
        if extra_ramp > 0.0 and args.rank_weight > 0.0:
            rho = torch.zeros_like(raw_scores)
            rank = pairwise_rank_loss(
                raw_scores,
                errors,
                rho,
                args.rank_margin,
                args.rank_temperature,
                args.rank_epsilon,
            )
        if extra_ramp > 0.0 and args.csc_weight > 0.0 and projector is not None:
            query = projector(pool5)
            sr = F.softmax(logits.detach(), dim=1).max(dim=1).values
            csc = confidence_supcon(
                query, targets, sr, queue, args.csc_temperature
            )
            queue.enqueue(query.detach(), targets, sr.detach())

        loss = (
            ce
            + lambda_t * correctness_bce
            + extra_ramp * (args.rank_weight * rank + args.csc_weight * csc)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite search-variant loss")

        backbone_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(backbone.parameters()) + list(calibrator.parameters())
        if projector is not None:
            params += list(projector.parameters())
        grad_norm = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        backbone_optimizer.step()
        head_optimizer.step()

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += ce.item() * batch_size
        totals["bce"] += correctness_bce.item() * batch_size
        totals["rank"] += rank.item() * batch_size
        totals["csc"] += csc.item() * batch_size
        totals["correct"] += correctness.sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("training loader produced no batches")
    return {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "bce": totals["bce"] / sample_count,
        "rank": totals["rank"] / sample_count,
        "csc": totals["csc"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "lambda_t": lambda_t,
        "ramp": extra_ramp,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SCSF CIFAR search variants")
    parser.add_argument("-d", "--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--variant",
        default="scsf",
        choices=["scsf", "rank", "csc", "rank_csc"],
    )
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--pretrain", default=100, type=int)
    parser.add_argument("--ramp-epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--meta-weight", default=1.0, type=float)
    parser.add_argument("--min-meta-weight", default=1e-4, type=float)
    parser.add_argument("--rank-weight", default=0.5, type=float)
    parser.add_argument("--rank-margin", default=0.0, type=float)
    parser.add_argument("--rank-temperature", default=1.0, type=float)
    parser.add_argument("--rank-epsilon", default=0.05, type=float)
    parser.add_argument("--csc-weight", default=0.5, type=float)
    parser.add_argument("--csc-temperature", default=0.1, type=float)
    parser.add_argument("--queue-size", default=3000, type=int)
    parser.add_argument("--proj-dim", default=128, type=int)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/search_scsf")
    parser.add_argument("--limit-train-batches", default=0, type=int)
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and less than --epochs")
    if args.variant == "scsf":
        args.rank_weight = 0.0
        args.csc_weight = 0.0
    elif args.variant == "rank":
        args.csc_weight = 0.0
    elif args.variant == "csc":
        args.rank_weight = 0.0
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, _, num_classes = get_dataset(args)
    backbone = VGG16BN_FeatureExtractor(num_classes=num_classes, input_size=32).to(
        device
    )
    calibrator = RawScoreCalibrator(512 * 4, 512, num_classes).to(device)
    use_csc = args.csc_weight > 0.0
    projector = (
        ProjectionHead(in_dim=512, out_dim=args.proj_dim).to(device)
        if use_csc
        else None
    )
    queue = (
        FeatureQueue(args.proj_dim, args.queue_size, device) if use_csc else None
    )

    backbone_optimizer = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    head_params = list(calibrator.parameters())
    if projector is not None:
        head_params += list(projector.parameters())
    head_optimizer = optim.Adam(head_params, lr=args.meta_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer,
        milestones=list(range(25, args.epochs, 25)),
        gamma=0.5,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    best_aurc = math.inf

    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                backbone,
                calibrator,
                projector,
                queue,
                train_loader,
                backbone_optimizer,
                head_optimizer,
                device,
                epoch,
                args,
            )
            scheduler.step()
            record = {"epoch": epoch, "train": train_metrics}
            summary = (
                f"Epoch {epoch:03d}/{args.epochs} {args.variant} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"ramp={train_metrics['ramp']:.3f}"
            )
            if epoch > args.pretrain:
                validation = evaluate(backbone, calibrator, val_loader, device)
                record["validation"] = validation
                summary += f" val_AURC={validation['aurc']:.6f}"
            print(summary)
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            payload = {
                "epoch": epoch,
                "variant": args.variant,
                "backbone": backbone.state_dict(),
                "calibrator": calibrator.state_dict(),
                "projector": None if projector is None else projector.state_dict(),
                "args": vars(args),
                "validation": record.get("validation"),
            }
            torch.save(payload, last_path)
            if epoch > args.pretrain and record["validation"]["aurc"] < best_aurc:
                best_aurc = record["validation"]["aurc"]
                torch.save(payload, best_path)

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    calibrator.load_state_dict(checkpoint["calibrator"])
    validation = evaluate(backbone, calibrator, val_loader, device)
    test = evaluate(backbone, calibrator, test_loader, device)
    results = {
        "eval_checkpoint": "last",
        "variant": args.variant,
        "last_epoch": checkpoint["epoch"],
        "best_val_aurc": best_aurc if best_aurc < math.inf else None,
        "validation": validation,
        "test": test,
        "args": {k: v for k, v in vars(args).items() if k != "group_matrix"},
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(
        f"Evaluated last checkpoint (epoch {checkpoint['epoch']}) variant={args.variant}; "
        f"test accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f}"
    )


if __name__ == "__main__":
    main()
