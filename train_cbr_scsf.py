#!/usr/bin/env python3
"""Initial runnable Confusion-Budgeted Risk (CBR-SCSF) prototype.

This file implements section 8 of docs/SCSF_Risk_Coverage_In_Training_Review.pdf
without changing the existing SCSF runners.  It deliberately uses one raw
confidence logit, natural-distribution mini-batches, and validation-only model
selection.
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

from train_scsf import (
    VGG16BN_FeatureExtractor,
    get_dataset,
    get_meta_weight_decay,
)


EPS = 1e-8

# Official CIFAR-100 fine → 20-superclass map from the downloaded train file.
CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def coarse_group_matrix(num_classes, device, dtype):
    if num_classes != len(CIFAR100_FINE_TO_COARSE):
        raise ValueError("coarse confusion is only defined for CIFAR-100")
    mapping = torch.tensor(CIFAR100_FINE_TO_COARSE, device=device)
    matrix = torch.zeros(num_classes, 20, device=device, dtype=dtype)
    matrix[torch.arange(num_classes, device=device), mapping] = 1.0
    return matrix


class RawConfidenceHead(nn.Module):
    """SCSF calibrator that returns one unbounded confidence score."""

    def __init__(self, pool4_dim, pool5_dim, num_classes, extra_dim=0):
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.network = nn.Sequential(
            nn.Linear(pool4_dim + pool5_dim + num_classes + self.extra_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pool4, pool5, logits, extra=None):
        # Features retain their gradient path; logits follow the reviewed SCSF
        # design and are stopped only on their direct path into this head.
        parts = [pool4, pool5, logits.detach()]
        if self.extra_dim > 0:
            if extra is None:
                extra = pool4.new_zeros(pool4.size(0), self.extra_dim)
            elif extra.ndim == 1:
                extra = extra.unsqueeze(1)
            parts.append(extra)
        features = torch.cat(parts, dim=1)
        return self.network(features).squeeze(1)


class _ImplicitSoftThreshold(torch.autograd.Function):
    """Solve mean(sigmoid((score-h)/T))=coverage with an implicit gradient."""

    @staticmethod
    def forward(ctx, scores, coverage, temperature, iterations):
        if scores.ndim != 1 or scores.numel() == 0:
            raise ValueError("scores must be a non-empty one-dimensional tensor")
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between zero and one")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        # This bracket makes sigmoid tails negligible even for extreme target
        # coverages while retaining the exact monotonic bisection invariant.
        with torch.no_grad():
            margin = temperature * (
                abs(math.log(coverage / (1.0 - coverage))) + 30.0
            )
            low = scores.min() - margin
            high = scores.max() + margin
            for _ in range(iterations):
                midpoint = (low + high) * 0.5
                observed = torch.sigmoid(
                    (scores - midpoint) / temperature
                ).mean()
                # Coverage decreases as h increases.
                if observed > coverage:
                    low = midpoint
                else:
                    high = midpoint
            threshold = (low + high) * 0.5
            mask = torch.sigmoid((scores - threshold) / temperature)
            slope = mask * (1.0 - mask)
        ctx.save_for_backward(slope)
        return threshold

    @staticmethod
    def backward(ctx, grad_threshold):
        (slope,) = ctx.saved_tensors
        denominator = slope.sum().clamp_min(torch.finfo(slope.dtype).tiny)
        # dh/ds_j = a_j(1-a_j) / sum_i a_i(1-a_i).
        grad_scores = grad_threshold * slope / denominator
        return grad_scores, None, None, None


def implicit_soft_threshold(scores, coverage, temperature, iterations=60):
    """Return a differentiable scalar soft-coverage threshold."""

    return _ImplicitSoftThreshold.apply(
        scores, float(coverage), float(temperature), int(iterations)
    )


def soft_coverage_masks(scores, coverages, temperature, iterations=60):
    """Construct nested soft acceptances from the same raw score."""

    thresholds = []
    masks = []
    for coverage in coverages:
        threshold = implicit_soft_threshold(
            scores, coverage, temperature, iterations
        )
        thresholds.append(threshold)
        masks.append(torch.sigmoid((scores - threshold) / temperature))
    return torch.stack(masks, dim=1), torch.stack(thresholds)


def class_coverages(masks, targets, num_classes):
    """Per-class soft coverage and a presence mask for each mini-batch."""

    one_hot = F.one_hot(targets, num_classes=num_classes).to(masks.dtype)
    support = one_hot.sum(dim=0)
    accepted = one_hot.transpose(0, 1) @ masks
    coverage = accepted / support.clamp_min(1.0).unsqueeze(1)
    return coverage, support > 0


def cbr_selective_losses(
    logits,
    masks,
    targets,
    coverages,
    confusion_temperature,
    group_matrix=None,
):
    """Compute global selective risk and per-coverage confusion smooth-max."""

    if confusion_temperature <= 0.0:
        raise ValueError("confusion_temperature must be positive")
    probabilities = F.softmax(logits, dim=1)
    true_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    errors = 1.0 - true_probability
    num_classes = logits.size(1)
    class_cov, present = class_coverages(masks, targets, num_classes)

    # The solved denominator is B*c up to bisection precision. Using the
    # realized sum is numerically safer without changing the objective.
    micro = (masks * errors.unsqueeze(1)).sum(dim=0) / masks.sum(dim=0).clamp_min(
        EPS
    )

    one_hot = F.one_hot(targets, num_classes=num_classes).to(probabilities.dtype)
    if group_matrix is None:
        confusion_one_hot = one_hot
        confusion_prob = probabilities
        confusion_present = present
    else:
        confusion_one_hot = one_hot @ group_matrix
        confusion_prob = probabilities @ group_matrix
        confusion_present = confusion_one_hot.sum(dim=0) > 0
    pair_mass = torch.einsum("bc,bk,bd->kcd", confusion_one_hot, masks, confusion_prob)
    class_mass = torch.einsum("bc,bk->kc", confusion_one_hot, masks).clamp_min(EPS)
    pair_risk = pair_mass / class_mass.unsqueeze(-1)
    group_count = pair_risk.size(-1)
    off_diagonal = ~torch.eye(group_count, dtype=torch.bool, device=logits.device)
    valid = confusion_present.unsqueeze(1) & off_diagonal

    tau = confusion_temperature
    confusion_terms = []
    pair_tables = []
    for coverage_index in range(len(coverages)):
        valid_pairs = pair_risk[coverage_index][valid]
        if valid_pairs.numel():
            confusion_terms.append(
                tau
                * (
                    torch.logsumexp(valid_pairs / tau, dim=0)
                    - math.log(valid_pairs.numel())
                )
            )
        else:
            confusion_terms.append(logits.sum() * 0.0)
        pair_tables.append(
            pair_risk[coverage_index].detach().masked_fill(~valid, float("nan"))
        )

    return {
        "micro": micro.mean(),
        "confusion": torch.stack(confusion_terms).mean(),
        "class_coverage": class_cov,
        "present": present,
        "pair_risks": torch.stack(pair_tables),
    }


class ProjectedCoverageDual:
    """EMA-smoothed nonnegative dual ascent for per-class coverage floors."""

    def __init__(
        self,
        num_classes,
        coverages,
        floor_ratio,
        learning_rate,
        ema_decay,
        max_value,
        device,
    ):
        if not 0.0 < floor_ratio <= 1.0:
            raise ValueError("floor_ratio must be in (0, 1]")
        self.floor_ratio = floor_ratio
        self.learning_rate = learning_rate
        self.ema_decay = ema_decay
        self.max_value = max_value
        shape = (num_classes, len(coverages))
        self.values = torch.zeros(shape, device=device)
        self.residual_ema = torch.zeros(shape, device=device)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self.floors = (
            torch.as_tensor(coverages, device=device).unsqueeze(0) * floor_ratio
        )

    def residual(self, observed_class_coverage):
        return self.floors - observed_class_coverage

    def penalty(self, observed_class_coverage, present):
        residual = self.residual(observed_class_coverage)
        return (
            self.values.detach()
            * residual
            * present[:, None].to(residual.dtype)
        ).sum()

    @torch.no_grad()
    def update(self, observed_class_coverage, present):
        residual = self.residual(observed_class_coverage.detach())
        active = present[:, None].expand_as(residual)
        first = active & ~self.initialized
        later = active & self.initialized
        self.residual_ema[first] = residual[first]
        self.residual_ema[later] = (
            self.ema_decay * self.residual_ema[later]
            + (1.0 - self.ema_decay) * residual[later]
        )
        self.initialized[active] = True
        updated = self.values + self.learning_rate * self.residual_ema
        updated = updated.clamp(min=0.0, max=self.max_value)
        # Absent classes are intentionally unchanged, not treated as zero risk
        # or zero coverage.
        self.values[active] = updated[active]

    def state_dict(self):
        return {
            "values": self.values,
            "residual_ema": self.residual_ema,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state):
        self.values.copy_(state["values"])
        self.residual_ema.copy_(state["residual_ema"])
        self.initialized.copy_(state["initialized"])


def ramp_weight(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def train_epoch(
    backbone,
    confidence_head,
    loader,
    backbone_optimizer,
    meta_optimizer,
    dual,
    device,
    epoch,
    args,
):
    backbone.train()
    confidence_head.train()
    totals = defaultdict(float)
    sample_count = 0
    extra_ramp = ramp_weight(epoch, args.pretrain, args.cbr_ramp_epochs)
    lambda_t = (
        0.0
        if epoch <= args.pretrain
        else get_meta_weight_decay(
            epoch,
            args.pretrain,
            args.epochs,
            args.correctness_weight,
            args.min_meta_weight,
        )
    )

    for batch_index, (inputs, targets) in enumerate(loader):
        if args.limit_train_batches and batch_index >= args.limit_train_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, pool4, pool5 = backbone(inputs, return_features=True)
        raw_scores = confidence_head(pool4, pool5, logits)
        ce = F.cross_entropy(logits, targets)

        with torch.no_grad():
            correctness = logits.argmax(dim=1).eq(targets).to(logits.dtype)
        correctness_bce = F.binary_cross_entropy_with_logits(
            raw_scores, correctness
        )

        micro = logits.sum() * 0.0
        confusion = logits.sum() * 0.0
        floor_penalty = logits.sum() * 0.0
        present_count = 0
        if extra_ramp > 0.0:
            masks, _ = soft_coverage_masks(
                raw_scores,
                args.coverages,
                args.soft_temperature,
                args.threshold_iterations,
            )
            selective = cbr_selective_losses(
                logits,
                masks,
                targets,
                args.coverages,
                args.confusion_temperature,
                group_matrix=getattr(args, "group_matrix", None),
            )
            micro = selective["micro"]
            confusion = selective["confusion"]
            floor_penalty = dual.penalty(
                selective["class_coverage"], selective["present"]
            )
            present_count = selective["present"].sum().item()

        loss = (
            ce
            + lambda_t * correctness_bce
            + extra_ramp
            * (
                args.micro_weight * micro
                + args.confusion_weight * confusion
                + floor_penalty
            )
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite CBR loss; inspect scores, LR, temperatures, and weights"
            )

        backbone_optimizer.zero_grad(set_to_none=True)
        meta_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(confidence_head.parameters()),
            args.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm before optimizer step")
        backbone_optimizer.step()
        meta_optimizer.step()
        if extra_ramp > 0.0:
            dual.update(selective["class_coverage"], selective["present"])

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += ce.item() * batch_size
        totals["correctness_bce"] += correctness_bce.item() * batch_size
        totals["micro"] += micro.item() * batch_size
        totals["confusion"] += confusion.item() * batch_size
        totals["floor_penalty"] += floor_penalty.item() * batch_size
        totals["correct"] += correctness.sum().item()
        totals["present_classes"] += present_count
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("training loader produced no batches")
    return {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "correctness_bce": totals["correctness_bce"] / sample_count,
        "micro": totals["micro"] / sample_count,
        "confusion": totals["confusion"] / sample_count,
        "floor_penalty": totals["floor_penalty"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "mean_present_classes_per_batch": totals["present_classes"]
        / max(1, math.ceil(sample_count / args.batch_size)),
        "ramp": extra_ramp,
        "lambda_t": lambda_t,
    }


@torch.no_grad()
def evaluate(backbone, confidence_head, loader, coverages, num_classes, device):
    """Hard top-k held-out diagnostics using only the single learned score."""

    backbone.eval()
    confidence_head.eval()
    all_scores = []
    all_targets = []
    all_predictions = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits, pool4, pool5 = backbone(inputs, return_features=True)
        all_scores.append(confidence_head(pool4, pool5, logits).cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_targets.append(targets.cpu())

    scores = torch.cat(all_scores)
    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    correctness = predictions.eq(targets)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_errors = (~correctness[order]).to(torch.float32)
    prefix_risk = sorted_errors.cumsum(0) / torch.arange(
        1, len(scores) + 1, dtype=torch.float32
    )

    operating_points = {}
    for coverage in coverages:
        accepted_count = max(1, int(math.floor(coverage * len(scores))))
        accepted = order[:accepted_count]
        accepted_targets = targets[accepted]
        accepted_predictions = predictions[accepted]
        per_class_coverage = []
        confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )
        for true_class in range(num_classes):
            class_total = targets.eq(true_class).sum().item()
            class_accepted = accepted_targets.eq(true_class)
            per_class_coverage.append(
                class_accepted.sum().item() / class_total
                if class_total
                else None
            )
            if class_accepted.any():
                counts = torch.bincount(
                    accepted_predictions[class_accepted], minlength=num_classes
                )
                confusion[true_class] = counts
        operating_points[f"{coverage:.6g}"] = {
            "accepted": accepted_count,
            "realized_coverage": accepted_count / len(scores),
            "risk": (~correctness[accepted]).float().mean().item(),
            "per_class_coverage": per_class_coverage,
            "accepted_confusion": confusion.tolist(),
        }

    return {
        "samples": len(scores),
        "accuracy": correctness.float().mean().item(),
        "aurc": prefix_risk.mean().item(),
        "score_mean": scores.mean().item(),
        "score_std": scores.std(unbiased=False).item(),
        "operating_points": operating_points,
    }


def jsonable_args(args):
    payload = {}
    for key, value in vars(args).items():
        if torch.is_tensor(value):
            continue
        payload[key] = value
    return payload


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_coverages(value):
    try:
        coverages = sorted({float(item) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("coverages must be comma-separated floats") from error
    if not coverages or any(not 0.0 < item < 1.0 for item in coverages):
        raise argparse.ArgumentTypeError("every coverage must be in (0, 1)")
    return coverages


def parse_args():
    parser = argparse.ArgumentParser(description="Train CBR-SCSF on VGG/CIFAR")
    parser.add_argument("-d", "--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--pretrain", default=100, type=int)
    parser.add_argument("--cbr-ramp-epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--coverages", default=parse_coverages("0.70,0.80,0.90,0.95"), type=parse_coverages)
    parser.add_argument("--soft-temperature", default=0.2, type=float)
    parser.add_argument("--confusion-temperature", default=0.1, type=float)
    parser.add_argument("--threshold-iterations", default=60, type=int)
    parser.add_argument("--correctness-weight", default=1.0, type=float)
    parser.add_argument(
        "--min-meta-weight",
        default=1e-4,
        type=float,
        help="Floor for SCSF cosine-decay λ during the joint phase",
    )
    parser.add_argument("--micro-weight", default=0.5, type=float)
    parser.add_argument("--confusion-weight", default=0.5, type=float)
    parser.add_argument(
        "--confusion-groups",
        default="fine",
        choices=["fine", "coarse"],
        help="coarse uses CIFAR-100's 20 superclasses for the confusion term",
    )
    parser.add_argument("--coverage-floor-ratio", default=0.8, type=float)
    parser.add_argument("--dual-lr", default=0.05, type=float)
    parser.add_argument("--dual-ema-decay", default=0.9, type=float)
    parser.add_argument("--dual-max", default=10.0, type=float)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/cbr_scsf")
    parser.add_argument(
        "--limit-train-batches",
        default=0,
        type=int,
        help="debug-only per-epoch batch cap (0 uses all natural batches)",
    )
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and smaller than --epochs")
    if args.cbr_ramp_epochs < 1:
        parser.error("--cbr-ramp-epochs must be positive")
    if args.soft_temperature <= 0 or args.confusion_temperature <= 0:
        parser.error("temperatures must be positive")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.confusion_groups == "coarse" and args.dataset != "cifar100":
        parser.error("--confusion-groups coarse requires -d cifar100")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get_dataset supplies a shuffled, unbalanced train loader: this preserves
    # natural class priors as required by the global coverage equation.
    train_loader, val_loader, test_loader, test_loader_full, num_classes = get_dataset(args)
    args.group_matrix = (
        coarse_group_matrix(num_classes, device, torch.float32)
        if args.confusion_groups == "coarse"
        else None
    )
    backbone = VGG16BN_FeatureExtractor(
        num_classes=num_classes, input_size=32
    ).to(device)
    confidence_head = RawConfidenceHead(
        pool4_dim=512 * 4,
        pool5_dim=512,
        num_classes=num_classes,
    ).to(device)
    backbone_optimizer = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    meta_optimizer = optim.Adam(confidence_head.parameters(), lr=args.meta_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer,
        milestones=list(range(25, args.epochs, 25)),
        gamma=0.5,
    )
    dual = ProjectedCoverageDual(
        num_classes,
        args.coverages,
        args.coverage_floor_ratio,
        args.dual_lr,
        args.dual_ema_decay,
        args.dual_max,
        device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    best_validation_aurc = math.inf

    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                backbone,
                confidence_head,
                train_loader,
                backbone_optimizer,
                meta_optimizer,
                dual,
                device,
                epoch,
                args,
            )
            scheduler.step()
            record = {"epoch": epoch, "train": train_metrics}
            summary = (
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"ramp={train_metrics['ramp']:.3f}"
            )

            # Validation is logged for analysis. Official eval uses last.pth.
            if epoch > args.pretrain:
                validation = evaluate(
                    backbone,
                    confidence_head,
                    val_loader,
                    args.coverages,
                    num_classes,
                    device,
                )
                record["validation"] = validation
                summary += f" val_AURC={validation['aurc']:.6f}"
            print(summary)
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            payload = {
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "confidence_head": confidence_head.state_dict(),
                "dual": dual.state_dict(),
                "validation": record.get("validation"),
                "args": jsonable_args(args),
            }
            torch.save(payload, last_path)
            if epoch > args.pretrain and record["validation"]["aurc"] < best_validation_aurc:
                best_validation_aurc = record["validation"]["aurc"]
                torch.save(payload, best_path)

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    confidence_head.load_state_dict(checkpoint["confidence_head"])
    dual.load_state_dict(checkpoint["dual"])

    validation = evaluate(
        backbone,
        confidence_head,
        val_loader,
        args.coverages,
        num_classes,
        device,
    )
    test = evaluate(
        backbone,
        confidence_head,
        test_loader_full,
        args.coverages,
        num_classes,
        device,
    )
    results = {
        "eval_checkpoint": "last",
        "last_epoch": checkpoint["epoch"],
        "best_val_aurc": best_validation_aurc
        if best_validation_aurc < math.inf
        else None,
        "validation": validation,
        "test": test,
        "dual_values": dual.values.cpu().tolist(),
        "args": jsonable_args(args),
    }
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as results_file:
        json.dump(results, results_file, indent=2)

    print(
        f"Evaluated last checkpoint (epoch {checkpoint['epoch']}); "
        f"test accuracy={100.0 * test['accuracy']:.2f}% "
        f"test AURC={test['aurc']:.6f}"
    )
    for coverage, diagnostics in test["operating_points"].items():
        print(
            f"coverage={coverage} risk={diagnostics['risk']:.6f} "
            f"per-class coverage={diagnostics['per_class_coverage']}"
        )
        print(f"accepted confusion={diagnostics['accepted_confusion']}")


if __name__ == "__main__":
    main()
