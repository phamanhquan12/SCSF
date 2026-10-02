#!/usr/bin/env python3
"""Initial runnable Risk-weighted Repair-or-Rank (R3-SCSF) prototype.

The virtual probe is an exact one-example SGD step on VGG's final linear
classifier.  It is deliberately first-order and detached: no model parameter,
optimizer state, or BatchNorm running statistic is changed by the probe.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from train_scsf import (
    COVERAGE_POINTS,
    VGG16BN_FeatureExtractor,
    get_dataset,
    get_meta_weight_decay,
)


class TwoViewDataset(Dataset):
    """Return two independently augmented views of the same training item."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        view_u, target_u = self.dataset[index]
        view_v, target_v = self.dataset[index]
        if target_u != target_v:
            raise RuntimeError("Two views of one sample produced different targets")
        return view_u, view_v, target_u


class RawScoreCalibrator(nn.Module):
    """SCSF calibrator that exposes its pre-sigmoid confidence score."""

    def __init__(self, pool4_dim, pool5_dim, num_classes):
        super().__init__()
        input_dim = pool4_dim + pool5_dim + num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
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

    def forward(self, pool4, pool5, logits):
        # Preserve the reviewed SCSF route: feature gradients remain open, but
        # confidence losses cannot directly alter classification logits.
        inputs = torch.cat([pool4, pool5, logits.detach()], dim=1)
        return self.network(inputs).squeeze(1)


def empirical_aurc_influence(raw_scores):
    """Detached exact W_B(rank) for the all-prefix empirical AURC."""

    batch_size = raw_scores.numel()
    if batch_size == 0:
        return raw_scores.detach().clone()
    order = torch.argsort(raw_scores.detach(), descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(batch_size, device=raw_scores.device)
    reciprocal = 1.0 / torch.arange(
        1, batch_size + 1, device=raw_scores.device, dtype=raw_scores.dtype
    )
    by_rank = torch.flip(
        torch.cumsum(torch.flip(reciprocal, dims=[0]), dim=0), dims=[0]
    ) / batch_size
    return by_rank[ranks].detach()


def normalized_error_influence(raw_scores, errors, max_weight=5.0):
    """Mean-one AURC influence on errors, with a detached upper clip."""

    weights = empirical_aurc_influence(raw_scores)
    if errors.any():
        scale = weights[errors].mean().clamp_min(torch.finfo(weights.dtype).eps)
        weights = (weights / scale).clamp(max=max_weight)
    return weights.detach()


def pairwise_rank_loss(
    raw_scores,
    errors,
    rho,
    margin=0.0,
    temperature=1.0,
    rank_epsilon=0.05,
):
    """RC-impact weighted softplus loss for every error/correct pair."""

    wrong = torch.where(errors)[0]
    correct = torch.where(~errors)[0]
    if wrong.numel() == 0 or correct.numel() == 0:
        return raw_scores.sum() * 0.0

    influence = empirical_aurc_influence(raw_scores)
    impact = (
        influence[wrong].unsqueeze(1) - influence[correct].unsqueeze(0)
    ).abs().detach()
    pair_logits = (
        margin
        + raw_scores[wrong].unsqueeze(1)
        - raw_scores[correct].unsqueeze(0)
    )
    softplus = temperature * F.softplus(pair_logits / temperature)
    routing = (rank_epsilon + 1.0 - rho[wrong].detach()).unsqueeze(1)
    denominator = impact.sum().clamp_min(torch.finfo(raw_scores.dtype).eps)
    return (impact * routing * softplus).sum() / denominator


def repair_loss(logits, targets, raw_scores, errors, rho, max_weight=5.0):
    """Extra CE for errors, routed by repairability and RC influence."""

    if not errors.any():
        return logits.sum() * 0.0
    weights = normalized_error_influence(raw_scores, errors, max_weight)
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    routed = weights[errors] * rho[errors].detach() * per_sample[errors]
    return routed.sum() / errors.sum().clamp_min(1)


def virtual_repairability(
    logits_u,
    logits_v,
    penultimate_u,
    penultimate_v,
    targets,
    virtual_lr,
    eps=1e-8,
):
    """Exact detached repair proxy for one SGD step on the final linear layer.

    For sample i, grad_W CE_i = (p_i-y_i) outer h_u_i and
    grad_b CE_i = p_i-y_i.  Substitution into the other-view logits gives the
    exact post-step result without mutating W or b.
    """

    with torch.no_grad():
        probabilities = F.softmax(logits_u.detach(), dim=1)
        delta = probabilities - F.one_hot(
            targets, num_classes=logits_u.size(1)
        ).to(probabilities.dtype)
        feature_alignment = (penultimate_u * penultimate_v).sum(dim=1) + 1.0
        logits_v_after = (
            logits_v.detach()
            - virtual_lr * feature_alignment.unsqueeze(1) * delta
        )
        before = F.cross_entropy(logits_v.detach(), targets, reduction="none")
        after = F.cross_entropy(logits_v_after, targets, reduction="none")
        rho = ((before - after) / (before + eps)).clamp(0.0, 1.0)
    return rho.detach()


@contextmanager
def preserve_module_modes(module):
    """Restore every submodule's train/eval flag after a temporary probe."""

    modes = {child: child.training for child in module.modules()}
    try:
        yield
    finally:
        for child, training in modes.items():
            child.train(training)


def final_classifier_probe(backbone, inputs):
    """Run deterministic VGG features through the final classifier input."""

    with preserve_module_modes(backbone):
        backbone.eval()
        with torch.no_grad():
            features = backbone.base_model.features(inputs)
            hidden = features.flatten(1)
            classifier = backbone.base_model.classifier
            for layer in list(classifier.children())[:-1]:
                hidden = layer(hidden)
            final_layer = list(classifier.children())[-1]
            if not isinstance(final_layer, nn.Linear):
                raise TypeError("R3 virtual probe requires a final nn.Linear layer")
            logits = final_layer(hidden)
    return logits.detach(), hidden.detach()


def compute_batch_rho(backbone, view_u, view_v, targets, virtual_lr):
    """Two-view final-classifier repairability with no persistent state change."""

    logits_u, hidden_u = final_classifier_probe(backbone, view_u)
    logits_v, hidden_v = final_classifier_probe(backbone, view_v)
    return virtual_repairability(
        logits_u, logits_v, hidden_u, hidden_v, targets, virtual_lr
    )


def assert_finite_gradients(parameters):
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Non-finite R3 gradient")


def make_two_view_loader(base_loader, batch_size, workers):
    return DataLoader(
        TwoViewDataset(base_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


class LimitedLoader:
    """Debug adapter preserving the wrapped loader's dataset."""

    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = min(limit, len(loader))
        self.dataset = loader.dataset

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch

    def __len__(self):
        return self.limit


def train_epoch(
    backbone,
    calibrator,
    loader,
    backbone_optimizer,
    calibrator_optimizer,
    device,
    joint,
    meta_weight,
    rank_weight,
    repair_weight_value,
    virtual_lr,
    rank_margin,
    rank_temperature,
    rank_epsilon,
    max_rc_weight,
    max_grad_norm,
):
    backbone.train()
    calibrator.train()
    totals = defaultdict(float)
    sample_count = 0

    for view_u, view_v, targets in loader:
        view_u = view_u.to(device, non_blocking=True)
        view_v = view_v.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        logits, pool4, pool5 = backbone(view_u, return_features=True)
        ce = F.cross_entropy(logits, targets)
        raw_scores = calibrator(pool4, pool5, logits)
        with torch.no_grad():
            errors = logits.argmax(dim=1).ne(targets)

        zero = logits.sum() * 0.0
        correctness_bce = zero
        rank = zero
        repair = zero
        rho = torch.zeros(batch_size, device=device, dtype=logits.dtype)
        if joint:
            correctness = (~errors).to(raw_scores.dtype)
            correctness_bce = F.binary_cross_entropy_with_logits(
                raw_scores, correctness
            )
            rho = compute_batch_rho(
                backbone, view_u, view_v, targets, virtual_lr
            )
            rank = pairwise_rank_loss(
                raw_scores,
                errors,
                rho,
                margin=rank_margin,
                temperature=rank_temperature,
                rank_epsilon=rank_epsilon,
            )
            repair = repair_loss(
                logits,
                targets,
                raw_scores,
                errors,
                rho,
                max_weight=max_rc_weight,
            )

        loss = ce
        if joint:
            loss = (
                loss
                + meta_weight * correctness_bce
                + rank_weight * rank
                + repair_weight_value * repair
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite R3 loss; lower LR/loss weights or lengthen ramp"
            )

        backbone_optimizer.zero_grad(set_to_none=True)
        calibrator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(backbone.parameters()) + list(calibrator.parameters())
        assert_finite_gradients(parameters)
        if max_grad_norm > 0:
            gradient_norm = nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite R3 gradient norm")
        backbone_optimizer.step()
        calibrator_optimizer.step()

        with torch.no_grad():
            totals["loss"] += loss.item() * batch_size
            totals["ce"] += ce.item() * batch_size
            totals["bce"] += correctness_bce.item() * batch_size
            totals["rank"] += rank.item() * batch_size
            totals["repair"] += repair.item() * batch_size
            totals["correct"] += (~errors).sum().item()
            totals["errors"] += errors.sum().item()
            totals["rho_error_sum"] += rho[errors].sum().item()
            sample_count += batch_size

    error_count = totals["errors"]
    return {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "bce": totals["bce"] / sample_count,
        "rank": totals["rank"] / sample_count,
        "repair": totals["repair"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "mean_error_rho": totals["rho_error_sum"] / max(1.0, error_count),
    }


@torch.no_grad()
def evaluate(backbone, calibrator, loader, device):
    backbone.eval()
    calibrator.eval()
    scores = []
    correctness = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, pool4, pool5 = backbone(inputs, return_features=True)
        raw_scores = calibrator(pool4, pool5, logits)
        scores.append(raw_scores.cpu())
        correctness.append(logits.argmax(dim=1).eq(targets).cpu())

    scores = torch.cat(scores)
    correctness = torch.cat(correctness)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_errors = (~correctness[order]).float()
    prefix_risks = sorted_errors.cumsum(0) / torch.arange(
        1, len(scores) + 1, dtype=torch.float32
    )
    coverage_errors = {}
    for coverage in COVERAGE_POINTS:
        count = max(1, int(len(scores) * coverage / 100))
        coverage_errors[coverage] = 100.0 * prefix_risks[count - 1].item()
    return {
        "accuracy": 100.0 * correctness.float().mean().item(),
        "aurc": prefix_risks.mean().item(),
        "coverage_errors": coverage_errors,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train R3-SCSF on VGG/CIFAR")
    parser.add_argument("-d", "--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--pretrain", default=100, type=int)
    parser.add_argument("--ramp-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--meta-weight", default=1.0, type=float)
    parser.add_argument(
        "--min-meta-weight",
        default=1e-4,
        type=float,
        help="Floor for SCSF cosine-decay λ during the joint phase",
    )
    parser.add_argument("--rank-weight", default=0.5, type=float)
    parser.add_argument("--repair-weight", default=0.5, type=float)
    parser.add_argument("--virtual-lr", default=1e-3, type=float)
    parser.add_argument("--rank-margin", default=0.0, type=float)
    parser.add_argument("--rank-temperature", default=1.0, type=float)
    parser.add_argument("--rank-epsilon", default=0.05, type=float)
    parser.add_argument("--max-rc-weight", default=5.0, type=float)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/r3_scsf")
    parser.add_argument(
        "--limit-train-batches",
        default=0,
        type=int,
        help="Debug-only cap on train batches per epoch (0 means all)",
    )
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and less than --epochs")
    if args.ramp_epochs < 1:
        parser.error("--ramp-epochs must be positive")
    if args.virtual_lr < 0 or args.rank_temperature <= 0:
        parser.error("virtual LR must be nonnegative and rank temperature positive")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train, val_loader, test_loader, _, num_classes = get_dataset(args)
    train_loader = make_two_view_loader(base_train, args.batch_size, args.workers)
    if args.limit_train_batches > 0:
        train_loader = LimitedLoader(train_loader, args.limit_train_batches)

    backbone = VGG16BN_FeatureExtractor(
        num_classes=num_classes, input_size=32
    ).to(device)
    calibrator = RawScoreCalibrator(512 * 4, 512, num_classes).to(device)
    backbone_optimizer = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    calibrator_optimizer = optim.Adam(
        calibrator.parameters(), lr=args.meta_lr
    )
    milestones = [epoch for epoch in range(25, args.epochs, 25)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer, milestones=milestones, gamma=0.5
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    best_aurc = math.inf

    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            joint = epoch > args.pretrain
            if joint:
                # Match SCSF: λ cosine-decays from 1.0 → 1e-4 over the joint phase.
                lambda_t = get_meta_weight_decay(
                    epoch,
                    args.pretrain,
                    args.epochs,
                    args.meta_weight,
                    args.min_meta_weight,
                )
                extra_progress = min(
                    1.0, (epoch - args.pretrain) / args.ramp_epochs
                )
                extra_ramp = 0.5 * (1.0 - math.cos(math.pi * extra_progress))
            else:
                lambda_t = 0.0
                extra_ramp = 0.0
            train_metrics = train_epoch(
                backbone,
                calibrator,
                train_loader,
                backbone_optimizer,
                calibrator_optimizer,
                device,
                joint,
                lambda_t,
                extra_ramp * args.rank_weight,
                extra_ramp * args.repair_weight,
                args.virtual_lr,
                args.rank_margin,
                args.rank_temperature,
                args.rank_epsilon,
                args.max_rc_weight,
                args.max_grad_norm,
            )
            scheduler.step()
            val_metrics = evaluate(backbone, calibrator, val_loader, device)
            record = {
                "epoch": epoch,
                "joint": joint,
                "lambda_t": lambda_t,
                "extra_ramp": extra_ramp,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"{'joint' if joint else 'warmup'} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"rho={train_metrics['mean_error_rho']:.3f} "
                f"val_AURC={val_metrics['aurc']:.6f}"
            )
            payload = {
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "calibrator": calibrator.state_dict(),
                "args": vars(args),
                "validation": val_metrics,
            }
            # Papers in this area report the last epoch, not the best validation AURC.
            torch.save(payload, last_path)
            if joint and val_metrics["aurc"] < best_aurc:
                best_aurc = val_metrics["aurc"]
                torch.save(payload, best_path)

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    calibrator.load_state_dict(checkpoint["calibrator"])
    validation = evaluate(backbone, calibrator, val_loader, device)
    test = evaluate(backbone, calibrator, test_loader, device)
    results = {
        "eval_checkpoint": "last",
        "last_epoch": checkpoint["epoch"],
        "best_val_aurc": best_aurc if best_aurc < math.inf else None,
        "validation": validation,
        "test": test,
        "args": vars(args),
    }
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as results_file:
        json.dump(results, results_file, indent=2)
    print(
        f"Evaluated last checkpoint (epoch {checkpoint['epoch']}); "
        f"test accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f}"
    )


if __name__ == "__main__":
    main()
