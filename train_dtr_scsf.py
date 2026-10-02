#!/usr/bin/env python3
"""Depth-Transition Repair (DTR-SCSF) training prototype.

Implements the proposal in docs/SCSF_Risk_Coverage_In_Training_Review.pdf:
  * a Pool4 auxiliary classifier;
  * a four-state (shallow-correct, final-correct) confidence head;
  * transition-conditioned shallow-to-deep distillation; and
  * detached risk-coverage rank weights for repair examples.

The baseline entry point is intentionally left unchanged.
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
from torch.utils.data import DataLoader, Dataset

from train_scsf import COVERAGE_POINTS, VGG16BN_FeatureExtractor, get_dataset


class TwoViewDataset(Dataset):
    """Return two independently augmented views from a torchvision dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        first, target = self.dataset[index]
        second, second_target = self.dataset[index]
        if target != second_target:
            raise RuntimeError("Two views of one sample produced different targets")
        return first, second, target


class Pool4Probe(nn.Module):
    """Lightweight classifier for the flattened Pool4 representation."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features):
        return self.network(features)


class TransitionCalibrator(nn.Module):
    """Predict the four shallow/final correctness transition states."""

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
            nn.Linear(128, 4),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pool4, pool5, final_logits):
        # Match the reviewed SCSF gradient path: features remain trainable while
        # the direct path from final logits into the confidence head is blocked.
        combined = torch.cat([pool4, pool5, final_logits.detach()], dim=1)
        return self.network(combined)

    @staticmethod
    def confidence(transition_logits):
        probabilities = F.softmax(transition_logits, dim=1)
        # States are encoded as 2 * shallow_correct + final_correct:
        # 00, 01, 10, 11. Marginalize over final-correct states.
        return probabilities[:, 1] + probabilities[:, 3]


def transition_targets(probe_logits, final_logits, targets):
    shallow_correct = probe_logits.argmax(dim=1).eq(targets)
    final_correct = final_logits.argmax(dim=1).eq(targets)
    states = 2 * shallow_correct.long() + final_correct.long()
    return states, shallow_correct, final_correct


def rc_rank_weights(confidence, selected, max_weight=5.0):
    """Compute detached empirical AURC influence W_B(rank), mean one on selected."""

    batch_size = confidence.numel()
    order = torch.argsort(confidence.detach(), descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(batch_size, device=confidence.device)

    reciprocal = 1.0 / torch.arange(
        1, batch_size + 1, device=confidence.device, dtype=confidence.dtype
    )
    # W_B(r) = (1/B) * sum_{k=r}^B 1/k for one-indexed rank r.
    influence_by_rank = torch.flip(
        torch.cumsum(torch.flip(reciprocal, dims=[0]), dim=0), dims=[0]
    ) / batch_size
    weights = influence_by_rank[ranks]

    if selected.any():
        selected_mean = weights[selected].mean().clamp_min(torch.finfo(weights.dtype).eps)
        weights = (weights / selected_mean).clamp(max=max_weight)
    return weights.detach()


def conditional_repair_loss(
    probe_logits_u,
    probe_logits_v,
    final_logits_u,
    confidence,
    targets,
    temperature,
    max_rc_weight,
):
    """Distill only stable shallow-correct/final-wrong transitions."""

    with torch.no_grad():
        shallow_u_correct = probe_logits_u.argmax(dim=1).eq(targets)
        shallow_v_correct = probe_logits_v.argmax(dim=1).eq(targets)
        final_u_wrong = final_logits_u.argmax(dim=1).ne(targets)
        selected = shallow_u_correct & shallow_v_correct & final_u_wrong
        weights = rc_rank_weights(confidence, selected, max_weight=max_rc_weight)

    if not selected.any():
        return final_logits_u.sum() * 0.0, selected

    teacher = F.softmax(probe_logits_u.detach() / temperature, dim=1)
    student_log = F.log_softmax(final_logits_u / temperature, dim=1)
    per_sample = F.kl_div(student_log, teacher, reduction="none").sum(dim=1)
    loss = (
        weights[selected] * per_sample[selected]
    ).sum() / selected.sum().clamp_min(1)
    return temperature**2 * loss, selected


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_two_view_loader(base_loader, batch_size, workers):
    return DataLoader(
        TwoViewDataset(base_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train_epoch(
    backbone,
    probe,
    transition_head,
    loader,
    backbone_optimizer,
    head_optimizer,
    device,
    joint,
    probe_weight,
    transition_weight,
    repair_weight,
    temperature,
    max_rc_weight,
    max_grad_norm,
):
    backbone.train()
    probe.train()
    transition_head.train()
    totals = defaultdict(float)
    sample_count = 0

    for view_u, view_v, targets in loader:
        view_u = view_u.to(device, non_blocking=True)
        view_v = view_v.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        final_u, pool4_u, pool5_u = backbone(view_u, return_features=True)
        probe_u = probe(pool4_u)
        final_ce = F.cross_entropy(final_u, targets)
        probe_ce = F.cross_entropy(probe_u, targets)

        transition_loss = final_u.sum() * 0.0
        repair_loss = final_u.sum() * 0.0
        selected = torch.zeros_like(targets, dtype=torch.bool)
        confidence = torch.full(
            (batch_size,), 0.5, device=device, dtype=final_u.dtype
        )

        if joint:
            transition_logits = transition_head(pool4_u, pool5_u, final_u)
            states, _, _ = transition_targets(probe_u, final_u, targets)
            transition_loss = F.cross_entropy(transition_logits, states.detach())
            confidence = transition_head.confidence(transition_logits)

            # The second forward is only used to ensure the shallow teacher is
            # stable across two label-preserving augmentations.
            _, pool4_v, _ = backbone(view_v, return_features=True)
            probe_v = probe(pool4_v)
            repair_loss, selected = conditional_repair_loss(
                probe_u,
                probe_v,
                final_u,
                confidence,
                targets,
                temperature,
                max_rc_weight,
            )

        loss = final_ce + probe_weight * probe_ce
        if joint:
            loss = (
                loss
                + transition_weight * transition_loss
                + repair_weight * repair_loss
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite DTR loss; lower loss weights/LR or increase ramp duration"
            )

        backbone_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(probe.parameters()),
                max_grad_norm,
            )
            nn.utils.clip_grad_norm_(transition_head.parameters(), max_grad_norm)
        backbone_optimizer.step()
        head_optimizer.step()

        with torch.no_grad():
            final_correct = final_u.argmax(dim=1).eq(targets)
            totals["loss"] += loss.item() * batch_size
            totals["final_ce"] += final_ce.item() * batch_size
            totals["probe_ce"] += probe_ce.item() * batch_size
            totals["transition"] += transition_loss.item() * batch_size
            totals["repair"] += repair_loss.item() * batch_size
            totals["correct"] += final_correct.sum().item()
            totals["selected"] += selected.sum().item()
            totals["confidence"] += confidence.sum().item()
            sample_count += batch_size

    return {
        key: value / sample_count
        for key, value in totals.items()
        if key not in {"correct", "selected"}
    } | {
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "repair_rate": 100.0 * totals["selected"] / sample_count,
    }


@torch.no_grad()
def evaluate(backbone, probe, transition_head, loader, device):
    backbone.eval()
    probe.eval()
    transition_head.eval()
    scores = []
    correctness = []
    states = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        final_logits, pool4, pool5 = backbone(inputs, return_features=True)
        probe_logits = probe(pool4)
        transition_logits = transition_head(pool4, pool5, final_logits)
        confidence = transition_head.confidence(transition_logits)
        batch_states, _, final_correct = transition_targets(
            probe_logits, final_logits, targets
        )
        scores.append(confidence.cpu())
        correctness.append(final_correct.cpu())
        states.append(batch_states.cpu())

    scores = torch.cat(scores)
    correctness = torch.cat(correctness)
    states = torch.cat(states)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_errors = (~correctness[order]).float()
    prefix_risks = sorted_errors.cumsum(0) / torch.arange(
        1, len(scores) + 1, dtype=torch.float32
    )
    aurc = prefix_risks.mean().item()

    coverage_errors = {}
    accepted_state_10 = {}
    for coverage in COVERAGE_POINTS:
        count = max(1, int(len(scores) * coverage / 100))
        accepted = order[:count]
        coverage_errors[coverage] = 100.0 * (~correctness[accepted]).float().mean().item()
        accepted_errors = ~correctness[accepted]
        if accepted_errors.any():
            accepted_state_10[coverage] = (
                100.0
                * states[accepted][accepted_errors].eq(2).float().mean().item()
            )
        else:
            accepted_state_10[coverage] = 0.0

    state_counts = torch.bincount(states, minlength=4)
    state_rates = {
        f"{state // 2}{state % 2}": 100.0 * state_counts[state].item() / len(states)
        for state in range(4)
    }
    return {
        "accuracy": 100.0 * correctness.float().mean().item(),
        "aurc": aurc,
        "coverage_errors": coverage_errors,
        "state_rates": state_rates,
        "accepted_error_state_10": accepted_state_10,
    }


def print_metrics(name, metrics):
    rates = metrics["state_rates"]
    print(
        f"{name}: accuracy={metrics['accuracy']:.2f}% "
        f"AURC={metrics['aurc']:.6f} "
        f"states[00/01/10/11]="
        f"{rates['00']:.2f}/{rates['01']:.2f}/{rates['10']:.2f}/{rates['11']:.2f}% "
        f"(1,0) among accepted errors @90/95="
        f"{metrics['accepted_error_state_10'][90]:.2f}/"
        f"{metrics['accepted_error_state_10'][95]:.2f}%"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DTR-SCSF")
    parser.add_argument("-d", "--dataset", default="cifar10")
    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--pretrain", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--probe-weight", default=0.3, type=float)
    parser.add_argument("--transition-weight", default=1.0, type=float)
    parser.add_argument("--repair-weight", default=0.5, type=float)
    parser.add_argument("--temperature", default=2.0, type=float)
    parser.add_argument("--max-rc-weight", default=5.0, type=float)
    parser.add_argument(
        "--dtr-ramp-epochs",
        default=10,
        type=int,
        help="Cosine ramp duration for transition and repair losses",
    )
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/dtr_scsf")
    parser.add_argument(
        "--limit-train-batches",
        default=0,
        type=int,
        help="Debug-only cap on batches per epoch (0 means all batches)",
    )
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and less than --epochs")
    if args.dtr_ramp_epochs < 1:
        parser.error("--dtr-ramp-epochs must be at least 1")
    if args.dataset not in {"cifar10", "svhn", "catsdogs", "covid"}:
        parser.error("unsupported dataset")
    return args


class LimitedLoader:
    """Small debug-run adapter that preserves loader.dataset."""

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


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train, val_loader, test_loader, test_loader_full, num_classes = get_dataset(args)
    train_loader = make_two_view_loader(base_train, args.batch_size, args.workers)
    if args.limit_train_batches > 0:
        train_loader = LimitedLoader(train_loader, args.limit_train_batches)

    input_size = 64 if args.dataset in {"catsdogs", "covid"} else 32
    pool4_dim = 512 * 4
    pool5_dim = 512 * (1 if input_size == 32 else 4)

    backbone = VGG16BN_FeatureExtractor(num_classes, input_size).to(device)
    probe = Pool4Probe(pool4_dim, num_classes).to(device)
    transition_head = TransitionCalibrator(
        pool4_dim, pool5_dim, num_classes
    ).to(device)

    backbone_optimizer = optim.SGD(
        list(backbone.parameters()) + list(probe.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    head_optimizer = optim.Adam(transition_head.parameters(), lr=args.meta_lr)
    milestones = [epoch for epoch in range(25, args.epochs, 25)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer, milestones=milestones, gamma=0.5
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    best_aurc = math.inf

    print(f"Device: {device}; DTR warmup={args.pretrain}, total epochs={args.epochs}")
    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            joint = epoch > args.pretrain
            joint_epoch = max(0, epoch - args.pretrain)
            ramp_progress = min(1.0, joint_epoch / args.dtr_ramp_epochs)
            dtr_ramp = 0.5 * (1.0 - math.cos(math.pi * ramp_progress))
            train_metrics = train_epoch(
                backbone,
                probe,
                transition_head,
                train_loader,
                backbone_optimizer,
                head_optimizer,
                device,
                joint,
                args.probe_weight,
                dtr_ramp * args.transition_weight,
                dtr_ramp * args.repair_weight,
                args.temperature,
                args.max_rc_weight,
                args.max_grad_norm,
            )
            scheduler.step()

            record = {
                "epoch": epoch,
                "joint": joint,
                "dtr_ramp": dtr_ramp,
                "train": train_metrics,
            }
            summary = (
                f"Epoch {epoch:03d}/{args.epochs} "
                f"{'joint' if joint else 'warmup'} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"repair={train_metrics['repair_rate']:.2f}% "
                f"ramp={dtr_ramp:.3f}"
            )

            if joint:
                val_metrics = evaluate(
                    backbone, probe, transition_head, val_loader, device
                )
                record["validation"] = val_metrics
                summary += (
                    f" val_acc={val_metrics['accuracy']:.2f}% "
                    f"val_AURC={val_metrics['aurc']:.6f}"
                )
                if val_metrics["aurc"] < best_aurc:
                    best_aurc = val_metrics["aurc"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "backbone": backbone.state_dict(),
                            "probe": probe.state_dict(),
                            "transition_head": transition_head.state_dict(),
                            "args": vars(args),
                            "validation": val_metrics,
                        },
                        best_path,
                    )
            print(summary)
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    probe.load_state_dict(checkpoint["probe"])
    transition_head.load_state_dict(checkpoint["transition_head"])
    print(f"Loaded validation-best epoch {checkpoint['epoch']}")

    val_metrics = evaluate(backbone, probe, transition_head, val_loader, device)
    test_metrics = evaluate(backbone, probe, transition_head, test_loader_full, device)
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    results = {
        "best_epoch": checkpoint["epoch"],
        "validation": val_metrics,
        "test": test_metrics,
        "args": vars(args),
    }
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as results_file:
        json.dump(results, results_file, indent=2)


if __name__ == "__main__":
    main()
