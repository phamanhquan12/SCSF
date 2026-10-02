#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch import optim

from scsf.datasets import build_loaders
from scsf.methods import FAITHFULNESS_NOTES, build_method
from scsf.metrics import (
    DEFAULT_COVERAGES,
    accuracy,
    format_curve,
    per_sample_predictions,
    risk_coverage_curve,
    roc_curve_points,
    summarize_selective_metrics,
)
from scsf.reporting import coverage_rows, write_dict_csv, write_plots, write_summary_csv


SCSF_FEATURE_SPECS = [
    "logits",
    "early+logits",
    "mid+logits",
    "late+logits",
    "early+mid+logits",
    "early+late+logits",
    "mid+late+logits",
    "early+mid+late+logits",
    "mid+late",
    "early+mid+late",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SCSF/selective-classification experiment pipeline")
    parser.add_argument("--dataset", default="cifar10", help="cifar10, svhn, catsdogs, covid, organamnist, chexpert, or imagefolder")
    parser.add_argument("--dataset-root", default=None, help="Root for custom ImageFolder datasets")
    parser.add_argument("--data-dir", default="data", help="Central data directory inside SCSF")
    parser.add_argument("--datasets-file", default="datasets_to_run.md", help="Markdown table listing the paper medical suite")
    parser.add_argument("--download", action="store_true", help="Download torchvision/MedMNIST/Kaggle datasets when supported")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild cached preprocessed datasets/statistics")
    parser.add_argument("--prepare-only", action="store_true", help="Build dataset cache/loaders and exit before training")
    parser.add_argument("--num-classes", type=int, default=None, help="Required for --dataset imagefolder")
    parser.add_argument("--input-size", type=int, default=None, help="Override registered input size")
    parser.add_argument("--medical-split-seed", type=int, default=42, help="Seed for deterministic medical-suite train/val/test splits")
    parser.add_argument("--smoke-train-samples", type=int, default=None, help="Use the first N train samples for fast smoke tests")
    parser.add_argument("--smoke-eval-samples", type=int, default=None, help="Use the first N val/test samples for fast smoke tests")
    parser.add_argument("--chexpert-label", default="Pleural Effusion", help="CheXpert label column for binary classification")
    parser.add_argument("--chexpert-uncertain-policy", default="zero", choices=["zero", "one", "ignore"])
    parser.add_argument("--chexpert-frontal-only", action="store_true", help="Use only frontal CheXpert images")
    parser.add_argument("--chexpert-stats-samples", type=int, default=None, help="Optional train-image limit for CheXpert mean/std computation")

    parser.add_argument("--method", default="scsf", choices=["scsf", "ds_scsf", "ds_scsf_v2", "sr", "dg", "sat", "selectivenet", "ccl_sc", "residual_head", "dp_head", "spatial_head"])
    parser.add_argument("--arch", default="vgg16_bn", choices=["vgg16_bn", "resnet18", "resnet50", "resnet101", "densenet121"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet weights for ResNet/DenseNet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate on the validation set every N epochs. The final epoch is always evaluated.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[40, 70, 90])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--save-dir", default="save/experiments")
    parser.add_argument("--output-layout", default="legacy", choices=["legacy", "paper"])
    parser.add_argument("--variant", default="default")
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--figures-dir", default=None)

    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--meta-loss", default="weighted_nll", choices=["weighted_nll", "weighted_nll_pairwise", "mse"])
    parser.add_argument("--meta-confidence-mode", default="meta", choices=["meta", "product", "blend"],
                        help="Confidence expression optimized by the SCSF meta loss")
    parser.add_argument("--scsf-meta-target", default="tcp", choices=["tcp", "correctness"])
    parser.add_argument("--scsf-feature-spec", default="mid+late+logits", choices=SCSF_FEATURE_SPECS)
    parser.add_argument("--scsf-scorer", default="meta", choices=["meta", "sr", "meta_sr_product", "meta_sr_blend", "geometric", "meta_agreement", "min_sr_meta", "margin", "energy", "doctor"])
    parser.add_argument("--scsf-sr-alpha", type=float, default=0.5, help="Softmax-response weight for --scsf-scorer meta_sr_blend")
    parser.add_argument("--meta-weight-mode", default="cosine", choices=["cosine", "fixed"])
    parser.add_argument("--meta-weight", type=float, default=1.0, help="Fixed lambda when --meta-weight-mode fixed")
    parser.add_argument("--init-meta-weight", type=float, default=1.0)
    parser.add_argument("--min-meta-weight", type=float, default=1e-4)
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.1, help="Margin for --meta-loss weighted_nll_pairwise")
    parser.add_argument("--pairwise-alpha", type=float, default=0.9, help="Ranking-loss weight for --meta-loss weighted_nll_pairwise")
    parser.add_argument("--pairwise-hard-fraction", type=float, default=1.0, help="Top fraction of pairwise losses to average")
    parser.add_argument("--pairwise-type", default="hinge", choices=["hinge", "softplus"], help="Pairwise ranking loss shape")
    parser.add_argument("--pairwise-temperature", type=float, default=0.1, help="Temperature for --pairwise-type softplus")
    parser.add_argument("--sr-rank-weight", type=float, default=0.0,
                        help="Weight for auxiliary SR pairwise ranking loss after pretrain (0=disabled)")
    parser.add_argument("--scsf-meta-only-after-pretrain", action="store_true", help="After warmup, optimize only the SCSF meta loss")
    parser.add_argument("--freeze-backbone-after-pretrain", action="store_true", help="Freeze all non-calibrator SCSF parameters after warmup")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--calibrator-arch", default="standard", choices=["standard", "simple"],
                        help="MetaCalibrator architecture: 'standard' (4-layer MLP) or 'simple' (2-layer residual MLP)")
    parser.add_argument("--tcp-ema-momentum", type=float, default=0.0,
                        help="EMA momentum for smoothing TCP calibration targets (0=disabled, 0.9=recommended)")
    parser.add_argument("--meta-weight-warmup-fraction", type=float, default=0.0,
                        help="Fraction of joint-training epochs for lambda warmup (0=disabled, 0.1=recommended)")
    parser.add_argument("--meta-focal-gamma", type=float, default=0.0,
                        help="Focal exponent for meta-loss weighting (0=disabled, 1.0=linear focal, 2.0=quadratic)")
    parser.add_argument("--ds-aux-ce-weight", type=float, default=0.3, help="DS-SCSF auxiliary classifier CE weight")
    parser.add_argument("--ds-aux-cal-weight", type=float, default=1.0, help="DS-SCSF auxiliary calibrator loss weight")
    parser.add_argument("--ds-kd-temperature", type=float, default=2.0, help="DS-SCSF v2 self-distillation temperature")
    parser.add_argument("--ds-kd-weight", type=float, default=0.5, help="DS-SCSF v2 KD vs aux-CE mix (1.0 = KD only)")
    parser.add_argument("--ds-rank-alpha", type=float, default=0.3, help="DS-SCSF v2 fused ranking loss weight in combined meta loss")
    parser.add_argument("--ds-gate-entropy-weight", type=float, default=0.01, help="DS-SCSF v2 fusion-gate entropy bonus (0=off)")
    parser.add_argument(
        "--ds-fused-cal-weight",
        type=float,
        default=0.0,
        help="Relative weight for direct BCE supervision on DS-SCSF v2 fused confidence (0=old v2)",
    )
    parser.add_argument(
        "--ds-gate-supervision-weight",
        type=float,
        default=0.0,
        help="Weight for reliability-guided DS-SCSF v2 gate supervision from per-branch CE (0=off)",
    )
    parser.add_argument(
        "--ds-gate-supervision-temperature",
        type=float,
        default=0.5,
        help="Temperature for reliability-guided DS-SCSF v2 gate targets",
    )
    parser.add_argument(
        "--ds-class-balance",
        default="none",
        choices=["none", "inverse", "effective"],
        help="Class weighting for DS-SCSF auxiliary CE and calibrator BCE",
    )
    parser.add_argument("--ds-class-balance-beta", type=float, default=0.9999, help="Beta for effective-number class weights")
    parser.add_argument("--reward", type=float, default=2.2)
    parser.add_argument("--sat-momentum", type=float, default=0.9)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--selectivenet-alpha", type=float, default=0.5)
    parser.add_argument("--selectivenet-lambda", type=float, default=32.0)
    parser.add_argument("--ccl-weight", type=float, default=0.5)
    parser.add_argument("--ccl-temperature", type=float, default=0.07)
    parser.add_argument("--ccl-base-temperature", type=float, default=0.10)
    parser.add_argument("--ccl-queue-size", type=int, default=300)
    parser.add_argument("--ccl-momentum", type=float, default=0.999)
    parser.add_argument("--ccl-require-full-queue", action="store_true")
    # --- Proposed method hyperparameters ---
    parser.add_argument("--agree-weight", type=float, default=1.0, help="BCE agreement loss weight for residual_head/dp_head/spatial_head")
    parser.add_argument("--min-agree-weight", type=float, default=1e-4, help="Minimum BCE weight after cosine decay")
    parser.add_argument("--dp-proj-damping", type=float, default=1e-2, help="Damping epsilon for dp_head null-space projection")
    parser.add_argument("--spatial-d-head", type=int, default=128, help="Attention head dimension for spatial_head")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, device):
    if len(batch) == 3:
        inputs, targets, indices = batch
    else:
        inputs, targets = batch
        indices = None
    inputs = inputs.to(device)
    targets = targets.to(device)
    indices = indices.to(device) if indices is not None else None
    return inputs, targets, indices


def train_one_epoch(model, loader, optimizer, meta_optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    use_amp = bool(args.amp and device.type == "cuda")
    def has_grad(opt):
        return any(p.grad is not None for group in opt.param_groups for p in group["params"])

    for batch in loader:
        inputs, targets, indices = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if meta_optimizer is not None:
            meta_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(inputs)
            loss = model.training_loss(output, targets, indices, epoch, args)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if has_grad(optimizer):
                scaler.step(optimizer)
            if meta_optimizer is not None and has_grad(meta_optimizer):
                scaler.step(meta_optimizer)
            scaler.update()
        else:
            loss.backward()
            if has_grad(optimizer):
                optimizer.step()
            if meta_optimizer is not None and has_grad(meta_optimizer):
                meta_optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(output.eval_logits.detach(), targets) * batch_size
        total_seen += batch_size
    return total_loss / total_seen, total_acc / total_seen


def freeze_scsf_backbone_if_needed(model, epoch: int, args):
    if args.method != "scsf" or not getattr(args, "freeze_backbone_after_pretrain", False):
        return
    if epoch != args.pretrain + 1:
        return
    frozen = 0
    for name, param in model.named_parameters():
        if not name.startswith("calibrator."):
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen non-calibrator SCSF parameters after warmup: {frozen}")


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    logits_all = []
    targets_all = []
    confidence_all = []
    for batch in loader:
        inputs, targets, _ = unpack_batch(batch, device)
        output = model(inputs)
        logits_all.append(output.eval_logits.cpu())
        targets_all.append(targets.cpu())
        confidence_all.append(output.confidence.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    confidence = torch.cat(confidence_all)
    return logits, targets, confidence


@torch.no_grad()
def evaluate(model, loader, device):
    logits, targets, confidence = collect_outputs(model, loader, device)
    curve = risk_coverage_curve(logits, targets, confidence, DEFAULT_COVERAGES)
    summary = summarize_selective_metrics(logits, targets, confidence)
    return summary.accuracy, curve, summary


def write_results(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["coverage", "error", "accuracy", "selected"])
        writer.writeheader()
        writer.writerows(rows)


def _dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return torch.as_tensor(targets, dtype=torch.long).view(-1)
    if hasattr(dataset, "samples"):
        return torch.as_tensor([target for _, target in getattr(dataset, "samples")], dtype=torch.long)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = _dataset_targets(dataset.dataset)
        return base_targets[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if hasattr(dataset, "dataset"):
        return _dataset_targets(dataset.dataset)
    return None


def _full_dataset_targets(dataset):
    if hasattr(dataset, "dataset"):
        return _full_dataset_targets(dataset.dataset)
    return _dataset_targets(dataset)


def configure_ds_class_weights(args, loaders, num_classes: int):
    args.ds_class_weights = None
    if args.method not in {"ds_scsf", "ds_scsf_v2"} or args.ds_class_balance == "none":
        return
    targets = _full_dataset_targets(loaders["train"].dataset)
    if targets is None:
        raise RuntimeError("--ds-class-balance requires a train dataset exposing targets or samples.")
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    if bool((counts <= 0).any()):
        raise RuntimeError(f"Cannot compute DS class weights with empty classes: counts={counts.tolist()}")
    if args.ds_class_balance == "inverse":
        weights = counts.sum() / (num_classes * counts)
    else:
        beta = float(args.ds_class_balance_beta)
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    args.ds_class_weights = [float(x) for x in weights.tolist()]
    print(f"DS-SCSF class counts: {[int(x) for x in counts.tolist()]}")
    print(f"DS-SCSF class weights ({args.ds_class_balance}): {[round(x, 4) for x in args.ds_class_weights]}")


def save_root_for(args) -> Path:
    if args.output_layout == "paper":
        return Path(args.save_dir) / args.dataset / args.arch / args.method / args.variant / str(args.seed)
    return Path(args.save_dir) / args.dataset / args.method / args.arch


def metrics_root_for(args, checkpoint_name: str, save_root: Path) -> Path:
    if args.metrics_dir is None:
        return save_root
    if args.output_layout == "paper":
        return Path(args.metrics_dir) / args.dataset / args.arch / args.method / args.variant / checkpoint_name
    return Path(args.metrics_dir) / args.dataset / args.method / args.arch / checkpoint_name


def write_rich_evaluation(args, save_root: Path, checkpoint_name: str, logits: torch.Tensor, targets: torch.Tensor, confidence: torch.Tensor):
    summary = summarize_selective_metrics(logits, targets, confidence)
    curve = risk_coverage_curve(logits, targets, confidence, DEFAULT_COVERAGES)
    per_sample = per_sample_predictions(logits, targets, confidence)
    metrics_root = metrics_root_for(args, checkpoint_name, save_root)
    extra = {
        "dataset": args.dataset,
        "method": args.method,
        "arch": args.arch,
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint": checkpoint_name,
    }
    write_summary_csv(metrics_root / "summary.csv", summary, extra)
    write_dict_csv(metrics_root / "risk_coverage.csv", coverage_rows(curve, extra))
    roc_rows = roc_curve_points(per_sample["confidence"], per_sample["correct"])
    write_dict_csv(metrics_root / "roc_curve.csv", [{**row, **extra} for row in roc_rows])
    rows = []
    for idx in range(len(per_sample["target"])):
        rows.append(
            {
                **extra,
                "index": idx,
                "target": int(per_sample["target"][idx]),
                "prediction": int(per_sample["prediction"][idx]),
                "confidence": float(per_sample["confidence"][idx]),
                "correct": int(per_sample["correct"][idx]),
                "loss": float(per_sample["loss"][idx]),
                "softmax_response": float(per_sample["softmax_response"][idx]),
            }
        )
    write_dict_csv(metrics_root / "per_sample.csv", rows)
    if args.figures_dir is not None:
        write_plots(
            metrics_root / "risk_coverage.csv",
            metrics_root / "roc_curve.csv",
            Path(args.figures_dir) / args.dataset,
            args.dataset,
            args.method,
            args.variant,
            checkpoint_name,
        )
    return summary, curve


@torch.no_grad()
def collect_fusion_diagnostics(model, loader, device):
    if not hasattr(model, "layers"):
        return None
    layers = list(getattr(model, "layers"))
    if not layers:
        return None
    weights_all = []
    correct_all = []
    for batch in loader:
        inputs, targets, _ = unpack_batch(batch, device)
        output = model(inputs)
        aux_outputs = output.aux_outputs or {}
        fusion_weights = aux_outputs.get("fusion_weights")
        if fusion_weights is None:
            beta_logits = getattr(model, "beta_logits", None)
            if beta_logits is None:
                return None
            weights = torch.softmax(beta_logits.detach().float().cpu(), dim=0)
            fusion_weights = weights.unsqueeze(0).expand(targets.numel(), -1)
        weights_all.append(fusion_weights.detach().cpu().float())
        correct_all.append(output.eval_logits.argmax(dim=1).eq(targets).detach().cpu())
    if not weights_all:
        return None
    return layers, torch.cat(weights_all, dim=0), torch.cat(correct_all, dim=0)


def write_fusion_diagnostics(args, model, loader, device, output_dir: Path, checkpoint_name: str):
    diagnostics = collect_fusion_diagnostics(model, loader, device)
    if diagnostics is None:
        return
    layers, weights, correct = diagnostics
    rows = []
    masks = {
        "all": torch.ones(correct.numel(), dtype=torch.bool),
        "correct": correct.bool(),
        "incorrect": ~correct.bool(),
    }
    for group, mask in masks.items():
        for layer_idx, layer in enumerate(layers):
            values = weights[mask, layer_idx] if bool(mask.any()) else weights.new_empty(0)
            rows.append(
                {
                    "dataset": args.dataset,
                    "method": args.method,
                    "arch": args.arch,
                    "variant": args.variant,
                    "checkpoint": checkpoint_name,
                    "group": group,
                    "layer": layer,
                    "num_samples": int(values.numel()),
                    "mean_weight": float(values.mean().item()) if values.numel() else 0.0,
                    "std_weight": float(values.std(unbiased=False).item()) if values.numel() else 0.0,
                    "min_weight": float(values.min().item()) if values.numel() else 0.0,
                    "max_weight": float(values.max().item()) if values.numel() else 0.0,
                }
            )
    write_dict_csv(output_dir / "fusion_weights_summary.csv", rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, spec, num_classes = build_loaders(args)
    configure_ds_class_weights(args, loaders, num_classes)
    if args.prepare_only:
        print(f"Prepared dataset={args.dataset} classes={num_classes} input={spec.input_size} mean={spec.mean} std={spec.std}")
        print(
            f"Splits train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
            f"test={len(loaders['test'].dataset)}"
        )
        return

    model = build_method(args, num_classes=num_classes, input_size=spec.input_size, train_size=len(loaders["train"].dataset)).to(device)
    if args.method == "scsf":
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("calibrator.")]
        optimizer = optim.SGD(backbone_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        meta_optimizer = optim.Adam(model.calibrator.parameters(), lr=args.meta_lr)
    elif args.method in {"ds_scsf", "ds_scsf_v2"}:
        meta_prefixes = ("calibrators.", "beta_logits", "fusion.")
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith(meta_prefixes)]
        meta_params = [p for name, p in model.named_parameters() if name.startswith(meta_prefixes)]
        optimizer = optim.SGD(backbone_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        meta_optimizer = optim.Adam(meta_params, lr=args.meta_lr)
    elif args.method in {"residual_head", "dp_head", "spatial_head"}:
        meta_prefixes = ("residual_head.", "baseline_head.", "spatial_head.", "q_proj.", "k_proj.", "v_proj.", "conf_head.")
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith(meta_prefixes)]
        head_params = [p for name, p in model.named_parameters() if name.startswith(meta_prefixes)]
        optimizer = optim.SGD(backbone_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        meta_optimizer = optim.Adam(head_params, lr=args.meta_lr) if head_params else None
    else:
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        meta_optimizer = None
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"Dataset={args.dataset} classes={num_classes} input={spec.input_size} train={len(loaders['train'].dataset)}")
    print(f"Method={args.method} arch={args.arch} device={device} amp={args.amp and device.type == 'cuda'}")
    if args.method == "scsf":
        print(
            "SCSF hparams: "
            f"pretrain={args.pretrain} meta_loss={args.meta_loss} "
            f"meta_conf={args.meta_confidence_mode} "
            f"meta_target={args.scsf_meta_target} feature_spec={args.scsf_feature_spec} "
            f"scorer={args.scsf_scorer} sr_alpha={args.scsf_sr_alpha} "
            f"lambda={args.meta_weight_mode}({args.init_meta_weight}->{args.min_meta_weight}) "
            f"error_weight={args.error_weight} meta_lr={args.meta_lr} "
            f"sr_rank_weight={args.sr_rank_weight}"
        )
    if args.method == "ds_scsf":
        print(
            "DS-SCSF hparams: "
            f"pretrain={args.pretrain} meta_target={args.scsf_meta_target} "
            f"feature_spec={args.scsf_feature_spec} "
            f"lambda={args.meta_weight_mode}({args.init_meta_weight}->{args.min_meta_weight}) "
            f"error_weight={args.error_weight} meta_lr={args.meta_lr} "
            f"aux_ce={args.ds_aux_ce_weight} aux_cal={args.ds_aux_cal_weight} "
            f"fused_cal={args.ds_fused_cal_weight}"
        )
    if args.method == "ds_scsf_v2":
        print(
            "DS-SCSF v2 hparams: "
            f"pretrain={args.pretrain} meta_target={args.scsf_meta_target} "
            f"feature_spec={args.scsf_feature_spec} "
            f"lambda={args.meta_weight_mode}({args.init_meta_weight}->{args.min_meta_weight}) "
            f"warmup={args.meta_weight_warmup_fraction} tcp_ema={args.tcp_ema_momentum} "
            f"error_weight={args.error_weight} meta_lr={args.meta_lr} "
            f"aux_ce={args.ds_aux_ce_weight} aux_cal={args.ds_aux_cal_weight} "
            f"kd(T={args.ds_kd_temperature},w={args.ds_kd_weight}) "
            f"rank_alpha={args.ds_rank_alpha} gate_entropy={args.ds_gate_entropy_weight} "
            f"fused_cal={args.ds_fused_cal_weight} "
            f"gate_sup={args.ds_gate_supervision_weight}@T{args.ds_gate_supervision_temperature}"
        )
    if args.method == "ccl_sc":
        print(
            "CCL-SC hparams: "
            f"pretrain={args.pretrain} weight={args.ccl_weight} "
            f"T={args.ccl_temperature} queue={args.ccl_queue_size} "
            f"momentum_encoder={args.ccl_momentum} require_full_queue={args.ccl_require_full_queue}"
        )
    if args.method in FAITHFULNESS_NOTES:
        print(f"Faithfulness note: {FAITHFULNESS_NOTES[args.method]}")

    best_aurc = float("inf")
    save_root = save_root_for(args)
    save_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        freeze_scsf_backbone_if_needed(model, epoch, args)
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, meta_optimizer, scaler, device, epoch, args)
        scheduler.step()
        should_eval = args.eval_every <= 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            val_acc, val_curve, val_summary = evaluate(model, loaders["val"], device)
            val_aurc = val_summary.aurc
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={train_loss:.4f} train_acc={train_acc:.2f} val_acc={val_acc:.2f} "
                f"val_aurc={val_aurc:.5f} val_naurc={val_summary.naurc:.5f}"
            )
            if val_aurc < best_aurc:
                best_aurc = val_aurc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "num_classes": num_classes,
                        "epoch": epoch,
                        "val_aurc": val_aurc,
                        "checkpoint_type": "best",
                    },
                    save_root / "best.pt",
                )
                write_results(save_root / "val_curve.csv", [r.__dict__ for r in val_curve])
                write_summary_csv(
                    save_root / "val_summary.csv",
                    val_summary,
                    {
                        "dataset": args.dataset,
                        "method": args.method,
                        "arch": args.arch,
                        "variant": args.variant,
                        "seed": args.seed,
                        "checkpoint": "best",
                        "epoch": epoch,
                    },
                )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={train_loss:.4f} train_acc={train_acc:.2f} val=skipped"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
            "epoch": args.epochs,
            "best_val_aurc": best_aurc,
            "checkpoint_type": "last",
        },
        save_root / "last.pt",
    )
    logits, targets, confidence = collect_outputs(model, loaders["test"], device)
    test_summary, test_curve = write_rich_evaluation(args, save_root, "last", logits, targets, confidence)
    write_fusion_diagnostics(args, model, loaders["test"], device, metrics_root_for(args, "last", save_root), "last")
    val_logits, val_targets, val_confidence = collect_outputs(model, loaders["val"], device)
    write_rich_evaluation(args, save_root, "val_last", val_logits, val_targets, val_confidence)
    write_fusion_diagnostics(args, model, loaders["val"], device, metrics_root_for(args, "val_last", save_root), "val_last")
    write_results(save_root / "test_curve.csv", [r.__dict__ for r in test_curve])
    print(
        f"Test acc={test_summary.accuracy:.2f} AURC={test_summary.aurc:.5f} "
        f"NAURC={test_summary.naurc:.5f} AUROC={test_summary.auroc:.5f} "
        f"FPR@95TPR={test_summary.fpr_at_95_tpr:.5f}"
    )
    print(f"Coverage error: {format_curve(test_curve)}")


if __name__ == "__main__":
    main()
