#!/usr/bin/env python3
"""Fresh selective-classification methods that do NOT use acceptance-weighted SupCon.

Logit-only scores (no confidence head)
  msp       CE only; test = max softmax probability
  maxlogit  CE only; test = max logit
  energy    CE only; test = logsumexp(logits)  (closed-set energy)
  margin    CE only; test = top1 − top2 logit
  entropy   CE only; test = −predictive entropy

Head-based
  scsf      CE + hard correctness BCE (plain SCSF baseline)
  focal     CE + focal BCE on confident errors (γ=2)
  tcp       CE + MSE(sigmoid(s), true-class probability)
  addmsp    CE + BCE on s + softplus(α)·logit(MSP)  (α learned; MSP detached)
  risk      CE + BCE + soft selective-risk (known-toxic; kept for completeness)
  valblend  Train as scsf; select λ on val for z(s)+λ·z(MSP); apply to official 10k

dualaug and its ablations (x̃ = horizontal flip of x)
  dualaug         two-view CE + head predicts view agreement
  dualaug_detach  dualaug, but head sees detached pool4/pool5 (no head grad into backbone)
  dualce_scsf     two-view CE + head predicts correctness
  dualce_msp      two-view CE only; test = MSP
  agree_only      single-view CE + head predicts view agreement (x̃ forward without grad)

Official comparison: last.pth on the official CIFAR 10k test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dyn_scsf import REPORT_COVERAGES, binary_auroc
from train_cbr_scsf import RawConfidenceHead, seed_everything
from train_next_scsf import soft_selective_risk_loss
from train_scsf import VGG16BN_FeatureExtractor, get_dataset, get_meta_weight_decay


EPS = 1e-6
LOGIT_SCORE_VARIANTS = ("msp", "maxlogit", "energy", "margin", "entropy")
# Trained methods (not post-hoc score choice):
TRAINED_VARIANTS = (
    "scsf",
    "focal",
    "tcp",
    "softauc",
    "selnet",
    "temptrain",
    "dualaug",
    "dualaug_detach",
    "dualce_scsf",
    "dualce_msp",
    "agree_only",
    "addmsp",
    "risk",
    "valblend",  # train SCSF + val-tuned blend (semi post-hoc)
)
HEAD_VARIANTS = (
    "scsf",
    "focal",
    "tcp",
    "softauc",
    "selnet",
    "dualaug",
    "dualaug_detach",
    "dualce_scsf",
    "agree_only",
    "addmsp",
    "risk",
    "valblend",
)
DUALAUG_FAMILY = ("dualaug", "dualaug_detach", "dualce_scsf", "dualce_msp", "agree_only")
DUAL_CE_VARIANTS = ("dualaug", "dualaug_detach", "dualce_scsf", "dualce_msp")
AGREE_VARIANTS = ("dualaug", "dualaug_detach", "agree_only")
FRESH_VARIANTS = LOGIT_SCORE_VARIANTS + TRAINED_VARIANTS
# unique preserve order
FRESH_VARIANTS = tuple(dict.fromkeys(FRESH_VARIANTS))
VALBLEND_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def ramp_weight(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def parse_coverages(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(not 0.0 < v < 1.0 for v in values):
        raise argparse.ArgumentTypeError("coverages must lie in (0, 1)")
    return values


def logit_msp(logits):
    probs = F.softmax(logits.detach(), dim=1)
    msp = probs.max(dim=1).values.clamp(EPS, 1.0 - EPS)
    return torch.log(msp) - torch.log1p(-msp)


def logit_margin(logits):
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def score_from_logits(logits, variant):
    if variant == "msp":
        return F.softmax(logits, dim=1).max(dim=1).values
    if variant == "maxlogit":
        return logits.max(dim=1).values
    if variant == "energy":
        return torch.logsumexp(logits, dim=1)
    if variant == "margin":
        return logit_margin(logits)
    if variant == "entropy":
        probs = F.softmax(logits, dim=1).clamp_min(EPS)
        return (probs * probs.log()).sum(dim=1)  # −H = sum p log p
    raise ValueError(f"not a logit-score variant: {variant}")


class ResidualBlend(nn.Module):
    """score = s + softplus(α) · logit(MSP). α starts near 0 (pure SCSF)."""

    def __init__(self, init_alpha=-2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, raw_scores, logits):
        scale = F.softplus(self.alpha)
        return raw_scores + scale * logit_msp(logits)

    @torch.no_grad()
    def scale_value(self):
        return float(F.softplus(self.alpha).detach())


class LogitTemperature(nn.Module):
    """Learned temperature for CE and MSP(logits/T). Trained, not post-hoc."""

    def __init__(self, init=0.0, floor=0.1):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(init)))
        self.floor = float(floor)

    def forward(self):
        return F.softplus(self.raw) + self.floor

    @torch.no_grad()
    def value(self):
        return float(self.forward().detach())


def soft_auc_loss(scores, correctness):
    """Pairwise SoftAUC: push correct scores above incorrect ones."""

    correct = torch.where(correctness)[0]
    wrong = torch.where(~correctness)[0]
    if correct.numel() == 0 or wrong.numel() == 0:
        return scores.sum() * 0.0
    # sigmoid(s_correct - s_wrong); maximize ⇒ minimize 1 - mean
    diff = scores[correct].unsqueeze(1) - scores[wrong].unsqueeze(0)
    return 1.0 - torch.sigmoid(diff).mean()


def selective_net_loss(logits, scores, targets, coverage=0.85, cov_weight=1.0):
    """SelectiveNet-style: risk on selected set + coverage penalty + aux CE."""

    per = F.cross_entropy(logits, targets, reduction="none")
    g = torch.sigmoid(scores)
    selected_risk = (g * per).sum() / g.sum().clamp_min(EPS)
    coverage_pen = (g.mean() - float(coverage)) ** 2
    aux = per.mean()
    return selected_risk + float(cov_weight) * coverage_pen + 0.5 * aux, g


def metrics_from_scores(scores, correctness):
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_errors = (~correctness[order]).float()
    prefix = sorted_errors.cumsum(0) / torch.arange(
        1, len(scores) + 1, dtype=torch.float32
    )
    accuracy = float(correctness.float().mean())
    aurc = float(prefix.mean())
    coverage_errors = {}
    for coverage in REPORT_COVERAGES:
        count = max(1, int(len(scores) * coverage / 100))
        coverage_errors[coverage] = 100.0 * float(prefix[count - 1])
    return {
        "samples": int(len(scores)),
        "accuracy": 100.0 * accuracy,
        "aurc": aurc,
        "eaurc": aurc - (1.0 - accuracy),
        "auroc": binary_auroc(scores, correctness),
        "coverage_errors": coverage_errors,
    }


def zscore(t, mean=None, std=None):
    if mean is None:
        mean = t.mean()
    if std is None:
        std = t.std().clamp_min(EPS)
    return (t - mean) / std, mean, std


def collect_head_and_msp(backbone, confidence_head, loader, device):
    backbone.eval()
    confidence_head.eval()
    heads, msps, correctness = [], [], []
    with torch.no_grad():
        for inputs, targets in _iter_batches(loader, device):
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            heads.append(confidence_head(pool4, pool5, logits).cpu())
            msps.append(F.softmax(logits, dim=1).max(dim=1).values.cpu())
            correctness.append(logits.argmax(dim=1).eq(targets).cpu())
    return torch.cat(heads), torch.cat(msps), torch.cat(correctness)


def select_valblend_lambda(backbone, confidence_head, val_loader, device, lambdas):
    head, msp, correct = collect_head_and_msp(
        backbone, confidence_head, val_loader, device
    )
    z_h, mean_h, std_h = zscore(head)
    z_m, mean_m, std_m = zscore(msp)
    best = None
    grid = []
    for lam in lambdas:
        scores = z_h + float(lam) * z_m
        metrics = metrics_from_scores(scores, correct)
        grid.append({"lambda": float(lam), "aurc": metrics["aurc"]})
        row = {"lambda": float(lam), "aurc": metrics["aurc"], "metrics": metrics}
        if best is None or metrics["aurc"] < best["aurc"]:
            best = row
    return {
        "lambda": best["lambda"],
        "val_aurc": best["aurc"],
        "head_mean": float(mean_h),
        "head_std": float(std_h),
        "msp_mean": float(mean_m),
        "msp_std": float(std_m),
        "grid": grid,
    }


def apply_valblend(head, msp, stats):
    z_h = (head - stats["head_mean"]) / max(stats["head_std"], EPS)
    z_m = (msp - stats["msp_mean"]) / max(stats["msp_std"], EPS)
    return z_h + float(stats["lambda"]) * z_m


def _iter_batches(loader, device):
    for batch in loader:
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]
        yield (
            inputs.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )


def collect_fresh_scores(
    backbone, confidence_head, blend, loader, device, variant, valblend_stats=None, temperature=None
):
    backbone.eval()
    if confidence_head is not None:
        confidence_head.eval()
    if blend is not None:
        blend.eval()
    if temperature is not None:
        temperature.eval()
    scores = []
    correctness = []
    with torch.no_grad():
        for inputs, targets in _iter_batches(loader, device):
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            if variant == "temptrain":
                t = temperature()
                score = F.softmax(logits / t, dim=1).max(dim=1).values
            elif variant in LOGIT_SCORE_VARIANTS:
                score = score_from_logits(logits, variant)
            elif variant == "dualce_msp":
                score = score_from_logits(logits, "msp")
            elif variant == "addmsp":
                raw = confidence_head(pool4, pool5, logits)
                score = blend(raw, logits)
            elif variant == "valblend":
                raw = confidence_head(pool4, pool5, logits)
                msp = F.softmax(logits, dim=1).max(dim=1).values
                if valblend_stats is None:
                    score = raw
                else:
                    score = apply_valblend(raw, msp, valblend_stats)
            else:
                score = confidence_head(pool4, pool5, logits)
            scores.append(score.cpu())
            correctness.append(logits.argmax(dim=1).eq(targets).cpu())
    return {
        "scores": torch.cat(scores),
        "correctness": torch.cat(correctness),
    }


def evaluate_fresh(
    backbone,
    confidence_head,
    blend,
    loader,
    device,
    variant,
    valblend_stats=None,
    temperature=None,
):
    collected = collect_fresh_scores(
        backbone,
        confidence_head,
        blend,
        loader,
        device,
        variant,
        valblend_stats=valblend_stats,
        temperature=temperature,
    )
    return metrics_from_scores(collected["scores"], collected["correctness"])


def focal_bce_with_logits(scores, correctness, gamma=2.0):
    target = correctness.to(scores.dtype)
    prob = torch.sigmoid(scores)
    p_t = torch.where(correctness, prob, 1.0 - prob).clamp(EPS, 1.0 - EPS)
    bce = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    return ((1.0 - p_t) ** float(gamma) * bce).mean()


def train_epoch(
    backbone,
    confidence_head,
    blend,
    temperature,
    loader,
    backbone_optimizer,
    head_optimizer,
    device,
    epoch,
    args,
):
    backbone.train()
    if confidence_head is not None:
        confidence_head.train()
    if blend is not None:
        blend.train()
    if temperature is not None:
        temperature.train()

    extra_ramp = ramp_weight(epoch, args.pretrain, args.ramp_epochs)
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
    logit_only = args.variant in LOGIT_SCORE_VARIANTS

    for batch_index, batch in enumerate(loader):
        if args.limit_train_batches and batch_index >= args.limit_train_batches:
            break
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        bce = torch.zeros((), device=device)
        risk = torch.zeros((), device=device)
        score_for_log = None

        if args.variant == "temptrain":
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            t = temperature()
            ce = F.cross_entropy(logits / t, targets)
            loss = ce
            score_for_log = F.softmax(logits.detach() / t.detach(), dim=1).max(dim=1).values
            with torch.no_grad():
                correctness = logits.argmax(dim=1).eq(targets)
        elif args.variant in DUALAUG_FAMILY:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            if args.variant in DUAL_CE_VARIANTS:
                logits_b, _, _ = backbone(inputs.flip(-1), return_features=True)
                ce = 0.5 * (
                    F.cross_entropy(logits, targets) + F.cross_entropy(logits_b, targets)
                )
            else:
                ce = F.cross_entropy(logits, targets)
                with torch.no_grad():
                    logits_b, _, _ = backbone(inputs.flip(-1), return_features=True)
            with torch.no_grad():
                agree = logits.argmax(dim=1).eq(logits_b.argmax(dim=1))
                correctness = logits.argmax(dim=1).eq(targets)
            if args.variant == "dualce_msp":
                loss = ce
                score_for_log = score_from_logits(logits.detach(), "msp")
            else:
                if args.variant == "dualaug_detach":
                    pool4, pool5 = pool4.detach(), pool5.detach()
                scores = confidence_head(pool4, pool5, logits)
                target = agree if args.variant in AGREE_VARIANTS else correctness
                bce = F.binary_cross_entropy_with_logits(
                    scores, target.to(scores.dtype)
                )
                score_for_log = scores.detach()
                loss = ce + lambda_t * bce
        elif logit_only:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            ce = F.cross_entropy(logits, targets)
            with torch.no_grad():
                correctness = logits.argmax(dim=1).eq(targets)
            loss = ce
            score_for_log = score_from_logits(logits.detach(), args.variant)
        else:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            ce = F.cross_entropy(logits, targets)
            with torch.no_grad():
                correctness = logits.argmax(dim=1).eq(targets)
            raw = confidence_head(pool4, pool5, logits)
            if args.variant == "addmsp":
                scores = blend(raw, logits)
            else:
                scores = raw
            score_for_log = scores.detach()
            if args.variant == "selnet" and extra_ramp > 0.0:
                sel, _ = selective_net_loss(
                    logits,
                    scores,
                    targets,
                    coverage=args.selnet_coverage,
                    cov_weight=args.selnet_cov_weight,
                )
                # Keep a light correctness BCE so the score stays ranked.
                bce = F.binary_cross_entropy_with_logits(
                    scores, correctness.to(scores.dtype)
                )
                loss = sel + 0.3 * lambda_t * bce
                ce = sel  # for logging
            elif args.variant == "softauc" and extra_ramp > 0.0:
                bce = F.binary_cross_entropy_with_logits(
                    scores, correctness.to(scores.dtype)
                )
                risk = soft_auc_loss(scores, correctness)
                loss = ce + lambda_t * bce + extra_ramp * args.softauc_weight * risk
            elif args.variant == "tcp":
                tcp = (
                    F.softmax(logits.detach(), dim=1)
                    .gather(1, targets[:, None])
                    .squeeze(1)
                )
                bce = F.mse_loss(torch.sigmoid(scores), tcp)
                loss = ce + lambda_t * bce
            elif args.variant == "focal":
                bce = focal_bce_with_logits(scores, correctness, gamma=args.focal_gamma)
                loss = ce + lambda_t * bce
            else:
                bce = F.binary_cross_entropy_with_logits(
                    scores, correctness.to(scores.dtype)
                )
                if args.variant == "risk" and extra_ramp > 0.0 and args.risk_weight > 0.0:
                    risk = soft_selective_risk_loss(
                        scores,
                        ~correctness,
                        args.risk_coverages,
                        args.soft_temperature,
                        args.threshold_iterations,
                    )
                loss = ce + lambda_t * bce + extra_ramp * args.risk_weight * risk

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite fresh loss")

        backbone_optimizer.zero_grad(set_to_none=True)
        if head_optimizer is not None:
            head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(backbone.parameters())
        if confidence_head is not None:
            params += list(confidence_head.parameters())
        if blend is not None:
            params += list(blend.parameters())
        if temperature is not None:
            params += list(temperature.parameters())
        grad_norm = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        backbone_optimizer.step()
        if head_optimizer is not None:
            head_optimizer.step()

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += float(ce.detach()) * batch_size
        totals["bce"] += float(bce.detach()) * batch_size
        totals["risk"] += float(risk.detach()) * batch_size
        totals["correct"] += correctness.sum().item()
        if score_for_log is not None:
            totals["score_mean"] += float(score_for_log.mean()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("training loader produced no batches")
    out = {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "bce": totals["bce"] / sample_count,
        "risk": totals["risk"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "lambda_t": lambda_t,
        "ramp": extra_ramp,
        "score_mean": totals["score_mean"] / sample_count,
    }
    if blend is not None:
        out["blend_scale"] = blend.scale_value()
    if temperature is not None:
        out["temperature"] = temperature.value()
    return out


def apply_variant_defaults(args):
    args.use_head = args.variant in HEAD_VARIANTS
    args.use_blend = args.variant == "addmsp"
    args.use_temperature = args.variant == "temptrain"
    if args.variant == "addmsp":
        args.risk_weight = 0.0
    elif args.variant == "risk":
        if args.risk_weight <= 0.0:
            args.risk_weight = 0.05
    elif args.variant == "softauc":
        args.risk_weight = 0.0
        if args.softauc_weight <= 0.0:
            args.softauc_weight = 0.5
    elif args.variant == "selnet":
        args.risk_weight = 0.0
    elif args.variant in LOGIT_SCORE_VARIANTS or args.variant == "temptrain":
        args.risk_weight = 0.0
    elif args.variant in ("scsf", "focal", "tcp", "valblend") + DUALAUG_FAMILY:
        args.risk_weight = 0.0
    else:
        raise ValueError(f"unknown variant: {args.variant}")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Fresh SC methods (no acccon)")
    parser.add_argument("-d", "--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--variant", default="valblend", choices=list(FRESH_VARIANTS))
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
    parser.add_argument("--risk-weight", default=0.0, type=float)
    parser.add_argument(
        "--risk-coverages",
        default=parse_coverages("0.80,0.90,0.95"),
        type=parse_coverages,
    )
    parser.add_argument("--soft-temperature", default=0.2, type=float)
    parser.add_argument("--threshold-iterations", default=60, type=int)
    parser.add_argument("--blend-init-alpha", default=-2.0, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--softauc-weight", default=0.0, type=float)
    parser.add_argument("--selnet-coverage", default=0.85, type=float)
    parser.add_argument("--selnet-cov-weight", default=1.0, type=float)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/fresh_sc")
    parser.add_argument("--limit-train-batches", default=0, type=int)
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and less than --epochs")
    if args.risk_weight < 0.0:
        parser.error("--risk-weight must be nonnegative")
    apply_variant_defaults(args)
    return args


def write_train_log(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, test_loader_full, num_classes = get_dataset(
        args
    )

    backbone = VGG16BN_FeatureExtractor(num_classes=num_classes, input_size=32).to(
        device
    )
    confidence_head = (
        RawConfidenceHead(512 * 4, 512, num_classes).to(device)
        if args.use_head
        else None
    )
    blend = (
        ResidualBlend(init_alpha=args.blend_init_alpha).to(device)
        if args.use_blend
        else None
    )
    temperature = LogitTemperature().to(device) if args.use_temperature else None

    backbone_optimizer = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    head_params = []
    if confidence_head is not None:
        head_params += list(confidence_head.parameters())
    if blend is not None:
        head_params += list(blend.parameters())
    if temperature is not None:
        head_params += list(temperature.parameters())
    head_optimizer = (
        optim.Adam(head_params, lr=args.meta_lr) if head_params else None
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer,
        milestones=list(range(25, args.epochs, 25)),
        gamma=0.5,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2)

    best_aurc = math.inf
    train_log_rows = []

    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                backbone,
                confidence_head,
                blend,
                temperature,
                train_loader,
                backbone_optimizer,
                head_optimizer,
                device,
                epoch,
                args,
            )
            scheduler.step()
            # valblend monitors raw SCSF during train; λ chosen only at the end.
            val_variant = "scsf" if args.variant == "valblend" else args.variant
            val_metrics = evaluate_fresh(
                backbone,
                confidence_head,
                blend,
                val_loader,
                device,
                val_variant,
                temperature=temperature,
            )
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history_file.write(json.dumps(jsonable(record)) + "\n")
            history_file.flush()
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
            row.update(
                {
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                    if k != "coverage_errors"
                }
            )
            train_log_rows.append(row)

            blend_msg = (
                f" blend_scale={train_metrics['blend_scale']:.3f}"
                if "blend_scale" in train_metrics
                else ""
            )
            temp_msg = (
                f" T={train_metrics['temperature']:.3f}"
                if "temperature" in train_metrics
                else ""
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} {args.variant} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"bce={train_metrics['bce']:.4f} "
                f"risk={train_metrics['risk']:.4f} "
                f"ramp={train_metrics['ramp']:.3f} "
                f"val_AURC={val_metrics['aurc']:.6f}"
                f"{blend_msg}{temp_msg}",
                flush=True,
            )

            payload = {
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "confidence_head": None
                if confidence_head is None
                else confidence_head.state_dict(),
                "blend": None if blend is None else blend.state_dict(),
                "temperature": None if temperature is None else temperature.state_dict(),
                "args": vars(args),
            }
            torch.save(payload, last_path)
            if val_metrics["aurc"] < best_aurc:
                best_aurc = val_metrics["aurc"]
                torch.save(payload, best_path)

    write_train_log(os.path.join(args.output_dir, "train_log.csv"), train_log_rows)

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    if confidence_head is not None and checkpoint["confidence_head"] is not None:
        confidence_head.load_state_dict(checkpoint["confidence_head"])
    if blend is not None and checkpoint["blend"] is not None:
        blend.load_state_dict(checkpoint["blend"])
    if temperature is not None and checkpoint.get("temperature") is not None:
        temperature.load_state_dict(checkpoint["temperature"])

    valblend_stats = None
    if args.variant == "valblend":
        valblend_stats = select_valblend_lambda(
            backbone, confidence_head, val_loader, device, VALBLEND_LAMBDAS
        )
        print(
            f"valblend selected λ={valblend_stats['lambda']} "
            f"val_AURC={valblend_stats['val_aurc']:.6f}",
            flush=True,
        )

    test = evaluate_fresh(
        backbone,
        confidence_head,
        blend,
        test_loader_full,
        device,
        args.variant,
        valblend_stats=valblend_stats,
        temperature=temperature,
    )
    test_msp = evaluate_fresh(
        backbone, None, None, test_loader_full, device, "msp"
    )
    results = {
        "eval_checkpoint": "last",
        "eval_test_set": "official_10k",
        "variant": args.variant,
        "last_epoch": args.epochs,
        "best_val_aurc": best_aurc,
        "test": test,
        "test_msp": test_msp,
        "args": jsonable(vars(args)),
    }
    if blend is not None:
        results["blend_scale"] = blend.scale_value()
    if temperature is not None:
        results["temperature"] = temperature.value()
    if valblend_stats is not None:
        results["valblend"] = jsonable(valblend_stats)
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(results), handle, indent=2)
    with open(
        os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(jsonable(test), handle, indent=2)

    ce = test["coverage_errors"]
    print(
        f"Evaluated last checkpoint (epoch {args.epochs}) variant={args.variant}; "
        f"official_10k accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f} "
        f"@95={ce[95]:.2f} @90={ce[90]:.2f} @80={ce[80]:.2f} @10={ce[10]:.2f}",
        flush=True,
    )
    ce = test_msp["coverage_errors"]
    print(
        f"Same checkpoint scored by MSP: AURC={test_msp['aurc']:.6f} "
        f"@95={ce[95]:.2f} @90={ce[90]:.2f} @80={ce[80]:.2f} @10={ce[10]:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
