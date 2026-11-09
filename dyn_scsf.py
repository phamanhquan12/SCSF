"""Training-dynamics utilities for DSSPMI / ACCon variants (recipe §§7–15).

This module is side-effect free: it does not change existing trainers until
`train_next_scsf.py` opts in. Official test labels are never consumed here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


EPS = 1e-8
RANK_BANDS = (
    (0, 10),
    (10, 50),
    (50, 70),
    (70, 75),
    (75, 80),
    (80, 85),
    (85, 90),
    (90, 95),
    (95, 100),
)
REPORT_COVERAGES = (100, 95, 90, 85, 80, 75, 70, 50, 25, 10)
ENRICHMENT_KS = (1, 5, 10, 20)

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


@dataclass
class VariantConfig:
    use_acccon: bool = False
    query_weight_mode: str = "none"
    confidence_target: str = "hard"
    record_dynamics: bool = False
    record_el2n: bool = False
    use_depth_probes: bool = False
    append_disagreement: bool = False
    target_alpha: float = 0.0
    forget_gamma: float = 0.5
    margin_temperature: float = 1.0
    ambiguity_beta: float = 0.10
    stability_gamma: float = 2.0
    el2n_beta: float = 0.10
    extra_dim: int = 0


class IndexedDataset(Dataset):
    """Wrap a map-style dataset so each item also returns a stable sample id."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, int(index)


class LinearProbe(nn.Module):
    """Training-only linear classifier on a GAP'd intermediate feature."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features):
        return self.fc(features)


class TrainingDynamicsTracker:
    """Epoch-level rolling history keyed by stable sample ids.

    Current-epoch observations are staged and only committed after the loss
    for that epoch has been computed, so targets never leak the current label.
    """

    def __init__(self, n_samples, window=20):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if window < 1:
            raise ValueError("window must be positive")
        self.n = int(n_samples)
        self.window = int(window)

        self.correct_hist = torch.zeros(self.n, self.window, dtype=torch.uint8)
        self.ptrue_hist = torch.zeros(self.n, self.window, dtype=torch.float32)
        self.margin_hist = torch.zeros(self.n, self.window, dtype=torch.float32)
        self.selector_hist = torch.zeros(self.n, self.window, dtype=torch.float32)
        self.ptr = torch.zeros(self.n, dtype=torch.int64)
        self.count = torch.zeros(self.n, dtype=torch.int64)

        self._epoch_correct = torch.zeros(self.n, dtype=torch.uint8)
        self._epoch_ptrue = torch.zeros(self.n, dtype=torch.float32)
        self._epoch_margin = torch.zeros(self.n, dtype=torch.float32)
        self._epoch_selector = torch.zeros(self.n, dtype=torch.float32)
        self._epoch_seen = torch.zeros(self.n, dtype=torch.bool)

        self.prev_correct = torch.zeros(self.n, dtype=torch.bool)
        self.seen_prev = torch.zeros(self.n, dtype=torch.bool)
        self.forget_count = torch.zeros(self.n, dtype=torch.int32)
        self.learn_count = torch.zeros(self.n, dtype=torch.int32)

        self.el2n_sum = torch.zeros(self.n, dtype=torch.float32)
        self.el2n_n = torch.zeros(self.n, dtype=torch.int32)
        self.el2n_frozen = False

        self.sigma_ref = 1.0
        self.epochs_committed = 0

    def _ids_cpu(self, ids):
        if ids is None:
            raise ValueError("sample ids are required for dynamics tracking")
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        return ids.detach().to(device="cpu", dtype=torch.long)

    @torch.no_grad()
    def observe_batch(self, ids, logits, labels, selector=None):
        ids = self._ids_cpu(ids)
        logits = logits.detach()
        labels = labels.detach()
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        correct = pred.eq(labels)
        ptrue = probs.gather(1, labels[:, None]).squeeze(1)
        true_logit = logits.gather(1, labels[:, None]).squeeze(1)
        masked = logits.clone()
        masked[torch.arange(labels.size(0), device=logits.device), labels] = -torch.inf
        margin = true_logit - masked.max(dim=1).values

        self._epoch_correct[ids] = correct.cpu().to(torch.uint8)
        self._epoch_ptrue[ids] = ptrue.cpu().float()
        self._epoch_margin[ids] = margin.cpu().float()
        if selector is not None:
            self._epoch_selector[ids] = selector.detach().cpu().float()
        self._epoch_seen[ids] = True

    @torch.no_grad()
    def observe_el2n(self, ids, logits, labels):
        if self.el2n_frozen:
            return
        ids = self._ids_cpu(ids)
        probs = logits.detach().softmax(dim=1)
        onehot = F.one_hot(labels.detach(), num_classes=probs.size(1)).to(probs.dtype)
        values = (probs - onehot).norm(dim=1).cpu()
        self.el2n_sum[ids] += values
        self.el2n_n[ids] += 1

    def freeze_el2n(self):
        self.el2n_frozen = True

    @torch.no_grad()
    def commit_epoch(self):
        seen = self._epoch_seen
        if not seen.any():
            return
        correct = self._epoch_correct.bool()
        forgetting = seen & self.seen_prev & self.prev_correct & (~correct)
        learning = seen & self.seen_prev & (~self.prev_correct) & correct
        self.forget_count[forgetting] += 1
        self.learn_count[learning] += 1
        self.prev_correct[seen] = correct[seen]
        self.seen_prev[seen] = True

        ids = seen.nonzero(as_tuple=False).squeeze(1)
        slots = self.ptr[ids] % self.window
        self.correct_hist[ids, slots] = self._epoch_correct[ids]
        self.ptrue_hist[ids, slots] = self._epoch_ptrue[ids]
        self.margin_hist[ids, slots] = self._epoch_margin[ids]
        self.selector_hist[ids, slots] = self._epoch_selector[ids]
        self.ptr[ids] += 1
        self.count[ids] += 1

        self._epoch_correct.zero_()
        self._epoch_ptrue.zero_()
        self._epoch_margin.zero_()
        self._epoch_selector.zero_()
        self._epoch_seen.zero_()
        self.epochs_committed += 1
        self._refresh_sigma_ref()

    def discard_epoch(self):
        self._epoch_correct.zero_()
        self._epoch_ptrue.zero_()
        self._epoch_margin.zero_()
        self._epoch_selector.zero_()
        self._epoch_seen.zero_()

    def has_history(self, ids, min_obs=1):
        ids = self._ids_cpu(ids)
        return self.count[ids] >= int(min_obs)

    def _window_gather(self, hist, ids):
        ids = self._ids_cpu(ids)
        counts = self.count[ids].clamp(max=self.window)
        last = (self.ptr[ids] - 1).remainder(self.window)
        offsets = torch.arange(self.window)
        include = offsets.unsqueeze(0) < counts.unsqueeze(1)
        slots = (last.unsqueeze(1) - offsets.unsqueeze(0)).remainder(self.window)
        gathered = hist[ids.unsqueeze(1), slots]
        return gathered, include

    def _window_mean(self, hist, ids):
        gathered, include = self._window_gather(hist, ids)
        denom = include.sum(dim=1).clamp_min(1)
        return (gathered * include).sum(dim=1) / denom

    def _window_std(self, hist, ids):
        gathered, include = self._window_gather(hist, ids)
        n = include.sum(dim=1)
        mean = self._window_mean(hist, ids)
        var = ((gathered - mean.unsqueeze(1)).square() * include).sum(dim=1) / n.clamp_min(1)
        return torch.where(n >= 2, var.sqrt(), torch.zeros_like(var))

    def temporal_correctness(self, ids, window=None):
        del window
        return self._window_mean(self.correct_hist.float(), ids)

    def mean_confidence(self, ids, window=None):
        del window
        return self._window_mean(self.ptrue_hist, ids)

    def variability(self, ids, window=None):
        del window
        return self._window_std(self.ptrue_hist, ids)

    def mean_margin(self, ids, window=None):
        del window
        return self._window_mean(self.margin_hist, ids)

    def margin_variability(self, ids, window=None):
        del window
        return self._window_std(self.margin_hist, ids)

    def forgetting_count(self, ids):
        return self.forget_count[self._ids_cpu(ids)].to(torch.float32)

    def learning_count(self, ids):
        return self.learn_count[self._ids_cpu(ids)].to(torch.float32)

    def raw_el2n(self, ids):
        ids = self._ids_cpu(ids)
        return self.el2n_sum[ids] / self.el2n_n[ids].clamp_min(1).float()

    def normalized_el2n(self, ids):
        values = self.raw_el2n(ids)
        have = self.el2n_n[self._ids_cpu(ids)] > 0
        if have.any():
            observed = self.raw_el2n(self.el2n_n.nonzero(as_tuple=False).squeeze(1))
            low = observed.min()
            high = observed.max()
            values = (values - low) / (high - low).clamp_min(EPS)
        return values.clamp(0.0, 1.0)

    def last_selector(self, ids):
        gathered, include = self._window_gather(self.selector_hist, ids)
        # offset 0 is the most recent committed value
        return gathered[:, 0]

    def _refresh_sigma_ref(self):
        valid = self.count > 0
        if not valid.any():
            self.sigma_ref = 1.0
            return
        ids = valid.nonzero(as_tuple=False).squeeze(1)
        sigma = self.variability(ids)
        self.sigma_ref = float(torch.quantile(sigma, 0.95).clamp_min(EPS))

    def normalized_ambiguity(self, ids):
        sigma = self.variability(ids)
        return (sigma / (self.sigma_ref + EPS)).clamp(0.0, 1.0)

    def stability(self, ids, gamma):
        return torch.exp(-float(gamma) * self.normalized_ambiguity(ids))

    def to_numpy_summary(self):
        ids = torch.arange(self.n)
        valid = self.count > 0
        return {
            "count": self.count.numpy(),
            "temporal_correctness": self.temporal_correctness(ids).numpy(),
            "mean_confidence": self.mean_confidence(ids).numpy(),
            "variability": self.variability(ids).numpy(),
            "mean_margin": self.mean_margin(ids).numpy(),
            "margin_variability": self.margin_variability(ids).numpy(),
            "forget_count": self.forget_count.numpy(),
            "learn_count": self.learn_count.numpy(),
            "el2n": self.normalized_el2n(ids).numpy(),
            "el2n_raw": self.raw_el2n(ids).numpy(),
            "last_selector": self.last_selector(ids).numpy(),
            "valid": valid.numpy(),
            "sigma_ref": np.array(self.sigma_ref),
            "epochs_committed": np.array(self.epochs_committed),
        }


def make_query_weight(w, mode, beta=0.1, floor=0.2, ambiguity=None):
    if mode in ("none", "uniform"):
        return torch.ones_like(w)
    if mode == "acceptance":
        return w
    if mode == "boundary":
        return w + float(beta) * 4.0 * w * (1.0 - w)
    if mode == "floor":
        alpha = float(floor)
        return alpha + (1.0 - alpha) * w
    if mode == "cartography":
        if ambiguity is None:
            raise ValueError("cartography query weights require ambiguity")
        return w + float(beta) * w * ambiguity
    raise ValueError(f"unknown query weight mode: {mode}")


def make_confidence_target(
    hard_correct,
    ids,
    tracker,
    mode,
    alpha=0.0,
    forget_gamma=0.5,
    margin_temperature=1.0,
):
    """Soft/hard confidence-head targets from previous-epoch dynamics only."""

    hard = hard_correct.to(dtype=torch.float32)
    if mode == "hard" or tracker is None or ids is None:
        return hard
    valid = tracker.has_history(ids).to(device=hard.device)
    if not valid.any() and mode != "hard":
        return hard

    temp = tracker.temporal_correctness(ids).to(device=hard.device, dtype=hard.dtype)
    if mode == "temporal":
        return torch.where(valid, temp, hard)
    if mode == "hybrid":
        mixed = float(alpha) * hard + (1.0 - float(alpha)) * temp
        return torch.where(valid, mixed, hard)
    if mode == "forget":
        n_forget = tracker.forgetting_count(ids).to(device=hard.device, dtype=hard.dtype)
        stability = torch.exp(-float(forget_gamma) * n_forget)
        return torch.where(valid, temp * stability, hard)
    if mode == "margin":
        margin = tracker.mean_margin(ids).to(device=hard.device, dtype=hard.dtype)
        soft = torch.sigmoid(margin / float(margin_temperature))
        return torch.where(valid, soft, hard)
    raise ValueError(f"unknown confidence target: {mode}")


def apply_cartography_weights(accept, tracker, ids, beta, gamma, device):
    query = accept
    key = accept
    if tracker is None or ids is None:
        return query, key
    valid = tracker.has_history(ids, min_obs=2).to(device=device)
    if not valid.any():
        return query, key
    ambiguity = tracker.normalized_ambiguity(ids).to(device=device, dtype=accept.dtype)
    stability = tracker.stability(ids, gamma).to(device=device, dtype=accept.dtype)
    carto_q = make_query_weight(accept, "cartography", beta=beta, ambiguity=ambiguity)
    carto_k = accept * stability
    query = torch.where(valid, carto_q, query)
    key = torch.where(valid, carto_k, key)
    return query, key


def apply_el2n_weights(accept, tracker, ids, beta, device):
    query = accept
    key = accept
    if tracker is None or ids is None:
        return query, key
    have = (tracker.el2n_n[tracker._ids_cpu(ids)] > 0).to(device=device)
    if not have.any():
        return query, key
    el2n = tracker.normalized_el2n(ids).to(device=device, dtype=accept.dtype)
    el2n_q = accept + float(beta) * el2n * accept
    el2n_k = accept * (1.0 - el2n)
    query = torch.where(have, el2n_q, query)
    key = torch.where(have, el2n_k, key)
    return query, key


def depth_disagreement(logits3, logits4, logits5, logits):
    final = logits.argmax(dim=1)
    votes = (
        logits3.argmax(dim=1).ne(final).to(logits.dtype)
        + logits4.argmax(dim=1).ne(final).to(logits.dtype)
        + logits5.argmax(dim=1).ne(final).to(logits.dtype)
    )
    return votes / 3.0


def make_probes(num_classes, device):
    return nn.ModuleDict(
        {
            "pool3": LinearProbe(256, num_classes),
            "pool4": LinearProbe(2048, num_classes),
            "pool5": LinearProbe(512, num_classes),
        }
    ).to(device)


def unpack_batch(batch, device):
    inputs = batch[0].to(device, non_blocking=True)
    targets = batch[1].to(device, non_blocking=True)
    if len(batch) >= 3:
        ids = batch[2]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        ids = ids.to(device="cpu", dtype=torch.long)
    else:
        ids = None
    return inputs, targets, ids


def data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data")


def build_train_eval_loader(dataset_name, batch_size, workers):
    """Deterministic training-set loader with test transforms (recipe Option B)."""

    if dataset_name not in CIFAR_STATS:
        raise ValueError(f"train-eval loader is only defined for {tuple(CIFAR_STATS)}")
    mean, std = CIFAR_STATS[dataset_name]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    ctor = datasets.CIFAR100 if dataset_name == "cifar100" else datasets.CIFAR10
    raw = ctor(root=data_root(), train=True, download=False, transform=transform)
    return DataLoader(
        IndexedDataset(raw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def wrap_train_loader_ids(train_loader):
    """Return a new loader that also yields stable sample ids.

    RandomSampler draws its permutation at iteration time, so rebuilding here
    does not consume RNG or change the first-epoch shuffle relative to the
    original loader.
    """

    dataset = train_loader.dataset
    if not isinstance(dataset, IndexedDataset):
        dataset = IndexedDataset(dataset)
    return DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, "pin_memory", True),
    )


def binary_auroc(scores, positive):
    positive = positive.to(dtype=torch.bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, descending=False, stable=True)
    ranks = torch.empty(scores.size(0), dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.size(0) + 1, dtype=torch.float64)
    u = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def rank_band_analysis(scores, correctness):
    order = torch.argsort(scores, descending=True, stable=True)
    errors = (~correctness[order]).to(torch.int64)
    n = scores.numel()
    bands = {}
    for low, high in RANK_BANDS:
        start = int(n * low / 100)
        end = max(start + 1, int(n * high / 100)) if high < 100 else n
        sl = errors[start:end]
        n_err = int(sl.sum())
        n_samples = int(sl.numel())
        bands[f"{low}-{high}"] = {
            "n_samples": n_samples,
            "n_errors": n_err,
            "slice_error_rate": n_err / n_samples if n_samples else None,
        }
    return bands


def enrichment_metrics(scores, correctness):
    order = torch.argsort(scores, descending=True, stable=True)
    errors = ~correctness[order]
    n = scores.numel()
    n_err = int(errors.sum())
    payload = {}
    for k in ENRICHMENT_KS:
        top_n = max(1, int(round(n * k / 100)))
        bottom_n = max(1, int(round(n * k / 100)))
        top_err = int(errors[:top_n].sum())
        bottom_err = int(errors[-bottom_n:].sum())
        payload[f"top{k}_purity"] = 1.0 - top_err / top_n
        payload[f"bottom{k}_error_rate"] = bottom_err / bottom_n
        payload[f"bottom{k}_capture"] = bottom_err / n_err if n_err else None
    error_ranks = torch.nonzero(errors, as_tuple=False).squeeze(1).to(torch.float64) + 1
    if error_ranks.numel():
        payload["error_rank_mean"] = float(error_ranks.mean() / n)
        payload["error_rank_median"] = float(error_ranks.median() / n)
    else:
        payload["error_rank_mean"] = None
        payload["error_rank_median"] = None
    return payload


@torch.no_grad()
def collect_scores(
    backbone,
    confidence_head,
    loader,
    device,
    probes=None,
    extra_dim=0,
    use_ds_score=False,
    ds_branch=None,
    append_msp=False,
):
    backbone.eval()
    confidence_head.eval()
    if probes is not None:
        probes.eval()
    if ds_branch is not None:
        ds_branch.eval()
    scores = []
    correctness = []
    disagreements = []
    for batch in loader:
        inputs, targets, _ = unpack_batch(batch, device)
        extra = None
        if use_ds_score:
            logits, spatial3, spatial4, spatial5 = backbone(
                inputs, return_features=True, return_spatial=True
            )
            fused, _ = confidence_head(spatial3, spatial4, spatial5, logits)
            scores.append(fused.cpu())
        elif ds_branch is not None:
            logits, spatial3, spatial4, spatial5 = backbone(
                inputs, return_features=True, return_spatial=True
            )
            pool4 = F.adaptive_avg_pool2d(spatial4, 2).flatten(1)
            pool5 = F.adaptive_avg_pool2d(spatial5, 1).flatten(1)
            branch_out = ds_branch(spatial3, spatial4, spatial5)
            d = depth_disagreement(
                branch_out["logits3"],
                branch_out["logits4"],
                branch_out["logits5"],
                logits,
            )
            disagreements.append(d.cpu())
            if extra_dim > 0:
                extra = d.unsqueeze(1)
            scores.append(confidence_head(pool4, pool5, logits, extra=extra).cpu())
        elif probes is not None:
            logits, pool4, pool5, pool3 = backbone(
                inputs, return_features=True, return_pool3=True
            )
            logits3 = probes["pool3"](pool3)
            logits4 = probes["pool4"](pool4)
            logits5 = probes["pool5"](pool5)
            d = depth_disagreement(logits3, logits4, logits5, logits)
            disagreements.append(d.cpu())
            if extra_dim > 0:
                extra = d.unsqueeze(1)
            scores.append(confidence_head(pool4, pool5, logits, extra=extra).cpu())
        else:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            if append_msp and extra_dim > 0:
                msp = F.softmax(logits, dim=1).max(dim=1).values
                extra = msp.unsqueeze(1)
            scores.append(confidence_head(pool4, pool5, logits, extra=extra).cpu())
        correctness.append(logits.argmax(dim=1).eq(targets).cpu())
    payload = {
        "scores": torch.cat(scores),
        "correctness": torch.cat(correctness),
    }
    if disagreements:
        payload["disagreement"] = torch.cat(disagreements)
    return payload


def evaluate_selective(
    backbone,
    confidence_head,
    loader,
    device,
    probes=None,
    extra_dim=0,
    use_ds_score=False,
    ds_branch=None,
    append_msp=False,
):
    collected = collect_scores(
        backbone,
        confidence_head,
        loader,
        device,
        probes=probes,
        extra_dim=extra_dim,
        use_ds_score=use_ds_score,
        ds_branch=ds_branch,
        append_msp=append_msp,
    )
    scores = collected["scores"]
    correctness = collected["correctness"]
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_errors = (~correctness[order]).float()
    prefix = sorted_errors.cumsum(0) / torch.arange(1, len(scores) + 1, dtype=torch.float32)
    accuracy = float(correctness.float().mean())
    aurc = float(prefix.mean())
    coverage_errors = {}
    for coverage in REPORT_COVERAGES:
        count = max(1, int(len(scores) * coverage / 100))
        coverage_errors[coverage] = 100.0 * float(prefix[count - 1])
    metrics = {
        "samples": int(len(scores)),
        "accuracy": 100.0 * accuracy,
        "aurc": aurc,
        "eaurc": aurc - (1.0 - accuracy),
        "auroc": binary_auroc(scores, correctness),
        "coverage_errors": coverage_errors,
        "bands": rank_band_analysis(scores, correctness),
        "enrichment": enrichment_metrics(scores, correctness),
    }
    if "disagreement" in collected:
        d = collected["disagreement"]
        wrong = ~correctness
        metrics["depth"] = {
            "mean_disagreement": float(d.mean()),
            "corr_disagreement_error": _corr(d, wrong.float()),
            "error_by_disagreement": _error_by_disagreement(d, wrong),
        }
    return metrics


def _corr(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() < 3 or float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return None
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def _error_by_disagreement(disagreement, wrong):
    table = {}
    for value in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        mask = (disagreement - value).abs() < 1e-5
        n = int(mask.sum())
        table[f"{value:.2f}"] = {
            "n_samples": n,
            "error_rate": float(wrong[mask].float().mean()) if n else None,
        }
    return table


def dynamics_correlations(tracker):
    summary = tracker.to_numpy_summary()
    valid = summary["valid"]
    payload = {
        "n_tracked": int(valid.sum()),
        "epochs_committed": int(summary["epochs_committed"]),
        "sigma_ref": float(summary["sigma_ref"]),
        "mean_forget": float(summary["forget_count"][valid].mean()) if valid.any() else None,
        "mean_temporal": float(summary["temporal_correctness"][valid].mean())
        if valid.any()
        else None,
    }
    pairs = (
        ("temporal_correctness", "mean_margin"),
        ("temporal_correctness", "last_selector"),
        ("forget_count", "temporal_correctness"),
        ("forget_count", "last_selector"),
        ("variability", "last_selector"),
        ("el2n", "variability"),
        ("el2n", "forget_count"),
        ("el2n", "last_selector"),
        ("el2n", "mean_margin"),
        ("mean_confidence", "variability"),
    )
    for left, right in pairs:
        a = torch.from_numpy(summary[left][valid]).float()
        b = torch.from_numpy(summary[right][valid]).float()
        payload[f"corr_{left}_{right}"] = _corr(a, b)
    return payload


def save_dynamics_outputs(output_dir, tracker, extra_json=None):
    os.makedirs(output_dir, exist_ok=True)
    summary = tracker.to_numpy_summary()
    np.savez_compressed(os.path.join(output_dir, "dynamics_summary.npz"), **summary)
    analysis = dynamics_correlations(tracker)
    if extra_json:
        analysis.update(extra_json)
    with open(os.path.join(output_dir, "band_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    _try_dynamics_plots(output_dir, summary)
    return analysis


def _try_dynamics_plots(output_dir, summary):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    valid = summary["valid"]
    if not valid.any():
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        summary["mean_confidence"][valid],
        summary["variability"][valid],
        c=summary["temporal_correctness"][valid],
        s=4,
        cmap="viridis",
        alpha=0.4,
    )
    ax.set_xlabel("mean true-class confidence")
    ax.set_ylabel("confidence variability")
    ax.set_title("Dataset cartography (train dynamics)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cartography.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        summary["temporal_correctness"][valid],
        summary["last_selector"][valid],
        s=4,
        alpha=0.35,
    )
    ax.set_xlabel("rolling correctness")
    ax.set_ylabel("last selector score")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temporal_vs_selector.png"), dpi=140)
    plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def variant_defaults():
    return {
        "temporal_target": VariantConfig(
            confidence_target="temporal",
            record_dynamics=True,
        ),
        "temporal_hybrid": VariantConfig(
            confidence_target="hybrid",
            record_dynamics=True,
            target_alpha=0.5,
        ),
        "forget_target": VariantConfig(
            confidence_target="forget",
            record_dynamics=True,
            forget_gamma=0.5,
        ),
        "margin_target": VariantConfig(
            confidence_target="margin",
            record_dynamics=True,
            margin_temperature=1.0,
        ),
        "carto_acccon": VariantConfig(
            use_acccon=True,
            query_weight_mode="cartography",
            record_dynamics=True,
            ambiguity_beta=0.10,
            stability_gamma=2.0,
        ),
        "el2n_acccon": VariantConfig(
            use_acccon=True,
            query_weight_mode="el2n",
            record_el2n=True,
            el2n_beta=0.10,
        ),
        "depth_diag": VariantConfig(
            use_depth_probes=True,
        ),
        "depth_score": VariantConfig(
            use_depth_probes=True,
            append_disagreement=True,
            extra_dim=1,
        ),
        "temporal_depth": VariantConfig(
            confidence_target="temporal",
            record_dynamics=True,
            use_depth_probes=True,
            append_disagreement=True,
            extra_dim=1,
        ),
    }


def apply_dyn_config(args, config: VariantConfig):
    args.confidence_target = config.confidence_target
    args.record_dynamics = config.record_dynamics
    args.record_el2n = config.record_el2n
    args.use_depth_probes = config.use_depth_probes
    args.append_disagreement = config.append_disagreement
    args.query_weight_mode = config.query_weight_mode
    args.target_alpha = config.target_alpha
    args.forget_gamma = config.forget_gamma
    args.margin_temperature = config.margin_temperature
    args.ambiguity_beta = config.ambiguity_beta
    args.stability_gamma = config.stability_gamma
    args.el2n_beta = config.el2n_beta
    args.extra_dim = config.extra_dim
    if config.use_acccon:
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.con_mode = "accept"
        if args.con_weight <= 0.0:
            args.con_weight = 0.5
        args.coverages = [0.80, 0.90, 0.95]
    else:
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.con_weight = 0.0
        args.tail_weight = 0.0
        args.con_mode = "accept"
    return args


def config_dict(config: VariantConfig):
    return asdict(config)
