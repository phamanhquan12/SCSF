from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_COVERAGES = (100, 99, 98, 97, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10)


@dataclass(frozen=True)
class CoverageResult:
    coverage: float
    error: float
    accuracy: float
    selected: int


@dataclass(frozen=True)
class MetricSummary:
    accuracy: float
    aurc: float
    oracle_aurc: float
    eaurc: float
    naurc: float
    auroc: float
    fpr_at_95_tpr: float
    full_risk: float
    num_samples: int


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return pred.eq(targets).float().mean().mul(100.0).item()


def risk_coverage_curve(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
    coverages: Iterable[float] = DEFAULT_COVERAGES,
) -> list[CoverageResult]:
    logits_np = logits.detach().cpu()
    targets_np = targets.detach().cpu()
    confidence_np = confidence.detach().cpu().float()

    correct = logits_np.argmax(dim=1).eq(targets_np).numpy().astype(np.float32)
    order = np.argsort(-confidence_np.numpy())
    correct_sorted = correct[order]
    n = len(correct_sorted)
    results: list[CoverageResult] = []

    for coverage in coverages:
        selected = max(1, int(round((coverage / 100.0) * n)))
        acc = float(correct_sorted[:selected].mean() * 100.0)
        results.append(
            CoverageResult(
                coverage=float(coverage),
                error=100.0 - acc,
                accuracy=acc,
                selected=selected,
            )
        )
    return results


def aurc(results: Iterable[CoverageResult]) -> float:
    by_cov = sorted(results, key=lambda item: item.coverage, reverse=True)
    area = 0.0
    for left, right in zip(by_cov, by_cov[1:]):
        width = (left.coverage - right.coverage) / 100.0
        area += ((left.error / 100.0) + (right.error / 100.0)) * 0.5 * width
    return area


def per_sample_predictions(logits: torch.Tensor, targets: torch.Tensor, confidence: torch.Tensor) -> dict[str, np.ndarray]:
    logits_cpu = logits.detach().cpu().float()
    targets_cpu = targets.detach().cpu().long()
    confidence_np = confidence.detach().cpu().float().numpy()
    probs = F.softmax(logits_cpu, dim=1)
    losses = F.cross_entropy(logits_cpu, targets_cpu, reduction="none").numpy()
    pred = logits_cpu.argmax(dim=1)
    correctness = pred.eq(targets_cpu).numpy().astype(np.int64)
    return {
        "target": targets_cpu.numpy(),
        "prediction": pred.numpy(),
        "confidence": confidence_np,
        "correct": correctness,
        "loss": losses,
        "softmax_response": probs.max(dim=1).values.numpy(),
    }


def exact_aurc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    if confidence.size == 0:
        return 0.0
    order = np.argsort(-confidence)
    errors = 1.0 - correctness[order].astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(cumulative_risk.mean())


def oracle_aurc(correctness: np.ndarray) -> float:
    if correctness.size == 0:
        return 0.0
    order = np.argsort(-(correctness.astype(np.float64)))
    errors = 1.0 - correctness[order].astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(cumulative_risk.mean())


def naurc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    actual = exact_aurc(confidence, correctness)
    optimal = oracle_aurc(correctness)
    full_risk = float((1.0 - correctness.astype(np.float64)).mean())
    denom = full_risk - optimal
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((actual - optimal) / denom, 0.0, 1.0))


def roc_curve_points(confidence: np.ndarray, correctness: np.ndarray) -> list[dict[str, float]]:
    if confidence.size == 0:
        return [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    order = np.argsort(-confidence)
    sorted_scores = confidence[order]
    sorted_correct = correctness[order].astype(np.int64)
    positives = max(1, int(sorted_correct.sum()))
    negatives = max(1, int(sorted_correct.size - sorted_correct.sum()))
    tp = 0
    fp = 0
    rows = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    last_score = None
    for score, correct in zip(sorted_scores, sorted_correct):
        if last_score is not None and float(score) != float(last_score):
            rows.append({"threshold": float(last_score), "fpr": fp / negatives, "tpr": tp / positives})
        if correct:
            tp += 1
        else:
            fp += 1
        last_score = score
    rows.append({"threshold": float(last_score), "fpr": fp / negatives, "tpr": tp / positives})
    return rows


def auroc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    if np.unique(correctness).size < 2:
        return 0.5
    rows = roc_curve_points(confidence, correctness)
    fpr = np.asarray([row["fpr"] for row in rows], dtype=np.float64)
    tpr = np.asarray([row["tpr"] for row in rows], dtype=np.float64)
    return float(np.trapezoid(tpr, fpr))


def fpr_at_tpr(confidence: np.ndarray, correctness: np.ndarray, target_tpr: float = 0.95) -> float:
    if np.unique(correctness).size < 2:
        return 1.0
    for row in roc_curve_points(confidence, correctness):
        if row["tpr"] >= target_tpr:
            return float(row["fpr"])
    return 1.0


def summarize_selective_metrics(logits: torch.Tensor, targets: torch.Tensor, confidence: torch.Tensor) -> MetricSummary:
    per_sample = per_sample_predictions(logits, targets, confidence)
    confidence_np = per_sample["confidence"].astype(np.float64)
    correctness = per_sample["correct"].astype(np.float64)
    actual_aurc = exact_aurc(confidence_np, correctness)
    best_aurc = oracle_aurc(correctness)
    full_risk = float((1.0 - correctness).mean())
    denom = full_risk - best_aurc
    eaurc = max(0.0, actual_aurc - best_aurc)
    return MetricSummary(
        accuracy=float(correctness.mean() * 100.0),
        aurc=actual_aurc,
        oracle_aurc=best_aurc,
        eaurc=eaurc,
        naurc=0.0 if denom <= 1e-12 else float(np.clip(eaurc / denom, 0.0, 1.0)),
        auroc=auroc(confidence_np, correctness),
        fpr_at_95_tpr=fpr_at_tpr(confidence_np, correctness, 0.95),
        full_risk=full_risk,
        num_samples=int(correctness.size),
    )


def format_curve(results: Iterable[CoverageResult]) -> str:
    return " ".join(f"{r.coverage:.0f}:{r.error:.2f}" for r in results)
