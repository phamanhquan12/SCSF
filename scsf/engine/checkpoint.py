"""Checkpoint manager with the contract selection rule.

Selection rule (empirical contract §5): among all evaluated epochs, pick the
one with the **lowest validation AURC** among epochs whose validation accuracy
is within ``guard_delta_acc`` percentage points of the best validation accuracy
seen so far. The chosen epoch's model is the deployment checkpoint.
"""

from __future__ import annotations

import json
import os

import torch

GUARD_FALLBACK_DELTA = 1.0


def _sched_state(scheduler):
    try:
        return {"state_dict": scheduler.state_dict()}
    except Exception:
        return {}


def _build_scheduler(cfg, optimizer):
    spec = cfg.scheduler_spec()
    kind = spec["kind"]
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(spec["epochs"]), 1)
        )
    if kind == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=spec["milestones"] or [spec["epochs"] // 2], gamma=spec["gamma"]
        )
    if kind == "constant":
        class _Const(torch.optim.lr_scheduler.LRScheduler):
            def get_lr(self):
                return [g["lr"] for g in self.optimizer.param_groups]
        return _Const(optimizer)
    raise ValueError(f"unknown scheduler {kind!r}")


class SelectionTracker:
    """Tracks the best selected epoch under the contract rule."""

    def __init__(self, guard_delta_acc: float = GUARD_FALLBACK_DELTA):
        self.guard = float(guard_delta_acc)
        self.best_acc = float("-inf")
        self.best_aurc = float("inf")
        self.selected_epoch = None
        self.best_epoch = None
        self.candidates = []

    def update(self, epoch: int, val_metrics: dict) -> bool:
        acc = float(val_metrics["acc"]) * 100.0
        aurc = float(val_metrics["aurc"])
        self.candidates.append((epoch, acc, aurc))
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
        # eligible = within guard of best acc; pick lowest AURC (ties: earliest)
        eligible = [c for c in self.candidates if c[1] >= self.best_acc - self.guard]
        if not eligible:
            return False
        pick = min(eligible, key=lambda c: (c[2], c[0]))
        changed = pick[0] != self.selected_epoch
        self.selected_epoch = pick[0]
        self.best_aurc = pick[2]
        return changed

    def summary(self):
        return {
            "best_acc": None if self.best_epoch is None else self.best_acc,
            "best_acc_epoch": self.best_epoch,
            "selected_epoch": self.selected_epoch,
            "best_selected_aurc": None if self.selected_epoch is None else self.best_aurc,
            "selection_rule": f"min_val_aurc_among_acc>=best_acc-{self.guard}pp",
        }


class CheckpointManager:
    EXT = ".pt"

    def __init__(self, run_dir: str, mode: str = "selected"):
        self.run_dir = run_dir
        self.mode = mode  # 'selected' | 'last'
        os.makedirs(run_dir, exist_ok=True)

    def ckpt_path(self, tag: str) -> str:
        return os.path.join(self.run_dir, f"{tag}{self.EXT}")

    def save(self, tag: str, payload: dict) -> str:
        path = self.ckpt_path(tag)
        torch.save(payload, path)
        return path

    def best_path(self) -> str:
        return self.ckpt_path("best")

    def last_path(self) -> str:
        return self.ckpt_path("last")

    def epoch_path(self, epoch: int) -> str:
        return self.ckpt_path(f"epoch_{epoch:03d}")

    def load(self, tag: str, map_location="cpu"):
        return torch.load(self.ckpt_path(tag), map_location=map_location, weights_only=False)

    def exists(self, tag: str) -> bool:
        return os.path.exists(self.ckpt_path(tag))

    def deployment_path(self, selection: str) -> str:
        return self.ckpt_path(selection if selection in ("best", "last") else f"epoch_{selection:03d}")