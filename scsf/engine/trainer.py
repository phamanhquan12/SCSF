"""Trainer: exact-resume, deterministic training loop driven by Method hooks.

Unlike a generic engine, only the ``Method`` interface is exercised; the
history (loss term names, gradient semantics, scheduler policy) belongs to the
methods. Everything here is about reproducibility: persistent generators,
RNG snapshots in every checkpoint, and the contract selection rule.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import torch

from ..data import assert_no_official_test_leakage, get_split, split_hashes
from ..data.cifar import build_dataloader
from ..methods import build_method
from ..metrics import all_metrics, selective_risk_at_coverages
from ..version import __version__, package_versions
from .checkpoint import CheckpointManager, SelectionTracker, _build_scheduler
from .seeding import capture_global_state, make_generator, restore_global_state, seed_all
from .registry import BASE_COLUMNS
from .config import config_hash


def _build_optimizers(method, cfg):
    train = cfg["train"]
    opts = []
    for spec in method.optimizer_specs():
        params = list(spec["params"])
        if not params:
            continue
        base = {
            "lr": float(spec["lr"]),
            "weight_decay": float(spec.get("weight_decay", 0.0)),
            "momentum": float(spec.get("momentum", 0.0)),
        }
        kind = str(spec.get("kind", train.get("optimizer", "sgd")))
        if kind == "sgd":
            sgd_base = dict(base)
            sgd_base["momentum"] = float(spec.get("momentum", 0.0))
            sgd_base["nesterov"] = bool(spec.get("nesterov", False))
            opts.append(torch.optim.SGD(params, **sgd_base))
        elif kind == "adam":
            adam_base = dict(base)
            adam_base.pop("momentum", None)
            adam_base["betas"] = (float(spec.get("momentum", 0.9)), 0.999)
            opts.append(torch.optim.Adam(params, **adam_base))
        elif kind == "adamw":
            adamw_base = dict(base)
            adamw_base.pop("momentum", None)
            adamw_base["betas"] = (float(spec.get("momentum", 0.9)), 0.999)
            opts.append(torch.optim.AdamW(params, **adamw_base))
        else:
            raise ValueError(f"unsupported optimizer kind {kind!r}")
    if not opts:
        raise RuntimeError("method.optimizer_specs() returned no parameter groups")
    return opts


class Trainer:
    def __init__(self, cfg: dict, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.train_cfg = cfg["train"]
        self.device = torch.device(self.train_cfg["device"])
        self.manager = CheckpointManager(run_dir)
        self.method = None
        self.optimizers = []
        self.scheduler = None
        self.selection = None
        self.epoch = 0
        self.batch_index = 0

    def _build(self):
        if getattr(self, "_built", False):
            return
        seed_all(self.cfg["train"]["seed"])
        split = get_split(self.cfg)
        assert_no_official_test_leakage(split)
        self.split = split
        self.split_hashes = split_hashes(split)

        self.method = build_method(self.cfg["method_name"], self.cfg)
        self.method.to(self.device)
        self.optimizers = _build_optimizers(self.method, self.cfg)
        self.scheduler = _build_scheduler(self.method, self.optimizers[0])
        self.selection = SelectionTracker(self.train_cfg.get("guard_delta_acc", 1.0))

        # persistent generator: its consumed state is the data-order contract
        self.generator = make_generator(self.cfg["train"]["data_order_seed"])
        self.train_loader = build_dataloader(
            self.cfg, "train", generator=self.generator,
            return_indices=self.method.needs_indices,
            overfit=int(self.cfg["train"].get("overfit", 0)),
        )
        self.val_loader = build_dataloader(self.cfg, "val", shuffle=False, return_indices=True)
        self._built = True

    # -- evaluation --------------------------------------------------------
    def _eval_val(self) -> dict:
        self.method.eval()
        labels, preds, confs, ids = [], [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch[0], batch[1]
                mp = self.method.predict_batch(x.to(self.device))
                labels.append(y.numpy())
                preds.append(mp.prediction.detach().cpu().numpy())
                confs.append(mp.confidence.detach().cpu().numpy())
                ids.append(np.asarray(batch[2]))
        labels = np.concatenate(labels)
        metrics = all_metrics(labels, np.concatenate(preds), np.concatenate(confs),
                              np.concatenate(ids), self.cfg["data"]["num_classes"])
        for c in selective_risk_at_coverages(labels, np.concatenate(preds), np.concatenate(confs),
                                             np.concatenate(ids)):
            metrics[f"risk_at_cov_{int(c['coverage'])}"] = float(c["risk"])
        self.method.train()
        return metrics

    # -- checkpointing -----------------------------------------------------
    def _payload(self, tag: str, val_metrics: dict | None):
        state = capture_global_state(generator=self.generator)
        return {
            "scsf_version": __version__,
            "cfg": self.cfg,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "model_state": self.method.state_dict(),
            "optimizer_states": [o.state_dict() for o in self.optimizers],
            "scheduler_state": self.scheduler.state_dict(),
            "rng": state,
            "val_metrics": val_metrics,
            "selection_state": self.selection.state(),
            "split_hashes": self.split_hashes,
        }

    def _save_epoch_snapshot(self):
        if (self.epoch + 1) % int(self.train_cfg.get("save_every", 5)) != 0:
            return
        payload = self._payload(f"epoch_{self.epoch:03d}", None)
        self.manager.save(f"epoch_{self.epoch:03d}", payload)
        # Disk budget: keep the newest epoch snapshot (for exact resume after a
        # crash) plus the currently-selected epoch snapshot (so the contract's
        # selected checkpoint is always reconstructable). best/selected/last
        # are separate single files. Everything else is pruned.
        self._prune_old_epoch_snapshots()

    def _prune_old_epoch_snapshots(self):
        import glob
        pat = os.path.join(self.run_dir, "epoch_*.pt")
        tags = []
        for p in glob.glob(pat):
            t = os.path.basename(p)[:-3]
            if t[6:].isdigit():
                tags.append(t)
        if len(tags) <= 1:
            return
        newest = max(tags, key=lambda t: int(t[6:]))
        keep = {newest}
        sel = self.selection.selected_epoch
        if sel is not None:
            keep.add(f"epoch_{sel:03d}")
        for t in tags:
            if t not in keep:
                try:
                    os.remove(self.manager.ckpt_path(t))
                except OSError:
                    pass

    def _manifest(self) -> dict:
        from importlib import import_module
        versions = package_versions(import_module("torch"), import_module("torchvision"),
                                    import_module("timm"), np)
        commit, dirty = _git_state()
        n_params = sum(p.numel() for m in self.method.inference_modules() for p in m.parameters())
        return {
            **versions,
            "seed": int(self.cfg["train"]["seed"]),
            "device": str(self.device),
            "commit": commit,
            "dirty": dirty,
            "split_hashes": self.split_hashes,
            "config_hash": config_hash(self.cfg),
            "params_total": int(n_params),
            "logits": self.cfg["method_name"],
            "selection": self.selection.summary(),
        }

    # -- run -----------------------------------------------------------------
    def run(self, resume_from: str | None = None) -> dict:
        self._build()
        os.makedirs(self.run_dir, exist_ok=True)
        # Write the fully-resolved config at the start so a crashed run keeps a
        # reproducible config (and resume uses the exact same cfg hash).
        with open(os.path.join(self.run_dir, "cfg.json"), "w") as f:
            json.dump(self.cfg, f, indent=2, sort_keys=True, default=str)

        start_epoch = 0
        if resume_from:
            payload = self.manager.load(resume_from, map_location=self.device)
            self.method.load_state_dict(payload["model_state"])
            for o, s in zip(self.optimizers, payload.get("optimizer_states", [])):
                o.load_state_dict(s)
            self.scheduler.load_state_dict(payload["scheduler_state"])
            restore_global_state(payload["rng"], self.generator)
            if "selection_state" in payload:
                self.selection.restore(payload["selection_state"])
            elif "selection_summary" in payload:
                self.selection.selected_epoch = payload["selection_summary"]["selected_epoch"]
            start_epoch = int(payload.get("epoch", 0))
            if int(payload.get("batch_index", 0)) == 0:
                # checkpoint sits at an epoch boundary: resume from its start.
                # (Every live trainer snapshot is epoch-final; the generator
                # state was captured at the end of that epoch, so the next
                # draw is exactly the first batch of `start_epoch`.)
                start_epoch += 1
            else:
                # mid-epoch snapshot: keep the epoch and resume from the batch
                # after the saved one; the restored generator is already in the
                # right position, so no replay is needed.
                self.batch_index = int(payload.get("batch_index", 0))
            self.epoch = start_epoch - 1

        best_seen = {"acc": -1.0}
        final_metrics = None

        for epoch in range(start_epoch, int(self.cfg["train"]["epochs"])):
            self.epoch = epoch
            self.method.on_epoch_start(epoch)
            self.method.train()
            # On a mid-epoch resume, keep counting the current epoch's batches
            # from the resume position so batch-cadence triggers stay aligned.
            batch_off = self.batch_index if epoch == start_epoch else 0
            for bi_raw, batch in enumerate(self.train_loader):
                bi = bi_raw + batch_off
                self.batch_index = bi
                loss_dict = self.method.train_loss(
                    tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch), self
                )
                nonfinite = [
                    name for name, value in loss_dict.items()
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                ]
                if nonfinite:
                    raise FloatingPointError(
                        f"non-finite loss terms at epoch={epoch}, batch={bi}: "
                        f"{', '.join(nonfinite)}")
                total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=True)
                total.backward()
                for opt in self.optimizers:
                    opt.step()
            self.batch_index = 0
            self.scheduler.step()
            self.method.on_epoch_end(epoch, {})

            val_metrics = None
            if (epoch + 1) % int(self.train_cfg.get("eval_every", 1)) == 0 or epoch == int(self.cfg["train"]["epochs"]) - 1:
                val_metrics = self._eval_val()
                acc = float(val_metrics["acc"]) * 100.0
                changed = self.selection.update(epoch, val_metrics)
                if acc > best_seen["acc"]:
                    best_seen = {"acc": acc, "epoch": epoch}
                    self.manager.save("best", self._payload("best", val_metrics))
                self._save_epoch_snapshot()
                if changed:
                    # selection epoch == current epoch -> snapshot now; otherwise
                    # copy from the epoch snapshot (kept on the save_every grid,
                    # and pruning preserves the selected-epoch snapshot).
                    if self.selection.selected_epoch == epoch:
                        self.manager.save("selected", self._payload("selected", val_metrics))
                    elif self.manager.exists(f"epoch_{self.selection.selected_epoch:03d}"):
                        shutil.copyfile(
                            self.manager.ckpt_path(f"epoch_{self.selection.selected_epoch:03d}"),
                            self.manager.ckpt_path("selected"),
                        )
                final_metrics = val_metrics
                print(f"[{self.cfg['run_name']}] ep{epoch:03d} "
                      f"val_acc={float(val_metrics['acc'])*100:.2f}% "
                      f"aurc={float(val_metrics['aurc']):.4f} "
                      f"sel={self.selection.selected_epoch}")

            self.manager.save("last", self._payload("last", val_metrics or final_metrics))

        if not self.manager.exists("selected"):
            shutil.copyfile(self.manager.last_path(), self.manager.ckpt_path("selected"))

        manifest = self._manifest()
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        with open(os.path.join(self.run_dir, "cfg.json"), "w") as f:
            json.dump(self.cfg, f, indent=2, sort_keys=True, default=str)
        return {
            "final_val": final_metrics,
            "selection": self.selection.summary(),
            "manifest": manifest,
        }


def _git_state():
    # Execution mirrors intentionally omit .git.  The launcher can pin the
    # exact authoring commit explicitly; when present it is authoritative and
    # propagates unchanged to every train/eval subprocess.
    frozen = os.environ.get("SCSF_SOURCE_COMMIT", "").strip()
    if frozen:
        dirty = os.environ.get("SCSF_SOURCE_DIRTY", "0").strip().lower()
        return frozen, dirty in ("1", "true", "yes")
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        if not commit:
            return "", False
        return commit, dirty
    except Exception:
        return "", False
