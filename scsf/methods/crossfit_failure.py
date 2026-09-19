"""Cross-fitted failure supervision (NEXT5 method A).

Student = ``scsf_correctness`` + a training-only difficulty head on
``final_embedding``. Deployed score remains the current-correctness head.
OOF teacher errors are detached targets; missing IDs are hard errors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .next5_common import ScalarHead, final_repr, seed_extra_module
from .scsf_correctness import SCSFCorrectnessMethod
from ..engine.artifacts import OOF_SCHEMA, load_json, oof_error_table, validate_oof_payload


class CrossFitFailureMethod(SCSFCorrectnessMethod):
    method_name = "crossfit_failure"
    needs_indices = True

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit",
                "scsf_corr_raw", "scsf_corr")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.lambda_difficulty = float(m.get("lambda_difficulty", 0.3))
        self.control = str(m.get("control", "oof"))
        self.oof_path = m.get("oof_path")
        hidden = int(m.get("difficulty_hidden", 128))
        D = int(self.backbone.final_dim)
        self.difficulty = ScalarHead(D, hidden)
        seed_extra_module(
            self.difficulty,
            int(train_cfg["train"].get("seed", 13)),
            int(m.get("head_seed_offset", 10001)),
        )
        n = int(train_cfg["data"].get("official_train_size", 50000))
        self.register_buffer("_oof_error", torch.full((n,), float("nan")))
        self.register_buffer("_oof_valid", torch.zeros(n, dtype=torch.bool))
        self._teacher_logits = None
        if self.oof_path:
            self._load_oof(self.oof_path)

    def _load_oof(self, path: str) -> None:
        payload = load_json(path)
        if payload.get("schema") != OOF_SCHEMA and not payload.get("merged"):
            raise ValueError(f"OOF cache {path} has schema {payload.get('schema')!r}")
        n = int(self._oof_error.numel())
        expected = [int(r["id"]) for r in payload["rows"]]
        validate_oof_payload(payload, expected)
        err, valid, logits = oof_error_table(payload, n)
        if self.control == "shuffled":
            err = _shuffle_within_class(err, valid, payload)
        self._oof_error.copy_(torch.from_numpy(err))
        self._oof_valid.copy_(torch.from_numpy(valid))
        if logits is not None:
            self._teacher_logits = torch.from_numpy(logits)

    def train_loss(self, batch, state) -> dict:
        x, y, idx = batch[0], batch[1], batch[2].long()
        bo = self.backbone(x)
        taps = [bo.role(self.backbone, role) for role in self.tap_roles]
        logits = bo.logits
        h = final_repr(bo)
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}
        if state.epoch >= self.pretrain:
            s = self._calib(taps, logits)
            target01 = (logits.argmax(dim=1) == y).float()
            from .rc_training.losses import weighted_bce_correctness
            from .rc_training.schedules import meta_weight_cosine_decay
            meta = weighted_bce_correctness(s, target01, self.error_weight)
            w = meta_weight_cosine_decay(
                state.epoch, self.pretrain,
                int(self.cfg["train"]["epochs"]),
                self.init_meta_weight, self.min_meta_weight,
            )
            out["meta"] = w * meta
            out["meta_weight"] = torch.tensor(w, device=logits.device)
            out["meta_raw"] = meta.detach()
        dlogit = self.difficulty(h)
        target = self._oof_error[idx].to(dtype=dlogit.dtype, device=dlogit.device)
        valid = self._oof_valid[idx]
        if not bool(valid.all()):
            bad = idx[~valid][:8].detach().cpu().tolist()
            raise RuntimeError(
                f"crossfit_failure: missing OOF targets for ids {bad} "
                f"(control={self.control})"
            )
        target = target.detach()
        if self.control == "distill" and self._teacher_logits is not None:
            tlog = self._teacher_logits[idx].to(device=logits.device, dtype=logits.dtype)
            teacher = torch.softmax(tlog, dim=1).detach()
            distill = F.kl_div(
                F.log_softmax(logits, dim=1), teacher, reduction="batchmean",
            )
            out["difficulty"] = self.lambda_difficulty * distill
        else:
            out["difficulty"] = self.lambda_difficulty * F.binary_cross_entropy_with_logits(
                dlogit, target
            )
        student_err = (logits.argmax(dim=1) != y).float().detach()
        out["diag_teacher_error_rate"] = target.mean().detach()
        out["diag_student_error_rate"] = student_err.mean()
        if target.numel() > 1:
            out["diag_err_assoc"] = _batch_corr(target, student_err)
        return out

    def inference_modules(self):
        return [self.backbone, self._calib]

    def optimizer_specs(self):
        specs = list(super().optimizer_specs())
        extra = [p for p in self.difficulty.parameters() if p.requires_grad]
        if extra:
            t = self.cfg["train"]
            specs.append({
                "params": extra,
                "kind": t.get("optimizer", "sgd"),
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
            })
        return specs


def _batch_corr(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    den = (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-8)
    return (a * b).mean() / den


def _shuffle_within_class(err, valid, payload):
    rng = np_random(0)
    import numpy as np
    labels = {}
    for r in payload["rows"]:
        labels[int(r["id"])] = int(r["label"])
    by_c = {}
    ids = [int(r["id"]) for r in payload["rows"]]
    for i in ids:
        by_c.setdefault(labels[i], []).append(i)
    out = err.copy()
    for c, members in by_c.items():
        vals = [float(err[i]) for i in members]
        rng.shuffle(vals)
        for i, v in zip(members, vals):
            out[i] = v
    return out


def np_random(seed: int):
    import numpy as np
    return np.random.RandomState(int(seed) + 4242)


__all__ = ["CrossFitFailureMethod"]
