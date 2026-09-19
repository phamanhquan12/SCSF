"""Leave-one-out neighborhood evidence distillation (NEXT5 method D)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .next5_common import LinearClassHead, final_repr, seed_extra_module
from .scsf_correctness import SCSFCorrectnessMethod
from ..engine.artifacts import MEMORY_SCHEMA


class NeighborDistillMethod(SCSFCorrectnessMethod):
    method_name = "neighbor_distill"
    needs_indices = True

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit",
                "scsf_corr_raw", "scsf_corr", "support_score")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.lambda_support = float(m.get("lambda_support", 0.3))
        self.control = str(m.get("control", "distill"))
        self.memory_path = m.get("memory_path")
        self.frozen_features = bool(m.get("frozen_features", False) or self.control == "frozen_features")
        D = int(self.backbone.final_dim)
        seed = int(train_cfg["train"].get("seed", 13))
        self.head_k8 = LinearClassHead(D, self.num_classes)
        self.head_k32 = LinearClassHead(D, self.num_classes)
        seed_extra_module(self.head_k8, seed, int(m.get("head_seed_offset", 10004)))
        seed_extra_module(self.head_k32, seed, int(m.get("head_seed_offset", 10004)) + 1)
        n = int(train_cfg["data"].get("official_train_size", 50000))
        self.register_buffer("_tgt8", torch.zeros(n, self.num_classes))
        self.register_buffer("_tgt32", torch.zeros(n, self.num_classes))
        self.register_buffer("_mem_valid", torch.zeros(n, dtype=torch.bool))
        if self.memory_path:
            self._load_memory(self.memory_path)

    def _load_memory(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        schema = data["schema"].item() if "schema" in data else None
        if schema not in (MEMORY_SCHEMA, None):
            raise ValueError(f"memory schema mismatch: {schema!r}")
        ids = np.asarray(data["ids"]).astype(np.int64)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("memory contains duplicate IDs")
        t8 = np.asarray(data["target_k8"], dtype=np.float32)
        t32 = np.asarray(data["target_k32"], dtype=np.float32)
        if self.control == "shuffled":
            rng = np.random.RandomState(13 + 9001)
            perm = rng.permutation(len(ids))
            t8 = t8[perm]
            t32 = t32[perm]
        for i, row8, row32 in zip(ids, t8, t32):
            self._tgt8[int(i)] = torch.from_numpy(row8)
            self._tgt32[int(i)] = torch.from_numpy(row32)
            self._mem_valid[int(i)] = True

    def predict_batch(self, x):
        mp = super().predict_batch(x)
        with torch.no_grad():
            bo = self.backbone(x)
            h = final_repr(bo)
            supp = torch.softmax(self.head_k32(h), dim=1)
            pred = mp.prediction
            mp.scores["support_score"] = supp.gather(1, pred.view(-1, 1)).squeeze(1)
        return mp

    def train_loss(self, batch, state) -> dict:
        x, y, idx = batch[0], batch[1], batch[2].long()
        bo = self.backbone(x)
        taps = [bo.role(self.backbone, role) for role in self.tap_roles]
        logits = bo.logits
        h = final_repr(bo)
        if self.frozen_features:
            h = h.detach()
            taps = [t.detach() for t in taps]
            logits = logits.detach()
        ce = F.cross_entropy(logits, y) if not self.frozen_features else logits.new_zeros(())
        out = {"ce": ce}
        if state.epoch >= self.pretrain:
            from .rc_training.losses import weighted_bce_correctness
            from .rc_training.schedules import meta_weight_cosine_decay
            s = self._calib(taps, logits)
            target01 = (logits.argmax(dim=1) == y).float()
            w = meta_weight_cosine_decay(
                state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
                self.init_meta_weight, self.min_meta_weight,
            )
            out["meta"] = w * weighted_bce_correctness(s, target01, self.error_weight)
            out["meta_weight"] = torch.tensor(w, device=logits.device)
        valid = self._mem_valid[idx]
        if not bool(valid.all()):
            bad = idx[~valid][:8].detach().cpu().tolist()
            raise RuntimeError(f"neighbor_distill: missing memory targets for ids {bad}")
        t8 = self._tgt8[idx].to(device=h.device, dtype=h.dtype).detach()
        t32 = self._tgt32[idx].to(device=h.device, dtype=h.dtype).detach()
        kl8 = F.kl_div(F.log_softmax(self.head_k8(h), dim=1), t8, reduction="batchmean")
        kl32 = F.kl_div(F.log_softmax(self.head_k32(h), dim=1), t32, reduction="batchmean")
        out["support"] = self.lambda_support * 0.5 * (kl8 + kl32)
        out["diag_kl8"] = kl8.detach()
        out["diag_kl32"] = kl32.detach()
        return out

    def inference_modules(self):
        return [self.backbone, self._calib]

    def optimizer_specs(self):
        specs = list(super().optimizer_specs())
        extra = [p for p in list(self.head_k8.parameters()) + list(self.head_k32.parameters())
                 if p.requires_grad]
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


__all__ = ["NeighborDistillMethod"]
