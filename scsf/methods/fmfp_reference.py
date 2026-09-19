"""FMFP reference: official SAM + SWA under the shared SCSF recipe.

Port of https://github.com/Impression2805/FMFP (MIT). ``train_fmfp.py``
applies SAM to CE only (the ranking criterion constructed in ``main_fmfp.py``
is unused there). SWA averaging starts at ``swa_start`` (protocol: 180 on the
300-epoch recipe). Official cosine+SWALR is **not** used; the shared
``ccl_sc_reference`` step schedule remains on the live SAM optimizer.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .base import MethodPrediction
from .ce import CEMethod
from .scores import compute_scores


class FMFPReferenceMethod(CEMethod):
    method_name = "fmfp_reference"
    redeploy_after_train_end = True

    def default_score(self) -> str:
        return "msp"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.rho = float(m.get("rho", 0.05))
        self.swa_start = int(m.get("swa_start", 180))
        self._swa = None
        self._swa_n = 0

    def optimizer_specs(self):
        t = self.cfg["train"]
        return [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "kind": "sam_sgd",
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
                "rho": self.rho,
                "adaptive": False,
            }
        ]

    def run_step(self, batch, state, optimizers) -> dict:
        x, y = batch[0], batch[1]
        opt = optimizers[0]
        logits = self.backbone(x).logits
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.first_step(zero_grad=True)
        loss2 = F.cross_entropy(self.backbone(x).logits, y)
        loss2.backward()
        opt.second_step(zero_grad=True)
        self.after_step({"ce": loss.detach()}, state)
        return {
            "ce": loss.detach(),
            "ce_perturbed": loss2.detach(),
        }

    def on_epoch_end(self, epoch: int, val_metrics: dict) -> None:
        if int(epoch) >= int(self.swa_start):
            self._swa_update()

    @torch.no_grad()
    def _swa_update(self):
        if self._swa is None:
            self._swa = [p.detach().clone() for p in self.parameters()]
            self._swa_n = 0
        n = self._swa_n
        for s, p in zip(self._swa, self.parameters()):
            s.mul_(n / (n + 1.0)).add_(p.detach(), alpha=1.0 / (n + 1.0))
        self._swa_n = n + 1

    @contextlib.contextmanager
    def _swa_weights(self):
        if self._swa is None or self._swa_n <= 0:
            yield
            return
        live = [p.data.clone() for p in self.parameters()]
        try:
            for p, s in zip(self.parameters(), self._swa):
                p.data.copy_(s.to(device=p.device, dtype=p.dtype))
            yield
        finally:
            for p, b in zip(self.parameters(), live):
                p.data.copy_(b)

    def predict_batch(self, x):
        if (not self.training) and self._swa is not None and self._swa_n > 0:
            with self._swa_weights():
                return super().predict_batch(x)
        return super().predict_batch(x)

    def on_train_end(self, trainer) -> None:
        if self._swa is None or self._swa_n <= 0:
            return
        for p, s in zip(self.parameters(), self._swa):
            p.data.copy_(s.to(device=p.device, dtype=p.dtype))
        _update_bn_method(trainer.train_loader, self, trainer.device)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self._swa is not None:
            sd["_fmfp_swa"] = [t.detach().cpu().clone() for t in self._swa]
            sd["_fmfp_swa_n"] = int(self._swa_n)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        sd = dict(state_dict)
        swa = sd.pop("_fmfp_swa", None)
        n = sd.pop("_fmfp_swa_n", 0)
        result = super().load_state_dict(sd, strict=strict)
        if swa is not None:
            self._swa = [t.clone() for t in swa]
            self._swa_n = int(n)
        return result


def _update_bn_method(loader, method, device):
    """BN refresh when the module forward does not return a tensor."""
    momenta = {}
    method.train()
    for m in method.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            momenta[m] = m.momentum
            m.momentum = None
            m.num_batches_tracked *= 0
            m.reset_running_stats()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            method.backbone(x)
    for m, mom in momenta.items():
        m.momentum = mom


__all__ = ["FMFPReferenceMethod"]
