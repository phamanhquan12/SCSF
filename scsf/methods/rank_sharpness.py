"""Ranking-sharpness training (NEXT5 method E).

Two-pass update: freeze pair identities, perturb in the ranking-loss
direction, evaluate the combined loss at perturbed weights, restore, one
optimizer step. Pairless batches fall back to an unperturbed CE+BCE update.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .next5_common import sampled_rank_pair_loss, window_rank_weights
from .rc_training.losses import weighted_bce_correctness
from .rc_training.schedules import meta_weight_cosine_decay
from .scsf_correctness import SCSFCorrectnessMethod


class RankSharpnessMethod(SCSFCorrectnessMethod):
    method_name = "rank_sharpness"
    needs_indices = True

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit",
                "scsf_corr_raw", "scsf_corr")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.lambda_rank = float(m.get("lambda_rank", 0.5))
        self.margin = float(m.get("margin", 0.0))
        self.rho = float(m.get("rho", 0.05))
        self.cover_lo = float(m.get("cover_lo", 0.8))
        self.cover_hi = float(m.get("cover_hi", 1.0))
        self.max_err = int(m.get("max_err", 32))
        self.max_corr = int(m.get("max_corr", 32))
        self.control = str(m.get("control", "sharpness"))

    def _pack(self, batch, state, frozen=None):
        x, y, idx = batch[0], batch[1], batch[2]
        taps, logits = self._taps_and_logits(x)
        ce = F.cross_entropy(logits, y)
        s = self._calib(taps, logits)
        target01 = (logits.argmax(dim=1) == y).float()
        wmeta = meta_weight_cosine_decay(
            state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
            self.init_meta_weight, self.min_meta_weight,
        )
        meta = wmeta * weighted_bce_correctness(s, target01, self.error_weight)
        err = (target01.detach() <= 0.5)
        if frozen is None:
            weights = window_rank_weights(s.detach(), idx, self.cover_lo, self.cover_hi)
            frozen = {
                "err": err.detach().clone(),
                "weights": weights.detach().clone(),
                "idx": idx.detach().clone(),
            }
        rank = sampled_rank_pair_loss(
            s, frozen["err"], frozen["weights"],
            margin=self.margin, max_err=self.max_err, max_corr=self.max_corr,
        )
        pair_ok = bool(frozen["err"].any()) and bool((~frozen["err"]).any())
        out = {
            "ce": ce,
            "meta": meta,
            "rank": self.lambda_rank * rank,
            "meta_weight": torch.tensor(wmeta, device=x.device),
            "diag_pair_ok": torch.tensor(float(pair_ok), device=x.device),
            "diag_rank_raw": rank.detach(),
        }
        return out, frozen, pair_ok

    def train_loss(self, batch, state) -> dict:
        out, _, _ = self._pack(batch, state, frozen=None)
        return out

    def run_step(self, batch, state, optimizers) -> dict:
        if self.control == "unperturbed":
            return SCSFCorrectnessMethod.run_step(self, batch, state, optimizers)

        out1, frozen, pair_ok = self._pack(batch, state, frozen=None)
        rank = out1["rank"]
        params = [p for p in self.parameters() if p.requires_grad]
        backups = [p.data.clone() for p in params]
        try:
            do_perturb = (
                pair_ok and bool(torch.isfinite(rank))
                and float(rank.detach().abs()) > 0
                and self.control != "unperturbed"
            )
            nrm = rank.new_zeros(())
            if do_perturb:
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                rank.backward()
                sq = rank.new_zeros(())
                grads = []
                for p in params:
                    g = None if p.grad is None else p.grad.detach().clone()
                    grads.append(g)
                    if g is not None:
                        sq = sq + g.pow(2).sum()
                nrm = torch.sqrt(sq).clamp_min(1e-12)
                if float(nrm.detach()) > 0:
                    with torch.no_grad():
                        for p, g in zip(params, grads):
                            if g is not None:
                                p.add_(self.rho * g / nrm)
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    out2, _, _ = self._pack(batch, state, frozen=frozen)
                    total = _sum_grad(out2)
                    if total is not None:
                        total.backward()
                    with torch.no_grad():
                        for p, b in zip(params, backups):
                            p.copy_(b)
                    for opt in optimizers:
                        opt.step()
                    self.after_step(out2, state)
                    out2["diag_perturbed"] = torch.tensor(1.0, device=rank.device)
                    out2["diag_rank_grad_norm"] = nrm.detach()
                    return out2
            # Unperturbed fallback (no pairs, zero ranking grad, or control).
            with torch.no_grad():
                for p, b in zip(params, backups):
                    p.copy_(b)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            out_u, _, _ = self._pack(batch, state, frozen=frozen)
            total = _sum_grad({"ce" : out_u["ce"], "meta": out_u["meta"]})
            if total is not None:
                total.backward()
            for opt in optimizers:
                opt.step()
            self.after_step(out_u, state)
            out_u["diag_perturbed"] = torch.tensor(0.0, device=out_u["ce"].device)
            out_u["diag_rank_grad_norm"] = nrm.detach() if torch.is_tensor(nrm) else torch.tensor(0.0)
            return out_u
        except Exception:
            with torch.no_grad():
                for p, b in zip(params, backups):
                    p.copy_(b)
            raise


def _sum_grad(loss_dict):
    total = None
    for v in loss_dict.values():
        if torch.is_tensor(v) and v.requires_grad:
            total = v if total is None else total + v
    return total


__all__ = ["RankSharpnessMethod"]
