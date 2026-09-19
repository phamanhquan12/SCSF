"""Within-image intervention ranking (NEXT5 method C)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rc_training.losses import weighted_bce_correctness
from .rc_training.schedules import meta_weight_cosine_decay
from .scsf_correctness import SCSFCorrectnessMethod


class InterventionRankMethod(SCSFCorrectnessMethod):
    method_name = "intervention_rank"
    needs_two_views = True
    needs_indices = True

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit",
                "scsf_corr_raw", "scsf_corr")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.lambda_pair = float(m.get("lambda_pair", 0.5))
        self.margin = float(m.get("margin", 0.1))
        self.control = str(m.get("control", "pair"))

    def _forward_view(self, x):
        taps, logits = self._taps_and_logits(x)
        s = self._calib(taps, logits)
        return logits, s

    def train_loss(self, batch, state) -> dict:
        x, v, y, idx = batch[0], batch[1], batch[2], batch[3]
        logits_a, s_a = self._forward_view(x)
        logits_b, s_b = self._forward_view(v)
        ce = 0.5 * (F.cross_entropy(logits_a, y) + F.cross_entropy(logits_b, y))
        out = {"ce": ce}
        if state.epoch < self.pretrain:
            return out
        t_a = (logits_a.argmax(dim=1) == y).float()
        t_b = (logits_b.argmax(dim=1) == y).float()
        meta = 0.5 * (
            weighted_bce_correctness(s_a, t_a, self.error_weight)
            + weighted_bce_correctness(s_b, t_b, self.error_weight)
        )
        w = meta_weight_cosine_decay(
            state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
            self.init_meta_weight, self.min_meta_weight,
        )
        out["meta"] = w * meta
        out["meta_weight"] = torch.tensor(w, device=x.device)

        corr_a = t_a.detach() > 0.5
        corr_b = t_b.detach() > 0.5
        mixed = corr_a ^ corr_b
        n_mixed = mixed.float().sum()
        out["diag_valid_pair_rate"] = (n_mixed / max(mixed.numel(), 1)).detach()
        if self.control == "two_view_only":
            out["pair"] = torch.zeros((), device=x.device)
            return out
        if self.control == "consistency":
            out["pair"] = self.lambda_pair * (s_a - s_b).pow(2).mean()
            return out
        if self.control == "shuffled":
            # destroy pairing by rolling view-b scores; still require mixed mask
            s_b = torch.roll(s_b, 1, dims=0)
            corr_b = torch.roll(corr_b, 1, dims=0)
            mixed = corr_a ^ corr_b
            n_mixed = mixed.float().sum()
        if self.control == "cross_image":
            # rank across the batch instead of within image
            err = ~(t_a.detach() > 0.5)
            if bool(err.any()) and bool((~err).any()):
                pair = F.softplus(
                    s_a[err].unsqueeze(1) - s_a[~err].unsqueeze(0) + self.margin
                ).mean()
                out["pair"] = self.lambda_pair * pair
            else:
                out["pair"] = torch.zeros((), device=x.device)
            return out

        if not bool(n_mixed > 0):
            out["pair"] = torch.zeros((), device=x.device)
            return out
        # within-image: s_wrong - s_correct
        s_wrong = torch.where(corr_a, s_b, s_a)
        s_correct = torch.where(corr_a, s_a, s_b)
        pair_terms = F.softplus(s_wrong - s_correct + self.margin)
        out["pair"] = self.lambda_pair * pair_terms[mixed].mean()
        return out


__all__ = ["InterventionRankMethod"]
