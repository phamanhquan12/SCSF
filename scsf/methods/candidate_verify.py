"""Class-claim verification (NEXT5 method B).

CE classifier plus a shared class-conditioned verifier q(h, embedding(c)).
Inference uses only q(h, argmax z). Ground-truth indicators never enter q.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .next5_common import ClassVerifier, LinearClassHead, final_repr, seed_extra_module
from .rc_training.calibration import RawScoreCalibrator
from .rc_training.losses import weighted_bce_correctness
from .rc_training.schedules import meta_weight_cosine_decay
from .scores import compute_scores


class CandidateVerifyMethod(Method):
    method_name = "candidate_verify"

    def default_score(self) -> str:
        return "q_argmax"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "normalized_logit",
                "q_argmax")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.control = str(m.get("control", "verifier"))
        self.lambda_verify = float(m.get("lambda_verify", 1.0))
        self.n_random = int(m.get("n_random_negatives", 3))
        self.w_pos = float(m.get("weight_positive", 1.0))
        self.w_hard = float(m.get("weight_hard", 1.0))
        self.w_rand = float(m.get("weight_random", 0.5))
        self.hidden = int(m.get("verifier_hidden", 128))
        D = int(self.backbone.final_dim)
        seed = int(train_cfg["train"].get("seed", 13))
        offset = int(m.get("head_seed_offset", 10002))
        self.verifier = ClassVerifier(D, self.num_classes, self.hidden)
        seed_extra_module(self.verifier, seed, offset)
        self.ovr = None
        self.scalar = None
        self.second_ce = None
        self._calib = None
        if self.control == "ovr":
            self.ovr = nn_linear(D, self.num_classes)
            seed_extra_module(self.ovr, seed, offset)
        elif self.control == "scalar_correctness":
            self._install_correctness_head(m)
        elif self.control == "second_ce":
            self.second_ce = LinearClassHead(D, self.num_classes)
            seed_extra_module(self.second_ce, seed, offset)

    def _install_correctness_head(self, m):
        _, shapes = self.backbone.probe_tap_shapes(batch=1)
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        feature_dims = []
        for role in self.tap_roles:
            name = self.backbone.roles[role]
            shape = torch.Size(shapes[name])
            spatial = 4 if (shape[-2] > 1 or shape[-1] > 1) else max(shape[-2] * shape[-1], 1)
            feature_dims.append(int(shape[1] * spatial))
        self._calib = RawScoreCalibrator(
            feature_dims=feature_dims, logit_dim=self.num_classes,
            hidden_dims=tuple(int(d) for d in m.get("calibrator_hidden_dims", (1024, 512, 256, 128))),
            dropout=float(m.get("calibrator_dropout", 0.3)),
        )
        self.pretrain = int(m.get("pretrain", 0))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.error_weight = float(m.get("error_weight", 1.0))

    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        pred = logits.argmax(dim=1)
        h = final_repr(bo)
        q = torch.sigmoid(self.verifier(h, pred))
        scores = compute_scores(logits, self.default_scores())
        scores["q_argmax"] = q.detach()
        conf = q.detach()
        if self.control == "second_ce" and self.second_ce is not None:
            scores["second_ce_msp"] = torch.softmax(self.second_ce(h), dim=1).max(dim=1).values.detach()
        return MethodPrediction(logits, pred, conf, scores)

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        h = final_repr(bo)
        out = {"ce": F.cross_entropy(logits, y)}
        if self.control == "ovr" and self.ovr is not None:
            target = F.one_hot(y, self.num_classes).float()
            out["verify"] = self.lambda_verify * F.binary_cross_entropy_with_logits(
                self.ovr(h), target
            )
            return out
        if self.control == "second_ce" and self.second_ce is not None:
            out["verify"] = self.lambda_verify * F.cross_entropy(self.second_ce(h), y)
            return out
        if self.control == "scalar_correctness" and self._calib is not None:
            taps = [bo.role(self.backbone, role) for role in self.tap_roles]
            s = self._calib(taps, logits)
            target01 = (logits.argmax(dim=1) == y).float()
            w = meta_weight_cosine_decay(
                state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
                self.init_meta_weight, self.min_meta_weight,
            )
            out["meta"] = w * weighted_bce_correctness(s, target01, self.error_weight)
            out["meta_weight"] = torch.tensor(w, device=logits.device)
            return out

        q_loss, diag = _verification_loss(
            self.verifier, h, logits, y, self.num_classes,
            n_random=self.n_random, w_pos=self.w_pos, w_hard=self.w_hard,
            w_rand=self.w_rand,
        )
        out["verify"] = self.lambda_verify * q_loss
        out.update(diag)
        return out

    def inference_modules(self):
        return [self.backbone, self.verifier]


def nn_linear(in_dim, out_dim):
    import torch.nn as nn
    return nn.Linear(int(in_dim), int(out_dim))


def _verification_loss(verifier, h, logits, y, num_classes, n_random, w_pos, w_hard, w_rand):
    """BCE over (y, hard negative, random negatives). Class selection detached."""
    b, c = logits.shape
    device = logits.device
    pred_wrong = logits.detach().clone()
    pred_wrong.scatter_(1, y.view(-1, 1), float("-inf"))
    hard = pred_wrong.argmax(dim=1)

    queries = [y.detach(), hard]
    weights = [h.new_full((b,), w_pos), h.new_full((b,), w_hard)]
    used = torch.zeros(b, c, dtype=torch.bool, device=device)
    used.scatter_(1, y.view(-1, 1), True)
    used.scatter_(1, hard.view(-1, 1), True)

    for _ in range(int(n_random)):
        rand = torch.randint(0, c, (b,), device=device)
        for _retry in range(8):
            collision = used.gather(1, rand.view(-1, 1)).squeeze(1)
            if not bool(collision.any()):
                break
            rand = torch.where(
                collision,
                torch.randint(0, c, (b,), device=device),
                rand,
            )
        still = used.gather(1, rand.view(-1, 1)).squeeze(1)
        rand = torch.where(still, (y + 1) % c, rand)
        skip = used.gather(1, rand.view(-1, 1)).squeeze(1)
        used.scatter_(1, rand.view(-1, 1), True)
        queries.append(rand.detach())
        w = h.new_full((b,), w_rand)
        w = torch.where(skip, torch.zeros_like(w), w)
        weights.append(w)

    losses = []
    wsum = torch.zeros(b, device=device, dtype=h.dtype)
    for q_id, w in zip(queries, weights):
        logit = verifier(h, q_id)
        target = (q_id == y).float()
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        losses.append(w * bce)
        wsum = wsum + w
    total = torch.stack(losses, dim=0).sum(0) / wsum.clamp_min(1e-8)
    pred = logits.detach().argmax(dim=1)
    diag = {
        "diag_hard_eq_pred": (hard == pred).float().mean().detach(),
        "diag_n_queries": torch.tensor(float(len(queries)), device=device),
        "diag_verifier_params": torch.tensor(float(verifier.param_count()), device=device),
    }
    return total.mean(), diag


__all__ = ["CandidateVerifyMethod"]
