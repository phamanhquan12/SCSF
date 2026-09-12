"""CBR-SCSF — Confusion-Budgeted Risk (§8 spec; review pp. 11–13).

One raw confidence score ``s`` from the shared ``RawScoreCalibrator``, one
positive temperature ``Ts`` and a coverage grid. Per coverage the implicit
soft threshold ``h_c`` (``mean_i a_ic = c``) is solved with the analytic
tap-in derivative implemented as a custom autograd function
(``rc_training/softquantile.py``)::

    a_ic      = 1 - sigmoid((t - s_i)/Ts)        (accept when s_i > t)
    phi_a(c)  = sum_{y_i=a} a_ic / n_a
    U_ab(c)   = sum_{y_i=a} a_ic p_i(b) / (sum_{y_i=a} a_ic + eps)

    L = L_base + eta * sum_c w_c * L_micro(c)
        + beta * sum_c w_c * L_conf(c)
        + sum_{supported a,c} nu_ac * (kappa*c - phi_a(c))
    L_micro(c) = sum_i a_ic (1 - p_i(y_i)) / (B * c)
    L_conf(c)  = tau * [logsumexp(U_ab(c)/tau over supported edges)
                        - log(#supported edges)]

``nu_ac`` (dual over the per-class coverage floor) never enters any model
optimizer: ascent happens at the engine's optimizer-step boundary
(``after_step``), clamped to ``[0, dual_max]``, from detached residuals of
batch-present classes only. Edge support is refreshed periodically from
train-only confusion counts (`EdgeSupport`); absent classes / empty edge sets
are recorded as diagnostics, never reported as zero-risk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .rc_training.calibration import RawScoreCalibrator
from .rc_training.losses import (
    class_coverage_fractions,
    confusion_utilities,
    logsumexp_confusion,
    weighted_bce_correctness,
)
from .rc_training.schedules import meta_weight_cosine_decay, temperature_schedule
from .rc_training.softquantile import soft_coverage_threshold
from .rc_training.state import DualState, EdgeSupport
from .scores import compute_scores

DEFAULT_COVERAGES = (0.70, 0.80, 0.90, 0.95)


def _present_classes(y, num_classes, device):
    onehot = F.one_hot(y.detach().to(torch.long), int(num_classes)).float()
    return onehot.sum(0) > 0


class CBRSCSFMethod(Method):
    method_name = "cbr_scsf"

    def default_score(self) -> str:
        return "cbr_conf"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "cbr_raw", "cbr_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        self.coverages = tuple(float(c) for c in m.get("coverages", DEFAULT_COVERAGES))
        self.pretrain = int(m.get("pretrain", 0))
        self.Ts0 = float(m.get("Ts0", 1.0))
        self.Ts1 = float(m.get("Ts1", 4.0))
        self.eta = float(m.get("eta", 1.0))
        self.beta = float(m.get("beta", 1.0))
        self.tau = float(m.get("tau", 0.1))
        self.kappa = float(m.get("kappa", 1.0))
        self.eta_dual = float(m.get("eta_dual", 0.01))
        self.dual_max = float(m.get("dual_max", 100.0))
        w_c = m.get("w_c", None)
        self.w_c = tuple(float(x) for x in w_c) if w_c else None
        self.use_micro = bool(m.get("use_micro", True))
        self.use_confusion = bool(m.get("use_confusion", True))
        self.use_floor = bool(m.get("use_floor", True))
        self.group_by_class = bool(m.get("group_by_class", False))
        self.group_max = bool(m.get("group_max", False))
        self.edge_min_support = int(m.get("edge_min_support", 10))
        self.edge_update_every = int(m.get("edge_update_every", 1))
        self.meta_lr = float(m.get("meta_lr", 1e-4))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.error_weight = float(m.get("error_weight", 1.0))
        self.hidden_dims = tuple(int(d) for d in m.get("calibrator_hidden_dims", (1024, 512, 256, 128)))
        self.dropout = float(m.get("calibrator_dropout", 0.3))

        self._n_cov = len(self.coverages)
        if self.w_c is None:
            self.w_c = tuple(1.0 / self._n_cov for _ in range(len(self.coverages)))
        if len(self.w_c) != self._n_cov:
            raise ValueError("cbr.w_c length must match coverages")

        if self.use_confusion and self.group_by_class:
            raise ValueError(
                "cbr: use_confusion and group_by_class are mutually exclusive "
                "(cbr_true_class_group sets group_by_class=true, use_confusion=false)"
            )

        self._calib = None
        self._probe()
        self._edge = EdgeSupport(self.num_classes, next(self.backbone.parameters()).device)
        self._support = None
        self._duals = DualState(
            num_edges=self.num_classes * self._n_cov, dual_max=self.dual_max)
        self._pending_dual_grad = None

    def _probe(self):
        _, shapes = self.backbone.probe_tap_shapes(batch=1)
        feature_dims = []
        for role in self.tap_roles:
            name = self.backbone.roles[role]
            shape = torch.Size(shapes[name])
            h, w = shape[-2], shape[-1]
            spatial = 4 if (h > 1 or w > 1) else max(h * w, 1)
            feature_dims.append(int(shape[1] * spatial))
        self._calib = RawScoreCalibrator(
            feature_dims=feature_dims,
            logit_dim=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

    def on_epoch_start(self, epoch: int) -> None:
        # periodic, train-only supported-edge mask refresh
        self._support = self._edge.mask(min_support=self.edge_min_support).to(
            next(self.backbone.parameters()).device
        )
        self._duals.clamp_nu()

    def _taps_and_logits(self, bo):
        return [bo.role(self.backbone, role) for role in self.tap_roles], bo.logits

    def predict_batch(self, x):
        bo = self.backbone(x)
        taps, logits = self._taps_and_logits(bo)
        with torch.no_grad():
            raw = self._calib(taps, logits).detach()
            conf = torch.sigmoid(raw)
        scores = compute_scores(logits, self.default_scores())
        scores["cbr_raw"] = raw
        scores["cbr_conf"] = conf
        pred = logits.argmax(dim=1)
        return MethodPrediction(logits, pred, conf, scores)

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        bo = self.backbone(x)
        taps, logits = self._taps_and_logits(bo)
        out = {"ce": F.cross_entropy(logits, y)}
        if state.epoch < self.pretrain:
            return out

        s = self._calib(taps, logits)                          # raw score (B,)
        target01 = (logits.argmax(dim=1) == y).float()          # for BCE base
        w = meta_weight_cosine_decay(
            state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
            self.init_meta_weight, self.min_meta_weight)
        out["meta"] = w * weighted_bce_correctness(s, target01, self.error_weight)
        out["meta_weight"] = torch.tensor(w, device=logits.device)

        p = torch.softmax(logits, dim=1)                       # attached
        p_y = p.gather(1, y[:, None]).squeeze(1)
        B = p.shape[0]
        Ts = torch.tensor(
            temperature_schedule(state.epoch, int(self.cfg["train"]["epochs"]),
                                 self.Ts0, self.Ts1), dtype=s.dtype, device=x.device)
        coverages = torch.tensor(self.coverages, dtype=torch.float32, device=x.device)
        h = soft_coverage_threshold(s, coverages, Ts)
        # a is attached: dL/ds flows through the implicit threshold's analytic
        # derivative (SoftCoverageThreshold.backward), per spec §8.1.
        a = (1.0 - (h.unsqueeze(1) - s.unsqueeze(0)) / Ts).sigmoid()   # (C, B)

        # periodic, train-only edge-support accumulation
        self._edge.update_from_batch(logits.argmax(1).detach(), y.detach())

        phi = class_coverage_fractions(a.transpose(0, 1), y, self.num_classes)  # (C, nc)
        present = _present_classes(y, self.num_classes, x.device)

        dual_grad = torch.zeros(self.num_classes * self._n_cov, dtype=logits.dtype,
                                device=logits.device)
        diag_edges_used = []
        diag_hard_cov = []
        cidx = torch.arange(self.num_classes, device=x.device)
        group_acc = torch.zeros(self.num_classes, dtype=logits.dtype,
                                device=x.device) if self.group_by_class else None

        for ci, c in enumerate(self.coverages):
            wc = self.w_c[ci]
            a_c = a[ci]
            # L_micro(c): classifier risk over the accepted mass
            if self.use_micro:
                out[f"micro_{ci}"] = wc * self.eta * (
                    (a_c * (1.0 - p_y)).sum() / (B * c + 1e-8))
            # confusion / group risk
            if self.use_confusion and not self.group_by_class:
                support = self._support if self._support is not None else \
                    torch.ones(self.num_classes, self.num_classes, dtype=torch.bool, device=x.device)
                U = confusion_utilities(p, a_c[:, None], y, self.num_classes)[:, :, 0]  # (C, C)
                edges = [
                    (ia.item(), ib.item())
                    for ia in cidx if present[ia]
                    for ib in cidx
                    if ia != ib and bool(support[ia, ib].item())
                ]
                if edges:
                    uvals = torch.stack([U[ia, ib] for ia, ib in edges])
                    out[f"conf_{ci}"] = wc * self.beta * logsumexp_confusion(uvals, self.tau)
                    diag_edges_used.append(len(edges))
                else:
                    out[f"conf_{ci}"] = torch.zeros((), dtype=logits.dtype, device=logits.device)
                    diag_edges_used.append(0)
            elif self.group_by_class:
                # true-class group risk (variant): per-class coverage-weighted
                # accepted error, aggregated over the coverage grid below.
                onehot = F.one_hot(y.detach().to(torch.long), self.num_classes).float()
                denom = onehot.t() @ a_c
                num2 = onehot.t() @ (a_c * (1.0 - p_y))
                g_a = num2 / (denom.clamp_min(1e-8))
                g_a = torch.where(present, g_a, torch.zeros_like(g_a))
                group_acc = group_acc + wc * g_a
                diag_edges_used.append(0)
            # per-class coverage floor
            if self.use_floor:
                residual_c = self.kappa * c - phi[ci]
                residual_c = torch.where(present, residual_c, torch.zeros_like(residual_c))
                nu_c = self._duals.sample().view(self.num_classes, self._n_cov)[:, ci]
                out[f"floor_{ci}"] = (nu_c * residual_c).sum()
                dual_grad.view(self.num_classes, self._n_cov)[:, ci] = residual_c.detach()
            # hard-coverage diagnostic from the solved implicit thresholds
            with torch.no_grad():
                diag_hard_cov.append(float((s.detach() > h[ci]).float().mean()))

        self._pending_dual_grad = dual_grad
        if self.group_by_class:
            # aggregated over the coverage grid: worst true-class (group-DRO)
            # or mean true-class risk (cbr_true_class_group).
            if self.group_max:
                out["dro"] = self.beta * group_acc[present].max()
            else:
                n_present = max(int(present.sum()), 1)
                out["group"] = self.beta * group_acc[present].sum() / n_present
        out["diag_edges_used"] = torch.tensor(float(sum(diag_edges_used)), device=x.device)
        out["diag_hard_cov"] = torch.tensor(diag_hard_cov, device=x.device, dtype=logits.dtype)
        out["diag_Ts"] = Ts.detach()
        out["diag_phi_max"] = phi[:, present].max().detach() if bool(present.any()) \
            else torch.zeros((), device=x.device)
        return out

    def after_step(self, loss_items, state) -> None:
        if self._pending_dual_grad is None:
            return
        self._duals.add_dual_grad(self._pending_dual_grad, self.eta_dual)
        self._duals.clamp_nu()
        self._pending_dual_grad = None

    def optimizer_specs(self):
        t = self.cfg["train"]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        calib_params = [p for p in self._calib.parameters() if p.requires_grad]
        specs = [
            {
                "params": backbone_params,
                "kind": t.get("optimizer", "sgd"),
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
            },
            {
                "params": calib_params,
                "kind": "adam",
                "lr": self.meta_lr,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
        ]
        return [s for s in specs if s["params"]]

    def inference_modules(self):
        return [self.backbone, self._calib]


__all__ = ["CBRSCSFMethod", "DEFAULT_COVERAGES"]