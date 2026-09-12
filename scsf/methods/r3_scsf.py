"""R3-SCSF — Repair-or-Rank (§6 spec; review pp. 7–8).

For sampled **incorrect** training examples two views ``u, v`` are used to
measure *virtual repairability* — how much a single idealized SGD step on the
final-block+classifier subset ``omega`` would cut the loss on the second view:
::

    omega_plus_i = omega - eta_virtual * grad_omega CE(f(u_i), y_i)
    rho_i        = clip((CE(v_i) - CE_omega+(v_i)) / (CE(v_i) + eps), 0, 1)

``rho`` is measured per example with ``torch.func.functional_call`` on detached
parameter clones — the live model, optimizer, BN running stats and global RNG
are never touched. ``rho`` is detached everywhere; unmeasured examples use a
cached value (age <= 1 epoch) or the default ``rho_default``.

Objective (after ``pretrain``)::

    L = CE + w * BCE_correctness
        + beta * rank_pair_loss                            (RC-weighted ordering)
        + gamma * sum_{incorrect i} Wbar_i * rho_i * CE_i / (n_incorrect + eps)

The pair term is **not** normalized by the sum of ``rho`` (that cancels
routing strength); the repair term normalizes by the incorrect count. Ranking
updates calibrator + intermediate features; the calibrator's logits input stays
stop-gradient (review rule). The calibrator's raw score ``s`` is the RC
ordering score; ``sigmoid(s)`` is the deployed confidence.
"""

from __future__ import annotations

import contextlib
import random

import numpy as np
import torch
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .rc_training.calibration import RawScoreCalibrator
from .rc_training.losses import rank_pair_loss, weighted_bce_correctness
from .rc_training.rc_weights import harmonic_weights, normalize_clip
from .rc_training.schedules import meta_weight_cosine_decay
from .rc_training.state import RhoCache
from .scores import compute_scores

#: final-block + classifier parameter prefixes per backbone (§3.8).
#: VGG16-BN: feature-block weights strictly after ``pool4`` up to ``pool5``
#: (features indices 32/35/38 = conv5_1/5_2/5_3) plus the classifier.
OMEGA_PREFIXES = {
    "resnet18": ["base_model.layer4", "base_model.fc"],
    "vgg16_bn": [
        "base_model.features.32",
        "base_model.features.35",
        "base_model.features.38",
        "base_model.classifier",
    ],
}

_DEFAULT_RHO = 0.5


@contextlib.contextmanager
def _rng_gate(seed: int):
    """Capture + restore the global torch/numpy/python RNG streams."""
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(int(seed) & 0xFFFFFFFF)
        np.random.seed(int(seed) & 0xFFFFFFFF)
        random.seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)


class R3SCSFMethod(Method):
    method_name = "r3_scsf"
    needs_two_views = True

    def default_score(self) -> str:
        return "r3_conf"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "r3_rank_raw", "r3_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        self.pretrain = int(m.get("pretrain", 0))
        self.beta = float(m.get("beta", 1.0))
        self.gamma = float(m.get("gamma", 1.0))
        self.margin = float(m.get("margin", 0.0))
        self.temperature = float(m.get("temperature", 1.0))
        self.eps_rank = float(m.get("eps_rank", 1e-3))
        rho = m.get("rho", {}) if isinstance(m.get("rho", {}), dict) else {}
        self.rho_eta_virtual = float(m.get("eta_virtual", rho.get("eta_virtual", 1.0)))
        self.rho_max_errors = int(rho.get("max_errors", 8))
        self.rho_refresh_interval = int(rho.get("refresh_interval", 10))
        self.rho_default = float(rho.get("default", _DEFAULT_RHO))
        self.rho_cache_capacity = int(rho.get("cache_capacity", 65536))
        self.rho_measure_seed = int(rho.get("measure_seed", 13))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.error_weight = float(m.get("error_weight", 1.0))
        self.meta_lr = float(m.get("meta_lr", 1e-4))
        self.wbar_clip_max = m.get("wbar_clip_max")
        self.hidden_dims = tuple(int(d) for d in m.get("calibrator_hidden_dims", (1024, 512, 256, 128)))
        self.dropout = float(m.get("calibrator_dropout", 0.3))

        self._omega_names = self._resolve_omega(
            list(rho.get("omega_prefixes", OMEGA_PREFIXES.get(
                str(train_cfg["backbone"]).lower(), [])))
        )
        self._calib = None
        self._probe()
        self._rho_cache = RhoCache(
            capacity=self.rho_cache_capacity, device=self._omega_device,
            dtype=torch.float32,
        )

    @property
    def _omega_device(self):
        p = next(self.backbone.parameters())
        return p.device

    def _resolve_omega(self, prefixes):
        if not prefixes:
            raise ValueError(
                f"r3.omega_prefixes undefined for backbone {self.cfg['backbone']!r}; "
                f"known: {sorted(OMEGA_PREFIXES)}"
            )
        names = {n for n, _ in self.backbone.named_parameters()}
        found = [p for p in prefixes if any(n.startswith(p) for n in names)]
        if not found:
            raise ValueError(
                f"r3.omega_prefixes {prefixes} match no backbone parameters; "
                f"available prefixes (sample): {sorted({'.'.join(n.split('.')[:2]) for n in names})}"
            )
        sel = [n for n in names if any(n.startswith(p) for p in found)]
        return sel

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

    def _taps_and_logits(self, bo):
        return [bo.role(self.backbone, role) for role in self.tap_roles], bo.logits

    # -- virtual repairability measurement -----------------------------------
    def _measure_rho(self, xu, xv, yu):
        """Measure ``rho_i`` per incorrect example; returns a detached (n,) tensor.

        One virtual step per measured example over the detached ``omega``
        subset only (§7.1). The backbone is eval-pinned (BN uses running
        stats, never mutated) and the global RNG is gated and restored.
        """
        from torch.func import functional_call

        params = {n: p.detach() for n, p in self.backbone.named_parameters()}
        for n in self._omega_names:
            params[n].requires_grad_(True)   # leaf clones: omega-only graph
        buffers = {n: b.detach() for n, b in self.backbone.named_buffers()}
        rho = torch.full((xu.shape[0],), self.rho_default, dtype=torch.float32,
                         device=xu.device)
        n_e = xu.shape[0]
        if n_e == 0:
            return rho
        was_training = self.backbone.training
        try:
            self.backbone.eval()          # eval-BN, running stats never mutated
            for k in range(n_e):
                with _rng_gate(self.rho_measure_seed + k):
                    x_i, v_i, y_i = xu[k:k + 1], xv[k:k + 1], yu[k:k + 1]
                    with torch.enable_grad():
                        ce_u = F.cross_entropy(
                            functional_call(self.backbone, (params, buffers), x_i).logits, y_i)
                        grads = torch.autograd.grad(
                            ce_u, [params[n] for n in self._omega_names], create_graph=False)
                        omega_plus = {
                            n: params[n] - self.rho_eta_virtual * g
                            for n, g in zip(self._omega_names, grads)
                        }
                    # CE evaluation is label-free w.r.t. the graph (no grad kept)
                    with torch.no_grad():
                        ce_before = F.cross_entropy(
                            functional_call(self.backbone, (params, buffers), v_i).logits, y_i)
                        overrides = dict(params, **{n: omega_plus[n] for n in self._omega_names})
                        ce_after = F.cross_entropy(
                            functional_call(self.backbone, (overrides, buffers), v_i).logits, y_i)
                        rho[k] = torch.clamp(
                            (ce_before - ce_after) / (ce_before + 1e-8), 0.0, 1.0)
        finally:
            self.backbone.train(was_training)
        return rho.detach()

    def _rho_for_batch(self, x, v, y, idx, epoch, corr_u):
        """Per-example detached rho: measured / cached / default (bounded)."""
        err = (~corr_u).nonzero(as_tuple=True)[0]
        rho = torch.full((x.shape[0],), float(self.rho_default), dtype=torch.float32,
                         device=x.device)
        if err.numel() == 0:
            return rho, torch.zeros((), device=x.device), torch.zeros((), device=x.device)

        # deterministic sample: first max_errors in stable-id order
        order = torch.argsort(idx[err])
        measured = []
        cached = []
        for pos in order:
            i = err[pos].item()
            val = self._rho_cache.lookup(int(idx[i].item()), int(epoch))
            if val is None and len(measured) < self.rho_max_errors:
                val = self._measure_rho(x[i:i + 1], v[i:i + 1], y[i:i + 1])[0]
                self._rho_cache.update(int(idx[i].item()), val, int(epoch))
                measured.append(i)
            elif val is not None:
                cached.append(i)
            if val is not None:
                rho[i] = val
        n_measured = torch.tensor(float(len(measured)), device=x.device)
        n_cached = torch.tensor(float(len(cached)), device=x.device)
        return rho, n_measured, n_cached

    # -- prediction ----------------------------------------------------------
    def predict_batch(self, x):
        bo = self.backbone(x)
        taps, logits = self._taps_and_logits(bo)
        with torch.no_grad():
            raw = self._calib(taps, logits).detach()
            conf = torch.sigmoid(raw)
        scores = compute_scores(logits, self.default_scores())
        scores["r3_rank_raw"] = raw
        scores["r3_conf"] = conf
        pred = logits.argmax(dim=1)
        return MethodPrediction(logits, pred, conf, scores)

    # -- training ------------------------------------------------------------
    def train_loss(self, batch, state) -> dict:
        x, v, y, idx = batch[0], batch[1], batch[2], batch[3]
        bo = self.backbone(x)
        taps, logits = self._taps_and_logits(bo)
        out = {"ce": F.cross_entropy(logits, y)}
        if state.epoch < self.pretrain:
            return out

        s = self._calib(taps, logits)                        # raw score (B,)
        target01 = (logits.argmax(dim=1) == y).float()       # correctness, detached-later
        err = target01 <= 0.5

        rho, n_measured, n_cached = self._rho_for_batch(
            x, v, y, idx, state.epoch, corr_u=target01 > 0.5)
        rho = rho.detach()

        # L_base = CE + w * BCE_correctness (mirrors scsf_correctness)
        w = meta_weight_cosine_decay(
            state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
            self.init_meta_weight, self.min_meta_weight)
        out["meta"] = w * weighted_bce_correctness(s, target01, self.error_weight)
        out["meta_weight"] = torch.tensor(w, device=logits.device)

        # RC weights on the calibrator raw score (stable-id tie-break)
        w_raw = harmonic_weights(s.detach(), idx)
        w_bar = normalize_clip(w_raw, err, self.wbar_clip_max)

        if self.beta and bool(err.any()) and bool((~err).any()):
            w_diff = (w_raw[err][:, None] - w_raw[~err][None, :]).abs().detach()
            out["rank"] = self.beta * rank_pair_loss(
                s, target01, rho, w_diff,
                margin=self.margin, temperature=self.temperature,
                eps_rank=self.eps_rank,
            )
        else:
            out["rank"] = torch.zeros((), dtype=logits.dtype, device=logits.device)

        if self.gamma and bool(err.any()):
            ce_i = F.cross_entropy(logits[err], y[err], reduction="none")
            out["repair"] = self.gamma * (
                (w_bar[err] * rho[err] * ce_i).sum()
                / (err.sum().to(ce_i.dtype) + 1e-8)
            )
        else:
            out["repair"] = torch.zeros((), dtype=logits.dtype, device=logits.device)

        out["diag_rho_mean"] = rho[err].mean().detach() if bool(err.any()) else torch.zeros((), device=x.device)
        out["diag_rho_measured"] = n_measured.detach()
        out["diag_rho_cached"] = n_cached.detach()
        out["diag_w_bar_mean"] = w_bar[err].mean().detach() if bool(err.any()) else torch.zeros((), device=x.device)
        return out

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


__all__ = ["R3SCSFMethod", "OMEGA_PREFIXES"]