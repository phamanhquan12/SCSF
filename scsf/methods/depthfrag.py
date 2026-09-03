"""DepthFrag distillation method: probes regress depth-wise fragility targets.

Method structure
----------------
* Every tapped site ``l`` carries a small normalized probe ``q_l(h_l)``
  (LayerNorm + MLP -> scalar) regressing the **detached** per-site target
  ``target_l = sign(relative_rho_l) * log1p(|relative_rho_l|)`` with Huber
  loss. Correct/incorrect examples stay distinguishable because ``target_l``
  keeps its sign.
* The per-example depth profile from the probes is aggregated with a
  configurable ``soft_min`` / lower-tail CVaR / ``mean`` / ``min`` /
  ``terminal`` operator, and a small terminal head on ``final_embedding``
  regresses the **detached** aggregate of the *true* targets.
* Inference keeps only the terminal logits plus the small terminal fragility
  head; probes and their autograd machinery are discarded (they are never in
  the prediction path). Confidence is the head output (a robustness-style
  quantity: higher = keep).

BatchNorm treatment
-------------------
Targets are computed with a **separate target forward** whose BatchNorm
statistics are frozen/eval (the backbone is evaluated in eval role while the
parameters stay differentiable ``nn.Parameters``); the probe/distillation
losses use the *training* forward's features. This is the defensible default
for per-example independence. See :mod:`scsf.depthfrag.geometry` for the fast
vs exact modes and the dedicated BatchNorm-coupling demonstration test.

End-to-end vs frozen control
----------------------------
``freeze_backbone: true`` reproduces the identical probe/head capacity with a
frozen backbone (the control). Per-step gradient norms are accumulated by
post-accumulate hooks and logged each epoch; a gradient-path helper lists
which backbone parameters actually receive probe gradients.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..depthfrag.geometry import (
    aggregate_profile,
    pool_tap,
    radii_from_site,
    site_gradients,
    target_transform,
    true_class_margin,
)
from .base import Method, MethodPrediction
from .scores import compute_scores

__all__ = [
    "DepthFragMethod",
    "FragProbe",
    "FragHead",
    "GradNormAccumulator",
    "params_reached_by_probes",
    "probe_gradient_report",
]

BN_MODES = ("eval_targets", "train")


class FragProbe(nn.Module):
    """Small normalized probe ``q_l(h_l) -> scalar`` (training-only)."""

    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(h)).squeeze(-1)


class FragHead(nn.Module):
    """Small terminal fragility head on ``final_embedding`` (persistent)."""

    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(h)).squeeze(-1)


class GradNormAccumulator:
    """Per-parameter cumulative gradient norms via post-accumulate hooks."""

    def __init__(self):
        self._acc: Dict[str, float] = {}
        self._cnt: Dict[str, int] = {}
        self._param_names: Dict[int, str] = {}

    def register(self, name: str, param: nn.Parameter):
        if not param.requires_grad:
            return
        self._param_names[id(param)] = name
        key = name
        self._acc.setdefault(key, 0.0)
        self._cnt.setdefault(key, 0)

        def hook(grad):
            self._acc[key] += float(grad.norm())
            self._cnt[key] += 1

        param.register_post_accumulate_grad_hook(hook)

    def summary(self) -> Dict[str, float]:
        return {k: (self._acc[k] / self._cnt[k]) if self._cnt[k] else 0.0
                for k in self._acc}

    def reset(self):
        for k in self._acc:
            self._acc[k] = 0.0
            self._cnt[k] = 0


class DepthFragMethod(Method):
    """Distill depth-wise decision fragility into a terminal score."""

    method_name = "depthfrag"

    def default_score(self) -> str:
        return "depthfrag"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "depthfrag")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.p = float(m.get("p", 2))
        self.q = float(m.get("q", 2))
        self.eps = float(m.get("eps", 1e-12))
        self.bn_mode = str(m.get("bn_mode", "eval_targets"))
        if self.bn_mode not in BN_MODES:
            raise ValueError(f"bn_mode must be one of {BN_MODES}, got {self.bn_mode!r}")
        self.token = str(m.get("token", "cls"))
        self.agg = str(m.get("agg", "soft_min"))
        self.agg_tau = float(m.get("agg_tau", 2.0))
        self.cvar_frac = float(m.get("cvar_frac", 0.25))
        self.target_kind = str(m.get("target", "signed_log1p"))
        self.clip_cap = float(m.get("clip_cap", 1.0))
        self.probe_hidden = int(m.get("probe_hidden", 32))
        self.probe_scale = float(m.get("probe_scale", 1.0))
        self.head_scale = float(m.get("head_scale", 1.0))
        self.huber_delta = float(m.get("huber_delta", 1.0))
        self.use_probes = bool(m.get("use_probes", True))
        self.use_head = bool(m.get("use_head", True))
        self.freeze_backbone = bool(m.get("freeze_backbone", False))
        self.target_interval = int(m.get("target_interval", 1))
        self.warmup_epochs = int(m.get("warmup_epochs", 0))

        role_pr = m.get("probe_sites", "all")
        if role_pr == "all":
            self.site_names = list(self.backbone.taps.keys())
        else:
            role_list = list(role_pr)
            self.site_names = [self.backbone.roles[r] for r in role_list]
        terminal = m.get("terminal_site", "top_l1")
        self.terminal_site = self.backbone.roles.get(terminal, terminal)

        dims = self._probe_site_dims()
        self.probes = nn.ModuleDict(
            {s: FragProbe(dims[s], self.probe_hidden) for s in self.site_names}
        )
        self._probe_params = [
            p for pr in self.probes.values() for p in pr.parameters()
        ]
        self.head = FragHead(self.backbone.final_dim, self.probe_hidden)
        self._head_params = list(self.head.parameters())

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self._gn = GradNormAccumulator()
        self._register_grad_tracking()

        self._step_n = 0
        self._step_probe: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_head = 0.0
        self._step_agg_t = 0.0
        self._step_agg_p = 0.0
        self._target_ms = 0.0
        self._log: List[dict] = []

    # ------------------------------------------------------------------ init
    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels,
                            self.backbone.input_size, self.backbone.input_size)
            )
        return {s: int(pool_tap(bo.features[s], self.token).shape[-1])
                for s in self.site_names}

    def _register_grad_tracking(self):
        n = 0
        for base, mod in (("backbone", self.backbone),
                          ("probes", self.probes),
                          ("head", self.head)):
            for name, p in mod.named_parameters():
                self._gn.register(f"{base}.{name}", p)
                n += 1

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        if self.use_head:
            frag = self.head(bo.final_embedding)
            scores["depthfrag"] = frag
        else:
            scores["depthfrag"] = compute_scores(logits, ("logit_margin",))["logit_margin"]
        conf = self._pick(scores)
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def _pick(self, scores):
        if self.score in scores:
            return scores[self.score]
        if "depthfrag" in scores:
            return scores["depthfrag"]
        return scores["msp"]

    def stripped_predict_batch(self, x):
        """Deployment-graph inference: probes are absent, head-only."""
        return self.predict_batch(x)

    def to_deployment(self) -> nn.Module:
        return self

    def inference_modules(self):
        mods = [self.backbone]
        if self.use_head:
            mods.append(self.head)
        return mods

    def optimizer_specs(self):
        t = self.cfg["train"]
        params = list(self.backbone.parameters()) + self._probe_params
        if self.use_head:
            params = params + self._head_params
        params = [p for p in params if p.requires_grad]
        return [{
            "params": params,
            "kind": t.get("optimizer", "sgd"),
            "lr": float(t["lr"]),
            "momentum": float(t.get("momentum", 0.9)),
            "weight_decay": float(t.get("weight_decay", 5e-4)),
        }]

    # -------------------------------------------------------------- training
    def train_loss(self, batch, state):
        device = next(self.backbone.parameters()).device
        x = batch[0].to(device)
        y = batch[1].to(device)

        step = int(getattr(state, "batch_index", 0))
        epoch = int(getattr(state, "epoch", 0))
        in_warmup = self.warmup_epochs > 0 and epoch < self.warmup_epochs
        # `target_interval` semantics (documented): every K-th step recomputes
        # the eval-forward targets and runs the probe/head supervision; the
        # in-between steps train CE only. Targets are per-example, so a
        # stale-target reuse is never attempted.
        do_targets = self._should_recompute_targets(step) or self.bn_mode == "train"
        t0 = time.perf_counter()
        if do_targets and self.bn_mode == "train":
            bo = self.backbone(x)
            targets, rels, margins = self._targets_train_forward(bo, bo.logits, y)
        elif do_targets:
            targets, rels, margins = self._targets_eval_forward(x, y)
            bo = self.backbone(x)
        else:
            bo = self.backbone(x)
        self._target_ms += (time.perf_counter() - t0) * 1000.0

        logits = bo.logits[:, : self.num_classes]
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}

        if not do_targets or not (self.use_probes or self.use_head):
            self._step_n += 1
            return out

        true_profile = torch.stack([targets[s] for s in self.site_names], dim=1)
        agg_target = aggregate_profile(true_profile, self.agg,
                                       self.agg_tau, self.cvar_frac)

        if self.use_probes:
            pl = torch.zeros((), device=device)
            pred_columns = []
            for s in self.site_names:
                feat = pool_tap(bo.features[s], self.token)
                if in_warmup:
                    feat = feat.detach()
                q = self.probes[s](feat)
                pred_columns.append(q)
                pl = pl + F.huber_loss(q, targets[s].detach(), delta=self.huber_delta)
            pred_profile = torch.stack(pred_columns, dim=1)
            agg_pred = aggregate_profile(pred_profile, self.agg,
                                         self.agg_tau, self.cvar_frac)
            out["depthfrag_probe"] = self.probe_scale * pl / len(self.site_names)
            with torch.no_grad():
                self._step_agg_t += float(agg_target.detach().mean())
                self._step_agg_p += float(agg_pred.detach().mean())
                for s in self.site_names:
                    self._step_probe[s] += float(
                        F.huber_loss(q.detach(), targets[s].detach(),
                                     delta=self.huber_delta))

        if self.use_head:
            head_input = bo.final_embedding
            if in_warmup:
                head_input = head_input.detach()
            out["depthfrag_head"] = self.head_scale * F.huber_loss(
                self.head(head_input), agg_target.detach(),
                delta=self.huber_delta)
            self._step_head += 1.0
        self._step_n += 1
        return out

    def _should_recompute_targets(self, step: int) -> bool:
        return self.target_interval <= 1 or step % self.target_interval == 0

    def _targets_eval_forward(self, x, y):
        """Detached targets from a forward with BatchNorm in eval role.

        The parameters remain differentiable ``nn.Parameters``; the targets
        are detached before any probe/head loss, satisfying the contract. The
        forward runs on a fresh graph seeded from the input so that frozen
        backbones still expose per-site gradients.
        """
        was_training = self.backbone.training
        xg = x.detach().clone().requires_grad_(True)
        self.backbone.eval()
        store: Dict[str, torch.Tensor] = {}
        try:
            with torch.enable_grad():
                bo = _forward_with_taps(self.backbone, xg, store)
                m = true_class_margin(bo.logits, y)
                gs = site_gradients(m, store, self.site_names)
                rho, rel, target = {}, {}, {}
                for s in self.site_names:
                    r, rr = radii_from_site(m, gs[s], store[s], self.p, self.q,
                                            self.eps)
                    rho[s] = r
                    rel[s] = rr
                    target[s] = target_transform(rr, self.target_kind,
                                                 self.clip_cap).detach()
        finally:
            del store
            self.backbone.train(was_training)
        return target, rel, m.detach()

    def _targets_train_forward(self, bo, logits, y):
        """Targets from the *training* forward (BatchNorm batch stats).

        This mode couples examples through BatchNorm statistics and is only
        available for explicit benchmarking against ``eval_targets``; it is
        documented as coupling and demonstrated by a dedicated test.
        """
        m = true_class_margin(logits, y)
        store = dict(bo.features)
        gs = site_gradients(m, store, self.site_names)
        target = {}
        rel = {}
        for s in self.site_names:
            r, rr = radii_from_site(m, gs[s], store[s], self.p, self.q, self.eps)
            rel[s] = rr
            target[s] = target_transform(rr, self.target_kind, self.clip_cap).detach()
        return target, rel, m.detach()

    # ------------------------------------------------------------------ logs
    def on_epoch_end(self, epoch: int, val_metrics: dict):
        n = max(self._step_n, 1)
        in_warmup = self.warmup_epochs > 0 and epoch < self.warmup_epochs
        row = {
            "epoch": int(epoch),
            "target_ms": float(self._target_ms),
            "head_steps": int(self._step_head),
            "agg_target": float(self._step_agg_t / n),
            "agg_pred": float(self._step_agg_p / n),
            "gradnorm": self._gn.summary(),
            "warmup": in_warmup,
            "aux_grads_to_backbone": not in_warmup and not self.freeze_backbone,
        }
        for s in self.site_names:
            row[f"probe_huber_{s}"] = float(self._step_probe[s] / n)
        self._log.append(row)
        self._write_log()
        self._reset_step_stats()
        self._gn.reset()

    def _reset_step_stats(self):
        self._step_n = 0
        self._step_head = 0.0
        self._step_agg_t = 0.0
        self._step_agg_p = 0.0
        self._step_probe = {s: 0.0 for s in self.site_names}
        self._target_ms = 0.0

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "depthfrag.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            pass


def _forward_with_taps(backbone, x, store):
    from ..backbones import MultiHook

    hooks = MultiHook(backbone.taps, store)
    try:
        return backbone(x)
    finally:
        hooks.remove()


# ---------------------------------------------------------------------------
# gradient-path introspection (aux-only backward prefix test)
# ---------------------------------------------------------------------------
def params_reached_by_probes(method: DepthFragMethod, site: str,
                             num_examples: int = 2, use_probes_only: bool = True):
    """Backbone parameter names receiving gradient from a probe's loss.

    Architecture neutral: one forward + probe-head backward on a tiny
    synthetic batch records which backbone parameters end up with a non-zero
    gradient. In the frozen-backbone control this must be empty.
    """
    if site not in method.backbone.taps:
        raise KeyError(f"{site!r} is not a registered tap of "
                       f"{type(method.backbone).__name__}")
    dev = next(method.backbone.parameters()).device
    was_training = method.training
    method.eval()
    for p in method.backbone.parameters():
        p.grad = None
    x = torch.randn(num_examples, method.backbone.channels,
                    method.backbone.input_size, method.backbone.input_size, device=dev)
    y = torch.randint(0, method.num_classes, (num_examples,), device=dev)
    with torch.enable_grad(), method.probe_mode():
        bo = method.backbone(x)
        m = true_class_margin(bo.logits, y)
        store = dict(bo.features)
        gs = site_gradients(m, store, [site])
        h = store[site]
        r, rr = radii_from_site(m, gs[site], h, method.p, method.q, method.eps)
        target = target_transform(rr, method.target_kind, method.clip_cap).detach()
        feat = pool_tap(h, method.token)
        loss = F.huber_loss(method.probes[site](feat), target,
                            delta=method.huber_delta)
        loss.backward()
    reached = [n for n, p in method.backbone.named_parameters()
               if p.grad is not None and bool(torch.any(p.grad != 0))]
    for p in method.backbone.parameters():
        p.grad = None
    method.train(was_training)
    return reached


def probe_gradient_report(method: DepthFragMethod, x, y):
    """Run a full training step and report which params received gradients.

    Returns ``{backbone_reached: [names], probe_grad_norm, head_grad_norm,
    model_grad_norm}`` and zeroes grads afterwards.
    """
    dev = next(method.backbone.parameters()).device
    method.train()
    x = x.to(dev)
    y = y.to(dev)
    loss_dict = method.train_loss((x, y), None)
    total = sum(v for v in loss_dict.values()
                if torch.is_tensor(v) and v.requires_grad)
    for p in method.parameters():
        p.grad = None
    total.backward()
    out = {
        "backbone_reached": [n for n, p in method.backbone.named_parameters()
                             if p.grad is not None and bool(torch.any(p.grad != 0))],
        "probe_grad_norm": float(sum(
            (p.grad.norm().item() if p.grad is not None else 0.0)
            for p in method._probe_params)),
        "head_grad_norm": float(sum(
            (p.grad.norm().item() if p.grad is not None else 0.0)
            for p in method._head_params)),
        "backbone_grad_norm": float(sum(
            (p.grad.norm().item() if p.grad is not None else 0.0)
            for p in method.backbone.parameters())),
        "probe_terms": [k for k in loss_dict if "probe" in k],
    }
    for p in method.parameters():
        p.grad = None
    method.eval()
    return out