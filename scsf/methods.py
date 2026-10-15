from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .models import FeatureModel, VGGFeatureModel, build_backbone


FAITHFULNESS_NOTES = {
    "sr": "Softmax Response ranks by max softmax probability; used as SR confidence in SAT-selective-cls/train.py eval_converage.",
    "dg": "Deep Gamblers follows NIPS2019DeepGamblers/main.py: C+1 output, softmax reservation neuron, loss=-log(p_y + reservation/reward).",
    "sat": "SAT follows SAT-selective-cls/loss.py SelfAdativeTraining: momentum target history plus a reservation probability 1-p_y.",
    "selectivenet": "SelectiveNet follows selectivenet/models/*_vgg_selectivenet.py: class head, sigmoid selection head, auxiliary head, coverage penalty lambda=32.",
    "dualaug": "dualaug: train_fresh_sc.py dualaug on a torchvision backbone; VGG pool4/pool5 are replaced by pooled mid/late stages, the head gets the input LayerNorm of the SCSF calibrator, and there is no gradient clipping (no method in run_experiment.py clips).",
    "dualaug_correctness": "dualaug_correctness: dualaug with head target a0*(0.5+0.5*a1), a_i = 1[view i correct] (proposal C); --aux-detach detaches the head's pooled features.",
    "verify_global": "verify_global: proposal A; two-view CE + class-conditioned verifier MLP([GAP(layer3), e_c]) (embed 16, hidden 128), weight 0 in warm-up then --verify-weight; scored by u(x, argmax z).",
    "verify_spatial": "verify_spatial: proposal B; two-view CE + Conv1x1(layer3) -> C evidence maps, top-20% spatial mean (or GAP control); scored by u(x, argmax z).",
    "mg_scsf": "mg_scsf: SCSF (TCP head, Adam, cosine lambda, raw-logit score) + training-only margin-gated cosine classifier on GAP(layer3), alpha 0->--ds-aux-weight over --ds-aux-ramp epochs after warm-up; single-view CE.",
    "mp_ds": "mp_ds: single-view CE + K normalised prototypes per class on a bias-free 128-d projection of GAP(--ds-site: layer3, layer4 or both concatenated), CE(a/0.1, y); scored by the prototype support margin of the final prediction.",
    "se_ds": "se_ds: single-view CE + Conv1x1 class evidence maps on --ds-site (layer3, layer4, or layer3 + upsampled layer4), top-20% (or GAP control) pooling, CE(a, y); scored by the evidence margin of the final prediction.",
    "ccl_sc": "CCL-SC: --ccl-variant official reproduces lamda-bbo/CCL-SC train_CCL_SC.py + moco/CSC.py (per-epoch key EMA, queues frozen once full, max-shifted loss); --ccl-variant paper uses the paper's CSC loss with MoCo-style per-step updates.",
}


@dataclass
class MethodBatchOutput:
    train_logits: torch.Tensor
    eval_logits: torch.Tensor
    confidence: torch.Tensor
    aux_loss: torch.Tensor
    features: Optional[torch.Tensor] = None
    meta_logits: Optional[torch.Tensor] = None
    key_logits: Optional[torch.Tensor] = None
    key_features: Optional[torch.Tensor] = None
    aux_outputs: Optional[dict[str, dict[str, torch.Tensor]]] = None


class ExperimentMethod(nn.Module):
    needs_indices = False

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def training_loss(self, output: MethodBatchOutput, targets: torch.Tensor, indices: Optional[torch.Tensor], epoch: int, args) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        raise NotImplementedError


class BackboneMethod(ExperimentMethod):
    def __init__(self, backbone: FeatureModel, num_classes: int):
        super().__init__(num_classes)
        self.backbone = backbone

    def _backbone_logits_features(self, x: torch.Tensor):
        return self.backbone(x, return_features=True)


class SoftmaxResponseMethod(BackboneMethod):
    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, _ = self._backbone_logits_features(x)
        confidence = F.softmax(logits, dim=1).max(dim=1).values
        return MethodBatchOutput(logits, logits, confidence, logits.new_zeros(()))

    def training_loss(self, output, targets, indices, epoch, args):
        return F.cross_entropy(output.train_logits, targets)


class DeepGamblersMethod(BackboneMethod):
    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits_plus, _ = self._backbone_logits_features(x)
        class_logits = logits_plus[:, : self.num_classes]
        # logit(1 - p_abstain): same ranking as 1 - p_abstain without fp32 saturation ties.
        confidence = torch.logsumexp(class_logits.float(), dim=1) - logits_plus[:, -1].float()
        return MethodBatchOutput(logits_plus, class_logits, confidence, logits_plus.new_zeros(()))

    def training_loss(self, output, targets, indices, epoch, args):
        if epoch <= args.pretrain:
            return F.cross_entropy(output.train_logits[:, : self.num_classes], targets)
        probs = F.softmax(output.train_logits, dim=1)
        gain = probs[torch.arange(targets.shape[0], device=targets.device), targets]
        reservation = probs[:, -1]
        return -(gain.add(reservation.div(args.reward))).log().mean()


class SelfAdaptiveTraining:
    """Faithful port of SAT-selective-cls/loss.py SelfAdativeTraining."""

    def __init__(self, num_examples: int, num_classes: int, momentum: float = 0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.momentum = momentum
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        def compute_loss() -> torch.Tensor:
            logits_fp32 = logits.float()
            prob = F.softmax(logits_fp32.detach()[:, : self.num_classes], dim=1)
            onehot = torch.zeros_like(prob)
            batch = torch.arange(labels.shape[0], device=labels.device)
            onehot[batch, labels] = 1.0

            history = self.prob_history[indices.cpu()].clone().to(prob.device)
            is_updated = self.updated[indices.cpu()].to(prob.device).bool().unsqueeze(1)
            prob_mom = torch.where(is_updated, history, onehot)
            prob_mom = self.momentum * prob_mom + (1.0 - self.momentum) * prob

            self.updated[indices.cpu()] = 1
            self.prob_history[indices.cpu()] = prob_mom.detach().cpu()

            soft_label = torch.zeros_like(logits_fp32)
            soft_label[batch, labels] = prob_mom[batch, labels]
            soft_label[:, -1] = 1.0 - prob_mom[batch, labels]
            soft_label = F.normalize(soft_label, dim=1, p=1)
            return torch.sum(-F.log_softmax(logits_fp32, dim=1) * soft_label, dim=1).mean()

        if logits.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                return compute_loss()
        return compute_loss()


class SATMethod(DeepGamblersMethod):
    needs_indices = True

    def __init__(self, backbone: FeatureModel, num_classes: int, train_size: int, momentum: float):
        super().__init__(backbone, num_classes)
        self.sat_loss = SelfAdaptiveTraining(train_size, num_classes, momentum)

    def training_loss(self, output, targets, indices, epoch, args):
        if epoch <= args.pretrain:
            return F.cross_entropy(output.train_logits[:, : self.num_classes], targets)
        if indices is None:
            raise ValueError("SAT requires indexed training batches.")
        return self.sat_loss(output.train_logits, targets, indices)


class SelectiveNetMethod(BackboneMethod):
    def __init__(self, backbone: FeatureModel, num_classes: int, target_coverage: float = 0.8, alpha: float = 0.5):
        super().__init__(backbone, num_classes)
        self.target_coverage = target_coverage
        self.alpha = alpha
        self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        self.selector = nn.Sequential(nn.Linear(backbone.feature_dim, 1), nn.Sigmoid())
        self.aux_classifier = nn.Linear(backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        _, features = self._backbone_logits_features(x)
        logits = self.classifier(features)
        selection_logit = self.selector[0](features).squeeze(1)
        selection = self.selector[1](selection_logit)
        aux_logits = self.aux_classifier(features)
        packed = torch.cat([logits, selection.unsqueeze(1)], dim=1)
        # Rank by the pre-sigmoid selector logit: same order, no ties at exactly 1.0.
        return MethodBatchOutput(packed, logits, selection_logit.float(), aux_logits, features=features)

    def training_loss(self, output, targets, indices, epoch, args):
        logits = output.train_logits[:, : self.num_classes]
        selection = output.train_logits[:, -1].clamp_min(1e-6)
        ce = F.cross_entropy(logits, targets, reduction="none")
        selective_loss = (ce * selection).mean() / selection.mean()
        coverage_penalty = args.selectivenet_lambda * F.relu(self.target_coverage - selection.mean()).pow(2)
        aux_loss = F.cross_entropy(output.aux_loss, targets)
        return self.alpha * (selective_loss + coverage_penalty) + (1.0 - self.alpha) * aux_loss


class MetaCalibrator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, use_logits: bool = True):
        super().__init__()
        self.use_logits = use_logits
        input_dim = feature_dim + (num_classes if use_logits else 0)
        widths = [hidden_dim * 4, hidden_dim * 2, hidden_dim, max(1, hidden_dim // 2)]
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, widths[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(widths[0], widths[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(widths[1], widths[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(widths[2], widths[3]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(widths[3], 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.use_logits:
            inputs = torch.cat([features.float(), logits.detach().float()], dim=1)
        else:
            inputs = features.float()
        if inputs.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                return self.net(inputs).squeeze(1)
        return self.net(inputs).squeeze(1)


class SimpleMetaCalibrator(nn.Module):
    """Lightweight 2-hidden-layer calibrator with a residual connection and GELU.

    Architecture: LayerNorm → Linear(in, h) → GELU → Dropout → [Linear(in, h) residual]
                  → Linear(h, h) → GELU → Dropout → Linear(h, 1)

    Compared to MetaCalibrator this halves the depth (2 vs 4 hidden layers),
    uses a skip connection to prevent gradient vanishing, and swaps ReLU for
    GELU for smoother gradients on the regression task.
    """

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, use_logits: bool = True):
        super().__init__()
        self.use_logits = use_logits
        input_dim = feature_dim + (num_classes if use_logits else 0)
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(input_dim, hidden_dim)  # residual projection
        self.drop = nn.Dropout(0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.use_logits:
            inputs = torch.cat([features.float(), logits.detach().float()], dim=1)
        else:
            inputs = features.float()
        if inputs.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                return self._net(inputs)
        return self._net(inputs)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        res = self.proj(x)
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(F.gelu(self.fc2(x + res)))
        return self.head(x).squeeze(1)


class SCSFMethod(BackboneMethod):
    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        hidden_dim: int = 256,
        use_logits: bool = True,
        scorer: str = "meta",
        sr_alpha: float = 0.5,
        calibrator_arch: str = "standard",
        train_size: int = 0,
    ):
        super().__init__(backbone, num_classes)
        calib_cls = SimpleMetaCalibrator if calibrator_arch == "simple" else MetaCalibrator
        self.calibrator = calib_cls(backbone.feature_dim, num_classes, hidden_dim=hidden_dim, use_logits=use_logits)
        self.scorer = scorer
        self.sr_alpha = sr_alpha
        # EMA target history (Fix 1). Kept as plain CPU tensors (not registered
        # buffers) so they never get moved to GPU on model.to(device).
        if train_size > 0:
            self._tcp_ema = torch.full((train_size,), 0.5)          # always CPU
            self._tcp_initialized = torch.zeros(train_size, dtype=torch.bool)  # always CPU
        else:
            self._tcp_ema = None
            self._tcp_initialized = None

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._backbone_logits_features(x)
        meta_logits = self.calibrator(features, logits)
        meta_confidence = torch.sigmoid(meta_logits.float()).to(logits.dtype)
        sr_confidence = F.softmax(logits.float(), dim=1).max(dim=1).values.to(logits.dtype)
        if self.scorer == "meta":
            # Rank by the raw logit: sigmoid saturates to exactly 1.0 in fp32 (logit > ~17), creating ties.
            confidence = meta_logits.float()
        elif self.scorer == "sr":
            confidence = sr_confidence
        elif self.scorer == "meta_sr_product":
            confidence = meta_confidence * sr_confidence
        elif self.scorer == "meta_sr_blend":
            alpha = min(1.0, max(0.0, self.sr_alpha))
            confidence = alpha * sr_confidence + (1.0 - alpha) * meta_confidence
        elif self.scorer == "geometric":
            # Geometric mean: softer than product, requires both to be high
            confidence = (meta_confidence * sr_confidence).sqrt()
        elif self.scorer == "meta_agreement":
            # Penalise SR by the absolute disagreement between SR and Meta.
            # Stays high when both signals agree, drops when they diverge,
            # regardless of which one is higher.
            disagreement = (sr_confidence - meta_confidence).abs()
            confidence = sr_confidence * (1.0 - disagreement)
        elif self.scorer == "min_sr_meta":
            # Most conservative: confidence is the lower of the two signals.
            confidence = torch.min(sr_confidence, meta_confidence)
        elif self.scorer == "margin":
            # Prediction margin: top-1 minus top-2 softmax probability.
            # More discriminative than SR when two classes compete closely.
            probs = F.softmax(logits.float(), dim=1)
            top2, _ = probs.topk(min(2, probs.size(1)), dim=1)
            if top2.size(1) >= 2:
                confidence = (top2[:, 0] - top2[:, 1]).to(logits.dtype)
            else:
                confidence = top2[:, 0].to(logits.dtype)
        elif self.scorer == "energy":
            # Log-partition / free-energy score (Liu et al. 2020).
            # Higher log-sum-exp → model places large unnormalised mass on
            # some class → generally correlates with in-distribution confidence.
            confidence = torch.logsumexp(logits.float(), dim=1).to(logits.dtype)
        elif self.scorer == "doctor":
            # Doctor confidence (Granese et al. 2021):
            # 1 / (1 + Σ_c p_c*(1-p_c)); higher = more concentrated softmax.
            probs = F.softmax(logits.float(), dim=1)
            variance = (probs * (1.0 - probs)).sum(dim=1)
            confidence = (1.0 / (1.0 + variance)).to(logits.dtype)
        else:
            raise ValueError(f"Unknown SCSF scorer: {self.scorer}")
        return MethodBatchOutput(logits, logits, confidence, logits.new_zeros(()), features=features, meta_logits=meta_logits)

    def training_loss(self, output, targets, indices, epoch, args):
        ce = F.cross_entropy(output.train_logits, targets)
        if epoch <= args.pretrain:
            return ce
        with torch.no_grad():
            probs = F.softmax(output.train_logits, dim=1)
            correct = output.train_logits.argmax(dim=1).eq(targets).float()
            if getattr(args, "scsf_meta_target", "tcp") == "correctness":
                target_conf = correct
            else:
                current_tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                # Fix 1: EMA-smooth TCP targets when buffer is available and indices are known.
                ema_momentum = float(getattr(args, "tcp_ema_momentum", 0.0))
                if ema_momentum > 0.0 and self._tcp_ema is not None and indices is not None:
                    idx = indices.cpu()
                    current_cpu = current_tcp.detach().cpu()
                    initialized = self._tcp_initialized[idx]
                    self._tcp_ema[idx] = torch.where(
                        initialized,
                        ema_momentum * self._tcp_ema[idx] + (1.0 - ema_momentum) * current_cpu,
                        current_cpu,
                    )
                    self._tcp_initialized[idx] = True
                    target_conf = self._tcp_ema[idx].to(output.train_logits.device)
                else:
                    target_conf = current_tcp

        if args.meta_loss == "mse":
            meta_confidence = self._loss_confidence(output, args)
            focal_gamma = float(getattr(args, "meta_focal_gamma", 0.0))
            if focal_gamma > 0.0:
                # Fix 5: focal weighting — down-weight high-TCP (easy) samples
                focal_weight = (1.0 - target_conf.detach()).clamp(min=1e-6).pow(focal_gamma)
                meta_loss = (focal_weight * F.mse_loss(meta_confidence, target_conf, reduction="none")).mean()
            else:
                meta_loss = F.mse_loss(meta_confidence, target_conf)
        elif args.meta_loss in {"weighted_nll", "weighted_nll_pairwise"}:
            target_conf = target_conf.float()
            weights = torch.ones_like(correct)
            weights[correct < 0.5] = args.error_weight
            # Fix 5: focal weighting — down-weight high-TCP (easy) samples so
            # calibrator focuses on the uncertain/incorrect boundary.
            focal_gamma = float(getattr(args, "meta_focal_gamma", 0.0))
            if focal_gamma > 0.0:
                focal_weight = (1.0 - target_conf.detach()).clamp(min=1e-6).pow(focal_gamma)
                weights = weights * focal_weight
            meta_logits = output.meta_logits.float() if output.meta_logits is not None else torch.logit(output.confidence.float().clamp(1e-7, 1.0 - 1e-7))
            confidence_mode = getattr(args, "meta_confidence_mode", "meta")
            if confidence_mode == "meta":
                if meta_logits.device.type == "cuda":
                    with torch.amp.autocast("cuda", enabled=False):
                        bce_loss = F.binary_cross_entropy_with_logits(meta_logits, target_conf, weight=weights.float())
                else:
                    bce_loss = F.binary_cross_entropy_with_logits(meta_logits, target_conf, weight=weights.float())
                ranking_confidence = torch.sigmoid(meta_logits)
            else:
                ranking_confidence = self._loss_confidence(output, args)
                if ranking_confidence.device.type == "cuda":
                    with torch.amp.autocast("cuda", enabled=False):
                        bce_loss = F.binary_cross_entropy(
                            ranking_confidence.clamp(1e-7, 1.0 - 1e-7),
                            target_conf,
                            weight=weights.float(),
                        )
                else:
                    bce_loss = F.binary_cross_entropy(
                        ranking_confidence.clamp(1e-7, 1.0 - 1e-7),
                        target_conf,
                        weight=weights.float(),
                    )
            if args.meta_loss == "weighted_nll":
                meta_loss = bce_loss
            else:
                ranking_loss = pairwise_ranking_loss(ranking_confidence, correct, args)
                if ranking_loss is not None:
                    alpha = min(1.0, max(0.0, float(getattr(args, "pairwise_alpha", 0.9))))
                    meta_loss = alpha * ranking_loss + (1.0 - alpha) * bce_loss
                else:
                    meta_loss = bce_loss
        else:
            raise ValueError(f"Unknown SCSF meta loss: {args.meta_loss}")

        sr_rank_weight = max(0.0, float(getattr(args, "sr_rank_weight", 0.0)))
        if sr_rank_weight > 0.0:
            sr_confidence = F.softmax(output.train_logits.float(), dim=1).max(dim=1).values
            sr_ranking_loss = pairwise_ranking_loss(sr_confidence, correct, args)
            if sr_ranking_loss is not None:
                ce = ce + sr_rank_weight * sr_ranking_loss

        if getattr(args, "scsf_meta_only_after_pretrain", False):
            return self._meta_weight(epoch, args) * meta_loss
        return ce + self._meta_weight(epoch, args) * meta_loss

    @staticmethod
    def _loss_confidence(output, args) -> torch.Tensor:
        meta_logits = output.meta_logits.float() if output.meta_logits is not None else torch.logit(output.confidence.float().clamp(1e-7, 1.0 - 1e-7))
        meta_confidence = torch.sigmoid(meta_logits)
        confidence_mode = getattr(args, "meta_confidence_mode", "meta")
        if confidence_mode == "meta":
            return meta_confidence
        sr_confidence = F.softmax(output.train_logits.detach().float(), dim=1).max(dim=1).values
        if confidence_mode == "product":
            return meta_confidence * sr_confidence
        if confidence_mode == "blend":
            alpha = min(1.0, max(0.0, float(getattr(args, "scsf_sr_alpha", 0.5))))
            return alpha * sr_confidence + (1.0 - alpha) * meta_confidence
        raise ValueError(f"Unknown SCSF meta confidence mode: {confidence_mode}")

    @staticmethod
    def _meta_weight(epoch: int, args) -> float:
        if args.meta_weight_mode == "fixed":
            return args.meta_weight
        if args.meta_weight_mode != "cosine":
            raise ValueError(f"Unknown meta weight mode: {args.meta_weight_mode}")
        if epoch <= args.pretrain:
            return 0.0
        progress = (epoch - args.pretrain) / max(1, args.epochs - args.pretrain)
        # Fix 4: linear ramp-up over the first warmup_fraction of joint training,
        # then cosine decay. warmup_fraction=0 reproduces the original behaviour.
        warmup = float(getattr(args, "meta_weight_warmup_fraction", 0.0))
        if warmup > 0.0 and progress < warmup:
            return args.init_meta_weight * (progress / warmup)
        decay_progress = (progress - warmup) / max(1e-9, 1.0 - warmup)
        return args.min_meta_weight + 0.5 * (args.init_meta_weight - args.min_meta_weight) * (
            1.0 + math.cos(math.pi * decay_progress)
        )


def pairwise_ranking_loss(confidence: torch.Tensor, correct: torch.Tensor, args) -> Optional[torch.Tensor]:
    """Pairwise ranking loss shared by SCSF and DS-SCSF v2 (AURC-aligned)."""
    correct_mask = correct.bool()
    conf_correct = confidence[correct_mask]
    conf_wrong = confidence[~correct_mask]
    if conf_correct.numel() == 0 or conf_wrong.numel() == 0:
        return None
    margin = float(getattr(args, "pairwise_margin", 0.1))
    pairwise = conf_wrong.unsqueeze(0) - conf_correct.unsqueeze(1) + margin
    pairwise_type = getattr(args, "pairwise_type", "hinge")
    if pairwise_type == "hinge":
        pairwise_loss = pairwise.clamp_min(0.0)
    elif pairwise_type == "softplus":
        temperature = max(1e-6, float(getattr(args, "pairwise_temperature", 0.1)))
        pairwise_loss = F.softplus(pairwise / temperature) * temperature
    else:
        raise ValueError(f"Unknown pairwise loss type: {pairwise_type}")
    hard_fraction = min(1.0, max(0.0, float(getattr(args, "pairwise_hard_fraction", 1.0))))
    if 0.0 < hard_fraction < 1.0:
        flat_loss = pairwise_loss.flatten()
        k = max(1, int(math.ceil(flat_loss.numel() * hard_fraction)))
        return flat_loss.topk(k).values.mean()
    return pairwise_loss.mean()


class GatedConfidenceFusion(nn.Module):
    """Instance-adaptive fusion of per-depth confidence scores.

    Each depth contributes a (confidence, margin) pair. A depth-specific linear
    gate maps that pair to a logit; softmax over depths yields per-sample
    weights. A learnable depth prior gently favors later layers when gate
    inputs are ambiguous, reducing collapse to a single depth.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GatedConfidenceFusion requires at least one layer.")
        self.gates = nn.ModuleList(nn.Linear(2, 1) for _ in range(num_layers))
        prior = torch.linspace(-0.25, 0.25, num_layers)
        self.depth_prior = nn.Parameter(prior)
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(
        self, confidences: list[torch.Tensor], margins: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(confidences) != len(self.gates) or len(margins) != len(self.gates):
            raise ValueError("confidences/margins length must match number of gates.")
        gate_logits = []
        for idx, (confidence, margin) in enumerate(zip(confidences, margins)):
            gate_input = torch.stack([confidence.float(), margin.float()], dim=1)
            gate_logits.append(self.gates[idx](gate_input).squeeze(1) + self.depth_prior[idx])
        weights = F.softmax(torch.stack(gate_logits, dim=1), dim=1)
        dtype = confidences[0].dtype
        fused = sum(weights[:, idx].to(dtype) * confidences[idx] for idx in range(len(confidences)))
        return fused, weights


class DSFeatureCalibrator(nn.Module):
    """Per-depth confidence head used by DS-SCSF companion objectives."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = feature_dim + num_classes
        mid_dim = max(1, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid_dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([features.float(), logits.detach().float()], dim=1)
        if inputs.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                return self.net(inputs).squeeze(1)
        return self.net(inputs).squeeze(1)


class DSSCSFMethod(BackboneMethod):
    """Pipeline-native DS-SCSF.

    This ports the useful part of train_ds_scsf.py: auxiliary classifiers and
    calibrators at multiple feature depths, with learned confidence fusion.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        feature_dims: dict[str, int],
        hidden_dim: int = 256,
        layers: tuple[str, ...] = ("early", "mid", "late"),
    ):
        super().__init__(backbone, num_classes)
        self.layers = tuple(layer for layer in layers if layer in feature_dims)
        if not self.layers:
            raise ValueError("DS-SCSF requires at least one feature layer.")
        self.feature_dims = feature_dims
        self.aux_classifiers = nn.ModuleDict(
            {layer: nn.Linear(feature_dims[layer], num_classes) for layer in self.layers}
        )
        self.calibrators = nn.ModuleDict(
            {
                layer: DSFeatureCalibrator(feature_dims[layer], num_classes, hidden_dim=hidden_dim)
                for layer in self.layers
            }
        )
        self.beta_logits = nn.Parameter(torch.ones(len(self.layers)))

    def _split_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = {}
        offset = 0
        for layer in self.layers:
            dim = self.feature_dims[layer]
            parts[layer] = features[:, offset : offset + dim]
            offset += dim
        return parts

    def _fused_confidence(self, confidences: list[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.beta_logits.float(), dim=0).to(confidences[0].dtype)
        return sum(weights[idx] * confidences[idx] for idx in range(len(confidences)))

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._backbone_logits_features(x)
        layer_features = self._split_features(features)
        aux_logits = {}
        meta_logits = {}
        confidences = []
        for layer in self.layers:
            layer_logits = self.aux_classifiers[layer](layer_features[layer])
            layer_meta_logits = self.calibrators[layer](layer_features[layer], layer_logits)
            aux_logits[layer] = layer_logits
            meta_logits[layer] = layer_meta_logits
            confidences.append(torch.sigmoid(layer_meta_logits.float()).to(logits.dtype))
        confidence = self._fused_confidence(confidences)
        return MethodBatchOutput(
            logits,
            logits,
            confidence,
            logits.new_zeros(()),
            features=features,
            aux_outputs={"features": layer_features, "logits": aux_logits, "meta_logits": meta_logits},
        )

    def training_loss(self, output, targets, indices, epoch, args):
        ce = F.cross_entropy(output.train_logits, targets)
        aux_logits = output.aux_outputs["logits"]
        aux_meta_logits = output.aux_outputs["meta_logits"]
        class_weights = getattr(args, "ds_class_weights", None)
        ce_weight = None
        if class_weights is not None:
            ce_weight = torch.as_tensor(class_weights, device=targets.device, dtype=output.train_logits.float().dtype)
        aux_ce = sum(F.cross_entropy(aux_logits[layer], targets, weight=ce_weight) for layer in self.layers)
        loss = ce + float(getattr(args, "ds_aux_ce_weight", 0.3)) * aux_ce
        if epoch <= args.pretrain:
            return loss

        with torch.no_grad():
            probs = F.softmax(output.train_logits.float(), dim=1)
            correct = output.train_logits.argmax(dim=1).eq(targets).float()
            if getattr(args, "scsf_meta_target", "tcp") == "correctness":
                target_conf = correct
            else:
                target_conf = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        weights = torch.ones_like(correct)
        weights[correct < 0.5] = args.error_weight
        if class_weights is not None:
            sample_class_weights = ce_weight[targets].to(weights.dtype)
            weights = weights * sample_class_weights
        meta_losses = []
        for layer in self.layers:
            layer_meta_logits = aux_meta_logits[layer].float()
            if layer_meta_logits.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    meta_losses.append(
                        F.binary_cross_entropy_with_logits(layer_meta_logits, target_conf.float(), weight=weights.float())
                    )
            else:
                meta_losses.append(
                    F.binary_cross_entropy_with_logits(layer_meta_logits, target_conf.float(), weight=weights.float())
                )
        meta_loss = sum(meta_losses) / len(meta_losses)
        fused_cal_weight = max(0.0, float(getattr(args, "ds_fused_cal_weight", 0.0)))
        if fused_cal_weight > 0.0:
            fused_confidence = output.confidence.float().clamp(1e-7, 1.0 - 1e-7)
            if fused_confidence.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    fused_meta_loss = F.binary_cross_entropy(
                        fused_confidence,
                        target_conf.float(),
                        weight=weights.float(),
                    )
            else:
                fused_meta_loss = F.binary_cross_entropy(
                    fused_confidence,
                    target_conf.float(),
                    weight=weights.float(),
                )
            meta_loss = (meta_loss + fused_cal_weight * fused_meta_loss) / (1.0 + fused_cal_weight)
        return loss + float(getattr(args, "ds_aux_cal_weight", 1.0)) * SCSFMethod._meta_weight(epoch, args) * meta_loss


class DSSCSFv2Method(BackboneMethod):
    """DS-SCSF v2: instance-adaptive, self-distilled, rank-aware deep supervision.

    Three targeted changes over ``DSSCSFMethod``:

    1. **Fusion.** ``GatedConfidenceFusion`` replaces the global softmax mixture
       with per-sample fusion weights driven by each depth's confidence and margin.
    2. **Representation quality.** Auxiliary heads are self-distilled from the
       final classifier's detached soft predictions (BYOT-style). Better aux
       classifiers yield more reliable per-depth TCP inputs for the fusion gate.
    3. **Objective alignment.** Calibrators use EMA-smoothed, optionally
       focally-weighted BCE (Fix1/Fix5) plus a pairwise ranking loss on the
       *fused* confidence, aligned with AURC/NAURC evaluation.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        feature_dims: dict[str, int],
        hidden_dim: int = 256,
        layers: tuple[str, ...] = ("mid", "late"),
        train_size: int = 0,
        kd_temperature: float = 2.0,
        kd_weight: float = 0.5,
    ):
        super().__init__(backbone, num_classes)
        self.layers = tuple(layer for layer in layers if layer in feature_dims)
        if not self.layers:
            raise ValueError("DS-SCSF v2 requires at least one feature layer.")
        self.feature_dims = feature_dims
        self.aux_classifiers = nn.ModuleDict(
            {layer: nn.Linear(feature_dims[layer], num_classes) for layer in self.layers}
        )
        self.calibrators = nn.ModuleDict(
            {
                layer: DSFeatureCalibrator(feature_dims[layer], num_classes, hidden_dim=hidden_dim)
                for layer in self.layers
            }
        )
        self.fusion = GatedConfidenceFusion(len(self.layers))
        self.kd_temperature = kd_temperature
        self.kd_weight = min(1.0, max(0.0, kd_weight))
        if train_size > 0:
            self._tcp_ema = torch.full((train_size,), 0.5)
            self._tcp_initialized = torch.zeros(train_size, dtype=torch.bool)
        else:
            self._tcp_ema = None
            self._tcp_initialized = None

    def _split_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = {}
        offset = 0
        for layer in self.layers:
            dim = self.feature_dims[layer]
            parts[layer] = features[:, offset : offset + dim]
            offset += dim
        return parts

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        top2, _ = probs.topk(min(2, probs.size(1)), dim=1)
        if top2.size(1) >= 2:
            return top2[:, 0] - top2[:, 1]
        return top2[:, 0]

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._backbone_logits_features(x)
        layer_features = self._split_features(features)
        aux_logits: dict[str, torch.Tensor] = {}
        meta_logits: dict[str, torch.Tensor] = {}
        confidences, margins = [], []
        for layer in self.layers:
            layer_logits = self.aux_classifiers[layer](layer_features[layer])
            layer_meta_logits = self.calibrators[layer](layer_features[layer], layer_logits)
            aux_logits[layer] = layer_logits
            meta_logits[layer] = layer_meta_logits
            confidences.append(torch.sigmoid(layer_meta_logits.float()).to(logits.dtype))
            margins.append(self._margin(layer_logits).to(logits.dtype))

        confidence, fusion_weights = self.fusion(confidences, margins)
        return MethodBatchOutput(
            logits,
            logits,
            confidence,
            logits.new_zeros(()),
            features=features,
            aux_outputs={
                "features": layer_features,
                "logits": aux_logits,
                "meta_logits": meta_logits,
                "fusion_weights": fusion_weights,
            },
        )

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.kd_temperature
        log_p_student = F.log_softmax(student_logits.float() / temperature, dim=1)
        p_teacher = F.softmax(teacher_logits.float().detach() / temperature, dim=1)
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature**2)

    def training_loss(self, output, targets, indices, epoch, args):
        aux_logits = output.aux_outputs["logits"]
        aux_meta_logits = output.aux_outputs["meta_logits"]
        ce = F.cross_entropy(output.train_logits, targets)

        class_weights = getattr(args, "ds_class_weights", None)
        ce_weight = None
        if class_weights is not None:
            ce_weight = torch.as_tensor(class_weights, device=targets.device, dtype=output.train_logits.float().dtype)

        aux_ce = output.train_logits.new_zeros(())
        kd = output.train_logits.new_zeros(())
        for layer in self.layers:
            aux_ce = aux_ce + F.cross_entropy(aux_logits[layer], targets, weight=ce_weight)
            kd = kd + self._kd_loss(aux_logits[layer], output.train_logits)
        aux_ce = aux_ce / len(self.layers)
        kd = kd / len(self.layers)

        aux_weight = float(getattr(args, "ds_aux_ce_weight", 0.3))
        loss = ce + aux_weight * ((1.0 - self.kd_weight) * aux_ce + self.kd_weight * kd)
        if epoch <= args.pretrain:
            return loss

        with torch.no_grad():
            probs = F.softmax(output.train_logits.float(), dim=1)
            correct = output.train_logits.argmax(dim=1).eq(targets).float()
            if getattr(args, "scsf_meta_target", "tcp") == "correctness":
                target_conf = correct
            else:
                current_tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                ema_momentum = float(getattr(args, "tcp_ema_momentum", 0.0))
                if ema_momentum > 0.0 and self._tcp_ema is not None and indices is not None:
                    idx = indices.cpu()
                    current_cpu = current_tcp.detach().cpu()
                    initialized = self._tcp_initialized[idx]
                    self._tcp_ema[idx] = torch.where(
                        initialized,
                        ema_momentum * self._tcp_ema[idx] + (1.0 - ema_momentum) * current_cpu,
                        current_cpu,
                    )
                    self._tcp_initialized[idx] = True
                    target_conf = self._tcp_ema[idx].to(output.train_logits.device)
                else:
                    target_conf = current_tcp

        weights = torch.ones_like(correct)
        weights[correct < 0.5] = args.error_weight
        focal_gamma = float(getattr(args, "meta_focal_gamma", 0.0))
        if focal_gamma > 0.0:
            focal_weight = (1.0 - target_conf.detach()).clamp(min=1e-6).pow(focal_gamma)
            weights = weights * focal_weight
        if class_weights is not None:
            weights = weights * ce_weight[targets].to(weights.dtype)

        meta_losses = []
        for layer in self.layers:
            layer_meta_logits = aux_meta_logits[layer].float()
            if layer_meta_logits.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    meta_losses.append(
                        F.binary_cross_entropy_with_logits(layer_meta_logits, target_conf.float(), weight=weights.float())
                    )
            else:
                meta_losses.append(
                    F.binary_cross_entropy_with_logits(layer_meta_logits, target_conf.float(), weight=weights.float())
                )
        meta_loss = sum(meta_losses) / len(meta_losses)
        fused_cal_weight = max(0.0, float(getattr(args, "ds_fused_cal_weight", 0.0)))
        if fused_cal_weight > 0.0:
            fused_confidence = output.confidence.float().clamp(1e-7, 1.0 - 1e-7)
            if fused_confidence.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    fused_meta_loss = F.binary_cross_entropy(
                        fused_confidence,
                        target_conf.float(),
                        weight=weights.float(),
                    )
            else:
                fused_meta_loss = F.binary_cross_entropy(
                    fused_confidence,
                    target_conf.float(),
                    weight=weights.float(),
                )
            meta_loss = (meta_loss + fused_cal_weight * fused_meta_loss) / (1.0 + fused_cal_weight)

        ranking_loss = pairwise_ranking_loss(output.confidence.float(), correct, args)
        rank_alpha = min(1.0, max(0.0, float(getattr(args, "ds_rank_alpha", 0.3))))
        if ranking_loss is None:
            combined_meta = meta_loss
        else:
            combined_meta = (1.0 - rank_alpha) * meta_loss + rank_alpha * ranking_loss

        gate_entropy_weight = float(getattr(args, "ds_gate_entropy_weight", 0.01))
        if gate_entropy_weight > 0.0:
            fusion_weights = output.aux_outputs["fusion_weights"].float().clamp(min=1e-8)
            entropy = -(fusion_weights * fusion_weights.log()).sum(dim=1).mean()
            max_entropy = math.log(max(2, len(self.layers)))
            combined_meta = combined_meta - gate_entropy_weight * (entropy / max_entropy)

        gate_supervision_weight = max(0.0, float(getattr(args, "ds_gate_supervision_weight", 0.0)))
        if gate_supervision_weight > 0.0 and len(self.layers) > 1:
            temperature = max(1e-6, float(getattr(args, "ds_gate_supervision_temperature", 0.5)))
            with torch.no_grad():
                branch_losses = torch.stack(
                    [
                        F.cross_entropy(aux_logits[layer].float(), targets, reduction="none")
                        for layer in self.layers
                    ],
                    dim=1,
                )
                gate_targets = F.softmax(-branch_losses / temperature, dim=1)
            fusion_weights = output.aux_outputs["fusion_weights"].float().clamp(min=1e-8)
            gate_loss = F.kl_div(fusion_weights.log(), gate_targets, reduction="batchmean")
            combined_meta = combined_meta + gate_supervision_weight * gate_loss

        meta_weight = SCSFMethod._meta_weight(epoch, args)
        return loss + float(getattr(args, "ds_aux_cal_weight", 1.0)) * meta_weight * combined_meta


class AuxGradLogMixin:
    """Once per epoch after warm-up, record the auxiliary loss's gradient norm at ResNet layer2/3/4.

    The norms must be ~0 in the detached-feature controls; this checks the implementation, while the
    active-vs-detached performance comparison establishes the contribution of deep supervision.
    """

    def on_epoch_start(self, epoch: int, args):
        self._grad_log_pending = epoch > args.pretrain

    def _maybe_log_aux_grad(self, losses: torch.Tensor | dict[str, torch.Tensor], epoch: int):
        """losses: the auxiliary loss, or {name: loss term} to log each term's gradient norms separately."""
        if not isinstance(losses, dict):
            losses = {"aux": losses}
        if not getattr(self, "_grad_log_pending", False) or not any(loss.requires_grad for loss in losses.values()):
            return
        self._grad_log_pending = False
        blocks = {name: getattr(self.backbone, name) for name in ("layer2", "layer3", "layer4") if hasattr(self.backbone, name)}
        if not blocks:
            return
        params = {name: [p for p in block.parameters() if p.requires_grad] for name, block in blocks.items()}
        flat = [p for group in params.values() for p in group]
        row = {"epoch": epoch}
        for term, loss in losses.items():
            row[f"{term}_loss"] = float(loss.detach())
            grads = (
                torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
                if loss.requires_grad
                else [None] * len(flat)
            )
            offset = 0
            for name, group in params.items():
                squared = sum(float(g.float().pow(2).sum()) for g in grads[offset : offset + len(group)] if g is not None)
                row[f"{name}_{term}_grad_norm"] = squared ** 0.5
                offset += len(group)
        self.aux_grad_log.append(row)
        print("aux grad norms: " + " ".join(f"{k}={v:.3e}" for k, v in row.items() if k.endswith("norm")))


DUALAUG_TARGETS = ("agreement", "correctness", "correct_agree")


class DualAugMethod(AuxGradLogMixin, SCSFMethod):
    """dualaug (port of train_fresh_sc.py --variant dualaug).

    Loss = 0.5 * [CE(x) + CE(flip(x))] + lambda_t * BCE(head, target).
    The head is the SCSF calibrator on pooled mid+late features (gradient flows into the backbone unless
    detach_features) and stop-gradient logits. lambda_t is 0 during warm-up, then cosine-decays
    init -> min meta weight. Test inputs are scored by the raw head logit (single view, no flip).

    target: agreement = 1[argmax f(x) == argmax f(Tx)] (original dualaug);
            correctness = 1[argmax f(x) == y];
            correct_agree = a0 * (0.5 + 0.5 * a1), a_i = 1[view i is correct] (dualaug_correctness).
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        hidden_dim: int = 256,
        calibrator_arch: str = "standard",
        target: str = "agreement",
        detach_features: bool = False,
    ):
        super().__init__(backbone, num_classes, hidden_dim=hidden_dim, use_logits=True, scorer="meta", calibrator_arch=calibrator_arch)
        if target not in DUALAUG_TARGETS:
            raise ValueError(f"Unknown dualaug target '{target}'; choose from {DUALAUG_TARGETS}")
        self.target = target
        self.detach_features = detach_features
        self.aux_grad_log: list[dict] = []

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._backbone_logits_features(x)
        meta_logits = self.calibrator(features.detach() if self.detach_features else features, logits)
        aux_outputs = None
        if self.training:
            aux_outputs = {"flip_logits": self.backbone(x.flip(-1))}
        return MethodBatchOutput(
            logits,
            logits,
            meta_logits.float(),
            logits.new_zeros(()),
            features=features,
            meta_logits=meta_logits,
            aux_outputs=aux_outputs,
        )

    def _head_target(self, logits: torch.Tensor, flip_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred, flip_pred = logits.argmax(dim=1), flip_logits.argmax(dim=1)
        if self.target == "agreement":
            return pred.eq(flip_pred).float()
        correct = pred.eq(targets).float()
        if self.target == "correctness":
            return correct
        return correct * (0.5 + 0.5 * flip_pred.eq(targets).float())

    def training_loss(self, output, targets, indices, epoch, args):
        logits = output.train_logits
        flip_logits = output.aux_outputs["flip_logits"]
        ce = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(flip_logits, targets))
        if epoch <= args.pretrain:
            return ce
        with torch.no_grad():
            head_target = self._head_target(logits, flip_logits, targets)
        with torch.amp.autocast("cuda", enabled=False):
            bce = F.binary_cross_entropy_with_logits(output.meta_logits.float(), head_target)
        self._maybe_log_aux_grad(bce, epoch)
        return ce + self._meta_weight(epoch, args) * bce


class _ClassVerifierBase(AuxGradLogMixin, BackboneMethod):
    """Two-view CE classifier plus a class-conditioned verifier on ResNet layer3 (Proposals A/B).

    The verifier produces one raw logit u(x, c) per candidate class c. Training:
        L = 0.5 [CE(z(x), y) + CE(z(Tx), y)] + alpha_t / 2 [L_ver(x, y) + L_ver(Tx, y)],
    alpha_t = 0 during warm-up and verify_weight afterwards. Inference keeps argmax z(x) and scores
    it with u(x, argmax z(x)) from a single backbone pass.

    loss="binary": L_ver = 0.5 softplus(-u_y) + 1/(2(C-1)) sum_{c != y} softplus(u_c)
    loss="softmax": L_ver = CE(u, y), scored by log softmax(u)[pred] (same order as the probability,
                    without fp32 saturation ties); the auxiliary-classification control.
    detach_features: the verifier reads layer3.detach() (no auxiliary gradient into the backbone).
    """

    head_prefixes: tuple[str, ...] = ()

    def __init__(self, backbone: FeatureModel, num_classes: int, loss: str = "binary", detach_features: bool = False):
        super().__init__(backbone, num_classes)
        if not hasattr(backbone, "forward_with_maps") or not getattr(backbone, "_is_resnet", False):
            raise ValueError("Class-conditioned verifiers need a ResNet backbone (layer3 feature map).")
        if loss not in {"binary", "softmax"}:
            raise ValueError(f"Unknown verifier loss '{loss}'")
        if num_classes < 2:
            raise ValueError("Class-conditioned verification needs at least two classes.")
        self.loss = loss
        self.detach_features = detach_features
        self.aux_grad_log: list[dict] = []

    def _candidate_logits(self, layer3: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _verify(self, layer3: torch.Tensor) -> torch.Tensor:
        layer3 = layer3.detach() if self.detach_features else layer3
        if layer3.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                return self._candidate_logits(layer3.float())
        return self._candidate_logits(layer3.float())

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features, maps = self.backbone.forward_with_maps(x)
        candidates = self._verify(maps["mid"])
        scores = candidates if self.loss == "binary" else F.log_softmax(candidates, dim=1)
        confidence = scores.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
        aux_outputs = {"candidates": candidates}
        if self.training:
            flip_logits, _, flip_maps = self.backbone.forward_with_maps(x.flip(-1))
            aux_outputs.update({"flip_logits": flip_logits, "flip_candidates": self._verify(flip_maps["mid"])})
        return MethodBatchOutput(logits, logits, confidence, logits.new_zeros(()), features=features, aux_outputs=aux_outputs)

    def _verification_loss(self, candidates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss == "softmax":
            return F.cross_entropy(candidates, targets)
        positive = F.one_hot(targets, self.num_classes).bool()
        pos_term = F.softplus(-candidates[positive])
        neg_term = F.softplus(candidates).masked_fill(positive, 0.0).sum(dim=1) / (self.num_classes - 1)
        return (0.5 * pos_term + 0.5 * neg_term).mean()

    def training_loss(self, output, targets, indices, epoch, args):
        flip_logits = output.aux_outputs["flip_logits"]
        ce = 0.5 * (F.cross_entropy(output.train_logits, targets) + F.cross_entropy(flip_logits, targets))
        if epoch <= args.pretrain:
            return ce
        with torch.amp.autocast("cuda", enabled=False):
            verification = 0.5 * (
                self._verification_loss(output.aux_outputs["candidates"], targets)
                + self._verification_loss(output.aux_outputs["flip_candidates"], targets)
            )
        self._maybe_log_aux_grad(verification, epoch)
        return ce + args.verify_weight * verification


class VerifyGlobalMethod(_ClassVerifierBase):
    """Proposal A: u(x, c) = MLP([GAP(layer3(x)), e_c]) with a learned class embedding e_c."""

    head_prefixes = ("class_embed.", "verifier.")

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        loss: str = "binary",
        detach_features: bool = False,
    ):
        super().__init__(backbone, num_classes, loss=loss, detach_features=detach_features)
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.verifier = nn.Sequential(
            nn.Linear(backbone.layer3_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _candidate_logits(self, layer3: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(layer3, (1, 1)).flatten(1)
        batch, classes = pooled.size(0), self.num_classes
        pairs = torch.cat(
            [
                pooled.unsqueeze(1).expand(batch, classes, pooled.size(1)),
                self.class_embed.weight.unsqueeze(0).expand(batch, classes, -1),
            ],
            dim=2,
        )
        return self.verifier(pairs).squeeze(2)


class VerifySpatialMethod(_ClassVerifierBase):
    """Proposal B: A = Conv1x1(layer3) in R^{C x H x W}; u(x, c) = mean of the top ceil(topk_fraction * HW)
    values of A_c (pool="topk"), or the spatial mean of A_c (pool="gap", same parameters)."""

    head_prefixes = ("evidence_conv.",)

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        pool: str = "topk",
        topk_fraction: float = 0.2,
        loss: str = "binary",
        detach_features: bool = False,
    ):
        super().__init__(backbone, num_classes, loss=loss, detach_features=detach_features)
        if pool not in {"topk", "gap"}:
            raise ValueError(f"Unknown verify_spatial pool '{pool}'")
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("--verify-topk-fraction must be in (0, 1]")
        self.pool = pool
        self.topk_fraction = topk_fraction
        self.evidence_conv = nn.Conv2d(backbone.layer3_dim, num_classes, kernel_size=1)

    def _candidate_logits(self, layer3: torch.Tensor) -> torch.Tensor:
        evidence = self.evidence_conv(layer3).flatten(2)
        if self.pool == "gap":
            return evidence.mean(dim=2)
        k = max(1, math.ceil(self.topk_fraction * evidence.size(2)))
        return evidence.topk(k, dim=2).values.mean(dim=2)


# --- Single-view deep supervision (docs/medical_deep_supervision_method_proposals (1).md) ---
# none: auxiliary gradients enter the backbone; all: every auxiliary branch reads detached features
# (All-DS-detached); new: only the new branch is detached (MG-SCSF control; SCSF's TCP path stays active).
DS_DETACH_MODES = ("none", "all", "new")
DS_EPS = 1e-6


def ds_aux_coefficient(epoch: int, args) -> float:
    """alpha_e = w * min(1, max(0, (e - pretrain) / ramp)), epochs numbered from one."""
    ramp = max(1, int(getattr(args, "ds_aux_ramp", 5)))
    return float(getattr(args, "ds_aux_weight", 0.1)) * min(1.0, max(0.0, (epoch - args.pretrain) / ramp))


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=DS_EPS)


def class_margin(scores: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """scores[c] - max_{c' != c} scores[c'] for the given class of every row."""
    own = scores.gather(1, classes.unsqueeze(1)).squeeze(1)
    others = scores.scatter(1, classes.unsqueeze(1), float("-inf")).max(dim=1).values
    return own - others


@contextmanager
def isolated_init_rng(seed: int):
    """Initialise auxiliary parameters from their own CPU RNG stream, leaving the global stream untouched,
    so controls with different heads keep the backbone initialisation, data order and augmentations."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _fp32_region(device: torch.device):
    return torch.amp.autocast("cuda", enabled=False) if device.type == "cuda" else nullcontext()


class DSDiagnosticsMixin(AuxGradLogMixin):
    """Per-epoch training diagnostics, summed on device during the epoch and flushed to aux_diagnostics."""

    def _init_ds_logs(self):
        self.aux_grad_log: list[dict] = []
        self.aux_diagnostics: list[dict] = []
        self._diag_sums: dict[str, torch.Tensor] = {}
        self._diag_epoch = 0

    def on_epoch_start(self, epoch: int, args):
        super().on_epoch_start(epoch, args)
        self.flush_diagnostics()
        self._diag_epoch = epoch

    @torch.no_grad()
    def _accumulate(self, **values: torch.Tensor):
        for key, value in values.items():
            value = value.detach().float()
            self._diag_sums[key] = self._diag_sums[key] + value if key in self._diag_sums else value

    def flush_diagnostics(self):
        if not self._diag_sums:
            return
        sums = {key: value.cpu() for key, value in self._diag_sums.items()}
        self._diag_sums = {}
        row = {"epoch": self._diag_epoch}
        row.update(self._diagnostic_row(sums))
        self.aux_diagnostics.append(row)

    def _diagnostic_row(self, sums: dict[str, torch.Tensor]) -> dict:
        raise NotImplementedError

    @staticmethod
    def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        return float(numerator / denominator) if float(denominator) > 0 else float("nan")

    def _accumulate_disagreement(self, aux_scores: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
        pred = logits.argmax(dim=1)
        correct = pred.eq(targets).float()
        disagree = aux_scores.argmax(dim=1).ne(pred).float()
        self._accumulate(
            n_correct=correct.sum(),
            n_wrong=(1.0 - correct).sum(),
            disagree_correct=(disagree * correct).sum(),
            disagree_wrong=(disagree * (1.0 - correct)).sum(),
        )

    def _disagreement_row(self, sums: dict[str, torch.Tensor]) -> dict:
        return {
            "final_aux_disagree_given_correct": self._ratio(sums["disagree_correct"], sums["n_correct"]),
            "final_aux_disagree_given_wrong": self._ratio(sums["disagree_wrong"], sums["n_wrong"]),
        }


def _require_resnet(backbone: FeatureModel, method: str):
    if not hasattr(backbone, "forward_with_maps") or not getattr(backbone, "_is_resnet", False):
        raise ValueError(f"{method} needs a ResNet backbone (layer3 feature map).")


class MGSCSFMethod(DSDiagnosticsMixin, SCSFMethod):
    """MG-SCSF: SCSF plus a training-only, margin-gated cosine classifier on GAP(layer3).

    u_c = <normalize(GAP(layer3)), normalize(w_c)>, d = u_y - max_{c != y} u_c,
    L = CE(z, y) + lambda_e BCE(s_SCSF, sg[p_y]) + alpha_e mean([max(0, margin - d)]^2).
    loss="ce" is the mechanism control CE(ce_scale * u, y) with the identical branch and schedule.
    SCSF keeps its calibrator (Adam), TCP target and cosine lambda; inference uses the raw SCSF logit.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        hidden_dim: int = 256,
        use_logits: bool = True,
        calibrator_arch: str = "standard",
        train_size: int = 0,
        loss: str = "hinge",
        margin: float = 0.2,
        ce_scale: float = 10.0,
        detach: str = "none",
        init_seed: int = 0,
    ):
        super().__init__(
            backbone,
            num_classes,
            hidden_dim=hidden_dim,
            use_logits=use_logits,
            scorer="meta",
            calibrator_arch=calibrator_arch,
            train_size=train_size,
        )
        _require_resnet(backbone, "mg_scsf")
        if loss not in {"hinge", "ce"}:
            raise ValueError(f"Unknown mg_scsf loss '{loss}'")
        if detach not in DS_DETACH_MODES:
            raise ValueError(f"Unknown detach mode '{detach}'; choose from {DS_DETACH_MODES}")
        self.loss = loss
        self.margin = margin
        self.ce_scale = ce_scale
        self.detach = detach
        with isolated_init_rng(init_seed):
            self.margin_weight = nn.Parameter(torch.empty(num_classes, backbone.layer3_dim))
            nn.init.kaiming_uniform_(self.margin_weight, a=math.sqrt(5))
        self._init_ds_logs()

    def _margin_scores(self, layer3: torch.Tensor) -> torch.Tensor:
        if self.detach != "none":
            layer3 = layer3.detach()
        with _fp32_region(layer3.device):
            pooled = F.adaptive_avg_pool2d(layer3.float(), (1, 1)).flatten(1)
            return _unit(pooled) @ _unit(self.margin_weight).t()

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features, maps = self.backbone.forward_with_maps(x)
        meta_logits = self.calibrator(features.detach() if self.detach == "all" else features, logits)
        aux_outputs = {"margin_scores": self._margin_scores(maps["mid"])} if self.training else None
        return MethodBatchOutput(
            logits,
            logits,
            meta_logits.float(),
            logits.new_zeros(()),
            features=features,
            meta_logits=meta_logits,
            aux_outputs=aux_outputs,
        )

    def training_loss(self, output, targets, indices, epoch, args):
        total = SCSFMethod.training_loss(self, output, targets, indices, epoch, args)
        alpha = ds_aux_coefficient(epoch, args)
        if alpha <= 0.0:
            return total
        scores = output.aux_outputs["margin_scores"]
        with _fp32_region(scores.device):
            margins = class_margin(scores, targets)
            if self.loss == "hinge":
                aux = F.relu(self.margin - margins).pow(2).mean()
            else:
                aux = F.cross_entropy(self.ce_scale * scores, targets)
        onehot = F.one_hot(targets, self.num_classes).float()
        active = margins.lt(self.margin).float()
        self._accumulate(
            class_count=onehot.sum(0),
            class_active=(onehot * active.unsqueeze(1)).sum(0),
            class_margin=(onehot * margins.unsqueeze(1)).sum(0),
        )
        ce = F.cross_entropy(output.train_logits, targets)
        self._maybe_log_aux_grad({"ce": ce, "tcp": total - ce, "margin": alpha * aux}, epoch)
        return total + alpha * aux

    def _diagnostic_row(self, sums):
        count = sums["class_count"]
        row = {
            "active_frac": self._ratio(sums["class_active"].sum(), count.sum()),
            "mean_margin": self._ratio(sums["class_margin"].sum(), count.sum()),
        }
        for c in range(self.num_classes):
            row[f"class{c}_active_frac"] = self._ratio(sums["class_active"][c], count[c])
            row[f"class{c}_mean_margin"] = self._ratio(sums["class_margin"][c], count[c])
        return row


# Backbone stages read by the mp_ds/se_ds branch; layer3+layer4 mirrors SCSF's mid+late features.
DS_SITES = {"layer3": ("mid",), "layer4": ("late",), "layer3+layer4": ("mid", "late")}


class _LabelEvidenceDSBase(DSDiagnosticsMixin, BackboneMethod):
    """Single-view CE plus a class-label-supervised branch on layer3, layer4 or both (MP-DS, SE-DS).

    L = CE(z(x), y) + alpha_e CE(a(x) / temperature, y). Inference keeps argmax z and scores it with
    s(x) = a_yhat - max_{c != yhat} a_c (the final prediction, never the branch's own argmax).
    detach="all" feeds the branch detached stage maps (All-DS-detached).
    With a layer4 site the auxiliary loss also reaches layer4, i.e. the classifier's input features.
    """

    temperature = 1.0

    def __init__(self, backbone: FeatureModel, num_classes: int, detach: str, method: str, site: str = "layer3"):
        super().__init__(backbone, num_classes)
        _require_resnet(backbone, method)
        if num_classes < 2:
            raise ValueError(f"{method} needs at least two classes.")
        if detach not in {"none", "all"}:
            raise ValueError(f"{method} supports --ds-detach none or all, got '{detach}'")
        if site not in DS_SITES:
            raise ValueError(f"Unknown --ds-site '{site}'; choose from {tuple(DS_SITES)}")
        self.detach = detach
        self.site = site
        self.site_stages = DS_SITES[site]
        self.site_dim = sum(backbone._stage_dims[stage] for stage in self.site_stages)
        self._init_ds_logs()

    def _branch(self, stage_maps: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Class scores under key "scores" (plus optional extras) from the site's maps, in float32."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features, maps = self.backbone.forward_with_maps(x)
        stage_maps = [maps[stage].detach() if self.detach == "all" else maps[stage] for stage in self.site_stages]
        with _fp32_region(logits.device):
            branch = self._branch([stage_map.float() for stage_map in stage_maps])
        confidence = class_margin(branch["scores"], logits.argmax(dim=1))
        return MethodBatchOutput(logits, logits, confidence, logits.new_zeros(()), features=features, aux_outputs=branch)

    def training_loss(self, output, targets, indices, epoch, args):
        ce = F.cross_entropy(output.train_logits, targets)
        alpha = ds_aux_coefficient(epoch, args)
        if alpha <= 0.0:
            return ce
        scores = output.aux_outputs["scores"]
        with _fp32_region(scores.device):
            aux = F.cross_entropy(scores / self.temperature, targets)
        self._accumulate_disagreement(scores, output.train_logits, targets)
        self._accumulate_branch(output.aux_outputs, targets)
        self._maybe_log_aux_grad({"ce": ce, "aux": alpha * aux}, epoch)
        return ce + alpha * aux

    def _accumulate_branch(self, branch: dict[str, torch.Tensor], targets: torch.Tensor):
        pass

    def _diagnostic_row(self, sums):
        return self._disagreement_row(sums)


class MPDSMethod(_LabelEvidenceDSBase):
    """MP-DS: K learned, normalised prototypes per class on v = normalize(W h) (W bias-free, 128-d), where h is
    GAP(layer3), GAP(layer4) or their concatenation (--ds-site).

    a_c = max_k <v, normalize(p_ck)>, L_MP = CE(a / 0.1, y); prototypes are jointly optimised (no clustering,
    memory bank, occupancy balancing or restarts). K=1 is the mechanism control.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        prototypes: int = 3,
        dim: int = 128,
        temperature: float = 0.1,
        detach: str = "none",
        init_seed: int = 0,
        site: str = "layer3",
    ):
        super().__init__(backbone, num_classes, detach, "mp_ds", site=site)
        if prototypes < 1:
            raise ValueError("--mp-prototypes must be >= 1")
        self.num_prototypes = prototypes
        self.temperature = temperature
        with isolated_init_rng(init_seed):
            self.proto_proj = nn.Linear(self.site_dim, dim, bias=False)
            self.prototypes = nn.Parameter(torch.randn(num_classes, prototypes, dim))

    def _branch(self, stage_maps):
        pooled = torch.cat([F.adaptive_avg_pool2d(stage_map, (1, 1)).flatten(1) for stage_map in stage_maps], dim=1)
        sims = torch.einsum("bd,ckd->bck", _unit(self.proto_proj(pooled)), _unit(self.prototypes))
        return {"scores": sims.max(dim=2).values, "sims": sims}

    def _accumulate_branch(self, branch, targets):
        own = branch["sims"][torch.arange(targets.size(0), device=targets.device), targets]
        selected = own.argmax(dim=1)
        occupancy = torch.zeros(self.num_classes, self.num_prototypes, device=targets.device)
        occupancy.index_put_((targets, selected), torch.ones_like(selected, dtype=torch.float), accumulate=True)
        self._accumulate(center_occupancy=occupancy)

    @torch.no_grad()
    def _diagnostic_row(self, sums):
        row = self._disagreement_row(sums)
        occupancy = sums["center_occupancy"]
        prototypes = _unit(self.prototypes.detach()).cpu()
        for c in range(self.num_classes):
            total = occupancy[c].sum()
            for k in range(self.num_prototypes):
                row[f"class{c}_center{k}_occupancy"] = self._ratio(occupancy[c, k], total)
            row[f"class{c}_unused_centers"] = int((occupancy[c] == 0).sum())
            if self.num_prototypes > 1:
                cosine = prototypes[c] @ prototypes[c].t()
                off_diagonal = cosine[~torch.eye(self.num_prototypes, dtype=torch.bool)]
                row[f"class{c}_center_cos"] = float(off_diagonal.mean())
        return row


class SEDSMethod(_LabelEvidenceDSBase):
    """SE-DS: A = Conv1x1(H) (with bias) in R^{C x H x W}; a_c = mean of the top ceil(0.2 HW) values of A_c,
    L_SE = CE(a, y). pool="gap" (spatial mean, same convolution) is the mechanism control. Raw scores, no sigmoid.

    H is layer3, layer4, or (layer3+layer4) layer3 concatenated with layer4 nearest-upsampled to layer3's grid,
    so the evidence is Conv(layer3) + upsample(Conv(layer4)) at the finer resolution.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        num_classes: int,
        pool: str = "topk",
        topk_fraction: float = 0.2,
        detach: str = "none",
        init_seed: int = 0,
        site: str = "layer3",
    ):
        super().__init__(backbone, num_classes, detach, "se_ds", site=site)
        if pool not in {"topk", "gap"}:
            raise ValueError(f"Unknown se_ds pool '{pool}'")
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("--se-topk-fraction must be in (0, 1]")
        self.pool = pool
        self.topk_fraction = topk_fraction
        with isolated_init_rng(init_seed):
            self.evidence_conv = nn.Conv2d(self.site_dim, num_classes, kernel_size=1, bias=True)

    def _branch(self, stage_maps):
        size = stage_maps[0].shape[-2:]
        stacked = torch.cat(
            [m if m.shape[-2:] == size else F.interpolate(m, size=size, mode="nearest") for m in stage_maps], dim=1
        )
        evidence = self.evidence_conv(stacked).flatten(2)
        if self.pool == "gap":
            return {"scores": evidence.mean(dim=2)}
        k = max(1, math.ceil(self.topk_fraction * evidence.size(2)))
        return {"scores": evidence.topk(k, dim=2).values.mean(dim=2)}


class CCLSCMethod(BackboneMethod):
    """CCL-SC (Wu et al., ICML 2024) with a momentum encoder and correct/error feature queues.

    variant="official" reproduces lamda-bbo/CCL-SC (train_CCL_SC.py, moco/CSC.py), which produced
    the paper's numbers:
      - key encoder copied from the online model when warm-up ends, then EMA-updated once per epoch;
      - queues filled only after warm-up and frozen once both are full;
      - per-positive loss logsumexp(l - max(l)) - s_p with l = sims / tau (the max shift is never
        added back and s_p is not divided by tau).
    variant="paper" implements the CSC loss as written in the paper (Eq. 6):
      - key encoder copied when warm-up ends, then EMA-updated every step (MoCo);
      - queues updated FIFO every step after warm-up;
      - per-positive loss logsumexp(l) - s_p / tau.
    Both variants start the CSC term only once both queues are full, weight each anchor by its
    detached softmax response divided by its number of positives, and score test inputs by SR.
    """

    def __init__(
        self,
        backbone: FeatureModel,
        key_backbone: FeatureModel,
        num_classes: int,
        queue_size: int,
        momentum: float,
        temperature: float,
        base_temperature: float,
        variant: str = "official",
    ):
        super().__init__(backbone, num_classes)
        if variant not in {"official", "paper"}:
            raise ValueError(f"Unknown CCL-SC variant: {variant}")
        self.key_backbone = key_backbone
        self.key_backbone.load_state_dict(self.backbone.state_dict())
        for param in self.key_backbone.parameters():
            param.requires_grad = False

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.variant = variant
        self._epoch = 0
        self._key_initialized = False

        # Official VGG16-BN embedding: output of classifier[:3] (Linear -> ReLU -> BatchNorm1d).
        self._vgg_projection = isinstance(backbone, VGGFeatureModel)
        if self._vgg_projection:
            first_linear = next(m for m in backbone.base.classifier.modules() if isinstance(m, nn.Linear))
            feature_dim = first_linear.out_features
        else:
            feature_dim = backbone.feature_dim
        self.register_buffer("correct_queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
        self.register_buffer("error_queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
        self.register_buffer("correct_queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("error_queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("correct_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("error_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("correct_queue_full", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("error_queue_full", torch.zeros(1, dtype=torch.bool))

    def _encode(self, model: FeatureModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._vgg_projection:
            flat = model.forward_features(x)
            layers = list(model.base.classifier.children())
            projection = nn.Sequential(*layers[:3])(flat)
            logits = nn.Sequential(*layers[3:])(projection)
            return logits, projection
        return model(x, return_features=True)

    def _past_warmup(self, args) -> bool:
        return self._epoch > args.pretrain

    @torch.no_grad()
    def _copy_key_encoder(self):
        for online_param, key_param in zip(self.backbone.parameters(), self.key_backbone.parameters()):
            key_param.data.copy_(online_param.data)
        self._key_initialized = True

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for online_param, key_param in zip(self.backbone.parameters(), self.key_backbone.parameters()):
            key_param.data.mul_(self.momentum).add_(online_param.data, alpha=1.0 - self.momentum)

    def on_epoch_start(self, epoch: int, args):
        self._epoch = epoch
        if epoch <= args.pretrain:
            return
        if not self._key_initialized:
            self._copy_key_encoder()
        elif self.variant == "official":
            self._momentum_update_key_encoder()

    @torch.no_grad()
    def _enqueue(self, features: torch.Tensor, labels: torch.Tensor, prefix: str):
        if features.numel() == 0:
            return
        queue = getattr(self, f"{prefix}_queue")
        label_queue = getattr(self, f"{prefix}_queue_labels")
        ptr = getattr(self, f"{prefix}_queue_ptr")
        full = getattr(self, f"{prefix}_queue_full")

        features = features.detach()
        labels = labels.detach().long()
        if features.size(0) >= self.queue_size:
            features = features[-self.queue_size :]
            labels = labels[-self.queue_size :]

        start = int(ptr.item())
        count = features.size(0)
        end = start + count
        if end <= self.queue_size:
            queue[start:end].copy_(features)
            label_queue[start:end].copy_(labels)
        else:
            first = self.queue_size - start
            queue[start:].copy_(features[:first])
            label_queue[start:].copy_(labels[:first])
            queue[: end % self.queue_size].copy_(features[first:])
            label_queue[: end % self.queue_size].copy_(labels[first:])

        ptr[0] = end % self.queue_size
        if end >= self.queue_size or count >= self.queue_size:
            full[0] = True

    def _queue_view(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        queue = getattr(self, f"{prefix}_queue")
        labels = getattr(self, f"{prefix}_queue_labels")
        full = getattr(self, f"{prefix}_queue_full")
        ptr = getattr(self, f"{prefix}_queue_ptr")
        if bool(full.item()):
            return queue, labels
        size = int(ptr.item())
        return queue[:size], labels[:size]

    def _queues_full(self) -> bool:
        return bool(self.correct_queue_full.item()) and bool(self.error_queue_full.item())

    def csc_loss(self, logits: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CSC loss over the full queues; anchors without positives contribute zero."""
        q = F.normalize(features, dim=1)
        sr = F.softmax(logits, dim=1).max(dim=1).values.detach()

        # Snapshot: the queues are updated in place before backward() in the paper variant.
        pos_sim = q @ self.correct_queue.clone().t()
        pos_mask = targets.view(-1, 1).eq(self.correct_queue_labels.view(1, -1))
        neg_sim = q @ self.error_queue.clone().t()
        neg_mask = targets.view(-1, 1).eq(self.error_queue_labels.view(1, -1))
        # Same masking constant as the official code: exp() underflows to 0 with zero gradient.
        neg_logits = neg_sim.masked_fill(~neg_mask, -1e9) / self.temperature
        neg_lse = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        pos_logits = pos_sim / self.temperature
        lse = torch.logaddexp(pos_logits, neg_lse)
        if self.variant == "official":
            row_max = torch.maximum(pos_logits, neg_logits.max(dim=1, keepdim=True).values)
            per_positive = (lse - row_max) - pos_sim
        else:
            per_positive = lse - pos_logits

        pos_count = pos_mask.sum(dim=1, keepdim=True).clamp_min(1)
        weights = pos_mask.float() * (sr.view(-1, 1) / pos_count)
        loss = (per_positive * weights).sum() / targets.size(0)
        return (self.temperature / self.base_temperature) * loss

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._encode(self.backbone, x)
        confidence = F.softmax(logits, dim=1).max(dim=1).values
        key_logits = None
        key_features = None
        if self.training and self._key_initialized:
            if self.variant == "paper":
                self._momentum_update_key_encoder()
            with torch.no_grad():
                key_logits, key_features = self._encode(self.key_backbone, x)
        return MethodBatchOutput(
            logits,
            logits,
            confidence,
            logits.new_zeros(()),
            features=features,
            key_logits=key_logits,
            key_features=key_features,
        )

    def training_loss(self, output, targets, indices, epoch, args):
        ce = F.cross_entropy(output.train_logits, targets)
        if not self._past_warmup(args) or output.key_logits is None:
            return ce

        queues_full = self._queues_full()
        csc = None
        if queues_full and args.ccl_weight > 0:
            with torch.amp.autocast("cuda", enabled=False):
                csc = self.csc_loss(output.train_logits.float(), output.features.float(), targets)

        if self.variant == "paper" or not queues_full:
            with torch.no_grad():
                key_features = F.normalize(output.key_features, dim=1)
                key_predictions = output.key_logits.argmax(dim=1)
                key_correct = key_predictions.eq(targets)
                self._enqueue(key_features[key_correct], key_predictions[key_correct], "correct")
                self._enqueue(key_features[~key_correct], key_predictions[~key_correct], "error")

        if csc is None:
            return ce
        return ce + args.ccl_weight * csc


class ResidualHeadMethod(BackboneMethod):
    def __init__(self, backbone: FeatureModel, num_classes: int, hidden_dim: int, agree_weight: float = 1.0, min_agree_weight: float = 1e-4):
        super().__init__(backbone, num_classes)
        self.hidden_dim = hidden_dim
        self.agree_weight_max = agree_weight
        self.min_agree_weight = min_agree_weight
        
        layer3_dim = getattr(self.backbone, "layer3_dim", 1024)
        
        self.baseline_head = nn.Linear(num_classes, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(layer3_dim + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        
        self._baseline_frozen = False
        self._layer3_out = None

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        handle = None
        if hasattr(self.backbone, "layer3"):
            handle = self.backbone.layer3.register_forward_hook(
                lambda m, i, o: setattr(self, "_layer3_out", o)
            )
            
        logits, features = self._backbone_logits_features(x)
        
        if handle is not None:
            handle.remove()
            
        pool3 = F.adaptive_avg_pool2d(self._layer3_out, (1, 1)).flatten(1) if self._layer3_out is not None else torch.zeros(x.size(0), getattr(self.backbone, "layer3_dim", 1024), device=x.device)
            
        sg_logits = logits.detach()
        b_val = self.baseline_head(sg_logits)
        
        r_in = torch.cat([pool3, sg_logits], dim=1)
        r_val = self.residual_head(r_in)
        
        conf_logits = b_val + r_val
        confidence = torch.sigmoid(conf_logits).squeeze(-1)
        
        aux_outputs = {}
        if self.training:
            with torch.no_grad():
                flip_logits, _ = self._backbone_logits_features(x.flip(-1))
                agree = (logits.argmax(1) == flip_logits.argmax(1)).float()
            aux_outputs["agree"] = agree
            aux_outputs["conf_logits"] = conf_logits.squeeze(-1)
            
        return MethodBatchOutput(
            train_logits=logits,
            eval_logits=logits,
            confidence=confidence,
            aux_loss=None,
            features=features,
            meta_logits=None,
            aux_outputs=aux_outputs
        )

    def training_loss(self, output: MethodBatchOutput, targets: torch.Tensor, indices: Optional[torch.Tensor], epoch: int, args) -> torch.Tensor:
        ce_loss = F.cross_entropy(output.train_logits, targets)
        
        if epoch <= args.pretrain:
            return ce_loss
            
        if not self._baseline_frozen and epoch > args.pretrain + 5:
            self._baseline_frozen = True
            for p in self.baseline_head.parameters():
                p.requires_grad = False
                
        agree = output.aux_outputs["agree"]
        conf_logits = output.aux_outputs["conf_logits"]
        
        bce_loss = F.binary_cross_entropy_with_logits(conf_logits, agree)
        
        progress = min(1.0, max(0.0, (epoch - args.pretrain - 1) / max(1, args.epochs - args.pretrain - 1)))
        current_weight = self.min_agree_weight + 0.5 * (self.agree_weight_max - self.min_agree_weight) * (1 + math.cos(math.pi * progress))
        
        return ce_loss + current_weight * bce_loss

class DPHeadMethod(BackboneMethod):
    def __init__(self, backbone: FeatureModel, num_classes: int, hidden_dim: int, agree_weight: float = 1.0, min_agree_weight: float = 1e-4, proj_damping: float = 1e-2):
        super().__init__(backbone, num_classes)
        self.hidden_dim = hidden_dim
        self.agree_weight_max = agree_weight
        self.min_agree_weight = min_agree_weight
        self.proj_damping = proj_damping
        
        layer3_dim = getattr(self.backbone, "layer3_dim", 1024)
        
        self.baseline_head = nn.Linear(num_classes, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(layer3_dim + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        
        self._baseline_frozen = False
        self._layer3_out = None

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        handle = None
        if hasattr(self.backbone, "layer3"):
            handle = self.backbone.layer3.register_forward_hook(
                lambda m, i, o: setattr(self, "_layer3_out", o)
            )
            
        logits, features = self._backbone_logits_features(x)
        
        if handle is not None:
            handle.remove()
            
        layer3_out = self._layer3_out
        pool3 = F.adaptive_avg_pool2d(layer3_out, (1, 1)).flatten(1) if layer3_out is not None else torch.zeros(x.size(0), getattr(self.backbone, "layer3_dim", 1024), device=x.device)
            
        sg_logits = logits.detach()
        b_val = self.baseline_head(sg_logits)
        
        r_in = torch.cat([pool3, sg_logits], dim=1)
        r_val = self.residual_head(r_in)
        
        conf_logits = b_val + r_val
        confidence = torch.sigmoid(conf_logits).squeeze(-1)
        
        aux_outputs = {}
        if self.training:
            with torch.no_grad():
                flip_logits, _ = self._backbone_logits_features(x.flip(-1))
                agree = (logits.argmax(1) == flip_logits.argmax(1)).float()
            aux_outputs["agree"] = agree
            aux_outputs["conf_logits"] = conf_logits.squeeze(-1)
            aux_outputs["layer3_out"] = layer3_out
            
        return MethodBatchOutput(
            train_logits=logits,
            eval_logits=logits,
            confidence=confidence,
            aux_loss=None,
            features=features,
            meta_logits=None,
            aux_outputs=aux_outputs
        )

    def training_loss(self, output: MethodBatchOutput, targets: torch.Tensor, indices: Optional[torch.Tensor], epoch: int, args) -> torch.Tensor:
        ce_loss = F.cross_entropy(output.train_logits, targets)
        
        if epoch <= args.pretrain:
            return ce_loss
            
        if not self._baseline_frozen and epoch > args.pretrain + 5:
            self._baseline_frozen = True
            for p in self.baseline_head.parameters():
                p.requires_grad = False
                
        agree = output.aux_outputs["agree"]
        conf_logits = output.aux_outputs["conf_logits"]
        layer3_out = output.aux_outputs.get("layer3_out", None)
        
        bce_loss = F.binary_cross_entropy_with_logits(conf_logits, agree)
        
        progress = min(1.0, max(0.0, (epoch - args.pretrain - 1) / max(1, args.epochs - args.pretrain - 1)))
        current_weight = self.min_agree_weight + 0.5 * (self.agree_weight_max - self.min_agree_weight) * (1 + math.cos(math.pi * progress))
        
        weighted_bce = current_weight * bce_loss
        
        if layer3_out is not None and layer3_out.requires_grad:
            ce_grad = torch.autograd.grad(ce_loss, layer3_out, retain_graph=True)[0]
            
            def hook_fn(grad):
                aux_grad = grad - ce_grad
                ce_grad_flat = ce_grad.reshape(ce_grad.size(0), -1)
                aux_grad_flat = aux_grad.reshape(aux_grad.size(0), -1)
                
                dot = (aux_grad_flat * ce_grad_flat).sum(dim=1, keepdim=True)
                ce_norm_sq = (ce_grad_flat * ce_grad_flat).sum(dim=1, keepdim=True) + self.proj_damping
                
                proj_aux_flat = aux_grad_flat - (dot / ce_norm_sq) * ce_grad_flat
                proj_aux = proj_aux_flat.view_as(aux_grad)
                
                return ce_grad + proj_aux

            layer3_out.register_hook(hook_fn)
            
        return ce_loss + weighted_bce

class SpatialHeadMethod(BackboneMethod):
    def __init__(self, backbone: FeatureModel, num_classes: int, d_head: int = 128, hidden_dim: int = 128, agree_weight: float = 1.0, min_agree_weight: float = 1e-4):
        super().__init__(backbone, num_classes)
        self.d_head = d_head
        self.hidden_dim = hidden_dim
        self.agree_weight_max = agree_weight
        self.min_agree_weight = min_agree_weight
        
        layer3_dim = getattr(self.backbone, "layer3_dim", 1024)
        
        self.q_proj = nn.Linear(num_classes, d_head)
        self.k_proj = nn.Conv2d(layer3_dim, d_head, kernel_size=1)
        self.v_proj = nn.Conv2d(layer3_dim, layer3_dim, kernel_size=1)
        
        self.conf_head = nn.Sequential(
            nn.Linear(layer3_dim * 2 + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self._layer3_spatial = None

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        handle = None
        if hasattr(self.backbone, "layer3"):
            handle = self.backbone.layer3.register_forward_hook(
                lambda m, i, o: setattr(self, "_layer3_spatial", o)
            )
            
        logits, features = self._backbone_logits_features(x)
        
        if handle is not None:
            handle.remove()
            
        spatial_map = self._layer3_spatial
        if spatial_map is None:
            spatial_map = torch.zeros(x.size(0), getattr(self.backbone, "layer3_dim", 1024), 1, 1, device=x.device)
            
        B, C, H, W = spatial_map.shape
        
        sg_probs = F.softmax(logits.detach(), dim=1)
        q = self.q_proj(sg_probs)
        
        k = self.k_proj(spatial_map).view(B, self.d_head, -1)
        v = self.v_proj(spatial_map).view(B, C, -1)
        
        attn_logits = torch.bmm(q.unsqueeze(1), k).squeeze(1) / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        m = torch.bmm(v, attn_weights.unsqueeze(-1)).squeeze(-1)
        
        v_diff = v - m.unsqueeze(-1)
        v_var = torch.bmm((v_diff ** 2), attn_weights.unsqueeze(-1)).squeeze(-1)
        
        conf_in = torch.cat([m, v_var, logits.detach()], dim=1)
        conf_logits = self.conf_head(conf_in)
        confidence = torch.sigmoid(conf_logits).squeeze(-1)
        
        aux_outputs = {}
        if self.training:
            with torch.no_grad():
                flip_logits, _ = self._backbone_logits_features(x.flip(-1))
                agree = (logits.argmax(1) == flip_logits.argmax(1)).float()
            aux_outputs["agree"] = agree
            aux_outputs["conf_logits"] = conf_logits.squeeze(-1)
            
        return MethodBatchOutput(
            train_logits=logits,
            eval_logits=logits,
            confidence=confidence,
            aux_loss=None,
            features=features,
            meta_logits=None,
            aux_outputs=aux_outputs
        )

    def training_loss(self, output: MethodBatchOutput, targets: torch.Tensor, indices: Optional[torch.Tensor], epoch: int, args) -> torch.Tensor:
        ce_loss = F.cross_entropy(output.train_logits, targets)
        
        if epoch <= args.pretrain:
            return ce_loss
            
        agree = output.aux_outputs["agree"]
        conf_logits = output.aux_outputs["conf_logits"]
        
        bce_loss = F.binary_cross_entropy_with_logits(conf_logits, agree)
        
        progress = min(1.0, max(0.0, (epoch - args.pretrain - 1) / max(1, args.epochs - args.pretrain - 1)))
        current_weight = self.min_agree_weight + 0.5 * (self.agree_weight_max - self.min_agree_weight) * (1 + math.cos(math.pi * progress))
        
        return ce_loss + current_weight * bce_loss


def resolve_dg_reward(reward: float | None, num_classes: int) -> float:
    """Deep Gamblers requires 1 < o < num_classes; None picks min(2.2, (1 + num_classes) / 2)."""
    if reward is None:
        return min(2.2, (1.0 + num_classes) / 2.0)
    if not 1.0 < reward < num_classes:
        raise ValueError(f"Deep Gamblers reward must satisfy 1 < o < {num_classes}, got {reward}")
    return reward


def build_method(args, num_classes: int, input_size: int, train_size: int) -> ExperimentMethod:
    method_name = args.method
    model_classes = num_classes + 1 if method_name in {"dg", "sat"} else num_classes
    scsf_feature_spec = getattr(args, "scsf_feature_spec", "mid+late+logits")
    supports_scsf_feature_spec = args.arch in {"resnet18", "resnet50", "resnet101", "densenet121"}
    uses_scsf_features = method_name in {"scsf", "ds_scsf", "ds_scsf_v2", "dualaug", "dualaug_correctness", "mg_scsf"} and supports_scsf_feature_spec
    backbone = build_backbone(
        args.arch,
        model_classes,
        input_size=input_size,
        pretrained=args.pretrained,
        multi_layer_features=uses_scsf_features and scsf_feature_spec == "mid+late+logits",
        feature_spec=scsf_feature_spec if uses_scsf_features else None,
    )

    if method_name == "sr":
        return SoftmaxResponseMethod(backbone, num_classes)
    if method_name == "dg":
        args.reward = resolve_dg_reward(args.reward, num_classes)
        return DeepGamblersMethod(backbone, num_classes)
    if method_name == "sat":
        return SATMethod(backbone, num_classes, train_size, args.sat_momentum)
    if method_name == "selectivenet":
        return SelectiveNetMethod(backbone, num_classes, args.target_coverage, args.selectivenet_alpha)
    if method_name == "scsf":
        return SCSFMethod(
            backbone,
            num_classes,
            hidden_dim=args.hidden_dim,
            use_logits="logits" in scsf_feature_spec.split("+"),
            scorer=getattr(args, "scsf_scorer", "meta"),
            sr_alpha=getattr(args, "scsf_sr_alpha", 0.5),
            calibrator_arch=getattr(args, "calibrator_arch", "standard"),
            train_size=train_size,
        )
    if method_name in {"dualaug", "dualaug_correctness"}:
        return DualAugMethod(
            backbone,
            num_classes,
            hidden_dim=args.hidden_dim,
            calibrator_arch=getattr(args, "calibrator_arch", "standard"),
            target="correct_agree" if method_name == "dualaug_correctness" else getattr(args, "dualaug_target", "agreement"),
            detach_features=getattr(args, "aux_detach", False),
        )
    if method_name == "verify_global":
        return VerifyGlobalMethod(
            backbone,
            num_classes,
            embed_dim=args.verify_embed_dim,
            hidden_dim=args.verify_hidden_dim,
            loss=args.verify_loss,
            detach_features=args.aux_detach,
        )
    if method_name == "verify_spatial":
        return VerifySpatialMethod(
            backbone,
            num_classes,
            pool=args.verify_pool,
            topk_fraction=args.verify_topk_fraction,
            loss=args.verify_loss,
            detach_features=args.aux_detach,
        )
    ds_init_seed = int(args.seed) + 104729
    if method_name == "mg_scsf":
        return MGSCSFMethod(
            backbone,
            num_classes,
            hidden_dim=args.hidden_dim,
            use_logits="logits" in scsf_feature_spec.split("+"),
            calibrator_arch=getattr(args, "calibrator_arch", "standard"),
            train_size=train_size,
            loss=args.mg_loss,
            margin=args.mg_margin,
            ce_scale=args.mg_ce_scale,
            detach=args.ds_detach,
            init_seed=ds_init_seed,
        )
    if method_name == "mp_ds":
        return MPDSMethod(
            backbone,
            num_classes,
            prototypes=args.mp_prototypes,
            dim=args.mp_dim,
            temperature=args.mp_temperature,
            detach=args.ds_detach,
            init_seed=ds_init_seed,
            site=args.ds_site,
        )
    if method_name == "se_ds":
        return SEDSMethod(
            backbone,
            num_classes,
            pool=args.se_pool,
            topk_fraction=args.se_topk_fraction,
            detach=args.ds_detach,
            init_seed=ds_init_seed,
            site=args.ds_site,
        )
    if method_name == "ds_scsf":
        parts = [part.strip() for part in scsf_feature_spec.split("+") if part.strip()]
        layers = tuple(part for part in ("early", "mid", "late") if part in parts)
        if not layers:
            raise ValueError("DS-SCSF requires --scsf-feature-spec to include at least one of early/mid/late.")
        return DSSCSFMethod(
            backbone,
            num_classes,
            feature_dims={layer: backbone._stage_dims[layer] for layer in layers},
            hidden_dim=args.hidden_dim,
            layers=layers,
        )
    if method_name == "ds_scsf_v2":
        parts = [part.strip() for part in scsf_feature_spec.split("+") if part.strip() and part != "logits"]
        layers = tuple(part for part in ("early", "mid", "late") if part in parts)
        if not layers:
            layers = ("mid", "late")
        return DSSCSFv2Method(
            backbone,
            num_classes,
            feature_dims={layer: backbone._stage_dims[layer] for layer in layers},
            hidden_dim=args.hidden_dim,
            layers=layers,
            train_size=train_size,
            kd_temperature=float(getattr(args, "ds_kd_temperature", 2.0)),
            kd_weight=float(getattr(args, "ds_kd_weight", 0.5)),
        )
    if method_name == "ccl_sc":
        key_backbone = build_backbone(args.arch, model_classes, input_size=input_size, pretrained=args.pretrained)
        return CCLSCMethod(
            backbone,
            key_backbone,
            num_classes,
            queue_size=args.ccl_queue_size,
            momentum=args.ccl_momentum,
            temperature=args.ccl_temperature,
            base_temperature=args.ccl_base_temperature,
            variant=args.ccl_variant,
        )
    if method_name == "residual_head":
        return ResidualHeadMethod(
            backbone, num_classes,
            hidden_dim=args.hidden_dim,
            agree_weight=getattr(args, 'agree_weight', 1.0),
            min_agree_weight=getattr(args, 'min_agree_weight', 1e-4),
        )
    if method_name == "dp_head":
        return DPHeadMethod(
            backbone, num_classes,
            hidden_dim=args.hidden_dim,
            agree_weight=getattr(args, 'agree_weight', 1.0),
            min_agree_weight=getattr(args, 'min_agree_weight', 1e-4),
            proj_damping=getattr(args, 'dp_proj_damping', 1e-2),
        )
    if method_name == "spatial_head":
        return SpatialHeadMethod(
            backbone, num_classes,
            d_head=getattr(args, 'spatial_d_head', 128),
            hidden_dim=args.hidden_dim,
            agree_weight=getattr(args, 'agree_weight', 1.0),
            min_agree_weight=getattr(args, 'min_agree_weight', 1e-4),
        )
    raise ValueError(f"Unknown method: {method_name}")
