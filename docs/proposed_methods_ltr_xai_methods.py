from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .models import FeatureModel, build_backbone


FAITHFULNESS_NOTES = {
    "sr": "Softmax Response ranks by max softmax probability; used as SR confidence in SAT-selective-cls/train.py eval_converage.",
    "dg": "Deep Gamblers follows NIPS2019DeepGamblers/main.py: C+1 output, softmax reservation neuron, loss=-log(p_y + reservation/reward).",
    "sat": "SAT follows SAT-selective-cls/loss.py SelfAdativeTraining: momentum target history plus a reservation probability 1-p_y.",
    "selectivenet": "SelectiveNet follows selectivenet/models/*_vgg_selectivenet.py: class head, sigmoid selection head, auxiliary head, coverage penalty lambda=32.",
    "ccl_sc": "CCL-SC follows CCL-SC/train_CCL_SC.py and CCL-SC/moco/CSC.py: CE/SAT backbone with supervised contrastive consistency after pretrain.",
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
        probs = F.softmax(logits_plus, dim=1)
        class_logits = logits_plus[:, : self.num_classes]
        confidence = 1.0 - probs[:, -1]
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
        selection = self.selector(features).squeeze(1)
        aux_logits = self.aux_classifier(features)
        packed = torch.cat([logits, selection.unsqueeze(1)], dim=1)
        return MethodBatchOutput(packed, logits, selection, aux_logits, features=features)

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
            confidence = meta_confidence
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


class CCLSCMethod(BackboneMethod):
    """CCL-SC adapter with momentum encoder and correct/error feature queues."""

    def __init__(
        self,
        backbone: FeatureModel,
        key_backbone: FeatureModel,
        num_classes: int,
        queue_size: int,
        momentum: float,
        temperature: float,
        base_temperature: float,
        require_full_queue: bool,
    ):
        super().__init__(backbone, num_classes)
        self.key_backbone = key_backbone
        self.key_backbone.load_state_dict(self.backbone.state_dict())
        for param in self.key_backbone.parameters():
            param.requires_grad = False

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.require_full_queue = require_full_queue

        feature_dim = backbone.feature_dim
        self.register_buffer("correct_queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
        self.register_buffer("error_queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
        self.register_buffer("correct_queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("error_queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("correct_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("error_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("correct_queue_full", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("error_queue_full", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for online_param, key_param in zip(self.backbone.parameters(), self.key_backbone.parameters()):
            key_param.data.mul_(self.momentum).add_(online_param.data, alpha=1.0 - self.momentum)

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

    def _queues_ready(self) -> bool:
        if self.require_full_queue:
            return bool(self.correct_queue_full.item()) and bool(self.error_queue_full.item())
        correct_features, _ = self._queue_view("correct")
        error_features, _ = self._queue_view("error")
        return correct_features.size(0) > 0 and error_features.size(0) > 0

    def _csc_loss(self, logits: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        correct_features, correct_labels = self._queue_view("correct")
        error_features, error_labels = self._queue_view("error")
        if correct_features.size(0) == 0 or error_features.size(0) == 0:
            return None

        q = F.normalize(features, dim=1)
        sr = F.softmax(logits, dim=1).max(dim=1).values.detach().clamp_min(1e-6)
        losses = []
        for i in range(targets.size(0)):
            target = targets[i]
            pos_mask = correct_labels.eq(target)
            if not bool(pos_mask.any()):
                continue
            neg_mask = error_labels.eq(target)
            positive_sim = q[i : i + 1] @ correct_features[pos_mask].t()
            if bool(neg_mask.any()):
                negative_sim = q[i : i + 1] @ error_features[neg_mask].t()
            else:
                negative_sim = q.new_empty(1, 0)

            positive_sim = positive_sim.squeeze(0)
            negative_sim = negative_sim.squeeze(0)
            pos_count = positive_sim.numel()
            for positive in positive_sim:
                contrast_logits = torch.cat([positive.view(1), negative_sim], dim=0) / self.temperature
                logsumexp = torch.logsumexp(contrast_logits, dim=0)
                losses.append((logsumexp - positive) * sr[i] / pos_count)

        if not losses:
            return None
        return (self.temperature / self.base_temperature) * torch.stack(losses).sum() / targets.size(0)

    def forward(self, x: torch.Tensor) -> MethodBatchOutput:
        logits, features = self._backbone_logits_features(x)
        confidence = F.softmax(logits, dim=1).max(dim=1).values
        key_logits = None
        key_features = None
        if self.training:
            self._momentum_update_key_encoder()
            with torch.no_grad():
                key_logits, key_features = self.key_backbone(x, return_features=True)
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
        if output.features is None or output.key_logits is None or output.key_features is None:
            return ce

        csc_loss = None
        if epoch > args.pretrain and args.ccl_weight > 0 and self._queues_ready():
            csc_loss = self._csc_loss(output.train_logits, output.features, targets)

        with torch.no_grad():
            key_features = F.normalize(output.key_features, dim=1)
            key_predictions = output.key_logits.argmax(dim=1)
            key_correct = key_predictions.eq(targets)
            self._enqueue(key_features[key_correct], key_predictions[key_correct], "correct")
            self._enqueue(key_features[~key_correct], key_predictions[~key_correct], "error")

        if csc_loss is None:
            return ce
        return ce + args.ccl_weight * csc_loss


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


def build_method(args, num_classes: int, input_size: int, train_size: int) -> ExperimentMethod:
    method_name = args.method
    model_classes = num_classes + 1 if method_name in {"dg", "sat"} else num_classes
    scsf_feature_spec = getattr(args, "scsf_feature_spec", "mid+late+logits")
    supports_scsf_feature_spec = args.arch in {"resnet18", "resnet50", "resnet101", "densenet121"}
    uses_scsf_features = method_name in {"scsf", "ds_scsf", "ds_scsf_v2"} and supports_scsf_feature_spec
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
            require_full_queue=args.ccl_require_full_queue,
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
