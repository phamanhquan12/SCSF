#!/usr/bin/env python3
"""Second-wave CIFAR-100 variants built on the cbr_micro / CCL-SC leftover analysis.

cbr_micro won because it is a coverage-conditional soft risk (1-p_y on the accepted
set), not because of confusion, duals, pairwise rank, or SR-weighted CSC. Those
extras either late-collapsed or made leftover errors more overconfident.

This batch keeps the SCSF head and (when enabled) the micro term, then adds only
mechanisms that follow from that diagnosis:

  micro_hi      same micro risk, coverages 80/90/95 (the CCL-SC region)
  micro_ace     micro + accepted-set CE (sharper extra CE on the kept set)
  acccon        SCSF + acceptance-weighted SupCon (CCL-SC geometry, RC weights)
  acccon_bound  acccon with query weight w + β·4w(1-w) (core + decision boundary)
  acccon_floor  acccon with query floor q = α + (1-α)w (old shoulder + new core)
  acccon_tail   acccon + soft accepted-set 0/1 risk (thresholds detached)
  acccon_ds     acccon + DS-SCSF score (per-depth heads, fused s)
  dss_acccon    DSN aux CE (live) + SCSF + detached depth disagreement + qw acccon
  rank_acccon   qw acccon + hard-mined pairwise ranking on high-confidence errors
  msp_acccon    qw acccon + detached MSP appended to SCSF head (extra_dim=1)
  micro_acccon  micro + acceptance-weighted SupCon
  micro_leftcon micro + leftover-attract SupCon (pull high-score 1-p_y toward
                accepted-likely-correct same-class features)

Dynamics variants (recipe §§7–15), each isolated:

  temporal_target  confidence BCE on rolling correctness (K=20)
  temporal_hybrid  0.5 hard + 0.5 temporal
  forget_target    temporal × exp(-γ N_forget)
  margin_target    sigmoid(mean margin / T_m)
  carto_acccon     ACCon with ambiguity-boosted queries / stable keys
  el2n_acccon      ACCon with early-EL2N query/key prior
  depth_diag       SCSF + detached depth probes (analysis only)
  depth_score      SCSF + append cross-depth disagreement to the selector
  temporal_depth   temporal target + disagreement-augmented selector

Official checkpoint is last.pth. Confusion, duals, rank, and SR weights stay off.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dyn_scsf import (
    TrainingDynamicsTracker,
    apply_cartography_weights,
    apply_dyn_config,
    apply_el2n_weights,
    build_train_eval_loader,
    depth_disagreement,
    evaluate_selective,
    jsonable,
    make_confidence_target,
    make_probes,
    make_query_weight,
    save_dynamics_outputs,
    unpack_batch,
    variant_defaults,
    wrap_train_loader_ids,
)
from train_cbr_scsf import (
    EPS,
    RawConfidenceHead,
    implicit_soft_threshold,
    parse_coverages,
    ramp_weight,
    seed_everything,
    soft_coverage_masks,
)
from train_r3_scsf import evaluate
from train_scsf import VGG16BN_FeatureExtractor, get_dataset, get_meta_weight_decay
from train_search_scsf import FeatureQueue, ProjectionHead


NEXT_VARIANTS = (
    "micro_hi",
    "micro_ace",
    "acccon",
    "acccon_bound",
    "acccon_floor",
    "acccon_tail",
    "acccon_ds",
    "dss_acccon",
    "rank_acccon",
    "msp_acccon",
    "micro_acccon",
    "micro_leftcon",
)
DYN_VARIANTS = tuple(variant_defaults())
ACCON_SWEEP_VARIANTS = (
    "acccon_floor",
    "acccon_tail",
    "acccon_bound",
    "acccon_ds",
    "dss_acccon",
    "rank_acccon",
    "msp_acccon",
)


def uses_official_10k(variant):
    return (
        variant == "acccon"
        or variant in DYN_VARIANTS
        or variant in ACCON_SWEEP_VARIANTS
    )


class MiniClassifier(nn.Module):
    """GoogLeNet-style auxiliary classifier on a spatial pool map."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features):
        pooled = self.avgpool(features).flatten(1)
        return self.fc(pooled)


class DepthScoreHead(nn.Module):
    """Unbounded SCSF-style score from one depth's features + stopgrad logits."""

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feat_dim + num_classes, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features, logits):
        return self.network(torch.cat([features, logits.detach()], dim=1)).squeeze(1)


class DeeplySupervisedScore(nn.Module):
    """DS-SCSF selector: per-depth scores fused with learnable β.

    Mini-classifiers match train_ds_scsf (2×2 GAP). Score heads take the native
    flattened pool maps (CIFAR: 256×4×4, 512×2×2, 512×1×1). Scores stay unbounded.
    s3/s4 read aux logits; s5 reads the final classifier logits.
    """

    is_ds_score = True

    def __init__(
        self,
        num_classes,
        channels=(256, 512, 512),
        feat_dims=(256 * 4 * 4, 512 * 2 * 2, 512 * 1 * 1),
    ):
        super().__init__()
        self.mini3 = MiniClassifier(channels[0], num_classes)
        self.mini4 = MiniClassifier(channels[1], num_classes)
        self.mini5 = MiniClassifier(channels[2], num_classes)
        self.head3 = DepthScoreHead(feat_dims[0], num_classes)
        self.head4 = DepthScoreHead(feat_dims[1], num_classes)
        self.head5 = DepthScoreHead(feat_dims[2], num_classes)
        self.beta = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

    def fusion_weights(self):
        weights = self.beta.abs()
        return weights / weights.sum().clamp_min(EPS)

    def forward(self, spatial3, spatial4, spatial5, logits):
        logits3 = self.mini3(spatial3)
        logits4 = self.mini4(spatial4)
        logits5 = self.mini5(spatial5)
        feat3 = spatial3.flatten(1)
        feat4 = spatial4.flatten(1)
        feat5 = spatial5.flatten(1)
        score3 = self.head3(feat3, logits3)
        score4 = self.head4(feat4, logits4)
        score5 = self.head5(feat5, logits)
        weights = self.fusion_weights()
        fused = weights[0] * score3 + weights[1] * score4 + weights[2] * score5
        return fused, {
            "logits3": logits3,
            "logits4": logits4,
            "logits5": logits5,
            "score3": score3,
            "score4": score4,
            "score5": score5,
            "weights": weights,
        }


class DeepSupervisionBranch(nn.Module):
    """Live DSN mini-classifiers at pool3/4/5. Does not replace the SCSF score."""

    is_ds_branch = True

    def __init__(self, num_classes, channels=(256, 512, 512)):
        super().__init__()
        self.mini3 = MiniClassifier(channels[0], num_classes)
        self.mini4 = MiniClassifier(channels[1], num_classes)
        self.mini5 = MiniClassifier(channels[2], num_classes)

    def forward(self, spatial3, spatial4, spatial5):
        logits3 = self.mini3(spatial3)
        logits4 = self.mini4(spatial4)
        logits5 = self.mini5(spatial5)
        return {
            "logits3": logits3,
            "logits4": logits4,
            "logits5": logits5,
        }


def hard_pair_rank_loss(
    scores,
    correctness,
    margin=0.2,
    k_wrong=16,
    k_correct=4,
):
    """Hard-mined pairwise ranking: push top-K confident errors below top corrects.

    Recipe §5: for each high-confidence wrong j and correct i,
    L = softplus(s_j - s_i + m). Does not replace correctness BCE.
    """

    wrong = torch.where(~correctness)[0]
    correct = torch.where(correctness)[0]
    if wrong.numel() == 0 or correct.numel() == 0:
        return scores.sum() * 0.0

    k_w = min(int(k_wrong), int(wrong.numel()))
    k_c = min(int(k_correct), int(correct.numel()))
    wrong_top = wrong[torch.topk(scores[wrong], k=k_w, largest=True).indices]
    correct_top = correct[torch.topk(scores[correct], k=k_c, largest=True).indices]
    # Broadcast all hard pairs: (k_w, k_c)
    pair = (
        scores[wrong_top].unsqueeze(1)
        - scores[correct_top].unsqueeze(0)
        + float(margin)
    )
    return F.softplus(pair).mean()


def soft_selective_risk_loss(
    scores,
    wrong_mask,
    coverages=(0.70, 0.80, 0.90, 0.95),
    temperature=0.2,
    iterations=60,
):
    """Soft accepted 0/1 risk. Thresholds from detached scores; masks stay live."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    detached = scores.detach()
    errors = wrong_mask.detach().to(dtype=scores.dtype)
    risks = []
    for coverage in coverages:
        threshold = implicit_soft_threshold(
            detached, coverage, temperature, iterations
        )
        accept = torch.sigmoid((scores - threshold) / temperature)
        risks.append(accept.mul(errors).sum() / accept.sum().clamp_min(EPS))
    return torch.stack(risks).mean()


def soft_micro_risk(logits, masks, targets):
    """Mean soft selective risk using 1-p_y on nested coverage masks."""

    true_probability = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    errors = 1.0 - true_probability
    micro = (masks * errors.unsqueeze(1)).sum(dim=0) / masks.sum(dim=0).clamp_min(EPS)
    return micro.mean(), errors


def accepted_cross_entropy(logits, masks, targets):
    """Mean per-sample CE on the same soft accepted sets as micro risk."""

    per_sample = F.cross_entropy(logits, targets, reduction="none")
    accepted = (masks * per_sample.unsqueeze(1)).sum(dim=0) / masks.sum(dim=0).clamp_min(
        EPS
    )
    return accepted.mean()


def core_boundary_query_weight(accept, beta):
    """Query weight: trusted core plus a bump on the acceptance boundary.

    b = 4w(1-w) is 0 at w∈{0,1} and 1 at w=0.5. Keys stay plain acceptance.
    """

    if beta < 0.0:
        raise ValueError("boundary beta must be nonnegative")
    boundary = 4.0 * accept * (1.0 - accept)
    return accept + float(beta) * boundary


def rc_contrastive_weights(
    masks, logits, targets, mode, boundary_beta=1.0, query_floor=0.2
):
    """Detached RC sample weights. Never SR: acceptance or leftover mass."""

    true_probability = (
        F.softmax(logits.detach(), dim=1).gather(1, targets[:, None]).squeeze(1)
    )
    accept = masks.mean(dim=1).detach()
    leftover = (masks.detach() * (1.0 - true_probability).unsqueeze(1)).mean(dim=1)
    if mode == "accept":
        return accept, accept
    if mode == "accept_bound":
        return core_boundary_query_weight(accept, boundary_beta), accept
    if mode == "accept_floor":
        return make_query_weight(accept, "floor", floor=query_floor), accept
    if mode == "leftover":
        return leftover, accept * true_probability
    raise ValueError(f"unknown contrastive mode: {mode}")


def weighted_supcon(query, targets, query_weight, key_weight, queue, temperature):
    """Acceptance-weighted SupCon with query and key weights that do not cancel.

    Per query, positives are weighted only by key acceptance. The batch mean is
    then weighted by query acceptance, so a rejected anchor does not contribute
    as much as an accepted one.
    """

    if query.size(0) == 0:
        return query.sum() * 0.0
    if queue.filled > 0:
        keys = torch.cat([query, queue.z[: queue.filled]], dim=0)
        key_targets = torch.cat([targets, queue.y[: queue.filled]], dim=0)
        stored_key_weight = torch.cat(
            [key_weight, queue.c[: queue.filled]], dim=0
        )
    else:
        keys = query
        key_targets = targets
        stored_key_weight = key_weight

    logits = query @ keys.t() / temperature
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    eye = torch.eye(query.size(0), dtype=torch.bool, device=query.device)
    self_mask[:, : query.size(0)] = eye
    positives = key_targets.unsqueeze(0).eq(targets.unsqueeze(1)) & ~self_mask
    valid = positives.any(dim=1)
    if not valid.any():
        return query.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    key_w = stored_key_weight.clamp_min(0.0)
    positive_key_w = key_w.unsqueeze(0) * positives.to(key_w.dtype)
    safe_log_prob = log_prob.masked_fill(~positives, 0.0)
    per_query = -(positive_key_w * safe_log_prob).sum(dim=1) / (
        positive_key_w.sum(dim=1).clamp_min(EPS)
    )
    query_w = query_weight.clamp_min(0.0)
    return (query_w[valid] * per_query[valid]).sum() / query_w[valid].sum().clamp_min(
        EPS
    )


def apply_variant_defaults(args):
    """Lock unused extra-loss weights so a variant cannot silently mix recipes."""

    _ensure_dyn_fields(args)
    if args.variant in variant_defaults():
        return apply_dyn_config(args, variant_defaults()[args.variant])
    if args.variant == "micro_hi":
        args.coverages = parse_coverages("0.80,0.90,0.95")
        args.acceptce_weight = 0.0
        args.con_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "accept"
    elif args.variant == "micro_ace":
        args.con_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "accept"
    elif args.variant == "acccon":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = False
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
    elif args.variant == "acccon_bound":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = False
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept_bound"
        args.coverages = parse_coverages("0.80,0.90,0.95")
    elif args.variant == "acccon_floor":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = False
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept_floor"
        args.coverages = parse_coverages("0.80,0.90,0.95")
    elif args.variant == "acccon_tail":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.rank_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = False
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
        if args.tail_weight <= 0.0:
            args.tail_weight = 0.10
        args.tail_coverages = parse_coverages("0.70,0.80,0.90,0.95")
    elif args.variant == "acccon_ds":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
        args.use_ds_score = True
        args.use_ds_branch = False
        args.append_msp = False
        if args.aux_ce_weight <= 0.0:
            args.aux_ce_weight = 0.3
        if args.aux_bce_weight <= 0.0:
            args.aux_bce_weight = 0.3
    elif args.variant == "dss_acccon":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
        args.use_ds_score = False
        args.use_ds_branch = True
        args.append_msp = False
        args.append_disagreement = True
        args.extra_dim = 1
        args.aux_bce_weight = 0.0
        if args.aux_ce_weight <= 0.0:
            args.aux_ce_weight = 0.3
    elif args.variant == "rank_acccon":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = False
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
        if args.rank_weight <= 0.0:
            args.rank_weight = 0.05
        if args.rank_margin < 0.0:
            args.rank_margin = 0.2
    elif args.variant == "msp_acccon":
        args.micro_weight = 0.0
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.use_ds_score = False
        args.use_ds_branch = False
        args.append_msp = True
        args.extra_dim = 1
        args.aux_ce_weight = 0.0
        args.aux_bce_weight = 0.0
        args.con_mode = "accept"
        args.coverages = parse_coverages("0.80,0.90,0.95")
    elif args.variant == "micro_acccon":
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "accept"
    elif args.variant == "micro_leftcon":
        args.acceptce_weight = 0.0
        args.tail_weight = 0.0
        args.rank_weight = 0.0
        args.con_mode = "leftover"
    else:
        raise ValueError(f"unknown variant: {args.variant}")
    return args


def _ensure_dyn_fields(args):
    defaults = {
        "confidence_target": "hard",
        "record_dynamics": False,
        "record_el2n": False,
        "use_depth_probes": False,
        "append_disagreement": False,
        "query_weight_mode": "none",
        "target_alpha": 0.0,
        "forget_gamma": 0.5,
        "margin_temperature": 1.0,
        "ambiguity_beta": 0.10,
        "stability_gamma": 2.0,
        "el2n_beta": 0.10,
        "extra_dim": 0,
        "dyn_window": 20,
        "dyn_start_epoch": 101,
        "el2n_start": 10,
        "el2n_end": 20,
        "dynamics_mode": "eval",
        "tail_weight": 0.0,
        "tail_coverages": (0.70, 0.80, 0.90, 0.95),
        "rank_weight": 0.0,
        "rank_margin": 0.2,
        "rank_k_wrong": 16,
        "rank_k_correct": 4,
        "query_floor": 0.20,
        "use_ds_score": False,
        "use_ds_branch": False,
        "append_msp": False,
        "aux_ce_weight": 0.0,
        "aux_bce_weight": 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def needs_sample_ids(args):
    return bool(args.record_dynamics or args.record_el2n or args.use_depth_probes)


def should_record_dynamics(epoch, args):
    return args.record_dynamics and epoch >= args.dyn_start_epoch


def should_record_el2n(epoch, args):
    return args.record_el2n and args.el2n_start <= epoch <= args.el2n_end


def train_epoch(
    backbone,
    confidence_head,
    projector,
    queue,
    loader,
    backbone_optimizer,
    head_optimizer,
    device,
    epoch,
    args,
    tracker=None,
    probes=None,
    ds_branch=None,
):
    backbone.train()
    confidence_head.train()
    if projector is not None:
        projector.train()
    if probes is not None:
        probes.train()
    if ds_branch is not None:
        ds_branch.train()

    extra_ramp = ramp_weight(epoch, args.pretrain, args.ramp_epochs)
    use_ds = bool(getattr(args, "use_ds_score", False))
    use_ds_branch = bool(getattr(args, "use_ds_branch", False)) and ds_branch is not None
    lambda_t = (
        0.0
        if epoch <= args.pretrain
        else get_meta_weight_decay(
            epoch,
            args.pretrain,
            args.epochs,
            args.meta_weight,
            args.min_meta_weight,
        )
    )
    use_masks = extra_ramp > 0.0 and (
        args.micro_weight > 0.0
        or args.acceptce_weight > 0.0
        or args.con_weight > 0.0
    )
    use_contrast = extra_ramp > 0.0 and args.con_weight > 0.0 and projector is not None
    use_tail = extra_ramp > 0.0 and args.tail_weight > 0.0
    use_rank = extra_ramp > 0.0 and args.rank_weight > 0.0
    use_probes = probes is not None
    totals = defaultdict(float)
    sample_count = 0

    for batch_index, batch in enumerate(loader):
        if args.limit_train_batches and batch_index >= args.limit_train_batches:
            break
        inputs, targets, ids = unpack_batch(batch, device)
        ds_aux = None
        ds_branch_out = None
        extra = None
        probe_loss = None
        disagreement = None
        if use_ds:
            logits, spatial3, spatial4, spatial5 = backbone(
                inputs, return_features=True, return_spatial=True
            )
            pool5 = F.adaptive_avg_pool2d(spatial5, 1).flatten(1)
            pool4 = F.adaptive_avg_pool2d(spatial4, 2).flatten(1)
            raw_scores, ds_aux = confidence_head(spatial3, spatial4, spatial5, logits)
            probe_loss = logits.sum() * 0.0
            disagreement = logits.sum() * 0.0
        elif use_ds_branch:
            logits, spatial3, spatial4, spatial5 = backbone(
                inputs, return_features=True, return_spatial=True
            )
            pool5 = F.adaptive_avg_pool2d(spatial5, 1).flatten(1)
            pool4 = F.adaptive_avg_pool2d(spatial4, 2).flatten(1)
            ds_branch_out = ds_branch(spatial3, spatial4, spatial5)
            disagreement = depth_disagreement(
                ds_branch_out["logits3"],
                ds_branch_out["logits4"],
                ds_branch_out["logits5"],
                logits,
            )
            if args.append_disagreement and epoch > args.pretrain:
                extra = disagreement.detach().unsqueeze(1)
            raw_scores = confidence_head(pool4, pool5, logits, extra=extra)
            probe_loss = logits.sum() * 0.0
        elif use_probes:
            logits, pool4, pool5, pool3 = backbone(
                inputs, return_features=True, return_pool3=True
            )
            probe_loss = logits.sum() * 0.0
            disagreement = logits.sum() * 0.0
            logits3 = probes["pool3"](pool3.detach())
            logits4 = probes["pool4"](pool4.detach())
            logits5 = probes["pool5"](pool5.detach())
            probe_loss = (
                F.cross_entropy(logits3, targets)
                + F.cross_entropy(logits4, targets)
                + F.cross_entropy(logits5, targets)
            ) / 3.0
            disagreement = depth_disagreement(logits3, logits4, logits5, logits)
            if args.append_disagreement and epoch > args.pretrain:
                extra = disagreement.detach().unsqueeze(1)
            raw_scores = confidence_head(pool4, pool5, logits, extra=extra)
        else:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            probe_loss = logits.sum() * 0.0
            disagreement = logits.sum() * 0.0
            if getattr(args, "append_msp", False) and epoch > args.pretrain:
                msp = F.softmax(logits.detach(), dim=1).max(dim=1).values
                extra = msp.unsqueeze(1)
            raw_scores = confidence_head(pool4, pool5, logits, extra=extra)

        ce = F.cross_entropy(logits, targets)
        with torch.no_grad():
            correctness = logits.argmax(dim=1).eq(targets)
        target = make_confidence_target(
            correctness,
            ids,
            tracker,
            args.confidence_target,
            alpha=args.target_alpha,
            forget_gamma=args.forget_gamma,
            margin_temperature=args.margin_temperature,
        )
        correctness_bce = F.binary_cross_entropy_with_logits(
            raw_scores, target.to(raw_scores.dtype)
        )
        aux_ce = logits.sum() * 0.0
        aux_bce = logits.sum() * 0.0
        if ds_aux is not None:
            aux_ce = (
                F.cross_entropy(ds_aux["logits3"], targets)
                + F.cross_entropy(ds_aux["logits4"], targets)
                + F.cross_entropy(ds_aux["logits5"], targets)
            ) / 3.0
            aux_bce = (
                F.binary_cross_entropy_with_logits(
                    ds_aux["score3"], target.to(raw_scores.dtype)
                )
                + F.binary_cross_entropy_with_logits(
                    ds_aux["score4"], target.to(raw_scores.dtype)
                )
                + F.binary_cross_entropy_with_logits(
                    ds_aux["score5"], target.to(raw_scores.dtype)
                )
            ) / 3.0
        elif ds_branch_out is not None:
            aux_ce = (
                F.cross_entropy(ds_branch_out["logits3"], targets)
                + F.cross_entropy(ds_branch_out["logits4"], targets)
                + F.cross_entropy(ds_branch_out["logits5"], targets)
            ) / 3.0

        micro = logits.sum() * 0.0
        accept_ce = logits.sum() * 0.0
        contrast = logits.sum() * 0.0
        tail = logits.sum() * 0.0
        rank = logits.sum() * 0.0
        if use_tail:
            tail = soft_selective_risk_loss(
                raw_scores,
                ~correctness,
                args.tail_coverages,
                args.soft_temperature,
                args.threshold_iterations,
            )
        if use_rank:
            rank = hard_pair_rank_loss(
                raw_scores,
                correctness,
                margin=args.rank_margin,
                k_wrong=args.rank_k_wrong,
                k_correct=args.rank_k_correct,
            )
        if use_masks:
            score_for_masks = (
                raw_scores
                if args.micro_weight > 0.0 or args.acceptce_weight > 0.0
                else raw_scores.detach()
            )
            masks, _ = soft_coverage_masks(
                score_for_masks,
                args.coverages,
                args.soft_temperature,
                args.threshold_iterations,
            )
            if args.micro_weight > 0.0:
                micro, _ = soft_micro_risk(logits, masks, targets)
            if args.acceptce_weight > 0.0:
                accept_ce = accepted_cross_entropy(logits, masks, targets)
            if use_contrast:
                query = projector(pool5)
                query_weight, key_weight = rc_contrastive_weights(
                    masks,
                    logits,
                    targets,
                    args.con_mode,
                    boundary_beta=args.boundary_beta,
                    query_floor=args.query_floor,
                )
                if args.query_weight_mode == "cartography":
                    query_weight, key_weight = apply_cartography_weights(
                        query_weight,
                        tracker,
                        ids,
                        args.ambiguity_beta,
                        args.stability_gamma,
                        device,
                    )
                elif args.query_weight_mode == "el2n":
                    query_weight, key_weight = apply_el2n_weights(
                        query_weight, tracker, ids, args.el2n_beta, device
                    )
                contrast = weighted_supcon(
                    query,
                    targets,
                    query_weight,
                    key_weight,
                    queue,
                    args.con_temperature,
                )
                queue.enqueue(query.detach(), targets, key_weight.detach())

        loss = (
            ce
            + lambda_t * correctness_bce
            + extra_ramp
            * (
                args.micro_weight * micro
                + args.acceptce_weight * accept_ce
                + args.con_weight * contrast
                + args.tail_weight * tail
                + args.rank_weight * rank
            )
            + args.aux_ce_weight * aux_ce
            + lambda_t * args.aux_bce_weight * aux_bce
            + probe_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite next-variant loss")

        backbone_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(backbone.parameters()) + list(confidence_head.parameters())
        if projector is not None:
            params += list(projector.parameters())
        if probes is not None:
            params += list(probes.parameters())
        if ds_branch is not None:
            params += list(ds_branch.parameters())
        grad_norm = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        backbone_optimizer.step()
        head_optimizer.step()

        if (
            tracker is not None
            and args.dynamics_mode == "train"
            and should_record_dynamics(epoch, args)
        ):
            tracker.observe_batch(ids, logits, targets, raw_scores.detach())
        if (
            tracker is not None
            and args.dynamics_mode == "train"
            and should_record_el2n(epoch, args)
        ):
            tracker.observe_el2n(ids, logits, targets)

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += ce.item() * batch_size
        totals["bce"] += correctness_bce.item() * batch_size
        totals["micro"] += micro.item() * batch_size
        totals["accept_ce"] += accept_ce.item() * batch_size
        totals["contrast"] += contrast.item() * batch_size
        totals["tail"] += tail.item() * batch_size
        totals["rank"] += rank.item() * batch_size
        totals["aux_ce"] += aux_ce.item() * batch_size
        totals["aux_bce"] += aux_bce.item() * batch_size
        if ds_aux is not None:
            totals["beta3"] += float(ds_aux["weights"][0].detach()) * batch_size
            totals["beta4"] += float(ds_aux["weights"][1].detach()) * batch_size
            totals["beta5"] += float(ds_aux["weights"][2].detach()) * batch_size
        totals["probe"] += float(probe_loss.detach()) * batch_size
        totals["disagreement"] += float(disagreement.detach().mean()) * batch_size
        totals["correct"] += correctness.sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("training loader produced no batches")
    return {
        "loss": totals["loss"] / sample_count,
        "ce": totals["ce"] / sample_count,
        "bce": totals["bce"] / sample_count,
        "micro": totals["micro"] / sample_count,
        "accept_ce": totals["accept_ce"] / sample_count,
        "contrast": totals["contrast"] / sample_count,
        "tail": totals["tail"] / sample_count,
        "rank": totals["rank"] / sample_count,
        "aux_ce": totals["aux_ce"] / sample_count,
        "aux_bce": totals["aux_bce"] / sample_count,
        "beta3": totals["beta3"] / sample_count,
        "beta4": totals["beta4"] / sample_count,
        "beta5": totals["beta5"] / sample_count,
        "probe": totals["probe"] / sample_count,
        "disagreement": totals["disagreement"] / sample_count,
        "accuracy": 100.0 * totals["correct"] / sample_count,
        "lambda_t": lambda_t,
        "ramp": extra_ramp,
    }


@torch.no_grad()
def record_eval_pass(
    backbone,
    confidence_head,
    loader,
    device,
    tracker,
    probes,
    args,
    epoch,
    ds_branch=None,
):
    backbone.eval()
    confidence_head.eval()
    if probes is not None:
        probes.eval()
    if ds_branch is not None:
        ds_branch.eval()
    record_dyn = should_record_dynamics(epoch, args)
    record_el2n = should_record_el2n(epoch, args)
    if tracker is None or not (record_dyn or record_el2n):
        return
    for batch in loader:
        inputs, targets, ids = unpack_batch(batch, device)
        extra = None
        if ds_branch is not None and getattr(args, "use_ds_branch", False):
            logits, spatial3, spatial4, spatial5 = backbone(
                inputs, return_features=True, return_spatial=True
            )
            pool5 = F.adaptive_avg_pool2d(spatial5, 1).flatten(1)
            pool4 = F.adaptive_avg_pool2d(spatial4, 2).flatten(1)
            branch_out = ds_branch(spatial3, spatial4, spatial5)
            if args.append_disagreement and epoch > args.pretrain:
                d = depth_disagreement(
                    branch_out["logits3"],
                    branch_out["logits4"],
                    branch_out["logits5"],
                    logits,
                )
                extra = d.unsqueeze(1)
        elif probes is not None:
            logits, pool4, pool5, pool3 = backbone(
                inputs, return_features=True, return_pool3=True
            )
            if args.append_disagreement and epoch > args.pretrain:
                d = depth_disagreement(
                    probes["pool3"](pool3),
                    probes["pool4"](pool4),
                    probes["pool5"](pool5),
                    logits,
                )
                extra = d.unsqueeze(1)
        else:
            logits, pool4, pool5 = backbone(inputs, return_features=True)
            if getattr(args, "append_msp", False) and epoch > args.pretrain:
                msp = F.softmax(logits, dim=1).max(dim=1).values
                extra = msp.unsqueeze(1)
        scores = confidence_head(pool4, pool5, logits, extra=extra)
        if record_dyn:
            tracker.observe_batch(ids, logits, targets, scores)
        if record_el2n:
            tracker.observe_el2n(ids, logits, targets)
    if record_dyn:
        tracker.commit_epoch()
    else:
        tracker.discard_epoch()
    if epoch == args.el2n_end and args.record_el2n:
        tracker.freeze_el2n()


def run_evaluation(backbone, confidence_head, loader, device, args, probes, ds_branch=None):
    if uses_official_10k(args.variant):
        return evaluate_selective(
            backbone,
            confidence_head,
            loader,
            device,
            probes=probes if args.use_depth_probes else None,
            extra_dim=args.extra_dim,
            use_ds_score=bool(getattr(args, "use_ds_score", False)),
            ds_branch=ds_branch if getattr(args, "use_ds_branch", False) else None,
            append_msp=bool(getattr(args, "append_msp", False)),
        )
    return evaluate(backbone, confidence_head, loader, device)


def parse_args():
    parser = argparse.ArgumentParser(description="SCSF next-wave CIFAR search")
    parser.add_argument("-d", "--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--variant",
        default="micro_acccon",
        choices=list(NEXT_VARIANTS) + list(DYN_VARIANTS),
    )
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--pretrain", default=100, type=int)
    parser.add_argument("--ramp-epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--meta-weight", default=1.0, type=float)
    parser.add_argument("--min-meta-weight", default=1e-4, type=float)
    parser.add_argument(
        "--coverages",
        default=parse_coverages("0.70,0.80,0.90,0.95"),
        type=parse_coverages,
    )
    parser.add_argument("--soft-temperature", default=0.2, type=float)
    parser.add_argument("--threshold-iterations", default=60, type=int)
    parser.add_argument("--micro-weight", default=0.5, type=float)
    parser.add_argument("--acceptce-weight", default=0.5, type=float)
    parser.add_argument("--con-weight", default=0.5, type=float)
    parser.add_argument(
        "--con-mode",
        default="accept",
        choices=["accept", "accept_bound", "accept_floor", "leftover"],
    )
    parser.add_argument(
        "--boundary-beta",
        default=1.0,
        type=float,
        help="For acccon_bound: q = w + β·4w(1-w). Keys stay w.",
    )
    parser.add_argument(
        "--query-floor",
        default=0.20,
        type=float,
        help="For acccon_floor: q = α + (1-α)w. Keys stay w.",
    )
    parser.add_argument(
        "--tail-weight",
        default=0.0,
        type=float,
        help="Weight for soft accepted-set 0/1 risk. acccon_tail defaults to 0.10.",
    )
    parser.add_argument(
        "--tail-coverages",
        default=parse_coverages("0.70,0.80,0.90,0.95"),
        type=parse_coverages,
        help="Coverages for the tail risk term only. Separate from acccon masks.",
    )
    parser.add_argument(
        "--rank-weight",
        default=0.0,
        type=float,
        help="Hard-mined pairwise ranking weight. rank_acccon defaults to 0.05.",
    )
    parser.add_argument(
        "--rank-margin",
        default=0.2,
        type=float,
        help="Margin m in softplus(s_wrong - s_correct + m).",
    )
    parser.add_argument(
        "--rank-k-wrong",
        default=16,
        type=int,
        help="Top-K high-confidence errors mined per batch.",
    )
    parser.add_argument(
        "--rank-k-correct",
        default=4,
        type=int,
        help="Top-K correct partners per mined error.",
    )
    parser.add_argument(
        "--aux-ce-weight",
        default=0.0,
        type=float,
        help="DSN/GoogLeNet auxiliary classifier CE. acccon_ds defaults to 0.3.",
    )
    parser.add_argument(
        "--aux-bce-weight",
        default=0.0,
        type=float,
        help="Per-depth score BCE companion. acccon_ds defaults to 0.3.",
    )
    parser.add_argument("--con-temperature", default=0.1, type=float)
    parser.add_argument("--queue-size", default=3000, type=int)
    parser.add_argument("--proj-dim", default=128, type=int)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="./save/next_scsf")
    parser.add_argument("--limit-train-batches", default=0, type=int)
    parser.add_argument("--dyn-window", default=20, type=int)
    parser.add_argument("--dyn-start-epoch", default=101, type=int)
    parser.add_argument(
        "--dynamics-mode",
        default="eval",
        choices=["eval", "train"],
        help="eval=deterministic train-set pass (Option B); train=aug pass (Option A)",
    )
    parser.add_argument("--el2n-start", default=10, type=int)
    parser.add_argument("--el2n-end", default=20, type=int)
    parser.add_argument("--ambiguity-beta", default=0.10, type=float)
    parser.add_argument("--stability-gamma", default=2.0, type=float)
    parser.add_argument("--el2n-beta", default=0.10, type=float)
    parser.add_argument("--forget-gamma", default=0.5, type=float)
    parser.add_argument("--margin-temperature", default=1.0, type=float)
    parser.add_argument("--target-alpha", default=0.0, type=float)
    args = parser.parse_args()
    if args.pretrain < 1 or args.pretrain >= args.epochs:
        parser.error("--pretrain must be at least 1 and less than --epochs")
    if args.ramp_epochs < 1:
        parser.error("--ramp-epochs must be positive")
    if args.soft_temperature <= 0 or args.con_temperature <= 0:
        parser.error("temperatures must be positive")
    if args.boundary_beta < 0.0:
        parser.error("--boundary-beta must be nonnegative")
    if not 0.0 <= args.query_floor < 1.0:
        parser.error("--query-floor must be in [0, 1)")
    if args.tail_weight < 0.0:
        parser.error("--tail-weight must be nonnegative")
    if args.rank_weight < 0.0:
        parser.error("--rank-weight must be nonnegative")
    if args.rank_k_wrong < 1 or args.rank_k_correct < 1:
        parser.error("--rank-k-wrong and --rank-k-correct must be positive")
    if args.aux_ce_weight < 0.0 or args.aux_bce_weight < 0.0:
        parser.error("aux weights must be nonnegative")
    if args.dyn_window < 1:
        parser.error("--dyn-window must be positive")
    apply_variant_defaults(args)
    return args


def write_train_log(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, test_loader_full, num_classes = get_dataset(
        args
    )
    n_train = len(train_loader.dataset)
    if needs_sample_ids(args):
        train_loader = wrap_train_loader_ids(train_loader)
    tracker = (
        TrainingDynamicsTracker(n_train, window=args.dyn_window)
        if args.record_dynamics or args.record_el2n
        else None
    )
    train_eval_loader = None
    if tracker is not None and args.dynamics_mode == "eval":
        train_eval_loader = build_train_eval_loader(
            args.dataset, args.batch_size, args.workers
        )

    extra_dim = int(getattr(args, "extra_dim", 0))
    backbone = VGG16BN_FeatureExtractor(num_classes=num_classes, input_size=32).to(
        device
    )
    if getattr(args, "use_ds_score", False):
        confidence_head = DeeplySupervisedScore(num_classes).to(device)
        ds_branch = None
    else:
        confidence_head = RawConfidenceHead(
            512 * 4, 512, num_classes, extra_dim=extra_dim
        ).to(device)
        ds_branch = (
            DeepSupervisionBranch(num_classes).to(device)
            if getattr(args, "use_ds_branch", False)
            else None
        )
    probes = make_probes(num_classes, device) if args.use_depth_probes else None
    use_contrast = args.con_weight > 0.0
    projector = (
        ProjectionHead(in_dim=512, out_dim=args.proj_dim).to(device)
        if use_contrast
        else None
    )
    queue = (
        FeatureQueue(args.proj_dim, args.queue_size, device) if use_contrast else None
    )

    backbone_optimizer = optim.SGD(
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    head_params = list(confidence_head.parameters())
    if projector is not None:
        head_params += list(projector.parameters())
    if probes is not None:
        head_params += list(probes.parameters())
    if ds_branch is not None:
        head_params += list(ds_branch.parameters())
    head_optimizer = optim.Adam(head_params, lr=args.meta_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        backbone_optimizer,
        milestones=list(range(25, args.epochs, 25)),
        gamma=0.5,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2)
    best_aurc = math.inf
    train_log_rows = []

    with open(history_path, "w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                backbone,
                confidence_head,
                projector,
                queue,
                train_loader,
                backbone_optimizer,
                head_optimizer,
                device,
                epoch,
                args,
                tracker=tracker,
                probes=probes,
                ds_branch=ds_branch,
            )
            if tracker is not None and args.dynamics_mode == "train":
                if should_record_dynamics(epoch, args):
                    tracker.commit_epoch()
                else:
                    tracker.discard_epoch()
                if epoch == args.el2n_end and args.record_el2n:
                    tracker.freeze_el2n()
            if train_eval_loader is not None:
                record_eval_pass(
                    backbone,
                    confidence_head,
                    train_eval_loader,
                    device,
                    tracker,
                    probes,
                    args,
                    epoch,
                    ds_branch=ds_branch,
                )
            scheduler.step()
            record = {"epoch": epoch, "train": train_metrics}
            summary = (
                f"Epoch {epoch:03d}/{args.epochs} {args.variant} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"micro={train_metrics['micro']:.4f} "
                f"con={train_metrics['contrast']:.4f} "
                f"tail={train_metrics['tail']:.4f} "
                f"ramp={train_metrics['ramp']:.3f}"
            )
            if getattr(args, "use_ds_score", False):
                summary += (
                    f" aux_ce={train_metrics['aux_ce']:.4f}"
                    f" β=({train_metrics['beta3']:.2f},"
                    f"{train_metrics['beta4']:.2f},"
                    f"{train_metrics['beta5']:.2f})"
                )
            elif getattr(args, "use_ds_branch", False):
                summary += (
                    f" aux_ce={train_metrics['aux_ce']:.4f}"
                    f" D={train_metrics['disagreement']:.3f}"
                )
            if epoch > args.pretrain:
                validation = run_evaluation(
                    backbone,
                    confidence_head,
                    val_loader,
                    device,
                    args,
                    probes,
                    ds_branch=ds_branch,
                )
                record["validation"] = validation
                summary += f" val_AURC={validation['aurc']:.6f}"
            print(summary, flush=True)
            history_file.write(json.dumps(jsonable(record)) + "\n")
            history_file.flush()
            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            if "validation" in record:
                row["val_aurc"] = record["validation"]["aurc"]
                row["val_accuracy"] = record["validation"]["accuracy"]
            train_log_rows.append(row)
            payload = {
                "epoch": epoch,
                "variant": args.variant,
                "backbone": backbone.state_dict(),
                "confidence_head": confidence_head.state_dict(),
                "calibrator": confidence_head.state_dict(),
                "projector": None if projector is None else projector.state_dict(),
                "probes": None if probes is None else probes.state_dict(),
                "ds_branch": None if ds_branch is None else ds_branch.state_dict(),
                "args": jsonable(vars(args)),
                "validation": record.get("validation"),
            }
            torch.save(payload, last_path)
            if epoch > args.pretrain and record["validation"]["aurc"] < best_aurc:
                best_aurc = record["validation"]["aurc"]
                torch.save(payload, best_path)

    write_train_log(os.path.join(args.output_dir, "train_log.csv"), train_log_rows)
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    confidence_head.load_state_dict(checkpoint["confidence_head"])
    if probes is not None and checkpoint.get("probes") is not None:
        probes.load_state_dict(checkpoint["probes"])
    if ds_branch is not None and checkpoint.get("ds_branch") is not None:
        ds_branch.load_state_dict(checkpoint["ds_branch"])
    validation = run_evaluation(
        backbone, confidence_head, val_loader, device, args, probes, ds_branch=ds_branch
    )
    if uses_official_10k(args.variant):
        test = run_evaluation(
            backbone,
            confidence_head,
            test_loader_full,
            device,
            args,
            probes,
            ds_branch=ds_branch,
        )
        test_8k = run_evaluation(
            backbone,
            confidence_head,
            test_loader,
            device,
            args,
            probes,
            ds_branch=ds_branch,
        )
        eval_test_set = "official_10k"
    else:
        test = run_evaluation(
            backbone,
            confidence_head,
            test_loader,
            device,
            args,
            probes,
            ds_branch=ds_branch,
        )
        test_8k = None
        eval_test_set = "split_8k"
    results = {
        "eval_checkpoint": "last",
        "eval_test_set": eval_test_set,
        "variant": args.variant,
        "last_epoch": checkpoint["epoch"],
        "best_val_aurc": best_aurc if best_aurc < math.inf else None,
        "validation": validation,
        "test": test,
        "args": jsonable(vars(args)),
    }
    if test_8k is not None:
        results["test_8k"] = test_8k
    if getattr(confidence_head, "is_ds_score", False):
        results["fusion_weights"] = (
            confidence_head.fusion_weights().detach().cpu().tolist()
        )
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(results), handle, indent=2)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(test), handle, indent=2)
    if tracker is not None:
        save_dynamics_outputs(
            args.output_dir,
            tracker,
            extra_json={"variant": args.variant, "test": jsonable(test)},
        )
    print(
        f"Evaluated last checkpoint (epoch {checkpoint['epoch']}) "
        f"variant={args.variant}; "
        f"{eval_test_set} accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
