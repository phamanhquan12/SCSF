"""CCL-SC: Confidence-aware Contrastive Learning for Selective Classification.

Ported from the official ICML 2024 repository ``lamda-bbo/CCL-SC`` (MIT).
The algorithmic content is preserved exactly:

* Two class-conditioned queues — a **correct** queue of memory-encoder
  features whose (argmax) predicted label equals the target, and an **error**
  queue of memory-encoder features whose predicted label does not; each queue
  stores its prediction labels and uses the official cyclic ``_dequeue_and_enqueue``
  pointers.
* A frozen **momentum encoder** ``model_k`` (same architecture), updated once
  per epoch: a full parameter copy at ``epoch == pretrain``, then
  ``k <- m * k + (1 - m) * q`` (official ``momentum_update``).
* Confidence-aware InfoNCE: positives are same-class correct-queue entries,
  negatives are same-class error-queue entries, each positive weighted
  inversely by the official ``sr = max(logits)`` statistic; temperature ``T``
  is scaled by ``base_temperature = 0.1``.
* Queries are the online encoder's features (gradients flow); keys are the
  memory encoder's features (detached). The loss activates only after the
  queues are full and only after the pretrain transition.

Two recorded adaptations to the uploaded official code (see
:file:`external_sources.yaml`):
* the uploaded code only *fills* queues inside an ``epoch >= pretrain and
  full_k1 and full_k2`` gate whose initial flags are False, which can never
  activate the contrastive loss; we fill the queues from the first
  ``epoch >= pretrain`` batch and activate the loss once both are full, matching
  the paper's description;
* the official ``CSC.MoCo(dim=128, ...)`` is inconsistent with the 512-d
  features it is actually handed, so we set ``dim = backbone.final_dim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from .base import Method, MethodPrediction
from .scores import compute_scores


class CSCMoCo(nn.Module):
    """Class-conditioned MoCo queues with the official cyclic pointers."""

    def __init__(self, dim: int, K: int = 3000, m: float = 0.999, T: float = 0.1,
                 num_class: int = 10):
        super().__init__()
        self.base_temperature = 0.10
        self.K = int(K)
        self.K2 = int(K)
        self.m = float(m)
        self.T = float(T)

        # error queue
        self.register_buffer("queue", torch.randn(dim, self.K2))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("prediction_queue",
                             torch.randint(0, num_class, (self.K2,), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # correct queue
        self.register_buffer("correct_queue", torch.randn(dim, self.K))
        self.correct_queue = F.normalize(self.correct_queue, dim=0)
        self.register_buffer("correct_prediction_queue",
                             torch.randint(0, num_class, (self.K,), dtype=torch.long))
        self.register_buffer("correct_queue_ptr", torch.zeros(1, dtype=torch.long))
        # full flags persist across checkpoints
        self.register_buffer("full_k1", torch.zeros((), dtype=torch.bool))
        self.register_buffer("full_k2", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_error, k_correct, correct_predicts, error_predicts):
        """Official cyclic pointers (advance in *both* branches)."""
        full_k1, full_k2 = bool(self.full_k1), bool(self.full_k2)

        if k_error.numel():
            b_e = k_error.shape[0]
            ptr = int(self.queue_ptr)
            if ptr + b_e >= self.K2:
                self.queue[:, ptr:self.K2] = k_error[:self.K2 - ptr, :].T
                self.queue[:, :b_e - self.K2 + ptr] = k_error[self.K2 - ptr:, :].T
                self.prediction_queue[ptr:self.K2] = error_predicts[:self.K2 - ptr]
                self.prediction_queue[:b_e - self.K2 + ptr] = error_predicts[self.K2 - ptr:]
                full_k1 = True
            else:
                self.queue[:, ptr:ptr + b_e] = k_error[:b_e, :].T
                self.prediction_queue[ptr:ptr + b_e] = error_predicts[:b_e]
            self.queue_ptr[0] = (ptr + b_e) % self.K2

        if k_correct.numel():
            b_c = k_correct.shape[0]
            ptr2 = int(self.correct_queue_ptr)
            if ptr2 + b_c >= self.K2:
                self.correct_queue[:, ptr2:self.K2] = k_correct[:self.K2 - ptr2, :].T
                self.correct_queue[:, :b_c - self.K2 + ptr2] = k_correct[self.K2 - ptr2:, :].T
                self.correct_prediction_queue[ptr2:self.K2] = correct_predicts[:self.K2 - ptr2]
                self.correct_prediction_queue[:b_c - self.K2 + ptr2] = correct_predicts[self.K2 - ptr2:]
                full_k2 = True
            else:
                self.correct_queue[:, ptr2:ptr2 + b_c] = k_correct[:b_c, :].T
                self.correct_prediction_queue[ptr2:ptr2 + b_c] = correct_predicts[:b_c]
            self.correct_queue_ptr[0] = (ptr2 + b_c) % self.K2

        self.full_k1.fill_(full_k1)
        self.full_k2.fill_(full_k2)
        return full_k1, full_k2

    def forward(self, q, targets, outputs, outputs_k):
        """Official confidence-aware InfoNCE (part 1 'TT/TF' + part 2 'FF/FT').

        Momentum predictions gate the *key* masks; online predictions and the
        online softmax ``sr`` gate the *query*-side loss, exactly as official.
        """
        q = F.normalize(q, dim=1)
        predicted_m = outputs_k.argmax(dim=1)
        correct_mask_m = predicted_m == targets
        predicted = outputs.argmax(dim=1)
        outputs = F.softmax(outputs, dim=1)
        sr = outputs.max(dim=1).values.detach()
        correct_mask = predicted == targets
        error_predicts = predicted[~correct_mask]

        # Snapshot the queues (clone after detach): the loss graph would save
        # these for q's gradient, and the in-place enqueue that follows would
        # otherwise trip autograd's version check.
        error_queue = self.queue.detach().clone()
        error_queue_labels = self.prediction_queue.detach().clone()
        correct_queue = self.correct_queue.detach().clone()
        correct_queue_labels = self.correct_prediction_queue.detach().clone()

        # part 1: positives = same-target-class correct-queue entries (TT),
        # negatives = same-target-class error-queue entries (TF).
        sim_matrix = q @ error_queue
        eq = targets.view(-1, 1) == error_queue_labels.view(1, -1)
        sim_matrix = sim_matrix * eq
        sim_matrix = sim_matrix + (~eq).float() * -1e9

        sim_matrix_tp = q @ correct_queue
        eq_tp = targets.view(-1, 1) == correct_queue_labels.view(1, -1)
        sim_matrix_tp = sim_matrix_tp * eq_tp
        pos_sims = sim_matrix_tp[eq_tp]
        non_zero_counts = eq_tp.sum(dim=1)
        expanded_non_zero_counts = (non_zero_counts / sr).repeat_interleave(non_zero_counts)
        expanded_sim_matrix = sim_matrix.repeat_interleave(non_zero_counts, dim=0)

        logits_t = torch.cat([pos_sims.unsqueeze(-1), expanded_sim_matrix], dim=1)
        logits_t = logits_t / self.T
        logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values
        logsumexp_t = logits_t.exp().sum(dim=1).log()
        info_nce_loss_t = logsumexp_t - pos_sims
        info_nce_loss_t = info_nce_loss_t / expanded_non_zero_counts
        info_nce_loss_t = torch.sum(info_nce_loss_t) / q.shape[0]

        # part 2: error-class queries vs correct queue (official 'FF/FT')
        if error_predicts.numel() == 0:
            return (self.T / self.base_temperature) * info_nce_loss_t

        fq = q[~correct_mask]
        sim_matrix_ft = fq @ correct_queue
        eq_ft = error_predicts.view(-1, 1) == correct_queue_labels.view(1, -1)
        sim_matrix_ft = sim_matrix_ft * eq_ft
        sim_matrix_ft = sim_matrix_ft + (~eq_ft).float() * -1e9

        sim_matrix_ff = fq @ correct_queue
        eq_ff = targets[~correct_mask].view(-1, 1) == correct_queue_labels.view(1, -1)
        sim_matrix_ff = sim_matrix_ff * eq_ff
        pos_ff = sim_matrix_ff[eq_ff]
        non_zero_counts_f = eq_ff.sum(dim=1)
        expanded_non_zero_counts_f = non_zero_counts_f.repeat_interleave(non_zero_counts_f)
        expanded_ft = sim_matrix_ft.repeat_interleave(non_zero_counts_f, dim=0)

        logits_f = torch.cat([pos_ff.unsqueeze(-1), expanded_ft], dim=1)
        logits_f = logits_f / self.T
        logits_f = logits_f - logits_f.max(dim=1, keepdim=True).values
        logsumexp_f = logits_f.exp().sum(dim=1).log()
        info_nce_loss_f = logsumexp_f - pos_ff
        info_nce_loss_f = info_nce_loss_f / expanded_non_zero_counts_f
        info_nce_loss_f = torch.sum(info_nce_loss_f) / error_predicts.shape[0]

        return (self.T / self.base_temperature) * (info_nce_loss_t + info_nce_loss_f)


class CCLSCMethod(Method):
    method_name = "ccl_sc"

    def default_score(self) -> str:
        return "msp"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.pretrain = int(m.get("pretrain", 0))
        self.m = float(m.get("memo_m", 0.999))
        self.K = int(m.get("queue_size", 3000))
        self.T = float(m.get("temperature", 0.1))
        self.reward = float(m.get("reward", 1.0))

        # momentum encoder: same architecture, fully frozen
        self.model_k = build_backbone(train_cfg["backbone"], self.num_outputs, train_cfg)
        for p in self.model_k.parameters():
            p.requires_grad = False
        self.moco = CSCMoCo(
            dim=int(self.backbone.final_dim), K=self.K, m=self.m,
            T=self.T, num_class=self.num_classes,
        )

    def on_epoch_start(self, epoch: int):
        if epoch == self.pretrain:
            for pq, pk in zip(self.backbone.parameters(), self.model_k.parameters()):
                pk.data.copy_(pq.data)
        elif epoch > self.pretrain:
            for pq, pk in zip(self.backbone.parameters(), self.model_k.parameters()):
                pk.data.mul_(self.m).add_((1.0 - self.m) * pq.data)

    def predict_batch(self, x):
        bo = self.backbone(x)
        scores = compute_scores(bo.logits, self.default_scores())
        return MethodPrediction(bo.logits, bo.logits.argmax(dim=1), scores[self.score], scores)

    def optimizer_specs(self):
        t = self.cfg["train"]
        # model_k is EMA-updated and must never be touched by the optimizer
        return [
            {
                "params": self.backbone.parameters(),
                "kind": t.get("optimizer", "sgd"),
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
            }
        ]

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        bo = self.backbone(x)
        ce = F.cross_entropy(bo.logits, y)
        out = {"ce": ce}
        if state.epoch >= self.pretrain:
            with torch.no_grad():
                bo_k = self.model_k(x)
                k = F.normalize(bo_k.final_embedding, dim=1)
            preds = bo.logits.argmax(dim=1)
            preds_m = bo_k.logits.argmax(dim=1)
            err_m = preds_m != y
            corr_m = ~err_m
            # official order: loss from the *already-filled* queues first ...
            if bool(self.moco.full_k1) and bool(self.moco.full_k2):
                loss2 = self.moco(bo.final_embedding, y, bo.logits, bo_k.logits)
                out["csc"] = self.reward * loss2
            # ... then enqueue this batch's key features (momentum-masked).
            self.moco._dequeue_and_enqueue(
                k[err_m], k[corr_m],
                preds_m[corr_m], preds_m[err_m],
            )
        return out

    def inference_modules(self):
        # momentum encoder + queues are training-only artefacts
        return [self.backbone]


__all__ = ["CCLSCMethod", "CSCMoCo"]