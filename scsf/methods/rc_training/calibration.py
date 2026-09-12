"""Shared raw-score confidence heads for the review-aligned family.

* ``RawScoreCalibrator`` — one raw score logit ``s``; primary confidence is
  ``sigmoid(s)``. Used by ``scsf_correctness``, ``r3_scsf``, ``cbr_scsf``.
* ``TransitionCalibrator`` — four-state output for ``dtr_scsf`` (softmax over
  the fixed state order ``[00, 01, 10, 11]``, index ``2*tS + tL``); the
  primary deployed confidence is ``softmax[01] + softmax[11]``
  (final-correctness marginal).

Both consume tapped features (attached) and classifier logits (detached) and
follow the review rule: features flow gradients, logits entering the head are
stop-gradient. Architecture mirrors the v1 MetaCalibrator family (taps pooled
to at most 2x2 then flattened; Linear->ReLU->Dropout stack; Xavier init) so
confidence-head capacity is comparable across methods.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ...backbones import adaptive_flatten

DEFAULT_HIDDEN_DIMS = (1024, 512, 256, 128)
STATE_ORDER = ("00", "01", "10", "11")


class _CalibratorMLP(nn.Module):
    """Feature-concat + detached-logits MLP producing ``out_width`` logits."""

    def __init__(
        self,
        feature_dims: Sequence[int],
        logit_dim: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        out_width: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dims = [int(d) for d in feature_dims]
        self.logit_dim = int(logit_dim)
        self.out_width = int(out_width)
        input_dim = sum(self.feature_dims) + self.logit_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), nn.ReLU(inplace=False), nn.Dropout(dropout)]
            prev = int(h)
        layers += [nn.Linear(prev, self.out_width)]
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def _feats(self, tapped_features):
        parts = [adaptive_flatten(f, out=2) for f in tapped_features]
        return torch.cat(parts, dim=1)

    def forward(self, tapped_features, logits):
        feats = self._feats(tapped_features)
        combined = torch.cat([feats, logits.detach()], dim=1)
        return self.network(combined)


class RawScoreCalibrator(_CalibratorMLP):
    """One raw score logit ``s`` (pre-sigmoid) from features + detached logits."""

    def __init__(self, feature_dims, logit_dim, hidden_dims=DEFAULT_HIDDEN_DIMS,
                 dropout=0.3):
        super().__init__(feature_dims, logit_dim, hidden_dims=hidden_dims,
                         out_width=1, dropout=dropout)

    def forward(self, tapped_features, logits):
        return super().forward(tapped_features, logits).squeeze(-1)  # (B,)

    def confidence(self, tapped_features, logits):
        return torch.sigmoid(self.forward(tapped_features, logits))


class TransitionCalibrator(_CalibratorMLP):
    """Four-state transition head (index ``2*tS + tL``) for DTR."""

    def __init__(self, feature_dims, logit_dim, hidden_dims=DEFAULT_HIDDEN_DIMS,
                 dropout=0.3):
        super().__init__(feature_dims, logit_dim, hidden_dims=hidden_dims,
                         out_width=4, dropout=dropout)

    def forward(self, tapped_features, logits):
        return super().forward(tapped_features, logits)  # (B, 4)

    def final_correctness_confidence(self, tapped_features, logits):
        """P(final classifier correct) = softmax(Q)[01] + softmax(Q)[11].

        Never "either depth correct" — a (1,0) sample is still an error of the
        final classifier until repaired.
        """
        q = torch.softmax(self.forward(tapped_features, logits), dim=1)
        return q[:, 1] + q[:, 3]


__all__ = [
    "_CalibratorMLP",
    "DEFAULT_HIDDEN_DIMS",
    "RawScoreCalibrator",
    "STATE_ORDER",
    "TransitionCalibrator",
]