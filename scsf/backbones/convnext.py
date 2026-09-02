"""ConvNeXt-Tiny (timm) with four native stage-output taps.

ConvNeXt heads on 32x32 CIFAR are intentionally *not* redesigned here: the
stem (stride-4 patchify conv) is unchanged, so the four stage outputs are at
8x8, 4x4, 2x2, 1x1. ``final_embedding`` = post-``norm_pre`` features flattened
to (B, 768). The only modification is the classification head width, which is
a registry-level concern shared by every method (no method touches module
names inside this backbone).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from .base import Backbone, BackboneOutput, flatten_spatial

__all__ = ["ConvNeXtTiny", "convnext_tiny"]


class ConvNeXtTiny(Backbone):
    registry_name = "convnext_tiny"

    def __init__(self, num_classes: int, input_size: int = 32, channels: int = 3, pretrained: bool = False):
        super().__init__(num_classes, input_size, channels)
        import timm

        timm_model = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=num_classes, in_chans=channels
        )
        self.base_model = timm_model
        self.final_dim = timm_model.num_features  # 768
        self.stages = timm_model.stages
        self.taps = OrderedDict(
            (f"stage{i + 1}", timm_model.get_submodule(f"stages.{i}")) for i in range(4)
        )
        self.roles = {"top_l1": "stage4", "top_l2": "stage3"}
        # Timm 1.0.29 attributes; verified on the execution server.
        assert hasattr(timm_model, "stem") and hasattr(timm_model, "norm_pre")

    def forward_backbone(self, x) -> BackboneOutput:
        features, x = self._embed(x)
        logits = self.base_model.head(x)
        return BackboneOutput(logits=logits, features=features, final_embedding=flatten_spatial(x))

    def forward_embedding(self, x):
        features, x = self._embed(x)
        return features, flatten_spatial(x)

    def _embed(self, x):
        m = self.base_model
        features = OrderedDict()
        x = m.stem(x)
        for i in range(4):
            x = m.stages[i](x)
            features[f"stage{i + 1}"] = x
        x = m.norm_pre(x)  # Identity in timm 1.0.29 convnext_tiny
        return features, x


def convnext_tiny(num_classes: int, input_size: int = 32, channels: int = 3) -> ConvNeXtTiny:
    return ConvNeXtTiny(num_classes, input_size, channels)