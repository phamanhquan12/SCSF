"""CIFAR-adapted DeiT-S with per-block taps (timm VisionTransformer).

Adaptation (frozen before any result is produced, see configs/backbones/deit_s.yaml
and docs/EMPIRICAL_CONTRACT.md): DeiT-S uses 12 blocks, embed_dim 384, 6 heads,
MLP ratio 4 — identical to the DeiT-S config — but we use a 32x32 input with a
4x4 patch (DeiT-S/16 at 224 px is computationally prohibitive for the gate
matrix on a single 4090). This keeps the same first-stage receptive geometry
(8x8 patches) as DeiT-S/16 on 224px. No distillation token is used.

Taps: every transformer block output is captured (``block0..block11``). For
analysis compute, adjacent blocks may be grouped (2,2,2,2,2,2) into semantic
roles ``stage_group0..stage_group5``; grouping is logged in the config.
``final_embedding`` = the CLS token after the final LayerNorm (384-d).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Backbone, BackboneOutput

__all__ = ["DeiTS", "deit_s", "deit_small_cifar"]

#: grouping of the 12 blocks into semantic super-stages (documented).
DEIT_BLOCK_GROUPS = (2, 2, 2, 2, 2, 2)


class DeiTS(Backbone):
    registry_name = "deit_s"

    def __init__(
        self,
        num_classes: int,
        input_size: int = 32,
        channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        block_groups: tuple = DEIT_BLOCK_GROUPS,
    ):
        super().__init__(num_classes, input_size, channels)
        from timm.models.vision_transformer import VisionTransformer

        if input_size == 32:
            self.patch_size = patch_size  # 4
        else:
            self.patch_size = patch_size
        self.block_groups = tuple(block_groups)
        if sum(self.block_groups) != depth:
            raise ValueError("block_groups must partition the DeiT-S depth (12)")
        self.base_model = VisionTransformer(
            img_size=input_size,
            patch_size=self.patch_size,
            in_chans=channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            class_token=True,
            no_embed_class=False,
            global_pool="",
            norm_layer=nn.LayerNorm,
        )
        self.final_dim = embed_dim
        self.embed_dim = embed_dim
        self.taps = OrderedDict(
            (f"block{i}", self.base_model.blocks[i]) for i in range(depth)
        )
        self.roles = self._make_group_roles()

    def _make_group_roles(self):
        names = self._make_group_names()
        roles = {}
        offset = 0
        for gi, n in enumerate(self.block_groups):
            group_names = [f"block{i}" for i in range(offset, offset + n)]
            roles[f"stage_group{gi}"] = group_names[-1]
            offset += n
        roles["top_l1"] = names[-1]
        roles["top_l2"] = names[-2]
        return roles

    def _make_group_names(self):
        names, offset = [], 0
        for n in self.block_groups:
            names.append(f"block{offset + n - 1}")
            offset += n
        return names

    def forward_backbone(self, x) -> BackboneOutput:
        features, cls = self._embed(x)
        logits = self.base_model.head(cls)
        return BackboneOutput(logits=logits, features=features, final_embedding=cls)

    def forward_embedding(self, x):
        return self._embed(x)

    def _embed(self, x):
        m = self.base_model
        features = OrderedDict()
        x = m.patch_embed(x)
        if m.cls_token is not None:
            x = torch.cat((m.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        for i, block in enumerate(m.blocks):
            x = block(x)
            features[f"block{i}"] = x
        x = m.norm(x)
        cls = x[:, 0]
        return features, cls


def deit_s(num_classes: int, input_size: int = 32, channels: int = 3, patch_size: int = 4) -> DeiTS:
    return DeiTS(num_classes, input_size, channels, patch_size=patch_size)


# Backwards-compatible spelling used in older configs.
deit_small_cifar = deit_s