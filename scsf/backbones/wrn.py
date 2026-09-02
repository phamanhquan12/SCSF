"""WideResNet-28-10 (so-called, per the gate contract) with 3 residual groups.

Standard WideResNet (Zagoruyko & Komodakis 2016) with depth 28, widen factor
10, and dropout 0.0. Taps: outputs at the end of each of the 3 residual
blocks. ``final_embedding`` = ReLU(group3) -> global average pool -> 640-d.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Backbone, BackboneOutput

__all__ = ["WideResNet28_10", "wideresnet28_10"]


class _WRNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.dropout = dropout
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class _WRNBase(nn.Module):
    def __init__(self, layers, num_classes, widen_factor=10, dropout=0.0):
        super().__init__()
        in_planes = 16
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.in_planes = in_planes
        self.group1 = self._make_group(layers[0], in_planes * widen_factor, stride=1, dropout=dropout)
        self.group2 = self._make_group(layers[1], in_planes * widen_factor * 2, stride=2, dropout=dropout)
        self.group3 = self._make_group(layers[2], in_planes * widen_factor * 4, stride=2, dropout=dropout)
        self.bn1 = nn.BatchNorm2d(in_planes * widen_factor * 4)
        self.linear = nn.Linear(in_planes * widen_factor * 4, num_classes)
        self.final_dim = in_planes * widen_factor * 4

    def _make_group(self, n, out_planes, stride, dropout):
        block = _WRNBlock(self.in_planes, out_planes, stride, dropout)
        self.in_planes = out_planes
        for _ in range(1, n):
            block = _make_sequential_chain(block, _WRNBlock(self.in_planes, out_planes, 1, dropout))
        return block

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn1(out))
        return out


def _make_sequential_chain(*mods):
    if len(mods) == 1:
        return mods[0]
    seq = nn.Sequential()
    for i, m in enumerate(mods):
        seq.add_module(str(i), m)
    return seq


class WideResNet28_10(Backbone):
    registry_name = "wideresnet28_10"

    def __init__(self, num_classes: int, input_size: int = 32, channels: int = 3, dropout: float = 0.0):
        super().__init__(num_classes, input_size, channels)
        self.base = _WRNBase([4, 4, 4], num_classes=num_classes, widen_factor=10, dropout=dropout)
        self.final_dim = self.base.final_dim
        self.taps = OrderedDict(
            (f"group{i + 1}", self.base.get_submodule(f"group{i + 1}")) for i in range(3)
        )
        self.roles = {"top_l1": "group3", "top_l2": "group2"}

    def forward_backbone(self, x) -> BackboneOutput:
        features, embedding = self._embed(x)
        logits = self.base.linear(embedding)
        return BackboneOutput(logits=logits, features=features, final_embedding=embedding)

    def forward_embedding(self, x):
        return self._embed(x)

    def _embed(self, x):
        b = self.base
        x = b.conv1(x)
        g1 = b.group1(x)
        g2 = b.group2(g1)
        g3 = b.group3(g2)
        features = OrderedDict([("group1", g1), ("group2", g2), ("group3", g3)])
        out = F.relu(b.bn1(g3))
        embedding = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return features, embedding


def wideresnet28_10(num_classes: int, input_size: int = 32, channels: int = 3) -> WideResNet28_10:
    return WideResNet28_10(num_classes, input_size, channels)