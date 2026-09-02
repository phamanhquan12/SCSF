"""ResNet-18 with ``layer1..layer4`` taps (torchvision implementation)."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Backbone, BackboneOutput

__all__ = ["ResNet18", "resnet18"]


class ResNet18(Backbone):
    registry_name = "resnet18"

    def __init__(self, num_classes: int, input_size: int = 32, channels: int = 3):
        super().__init__(num_classes, input_size, channels)
        import torchvision.models as tv

        self.base_model = tv.resnet18(num_classes=num_classes)
        self.final_dim = 512
        self.taps = OrderedDict(
            (f"layer{i + 1}", self.base_model.get_submodule(f"layer{i + 1}"))
            for i in range(4)
        )
        self.roles = {"top_l1": "layer4", "top_l2": "layer3"}

    def forward_backbone(self, x) -> BackboneOutput:
        features, embedding = self._embed(x)
        logits = self.base_model.fc(embedding)
        return BackboneOutput(logits=logits, features=features, final_embedding=embedding)

    def forward_embedding(self, x):
        return self._embed(x)

    def _embed(self, x):
        m = self.base_model
        features = OrderedDict()
        x = _stem(x, m)
        block1 = m.layer1(x)
        block2 = m.layer2(block1)
        block3 = m.layer3(block2)
        block4 = m.layer4(block3)
        features["layer1"], features["layer2"] = block1, block2
        features["layer3"], features["layer4"] = block3, block4
        pooled = m.avgpool(block4)
        embedding = torch.flatten(pooled, 1)
        return features, embedding


def _stem(x, m):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    return x


def resnet18(num_classes: int, input_size: int = 32, channels: int = 3) -> ResNet18:
    return ResNet18(num_classes, input_size, channels)