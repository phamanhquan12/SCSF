"""CIFAR VGG16-BN with native max-pool taps (SCSF / Deep Gamblers / CCL-SC).

This is the *exact* VGG16-BN used by the original SCSF, Deep Gamblers and
CCL-SC papers: the standardized 16-layer feature stack with the compact
CIFAR head ``Linear(512,512) ReLU BN(512) Dropout2d Linear(C)`` (legacy
``models/cifar/vgg.py`` in this repo). Torchvision's 224px VGG (25088-dim
classifier) is not used on CIFAR because that would break direct
SCSF/CCL-SC compatibility.

Taps: outputs right after each of the 5 max-pool layers (`pool1..pool5`).
``final_embedding`` = the post-BN 512-d projection features (the exact hook
point CCL-SC's official code uses: ``classifier[:3]`` output).
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import Backbone, BackboneOutput

__all__ = ["VGG16BN", "vgg16_bn"]


def _make_features(cfg, batch_norm=False):
    """The Deep Gamblers VGG feature stack (order preserved from v1)."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif isinstance(v, int):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif isinstance(v, float):
            layers.append(nn.Dropout2d(v))
    return nn.Sequential(*layers)


CIFAR_VGG16_CFG = [64, 0.3, 64, "M", 128, 0.4, 128, "M", 256, 0.4, 256, 0.4, 256,
                   "M", 512, 0.4, 512, 0.4, 512, "M", 512, 0.4, 512, 0.4, 512, "M", 0.5]


class _VGGHead(nn.Sequential):
    """Compact CIFAR classifier shared with the DG/SCSF/CCL-SC baselines.

    A flat ``nn.Sequential`` keeps parameter keys identical to the legacy
    layout ``base_model.classifier.<i>.*`` (used by v1 checkpoints).
    """

    def __init__(self, num_classes: int):
        super().__init__(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout2d(0.5),
            nn.Linear(512, num_classes),
        )


class VGG16BN(Backbone):
    registry_name = "vgg16_bn"

    def __init__(self, num_classes: int, input_size: int = 32, channels: int = 3):
        super().__init__(num_classes, input_size, channels)
        self.base_model = nn.Module()
        self.base_model.features = _make_features(CIFAR_VGG16_CFG, batch_norm=True)
        self.base_model.classifier = _VGGHead(num_classes)
        # Match models/cifar/vgg.py from Deep Gamblers/CCL-SC: initialize the
        # complete model only after both feature and classifier modules exist.
        # Initializing only _VGGHead would leave convolutions at PyTorch's
        # default Kaiming-uniform distribution.
        self._initialize_weights()
        self.features = self.base_model.features
        self.classifier = self.base_model.classifier
        self.final_dim = 512
        self._pool_idx = [i for i, m in enumerate(self.features) if isinstance(m, nn.MaxPool2d)]
        assert len(self._pool_idx) == 5, "expected 5 max pools"
        self.taps = OrderedDict((f"pool{i + 1}", self.features[idx]) for i, idx in enumerate(self._pool_idx))
        self.roles = {"top_l1": "pool5", "top_l2": "pool4"}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward_backbone(self, x) -> BackboneOutput:
        features, flat = self._embed(x)
        # Do not slice self.classifier: nn.Sequential.__getitem__ re-invokes
        # _VGGHead.__init__ (wrong signature). Iterate modules directly to keep
        # the legacy base_model.classifier.<i>.* parameter layout intact.
        proj = None
        for i, mod in enumerate(self.classifier):
            flat = mod(flat)
            if i == 2:
                proj = flat  # Linear -> ReLU -> BN (512-d)
        logits = flat          # Dropout2d -> Linear
        return BackboneOutput(logits=logits, features=features, final_embedding=proj)

    def forward_embedding(self, x):
        features, flat = self._embed(x)
        return features, flat

    def _embed(self, x):
        features = OrderedDict()
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._pool_idx:
                features[f"pool{self._pool_idx.index(i) + 1}"] = x
        flat = x.view(x.size(0), -1)
        return features, flat


def vgg16_bn(num_classes: int, input_size: int = 32, channels: int = 3) -> VGG16BN:
    return VGG16BN(num_classes, input_size, channels)
