from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tv_models

from models.cifar import vgg


class FeatureModel(nn.Module):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VGGFeatureModel(FeatureModel):
    def __init__(self, num_classes: int, input_size: int):
        super().__init__()
        self.adaptive_pool = input_size not in {32, 64}
        self.base = vgg.vgg16_bn(num_classes=num_classes, input_size=32 if self.adaptive_pool else input_size)

    @property
    def feature_dim(self) -> int:
        first_linear = next(m for m in self.base.classifier.modules() if isinstance(m, nn.Linear))
        return first_linear.in_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.features(x)
        if self.adaptive_pool:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.base.classifier(features)
        return (logits, features) if return_features else logits


class TorchvisionFeatureModel(FeatureModel):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool,
        multi_layer_features: bool = False,
        feature_spec: str | None = None,
    ):
        super().__init__()
        weights = None
        if pretrained:
            weights = "DEFAULT"
        self.feature_spec = feature_spec
        self.multi_layer_features = multi_layer_features
        if arch in {"resnet18", "resnet50", "resnet101"}:
            model_fn = {
                "resnet18": tv_models.resnet18,
                "resnet50": tv_models.resnet50,
                "resnet101": tv_models.resnet101,
            }[arch]
            model = model_fn(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Linear(feature_dim, num_classes)
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.layer2_dim = self._resnet_block_out_channels(model.layer2[-1])
            self.layer3_dim = self._resnet_block_out_channels(model.layer3[-1])
            self.layer4_dim = feature_dim
            self.classifier = model.fc
        elif arch == "densenet121":
            model = tv_models.densenet121(weights=weights)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Linear(feature_dim, num_classes)
            self.features = model.features
            self.densenet_stage_dims = {
                "early": model.features.transition2.conv.out_channels,
                "mid": model.features.transition3.conv.out_channels,
                "late": feature_dim,
            }
            self.classifier = model.classifier
        else:
            raise ValueError(f"Unsupported torchvision arch: {arch}")
        self.arch = arch
        self._classifier_feature_dim = feature_dim
        self._feature_layers = self._parse_feature_layers(feature_spec, multi_layer_features)
        if self._supports_feature_layers and self._feature_layers:
            dims = self._stage_dims
            self._feature_dim = sum(dims[layer] for layer in self._feature_layers)
        elif self._supports_feature_layers and feature_spec == "logits":
            self._feature_dim = 0
        else:
            self._feature_dim = feature_dim

    @property
    def _is_resnet(self) -> bool:
        return self.arch in {"resnet18", "resnet50", "resnet101"}

    @property
    def _is_densenet(self) -> bool:
        return self.arch == "densenet121"

    @property
    def _supports_feature_layers(self) -> bool:
        return self._is_resnet or self._is_densenet

    @property
    def _stage_dims(self) -> dict[str, int]:
        if self._is_resnet:
            return {"early": self.layer2_dim, "mid": self.layer3_dim, "late": self.layer4_dim}
        if self._is_densenet:
            return self.densenet_stage_dims
        return {}

    @staticmethod
    def _resnet_block_out_channels(block: nn.Module) -> int:
        if hasattr(block, "conv3"):
            return block.conv3.out_channels
        return block.conv2.out_channels

    @staticmethod
    def _parse_feature_layers(feature_spec: str | None, multi_layer_features: bool) -> list[str]:
        if feature_spec is None:
            return ["mid", "late"] if multi_layer_features else []
        parts = [part.strip() for part in feature_spec.split("+") if part.strip()]
        valid = {"early", "mid", "late", "logits"}
        invalid = sorted(set(parts) - valid)
        if invalid:
            raise ValueError(f"Invalid SCSF feature spec part(s): {invalid}. Valid parts: {sorted(valid)}")
        return [part for part in ("early", "mid", "late") if part in parts]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self._forward_features_and_classifier_input(x)
        return features

    def _forward_features_and_classifier_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._is_resnet:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            layer2 = x
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)
            final_features = nn.functional.adaptive_avg_pool2d(layer4, (1, 1)).flatten(1)
            if not self._feature_layers:
                if self.feature_spec == "logits":
                    return final_features.new_zeros(final_features.size(0), 0), final_features
                return final_features, final_features
            pooled = {
                "early": nn.functional.adaptive_avg_pool2d(layer2, (1, 1)).flatten(1),
                "mid": nn.functional.adaptive_avg_pool2d(layer3, (1, 1)).flatten(1),
                "late": final_features,
            }
            return torch.cat([pooled[layer] for layer in self._feature_layers], dim=1), final_features

        if self._is_densenet:
            stage_maps = {}
            for name, module in self.features.named_children():
                x = module(x)
                if name == "transition2":
                    stage_maps["early"] = x
                elif name == "transition3":
                    stage_maps["mid"] = x
                elif name == "norm5":
                    x = torch.relu(x)
                    stage_maps["late"] = x

            final_features = nn.functional.adaptive_avg_pool2d(stage_maps["late"], (1, 1)).flatten(1)
            if not self._feature_layers:
                if self.feature_spec == "logits":
                    return final_features.new_zeros(final_features.size(0), 0), final_features
                return final_features, final_features
            pooled = {
                layer: nn.functional.adaptive_avg_pool2d(stage_maps[layer], (1, 1)).flatten(1)
                for layer in self._feature_layers
            }
            return torch.cat([pooled[layer] for layer in self._feature_layers], dim=1), final_features

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        return features, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features, classifier_features = self._forward_features_and_classifier_input(x)
        logits = self.classifier(classifier_features)
        return (logits, features) if return_features else logits


def build_backbone(
    arch: str,
    num_classes: int,
    input_size: int,
    pretrained: bool = False,
    multi_layer_features: bool = False,
    feature_spec: str | None = None,
) -> FeatureModel:
    if arch == "vgg16_bn":
        if multi_layer_features:
            raise ValueError("multi_layer_features is currently implemented for ResNet backbones only.")
        return VGGFeatureModel(num_classes=num_classes, input_size=input_size)
    if arch in {"resnet18", "resnet50", "resnet101", "densenet121"}:
        return TorchvisionFeatureModel(
            arch,
            num_classes=num_classes,
            pretrained=pretrained,
            multi_layer_features=multi_layer_features,
            feature_spec=feature_spec,
        )
    raise ValueError("Unsupported arch '{}'. Choices: vgg16_bn, resnet18, resnet50, resnet101, densenet121".format(arch))
