#!/usr/bin/env python3
"""Query-weighted acccon on CCL-SC's large-image datasets (CelebA, ImageNet).

Keeps the CIFAR acccon loss (CE + correctness BCE + accept-weighted SupCon)
and matches each dataset's CCL-SC backbone / optimizer / schedule:

  CelebA    ResNet-18, Adam 1e-5, 50 epochs, Es=1, official val for selection
  ImageNet  ResNet-34, SGD 0.1, 150 epochs, Es=50, last.pth
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from dyn_scsf import evaluate_selective, jsonable
from train_cbr_scsf import RawConfidenceHead, parse_coverages, seed_everything
from train_next_scsf import apply_variant_defaults, train_epoch
from train_search_scsf import FeatureQueue, ProjectionHead


CELEBA_MEAN = (0.5063486, 0.4258108, 0.38318512)
CELEBA_STD = (0.26577517, 0.24520662, 0.24129295)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ATTRACTIVE_INDEX = 2


class ResNetFeatureExtractor(nn.Module):
    """ResNet with SCSF-style late features: layer3 (2x2) and layer4 (GAP)."""

    def __init__(self, arch, num_classes):
        super().__init__()
        if arch == "resnet18":
            net = models.resnet18(weights=None)
            layer3_ch, layer4_ch = 256, 512
        elif arch == "resnet34":
            net = models.resnet34(weights=None)
            layer3_ch, layer4_ch = 256, 512
        else:
            raise ValueError(f"unsupported arch: {arch}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.fc = nn.Linear(net.fc.in_features, num_classes)
        self.gap2 = nn.AdaptiveAvgPool2d((2, 2))
        self.layer3_dim = layer3_ch * 4
        self.layer4_dim = layer4_ch

    def forward(self, x, return_features=False, return_pool3=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat3 = self.layer3(x)
        feat4 = self.layer4(feat3)
        pooled = torch.flatten(self.avgpool(feat4), 1)
        logits = self.fc(pooled)
        if not return_features:
            return logits
        pool4 = torch.flatten(self.gap2(feat3), 1)
        pool5 = torch.flatten(self.avgpool(feat4), 1)
        if return_pool3:
            pool3 = torch.flatten(F_adaptive_avg(feat3), 1)
            return logits, pool4, pool5, pool3
        return logits, pool4, pool5


def F_adaptive_avg(feat):
    return torch.nn.functional.adaptive_avg_pool2d(feat, 1)


class AttractiveCelebA(torch.utils.data.Dataset):
    """CelebA with the Attractive attribute as a binary label (CCL-SC)."""

    def __init__(self, root, split, transform):
        self.base = datasets.CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=False,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, attrs = self.base[index]
        return image, int(attrs[ATTRACTIVE_INDEX].item())


def data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data")


def dataset_presets(name):
    if name == "celeba":
        return {
            "arch": "resnet18",
            "num_classes": 2,
            "epochs": 50,
            "pretrain": 1,
            "ramp_epochs": 2,
            "batch_size": 64,
            "lr": 1e-5,
            "optimizer": "adam",
            "queue_size": 300,
            "select_by": "val_accuracy",
            "eval_test_set": "official_test",
        }
    if name == "imagenet":
        return {
            "arch": "resnet34",
            "num_classes": 1000,
            "epochs": 150,
            "pretrain": 50,
            "ramp_epochs": 10,
            "batch_size": 256,
            "lr": 0.1,
            "optimizer": "sgd",
            "queue_size": 10000,
            "select_by": "last",
            "eval_test_set": "official_val",
        }
    raise ValueError(f"unsupported dataset: {name}")


def build_celeba_loaders(args):
    root = data_root()
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ]
    )
    train_set = AttractiveCelebA(root, "train", train_tf)
    val_set = AttractiveCelebA(root, "valid", eval_tf)
    test_set = AttractiveCelebA(root, "test", eval_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


def build_imagenet_loaders(args):
    root = os.path.join(data_root(), "imagenet")
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet folders not found under {root}. "
            "This 100GB VAST overlay cannot hold ImageNet (~140GB)."
        )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_loader = DataLoader(
        datasets.ImageFolder(train_dir, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, eval_tf),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader, val_loader


def build_loaders(args):
    if args.dataset == "celeba":
        return build_celeba_loaders(args)
    if args.dataset == "imagenet":
        return build_imagenet_loaders(args)
    raise ValueError(f"unsupported dataset: {args.dataset}")


def fill_acccon_args(args, preset):
    args.variant = "acccon"
    args.micro_weight = 0.0
    args.acceptce_weight = 0.0
    args.con_weight = 0.5
    args.tail_weight = 0.0
    args.con_mode = "accept"
    args.coverages = parse_coverages("0.80,0.90,0.95")
    args.soft_temperature = 0.2
    args.threshold_iterations = 60
    args.con_temperature = 0.1
    args.proj_dim = 128
    args.meta_weight = 1.0
    args.min_meta_weight = 1e-4
    args.max_grad_norm = 5.0
    args.limit_train_batches = 0
    args.boundary_beta = 1.0
    args.query_floor = 0.20
    args.query_weight_mode = "none"
    args.confidence_target = "hard"
    apply_variant_defaults(args)
    args.epochs = preset["epochs"] if args.epochs is None else args.epochs
    args.pretrain = preset["pretrain"] if args.pretrain is None else args.pretrain
    args.ramp_epochs = (
        preset["ramp_epochs"] if args.ramp_epochs is None else args.ramp_epochs
    )
    args.batch_size = (
        preset["batch_size"] if args.batch_size is None else args.batch_size
    )
    args.queue_size = (
        preset["queue_size"] if args.queue_size is None else args.queue_size
    )
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="acccon (qw) on CelebA / ImageNet")
    parser.add_argument("-d", "--dataset", required=True, choices=["celeba", "imagenet"])
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--pretrain", default=None, type=int)
    parser.add_argument("--ramp-epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--queue-size", default=None, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    args = parser.parse_args()
    preset = dataset_presets(args.dataset)
    if args.output_dir is None:
        args.output_dir = f"./save/next_acccon_{args.dataset}_seed{args.seed}_qw"
    fill_acccon_args(args, preset)
    args.arch = preset["arch"]
    args.num_classes = preset["num_classes"]
    args.lr = preset["lr"]
    args.optimizer = preset["optimizer"]
    args.select_by = preset["select_by"]
    args.eval_test_set = preset["eval_test_set"]
    return args, preset


def main():
    args, preset = parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(args)

    backbone = ResNetFeatureExtractor(args.arch, args.num_classes).to(device)
    confidence_head = RawConfidenceHead(
        backbone.layer3_dim, backbone.layer4_dim, args.num_classes
    ).to(device)
    projector = ProjectionHead(in_dim=backbone.layer4_dim, out_dim=args.proj_dim).to(
        device
    )
    queue = FeatureQueue(args.proj_dim, args.queue_size, device)

    if args.optimizer == "adam":
        backbone_optimizer = optim.Adam(backbone.parameters(), lr=args.lr)
        scheduler = None
    else:
        backbone_optimizer = optim.SGD(
            backbone.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            backbone_optimizer,
            milestones=list(range(10, args.epochs + 1, 10)),
            gamma=0.5,
        )
    head_optimizer = optim.Adam(
        list(confidence_head.parameters()) + list(projector.parameters()),
        lr=args.meta_lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = os.path.join(args.output_dir, "last.pth")
    best_path = os.path.join(args.output_dir, "best.pth")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2)

    best_val_acc = -math.inf
    best_val_aurc = math.inf
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
            )
            if scheduler is not None:
                scheduler.step()
            validation = evaluate_selective(
                backbone, confidence_head, val_loader, device
            )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
            print(
                f"Epoch {epoch:03d}/{args.epochs} acccon {args.dataset} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.2f}% "
                f"con={train_metrics['contrast']:.4f} "
                f"val_acc={validation['accuracy']:.2f}% "
                f"val_AURC={validation['aurc']:.6f}",
                flush=True,
            )
            history_file.write(json.dumps(jsonable(record)) + "\n")
            history_file.flush()
            payload = {
                "epoch": epoch,
                "variant": "acccon",
                "dataset": args.dataset,
                "backbone": backbone.state_dict(),
                "confidence_head": confidence_head.state_dict(),
                "projector": projector.state_dict(),
                "args": jsonable(vars(args)),
                "validation": validation,
            }
            torch.save(payload, last_path)
            improved = (
                validation["accuracy"] > best_val_acc
                if args.select_by == "val_accuracy"
                else validation["aurc"] < best_val_aurc
            )
            if improved:
                best_val_acc = validation["accuracy"]
                best_val_aurc = validation["aurc"]
                torch.save(payload, best_path)

    eval_path = last_path if args.select_by == "last" else best_path
    checkpoint = torch.load(eval_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone"])
    confidence_head.load_state_dict(checkpoint["confidence_head"])
    test = evaluate_selective(backbone, confidence_head, test_loader, device)
    last_ckpt = torch.load(last_path, map_location=device, weights_only=False)
    backbone.load_state_dict(last_ckpt["backbone"])
    confidence_head.load_state_dict(last_ckpt["confidence_head"])
    test_last = evaluate_selective(backbone, confidence_head, test_loader, device)
    results = {
        "eval_checkpoint": "best_val_acc" if args.select_by == "val_accuracy" else "last",
        "eval_test_set": args.eval_test_set,
        "variant": "acccon",
        "dataset": args.dataset,
        "arch": args.arch,
        "selected_epoch": checkpoint["epoch"],
        "best_val_accuracy": best_val_acc if best_val_acc > -math.inf else None,
        "best_val_aurc": best_val_aurc if best_val_aurc < math.inf else None,
        "test": test,
        "test_last": test_last,
        "args": jsonable(vars(args)),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(results), handle, indent=2)
    with open(
        os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(jsonable(test), handle, indent=2)
    print(
        f"Evaluated {results['eval_checkpoint']} (epoch {checkpoint['epoch']}) "
        f"dataset={args.dataset}; {args.eval_test_set} "
        f"accuracy={test['accuracy']:.2f}% AURC={test['aurc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
