#!/usr/bin/env python3
"""E02: export frozen PD features, validation first and test only after release."""
from __future__ import annotations

import argparse
from pathlib import Path

from pd_common import (load_config, load_checkpoint, verified_data, state_sha256,
                       require_test_release)
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from scsf.datasets import DatasetSpec, _transforms
from scsf.dsspmi_contract import atomic_json, file_sha256, digest
from scsf.dsspmi_campaign import CORE4, FULL


def export_one(config, entry, split="val"):
    if split not in {"val", "test"}:
        raise ValueError("Only deterministic validation/test exports are supported")
    release_sha = require_test_release(config.get("test_release_lock")) if split == "test" else None
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, args, _ = load_checkpoint(entry["checkpoint"], device, config["data_dir"])
    if (args.dataset not in CORE4 or args.seed not in (0, 1, 2) or args.method != "pd_scsf"
            or args.variant != FULL):
        raise ValueError("E02 requires an original full DS+ Core4 checkpoint with seed 0/1/2")
    data = verified_data(entry, args, config["data_dir"])
    before = state_sha256(model)
    model.eval()
    model.requires_grad_(False)
    metadata = data["identity"]
    spec = DatasetSpec(args.dataset, len(metadata["classes"][split]), args.input_size or 224,
                       tuple(metadata["mean"]), tuple(metadata["std"]))
    _, transform = _transforms(spec)
    root = Path(config["data_dir"]) / "preprocessed" / f"{args.dataset}_{spec.input_size}"
    dataset = ImageFolder(root / split, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=config.get("workers", 4))
    checkpoint_sha = file_sha256(entry["checkpoint"])
    out = Path(config["output_dir"]) / "frozen_features" / args.dataset / f"seed_{args.seed}" / split
    if out.exists() and any(out.iterdir()):
        manifest = load_config(out / "manifest.json")
        if (manifest["checkpoint_sha256"] != checkpoint_sha or manifest["data_sha256"] != data["sha256"]
                or manifest.get("test_release_sha256") != release_sha):
            raise ValueError("Existing feature export has a different contract")
        for chunk in manifest["chunks"]:
            if file_sha256(out / chunk["file"]) != chunk["sha256"]:
                raise ValueError("Export chunk changed")
        return out
    out.mkdir(parents=True, exist_ok=True)
    chunks, offset = [], 0
    all_ids = []
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            logits, _, maps = model.backbone.forward_with_maps(images.to(device))
            pooled = [torch.nn.functional.adaptive_avg_pool2d(maps[k], 1).flatten(1).cpu() for k in ("mid", "late")]
            ids = [Path(path).relative_to(root).as_posix() for path, _ in dataset.samples[offset:offset + len(labels)]]
            offset += len(labels)
            all_ids.extend(ids)
            payload = {"logits": logits.cpu(), "features": torch.cat(pooled, 1), "target": labels,
                       "prediction": logits.argmax(1).cpu(), "correct": logits.argmax(1).cpu().eq(labels).float(),
                       "sample_ids": ids}
            name = f"chunk_{index:05d}.pt"
            torch.save(payload, out / name)
            chunks.append({"file": name, "sha256": file_sha256(out / name), "n": len(labels)})
    if len(set(all_ids)) != len(all_ids) or state_sha256(model) != before:
        raise ValueError("Duplicate sample IDs or frozen model mutated")
    atomic_json(out / "manifest.json", {"dataset": args.dataset, "seed": args.seed, "split": split,
                "checkpoint_sha256": checkpoint_sha, "data_sha256": data["sha256"],
                "model_state_sha256": before, "sample_ids_sha256": digest(all_ids),
                "n": offset, "chunks": chunks, "test_release_sha256": release_sha})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    config = load_config(args.config)
    entries = config["checkpoints"]
    if not entries:
        parser.error("Configure the original DS+ checkpoint/provenance paths first")
    for entry in entries:
        print(export_one(config, entry, args.split))


if __name__ == "__main__":
    main()
