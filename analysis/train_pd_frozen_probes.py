#!/usr/bin/env python3
"""E02: three capacity-controlled probes, fit only on frozen validation features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pd_common import load_config, require_test_release
from export_pd_frozen_features import export_one
import numpy as np
import torch
from torch import nn

from scsf.dsspmi_contract import atomic_json, file_sha256, digest
from scsf.dsspmi_campaign import write_csv
from scsf.metrics import summary_from_correctness

FAMILIES = ("logits_only", "features_only", "combined")
WEIGHT_DECAYS = (0.0, 0.0001, 0.01)


def load_features(directory):
    manifest = load_config(directory / "manifest.json")
    parts = []
    for chunk in manifest["chunks"]:
        path = directory / chunk["file"]
        if file_sha256(path) != chunk["sha256"]:
            raise ValueError("Feature chunk fingerprint mismatch")
        parts.append(torch.load(path, map_location="cpu", weights_only=False))
    values = {k: torch.cat([p[k] for p in parts]) for k in ("logits", "features", "target", "prediction", "correct")}
    values["sample_ids"] = [sample for part in parts for sample in part["sample_ids"]]
    if len(set(values["sample_ids"])) != manifest["n"] or digest(values["sample_ids"]) != manifest["sample_ids_sha256"]:
        raise ValueError("Feature sample IDs changed")
    if not torch.equal(values["prediction"], values["logits"].argmax(1)):
        raise ValueError("Fixed classifier predictions do not match logits")
    return values, manifest


def inputs(values, family):
    if family == "logits_only":
        return values["logits"].float()
    if family == "features_only":
        return values["features"].float()
    return torch.cat([values["features"], values["logits"]], 1).float()


def stratified_folds(correct, seed=0):
    if min(int(correct.sum()), int((1 - correct).sum())) < 15:
        return None
    generator = torch.Generator().manual_seed(seed)
    folds = [[] for _ in range(3)]
    for label in (0, 1):
        idx = torch.where(correct == label)[0]
        idx = idx[torch.randperm(len(idx), generator=generator)]
        for i, part in enumerate(idx.tensor_split(3)):
            folds[i].extend(part.tolist())
    return [torch.tensor(sorted(f), dtype=torch.long) for f in folds]


def fit(x, correct, seed, decay, device="cpu"):
    torch.manual_seed(seed)
    mean = x.mean(0)
    std = x.std(0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    normalized = ((x - mean) / std).to(device)
    y = correct.to(device)
    model = nn.Sequential(nn.Linear(x.size(1), 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=decay)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(200):
        for idx in torch.randperm(len(x), generator=generator).split(64):
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(normalized[idx]).squeeze(1), y[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            optimizer.step()
    model.eval()
    return model, mean, std


@torch.no_grad()
def scores(model, x, mean, std, device):
    return model(((x - mean) / std).to(device)).squeeze(1).cpu().numpy()


def run(config, evaluate=False):
    if not config.get("checkpoints"):
        raise ValueError("Configure original checkpoint paths before probe fitting")
    release = require_test_release(config.get("test_release_lock")) if evaluate else None
    device = config.get("device", "cpu")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    rows = []
    out = Path(config["output_dir"])
    for entry in config["checkpoints"]:
        try:
            validation = export_one(config, entry, "val")
            values, manifest = load_features(validation)
            folds = stratified_folds(values["correct"])
            if folds is None:
                rows.append({"dataset": manifest["dataset"], "backbone_seed": manifest["seed"],
                             "status": "insufficient_calibration_errors"})
                continue
            test, test_manifest = (load_features(export_one(config, entry, "test")) if evaluate else (None, None))
            for family in FAMILIES:
                x = inputs(values, family)
                prefix = out / "probe_checkpoints" / manifest["dataset"] / f"seed_{manifest['seed']}" / family
                prefix.mkdir(parents=True, exist_ok=True)
                selection_path = prefix / "selection.json"
                if not evaluate:
                    if selection_path.exists():
                        selected = load_config(selection_path)
                        if selected["validation_manifest_sha256"] != file_sha256(validation / "manifest.json"):
                            raise ValueError("Probe selection data changed")
                        decay = selected["weight_decay"]
                    else:
                        trials = []
                        for decay_value in WEIGHT_DECAYS:
                            areas = []
                            for held in folds:
                                mask = torch.ones(len(x), dtype=torch.bool); mask[held] = False
                                model, mean, std = fit(x[mask], values["correct"][mask], 0, decay_value, device)
                                score = scores(model, x[held], mean, std, device)
                                areas.append(summary_from_correctness(score, values["correct"][held].numpy()).aurc)
                            trials.append({"weight_decay": decay_value, "internal_aurc": float(np.mean(areas))})
                        selected = min(trials, key=lambda r: (r["internal_aurc"], -r["weight_decay"]))
                        decay = selected["weight_decay"]
                        atomic_json(selection_path, {**selected, "trials": trials,
                                    "validation_manifest_sha256": file_sha256(validation / "manifest.json")})
                for seed in (0, 1, 2):
                    path = prefix / f"probe_{seed}.pt"
                    if evaluate:
                        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                        if checkpoint["validation_manifest_sha256"] != file_sha256(validation / "manifest.json"):
                            raise ValueError("Probe training manifest changed")
                        model = nn.Sequential(nn.Linear(x.size(1), 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
                        model.load_state_dict(checkpoint["model"], strict=True); model.eval()
                        mean, std = checkpoint["mean"], checkpoint["std"]
                        score = scores(model, inputs(test, family), mean, std, device)
                        metric = summary_from_correctness(score, test["correct"].numpy())
                        score_path = out / "probe_scores" / manifest["dataset"] / f"seed_{manifest['seed']}" / family / f"probe_{seed}.csv"
                        write_csv(score_path, [{"sample_id": sample, "prediction": int(pred), "target": int(target),
                                   "correct": int(correct), "confidence": float(s)} for sample, pred, target, correct, s in
                                  zip(test["sample_ids"], test["prediction"], test["target"], test["correct"], score)])
                        rows.append({"dataset": manifest["dataset"], "backbone_seed": manifest["seed"],
                                     "family": family, "probe_seed": seed, "status": "evaluated", "aurc": metric.aurc,
                                     "auroc": metric.auroc, "test_n": metric.num_samples, "test_release_sha256": release})
                    else:
                        if not path.exists():
                            model, mean, std = fit(x, values["correct"], seed, decay, device)
                            torch.save({"model": model.cpu().state_dict(), "mean": mean, "std": std,
                                        "family": family, "seed": seed, "weight_decay": decay,
                                        "validation_manifest_sha256": file_sha256(validation / "manifest.json"),
                                        "parameter_count": sum(p.numel() for p in model.parameters()),
                                        "epochs": 200, "hidden_width": 64, "batch_size": 64, "lr": 0.001}, path)
                        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                        if checkpoint["validation_manifest_sha256"] != file_sha256(validation / "manifest.json"):
                            raise ValueError("Existing probe uses different validation data")
                        rows.append({"dataset": manifest["dataset"], "backbone_seed": manifest["seed"],
                                     "family": family, "probe_seed": seed, "status": "fitted", "weight_decay": decay,
                                     "parameter_count": checkpoint["parameter_count"],
                                     "correct_n": int(values["correct"].sum()), "wrong_n": int((1-values["correct"]).sum())})
        except (OSError, ValueError, RuntimeError) as exc:
            rows.append({"checkpoint": entry.get("checkpoint", ""), "status": "blocked_or_failed", "reason": str(exc)})
    write_csv(out / ("probe_test_runs.csv" if evaluate else "probe_runs.csv"), rows)
    if any(r["status"] == "blocked_or_failed" for r in rows):
        raise RuntimeError("Some probe jobs were blocked/failed; inspect probe run status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), args.evaluate)


if __name__ == "__main__":
    main()
