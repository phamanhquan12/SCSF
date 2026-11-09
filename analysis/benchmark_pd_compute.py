#!/usr/bin/env python3
"""E16: checkpoint-based FP32 CUDA compute/memory measurements."""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from pd_common import load_checkpoint, load_config, verified_data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from scsf.dsspmi_campaign import write_csv
from scsf.dsspmi_contract import atomic_json, audit_cache, file_sha256
from scsf.datasets import DatasetSpec, _transforms


def optimizers(model, args):
    if args.method == "sr":
        return [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)]
    return [torch.optim.SGD([p for n, p in model.named_parameters() if not n.startswith("calibrator.")],
                            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
            torch.optim.Adam(model.calibrator.parameters(), lr=args.meta_lr)]


def benchmark(config):
    if not torch.cuda.is_available():
        raise RuntimeError("E16 requires CUDA on the H100; no CPU timing substitution")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Primary E16 measurements require an H100")
    # Reject shared execution rather than report contaminated timings.
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    query = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"]
    processes = subprocess.check_output(query, text=True).strip().splitlines()
    for line in processes:
        pid, gpu = [part.strip() for part in line.split(",", 1)]
        if (not uuid or gpu == uuid) and int(pid) != os.getpid():
            raise RuntimeError(f"Other compute process on benchmark GPU: {line}")
    entries = config.get("checkpoints", [])
    if len(entries) != 3:
        raise ValueError("Configure exactly three matching SR/scsf/pd_scsf reference checkpoints")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows, params, manifests, errors = [], [], [], []
    identities, methods = set(), set()
    for entry in entries:
        model, args, _ = load_checkpoint(entry["checkpoint"], "cpu", config["data_dir"])
        data = verified_data(entry, args, config["data_dir"])
        identities.add((args.dataset, args.seed, data["sha256"]))
        methods.add(args.method)
        if args.method not in {"sr", "scsf", "pd_scsf"}:
            raise ValueError("Unsupported E16 method")
        params.append({"method": args.method, "total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "backbone": sum(p.numel() for p in model.backbone.parameters()),
                       "confidence_head": sum(p.numel() for p in getattr(model, "calibrator", torch.nn.Identity()).parameters()),
                       "auxiliary_heads": sum(p.numel() for p in getattr(model, "heads", torch.nn.Identity()).parameters())})
        manifests.append({"method": args.method, "dataset": args.dataset, "seed": args.seed,
                          "path": entry["checkpoint"], "sha256": file_sha256(entry["checkpoint"]),
                          "data_sha256": data["sha256"]})
        del model
    if len(identities) != 1 or methods != {"sr", "scsf", "pd_scsf"}:
        raise ValueError("Benchmarks require matching dataset/seed/cache across all three methods")
    environment = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                   "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "timing": "CUDA events; synchronized per iteration", "precision": "FP32; AMP off"}
    atomic_json(out / "run_environment.json", environment)
    atomic_json(out / "benchmark_config.json", config)
    write_csv(out / "checkpoint_manifest.csv", manifests)
    write_csv(out / "parameter_counts.csv", params)
    for entry in entries:
        for kind in ("inference", "train_step"):
            modes = ("primary_only", "with_aux_diagnostics") if kind == "inference" else ("training",)
            for mode in modes:
                for batch in ((1, 8, 32, 64) if kind == "inference" else (64,)):
                    for round_id in range(3):
                        try:
                            torch.manual_seed(0)
                            torch.cuda.empty_cache()
                            model, args, _ = load_checkpoint(entry["checkpoint"], "cuda", config["data_dir"])
                            if hasattr(model, "emit_aux_confidence"):
                                model.emit_aux_confidence = mode != "primary_only"
                            x = torch.randn(batch, 3, 224, 224, device="cuda")
                            y = torch.arange(batch, device="cuda") % model.num_classes
                            if kind == "train_step":
                                model.train()
                                opts = optimizers(model, args)
                                def step():
                                    for opt in opts:
                                        opt.zero_grad(set_to_none=True)
                                    output = model(x)
                                    loss = model.training_loss(output, y, None, 30, args)
                                    if not bool(torch.isfinite(loss)):
                                        raise FloatingPointError("Nonfinite benchmark loss")
                                    loss.backward()
                                    for opt in opts:
                                        opt.step()
                            else:
                                model.eval()
                                opts = []
                                @torch.inference_mode()
                                def step():
                                    model(x)
                            warmup, repetitions = (20, 100) if kind == "inference" else (10, 30)
                            for _ in range(warmup):
                                step()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            for repetition in range(repetitions):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record(); step(); end.record(); end.synchronize()
                                elapsed = start.elapsed_time(end)
                                rows.append({"method": args.method, "kind": kind, "mode": mode, "batch": batch,
                                             "round": round_id, "iteration": repetition, "milliseconds": elapsed,
                                             "images_per_second": 1000 * batch / elapsed,
                                             "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
                        except torch.cuda.OutOfMemoryError as exc:
                            errors.append({"checkpoint": entry["checkpoint"], "kind": kind, "mode": mode,
                                           "batch": batch, "round": round_id, "error": str(exc)})
                        finally:
                            if "model" in locals():
                                del model
                            if "opts" in locals():
                                del opts
                            # Drop the closure so prior model/optimizer tensors do not survive into setup.
                            if "step" in locals():
                                del step
                            if "x" in locals():
                                del x, y
                            torch.cuda.empty_cache()
    for kind in ("inference", "train_step"):
        subset = [r for r in rows if r["kind"] == kind]
        write_csv(out / f"{kind}_raw_times.csv", subset)
        groups = {}
        for row in subset:
            groups.setdefault((row["method"], row["mode"], row["batch"], row["round"]), []).append(row)
        summaries = []
        for (method, mode, batch, round_id), measurements in groups.items():
            times = [r["milliseconds"] for r in measurements]
            summaries.append({"method": method, "mode": mode, "batch": batch, "round": round_id,
                              "n": len(times), "median_ms": statistics.median(times),
                              "p90_ms": float(np.percentile(times, 90)),
                              "peak_allocated_mib": max(r["peak_allocated_mib"] for r in measurements),
                              "peak_reserved_mib": max(r["peak_reserved_mib"] for r in measurements)})
        write_csv(out / f"{kind}_summary.csv", summaries)
    write_csv(out / "errors.csv", errors)
    loader_rows = []
    for dataset_name in ("ham10000", "pcam"):
        directory = Path(config["data_dir"]) / "preprocessed" / f"{dataset_name}_224"
        if not directory.exists():
            loader_rows.append({"dataset": dataset_name, "status": "missing_cache"})
            continue
        info = audit_cache(config["data_dir"], dataset_name)["identity"]
        spec = DatasetSpec(dataset_name, len(info["classes"]["train"]), 224, tuple(info["mean"]), tuple(info["std"]))
        transform, _ = _transforms(spec)
        dataset = ImageFolder(directory / "train", transform)
        for round_id in range(3):
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True,
                                persistent_workers=True, prefetch_factor=4)
            iterator = iter(loader)
            for _ in range(min(5, len(loader) - 1)):
                next(iterator)
            start = time.perf_counter(); count = 0; batches = 0
            for x, _ in iterator:
                count += len(x); batches += 1
                if batches >= 50:
                    break
            seconds = time.perf_counter() - start
            loader_rows.append({"dataset": dataset_name, "round": round_id, "status": "measured",
                                "n": count, "seconds": seconds, "images_per_second": count / seconds,
                                "train_n": len(dataset), "includes_gpu_compute": False})
            del iterator, loader
    write_csv(out / "pcam_loader_throughput.csv", loader_rows)
    estimates = []
    pcam = [r for r in loader_rows if r["dataset"] == "pcam" and r["status"] == "measured"]
    if pcam:
        loader_seconds = statistics.median(r["train_n"] / r["images_per_second"] for r in pcam)
        for method in methods:
            times = [r["milliseconds"] for r in rows if r["kind"] == "train_step" and r["method"] == method]
            if times:
                compute = math.ceil(pcam[0]["train_n"] / 64) * statistics.median(times) / 1000
                estimates.append({"method": method, "estimated_100_training_epoch_seconds": 100 * max(compute, loader_seconds),
                                  "status": "projection_from_fixed_shape_compute_and_real_loader",
                                  "excludes": "validation/checkpoint/initialization; not observed PCam full-run timing"})
    write_csv(out / "projected_pcam_cost.csv", estimates)
    if errors:
        raise RuntimeError("Some benchmark cases ran out of memory; inspect errors.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    benchmark(load_config(args.config))


if __name__ == "__main__":
    main()
