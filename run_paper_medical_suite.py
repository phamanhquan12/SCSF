#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from scsf.medical_registry import parse_datasets_to_run


DEFAULT_METHODS = ["sr", "ccl_sc", "sat", "dg", "selectivenet", "scsf", "dualaug"]


def union_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def parse_args():
    parser = argparse.ArgumentParser(description="Run the paper-ready SCSF medical dataset suite")
    parser.add_argument("--datasets-file", default="datasets_to_run.md")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-root", default="results/paper/medical_suite")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--arch", default="resnet50", choices=["vgg16_bn", "resnet18", "resnet50", "resnet101", "densenet121"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate validation metrics every N epochs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--milestones", type=int, nargs="+", default=[40, 70, 90])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1], help="Training seeds; mean/std are taken across these runs")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--medical-split-seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=None)
    parser.add_argument("--smoke-eval-samples", type=int, default=None)
    parser.add_argument("--scsf-feature-spec", default="mid+late+logits")
    parser.add_argument("--scsf-meta-target", default="tcp", choices=["tcp", "correctness"])
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--meta-loss", default="weighted_nll", choices=["weighted_nll", "weighted_nll_pairwise", "mse"])
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--min-meta-weight", type=float, default=1e-4)
    parser.add_argument("--scsf-scorer", default="meta", choices=["meta", "sr", "meta_sr_product", "meta_sr_blend", "geometric", "meta_agreement", "min_sr_meta", "margin", "energy", "doctor"])
    parser.add_argument("--scsf-sr-alpha", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for ResNet/DenseNet")
    # --- Proposed method hyperparameters ---
    parser.add_argument("--agree-weight", type=float, default=1.0, help="BCE agreement loss weight for proposed methods")
    parser.add_argument("--min-agree-weight", type=float, default=1e-4, help="Minimum BCE weight after cosine decay")
    parser.add_argument("--dp-proj-damping", type=float, default=1e-2, help="Damping for dp_head null-space projection")
    parser.add_argument("--spatial-d-head", type=int, default=128, help="Attention dim for spatial_head")
    # --- Baselines ---
    parser.add_argument("--dg-reward", type=float, default=None,
                        help="Deep Gamblers o; default lets run_experiment pick min(2.2, (1 + classes) / 2) per dataset")
    parser.add_argument("--sn-target-coverage", type=float, default=0.8)
    # --- CCL-SC (paper defaults for few-class datasets: CIFAR-10 / CelebA) ---
    parser.add_argument("--ccl-variants", nargs="+", default=["official"], choices=["official", "paper"],
                        help="CCL-SC implementations to run; each becomes its own variant directory")
    parser.add_argument("--ccl-weight", type=float, default=0.5)
    parser.add_argument("--ccl-temperature", type=float, default=0.1)
    parser.add_argument("--ccl-queue-size", type=int, default=300)
    parser.add_argument("--ccl-momentum", type=float, default=0.999)
    return parser.parse_args()


def add_if_present(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def scsf_variant(args) -> str:
    return (
        f"features-{args.scsf_feature_spec.replace('+', '_')}"
        f"__target-{args.scsf_meta_target}"
        f"__ew-{args.error_weight:g}"
        f"__mlr-{args.meta_lr:g}"
        f"__hd-{args.hidden_dim}"
        f"__minlam-{args.min_meta_weight:g}"
        f"__scorer-{args.scsf_scorer}"
    )


def write_run_config(args, run_root: Path, entries):
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.datasets_file, config_dir / "datasets_to_run.md")
    with (config_dir / "resolved_datasets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["display_name", "slug", "kaggle_slug", "url"])
        writer.writeheader()
        writer.writerows([entry.__dict__ for entry in entries])
    (config_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version}",
                f"platform={platform.platform()}",
                f"executable={sys.executable}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True)
    except subprocess.SubprocessError as exc:
        status = f"git status failed: {exc}\n"
    (config_dir / "git_status.txt").write_text(status, encoding="utf-8")
    (config_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def run_command(cmd: list[str], commands_file: Path, dry_run: bool):
    line = " ".join(cmd)
    print("\n" + "=" * 80)
    print(line)
    print("=" * 80)
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if not dry_run:
        subprocess.run(cmd, check=True)


def log_skip(message: str, commands_file: Path):
    print(message)
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(f"# {message}\n")


def method_variants(args, method: str) -> list[str]:
    if method == "scsf":
        return [scsf_variant(args)]
    if method == "ccl_sc":
        return [f"ccl-{name}" for name in args.ccl_variants]
    return ["default"]


def train_command(args, run_root: Path, dataset: str, method: str, variant: str, seed: int) -> tuple[list[str], Path]:
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--dataset",
        dataset,
        "--data-dir",
        args.data_dir,
        "--datasets-file",
        args.datasets_file,
        "--method",
        method,
        "--arch",
        args.arch,
        "--input-size",
        str(args.input_size),
        "--medical-split-seed",
        str(args.medical_split_seed),
        "--epochs",
        str(args.epochs),
        "--pretrain",
        str(args.pretrain),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--eval-every",
        str(args.eval_every),
        "--workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--milestones",
        *[str(milestone) for milestone in args.milestones],
        "--lr-gamma",
        str(args.lr_gamma),
        "--seed",
        str(seed),
        "--save-dir",
        str(run_root / "checkpoints"),
        "--metrics-dir",
        str(run_root / "metrics"),
        "--figures-dir",
        str(run_root / "figures"),
        "--output-layout",
        "paper",
        "--variant",
        variant,
    ]
    if args.download:
        cmd.append("--download")
    if args.force_preprocess:
        cmd.append("--force-preprocess")
    if args.amp:
        cmd.append("--amp")
    if args.pretrained:
        cmd.append("--pretrained")
    add_if_present(cmd, "--gpu", args.gpu)
    add_if_present(cmd, "--smoke-train-samples", args.smoke_train_samples)
    add_if_present(cmd, "--smoke-eval-samples", args.smoke_eval_samples)
    # Thread proposed-method hyperparameters
    if method in {"residual_head", "dp_head", "spatial_head"}:
        cmd.extend([
            "--agree-weight", str(args.agree_weight),
            "--min-agree-weight", str(args.min_agree_weight),
            "--dp-proj-damping", str(args.dp_proj_damping),
            "--spatial-d-head", str(args.spatial_d_head),
            "--hidden-dim", str(args.hidden_dim),
            "--meta-lr", str(args.meta_lr),
            "--pretrain", str(args.pretrain),
        ])
    if method == "ccl_sc":
        cmd.extend([
            "--ccl-variant", variant.removeprefix("ccl-"),
            "--ccl-weight", str(args.ccl_weight),
            "--ccl-temperature", str(args.ccl_temperature),
            "--ccl-queue-size", str(args.ccl_queue_size),
            "--ccl-momentum", str(args.ccl_momentum),
        ])
    if method == "dualaug":
        cmd.extend([
            "--scsf-feature-spec", args.scsf_feature_spec,
            "--hidden-dim", str(args.hidden_dim),
            "--meta-lr", str(args.meta_lr),
            "--min-meta-weight", str(args.min_meta_weight),
        ])
    if method == "dg" and args.dg_reward is not None:
        cmd.extend(["--reward", str(args.dg_reward)])
    if method == "selectivenet":
        cmd.extend(["--target-coverage", str(args.sn_target_coverage)])
    if method == "scsf":
        cmd.extend(
            [
                "--scsf-feature-spec",
                args.scsf_feature_spec,
                "--scsf-meta-target",
                args.scsf_meta_target,
                "--meta-lr",
                str(args.meta_lr),
                "--meta-loss",
                args.meta_loss,
                "--error-weight",
                str(args.error_weight),
                "--hidden-dim",
                str(args.hidden_dim),
                "--min-meta-weight",
                str(args.min_meta_weight),
                "--scsf-scorer",
                args.scsf_scorer,
                "--scsf-sr-alpha",
                str(args.scsf_sr_alpha),
            ]
        )
    checkpoint_dir = run_root / "checkpoints" / dataset / args.arch / method / variant / str(seed)
    return cmd, checkpoint_dir


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_tables(run_root: Path, methods: list[str], seeds: list[int]):
    """Per-seed results plus mean/std (ddof=1) across training seeds, all on the full test set."""
    metrics_root = run_root / "metrics"
    allowed_seeds = {f"seed_{seed}" for seed in seeds}
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for summary_path in sorted(metrics_root.glob("*/*/*/*/seed_*/last/summary.csv")):
        dataset, arch, method, variant, seed_dir = summary_path.relative_to(metrics_root).parts[:5]
        if method not in methods or seed_dir not in allowed_seeds:
            continue
        with summary_path.open(newline="") as f:
            row = next(csv.DictReader(f))
        row.update({"dataset": dataset, "arch": arch, "method": method, "variant": variant, "seed": seed_dir.removeprefix("seed_")})
        seed_rows.append(row)
    if not seed_rows:
        return
    with (tables_dir / "per_seed_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_fieldnames(seed_rows))
        writer.writeheader()
        writer.writerows(seed_rows)

    id_keys = ["dataset", "arch", "method", "variant"]
    skip_keys = set(id_keys) | {"seed", "checkpoint"}
    groups: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        groups.setdefault(tuple(row[k] for k in id_keys), []).append(row)
    summary_rows = []
    for key, rows in groups.items():
        out = dict(zip(id_keys, key))
        out["n_seeds"] = len(rows)
        out["seeds"] = " ".join(sorted(r["seed"] for r in rows))
        for metric in rows[0]:
            if metric in skip_keys:
                continue
            values = [to_float(r.get(metric)) for r in rows]
            if any(v is None for v in values):
                continue
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std(values)
        summary_rows.append(out)
    with (tables_dir / "main_results_mean_std.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_fieldnames(summary_rows))
        writer.writeheader()
        writer.writerows(summary_rows)

    rc_groups: dict[tuple, dict[str, list[float]]] = {}
    for curve_path in sorted(metrics_root.glob("*/*/*/*/seed_*/last/risk_coverage.csv")):
        dataset, arch, method, variant, seed_dir = curve_path.relative_to(metrics_root).parts[:5]
        if method not in methods or seed_dir not in allowed_seeds:
            continue
        with curve_path.open(newline="") as f:
            for row in csv.DictReader(f):
                bucket = rc_groups.setdefault((dataset, arch, method, variant, row["coverage"]), {"risk": [], "accuracy": []})
                bucket["risk"].append(float(row["risk"]))
                bucket["accuracy"].append(float(row["accuracy"]))
    rc_rows = []
    for (dataset, arch, method, variant, coverage), bucket in rc_groups.items():
        risk_mean, risk_std = mean_std(bucket["risk"])
        acc_mean, acc_std = mean_std(bucket["accuracy"])
        rc_rows.append({
            "dataset": dataset, "arch": arch, "method": method, "variant": variant, "coverage": coverage,
            "n_seeds": len(bucket["risk"]), "risk_mean": risk_mean, "risk_std": risk_std,
            "accuracy_mean": acc_mean, "accuracy_std": acc_std,
        })
    if rc_rows:
        with (tables_dir / "risk_coverage_mean_std.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=union_fieldnames(rc_rows))
            writer.writeheader()
            writer.writerows(rc_rows)


def main():
    args = parse_args()
    entries = parse_datasets_to_run(args.datasets_file)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.results_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_run_config(args, run_root, entries)
    commands_file = run_root / "config" / "commands.txt"
    if not args.skip_existing:
        commands_file.write_text("", encoding="utf-8")
    else:
        with commands_file.open("a", encoding="utf-8") as f:
            f.write("\n# Resuming with --skip-existing\n")

    for entry in entries:
        for method in args.methods:
            for variant in method_variants(args, method):
                for seed in args.seeds:
                    cmd, checkpoint_dir = train_command(args, run_root, entry.slug, method, variant, seed)
                    summary = run_root / "metrics" / entry.slug / args.arch / method / variant / f"seed_{seed}" / "last" / "summary.csv"
                    if args.skip_existing and (checkpoint_dir / "last.pt").exists() and summary.exists():
                        log_skip(f"SKIP existing {entry.slug}/{method}/{variant}/seed_{seed}", commands_file)
                        continue
                    run_command(cmd, commands_file, args.dry_run)

    if not args.dry_run:
        aggregate_tables(run_root, args.methods, args.seeds)
    print(f"Paper medical suite outputs: {run_root}")


if __name__ == "__main__":
    main()
