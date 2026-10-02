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


DEFAULT_METHODS = ["ccl_sc", "residual_head", "dp_head", "spatial_head"]


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
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--scsf-multi-trial-scorers",
        nargs="+",
        default=None,
        choices=["meta", "sr", "meta_sr_product", "meta_sr_blend", "geometric", "meta_agreement", "min_sr_meta", "margin", "energy", "doctor"],
        help="Optional SCSF scorer list to evaluate post-hoc from the same checkpoint.",
    )
    parser.add_argument("--multi-trial-checkpoint", default="last", choices=["best", "last"])
    parser.add_argument("--multi-trial-seeds", type=int, nargs="+", default=[10, 42, 123])
    parser.add_argument("--multi-trial-val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-multi-trial", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for ResNet/DenseNet")
    # --- Proposed method hyperparameters ---
    parser.add_argument("--agree-weight", type=float, default=1.0, help="BCE agreement loss weight for proposed methods")
    parser.add_argument("--min-agree-weight", type=float, default=1e-4, help="Minimum BCE weight after cosine decay")
    parser.add_argument("--dp-proj-damping", type=float, default=1e-2, help="Damping for dp_head null-space projection")
    parser.add_argument("--spatial-d-head", type=int, default=128, help="Attention dim for spatial_head")
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


def scsf_variant_for_scorer(variant: str, scorer: str) -> str:
    parts = variant.split("__")
    replaced = False
    for idx, part in enumerate(parts):
        if part.startswith("scorer-"):
            parts[idx] = f"scorer-{scorer}"
            replaced = True
            break
    if not replaced:
        parts.append(f"scorer-{scorer}")
    return "__".join(parts)


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


def train_command(args, run_root: Path, dataset: str, method: str) -> tuple[list[str], Path, str]:
    variant = scsf_variant(args) if method == "scsf" else "default"
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
        str(args.seed),
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
    checkpoint_dir = run_root / "checkpoints" / dataset / args.arch / method / variant / str(args.seed)
    return cmd, checkpoint_dir, variant


def eval_command(args, checkpoint: Path, output_dir: Path, method: str) -> list[str]:
    cmd = [
        sys.executable,
        "run_multi_trial_eval.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *[str(seed) for seed in args.multi_trial_seeds],
        "--val-fraction",
        str(args.multi_trial_val_fraction),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.workers),
    ]
    if args.download:
        cmd.append("--download")
    if method == "scsf":
        cmd.extend(["--scsf-scorer", args.scsf_scorer, "--scsf-sr-alpha", str(args.scsf_sr_alpha)])
    add_if_present(cmd, "--gpu", args.gpu)
    return cmd


def eval_commands_for_method(args, run_root: Path, dataset: str, method: str, variant: str, checkpoint: Path) -> list[tuple[list[str], Path, str]]:
    scorers = args.scsf_multi_trial_scorers if method == "scsf" and args.scsf_multi_trial_scorers else [None]
    commands = []
    for scorer in scorers:
        eval_variant = scsf_variant_for_scorer(variant, scorer) if scorer is not None else variant
        output_dir = run_root / "metrics" / dataset / args.arch / method / eval_variant / f"{args.multi_trial_checkpoint}_multi_trial"
        cmd = eval_command(args, checkpoint, output_dir, method)
        if scorer is not None:
            cmd = [token for token in cmd if token not in ["--scsf-scorer", args.scsf_scorer]]
            cmd.extend(["--scsf-scorer", scorer])
        commands.append((cmd, output_dir, eval_variant))
    return commands


def aggregate_tables(run_root: Path):
    allowed_methods = None
    args_path = run_root / "config" / "args.json"
    if args_path.exists():
        try:
            allowed_methods = set(json.loads(args_path.read_text(encoding="utf-8")).get("methods", []))
        except json.JSONDecodeError:
            allowed_methods = None

    rows = []
    for summary_path in sorted((run_root / "metrics").glob("*/*/*/*/*_multi_trial/multi_trial_summary.csv")):
        parts = summary_path.relative_to(run_root / "metrics").parts
        if allowed_methods is not None and parts[2] not in allowed_methods:
            continue
        with summary_path.open(newline="") as f:
            row = next(csv.DictReader(f))
        row.update({"dataset": parts[0], "arch": parts[1], "method": parts[2], "variant": parts[3], "checkpoint_eval": parts[4]})
        rows.append(row)
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (tables_dir / "main_results_mean_std.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=union_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)

    rc_rows = []
    for curve_path in sorted((run_root / "metrics").glob("*/*/*/*/*_multi_trial/multi_trial_curves.csv")):
        parts = curve_path.relative_to(run_root / "metrics").parts
        if allowed_methods is not None and parts[2] not in allowed_methods:
            continue
        with curve_path.open(newline="") as f:
            for row in csv.DictReader(f):
                row.update({"dataset": parts[0], "arch": parts[1], "method": parts[2], "variant": parts[3], "checkpoint_eval": parts[4]})
                rc_rows.append(row)
    if rc_rows:
        with (tables_dir / "risk_coverage_table.csv").open("w", newline="") as f:
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
            cmd, checkpoint_dir, variant = train_command(args, run_root, entry.slug, method)
            checkpoint = checkpoint_dir / f"{args.multi_trial_checkpoint}.pt"
            eval_commands = [] if args.skip_multi_trial else eval_commands_for_method(args, run_root, entry.slug, method, variant, checkpoint)
            summary_paths = [output_dir / "multi_trial_summary.csv" for _, output_dir, _ in eval_commands]
            if args.skip_existing and checkpoint.exists() and (args.skip_multi_trial or all(path.exists() for path in summary_paths)):
                log_skip(f"SKIP existing {entry.slug}/{method}/{variant}", commands_file)
                continue
            run_command(cmd, commands_file, args.dry_run)
            if args.skip_multi_trial:
                continue
            for eval_cmd, output_dir, eval_variant in eval_commands:
                summary_path = output_dir / "multi_trial_summary.csv"
                if args.skip_existing and summary_path.exists():
                    log_skip(f"SKIP existing multi-trial {entry.slug}/{method}/{eval_variant}", commands_file)
                    continue
                run_command(eval_cmd, commands_file, args.dry_run)

    if not args.dry_run:
        aggregate_tables(run_root)
    print(f"Paper medical suite outputs: {run_root}")


if __name__ == "__main__":
    main()
