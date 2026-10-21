#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from scsf.medical_registry import parse_datasets_to_run
from scsf.run_status import complete_runs, is_complete


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
    parser.add_argument("--datasets", nargs="+", default=None, help="Subset of dataset slugs from --datasets-file")
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
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild tables/ from finished runs without training")
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
    # --- Single-view deep supervision (mg_scsf, mp_ds, se_ds); each listed value becomes a variant directory ---
    parser.add_argument("--ds-detach-modes", nargs="+", default=["none"], choices=["none", "all", "new"],
                        help="none: full DS; all: All-DS-detached; new: detach only mg_scsf's margin branch "
                             "(skipped for mp_ds/se_ds, where it equals all)")
    parser.add_argument("--ds-sites", nargs="+", default=["layer3+layer4"], choices=["layer3", "layer4", "layer3+layer4"],
                        help="mp_ds/se_ds branch sites (mg_scsf's margin branch is always layer3)")
    parser.add_argument("--mg-losses", nargs="+", default=["hinge"], choices=["hinge", "ce"])
    parser.add_argument("--mp-prototypes", nargs="+", type=int, default=[3])
    parser.add_argument("--se-pools", nargs="+", default=["topk"], choices=["topk", "gap"])
    parser.add_argument("--ds-aux-weight", type=float, default=0.1)
    parser.add_argument("--ds-aux-ramp", type=int, default=5)
    parser.add_argument("--mg-margin", type=float, default=0.2)
    parser.add_argument("--mg-ce-scale", type=float, default=10.0)
    parser.add_argument("--mp-dim", type=int, default=128)
    parser.add_argument("--mp-temperature", type=float, default=0.1)
    parser.add_argument("--se-topk-fraction", type=float, default=0.2)
    # --- Round 2 DS (ro_ds, ag_ds, cm_ds) ---
    parser.add_argument("--ds-primary-score", default="msp", choices=["msp", "aux"])
    parser.add_argument("--ds-head-dim", type=int, default=128)
    parser.add_argument("--ro-rank-weights", nargs="+", type=float, default=[0.5],
                        help="ro_ds rank coefficients; include 0 for the ordinary-DS control")
    parser.add_argument("--ro-rank-margin", type=float, default=0.2)
    parser.add_argument("--ro-rank-temperature", type=float, default=0.1)
    parser.add_argument("--ag-align-modes", nargs="+", default=["on"], choices=["on", "off"],
                        help="ag_ds: projected/capped grads (on) or ordinary aux CE (off)")
    parser.add_argument("--ag-align-rho", type=float, default=0.25)
    parser.add_argument("--cm-complements", nargs="+", default=["targeted"], choices=["targeted", "off", "random"])
    parser.add_argument("--cm-topk-fraction", type=float, default=0.1)
    parser.add_argument("--cm-gate-prob", type=float, default=0.7)
    # --- Round 3 DS for the SCSF head (pd_scsf, ro_scsf); ro_scsf reuses --ds-sites/--ro-rank-weights ---
    parser.add_argument("--pd-depths", nargs="+", default=["layer2+layer3"], choices=["layer2+layer3", "layer3"])
    parser.add_argument("--pd-betas", nargs="+", type=float, default=[0.5])
    parser.add_argument("--pd-targets", nargs="+", default=["depth"], choices=["depth", "final"],
                        help="pd_scsf: depth-distilled TCP target, or the SCSF target with the same heads (control)")
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


def _git_output(*git_args: str) -> str:
    try:
        return subprocess.check_output(["git", *git_args], text=True, stderr=subprocess.STDOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"git {' '.join(git_args)} failed: {exc}\n"


def write_run_config(args, run_root: Path, entries):
    """Record this invocation under config/invocations/<timestamp>/ and append it to config/invocations.jsonl.

    Daily invocations share one run, so per-invocation records are kept instead of overwriting a single copy.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config_dir = run_root / "config" / "invocations" / stamp
    config_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_output("rev-parse", "HEAD").strip()
    with (run_root / "config" / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "invocation": stamp,
            "argv": sys.argv,
            "git_commit": commit,
            "methods": args.methods,
            "datasets": [entry.slug for entry in entries],
            "seeds": args.seeds,
        }) + "\n")
    (config_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
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
    (config_dir / "git_status.txt").write_text(_git_output("status", "--short"), encoding="utf-8")
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


def method_variants(args, method: str) -> list[tuple[str, list[str]]]:
    """(variant directory, variant-specific run_experiment flags) for every configuration of `method`."""
    if method == "scsf":
        return [(scsf_variant(args), [])]
    if method == "ccl_sc":
        return [(f"ccl-{name}", ["--ccl-variant", name]) for name in args.ccl_variants]
    if method in {"mg_scsf", "mp_ds", "se_ds", "ro_ds", "ag_ds", "cm_ds", "pd_scsf", "ro_scsf"}:
        # Directory = branch site, head setting, coefficient/ramp and gradient mode.
        schedule = f"w{args.ds_aux_weight:g}r{args.ds_aux_ramp}"
        site_tags = {"layer3": "l3", "layer4": "l4", "layer3+layer4": "l34", "layer2+layer3": "l23"}
        sites = {"mg_scsf": ["layer3"], "pd_scsf": args.pd_depths}.get(method, args.ds_sites)
        modes = [mode for mode in args.ds_detach_modes if mode != "new" or method in {"mg_scsf", "pd_scsf", "ro_scsf"}]
        if method == "mg_scsf":
            heads = [
                (f"{loss}-m{args.mg_margin:g}" if loss == "hinge" else f"ce-s{args.mg_ce_scale:g}", ["--mg-loss", loss])
                for loss in args.mg_losses
            ]
        elif method == "mp_ds":
            heads = [
                (f"k{k}-d{args.mp_dim}-t{args.mp_temperature:g}", ["--mp-prototypes", str(k)])
                for k in args.mp_prototypes
            ]
        elif method == "se_ds":
            heads = [
                (f"{pool}{args.se_topk_fraction:g}" if pool == "topk" else pool, ["--se-pool", pool])
                for pool in args.se_pools
            ]
        elif method == "pd_scsf":
            heads = [
                (f"tgt{target}-b{beta:g}-d{args.ds_head_dim}", ["--pd-target", target, "--pd-beta", str(beta)])
                for target in args.pd_targets
                for beta in (args.pd_betas if target == "depth" else [0.0])
            ]
        elif method in {"ro_ds", "ro_scsf"}:
            heads = [
                (f"rank{weight:g}-d{args.ds_head_dim}", ["--ro-rank-weight", str(weight)])
                for weight in args.ro_rank_weights
            ]
        elif method == "ag_ds":
            heads = [
                (
                    f"align{mode}-r{args.ag_align_rho:g}-d{args.ds_head_dim}",
                    ["--ag-align" if mode == "on" else "--no-ag-align"],
                )
                for mode in args.ag_align_modes
            ]
        else:
            heads = [
                (f"comp-{comp}-k{args.cm_topk_fraction:g}-d{args.ds_head_dim}", ["--cm-complement", comp])
                for comp in args.cm_complements
            ]
        primary_flags = (
            ["--ds-primary-score", args.ds_primary_score]
            if method in {"ro_ds", "ag_ds", "cm_ds"}
            else []
        )
        variants = []
        for site in sites:
            site_flags = {"mg_scsf": [], "pd_scsf": ["--pd-depths", site]}.get(method, ["--ds-site", site])
            for head, flags in heads:
                for mode in modes:
                    name = f"{site_tags[site]}-{head}__{schedule}__detach-{mode}"
                    if method in {"ro_ds", "ag_ds", "cm_ds"}:
                        name = f"{name}__score-{args.ds_primary_score}"
                    variants.append((name, [*flags, *site_flags, *primary_flags, "--ds-detach", mode]))
        return variants
    return [("default", [])]


def train_command(
    args, run_root: Path, dataset: str, method: str, variant: str, variant_flags: list[str], seed: int
) -> tuple[list[str], Path]:
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
    cmd.extend(variant_flags)
    if method == "ccl_sc":
        cmd.extend([
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
    if method in {"mg_scsf", "mp_ds", "se_ds", "ro_ds", "ag_ds", "cm_ds", "pd_scsf", "ro_scsf"}:
        cmd.extend([
            "--ds-aux-weight", str(args.ds_aux_weight),
            "--ds-aux-ramp", str(args.ds_aux_ramp),
            "--mg-margin", str(args.mg_margin),
            "--mg-ce-scale", str(args.mg_ce_scale),
            "--mp-dim", str(args.mp_dim),
            "--mp-temperature", str(args.mp_temperature),
            "--se-topk-fraction", str(args.se_topk_fraction),
            "--ds-head-dim", str(args.ds_head_dim),
            "--ro-rank-margin", str(args.ro_rank_margin),
            "--ro-rank-temperature", str(args.ro_rank_temperature),
            "--ag-align-rho", str(args.ag_align_rho),
            "--cm-topk-fraction", str(args.cm_topk_fraction),
            "--cm-gate-prob", str(args.cm_gate_prob),
        ])
    if method == "dg" and args.dg_reward is not None:
        cmd.extend(["--reward", str(args.dg_reward)])
    if method == "selectivenet":
        cmd.extend(["--target-coverage", str(args.sn_target_coverage)])
    if method in {"scsf", "mg_scsf", "pd_scsf", "ro_scsf"}:
        # SCSF-based DS methods inherit the SCSF configuration of the matched baseline.
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
    if not values:
        return float("nan"), float("nan")
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


def aggregate_tables(run_root: Path):
    """Per-seed results plus mean/std (ddof=1) across training seeds, all on the full test set.

    Covers every complete run under run_root, so tables accumulate across partial (daily) invocations.
    Undefined metrics (NaN, e.g. AUROC without errors) are left out of mean/std; when that happens the
    number of valid seeds is reported in <metric>_n_valid.
    """
    metrics_root = run_root / "metrics"
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    finished = list(complete_runs(run_root))
    seed_rows = []
    for dataset, arch, method, variant, seed in finished:
        summary_path = metrics_root / dataset / arch / method / variant / f"seed_{seed}" / "last" / "summary.csv"
        with summary_path.open(newline="") as f:
            row = next(csv.DictReader(f))
        row.update({"dataset": dataset, "arch": arch, "method": method, "variant": variant, "seed": seed})
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
            valid = [v for v in values if not math.isnan(v)]
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std(valid)
            if len(valid) < len(values):
                out[f"{metric}_n_valid"] = len(valid)
        summary_rows.append(out)
    with (tables_dir / "main_results_mean_std.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_fieldnames(summary_rows))
        writer.writeheader()
        writer.writerows(summary_rows)

    rc_groups: dict[tuple, dict[str, list[float]]] = {}
    for dataset, arch, method, variant, seed in finished:
        curve_path = metrics_root / dataset / arch / method / variant / f"seed_{seed}" / "last" / "risk_coverage.csv"
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
    if args.datasets:
        unknown = set(args.datasets) - {entry.slug for entry in entries}
        if unknown:
            raise SystemExit(f"--datasets not in {args.datasets_file}: {sorted(unknown)}")
        entries = [entry for entry in entries if entry.slug in args.datasets]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.results_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_tables(run_root)
        print(f"Tables rebuilt: {run_root / 'tables'}")
        return
    write_run_config(args, run_root, entries)
    commands_file = run_root / "config" / "commands.txt"
    if not args.skip_existing:
        commands_file.write_text("", encoding="utf-8")
    else:
        with commands_file.open("a", encoding="utf-8") as f:
            f.write("\n# Resuming with --skip-existing\n")

    for entry in entries:
        for method in args.methods:
            for variant, variant_flags in method_variants(args, method):
                for seed in args.seeds:
                    cmd, _ = train_command(args, run_root, entry.slug, method, variant, variant_flags, seed)
                    if args.skip_existing:
                        if is_complete(run_root, entry.slug, args.arch, method, variant, seed):
                            log_skip(f"SKIP complete {entry.slug}/{method}/{variant}/seed_{seed}", commands_file)
                            continue
                        # A finished last.pt of the same configuration is re-evaluated instead of retrained.
                        cmd.append("--reuse-last")
                    run_command(cmd, commands_file, args.dry_run)

    if not args.dry_run:
        aggregate_tables(run_root)
    print(f"Paper medical suite outputs: {run_root}")


if __name__ == "__main__":
    main()
