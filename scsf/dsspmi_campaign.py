"""New-only DSSPMI experiment matrix and server launcher."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import fcntl
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .dsspmi_contract import atomic_json, audit_cache, digest, file_sha256, require_contract

ROOT = Path(__file__).resolve().parents[1]
CORE3 = ("covid_qu_ex", "ham10000", "chest_ct_scan")
CORE4 = ("covid_qu_ex", "ham10000", "aptos2019", "chest_ct_scan")
SCREEN = ("covid_qu_ex", "chest_ct_scan")
FULL = "l23-tgtdepth-b0.5-d128__w0.1r5__detach-none"
LAYER3 = "l3-tgtdepth-b0.5-d128__w0.1r5__detach-none"
FINAL = "l23-tgtfinal-b0-d128__w0.1r5__detach-none"


@dataclass(frozen=True)
class Variant:
    experiment: str
    depths: str = "layer2+layer3"
    aggregation: str = "geometric"
    beta: float = 0.5
    weight: float = 0.5
    schedule: str = "legacy"

    @property
    def name(self):
        site = "l2" if self.depths == "layer2" else "l23"
        head = f"{site}-tgtdepth-b{self.beta:g}-d128"
        if self.aggregation != "geometric":
            head += f"-agg-{self.aggregation}"
        if self.weight != 0.5:
            head += f"-dw2-{self.weight:g}"
        schedule = "w0.1r5" if self.schedule == "legacy" else f"sched-{self.schedule}-w0.1"
        return f"{head}__{schedule}__detach-none"

    @property
    def datasets(self):
        return CORE4 if self.experiment in {"E05", "E06"} else CORE3

    def flags(self):
        return ["--pd-depths", self.depths, "--pd-betas", str(self.beta),
                "--pd-target-aggregations", self.aggregation,
                "--pd-depth-weight-layer2-values", str(self.weight),
                "--pd-aux-gradient-schedules", self.schedule]


VARIANTS = (Variant("E05", depths="layer2"), Variant("E06", aggregation="arithmetic"),
            *(Variant("E07B", beta=b) for b in (0.25, 0.75, 1.0)),
            *(Variant("E07W", weight=w) for w in (0.25, 0.75)),
            Variant("E09", schedule="step"), Variant("E09", schedule="early"))
ALLOWED = {"E05": {"pd_depths"}, "E06": {"pd_target_aggregation"},
           "E07B": {"pd_beta"}, "E07W": {"pd_depth_weight_layer2"},
           "E09": {"pd_aux_gradient_schedule"}}


def matrix(datasets=None, seeds=(0, 1, 2)):
    return [(v, dataset, seed) for v in VARIANTS
            for dataset in (datasets if datasets is not None else v.datasets) for seed in seeds]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def suite_flags(args, job, evaluate=False):
    variant, dataset, seed = job
    flags = ["--results-root", str(args.results_root), "--run-id", args.run_id,
             "--data-dir", str(args.data_dir), "--datasets-file", str(ROOT / "datasets_thesis.md"),
             "--no-download", "--arch", "resnet50", "--methods", "pd_scsf",
             "--datasets", dataset, "--seeds", str(seed), "--epochs", "100", "--pretrain", "20",
             "--batch-size", "64", "--eval-batch-size", "128", "--workers", "4", "--eval-every", "1",
             "--medical-split-seed", "42", "--lr", "0.01", "--milestones", "40", "70", "90",
             "--lr-gamma", "0.1", "--meta-lr", "0.001", "--meta-loss", "weighted_nll",
             "--error-weight", "1", "--hidden-dim", "256", "--min-meta-weight", "0.0001",
             "--scsf-feature-spec", "mid+late+logits", "--scsf-meta-target", "tcp", "--scsf-scorer", "meta",
             "--ds-head-dim", "128", "--ds-aux-weight", "0.1", "--ds-aux-ramp", "5",
             "--pd-targets", "depth", "--pd-head-types", "cosine", "--ds-detach-modes", "none",
             "--skip-existing", "--strict-run-contract", *variant.flags()]
    if args.gpu is not None:
        flags += ["--gpu", args.gpu]
    if evaluate:
        flags += ["--evaluate-last-only", "--test-release-lock", str(args.run_root / "config/test_release.json")]
    else:
        flags += ["--defer-test-evaluation"]
    return flags


def checkpoint_dir(args, job):
    v, dataset, seed = job
    return args.run_root / "checkpoints" / dataset / "resnet50/pd_scsf" / v.name / str(seed)


def resolved_args(args, job, evaluate=False):
    # Exercise the same Python parsers/generator used by the shell wrapper.
    import run_paper_medical_suite as suite
    import run_experiment as experiment
    saved = sys.argv
    try:
        flags = suite_flags(args, job, evaluate)
        flags.remove("--no-download")  # shell consumes this switch
        sys.argv = ["suite", "--pretrained", *flags]
        options = suite.parse_args()
        variants = suite.method_variants(options, "pd_scsf")
        if len(variants) != 1 or variants[0][0] != job[0].name:
            raise ValueError("Shell/Python variant identity mismatch")
        cmd, _ = suite.train_command(options, args.run_root, job[1], "pd_scsf", *variants[0], job[2])
        sys.argv = cmd[1:]
        effective = experiment.parse_args()
        effective.ds_class_weights = None
        return effective
    finally:
        sys.argv = saved


def validate_protocol(effective):
    template = json.loads((ROOT / "configs/dsspmi_new_runs_lock.yaml").read_text())
    fixed = {k: v for k, v in template["protocol"].items() if k not in {"primary_score", "checkpoint"}}
    fixed.update(method="pd_scsf", scsf_scorer="meta", scsf_meta_target="tcp", meta_loss="weighted_nll",
                 scsf_feature_spec="mid+late+logits", hidden_dim=256, ds_head_dim=128,
                 ds_aux_weight=0.1, ds_aux_ramp=5, ds_detach="none", error_weight=1.0,
                 meta_weight_mode="cosine", init_meta_weight=1.0, min_meta_weight=0.0001)
    mismatches = {k: (getattr(effective, k, None), v) for k, v in fixed.items()
                  if getattr(effective, k, None) != v}
    if mismatches or effective.smoke_train_samples or effective.smoke_eval_samples:
        raise ValueError(f"Resolved command violates the locked production protocol: {mismatches}")


def expected_contract(args, job, evaluate=False):
    from .dsspmi_contract import run_contract
    effective = resolved_args(args, job, evaluate)
    validate_protocol(effective)
    data = audit_cache(args.data_dir, job[1])
    return run_contract(effective, data)


def reference_check(args, job):
    """Require checkpoint-bound historic data evidence, not inferred current-cache identity."""
    import torch
    from .dsspmi_contract import normalized_config
    variant, dataset, seed = job
    entries = json.loads(args.reference_config.read_text()).get("references", [])
    required = [FULL] + ([LAYER3] if variant.experiment == "E05" else [FINAL] if variant.experiment == "E07B" else [])
    expected = expected_contract(args, job)
    wanted = expected["effective_config"]
    for name in required:
        matches = [r for r in entries if r.get("dataset") == dataset and r.get("seed") == seed
                   and r.get("method") == "pd_scsf" and r.get("variant") == name]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous historic reference: {dataset}/{name}/{seed}")
        entry = matches[0]
        checkpoint_path = Path(entry["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_type") != "last" or checkpoint.get("epoch") != 100:
            raise ValueError("Reference is not epoch-100 last checkpoint")
        if expected.get("num_classes") is not None and checkpoint.get("num_classes") != expected["num_classes"]:
            raise ValueError("Reference classifier class count does not match the original cache")
        evidence = json.loads(Path(entry["provenance"]).read_text())
        if evidence.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise ValueError("Historic provenance does not bind this checkpoint")
        if evidence.get("data_sha256") != expected["data_sha256"]:
            raise ValueError("UNVERIFIED_SPLIT: historic data identity absent or mismatched")
        old = normalized_config(checkpoint["args"])
        if name == FINAL:
            old.update(pd_target="depth", pd_beta=0.0)
        # Compare the whole effective namespace, defaulting only fields added by this implementation.
        old.setdefault("ds_class_weights", None)
        for key in ("ds_class_weights",):
            old.setdefault(key, wanted.get(key))
        differences = {k for k in set(old) | set(wanted) if old.get(k) != wanted.get(k)}
        if differences - ALLOWED[variant.experiment]:
            raise ValueError(f"Reference configuration differs: {sorted(differences - ALLOWED[variant.experiment])}")
    return expected


@contextmanager
def campaign_lease(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another campaign launcher is active under this run ID") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def clean_source():
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Production requires a committed, clean checkout (use an isolated server checkout)")


def selected(args):
    jobs = matrix(args.datasets, args.seeds)
    if args.experiment:
        jobs = [j for j in jobs if j[0].experiment == args.experiment]
    if args.day:
        chosen = VARIANTS[args.day - 1]
        jobs = [j for j in jobs if j[0] == chosen]
    return jobs


def freeze_matrix(args):
    path = args.run_root / "config/new_experiment_matrix.yaml"
    template = json.loads((ROOT / "configs/dsspmi_new_runs_lock.yaml").read_text())
    payload = {"schema": 1, "protocol": template["protocol"], "jobs": [{"variant": v.name, "dataset": d, "seed": s}
                                        for v, d, s in matrix(args.datasets, args.seeds)]}
    payload["sha256"] = digest(payload)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Existing campaign matrix differs; use a distinct run ID")
    atomic_json(path, payload)  # JSON is also valid YAML 1.2.
    environment_path = args.run_root / "config/packages.txt"
    if not environment_path.exists():
        environment_path.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    return payload


def job_contract(args, job):
    # Simple shell workflow: validate each new run, without requiring old reference records.
    if getattr(args, "simple", False):
        return expected_contract(args, job)
    return reference_check(args, job)


def run_jobs(args, jobs, evaluate=False):
    if not getattr(args, "simple", False):
        clean_source()
    lock = freeze_matrix(args)
    if not evaluate and (args.run_root / "config/test_release.json").exists():
        raise ValueError("Test release is frozen; no further training under this run ID")
    failures = []
    contract_rows = []
    for job in jobs:
        v, dataset, seed = job
        try:
            expected = job_contract(args, job)
            contract_rows.append({"experiment": v.experiment, "variant": v.name, "dataset": dataset,
                                  "seed": seed, "status": "new_run_verified" if getattr(args, "simple", False) else "matched",
                                  "reason": "Historical compatibility not certified" if getattr(args, "simple", False) else ""})
            if evaluate:
                marker = json.loads((checkpoint_dir(args, job) / "trained_complete.json").read_text())
                require_contract(marker.get("contract"), expected)
            flags = suite_flags(args, job, evaluate)
            command = ["bash", str(ROOT / "run_medical_suite.sh"), *flags]
            logs = args.run_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{v.name}_{dataset}_{seed}.log").open("a") as handle:
                handle.write(shlex.join(command) + "\n"); handle.flush()
                subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        except Exception as exc:
            failures.append({"experiment": v.experiment, "variant": v.name, "dataset": dataset,
                             "seed": seed, "status": "blocked_or_failed", "reason": str(exc)})
            print(f"BLOCKED/FAILED {dataset}/{v.name}/{seed}: {exc}", file=sys.stderr)
    contract_path = args.run_root / "tables/comparison_contract.csv"
    previous = []
    if contract_path.exists():
        with contract_path.open() as handle:
            previous = list(csv.DictReader(handle))
    by_job = {(r["variant"], r["dataset"], str(r["seed"])): r for r in previous + contract_rows + failures}
    write_csv(contract_path, list(by_job.values()))
    if failures:
        raise SystemExit(1)


def release_tests(args, jobs):
    if not getattr(args, "simple", False):
        clean_source()
    frozen = freeze_matrix(args)
    full = matrix(args.datasets, args.seeds)
    if len(jobs) != len(full) and not args.release_partial:
        raise ValueError("Partial test release requires --release-partial and a reason")
    checkpoints, omissions = {}, []
    for job in full:
        path = checkpoint_dir(args, job)
        try:
            expected = job_contract(args, job)
            marker = json.loads((path / "trained_complete.json").read_text())
            require_contract(marker.get("contract"), expected)
            sha = file_sha256(path / "last.pt")
            if marker["checkpoint_sha256"] != sha:
                raise ValueError("Checkpoint changed after training")
            checkpoints[expected["sha256"]] = sha
        except Exception as exc:
            if not args.release_partial:
                raise ValueError(f"Cannot release incomplete matrix: {path}: {exc}") from exc
            omissions.append({"variant": job[0].name, "dataset": job[1], "seed": job[2], "reason": str(exc)})
    payload = {"matrix_sha256": frozen["sha256"], "checkpoints": checkpoints,
               "omissions": omissions, "partial_reason": args.release_partial}
    path = args.run_root / "config/test_release.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Existing test-release lock is immutable")
    atomic_json(path, payload)
    available = [j for j in jobs if expected_contract(args, j)["sha256"] in checkpoints]
    run_jobs(args, available, evaluate=True)


def status(args):
    rows = []
    for job in selected(args):
        path = checkpoint_dir(args, job)
        state, reason = "pending", ""
        try:
            if (path / "trained_complete.json").exists():
                expected = expected_contract(args, job)
                marker = json.loads((path / "trained_complete.json").read_text())
                require_contract(marker.get("contract"), expected)
                if marker.get("checkpoint_sha256") != file_sha256(path / "last.pt"):
                    raise ValueError("Checkpoint checksum mismatch")
                state = "trained"
                if (path / "complete.json").exists():
                    complete = json.loads((path / "complete.json").read_text())
                    require_contract(complete.get("contract"), expected)
                    metrics = args.run_root / "metrics" / job[1] / "resnet50/pd_scsf" / job[0].name / f"seed_{job[2]}" / "last"
                    if not all((metrics / f).is_file() and (metrics / f).stat().st_size for f in
                               ("summary.csv", "risk_coverage.csv", "per_sample.csv", "roc_curve.csv")):
                        raise ValueError("Incomplete official evaluation")
                    state = "evaluated"
        except Exception as exc:
            state, reason = "incompatible", str(exc)
        rows.append({"experiment": job[0].experiment, "variant": job[0].name, "dataset": job[1],
                     "seed": job[2], "status": state, "reason": reason})
    write_csv(args.run_root / "tables/new_runs_status.csv", rows)
    for state in ("pending", "trained", "evaluated", "incompatible"):
        print(f"{state}: {sum(r['status'] == state for r in rows)}/{len(rows)}")
    return rows


def aggregate(args):
    rows = status(args)
    results = []
    for row in rows:
        if row["status"] != "evaluated":
            continue
        base = args.run_root / "metrics" / row["dataset"] / "resnet50/pd_scsf" / row["variant"] / f"seed_{row['seed']}" / "last"
        trained_path = checkpoint_dir(args, next(j for j in selected(args)
                          if j[0].name == row["variant"] and j[1] == row["dataset"] and j[2] == row["seed"])) / "trained_complete.json"
        timing = json.loads(trained_path.read_text())
        timing = {k: timing[k] for k in ("train_seconds", "validation_seconds", "checkpoint_seconds", "total_wall_seconds")}
        with (base / "summary.csv").open() as f:
            results.append({**row, **next(csv.DictReader(f)), **timing})
    write_csv(args.run_root / "tables/new_runs_per_seed.csv", results)
    write_csv(args.run_root / "benchmarks/E16/new_runs_full_wall_clock.csv",
              [{k: row[k] for k in ("dataset", "variant", "seed", "train_seconds", "validation_seconds",
                                     "checkpoint_seconds", "total_wall_seconds")} for row in results])
    import statistics
    import math
    groups = {}
    for row in results:
        groups.setdefault((row["dataset"], row["variant"]), []).append(row)
    summaries = []
    for (dataset, variant), values in groups.items():
        out = {"dataset": dataset, "variant": variant, "n_seeds": len(values)}
        for key in ("accuracy", "aurc", "eaurc", "naurc", "auroc", "fpr_at_95_tpr"):
            numbers = [float(v[key]) for v in values if math.isfinite(float(v[key]))]
            out[key + "_n_valid"] = len(numbers)
            out[key + "_mean"] = statistics.mean(numbers) if numbers else float("nan")
            out[key + "_std"] = statistics.stdev(numbers) if len(numbers) > 1 else float("nan")
        summaries.append(out)
    write_csv(args.run_root / "tables/new_runs_mean_std.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", help="status/next/1..9/all/aggregate/dry-run/smoke/screen/confirm/evaluate/probe/profile/preflight")
    parser.add_argument("--simple", action="store_true", help="Use the old shell workflow without historical-reference setup")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "dsspmi_new_ablations_v4"))
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("RESULTS_ROOT", ROOT / "results/paper/medical_suite")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", ROOT / "data")))
    parser.add_argument("--reference-config", type=Path, default=ROOT / "configs/dsspmi_reference_contract.json")
    parser.add_argument("--datasets", nargs="+", default=os.environ.get("DSSPMI_DATASETS", "").split() or None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[int(x) for x in os.environ.get("DSSPMI_SEEDS", "0 1 2").split()])
    parser.add_argument("--experiment", choices=sorted(ALLOWED))
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--release-partial", metavar="REASON")
    args = parser.parse_args()
    if args.datasets and not set(args.datasets) <= set(CORE4):
        parser.error("New training is restricted to the document's four primary datasets")
    if Path(args.run_id).name != args.run_id or args.run_id in {"round3", "thesis", "r101", "R50_pcam"}:
        parser.error("Use a new single-directory campaign run ID, not a historic run name")
    if len(set(args.seeds)) != len(args.seeds) or (args.datasets and len(set(args.datasets)) != len(args.datasets)):
        parser.error("Duplicate datasets/seeds would collide in the run matrix")
    args.run_root = args.results_root / args.run_id
    args.day = int(args.command) if args.command.isdigit() else None
    if args.day and args.day not in range(1, 10):
        parser.error("Day must be 1..9")
    jobs = selected(args)
    screen = [j for j in jobs if j[1] in SCREEN and j[2] == 0]
    if args.command == "dry-run":
        for job in jobs:
            print(shlex.join(["bash", str(ROOT / "run_medical_suite.sh"), *suite_flags(args, job)]))
        print(f"# {len(jobs)} jobs; {len(screen)} screen jobs")
    elif args.command == "status":
        status(args)
    elif args.command in {"aggregate", "aggregate-new"}:
        aggregate(args)
    elif args.command == "preflight":
        if not args.simple:
            clean_source()
        failures = []
        for job in jobs:
            try:
                job_contract(args, job)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise SystemExit("\n".join(failures))
        print(f"Matched: {len(jobs)} jobs")
    elif args.command == "screen":
        with campaign_lease(args.run_root):
            run_jobs(args, screen)
    elif args.command in {"confirm", "all"}:
        # Screen correctness must pass before spending time on confirmation.
        with campaign_lease(args.run_root):
            run_jobs(args, screen)
            run_jobs(args, [j for j in jobs if j not in screen])
    elif args.command == "next":
        rows = status(args)
        remaining = [j for j, r in zip(jobs, rows) if r["status"] not in {"trained", "evaluated"}]
        if remaining:
            v = remaining[0][0]
            with campaign_lease(args.run_root):
                run_jobs(args, [j for j in remaining if j[0] == v])
    elif args.day:
        with campaign_lease(args.run_root):
            run_jobs(args, jobs)
    elif args.command == "evaluate":
        with campaign_lease(args.run_root):
            release_tests(args, jobs)
    elif args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "test_pd_remaining_new", "test_dsspmi_contract", "test_dsspmi_probes"], cwd=ROOT, check=True)
    elif args.command in {"probe", "profile"}:
        if not args.config:
            parser.error("probe/profile requires --config")
        script = "train_pd_frozen_probes.py" if args.command == "probe" else "benchmark_pd_compute.py"
        subprocess.run([sys.executable, str(ROOT / "analysis" / script), "--config", str(args.config)], check=True)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
