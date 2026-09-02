"""Deterministic run-manifest generator for the locked CIFAR gate matrix.

Writes one TSV per stage under ``<results_root>/manifests/`` with columns:

    priority  stage  dataset  backbone  method_name  mode  seed  run_dir  args

``args`` is the full space-separated ``key=value`` CLI override list for
``python -m scsf.train`` (and ``run_dir`` for ``scsf.evaluate``). Run names and
CLI args are produced by the exact config resolver so the manifest can never
drift from what the trainer computes. The scheduler executes rows top-to-bottom
and is idempotent; stages overlap on purpose (Stage C re-runs nothing because
the scheduler skips complete runs).

Recipes: the gate uses the frozen backbone-transfer track
(``recipe=backbone_transfer``) for every cell, per docs/EMPIRICAL_CONTRACT.md.
Paper-track cells (SCSF/CCL-SC on VGG16-BN) are emitted separately as the
reproduction sanity check (``recipe=paper``).

Usage (run on the execution host so torch is available)::

    python -m scripts.gen_manifest  --results_root results
"""

from __future__ import annotations

import argparse
import os
import sys

# keep this importable as `python scripts/gen_manifest.py` AND runnable from
# the repo root; add repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scsf.engine.config import resolve  # noqa: E402

SEEDS = (13, 17, 23, 29, 31)
B_BASELINE_SEEDS = (13, 17, 23)  # cheap screen (Stage B)
RECIPE = "backbone_transfer"

BASELINES = [
    ("ce", None),
    ("dg", None),
    ("selectivenet", None),
    ("sat", None),
    ("scsf", "posthoc"),
    ("scsf", "e2e"),
    ("ccl_sc", None),
]

METHODS = BASELINES + [
    ("sage_ds", None),
    ("depthfrag", None),
    ("riskflow", None),
]

B_SCREEN_METHODS = [
    ("sage_ds", None),
    ("depthfrag", None),
    ("riskflow", None),
    ("riskflow_frozen", None),
    ("depthfrag_frozen", None),
]

BACKBONES = ("resnet18", "vgg16_bn", "wideresnet28_10", "convnext_tiny", "deit_s")
DATASETS = ("cifar10", "cifar100")

# Skip rules: only skip if the ORIGINAL baseline paper published results for
# that exact backbone+dataset.  When uncertain, err on the side of running.
SKIP_RULES = {
    # Deep Gamblers (Chen et al. 2020): published ResNet-18 on CIFAR-10/100
    "dg": {"resnet18"},
    # SelectiveNet (Geifman & El-Yaniv 2019): published ResNet-18 on CIFAR-10/100
    "selectivenet": {"resnet18"},
    # SAT (Zhang et al. 2019): published ResNet-18 on CIFAR-10/100
    "sat": {"resnet18"},
    # CCL-SC published VGG16-BN results, but it remains a required *matched*
    # baseline for the new VGG16-BN gate.  Published numbers are only a sanity
    # check because our split/recipe/selection protocol differs.  ResNet-18 is
    # the sole paper-result skip.
    "ccl_sc": {"resnet18"},
}


def _skip(method: str, backbone: str) -> bool:
    """Return True if this baseline should be skipped for this backbone."""
    return backbone in SKIP_RULES.get(method, set())

# Stage D methods per the contract section 8 (each method's own ablation
# ladder), run on the two ablation cells.
SAGE_DS_ABLATIONS = [
    ("sage_ds_fixed_late", None),
    ("sage_ds_all_equal", None),
    ("sage_ds_learned_dense", None),
    ("sage_ds_sparse", None),
    ("sage_ds_ss0_1", None),
    ("sage_ds_ss0_3", None),
    ("sage_ds_ss1_0", None),
    ("sage_ds_ss3_0", None),
]
DEPTHFRAG_ABLATIONS = [
    ("depthfrag_terminal_margin", None),
    ("depthfrag_terminal", None),
    ("depthfrag_intermediate", None),
    ("depthfrag_raw", None),
    ("depthfrag", None),
    ("depthfrag_frozen", None),
    ("depthfrag_clip", None),
]
RISKFLOW_COMPARISONS = [
    ("riskflow_concat", None),
    ("riskflow_heads", None),
    ("riskflow_cum", None),
    ("riskflow_resid", None),
    ("riskflow", None),
    ("riskflow_frozen", None),
    ("riskflow_hard", None),
]


def _cli(method: str, mode, dataset: str, backbone: str, seed: int, recipe: str,
         results_root: str) -> tuple[str, str]:
    """Return (run_dir, args) exactly as the resolver names them."""
    overrides = {
        "dataset": dataset,
        "backbone": backbone,
        "method_name": method,
        "seed": seed,
        "recipe": recipe,
        "results_root": results_root,
        "train": {"device": "cuda", "seed": seed},
        "data": {"num_workers": 8},
    }
    if mode:
        overrides["method"] = {"mode": mode}
    cfg = resolve(overrides)
    run_name = cfg["run_name"]
    args = (
        f"dataset={dataset} backbone={backbone} method_name={method} seed={seed} "
        f"recipe={recipe} results_root={results_root} train.device=cuda "
        f"train.seed={seed} data.num_workers=8"
    )
    if mode:
        args += f" method.mode={mode}"
    return f"{results_root}/{run_name}", args


def _rows():
    pri = 0
    # Stage A: baselines on the two stage-A backbones, all five seeds.
    # Apply skip rules: only skip if the original paper published on that backbone.
    for dataset in DATASETS:
        for backbone in ("resnet18", "vgg16_bn"):
            for method, mode in BASELINES:
                if _skip(method, backbone):
                    continue
                for seed in SEEDS:
                    run_dir, args = _cli(method, mode, dataset, backbone, seed, RECIPE, "{RR}")
                    yield (pri, "A", dataset, backbone, method, mode or "", seed, run_dir, args)
                    pri += 1
    # Stage B: cheap method screen, seeds 13/17/23.
    for cell in (("cifar10", "resnet18"), ("cifar100", "vgg16_bn")):
        for method, mode in B_SCREEN_METHODS:
            for seed in B_BASELINE_SEEDS:
                run_dir, args = _cli(method, mode, cell[0], cell[1], seed, RECIPE, "{RR}")
                yield (pri, "B", cell[0], cell[1], method, mode or "", seed, run_dir, args)
                pri += 1
    # Stage C: full dataset x backbone x method x seed matrix.
    # Apply skip rules per backbone.
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for method, mode in METHODS:
                if _skip(method, backbone):
                    continue
                for seed in SEEDS:
                    run_dir, args = _cli(method, mode, dataset, backbone, seed, RECIPE, "{RR}")
                    yield (pri, "C", dataset, backbone, method, mode or "", seed, run_dir, args)
                    pri += 1
    # Stage D: ablation ladders on the two required cells (5 seeds where the
    # ladder config is a primary cell input, else 13/17/23 for the ladders).
    for dataset, backbone in (("cifar100", "vgg16_bn"), ("cifar100", "deit_s")):
        for method, mode in SAGE_DS_ABLATIONS + DEPTHFRAG_ABLATIONS + RISKFLOW_COMPARISONS:
            for seed in B_BASELINE_SEEDS:
                run_dir, args = _cli(method, mode, dataset, backbone, seed, RECIPE, "{RR}")
                yield (pri, "D", dataset, backbone, method, mode or "", seed, run_dir, args)
                pri += 1
    # Paper-track reproduction sanity check: SCSF/CCL-SC on VGG16-BN (all seeds).
    for dataset in DATASETS:
        for method, mode in BASELINES:
            if _skip(method, "vgg16_bn"):
                continue
            for seed in SEEDS:
                run_dir, args = _cli(method, mode, dataset, "vgg16_bn", seed, "paper", "{RR}")
                yield (pri, "P", dataset, "vgg16_bn", method, mode or "", seed, run_dir, args)
                pri += 1


def _gate_rows():
    """Unified gate manifest: VGG16-BN first (gate-critical), then ResNet-18."""
    pri = 0
    # Gate-critical: VGG16-BN baselines + methods, all seeds
    for dataset in DATASETS:
        for method, mode in BASELINES:
            if _skip(method, "vgg16_bn"):
                continue
            for seed in SEEDS:
                run_dir, args = _cli(method, mode, dataset, "vgg16_bn", seed, RECIPE, "{RR}")
                yield (pri, "A", dataset, "vgg16_bn", method, mode or "", seed, run_dir, args)
                pri += 1
        for method, mode in [("sage_ds", None), ("depthfrag", None), ("riskflow", None)]:
            for seed in SEEDS:
                run_dir, args = _cli(method, mode, dataset, "vgg16_bn", seed, RECIPE, "{RR}")
                yield (pri, "C", dataset, "vgg16_bn", method, mode or "", seed, run_dir, args)
                pri += 1
    # Historical: ResNet-18 baselines + methods, all seeds
    for dataset in DATASETS:
        for method, mode in BASELINES:
            if _skip(method, "resnet18"):
                continue
            for seed in SEEDS:
                run_dir, args = _cli(method, mode, dataset, "resnet18", seed, RECIPE, "{RR}")
                yield (pri, "A", dataset, "resnet18", method, mode or "", seed, run_dir, args)
                pri += 1
        for method, mode in [("sage_ds", None), ("depthfrag", None), ("riskflow", None)]:
            for seed in SEEDS:
                run_dir, args = _cli(method, mode, dataset, "resnet18", seed, RECIPE, "{RR}")
                yield (pri, "C", dataset, "resnet18", method, mode or "", seed, run_dir, args)
                pri += 1
    # Additional backbones: WRN, ConvNeXt, DeiT (all methods, all seeds)
    for backbone in ("wideresnet28_10", "convnext_tiny", "deit_s"):
        for dataset in DATASETS:
            for method, mode in METHODS:
                if _skip(method, backbone):
                    continue
                for seed in SEEDS:
                    run_dir, args = _cli(method, mode, dataset, backbone, seed, RECIPE, "{RR}")
                    yield (pri, "C", dataset, backbone, method, mode or "", seed, run_dir, args)
                    pri += 1
    # Ablations on cifar100/vgg16_bn
    for dataset, backbone in (("cifar100", "vgg16_bn"),):
        for method, mode in SAGE_DS_ABLATIONS + DEPTHFRAG_ABLATIONS + RISKFLOW_COMPARISONS:
            for seed in B_BASELINE_SEEDS:
                run_dir, args = _cli(method, mode, dataset, backbone, seed, RECIPE, "{RR}")
                yield (pri, "D", dataset, backbone, method, mode or "", seed, run_dir, args)
                pri += 1


def _ccl_vgg_supplement_rows():
    """The ten matched CCL-SC jobs omitted by the original live VGG queue."""
    pri = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            run_dir, args = _cli("ccl_sc", None, dataset, "vgg16_bn", seed,
                                 RECIPE, "{RR}")
            yield (pri, "A", dataset, "vgg16_bn", "ccl_sc", "", seed,
                   run_dir, args)
            pri += 1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gate", action="store_true",
                    help="Generate a single gate.tsv manifest (VGG16-BN first)")
    ap.add_argument("--ccl-vgg-supplement", action="store_true",
                    help="Generate only the ten matched VGG16-BN CCL-SC jobs")
    args = ap.parse_args(argv)
    rr = args.results_root
    out_dir = args.out_dir or os.path.join(rr, "manifests")
    os.makedirs(out_dir, exist_ok=True)
    import csv
    if args.gate or args.ccl_vgg_supplement:
        rows = []
        source = (_ccl_vgg_supplement_rows() if args.ccl_vgg_supplement
                  else _gate_rows())
        for row in source:
            run_dir, cli = row[7], row[8].replace("{RR}", rr)
            rows.append((row[0], row[1], row[2], row[3], row[4], row[5],
                         str(row[6]), run_dir.replace("{RR}", rr), cli))
        filename = "gate_vgg_ccl_sc.tsv" if args.ccl_vgg_supplement else "gate.tsv"
        path = os.path.join(out_dir, filename)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["priority", "stage", "dataset", "backbone", "method_name",
                        "mode", "seed", "run_dir", "args"])
            for r in rows:
                w.writerow(r)
        print(f"wrote {len(rows)} runs to {path}")
        by_bb = {}
        for r in rows:
            by_bb[r[3]] = by_bb.get(r[3], 0) + 1
        print("  by backbone: " + ", ".join(f"{k}={v}" for k, v in sorted(by_bb.items())))
        return
    per_stage = {}
    n = 0
    for row in _rows():
        run_dir, cli = row[7], row[8].replace("{RR}", rr)
        rows = per_stage.setdefault(row[1], [])
        rows.append((row[0], row[1], row[2], row[3], row[4], row[5],
                     str(row[6]), run_dir.replace("{RR}", rr), cli))
        n += 1
    for stage, rows in sorted(per_stage.items()):
        with open(os.path.join(out_dir, f"stage_{stage}.tsv"), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["priority", "stage", "dataset", "backbone", "method_name",
                        "mode", "seed", "run_dir", "args"])
            for r in rows:
                w.writerow(r)
    print(f"wrote {n} runs to {out_dir}: " +
          ", ".join(f"stage_{s}={len(per_stage[s])}" for s in sorted(per_stage)))
    # summary table
    print("\nrun counts by stage:")
    for s in sorted(per_stage):
        by_method = {}
        for r in per_stage[s]:
            by_method[r[4]] = by_method.get(r[4], 0) + 1
        print(f"  {s}: {len(per_stage[s])} runs  " +
              ";  ".join(f"{k}={v}" for k, v in sorted(by_method.items())))


if __name__ == "__main__":
    main()
