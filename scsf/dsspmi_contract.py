"""Immutable run contracts for the new DSSPMI campaign (standard-library only)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import platform
from pathlib import Path

PD_DEFAULTS = {
    "pd_target_aggregation": "geometric", "pd_depth_weight_layer2": 0.5,
    "pd_aux_gradient_schedule": "legacy", "calibrator_arch": "standard",
    "tcp_ema_momentum": 0.0, "meta_weight_warmup_fraction": 0.0,
    "meta_focal_gamma": 0.0,
    # Added after the first Round-3 runs; older checkpoints always used the cosine head.
    "pd_head_type": "cosine",
}
RUNTIME_KEYS = {
    "gpu", "save_dir", "metrics_dir", "figures_dir", "output_layout", "variant",
    "data_dir", "dataset_root", "datasets_file", "download", "force_preprocess",
    "prepare_only", "reuse_last", "defer_test_evaluation", "evaluate_last_only",
    "strict_run_contract", "test_release_lock",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


_HASH_CACHE = {}


def cached_file_sha256(path):
    """file_sha256 memoised per process; invalidated when size or mtime change."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = file_sha256(path)
    return _HASH_CACHE[key]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def normalized_config(args):
    values = dict(PD_DEFAULTS)
    values.update(vars(args) if hasattr(args, "__dict__") else args)
    return {k: v for k, v in values.items() if k not in RUNTIME_KEYS}


CODE_FILES = ("run_experiment.py", "run_paper_medical_suite.py", "run_medical_suite.sh")
CODE_EXCLUDE = {"dsspmi_campaign.py"}  # launcher only; it does not change training results


def code_sha256():
    """Hash of the training-relevant source contents (independent of git history)."""
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in CODE_FILES]
    files += sorted(p for p in (root / "scsf").glob("*.py") if p.name not in CODE_EXCLUDE)
    result = hashlib.sha256()
    for path in files:
        result.update(path.relative_to(root).as_posix().encode() + b"\0")
        result.update(path.read_bytes() + b"\0")
    return result.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return b""


def source_identity():
    """code_sha256 + environment are fingerprinted; commit/diff are recorded for provenance only.

    Fingerprinting the commit hash would invalidate every trained checkpoint after any
    unrelated commit (e.g. docs), so reuse is keyed on the code contents instead.
    """
    environment = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy"):
        module = sys.modules.get(name)
        if module is not None:
            environment[name] = str(module.__version__)
    torch = sys.modules.get("torch")
    if torch is not None:
        environment.update(cuda=torch.version.cuda,
                           tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                           tf32_cudnn=torch.backends.cudnn.allow_tf32,
                           cudnn_benchmark=torch.backends.cudnn.benchmark,
                           cudnn_deterministic=torch.backends.cudnn.deterministic)
    return {"code_sha256": code_sha256(), "environment": environment,
            "commit": _git("rev-parse", "HEAD").decode().strip() or None,
            "tracked_diff_sha256": hashlib.sha256(_git("diff", "HEAD", "--", "*.py", "*.sh")).hexdigest()}


def fingerprinted_source():
    source = source_identity()
    return {"code_sha256": source["code_sha256"], "environment": source["environment"]}


def max_cross_split_duplicates():
    """Tolerated count of identical-content images that appear in more than one split.

    Default 0 (strict). Some public datasets (e.g. Covid-QU-Ex) ship a few cross-split
    duplicates; historic runs were trained on them, so a campaign comparing against those
    runs must keep the split. Raise via DSSPMI_MAX_CROSS_SPLIT_DUPLICATES; the exact
    duplicates are recorded in the data identity (and therefore in the run contract).
    """
    return int(os.environ.get("DSSPMI_MAX_CROSS_SPLIT_DUPLICATES", "0"))


def audit_cache(data_dir, dataset, input_size=224, split_seed=42):
    """Inspect an existing cache without creating, upgrading, or repairing it."""
    root = Path(data_dir) / "preprocessed" / f"{dataset}_{input_size}"
    manifest = root / "split_manifest.csv"
    metadata_path = root / "preprocess_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split_seed") != split_seed or metadata.get("input_size") != input_size:
        raise ValueError(f"Cache split seed/input size mismatch: {root}")
    if metadata.get("cache_format_version") != 3:
        raise ValueError(f"Cache must already be prepared with format 3: {root}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty split manifest")
    ids, source_splits, content_splits, counts = {}, {}, {}, {}
    cross_split = []
    classes = {}
    aliases = ("adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma")
    for split in ("train", "val", "test"):
        directory = root / split
        names = sorted(p.name for p in directory.iterdir() if p.is_dir())
        if not names:
            raise ValueError(f"Missing independent {split} classes: {directory}")
        canonical = []
        for name in names:
            if dataset == "chest_ct_scan":
                matches = [x for x in aliases if name == x or name.startswith(x + "_")]
                if len(matches) != 1:
                    raise ValueError(f"Unknown Chest CT semantic label: {name}")
                canonical.append(matches[0])
            else:
                canonical.append(name)
        classes[split] = {name: i for i, name in enumerate(canonical)}
        if len(classes[split]) != len(names):
            raise ValueError("Ambiguous class aliases")
    if not classes["train"] == classes["val"] == classes["test"]:
        raise ValueError(f"Semantic class-index mismatch: {classes}")
    for row in rows:
        split, label = row["split"], row["class_name"]
        if split not in classes:
            raise ValueError(f"Unexpected split: {split}")
        path = root / split / label / Path(row["cached_path"]).name
        sample_id = path.relative_to(root).as_posix()
        if sample_id in ids or not path.is_file():
            raise ValueError(f"Duplicate/missing cached sample: {sample_id}")
        source = row["source_path"]
        previous = source_splits.setdefault(source, split)
        if previous != split:
            raise ValueError(f"Source overlaps splits: {source}")
        content = cached_file_sha256(path)
        previous = content_splits.setdefault(content, split)
        if previous != split:
            cross_split.append(sample_id)
        ids[sample_id] = content
        counts[split] = counts.get(split, 0) + 1
    actual = {p.relative_to(root).as_posix() for split in classes
              for p in (root / split).rglob("*") if p.is_file()}
    if set(ids) != actual:
        raise ValueError("Cache directories differ from split manifest")
    allowed = max_cross_split_duplicates()
    if len(cross_split) > allowed:
        raise ValueError(f"Image content overlaps splits ({len(cross_split)} > allowed {allowed}); "
                         f"first: {cross_split[0]}. Deduplicate the cache or set "
                         "DSSPMI_MAX_CROSS_SPLIT_DUPLICATES to accept the known historic duplicates.")
    if cross_split:
        print(f"WARNING: {dataset}: {len(cross_split)} cross-split duplicate image(s) accepted: {sorted(cross_split)}",
              file=sys.stderr)
    payload = {"dataset": dataset, "input_size": input_size, "split_seed": split_seed,
               "classes": classes, "counts": counts, "samples": ids,
               "cross_split_duplicates": sorted(cross_split),
               "mean": metadata["mean"], "std": metadata["std"],
               "transform": f"resize-shorter/randomcrop-padding{max(4, input_size // 10)}/hflip;eval-centercrop;RGB;v1"}
    return {"sha256": digest(payload), "manifest_sha256": file_sha256(manifest),
            "metadata_sha256": file_sha256(metadata_path), "identity": payload}


def run_contract(args, data):
    effective = normalized_config(args)
    classes = data.get("identity", {}).get("classes", {}).get("train")
    payload = {"effective_config": effective, "data_sha256": data["sha256"],
               "num_classes": len(classes) if classes is not None else None,
               "source": fingerprinted_source()}
    return {**payload, "sha256": digest(payload)}


def require_contract(saved, expected):
    if not saved or saved.get("sha256") != expected["sha256"]:
        raise ValueError("Run configuration/data/source fingerprint mismatch; refusing reuse")


def require_release(path, contract_sha, checkpoint_sha):
    if not path:
        raise ValueError("Evaluation requires --test-release-lock")
    lock = json.loads(Path(path).read_text())
    entry = lock.get("checkpoints", {}).get(contract_sha)
    if entry != checkpoint_sha:
        raise ValueError("Checkpoint is absent from the frozen test-release lock")
