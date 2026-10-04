from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Bump when the cache layout or split logic changes so stale caches are rebuilt.
CACHE_FORMAT_VERSION = 2
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
SKIP_PARTS = {
    "mask",
    "masks",
    "lung masks",
    "infection masks",
    "segmentation",
    "segmentations",
    "mask images",
}


@dataclass(frozen=True)
class MedicalDatasetEntry:
    display_name: str
    slug: str
    kaggle_slug: str
    url: str


@dataclass(frozen=True)
class PreparedMedicalDataset:
    entry: MedicalDatasetEntry
    root: Path
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    classes: list[str]
    manifest_path: Path
    stats_path: Path


KNOWN_MEDICAL_DATASETS: dict[str, tuple[str, str]] = {
    "covidquex": ("covid_qu_ex", "anasmohammedtahir/covidqu"),
    "braintumormri": ("brain_tumor_mri", "mohammadhossein77/brain-tumors-dataset"),
    "braintumormrimasoud": ("brain_tumor_mri_masoud", "masoudnickparvar/brain-tumor-mri-dataset"),
    "alzheimersmri": ("alzheimers_mri", "preetpalsingh25/alzheimers-dataset-4-class-of-images"),
    "malariamicroscopic": ("malaria_microscopic", "iarunava/cell-images-for-detecting-malaria"),
    "chestctscan": ("chest_ct_scan", "mohamedhanyyy/chest-ctscan-images"),
    "tuberculosisxray": ("tuberculosis_xray", "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"),
    "braincancermri": ("brain_cancer_mri", "orvile/brain-cancer-mri-dataset"),
    "sarscov2ct": ("sars_cov2_ct", "plameneduardo/sarscov2-ctscan-dataset"),
    "breastultrasound": ("breast_ultrasound", "aryashah2k/breast-ultrasound-images-dataset"),
    "retinaloct": ("retinal_oct", "paultimothymooney/kermany2018"),
    "ham10000": ("ham10000", "kmader/skin-cancer-mnist-ham10000"),
    "aptos2019": ("aptos2019", "mariaherrerot/aptos2019"),
}


def _normalize_name(name: str) -> str:
    name = name.replace("’", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def slug_to_display_name(slug: str) -> str:
    for key, (known_slug, _) in KNOWN_MEDICAL_DATASETS.items():
        if slug == known_slug:
            words = re.sub(r"(?<!^)([A-Z])", r" \1", key).strip()
            return words or slug
    return slug.replace("_", " ").title()


def parse_datasets_to_run(path: Path | str = "datasets_to_run.md") -> list[MedicalDatasetEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Medical suite file not found: {path}")
    entries: list[MedicalDatasetEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|") or "kaggle.com/datasets" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        raw_name, raw_url = cells[0], cells[1]
        display_name = re.sub(r"[*`]", "", raw_name).strip()
        match = re.search(r"https://www\.kaggle\.com/datasets/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", raw_url)
        if not display_name or not match:
            continue
        kaggle_slug = match.group(1)
        normalized = _normalize_name(display_name)
        if normalized not in KNOWN_MEDICAL_DATASETS:
            raise ValueError(f"Unknown medical dataset row '{display_name}' in {path}; add it to KNOWN_MEDICAL_DATASETS.")
        slug, expected_kaggle_slug = KNOWN_MEDICAL_DATASETS[normalized]
        if kaggle_slug != expected_kaggle_slug:
            raise ValueError(
                f"Kaggle slug mismatch for {display_name}: expected {expected_kaggle_slug}, got {kaggle_slug}"
            )
        entries.append(MedicalDatasetEntry(display_name, slug, kaggle_slug, f"https://www.kaggle.com/datasets/{kaggle_slug}"))
    if not entries:
        raise ValueError(f"No known medical datasets found in {path}.")
    return entries


def get_medical_entry(slug: str, datasets_file: Path | str = "datasets_to_run.md") -> MedicalDatasetEntry:
    for entry in parse_datasets_to_run(datasets_file):
        if entry.slug == slug:
            return entry
    raise KeyError(f"Medical dataset '{slug}' is not listed in {datasets_file}")


def medical_dataset_slugs(datasets_file: Path | str = "datasets_to_run.md") -> list[str]:
    return [entry.slug for entry in parse_datasets_to_run(datasets_file)]


def _download_kaggle_dataset(entry: MedicalDatasetEntry, raw_root: Path):
    kaggle_exe = shutil.which("kaggle")
    kaggle_cmd: list[str] | None = [kaggle_exe] if kaggle_exe is not None else None
    if kaggle_exe is None:
        venv_kaggle = Path(sys.executable).parent / "kaggle"
        if venv_kaggle.exists():
            kaggle_cmd = [str(venv_kaggle)]
    if kaggle_cmd is None:
        try:
            import kaggle  # noqa: F401
        except ImportError:
            pass
        else:
            kaggle_cmd = [sys.executable, "-m", "kaggle"]
    if kaggle_cmd is None:
        raise RuntimeError(
            "Kaggle auto-download requires the Kaggle CLI. Install it with `pip install kaggle` "
            "and place your token at ~/.kaggle/kaggle.json."
        )
    raw_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([*kaggle_cmd, "datasets", "download", "-d", entry.kaggle_slug, "-p", str(raw_root), "--unzip"], check=True)


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            lowered_parts = {part.lower() for part in path.parts}
            if lowered_parts.intersection(SKIP_PARTS):
                continue
            yield path


def _split_from_parts(parts: tuple[str, ...]) -> tuple[str | None, int | None]:
    for idx, part in enumerate(parts):
        split = SPLIT_ALIASES.get(part.lower())
        if split is not None:
            return split, idx
    return None, None


def _class_from_path(path: Path, raw_root: Path) -> tuple[str | None, str | None]:
    rel = path.relative_to(raw_root)
    split, split_idx = _split_from_parts(rel.parts)
    if split_idx is not None:
        for part in rel.parts[split_idx + 1 : -1]:
            lowered = part.lower()
            if lowered in {"images", "image", "imgs", "files"} or lowered in SKIP_PARTS:
                continue
            return split, part
        return split, None

    for parent in reversed(rel.parts[:-1]):
        lowered = parent.lower()
        if lowered in {"images", "image", "imgs", "files"} or lowered in SKIP_PARTS:
            continue
        return None, parent
    return None, None


def _discover_samples(raw_root: Path) -> list[dict]:
    if (raw_root / "HAM10000_metadata.csv").exists():
        return _discover_ham10000_samples(raw_root)
    aptos_rows = _discover_aptos2019_samples(raw_root)
    if aptos_rows:
        return aptos_rows

    rows = []
    seen: set[tuple[str, str]] = set()
    for path in _iter_images(raw_root):
        split, class_name = _class_from_path(path, raw_root)
        if class_name is None:
            continue
        fingerprint = (class_name, hashlib.sha1(path.read_bytes()).hexdigest())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rows.append({"source_path": path, "split": split, "class_name": class_name})
    if not rows:
        raise RuntimeError(f"No class-labelled images found under {raw_root}")
    return rows


def _image_lookup(raw_root: Path) -> dict[str, Path]:
    lookup = {}
    for path in _iter_images(raw_root):
        lookup[path.stem] = path
    return lookup


def _discover_ham10000_samples(raw_root: Path) -> list[dict]:
    metadata_path = raw_root / "HAM10000_metadata.csv"
    image_lookup = _image_lookup(raw_root)
    rows = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("image_id")
            diagnosis = row.get("dx")
            if not image_id or not diagnosis or image_id not in image_lookup:
                continue
            # Several images per lesion: split by lesion_id so a lesion never spans train and test.
            group = row.get("lesion_id") or image_id
            rows.append({"source_path": image_lookup[image_id], "split": None, "class_name": diagnosis, "group": group})
    if not rows:
        raise RuntimeError(f"HAM10000 metadata found at {metadata_path}, but no labelled images were resolved.")
    return rows


def _discover_aptos2019_samples(raw_root: Path) -> list[dict]:
    csv_paths = sorted(raw_root.rglob("*.csv"))
    label_csvs: list[tuple[Path, str | None]] = []
    for path in csv_paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and {"id_code", "diagnosis"}.issubset(set(reader.fieldnames)):
                    stem = path.stem.lower()
                    if stem.startswith("train"):
                        split = "train"
                    elif stem in {"valid", "validation", "val"} or stem.startswith("valid"):
                        split = "val"
                    elif stem.startswith("test"):
                        split = "test"
                    else:
                        split = None
                    label_csvs.append((path, split))
        except UnicodeDecodeError:
            continue
    if not label_csvs:
        return []

    image_lookup = _image_lookup(raw_root)
    rows = []
    seen: set[str] = set()
    official_split_found = any(split is not None for _, split in label_csvs)
    for label_csv, split in label_csvs:
        if official_split_found and split is None:
            continue
        with label_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_id = row.get("id_code")
                diagnosis = row.get("diagnosis")
                if image_id is None or diagnosis is None or image_id not in image_lookup or image_id in seen:
                    continue
                seen.add(image_id)
                rows.append({"source_path": image_lookup[image_id], "split": split, "class_name": str(diagnosis)})
    if not rows:
        csv_list = ", ".join(str(path) for path, _ in label_csvs)
        raise RuntimeError(f"APTOS label CSV(s) found at {csv_list}, but no labelled images were resolved.")
    return rows


def _stable_split(rows: list[dict], seed: int) -> list[dict]:
    """Stratified 70/10/20 split. Rows sharing a "group" (e.g. HAM10000 lesion_id) stay together."""
    by_class: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        group = str(row.get("group") or row["source_path"])
        by_class.setdefault(row["class_name"], {}).setdefault(group, []).append(row)
    rng = random.Random(seed)
    split_rows: list[dict] = []
    for class_name, groups in sorted(by_class.items()):
        ordered = [groups[key] for key in sorted(groups)]
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.10))) if n >= 3 else 0
        if n_train + n_val >= n and n > 1:
            n_train = n - 1
            n_val = 0
        for idx, group_rows in enumerate(ordered):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            split_rows.extend({**row, "split": split} for row in group_rows)
    return split_rows


def _resolve_splits(rows: list[dict], seed: int) -> list[dict]:
    if any(row["split"] is None for row in rows):
        return _stable_split([{**row, "split": None} for row in rows], seed)
    splits = {row["split"] for row in rows}
    if "train" not in splits or "test" not in splits:
        return _stable_split([{**row, "split": None} for row in rows], seed)
    if "val" in splits:
        return rows
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    val_rows = _stable_split(train_rows, seed)
    promoted_val_sources = {row["source_path"] for row in val_rows if row["split"] == "val"}
    resolved = []
    for row in train_rows:
        resolved.append({**row, "split": "val" if row["source_path"] in promoted_val_sources else "train"})
    resolved.extend(test_rows)
    return resolved


def _copy_and_resize(rows: list[dict], cache_root: Path, input_size: int) -> list[dict]:
    """Cache RGB PNGs with the shorter side resized to input_size (same op as the loader's
    T.Resize(input_size), which then becomes a no-op), so epochs never decode full-size originals."""
    resize = T.Resize(input_size)

    def cache_one(row: dict) -> dict | None:
        source = Path(row["source_path"])
        class_name = str(row["class_name"]).strip().replace("/", "_")
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:16]
        dest = cache_root / row["split"] / class_name / f"{source.stem}_{digest}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source) as image:
                resized = resize(image.convert("RGB"))
        except (OSError, UnidentifiedImageError):
            return None
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        resized.save(dest, format="PNG")
        return {
            "split": row["split"],
            "class_name": class_name,
            "source_path": str(source),
            "cached_path": str(dest),
        }

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        results = list(pool.map(cache_one, rows))
    return [row for row in results if row is not None]


def _stats_from_manifest(rows: list[dict], input_size: int):
    train_paths = [Path(row["cached_path"]) for row in rows if row["split"] == "train"]
    if not train_paths:
        raise RuntimeError("Cannot compute dataset statistics without train samples.")
    resize = T.Resize((input_size, input_size))
    to_tensor = T.PILToTensor()
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for path in train_paths:
        with Image.open(path) as image:
            tensor = to_tensor(resize(image.convert("RGB"))).float().div(255.0)
        channel_sum += tensor.sum(dim=(1, 2)).double()
        channel_sq_sum += tensor.square().sum(dim=(1, 2)).double()
        pixel_count += tensor.size(1) * tensor.size(2)
    mean = (channel_sum / pixel_count).float()
    variance = (channel_sq_sum / pixel_count).float() - mean.square()
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def prepare_medical_imagefolder(
    entry: MedicalDatasetEntry,
    data_dir: Path,
    input_size: int,
    download: bool,
    force: bool,
    seed: int = 42,
) -> PreparedMedicalDataset:
    raw_root = data_dir / "raw" / entry.slug
    cache_root = data_dir / "preprocessed" / f"{entry.slug}_{input_size}"
    manifest_path = cache_root / "split_manifest.csv"
    stats_path = cache_root / "preprocess_manifest.json"

    if not raw_root.exists() or not any(_iter_images(raw_root)):
        if not download:
            raise FileNotFoundError(
                f"{entry.display_name} is not prepared under {raw_root}. Re-run with --download or place the Kaggle files there."
            )
        _download_kaggle_dataset(entry, raw_root)

    if force and cache_root.exists():
        shutil.rmtree(cache_root)

    if cache_root.exists() and stats_path.exists():
        cached_version = json.loads(stats_path.read_text(encoding="utf-8")).get("cache_format_version", 1)
        if cached_version != CACHE_FORMAT_VERSION:
            print(f"Rebuilding {cache_root}: cache format {cached_version} -> {CACHE_FORMAT_VERSION}")
            shutil.rmtree(cache_root)

    if manifest_path.exists() and stats_path.exists():
        metadata = json.loads(stats_path.read_text(encoding="utf-8"))
        return PreparedMedicalDataset(
            entry=entry,
            root=cache_root,
            mean=tuple(metadata["mean"]),
            std=tuple(metadata["std"]),
            classes=list(metadata["classes"]),
            manifest_path=manifest_path,
            stats_path=stats_path,
        )

    discovered = _discover_samples(raw_root)
    resolved = _resolve_splits(discovered, seed)
    manifest_rows = _copy_and_resize(resolved, cache_root, input_size)
    classes = sorted({row["class_name"] for row in manifest_rows})
    if len(classes) < 2:
        raise RuntimeError(f"{entry.display_name} produced fewer than two classes: {classes}")
    required_splits = {"train", "test"}
    found_splits = {row["split"] for row in manifest_rows}
    if not required_splits.issubset(found_splits):
        raise RuntimeError(f"{entry.display_name} missing required splits after preprocessing: {found_splits}")

    mean, std = _stats_from_manifest(manifest_rows, input_size)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "source_path", "cached_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts: dict[str, int] = {}
    for row in manifest_rows:
        key = f"{row['split']}:{row['class_name']}"
        split_counts[key] = split_counts.get(key, 0) + 1
    metadata = {
        "dataset": entry.slug,
        "display_name": entry.display_name,
        "kaggle_slug": entry.kaggle_slug,
        "url": entry.url,
        "input_size": input_size,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "split_seed": seed,
        "classes": classes,
        "num_classes": len(classes),
        "mean": mean,
        "std": std,
        "split_counts": split_counts,
        "preprocessing": [
            f"download/unpack Kaggle dataset to {raw_root}",
            f"resize RGB images to {input_size}x{input_size} in the shared loader",
            f"store ImageFolder split cache as RGB PNGs with the shorter side resized to {input_size}",
            "split HAM10000 by lesion_id so no lesion appears in more than one split",
            "normalize with train-split dataset-specific mean/std",
        ],
    }
    stats_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PreparedMedicalDataset(entry, cache_root, mean, std, classes, manifest_path, stats_path)
