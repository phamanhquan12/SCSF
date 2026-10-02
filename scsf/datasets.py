from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.datasets as tv_datasets
import torchvision.transforms as T
from PIL import Image

from .medical_registry import get_medical_entry, medical_dataset_slugs, prepare_medical_imagefolder


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    val_size: Optional[int] = None
    imagefolder_subdir: Optional[str] = None
    download: bool = False
    medmnist_flag: Optional[str] = None
    chexpert: bool = False
    medical_suite: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        input_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        val_size=2000,
        download=True,
    ),
    "svhn": DatasetSpec(
        name="svhn",
        num_classes=10,
        input_size=32,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        val_size=5000,
        imagefolder_subdir="svhn",
    ),
    "catsdogs": DatasetSpec(
        name="catsdogs",
        num_classes=2,
        input_size=64,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        val_size=2000,
        imagefolder_subdir="cats_dogs",
    ),
    "covid": DatasetSpec(
        name="covid",
        num_classes=3,
        input_size=64,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        imagefolder_subdir="covid",
    ),
    "organamnist": DatasetSpec(
        name="organamnist",
        num_classes=11,
        input_size=224,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        medmnist_flag="organamnist",
    ),
    "pathmnist": DatasetSpec(
        name="pathmnist",
        num_classes=9,
        input_size=224,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        medmnist_flag="pathmnist",
    ),
    "dermamnist": DatasetSpec(
        name="dermamnist",
        num_classes=7,
        input_size=224,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        medmnist_flag="dermamnist",
    ),
    "chexpert": DatasetSpec(
        name="chexpert",
        num_classes=2,
        input_size=224,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        chexpert=True,
    ),
}


CHEXPERT_KAGGLE_SLUG = "ashery/chexpert"
CHEXPERT_LABEL_COLUMNS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class IndexedDataset(Dataset):
    """Adds a stable sample index required by SAT's probability history."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, target, index


class TensorClassificationDataset(Dataset):
    def __init__(self, images: torch.Tensor, targets: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        self.images = images
        self.targets = targets.long().view(-1)
        self.mean = mean.float().view(-1, 1, 1)
        self.std = std.float().view(-1, 1, 1)

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, index: int):
        image = self.images[index].float().div(255.0)
        mean = self.mean
        std = self.std
        if image.size(0) == 1 and mean.size(0) == 3:
            image = image.expand(3, -1, -1)
        elif image.size(0) == 3 and mean.size(0) == 1:
            mean = mean.expand(3, -1, -1)
            std = std.expand(3, -1, -1)
        image = (image - mean) / std
        if image.size(0) == 1:
            image = image.expand(3, -1, -1)
        return image, self.targets[index]


class ImagePathClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        input_size: int,
        train: bool,
    ):
        self.samples = samples
        transforms = [T.Resize((input_size, input_size))]
        if train:
            transforms.extend(
                [
                    T.RandomCrop(input_size, padding=max(4, input_size // 10)),
                ]
            )
        transforms.extend([T.ToTensor(), T.Normalize(mean, std)])
        self.transform = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, target


def _transforms(spec: DatasetSpec):
    train_transform = T.Compose(
        [
            T.Resize(spec.input_size),
            T.RandomCrop(spec.input_size, padding=max(4, spec.input_size // 10)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize(spec.input_size),
            T.CenterCrop(spec.input_size),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
    return train_transform, eval_transform


def _limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def _split_test(testset: Dataset, val_size: Optional[int], seed: int):
    if not val_size:
        return testset, testset
    val_size = min(val_size, len(testset) - 1)
    generator = torch.Generator().manual_seed(seed)
    return random_split(testset, [val_size, len(testset) - val_size], generator=generator)


def _imagefolder(root: Path, train_transform, eval_transform, val_size: Optional[int], seed: int):
    train_path = root / "train"
    val_path = root / "val"
    test_path = root / "test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected ImageFolder layout with train/ and test/ under {root}")
    trainset = tv_datasets.ImageFolder(train_path, transform=train_transform)
    testset = tv_datasets.ImageFolder(test_path, transform=eval_transform)
    valset = tv_datasets.ImageFolder(val_path, transform=eval_transform) if val_path.exists() else None
    if valset is None:
        valset, final_test = _split_test(testset, val_size, seed)
    else:
        final_test = testset
    return trainset, valset, final_test, testset, len(trainset.classes)


def _medical_suite_datasets(data_dir: Path, spec: DatasetSpec, args):
    entry = get_medical_entry(spec.name, getattr(args, "datasets_file", "datasets_to_run.md"))
    prepared = prepare_medical_imagefolder(
        entry=entry,
        data_dir=data_dir,
        input_size=spec.input_size,
        download=args.download,
        force=args.force_preprocess,
        seed=getattr(args, "medical_split_seed", 42),
    )
    resolved_spec = DatasetSpec(
        **{
            **spec.__dict__,
            "num_classes": len(prepared.classes),
            "mean": prepared.mean,
            "std": prepared.std,
            "imagefolder_subdir": str(prepared.root),
        }
    )
    train_transform, eval_transform = _transforms(resolved_spec)
    trainset, valset, final_test, testset, discovered_classes = _imagefolder(
        prepared.root,
        train_transform,
        eval_transform,
        resolved_spec.val_size,
        args.seed,
    )
    return trainset, valset, final_test, testset, discovered_classes, resolved_spec


def _medmnist_cache_path(data_dir: Path, flag: str, input_size: int) -> Path:
    return data_dir / "preprocessed" / f"{flag}_{input_size}.pt"


def _load_medmnist_raw(flag: str, split: str, root: Path, download: bool, size: int):
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise ImportError(
            "OrganAMNIST support requires medmnist. Install it in your experiment environment with: "
            "pip install medmnist"
        ) from exc

    if flag not in INFO:
        raise ValueError(f"Unknown MedMNIST flag '{flag}'")
    root.mkdir(parents=True, exist_ok=True)
    dataset_cls = getattr(medmnist, INFO[flag]["python_class"])
    return dataset_cls(split=split, root=str(root), download=download, size=size)


def _target_to_int(target) -> int:
    if torch.is_tensor(target):
        return int(target.view(-1)[0].item())
    try:
        return int(target[0])
    except (TypeError, IndexError):
        return int(target)


class RawMedMNISTClassificationDataset(Dataset):
    def __init__(self, raw_dataset: Dataset, mean: torch.Tensor, std: torch.Tensor, input_size: int):
        self.raw_dataset = raw_dataset
        self.transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(tuple(float(x) for x in mean), tuple(float(x) for x in std)),
            ]
        )

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int):
        image, target = self.raw_dataset[index]
        image = self.transform(image.convert("RGB"))
        return image, _target_to_int(target)


def _preprocess_medmnist_split(raw_dataset, input_size: int):
    resize = T.Resize((input_size, input_size))
    to_tensor = T.PILToTensor()
    images = []
    targets = []
    for image, target in raw_dataset:
        image = image.convert("RGB")
        image = resize(image)
        images.append(to_tensor(image))
        targets.append(_target_to_int(target))
    return torch.stack(images), torch.tensor(targets, dtype=torch.long)


def _compute_stats(images: torch.Tensor):
    channel_sum = torch.zeros(images.size(1), dtype=torch.float64)
    channel_sq_sum = torch.zeros(images.size(1), dtype=torch.float64)
    pixel_count = 0
    for start in range(0, images.size(0), 512):
        batch = images[start : start + 512].float().div(255.0)
        channel_sum += batch.sum(dim=(0, 2, 3)).double()
        channel_sq_sum += batch.square().sum(dim=(0, 2, 3)).double()
        pixel_count += batch.size(0) * batch.size(2) * batch.size(3)
    mean = (channel_sum / pixel_count).float()
    variance = (channel_sq_sum / pixel_count).float() - mean.square()
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean, std


def _compute_medmnist_raw_stats(raw_dataset, input_size: int):
    resize = T.Resize((input_size, input_size))
    to_tensor = T.PILToTensor()
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for image, _ in raw_dataset:
        tensor = to_tensor(resize(image.convert("RGB"))).float().div(255.0)
        channel_sum += tensor.sum(dim=(1, 2)).double()
        channel_sq_sum += tensor.square().sum(dim=(1, 2)).double()
        pixel_count += tensor.size(1) * tensor.size(2)
    mean = (channel_sum / pixel_count).float()
    variance = (channel_sq_sum / pixel_count).float() - mean.square()
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean, std


def _use_lazy_medmnist_cache(spec: DatasetSpec) -> bool:
    return spec.medmnist_flag == "pathmnist" and spec.input_size >= 224


def _find_chexpert_root(root: Path) -> Optional[Path]:
    if (root / "train.csv").exists() and (root / "valid.csv").exists():
        return root
    for train_csv in root.rglob("train.csv"):
        candidate = train_csv.parent
        if (candidate / "valid.csv").exists() and (candidate / "train").exists() and (candidate / "valid").exists():
            return candidate
    return None


def _download_chexpert(root: Path):
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "CheXpert auto-download requires the Kaggle CLI. Install it with `pip install kaggle`, "
            "then place your Kaggle API token at ~/.kaggle/kaggle.json."
        )
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", CHEXPERT_KAGGLE_SLUG, "-p", str(root), "--unzip"], check=True)


def _ensure_chexpert_root(root: Path, download: bool) -> Path:
    chexpert_root = _find_chexpert_root(root)
    if chexpert_root is not None:
        return chexpert_root
    if not download:
        raise FileNotFoundError(
            f"CheXpert was not found under {root}. Re-run with --download after configuring the Kaggle CLI, "
            "or pass --dataset-root pointing at CheXpert-v1.0-small."
        )
    _download_chexpert(root)
    chexpert_root = _find_chexpert_root(root)
    if chexpert_root is None:
        raise FileNotFoundError(
            f"Downloaded {CHEXPERT_KAGGLE_SLUG}, but could not find train.csv, valid.csv, train/, and valid/ under {root}."
        )
    return chexpert_root


def _parse_chexpert_target(raw_value: str, uncertain_policy: str) -> Optional[int]:
    value = raw_value.strip()
    if value == "":
        return 0
    numeric = float(value)
    if numeric == 1.0:
        return 1
    if numeric == -1.0:
        if uncertain_policy == "one":
            return 1
        if uncertain_policy == "ignore":
            return None
        return 0
    return 0


def _resolve_chexpert_path(csv_root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    candidates = [csv_root / rel, csv_root.parent / rel]
    if len(rel.parts) > 1:
        stripped = Path(*rel.parts[1:])
        candidates.extend([csv_root / stripped, csv_root.parent / stripped])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_chexpert_csv(csv_path: Path, label_column: str, uncertain_policy: str, frontal_only: bool) -> list[tuple[Path, int]]:
    samples = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if label_column not in (reader.fieldnames or []):
            raise ValueError(f"CheXpert label '{label_column}' not found in {csv_path}. Choices: {', '.join(CHEXPERT_LABEL_COLUMNS)}")
        for row in reader:
            if frontal_only and row.get("Frontal/Lateral") != "Frontal":
                continue
            target = _parse_chexpert_target(row.get(label_column, ""), uncertain_policy)
            if target is None:
                continue
            samples.append((_resolve_chexpert_path(csv_path.parent, row["Path"]), target))
    if not samples:
        raise RuntimeError(f"No CheXpert samples found in {csv_path} for label={label_column} policy={uncertain_policy}")
    return samples


def _chexpert_stats_cache_path(data_dir: Path, label_column: str, uncertain_policy: str, input_size: int, frontal_only: bool) -> Path:
    label_slug = label_column.lower().replace(" ", "_").replace("/", "_")
    view_slug = "frontal" if frontal_only else "allviews"
    return data_dir / "preprocessed" / f"chexpert_{label_slug}_{uncertain_policy}_{view_slug}_{input_size}_stats.pt"


def _compute_image_path_stats(samples: list[tuple[Path, int]], input_size: int, max_samples: Optional[int]):
    resize = T.Resize((input_size, input_size))
    to_tensor = T.PILToTensor()
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    selected = samples if max_samples is None or max_samples <= 0 else samples[:max_samples]
    for path, _ in selected:
        with Image.open(path) as image:
            image = resize(image.convert("RGB"))
            tensor = to_tensor(image).float().div(255.0)
        channel_sum += tensor.sum(dim=(1, 2)).double()
        channel_sq_sum += tensor.square().sum(dim=(1, 2)).double()
        pixel_count += tensor.size(1) * tensor.size(2)
    mean = (channel_sum / pixel_count).float()
    variance = (channel_sq_sum / pixel_count).float() - mean.square()
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean, std


def _chexpert_datasets(data_dir: Path, spec: DatasetSpec, args):
    root = Path(args.dataset_root) if args.dataset_root else data_dir / "chexpert"
    chexpert_root = _ensure_chexpert_root(root, args.download)
    if args.chexpert_label not in CHEXPERT_LABEL_COLUMNS:
        raise ValueError(f"Unknown CheXpert label '{args.chexpert_label}'. Choices: {', '.join(CHEXPERT_LABEL_COLUMNS)}")

    train_samples = _load_chexpert_csv(chexpert_root / "train.csv", args.chexpert_label, args.chexpert_uncertain_policy, args.chexpert_frontal_only)
    val_samples = _load_chexpert_csv(chexpert_root / "valid.csv", args.chexpert_label, args.chexpert_uncertain_policy, args.chexpert_frontal_only)

    stats_path = _chexpert_stats_cache_path(data_dir, args.chexpert_label, args.chexpert_uncertain_policy, spec.input_size, args.chexpert_frontal_only)
    if stats_path.exists() and not args.force_preprocess:
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = _compute_image_path_stats(train_samples, spec.input_size, args.chexpert_stats_samples)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dataset": spec.name,
                "source": CHEXPERT_KAGGLE_SLUG,
                "root": str(chexpert_root),
                "label": args.chexpert_label,
                "uncertain_policy": args.chexpert_uncertain_policy,
                "frontal_only": args.chexpert_frontal_only,
                "input_size": spec.input_size,
                "stats_samples": args.chexpert_stats_samples,
                "mean": mean,
                "std": std,
                "preprocessing": [
                    f"resize to {spec.input_size}x{spec.input_size}",
                    "normalize with CheXpert train-split dataset-specific mean/std",
                    "binary target from selected CheXpert label column",
                ],
            },
            stats_path,
        )

    mean_tuple = tuple(float(x) for x in mean)
    std_tuple = tuple(float(x) for x in std)
    trainset = ImagePathClassificationDataset(train_samples, mean_tuple, std_tuple, spec.input_size, train=True)
    valset = ImagePathClassificationDataset(val_samples, mean_tuple, std_tuple, spec.input_size, train=False)
    resolved_spec = DatasetSpec(**{**spec.__dict__, "mean": mean_tuple, "std": std_tuple})
    return trainset, valset, valset, valset, resolved_spec.num_classes, resolved_spec


def _prepare_medmnist_cache(data_dir: Path, spec: DatasetSpec, args) -> dict:
    assert spec.medmnist_flag is not None
    cache_path = _medmnist_cache_path(data_dir, spec.medmnist_flag, spec.input_size)
    if cache_path.exists() and not args.force_preprocess:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    raw_root = data_dir / "medmnist_raw"
    if _use_lazy_medmnist_cache(spec):
        raw_train = _load_medmnist_raw(spec.medmnist_flag, "train", raw_root, args.download, spec.input_size)
        mean, std = _compute_medmnist_raw_stats(raw_train, spec.input_size)
        payload = {
            "dataset": spec.name,
            "medmnist_flag": spec.medmnist_flag,
            "input_size": spec.input_size,
            "mean": mean,
            "std": std,
            "lazy_raw": True,
            "preprocessing": [
                "download native MedMNIST 224 archive",
                "PIL convert RGB",
                f"resize to {spec.input_size}x{spec.input_size}",
                "normalize with train-split dataset-specific mean/std during loading",
                "do not duplicate large native archive into a tensor cache",
            ],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        return payload

    splits = {}
    for split in ("train", "val", "test"):
        raw = _load_medmnist_raw(spec.medmnist_flag, split, raw_root, args.download, spec.input_size)
        images, targets = _preprocess_medmnist_split(raw, spec.input_size)
        splits[split] = {"images": images, "targets": targets}

    mean, std = _compute_stats(splits["train"]["images"])
    payload = {
        "dataset": spec.name,
        "medmnist_flag": spec.medmnist_flag,
        "input_size": spec.input_size,
        "mean": mean,
        "std": std,
        "splits": splits,
        "preprocessing": [
            "PIL convert RGB",
            f"resize to {spec.input_size}x{spec.input_size}",
            "cache uint8 tensor",
            "normalize with train-split dataset-specific mean/std during loading",
            "ensure 3-channel tensor for shared backbones",
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _medmnist_datasets(data_dir: Path, spec: DatasetSpec, args):
    payload = _prepare_medmnist_cache(data_dir, spec, args)
    mean = payload["mean"]
    std = payload["std"]
    if payload.get("lazy_raw"):
        raw_root = data_dir / "medmnist_raw"
        train_raw = _load_medmnist_raw(spec.medmnist_flag, "train", raw_root, args.download, spec.input_size)
        val_raw = _load_medmnist_raw(spec.medmnist_flag, "val", raw_root, args.download, spec.input_size)
        test_raw = _load_medmnist_raw(spec.medmnist_flag, "test", raw_root, args.download, spec.input_size)
        trainset = RawMedMNISTClassificationDataset(train_raw, mean, std, spec.input_size)
        valset = RawMedMNISTClassificationDataset(val_raw, mean, std, spec.input_size)
        testset = RawMedMNISTClassificationDataset(test_raw, mean, std, spec.input_size)
    else:
        trainset = TensorClassificationDataset(payload["splits"]["train"]["images"], payload["splits"]["train"]["targets"], mean, std)
        valset = TensorClassificationDataset(payload["splits"]["val"]["images"], payload["splits"]["val"]["targets"], mean, std)
        testset = TensorClassificationDataset(payload["splits"]["test"]["images"], payload["splits"]["test"]["targets"], mean, std)
    mean = tuple(float(x) for x in payload["mean"])
    std = tuple(float(x) for x in payload["std"])
    resolved_spec = DatasetSpec(**{**spec.__dict__, "mean": mean, "std": std})
    return trainset, valset, testset, testset, resolved_spec.num_classes, resolved_spec


def get_dataset_spec(
    name: str,
    dataset_root: Optional[str],
    num_classes: Optional[int],
    input_size: Optional[int],
    datasets_file: str | Path = "datasets_to_run.md",
) -> DatasetSpec:
    if name == "imagefolder":
        if dataset_root is None or num_classes is None:
            raise ValueError("--dataset imagefolder requires --dataset-root and --num-classes")
        size = input_size or 224
        return DatasetSpec(
            name="imagefolder",
            num_classes=num_classes,
            input_size=size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    if name not in DATASETS:
        datasets_file = Path(datasets_file)
        medical_slugs = medical_dataset_slugs(datasets_file) if datasets_file.exists() else []
        if name in medical_slugs:
            return DatasetSpec(
                name=name,
                num_classes=num_classes or 2,
                input_size=input_size or 224,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                medical_suite=True,
            )
        choices = sorted([*DATASETS, *medical_slugs])
        raise ValueError(f"Unknown dataset '{name}'. Choices: {', '.join(choices)}, imagefolder")
    spec = DATASETS[name]
    if input_size is not None:
        spec = DatasetSpec(**{**spec.__dict__, "input_size": input_size})
    return spec


def build_loaders(args):
    base_data_dir = Path(args.data_dir)
    spec = get_dataset_spec(
        args.dataset,
        args.dataset_root,
        args.num_classes,
        args.input_size,
        getattr(args, "datasets_file", "datasets_to_run.md"),
    )

    if spec.medmnist_flag is not None:
        trainset, valset, final_test, testset, num_classes, spec = _medmnist_datasets(base_data_dir, spec, args)
    elif spec.chexpert:
        trainset, valset, final_test, testset, num_classes, spec = _chexpert_datasets(base_data_dir, spec, args)
    elif spec.medical_suite:
        trainset, valset, final_test, testset, num_classes, spec = _medical_suite_datasets(base_data_dir, spec, args)
    else:
        train_transform, eval_transform = _transforms(spec)

    if spec.medmnist_flag is not None:
        pass
    elif spec.chexpert:
        pass
    elif spec.medical_suite:
        pass
    elif spec.name == "cifar10":
        trainset = tv_datasets.CIFAR10(base_data_dir, train=True, download=args.download, transform=train_transform)
        testset = tv_datasets.CIFAR10(base_data_dir, train=False, download=args.download, transform=eval_transform)
        valset, final_test = _split_test(testset, spec.val_size, args.seed)
        num_classes = 10
    elif spec.name == "svhn":
        svhn_root = base_data_dir / "svhn"
        trainset = tv_datasets.SVHN(svhn_root, split="train", download=args.download, transform=train_transform)
        testset = tv_datasets.SVHN(svhn_root, split="test", download=args.download, transform=eval_transform)
        valset, final_test = _split_test(testset, spec.val_size, args.seed)
        num_classes = 10
    else:
        root = Path(args.dataset_root) if args.dataset_root else base_data_dir / (spec.imagefolder_subdir or spec.name)
        trainset, valset, final_test, testset, discovered_classes = _imagefolder(
            root, train_transform, eval_transform, spec.val_size, args.seed
        )
        num_classes = args.num_classes or discovered_classes or spec.num_classes

    trainset = _limit_dataset(trainset, args.smoke_train_samples)
    valset = _limit_dataset(valset, args.smoke_eval_samples)
    final_test = _limit_dataset(final_test, args.smoke_eval_samples)
    testset = _limit_dataset(testset, args.smoke_eval_samples)

    indexed_trainset = IndexedDataset(trainset)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
    }
    if args.workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )
    loaders = {
        "train": DataLoader(indexed_trainset, batch_size=args.batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(final_test, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
        "test_full": DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
    }
    return loaders, spec, num_classes
