"""Configuration resolver: YAML layers + CLI overrides -> canonical cfg.

Merge order (later wins): built-in defaults < dataset < backbone < method
< recipe < CLI overrides. The resolved object matches the schema the method
classes and the engine are written against:

    {
      "dataset": "cifar10",                 # string name
      "data": {...},                        # num_classes, split_seed, root, ...
      "backbone": "resnet18",
      "backbones": {"resnet18": {...}},     # input_size, patch_size, ...
      "method_name": "ce",
      "method": {...},                      # score, mode, pretrain, queue_size...
      "train": {...},                       # epochs, batch_size, seed, lr, ...
      "recipe": "singlerun",
      "meta_lr": 1e-4,
      "run_name": "cifar10-resnet18-ce-r...-s13",
      "results_root": "results",
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy

from .seeding import _DEFAULT

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_ROOT = os.path.join(_ROOT, "configs")

_DEFAULTS = {
    "device": "auto",
    "results_root": "results",
    "torch_threads": _DEFAULT["torch_threads"],
}

_DATASET_DEFAULT = {
    "cifar10": {"num_classes": 10, "split_seed": 20260902, "n_train": 45000, "n_val": 5000},
    "cifar100": {"num_classes": 100, "split_seed": 20260902, "n_train": 45000, "n_val": 5000},
    "official_train_size": 50000,
}


def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_layer(kind, name):
    return _load_yaml(os.path.join(CONFIG_ROOT, kind, f"{name}.yaml"))


def _coerce(value):
    if isinstance(value, str):
        low = value.lower()
        if low in ("true",):
            return True
        if low in ("false",):
            return False
        if low in ("none", "null"):
            return None
        if low.startswith("[") and low.endswith("]"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _set_dotted(target, key, value):
    parts = key.split(".")
    node = target
    for p in parts[:-1]:
        nxt = node.get(p) if isinstance(node, dict) else None
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = _coerce(value)
    return target


def _deep_merge(target, source):
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = _coerce(v)


def overrides_from_cli(argv=None):
    """Parse ``k=v [k=v ...]`` CLI arguments (dotted keys, value coercion)."""
    argv = sys.argv[1:] if argv is None else argv
    out = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected 'key=value', got {arg!r}")
        k, v = arg.split("=", 1)
        # Hydra-style "\u002b" flags (e.g. "+resume_from=epoch_003") collapse to
        # the plain override key; the trainer pops "resume_from" from this dict.
        # Without this strip a documented resume flag would be silently ignored
        # and a killed job would restart instead of resume.
        k = k.strip().lstrip("+")
        _set_dotted(out, k, v)
    return out


def resolve(overrides: dict) -> dict:
    """Resolve layered config into the canonical engine/method cfg."""
    dataset = str(overrides.get("dataset", "cifar10"))
    backbone = str(overrides.get("backbone", "resnet18"))
    method_name = str(overrides.get("method_name", overrides.get("method", "ce")))
    recipe = str(overrides.get("recipe", "singlerun"))

    ds_layer = _load_layer("datasets", dataset)
    bb_layer = _load_layer("backbones", backbone)
    mtd_layer = _load_layer("methods", method_name)
    rcp_layer = _load_layer("recipes", recipe)

    cfg = dict(_DEFAULTS)
    cfg.update(ds_layer)
    cfg.update(bb_layer)
    cfg.update(mtd_layer)
    cfg.update(rcp_layer)

    # Per-backbone recipe dispatch (e.g. AdamW for transformers, SGD for CNNs).
    dispatch = (rcp_layer.get("by_backbone", {}) or {}).get(backbone, {})
    if dispatch:
        cfg.setdefault("train", {}).update(dispatch)

    # Per-dataset recipe dispatch.  Reference protocols frequently share one
    # training recipe across CIFAR-10/100 while retaining dataset-specific
    # method constants (e.g. CCL-SC queue size and DG reward).  Keeping those
    # constants in the recipe makes the resolved cfg the auditable source of
    # truth instead of hiding them in manifest-generation conditionals.
    dataset_dispatch = (rcp_layer.get("by_dataset", {}) or {}).get(dataset, {})
    if dataset_dispatch:
        _deep_merge(cfg, dataset_dispatch)

    # Per-method recipe overrides (e.g. paper pretrain lengths).
    per_method = (rcp_layer.get("methods", {}) or {}).get(method_name, {})
    if per_method:
        meth = dict(cfg.get("method", {}) or {})
        meth.update(per_method.get("method", per_method))
        cfg["method"] = meth

    # Dataset-specific method constants take precedence over the global
    # method block.  Schema: by_dataset.<dataset>.methods.<method_name>.
    dataset_method = (dataset_dispatch.get("methods", {}) or {}).get(method_name, {})
    if dataset_method:
        meth = dict(cfg.get("method", {}) or {})
        meth.update(dataset_method.get("method", dataset_method))
        cfg["method"] = meth

    # canonical schema keys
    cfg["dataset"] = dataset
    cfg["backbone"] = backbone
    cfg["method_name"] = method_name
    cfg["recipe"] = recipe

    data = dict(cfg.get("data", {}))
    data.setdefault("num_classes", _DATASET_DEFAULT[dataset]["num_classes"])
    data.setdefault("split_seed", _DATASET_DEFAULT[dataset]["split_seed"])
    data.setdefault("root", os.environ.get("SCSF_DATA_ROOT", os.path.join(_ROOT, "data")))
    data.setdefault("normalize",
                    {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]})
    data.setdefault("num_workers", 4)
    data.setdefault("download", False)
    data.setdefault("split_index_dir", os.path.join(cfg.get("results_root", "results"), "splits"))
    data.setdefault("use_serialized_splits", True)
    cfg["data"] = data

    bb_cfg = cfg.get("backbones", {}).get(backbone, {})
    bbs = {backbone: bb_cfg}
    bbs[backbone].setdefault("input_size", 32)
    cfg["backbones"] = bbs

    method = dict(cfg.get("method", {}))
    method.setdefault("score", cfg.get("score", "msp") if not cfg.get("method") else "msp")
    cfg["method"] = method

    train = dict(cfg.get("train", {}))
    train.setdefault("epochs", 200)
    train.setdefault("batch_size", 128)
    train.setdefault("seed", 13)
    train.setdefault("lr", 0.1)
    train.setdefault("momentum", 0.9)
    train.setdefault("weight_decay", 5e-4)
    train.setdefault("optimizer", "sgd")
    train.setdefault("scheduler", "cosine")
    train.setdefault("data_order_seed", train["seed"])
    train.setdefault("guard_delta_acc", 1.0)
    train.setdefault("save_every", 5)
    train.setdefault("eval_every", 1)
    train.setdefault("overfit", 0)
    train.setdefault("device", cfg.get("device", "auto"))
    cfg["train"] = train

    cfg.setdefault("meta_lr", 1e-4)

    # apply remaining CLI overrides last (deep-merged so normalized subtrees
    # such as data.root / train.defaults survive nonzero top-level keys)
    for k, v in overrides.items():
        _deep_merge(cfg, {k: v})

    # finalize dependent fields
    if cfg["train"].get("device") == "auto":
        import torch
        cfg["train"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("run_name", run_name_for(cfg))
    return cfg


def run_name_for(cfg: dict) -> str:
    name = f"{cfg['dataset']}-{cfg['backbone']}-{cfg['method_name']}"
    score = cfg.get("method", {}).get("score")
    default_score = cfg.get("default_score")
    if score and score != default_score:
        name += f".{score}"
    # SCSF posthoc/e2e use the same method_name+score but distinct gradient
    # semantics; disambiguate non-default modes so run dirs never collide.
    mode = cfg.get("method", {}).get("mode")
    if cfg.get("method_name") == "scsf" and mode and mode != "posthoc":
        name += f".{mode}"
    return f"{name}-r{cfg['recipe']}-s{cfg['train'].get('seed', 13)}"


def config_hash(cfg: dict) -> str:
    """Canonical SHA-256 over the fully resolved config (manifest/registry)."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
