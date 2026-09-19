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

# Torch-free: do not import seeding (numpy/torch) at module import.
_DEFAULT = {"seed": 13, "torch_threads": 4}

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_ROOT = os.path.join(_ROOT, "configs")

_DEFAULTS = {
    "device": "auto",
    "results_root": "results",
    "torch_threads": _DEFAULT["torch_threads"],
}

# Runtime keys stripped from scientific_hash / comparison_signature so a
# plan-time resolve (device=auto, host data root) matches execute-time cfg.
_RUNTIME_TOP = ("device", "results_root", "run_name")
_RUNTIME_TRAIN = ("device",)
_RUNTIME_DATA = ("root", "num_workers", "split_index_dir")
# Artifact locators are host paths under results_root; identity is the
# construction (folds, CE anchor), not the absolute file path.
_RUNTIME_METHOD = ("oof_path", "memory_path")

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


def _load_method_layer(name, _seen=None):
    """Load a method YAML, resolving ``extends`` inheritance (deep merge).

    Parent layer is deep-merged first, then the child layer overrides it,
    so nested method subtrees inherit without clobbering (a child that only
    sets ``method.pretrain`` keeps its parent's ``method.taps``). Cycles are
    detected and hard-fail.
    """
    _seen = set() if _seen is None else set(_seen)
    if name in _seen:
        raise ValueError(f"cyclic method `extends` involving {name!r}")
    _seen.add(name)
    layer = _load_layer("methods", name)
    parent = layer.get("extends")
    if parent:
        if not isinstance(parent, str):
            raise ValueError("method `extends` must be a single method name string")
        merged = deepcopy(_load_method_layer(parent, _seen))
        _deep_merge(merged, layer)
        return merged
    return layer


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


def _promote_launcher_keys(overrides: dict) -> dict:
    """Copy top-level ``seed`` / ``variant`` into the nested schema.

    The portable launcher and some tests pass these at the top level; the
    engine's run_name and training loop read ``train.seed`` and
    ``method.variant``. Promotion is setdefault-only so an explicit nested
    value still wins.
    """
    out = dict(overrides)
    seed = out.get("seed")
    if seed is not None:
        train_ov = dict(out.get("train") or {})
        train_ov.setdefault("seed", int(seed) if not isinstance(seed, bool) else seed)
        train_ov.setdefault("data_order_seed", train_ov["seed"])
        out["train"] = train_ov
    variant = out.get("variant")
    if variant is not None:
        method_ov = dict(out.get("method") or {})
        method_ov.setdefault("variant", variant)
        out["method"] = method_ov
    return out


def resolve(overrides: dict, resolve_device: bool = True) -> dict:
    """Resolve layered config into the canonical engine/method cfg.

    ``resolve_device=False`` leaves ``train.device`` as ``auto`` and never
    imports torch. Plan-only launchers must use that path.
    """
    overrides = _promote_launcher_keys(overrides)
    dataset = str(overrides.get("dataset", "cifar10"))
    backbone = str(overrides.get("backbone", "resnet18"))
    method_name = str(overrides.get("method_name", overrides.get("method", "ce")))
    recipe = str(overrides.get("recipe", "singlerun"))

    ds_layer = _load_layer("datasets", dataset)
    bb_layer = _load_layer("backbones", backbone)
    mtd_layer = _load_method_layer(method_name)
    rcp_layer = _load_layer("recipes", recipe)

    # Named variants live in a ``variants:`` mapping of the method YAML; the
    # selected variant's ``method`` subtree is merged late (after recipe and
    # dataset-method overrides, before CLI overrides) and never misleads the
    # run_name, which appends ``.variant`` to disambiguate run dirs.
    _variants = (mtd_layer.pop("variants", None) or {}) if isinstance(mtd_layer, dict) else {}

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
    data.setdefault("official_train_size", 50000)
    data.setdefault("n_folds", 2)
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

    # Selected named variant: its ``method`` subtree is deep-merged after every
    # other layer (recipe, dataset-method, YAML default, CLI) so the explicit
    # variant always wins. Unknown variants hard-fail at resolve time.
    variant_name = cfg.get("method", {}).get("variant")
    if variant_name and _variants:
        vcfg = _variants.get(str(variant_name))
        if vcfg is None:
            raise ValueError(
                f"unknown method variant {variant_name!r} for {method_name!r}; "
                f"available: {sorted(_variants)}"
            )
        vmeth = vcfg.get("method", vcfg)
        if isinstance(vmeth, dict):
            _deep_merge(cfg.setdefault("method", {}), vmeth)

    # finalize dependent fields
    if resolve_device and cfg["train"].get("device") == "auto":
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
    # Repeat-run variants of the same method+score+mode (e.g. r3_small,
    # coverage-window) must never share a run dir.
    variant = cfg.get("method", {}).get("variant")
    if variant:
        name += f".{variant}"
    return f"{name}-r{cfg['recipe']}-s{cfg['train'].get('seed', 13)}"


def config_hash(cfg: dict) -> str:
    """Canonical SHA-256 over the fully resolved config (manifest/registry)."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_view(cfg: dict) -> dict:
    """Deep-copy cfg with runtime/host keys removed (scientific identity)."""
    view = deepcopy(cfg)
    for k in _RUNTIME_TOP:
        view.pop(k, None)
    train = dict(view.get("train") or {})
    for k in _RUNTIME_TRAIN:
        train.pop(k, None)
    if train:
        view["train"] = train
    data = dict(view.get("data") or {})
    for k in _RUNTIME_DATA:
        data.pop(k, None)
    if data:
        view["data"] = data
    method = dict(view.get("method") or {})
    for k in _RUNTIME_METHOD:
        method.pop(k, None)
    if method:
        view["method"] = method
    return view


def scientific_hash(cfg: dict) -> str:
    """Run hash: scientific config including seed, excluding host/runtime paths.

    Identifies a specific run. Device and data-root differences do not change
    it, so a plan-time row matches the execute-time child.
    """
    payload = json.dumps(_identity_view(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def comparison_signature(cfg: dict, source_sha: str = "") -> str:
    """Group intended seeds while preserving variant/source/recipe identity.

    Excludes seed / data_order_seed. Does **not** collapse two variants that
    happen to share a factory class, and does **not** split one intended cell
    merely because per-seed run hashes differ.
    """
    ident = _identity_view(cfg)
    train = dict(ident.get("train") or {})
    train.pop("seed", None)
    train.pop("data_order_seed", None)
    payload = {
        "dataset": ident.get("dataset"),
        "backbone": ident.get("backbone"),
        "method_name": ident.get("method_name"),
        "variant": (ident.get("method") or {}).get("variant"),
        "recipe": ident.get("recipe"),
        "method": ident.get("method"),
        "train": train,
        "data_num_classes": (ident.get("data") or {}).get("num_classes"),
        "data_split_seed": (ident.get("data") or {}).get("split_seed"),
        "source_sha": str(source_sha or ""),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
