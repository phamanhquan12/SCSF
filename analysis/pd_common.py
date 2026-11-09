"""Shared frozen-checkpoint and artifact utilities for E02/E16."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scsf.dsspmi_contract import audit_cache, file_sha256


def load_config(path):
    return json.loads(Path(path).read_text())


def state_sha256(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_checkpoint(path, device="cpu", data_dir=None):
    import run_experiment
    from scsf.methods import build_method
    saved_argv = sys.argv
    try:
        sys.argv = ["checkpoint-loader"]
        defaults = vars(run_experiment.parse_args())
    finally:
        sys.argv = saved_argv
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "last" or checkpoint.get("epoch") != 100:
        raise ValueError("Expected original epoch-100 last checkpoint")
    defaults.update(checkpoint["args"])
    args = SimpleNamespace(**defaults)
    if args.arch != "resnet50" or args.smoke_train_samples or args.smoke_eval_samples or args.amp:
        raise ValueError("Expected full FP32 R50 reference")
    if data_dir is not None:
        args.data_dir = str(data_dir)
    args.download = args.force_preprocess = False
    classes = checkpoint["num_classes"]
    # Every weight is replaced by strict loading; avoid downloading initialization weights.
    build_args = SimpleNamespace(**vars(args))
    build_args.pretrained = False
    model = build_method(build_args, classes, args.input_size or 224, train_size=0)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), args, checkpoint


def verified_data(entry, args, data_dir):
    data = audit_cache(data_dir, args.dataset, args.input_size or 224, args.medical_split_seed)
    evidence = load_config(entry["provenance"])
    if evidence.get("checkpoint_sha256") != file_sha256(entry["checkpoint"]):
        raise ValueError("Checkpoint does not match original provenance")
    if evidence.get("data_sha256") != data["sha256"]:
        raise ValueError("UNVERIFIED_SPLIT: missing/mismatched historic cache identity")
    return data


def require_test_release(path):
    if not path:
        raise ValueError("Test export/scoring requires the campaign test-release lock")
    value = load_config(path)
    if not value.get("matrix_sha256") or not value.get("checkpoints"):
        raise ValueError("Missing frozen campaign test release")
    return file_sha256(path)
