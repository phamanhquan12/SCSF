"""Config resolver and run-registry contracts."""

import os

import pytest

from scsf.engine import config
from scsf.engine.registry import BASE_COLUMNS, append_rows, load_registry
from scsf.engine.trainer import _git_state


def test_cli_nested_overrides_and_coercion():
    ov = config.overrides_from_cli(["dataset=cifar10", "train.epochs=200",
                                    "method.alpha=0.5", "train.lr=0.05",
                                    "method.queue_size=64"])
    assert ov["dataset"] == "cifar10"
    assert ov["train"]["epochs"] == 200
    assert ov["method"]["queue_size"] == 64
    assert ov["train"]["lr"] == 0.05


def test_explicit_source_commit_is_authoritative(monkeypatch):
    monkeypatch.setenv("SCSF_SOURCE_COMMIT", "734569f")
    monkeypatch.setenv("SCSF_SOURCE_DIRTY", "0")
    assert _git_state() == ("734569f", False)


def test_bad_override_rejected():
    with pytest.raises(ValueError):
        config.overrides_from_cli(["dataset"])


def test_resume_flags_collapse_to_plain_keys():
    # ``+resume_from=epoch_003`` (documented CLI) must land on the plain
    # "resume_from" key exactly as trainer.main pops it; a missing '\u002B'
    # strip restarts a killed job instead of resuming (regression test).
    ov = config.overrides_from_cli(["+resume_from=epoch_003"])
    assert ov["resume_from"] == "epoch_003"
    assert "+resume_from" not in ov
    assert config.overrides_from_cli(["resume_from=last"])["resume_from"] == "last"


def test_resolve_defaults_and_env_root(monkeypatch):
    monkeypatch.setenv("SCSF_DATA_ROOT", "/tmp/scsf_data_root_check")
    cfg = config.resolve({"dataset": "cifar10", "results_root": "/tmp/opencode/cfg_tests"})
    assert cfg["data"]["root"] == "/tmp/scsf_data_root_check"
    assert cfg["data"]["download"] is False          # never auto-download
    assert cfg["data"]["use_serialized_splits"] is True
    assert cfg["data"]["num_classes"] == 10
    assert cfg["train"]["seed"] == 13
    assert cfg["train"]["optimizer"] == "sgd"
    assert cfg["train"]["scheduler"] == "cosine"
    assert cfg["run_name"] == "cifar10-resnet18-ce-rsinglerun-s13"


def test_resolve_method_and_backbone_layers_override():
    cfg = config.resolve({"method_name": "scsf", "backbone": "resnet18",
                          "method": {"mode": "e2e"}, "train": {"epochs": 30}})
    assert cfg["method"]["mode"] == "e2e"
    assert cfg["train"]["epochs"] == 30
    assert cfg["backbones"]["resnet18"]["input_size"] == 32


def test_resolve_keeps_run_name_unique_per_seed_and_score():
    a = config.resolve({"dataset": "cifar10", "train": {"seed": 1}})
    b = config.resolve({"dataset": "cifar10", "train": {"seed": 2}})
    assert a["run_name"].endswith("-s1")
    assert b["run_name"].endswith("-s2")
    assert a["run_name"] != b["run_name"]


def test_scsf_mode_disambiguates_run_name():
    post = config.resolve({"method_name": "scsf", "method": {"mode": "posthoc"}})
    e2e = config.resolve({"method_name": "scsf", "method": {"mode": "e2e"}})
    legacy = config.resolve({"method_name": "scsf",
                             "method": {"mode": "legacy_partial_detach"}})
    assert post["run_name"] != e2e["run_name"]
    assert post["run_name"] != legacy["run_name"]
    # the default (posthoc) keeps the classic name; non-defaults add a suffix
    assert post["run_name"] == f"cifar10-resnet18-scsf-rsinglerun-s13"
    assert e2e["run_name"].endswith("scsf.e2e-rsinglerun-s13")
    # non-SCSF methods are untouched
    ce = config.resolve({"method_name": "ce"})
    assert ce["run_name"] == "cifar10-resnet18-ce-rsinglerun-s13"


def test_config_hash_is_deterministic_and_sensitive():
    a = config.resolve({"dataset": "cifar10", "train": {"seed": 13}})
    b = config.resolve({"dataset": "cifar10", "train": {"seed": 13}})
    c = config.resolve({"dataset": "cifar10", "train": {"seed": 17}})
    assert config.config_hash(a) == config.config_hash(b)
    assert config.config_hash(a) != config.config_hash(c)
    assert len(config.config_hash(a)) == 64


def test_dotted_cli_overrides_behavior_differs_from_resolve_dict():
    # dotted keys are a CLI-parsing artifact (overrides_from_cli); a raw dict
    # passed to resolve must be nested, otherwise the key is taken literally.
    cli = config.resolve(config.overrides_from_cli(["train.seed=7"]))
    assert cli["train"]["seed"] == 7
    literal = config.resolve({"train.seed": 7})
    assert literal["train"]["seed"] == 13  # untouched literal key "train.seed"


def test_vgg_backbone_dispatch_defaults():
    cfg = config.resolve({"backbone": "vgg16_bn"})
    assert cfg["backbones"]["vgg16_bn"]["input_size"] == 32


def test_ccl_sc_reference_recipe_matches_paper_and_dispatches_by_dataset():
    common = {"backbone": "vgg16_bn", "recipe": "ccl_sc_reference"}
    c10 = config.resolve({**common, "dataset": "cifar10", "method_name": "ccl_sc"})
    c100 = config.resolve({**common, "dataset": "cifar100", "method_name": "ccl_sc"})
    for cfg in (c10, c100):
        assert cfg["train"]["epochs"] == 300
        assert cfg["train"]["batch_size"] == 64
        assert cfg["train"]["optimizer"] == "sgd"
        assert cfg["train"]["scheduler"] == "step"
        assert cfg["train"]["milestones"] == list(range(25, 300, 25))
        assert cfg["train"]["gamma"] == 0.5
    for key, value in {
        "pretrain": 150, "queue_size": 300, "memo_m": 0.999, "reward": 0.5,
    }.items():
        assert c10["method"][key] == value
    for key, value in {
        "pretrain": 150, "queue_size": 3000, "memo_m": 0.99, "reward": 1.0,
    }.items():
        assert c100["method"][key] == value

    dg10 = config.resolve({**common, "dataset": "cifar10", "method_name": "dg"})
    dg100 = config.resolve({**common, "dataset": "cifar100", "method_name": "dg"})
    assert (dg10["method"]["reward"], dg10["method"]["pretrain"]) == (2.2, 100)
    assert (dg100["method"]["reward"], dg100["method"]["pretrain"]) == (4.6, 200)
    assert dg10["method"]["score"] == "dg_conf"


# ---------------------------------------------------------------------------


def test_registry_column_set_is_locked():
    assert "run_dir" in BASE_COLUMNS
    assert "split_hash" in BASE_COLUMNS
    assert "config_hash" in BASE_COLUMNS
    assert "risk_at_cov_5" in BASE_COLUMNS
    assert "risk_at_cov_100" in BASE_COLUMNS
    assert "checkpoint_epoch" in BASE_COLUMNS
    assert BASE_COLUMNS.count("acc") == 1
    # one row per (run_dir, split): no duplicate column defs
    assert len(set(BASE_COLUMNS)) == len(BASE_COLUMNS)


def test_registry_append_dedup_and_roundtrip(tmp_path):
    p = os.path.join(tmp_path, "registry.csv")
    row0 = {"run_dir": "r1", "split": "val", "acc": "0.5", "aurc": "0.4",
            "dataset": "cifar10", "backbone": "resnet18", "method_name": "ce",
            "score": "msp", "recipe": "singlerun", "seed": "1", "complete": "1"}
    row1 = dict(row0, acc="0.6", aurc="0.3")
    append_rows(p, [dict(row0, split="val")])
    append_rows(p, [dict(row1, split="val")])   # same (run_dir, split) replaces
    append_rows(p, [dict(row1, split="test")])  # different split appends
    rows = load_registry(p)
    assert len(rows) == 2
    by_split = {r["split"]: r["acc"] for r in rows}
    assert by_split["val"] == "0.6"
    assert by_split["test"] == "0.6"


def test_registry_rejects_unknown_columns():
    import csv
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(BASE_COLUMNS + ["evil_col"])
        w.writerow([""] * len(BASE_COLUMNS) + ["drop me"])
        path = f.name
    # load is lenient: the extra column comes through
    rows = load_registry(path)
    assert "evil_col" in rows[0]
    # append enforces the locked column set: evil_col never reaches disk
    append_rows(path + ".out", [{**{c: "" for c in BASE_COLUMNS},
                                 "run_dir": "x", "split": "val"}])
    with open(path + ".out") as f:
        header = next(csv.reader(f))
    assert "evil_col" not in header
    assert set(BASE_COLUMNS).issubset(header)
