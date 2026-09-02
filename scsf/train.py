"""train entrypoint: ``python -m scsf.train dataset=cifar10 backbone=resnet18 method=ce seed=13``.

Run directories live under ``<results_root>/<run_name>/``.
"""

from __future__ import annotations

import os

from .engine.config import overrides_from_cli, resolve
from .engine.trainer import Trainer


def main(argv=None) -> dict:
    from .engine.seeding import _DEFAULT
    overrides = overrides_from_cli(argv)
    resume_from = overrides.pop("resume_from", None)
    cfg = resolve(overrides)
    os.environ.setdefault("SCSF_DATA_SEED", str(cfg["train"]["seed"]))
    run_dir = os.path.join(cfg["results_root"], cfg["run_name"])
    trainer = Trainer(cfg, run_dir)
    out = trainer.run(resume_from=resume_from)
    print(f"done: {run_dir}")
    return out


if __name__ == "__main__":
    main()