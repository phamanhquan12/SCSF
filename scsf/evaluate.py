"""evaluate entrypoint: ``python -m scsf.evaluate run_dir=results/... split=test``."""

from __future__ import annotations

import sys

from .engine.config import overrides_from_cli
from .engine.evaluator import evaluate_run


def main(argv=None) -> dict:
    overrides = overrides_from_cli(argv)
    run_dir = overrides.pop("run_dir")
    split = str(overrides.pop("split", "val"))
    checkpoint = str(overrides.pop("checkpoint", "selected"))
    append = str(overrides.pop("append", "true")).lower() != "false"
    device = overrides.pop("device", None)
    out = evaluate_run(run_dir, split=split, checkpoint=checkpoint, append=append, device=device)
    m = out["metrics"]
    print(f"{run_dir} [{split}] acc={m['acc']:.4f} aurc={m['aurc']:.4f} "
          f"auroc_err={m['auroc_error']:.4f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])