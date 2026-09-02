"""extract_depthfg entrypoint.

``python -m scsf.extract_depthfg run_dir=results/... split=val [options]``

Extracts sample-level signed radius profiles (sample IDs + site names) from an
ordinary CE checkpoint, evaluates the validation-fitted score ladder on the
untouched test split, and runs the analytic-vs-iterative boundary audit on a
fixed validation subset. Scratch artifacts go to ``out_dir`` (default
``<run_dir>/depthfrag``); they must not be committed.

Options (all optional)::

    run_dir=...            run directory containing cfg.json + selected.pt
    split=val              split to profile (val for fitting, or test)
    out_dir=...            artifact directory
    subset=N               cap the profiled split to N samples
    test_subset=N          cap the test split used for ladder evaluation
    p=2 q=2 eps=1e-12      geometry exponents / eps
    mode=fast|exact        analytic gradient mode (exact: per-example torch.func)
    exact_microbatch=1     examples per exact-mode step
    mid_roles=top_l2       intermediate-radius role
    iterative_subset=128   fixed validation subset for the DeepFool-style audit
    iterative_steps=50     max walk steps per sample
    checkpoint=selected    checkpoint tag
    device=cuda            torch device
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .data.cifar import TEST_SPLIT_DISABLED, set_test_allowed
from .engine.config import overrides_from_cli, resolve
from .metrics import all_metrics


def main(argv=None) -> dict:
    overrides = overrides_from_cli(argv)
    run_dir = str(overrides.pop("run_dir"))
    split = str(overrides.pop("split", "val"))
    out_dir = str(overrides.pop("out_dir", os.path.join(run_dir, "depthfrag")))
    subset = int(overrides.pop("subset", 0)) or None
    test_subset = int(overrides.pop("test_subset", 0)) or None
    p = float(overrides.pop("p", 2))
    q = float(overrides.pop("q", 2))
    eps = float(overrides.pop("eps", 1e-12))
    mode = str(overrides.pop("mode", "fast"))
    exact_microbatch = int(overrides.pop("exact_microbatch", 1))
    mid_roles = [r.strip() for r in str(overrides.pop("mid_roles", "top_l2")).split(",")]
    iterative_subset = int(overrides.pop("iterative_subset", 128))
    iterative_steps = int(overrides.pop("iterative_steps", 50))
    checkpoint = str(overrides.pop("checkpoint", "selected"))
    device = overrides.pop("device", None)

    if not os.path.exists(os.path.join(run_dir, "cfg.json")):
        raise FileNotFoundError(f"not a run dir: {run_dir} (missing cfg.json)")
    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg = json.load(f)
    cfg["train"]["device"] = torch.device(device or cfg["train"].get("device", "cpu"))

    from .depthfrag.extract import DepthFragExtractor

    ext = DepthFragExtractor(cfg, run_dir, checkpoint=checkpoint, device=device,
                             p=p, q=q, eps=eps, mode=mode,
                             exact_microbatch=exact_microbatch, mid_roles=mid_roles)

    # the ladder always reports val + untouched-test metrics
    prof_val = ext.profile_split("val", subset=subset, num_workers=0)
    was = TEST_SPLIT_DISABLED
    if was:
        set_test_allowed(True)
    try:
        prof_test = ext.profile_split("test", subset=test_subset, num_workers=0)
    finally:
        set_test_allowed(was)

    from .depthfrag.extract import evaluate_variants

    eval_out = evaluate_variants(prof_val, prof_test, ext.terminal_site,
                                 ext.mid_sites)

    audit = ext.iterative_audit(prof_val, subset=iterative_subset,
                                max_steps=iterative_steps)

    summary = {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "mode": mode,
        "p": p, "q": q, "eps": eps,
        "profile_wall_s_val": prof_val["wall_s"],
        "profile_wall_s_test": prof_test["wall_s"],
        "variants": eval_out["variants"],
        "iterative": audit,
    }
    os.makedirs(out_dir, exist_ok=True)
    ext.save_artifacts(out_dir, prof_val, {"mode": mode, "p": p, "q": q, "eps": eps})
    ext.save_artifacts(out_dir, prof_test, {"mode": mode, "p": p, "q": q, "eps": eps})
    audit_persist = {
        "subset_n": audit["subset_n"],
        "summary": audit["summary"],
        "analytic_vs_iter": audit["analytic_vs_iter"],
    }
    with open(os.path.join(out_dir, "iterative_audit.json"), "w") as f:
        json.dump(audit_persist, f, indent=2, sort_keys=True, default=float)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=float)
    _save_scores_csv(out_dir, eval_out, prof_val, prof_test)
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def _save_scores_csv(out_dir, eval_out, prof_val, prof_test):
    import csv

    names = sorted(set(eval_out["scores_val"]) | set(eval_out["scores_test"]))
    for split, prof, sv in (("val", prof_val, eval_out["scores_val"]),
                            ("test", prof_test, eval_out["scores_test"])):
        path = os.path.join(out_dir, f"scores_{split}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "prediction", "margin", "error"] + names)
            err = (prof["predictions"] != prof["labels"]).astype(int)
            for i in range(len(prof["ids"])):
                w.writerow([int(prof["ids"][i]), int(prof["labels"][i]),
                            int(prof["predictions"][i]), float(prof["margins"][i]),
                            int(err[i])] + [f"{float(sv[n][i]):.6f}" if n in sv else ""
                                            for n in names])


if __name__ == "__main__":
    main(sys.argv[1:])