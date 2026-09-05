"""SAGE-V2 controller/mechanism audit (read-only) for the seed-13 runs.

Reads ``sage_ds_v2_utility.jsonl`` + the selected-checkpoint controller state
of every provided run and emits compact CSVs and layer x epoch heatmaps.
Writes nothing into the run directories.

Epoch inference (see docs/SAGE_V2_REPRO.md section 5A):
the utility JSONL is written *epoch-major*: each epoch is exactly one block of
``ROWS_PER_EPOCH`` rows, sorted by in-epoch step (50,100,...,700).  So
``epoch(row) = row_index // ROWS_PER_EPOCH``.  (A ``step`` field stores the
in-epoch optimizer step offset, NOT a global counter; it must not be used for
epochting.)

Usage::

    python scripts/sage_v2_audit_a.py [run_dir ...] --out <out_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

ROWS_PER_EPOCH = 14
N_EPOCHS = 300
COLLAPSE_GATEP = 0.1
COLLAPSE_RUN_EPOCHS = 10


def _load_utility_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "sage_ds_v2_utility.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def _selected_gate_probs(run_dir: str, cfg: dict) -> dict:
    """Final hard-concrete gate probabilities from the *selected* checkpoint."""
    import torch
    from scsf.engine.checkpoint import CheckpointManager
    from scsf.methods import build_method

    manager = CheckpointManager(run_dir)
    if not manager.exists("selected"):
        return {}
    payload = manager.load("selected", map_location=torch.device("cpu"))
    method = build_method(cfg["method_name"], cfg)
    method.load_state_dict(payload["model_state"])
    with torch.no_grad():
        return {s: float(method.controller.gate_prob(s).detach().cpu())
                for s in method.site_names}


def _heatmap_matrices(rows: list[dict], epochs: np.ndarray) -> dict:
    """Per-(site, epoch) nan-mean matrices for gatep / effw / cos / raw."""
    sites = sorted({k[len("gatep_"):] for k in rows[0] if k.startswith("gatep_")})
    mats = {key: {s: np.full(N_EPOCHS, np.nan) for s in sites}
            for key in ("gatep", "effw", "cos", "raw")}
    fields = {"gatep": "gatep_", "effw": "eff_aux_w_", "cos": "cos_utility_",
              "raw": "raw_utility_"}
    for s in sites:
        for key, prefix in fields.items():
            vals = np.array([r.get(prefix + s, np.nan) for r in rows], dtype=float)
            arr = mats[key][s]
            for e in range(N_EPOCHS):
                m = epochs == e
                if m.any():
                    arr[e] = np.nanmean(vals[m])
    return mats


def audit_run(run_dir: str, cfg: dict) -> dict:
    rows = _load_utility_rows(run_dir)
    if not rows:
        raise RuntimeError(f"no utility telemetry in {run_dir}")
    n = len(rows)
    epochs = np.arange(n) // ROWS_PER_EPOCH
    epochs = np.clip(epochs, 0, N_EPOCHS - 1)
    sites = sorted({k[len("gatep_"):] for k in rows[0] if k.startswith("gatep_")})
    mats = _heatmap_matrices(rows, epochs)

    summary = {}
    for s in sites:
        g = np.array([r.get(f"gatep_{s}", np.nan) for r in rows], dtype=float)
        sg = np.array([r.get(f"sampled_gate_{s}", 0.0) for r in rows], dtype=float)
        w = np.array([r.get(f"eff_aux_w_{s}", np.nan) for r in rows], dtype=float)
        c = np.array([r.get(f"cos_utility_{s}", np.nan) for r in rows], dtype=float)
        rw = np.array([r.get(f"raw_utility_{s}", np.nan) for r in rows], dtype=float)
        u = np.array([r.get(f"uema_{s}", np.nan) for r in rows], dtype=float)
        gl = np.array([r.get(f"gl_norm_{s}", np.nan) for r in rows], dtype=float)
        til = np.array([r.get(f"tilde_gl_norm_{s}", np.nan) for r in rows], dtype=float)
        supp = np.array([r.get(f"support_frac_{s}", np.nan) for r in rows], dtype=float)
        ab = np.array([r.get(f"align_before_{s}", 0.0) for r in rows], dtype=float)

        gp = np.nan_to_num(g, nan=0.0)
        collapse = None
        run_start = -1
        for e in range(N_EPOCHS):
            sel = epochs == e
            below = bool(gp[sel].size) and float(gp[sel].mean()) < COLLAPSE_GATEP
            if below:
                if run_start < 0:
                    run_start = e
                if e - run_start >= COLLAPSE_RUN_EPOCHS - 1:
                    collapse = run_start
                    break
            else:
                run_start = -1

        tail = np.isfinite(u[-10:])
        summary[s] = dict(
            site=s, n_steps=len(rows),
            gatep_mean=float(np.nanmean(g)),
            gatep_first_epoch=float(np.nanmean(mats["gatep"][s][0:100])),
            gatep_mid_epoch=float(np.nanmean(mats["gatep"][s][100:200])),
            gatep_last_epoch=float(np.nanmean(mats["gatep"][s][200:300])),
            sampled_active_frac=float(np.mean(sg > 0.05)),
            effw_mean=float(np.nanmean(w)),
            cos_mean=float(np.nanmean(c)),
            cos_std=float(np.nanstd(c)),
            raw_mean=float(np.nanmean(rw)),
            raw_std=float(np.nanstd(rw)),
            uema_final=float(np.nanmean(u[-10:])) if tail.any() else np.nan,
            gl_norm_mean=float(np.nanmean(gl)),
            tilde_norm_mean=float(np.nanmean(til)),
            support_mean=float(np.nanmean(supp)),
            proj_activation_rate=float(np.nanmean(ab < 0.0)),
            aux_mass_sum=float(np.nansum(w * til)),
            collapse_epoch=("none" if collapse is None else int(collapse)),
        )

    topo = _selected_gate_probs(run_dir, cfg)
    for s in sites:
        summary[s]["final_gatep_selected"] = (
            float(topo[s]) if s in topo else np.nan)

    return dict(sites=sites, epochs=epochs, rows=rows,
                mats=mats, summary=summary)


def _write_heatmap(path: str, data: dict, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = list(data.keys())
    if not sites:
        return
    arr = np.stack([data[s] for s in sites])
    fig, ax = plt.subplots(figsize=(12, 2 + 0.6 * len(sites)))
    im = ax.imshow(arr, aspect="auto", cmap="viridis",
                   extent=[0, N_EPOCHS, len(sites) - 0.5, -0.5])
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites)
    ax.set_xlabel("epoch")
    ax.set_title(label)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="/root/scsf_v2_auditA")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    for run_dir in a.run_dirs:
        if not os.path.isdir(run_dir):
            raise SystemExit(f"missing run dir: {run_dir}")
        with open(os.path.join(run_dir, "cfg.json")) as f:
            cfg = json.load(f)
        name = cfg["run_name"]
        res = audit_run(run_dir, cfg)

        cols = list(res["summary"][res["sites"][0]].keys())
        csv_path = os.path.join(a.out, f"summary_{name}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in res["sites"]:
                w.writerow(res["summary"][s])

        for key, label in (("gatep", "gate probability"),
                           ("effw", "effective aux weight (z*s)"),
                           ("cos", "cosine utility U_cos"),
                           ("raw", "raw utility U_raw")):
            _write_heatmap(os.path.join(a.out, f"heatmap_{key}_{name}.png"),
                           res["mats"][key], f"{label}  [{name}]")

        print(f"audited {name}: {len(res['rows'])} estimates, "
              f"sites={res['sites']}")
        print("  site   | gatep_mean | collapse | aux_mass | sel_gatep")
        for s in res["sites"]:
            r = res["summary"][s]
            print(f"  {s:>6} | {r['gatep_mean']:.3f} | "
                  f"{str(r['collapse_epoch']):>8} | {r['aux_mass_sum']:.4f} | "
                  f"{r['final_gatep_selected']:.3f}")


if __name__ == "__main__":
    main()