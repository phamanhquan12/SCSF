# SAGE-V2 multi-seed confirmation protocol

Status: preregistered before any confirmation run is launched.
Method frozen at commit `9649eea` (exact authoring commit `dirty=False` for the
seed-13 exploratory runs; same code + config is re-frozen for every
confirmation run).

## 1. Decision and scope

SAGE-v2 (`sage_ds_v2`) passes the **seed-13 promotion test** on both datasets
(paired vs `sage_ds` v1 under the identical `ccl_sc_reference` recipe) but has
not passed the formal baseline gate.  This document preregisters the
confirmation protocol for seeds 17, 23, 29, 31 and the read-only seed-13
audits.

Promotion to confirmation does **not** mean the original gate passed.  The
formal empirical gate (multi-baseline comparison under the locked
`EMPIRICAL_CONTRACT.md` / `REFERENCE_PROTOCOL.md` splits and recipe) is
untouched by this document and is only evaluated after confirmation (section
6).

## 2. Frozen method (no tuning)

- Code: commit `9649eea` (short), `dirty=False`.  Runs are launched from a
  mirror pinned to that exact commit with `SCSF_SOURCE_COMMIT=9649eea`; the
  run `manifest.json` must record `commit=9649eea`.
- Backbone: `vgg16_bn`; recipe `ccl_sc_reference` (300 epochs, batch 64,
  SGD lr 0.1 / momentum 0.9 / wd 5e-4, milestone decay 0.5 per 25 epochs,
  locked 45k/5k split — dataloader indices exposed via `needs_indices=True`).
- Method config: `configs/methods/sage_ds_v2.yaml` unchanged:
  `topology=sparse_utility_safe`, `sparsity_cost=0.5`, `supervision_scale=1.0`,
  `utility_interval=50`, `controller_lr=0.05`, `utility_ema_beta=0.99`,
  `hard_concrete_tau=0.3`, `strength_cap=0.5`, `projection_eps=1e-8`.
- **No hyperparameter is tuned on seed-13 results.**  Any need to change a
  constant re-opens this protocol.

## 3. Confirmation runs

| dataset  | seeds               | checks |
|---|---|---|
| CIFAR-10 | 17, 23, 29, 31      | 4      |
| CIFAR-100| 17, 23, 29, 31      | 4      |

Run names follow the resolver naming scheme
(`<dataset>-vgg16_bn-sage_ds_v2-rccl_sc_reference-s<seed>`); seed 13 serves as
the already-collected fifth seed (exploratory evidence preserved verbatim).
All runs use a fresh results root (`/root/scsf_v2_confirm`); raw JSONL
telemetry is preserved unwritten and unedited.

## 4. Endpoints and statistics

Each evaluation is on the locked official **test** split at the checkpoint
selected by the pre-registered rule (`min_val_aurc_among_acc>=best_acc-1.0pp`,
the unchanged recipe guard).

Primary endpoint (paired, same seed, same split):

- test AURC: `delta_seed = AURC(sage_ds_v2, seed) - AURC(sage_ds, seed)`.

Accuracy guard (per seed and per mean):

- `acc(sage_ds_v2) >= acc(sage_ds) - 1.0pp`.  Any violation is reported
  loudly; the mean paired accuracy delta must satisfy the guard.

Secondary endpoints (paired, same seeds):

- excess-AURC, failure AUROC, AUPR, and the full risk-coverage grid
  (cov 99…01) deltas on test.
- mean-class AURC and **worst-class AURC** as safety diagnostics (world: the
  seed-13 CIFAR-100 worst-class AURC worsens 0.3985 → 0.5774; audit B
  section 5 must classify that before any multi-seed claim).

Report for every endpoint across the 5 paired seeds (13+17+23+29+31):

- mean and std of v1 and v2;
- paired per-seed deltas (all five listed individually);
- 95% bootstrap CI (percentile, 10k resamples, balanced within pairs);
- paired nonparametric sign-flip test (exact over the 5 deltas) for AURC on
  each dataset;
- win/tie/loss count on the paired deltas.

Promotion-after-confirmation criterion (fixed here, not the formal gate):

- mean paired AURC improves on **both** datasets, and the mean accuracy guard
  holds on both.  No interim checkpoint selection; every result reported uses
  the pre-registered selection rule only.

## 5. Read-only seed-13 audits (preregistered methodology)

Both audits run on the preserved exploratory artifacts and write nothing into
`/root/scsf_v2_results` or any v1 registry.

### A. Controller/mechanism audit (both seed-13 runs)

Inputs: `<run>/sage_ds_v2_utility.jsonl` (per-estimate telemetry),
`<run>/sage_ds_v2.jsonl` (per-epoch aggregate), `<run>/manifest.json`,
selected/best checkpoint controller state.

- Epoch inference (documented rule): the AURC surrogate runs once per
  `utility_interval=50` steps with a deterministic `batch_index`; steps per
  epoch = `ceil(45000/64) = 704`, so `epoch = floor(step / 704)` with
  `step` = the logged `step` field (first logged step 50 → epoch 0).
- Per-site series across training (plotted as layer×epoch heatmaps where
  sensible): gate probability `gatep`, sampled activation frequency
  (`sampled_gate` mean/frac>0), effective auxiliary weight `eff_aux_w`,
  cosine utility, raw utility, EMA utility, `gl_norm` / `tilde_gl_norm`,
  `support_frac`, and the CE-projection activation rate
  (`frac of estimates with align_before < 0`).
- Time-to-collapse: first epoch at which `gatep` falls and stays below 0.1
  for ≥10 consecutive epochs, per site.
- Total auxiliary-gradient mass per site:
  `sum over estimates of eff_aux_w_s * tilde_gl_norm_s` (normalized by number
  of estimates).
- Topology at the **selected** checkpoint (`selected.pt`, CIFAR-10 epoch 224,
  CIFAR-100 epoch 279 — not epoch 299), reported as final `gatep` per site.
- Verdict framing: is CIFAR-10 best described as learned **transient**
  supervision (supervision significant early, collapses late) rather than a
  persistent sparse topology?

### B. CIFAR-100 class audit (both v1 and v2 seed-13)

Inputs: v1 and v2 selected checkpoints + configs only; **read-only
recomputation of test-set metrics on the official test split** (no registry
writes, no checkpoint/config selection on test-class results).

- Identify the class with worst-class AURC = 0.577445 in v2 (and its v1
  counterpart).
- For that class: test set size, error count, v1 vs v2 accuracy, AURC,
  coverage at several confidence percentiles.
- Full per-class AURC delta distribution (100 classes): median, quartiles,
  upper tail (p90/p95/max), count of classes moved worse, count moved worse
  by >0.05.
- Classification of the finding: isolated instability (one small class) vs
  broad minority-class degradation (many classes drifting worse).

## 6. Gate separation

Confirmation (section 4) is not the formal gate.  The formal empirical gate
requires SAGE-v2 compared against the matched baseline anchors
(CE, DG, SelectiveNet, CCL-SC ref, SCSF) under the same recipe and locked
splits (`docs/EMPIRICAL_CONTRACT.md`, `docs/REFERENCE_PROTOCOL.md`).  This
protocol does not redefine that gate and no confirmation result alone may be
reported as "gate passed".

## 7. Required follow-up ablations (run only if confirmation succeeds)

1. Learned controller vs all-layers/equal supervision.
2. Learned controller vs a fixed early-only schedule.
3. Learned controller vs a fixed schedule matched for total auxiliary
   gradient mass (per site and in total).
4. Sparse gating vs learned continuous weights.
5. Cosine utility (`U_cos`) vs raw utility (`U_raw`).
6. Bilevel disjoint meta batches vs same-batch utility.
7. Classification-safety projection on/off.

These resolve the two open scientific warnings from seed 13 (CIFAR-100
worst-class regression; controller regimes "all-supervision-off late" on
CIFAR-10 vs "pool2-only" on CIFAR-100 — i.e. whether learned supervision is
transient rather than persistent).

## 8. Preservation

- Archive on the server: confirmation manifest, registry, progress log,
  per-run manifests, configs, and SHA256 checksums over all confirmed run
  telemetry (utility JSONL, aggregate JSONL) and the seed-13 artifacts.
- Committed to the repo: analysis scripts, compact tables (CSV), heatmaps
  (PNG), and the checksum/archive manifest — **never** the raw large JSONL.
- No raw registry or seed-13 artifact is rewritten.