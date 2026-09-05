# SAGE-V2 seed-13 audits (read-only)

Commit: seed-13 runs at `9649eea` (exploratory), baselines frozen registry s13.
Scripts: `scripts/sage_v2_audit_a.py` (controller/mechanism),
`scripts/sage_v2_audit_b.py` (CIFAR-100 per-class). Re-ran after fixing
epoch inference (see note below). Artifacts: `*.csv`, `heatmap_*.png`,
`preservation_manifest.txt`, `SHA256SUMS.txt`.

## Audit A — controller / mechanism (both seed-13 runs, 4,200 estimator calls each)

Telemetry is epoch-major, 14 rows/epoch (in-epoch steps 50..700), 300 epochs →
**epoch(row) = row_index // 14**. Earlier run of the script wrongly binned by
`step//704` (binning every estimate into epoch 0); the fix above is what the
numbers below use. Selected-checkpoint topology (`selected.pt`, cpu load):

| run     | pool1 | pool2 | pool3 | pool4 | pool5 | (unit)        |
|---------|------:|------:|------:|------:|------:|---------------|
| cifar10 | 0.000 | 0.000 | 0.000 | 0.759 | 1.000 | gate prob     |
| cifar100| 0.000 | 0.959 | 0.000 | 0.000 | 0.613 | gate prob     |

Per-site means over all estimates (see `summary_*.csv` for all columns):

| run     | site | gatep | first/mid/last epochs | collapse@ | sampled_active | aux_mass |
|---------|------|------:|----------------------|----------:|---------------:|---------:|
| cifar10 | p1   | 0.037 | 0.040/0.039/0.031     | 2         | 0.053          | 98.2     |
| cifar10 | p2   | 0.414 | 0.445/0.421/0.377     | 10        | 0.426          | 2028     |
| cifar10 | p3   | 0.686 | 0.729/0.664/0.665     | none      | 0.694          | 4987     |
| cifar10 | p4   | 0.739 | 0.770/0.720/0.726     | none      | 0.755          | 6596     |
| cifar10 | p5   | 0.903 | 0.912/0.897/0.901     | none      | 0.913          | 9257     |
| cifar100| p1   | 0.054 | 0.059/0.058/0.046     | 3         | 0.072          | 157      |
| cifar100| p2   | 0.624 | 0.647/0.624/0.602     | none      | 0.670          | 5849     |
| cifar100| p3   | 0.223 | 0.239/0.239/0.191     | 6         | 0.235          | 1295     |
| cifar100| p4   | 0.248 | 0.267/0.266/0.212     | 6         | 0.260          | 1802     |
| cifar100| p5   | 0.925 | 0.934/0.920/0.921     | none      | 0.931          | 21873    |

**Readout.** Supervision is *persistent*, not transient:
- CIFAR-10: keep-late prune-early. pool1 collapses @2, pool2 @10, pooled gate
  ~<0.05 (pool1) to ~0.9 (pool5); selected keeps only pool4-5 (0.759/1.000).
- CIFAR-100: pool2 + pool5 keep (0.959/0.613 selected), pool1/3/4 pruned early
  (@3/6/6). aux-gradient mass is late-concentrated on both datasets
  (CIFAR-10 pool5 ≈ 40%; CIFAR-100 pool5 ≈ 71%).
- uema_final trends: negative for pool3/4/5 on CIFAR-10 (−0.052/−0.055/−0.074),
  near-zero for pruned pool2 (+0.012) — late-layer utility has been *reduced*
  as supervision kept them active.

## Audit B — CIFAR-100 per-class, seed 13 (100 classes × 100 test samples)

Reproduction check: recomputed global metrics == registry rows exactly
(v1 acc .7293 aurc .0802; v2 acc .7356 aurc .0764).

| metric                    | v1  | v2  |
|---------------------------|----:|----:|
| test acc                  | 0.7293 | 0.7356 |
| test AURC                 | 0.0802 | 0.0764 |
| mean-class AURC           | 0.1117 | 0.1091 |
| worst-class AURC          | 0.3985 (cls 73) | 0.5764 (cls 35) |

Per-class AURC deltas, v2 − v1 (n=100): mean −0.0027, median −0.0034,
q25 −0.0110, q75 +0.0075, p90 +0.0231, p95 +0.0340, **max +0.1803 (class 35)**,
min −0.1121. 44/100 classes worsen, **5/100 worsen by ≥0.05** (35:+0.180,
92:+0.079, 47:+0.059, 96:+0.057, 11:+0.054).

Class 35 is the sole large regression and the new worst class:
n_err 45→55/100, acc 0.55→0.45, cov@50% 0.85→0.72, cov@70% 0.60→0.47,
AURC 0.3961→0.5764. v1's previous worst (class 73) did *not* worsen
(AURC 0.3985→0.3933). Classes 81/70/46/80 improve by ≥0.07.

**Readout.** Not a broad minority-class harm. The profile is "v1 plus one
isolated worst-class spike on class 35". To be re-checked across confirmation
seeds before calling either side.

## Preservation

`preservation_manifest.txt` hashes (SHA256) the seed-13 exploratory artifacts
(both datasets: cfg/manifest/utility JSONL/aggregate JSONL/registry/best.pt/
selected.pt), the frozen v1 seed-13 baselines (untouched), and the in-flight
confirmation run dirs (will be regenerated after all 8 finish).