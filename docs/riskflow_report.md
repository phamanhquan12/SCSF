# RiskFlow — implementation report

Method: **RiskFlow** (persistent sequential failure-evidence state with
conditional innovation supervision). This is *not* a weighted sum of
independent intermediate confidence heads.

- Pushed commit: `method(riskflow): accumulate conditional failure innovations`
- Files: `scsf/methods/riskflow.py`, `scsf/riskflow/{__init__,cka,diagnostics,overhead}.py`,
  7 method configs, `tests/test_riskflow.py` (27 tests), `scripts/smoke_riskflow.{py,sh}`, `docs/riskflow.md`.
- Baseline commit: `depthfrag` `95b7ef7`; this change adds RiskFlow on top.

## Why this is not TULIP / deep exits

See `docs/riskflow.md` for the full write-up. In short: TULIP/deep exits
supervise independent per-layer confidence predictors and combine them after
the fact; RiskFlow supervises the **conditional residual** (the change in
failure evidence the current depth adds relative to the previous cumulative
state) and **persists one state through depth**. The novelty claim (residual
updates are less redundant while cumulative risk is more predictive) is
measured directly below, not assumed.

## Architecture gist

```text
r_0 = learned base-risk state
delta_r_l, gate_logit_l = psi_l(adapter_l(h_l), r_{l-1})
r_l = r_{l-1} + sigmoid(gate_logit_l) * delta_r_l
s_l = readout_hard(r_l);   q_l = sigmoid(s_l)
```

Shared `RiskCell` (delta + optional gate nets) + shared readouts + per-example
gate trajectory; architecture-specific only in the per-stage **input adapters**
(LayerNorm + Linear into a `state_dim`=64 space). ViT consumes CLS token mean.
Soft channel (`readout_soft`) is auxiliary and excluded from deployment counts.
Loss: CE + per-depth Huber innovation losses vs detached pseudo-residuals
`epsilon_l = stopgrad(e - sigmoid(s_{l-1}))`, terminal proper-scoring BCE on
`s_L`, plus a configurable decorrelation penalty (L2-normalized innovations,
off-diagonal abs mean — robust to zero variance). The final gate is **not** run
or claimed at deployment; inference score is `confidence = -s_hard_L`.

## Required comparisons (one class, config-switched)

| config | mechanism |
|--------|-----------|
| `riskflow_concat` | comparison 1: SCSF-style concatenation head |
| `riskflow_heads` | comparison 2: independent heads + learned weighted sum |
| `riskflow_cum` | comparison 3: cumulative state, no residual targets |
| `riskflow_resid` | comparison 4: cumulative state + residual innovation targets |
| `riskflow` (default) | comparison 5: residual innovation + sample-dependent gates |
| `riskflow_frozen` | comparison 6: frozen-backbone RiskFlow |
| `riskflow_hard` | comparison 7: hard-error channel only (soft off) |

All share the `RiskFlowMethod` class (`variant`, `use_soft`, `freeze_backbone`,
`gate_mode`). All load/predict; the config load + ablation tests run in the
suite.

## Tests

`tests/test_riskflow.py` — 27 tests covering every bullet in the prompt:

- zero updates → final state equals base state
- constructed residual sequence sums to expected final logit
- stop-gradient boundaries
- gates differ across constructed samples (+ fixed-gates sample-independent)
- state ordering follows the backbone registry deterministically
- constant-variance batch → no NaN in decorrelation loss
- checkpoint resume reproduces state/gate outputs
- tiny ResNet-18 + DeiT-S smoke emit all artifacts (cuda-gated)

Results (server, torch 2.6, cu124):
`117 passed, 4 skipped` full suite (skips = smoke-gated on `SCSF_RUN_SMOKE`).
Enabling smokes: `SCSF_RUN_SMOKE=1 pytest tests/test_riskflow.py::test_riskflow_smoke_artifacts`
→ **2 passed** (real trained tiny runs, artifacts asserted).

## Smoke commands (server)

```bash
# full artifact job for both backbones
bash /mnt/d/OPD/scsf/scripts/smoke_riskflow.sh
# or single backbone
SCSF_DATA_ROOT=/root/scsf/data \
/root/scsf_venv/bin/python scripts/smoke_riskflow.py \
  --backbone resnet18 --results-root /root/scsf_scratch/riskflow_smoke \
  --data-root /root/scsf/data --device cuda --epochs 1 --overfit 64 --trace-subset 128
```

Artifacts per backbone (under the run's `riskflow_smoke/`):
`trace_train.npz` (gates, innovation logits/vectors, cumulative risk, hard+soft
pseudo-residual targets), `smoke_summary.json`, `trajectory_<category>.png`
(fixed-rule category assignment from config thresholds `cat_lo` / `cat_hi = 0.3 / 0.7`).

## Redundancy diagnostics (micro-epoch protocol, trained-vs-trained)

Cross-depth correlation and centered linear CKA for independent-head outputs
vs RiskFlow innovations, and cumulative-risk correlation with the final score:

| backbone | CKA heads | CKA innov | corr heads | corr innov | innov/CKA ratio | innov/corr ratio | cum↔final r |
|----------|-----------|-----------|------------|------------|-----------------|------------------|-------------|
| ResNet-18 | 0.128 | 0.098 | 0.322 | 0.225 | **0.77** | **0.70** | 0.54 |
| DeiT-S | 0.793 | 0.530 | 0.886 | 0.674 | **0.67** | **0.76** | 0.98 |

On both backbones the residual innovations are **less** redundant than
independent heads while the cumulative risk is predictive of the final score —
directional support for the novelty mechanism. *Caveat:* these are 1-epoch,
64-sample overfit micro runs that exercise the code paths, not full
training-protocol numbers; the same diagnostics rerun unchanged on any real run.

## Deployment overhead (RiskFlow vs SCSF concat baseline)

Measured on the micro runs (deployment modules only; backbone excluded from the
MACs estimate; latency wall-clock on cuda, batch 16, 32×32):

| backbone | method | deploy params | added MACs/ex | latency ms | peak mem MiB |
|----------|--------|---------------|---------------|------------|--------------|
| ResNet-18 | concat | 11,558,987 | 6,033,408 | 2.75 | 307 |
| ResNet-18 | RiskFlow | 11,257,932 | 1,803,264 | 4.29 | 307 |
| DeiT-S | concat | 22,620,811 | 20,451,328 | 5.18 | 2850 |
| DeiT-S | RiskFlow | 21,659,916 | 7,202,816 | 9.75 | 2850 |

RiskFlow has **fewer** deployment parameters and **fewer** added MACs than the
concat head (the recurrent cell is cheap), but **higher latency** (sequential
depth-wise state recurrence) at identical peak memory. Nonzero overhead is
real and quantified as required.

## Notes

- The final gate is not run or claimed at deployment; it is a training-time
  per-example modulator whose trajectory is diagnostic.
- Do not reopen killed/parked methods on BGE/MiniLM; AQR not implemented
  (per AGENTS.md).
