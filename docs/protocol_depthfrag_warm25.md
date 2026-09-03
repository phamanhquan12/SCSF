# Research Protocol: DepthFrag Signal Warmup (E_s=25)

## Pre-registration

**Date**: 2026-09-03
**Author**: SCSF team
**Commit**: 0fda4f8 (server-pinned)

## Hypothesis

DepthFrag's decision-geometry targets are least meaningful at random initialization. A short CE-only warmup phase (E_s=25 epochs) allows the backbone to learn basic features before auxiliary fragility supervision shapes it, potentially improving final selective classification performance.

## Configuration

- **Name**: `depthfrag_warm25`
- **Total epochs**: 300 (unchanged)
- **Warmup epochs**: 25
- **Datasets**: CIFAR-10 and CIFAR-100 (identical E_s value)
- **Backbone**: VGG16-BN (ccl_sc_reference recipe)
- **All other settings**: identical to `depthfrag` under `ccl_sc_reference`

## Warmup Behavior

### Epochs 0–24 (warmup phase)
- Backbone receives classification CE gradients only
- DepthFrag probes and terminal head learn from **detached** backbone features
- Fragility targets remain detached (as in standard DepthFrag)
- No probe/head auxiliary gradient may reach the backbone

### Epochs 25 onward (distillation phase)
- Restore existing end-to-end DepthFrag behavior
- probe_scale=1.0, head_scale=1.0 (unchanged)
- No reset of backbone, probes, optimizer, or scheduler
- No additional epochs added

## Implementation

- Architecture-neutral: no VGG-specific branching
- Uses `trainer.state.epoch`, not global variables or dataset names
- Resume from checkpoints preserves the exact warmup boundary
- Default DepthFrag behavior remains bitwise unchanged when `warmup_epochs=0`
- Logs warmup phase and whether auxiliary gradients are permitted

## Evaluation Plan

### First comparison (CIFAR-10, VGG16-BN, seed 13)
- `depthfrag` E_s=0 vs `depthfrag_warm25` E_s=25
- Same 300-epoch budget and selected-checkpoint rule

### Promotion rules
- warm25 must reduce validation AURC
- Validation accuracy must remain within 1 percentage-point guard
- Failure AUROC should not degrade materially
- Improvement should appear across the risk-coverage curve, not at a single point

### If CIFAR-10 passes
- Run CIFAR-100 seed 13 with E_s=25
- If both pass, confirm with seeds 17, 23, 29, 31
- E_s=25 then becomes a candidate DepthFrag default

### If either dataset fails
- Retain E_s=0
- Record warm25 as a negative ablation
- Do not search E_s separately per dataset

## Tests Required

1. At epoch 24, auxiliary gradients do not reach backbone parameters
2. At epoch 24, probe/head parameters still receive gradients
3. At epoch 25, auxiliary gradients reach the intended backbone prefixes
4. CE gradients reach the backbone in both phases
5. warmup_epochs=0 reproduces existing behavior
6. Resume immediately before and after the boundary is exact
7. Configuration resolves identically for CIFAR-10 and CIFAR-100 except dataset-defined fields
8. Existing test suite remains green

## Constraints

- Do NOT sync to GPU server while current 30-run scheduler is active
- Do NOT launch ablation while scheduler is active
- RiskFlow and SAGE-DS warmup experiments are conditional and NOT launched yet
- Never adopt a 100-epoch warmup merely to match Deep Gamblers
