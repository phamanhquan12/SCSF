# Audit: `meta_weight_cosine` schedule is inverted

Audited on branch `quan` at `1c80ad4e3101f03ff3e011b7215d610fb5620d2d`.

## Finding

`scsf/methods/scsf.py:meta_weight_cosine(epoch, pretrain, total_epochs,
start_weight=1.0, min_weight=1e-4)` is documented as a *cosine-decayed* meta
loss weight and returns:

```python
progress = clamp((epoch - pretrain) / (total_epochs - pretrain), 0, 1)
return min_weight + 0.5 * (start_weight - min_weight) * (1.0 - math.cos(math.pi * progress))
```

Evaluating the factor `0.5*(1 - cos(pi*progress))`:

| progress | factor | value |
|----------|--------|-------|
| 0.0      | 0.0    | `min_weight`   (the minimum) |
| 0.5      | 0.5    | `(min+start)/2` |
| 1.0      | 1.0    | `start_weight` (the maximum) |

With the v1 defaults the schedule **starts at the minimum** (`1e-4`) and
**climbs to the maximum** (`1.0`) over the joint phase — i.e. exactly inverted
relative to the "decay" description and to the review's `λ_max=1 → λ_min=1e-4`
specification.

Because `cos(pi·t)` is monotone decreasing on `[0,1]`, `(1 − cos(pi·t))` is
monotone **increasing**; a faithful decreasing schedule needs `(1 + cos)`.
The correct factor for a decay from `start_weight` to `min_weight` is
`0.5*(1 + cos(pi*progress))`.

For reference, the review's locked reference describes cosine decay from
`λ_max = 1` down to `λ_min = 10^-4`.

## Impact / callers

* `scsf/methods/scsf.py:191` — SCSFMethod.train_loss uses it with
  `init_meta_weight`, `min_meta_weight`; so the joint-phase meta loss weight
  **grows** over training instead of tapering.
* `train_scsf.py:1534` — legacy script, same helper.

## Decision

The existing helper, its callers and the locked schedule test
(`tests/test_methods.py::test_scsf_meta_weight_cosine_schedule`) are the
historical TCP–MSE baseline's contract; changing them would silently alter
historical method behavior and checkpoint comparability. **No silent fix.**

* Legacy behavior is preserved and re-documented here as unintentionally
  inverted.
* The new family (`scsf_correctness`, `r3_scsf`, `dtr_scsf`, `cbr_scsf`)
  uses a new, correctly named helper
  `rc_training/schedules.py:meta_weight_cosine_decay` with factor
  `0.5*(1 + cos(pi*progress))`: starts at `start_weight`, ends at
  `min_weight`, returns `0.0` before `pretrain`, `progress` clamped to
  `[0,1]`. Endpoints and the one-epoch edge cases are locked by tests
  (`tests/`; see `docs/RC_TRAINING_IMPLEMENTATION.md` §3.6).

## Tests

* Existing legacy test stays green unchanged (locked historical contract).
* New tests assert `meta_weight_cosine_decay(epoch<pretrain) == 0.0`,
  `== start_weight` at joint-phase start, `== min_weight` at the final epoch,
  monotone non-increasing across the joint phase, and identical values for
  `total_epochs == pretrain` boundary (`0.0`). These tests are authored and
  **NOT RUN** in this delivery; commands in `docs/VALIDATION_STATUS.md`.