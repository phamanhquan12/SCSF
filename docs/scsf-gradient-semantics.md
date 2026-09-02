# SCSF gradient semantics (scientific-integrity core)

## What this document is about

SCSF's training loss is

```
L = L_CE(θ) + λ(t) · MSE( ĉ(x; ψ, θ), TCP(x) )
```

where `ĉ = MetaCalibrator(taps(x; θ), logits(x; θ))` predicts the (detached)
True Class Probability `TCP(x) = softmax(logits)[true label]`. The **routes
gradients take through the backbone** determine whether the meta-loss can
modify the classifier at all — and that routing is what v1 got wrong.

## v1 bug audit

The original `train_scsf.py` exposed a boolean `end_to_end`. Its documented
default `end_to_end=False` claimed "post-hoc: block gradients", but the
implementation only detached **logits**. The tapped pool4/pool5 features
still carried the graph into the backbone, so the meta-loss **could** update
the classifier despite the README's claim. That is an integrity bug: any
reported post-hoc result was actually a softly-joint-train result.

We therefore replace the boolean with an explicit three-value `mode` and lock
it with tests in `tests/test_methods.py`.

## The three modes

| `mode` | taps | logits | effect |
|---|---|---|---|
| `posthoc` | `detach()` | `detach()` | meta-loss updates only MetaCalibrator weights (default) |
| `e2e` | flow | flow | meta-loss also trains the backbone (explicit joint training) |
| `legacy_partial_detach` | **flow** | `detach()` | byte-exact reproduction of v1 `end_to_end=False` (deprecated) |

* `posthoc` (correct default): the meta-loss moves nothing but the MLP;
  the backbone sees only the CE gradient.
* `e2e`: deliberately disclosed joint training; enables the meta-loss to shape
  the tapped features.
* `legacy_partial_detach`: *only* kept for v1 compatibility / AB testing; it
  is the exact re-implementation of the old boolean default and must never be
  the default again. It is deprecated (`DeprecationWarning` when constructed
  via the old kwarg).

The deprecated constructor kwarg maps

```
end_to_end=True  -> mode="e2e"
end_to_end=False -> mode="legacy_partial_detach"
```

and always emits a `DeprecationWarning`.

## Where the detach happens

`MetaCalibrator.forward(taps, logits)` (in `scsf/methods/scsf.py`):

* `mode == "posthoc"`: `feats = feats.detach()`, `combined = cat(feats, logits.detach())`.
* `mode == "legacy_partial_detach"`: `feats` kept, `combined = cat(feats, logits.detach())`.
* otherwise (`e2e`): `combined = cat(feats, logits)` with no detach.

`TCP` is always computed under `torch.no_grad()` (`tcp()` in `scsf/methods/scores.py`),
so the target never pulls gradients into the backbone in any mode.

## Meta-weight schedule

The meta-loss magnitude is a cosine schedule over the joint phase
(`meta_weight_cosine` in `scsf/methods/scsf.py`, math locked by tests):

```
w(t) = min_w + 0.5·(start_w - min_w)·(1 - cos(π·p)),   p = (epoch - pretrain) / (total - pretrain)
w = 0 for epoch < pretrain or total <= pretrain
```

Defaults `start_w = 1.0`, `min_w = 1e-4`. During `pretrain` epochs the backbone
trains on CE only; after that the meta-loss kicks in at `min_w` and rises
towards `start_w` by the final epoch. Configured via `method.pretrain`,
`method.init_meta_weight`, `method.min_meta_weight`.

## Recommended use

| intent | mode |
|---|---|
| reproduce "post-hoc calibrator" claims | `posthoc` (default) |
| study joint feature shaping | `e2e` |
| AB against the old script's behavior | `legacy_partial_detach` (deprecated) |

Verify with `python -m pytest tests/test_methods.py -k gradient`.