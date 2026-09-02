"""SCSF next-generation: leakage-free selective classification harness.

Package layout
--------------
scsf.data         deterministic, leakage-free CIFAR-10/CIFAR-100 contracts
scsf.backbones    architecture-native feature-tap registry + BackboneOutput
scsf.methods      CE, DG, SelectiveNet, SAT, SCSF (posthoc/e2e), CCL-SC
scsf.metrics      exact selective-classification metrics
scsf.engine       trainer, evaluator, checkpointing, config, registry
scsf.train        entrypoint: ``python -m scsf.train dataset=cifar10 ...``
scsf.evaluate     entrypoint: ``python -m scsf.evaluate run_dir=... split=test``
scsf.aggregate    entrypoint: ``python -m scsf.aggregate results/registry.csv``

Gradient semantics (README + tests for details):
  SCSF mode ``posthoc``            detach every tapped feature and the logits;
  SCSF mode ``e2e``                allow every intended auxiliary path;
  SCSF mode ``legacy_partial_detach`` (deprecated alias) reproduces the v1
  behaviour in which logits were detached but tapped features were not.
"""

from .version import __version__  # noqa: F401

__all__ = ["__version__"]