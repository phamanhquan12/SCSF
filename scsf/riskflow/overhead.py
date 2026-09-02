"""RiskFlow deployment-overhead accounting.

RiskFlow has a nonzero inference overhead relative to a plain backbone or the
SCSF concatenation baseline: the persistent state pass runs per-depth adapters,
a shared update cell, and the readout at every depth. This module reports that
overhead directly.

* ``deployment_params(method)`` — inference parameters (persistent modules only;
  the auxiliary soft readout is excluded).
* ``added_macs(method, batch=1)`` — **analytic** multiply-accumulate estimate of
  the RiskFlow-specific modules added on top of the backbone (adapters, cell,
  readouts, base state), per example. This is an analytic count (one MAC per
  weight-element product), documented as such, not a FLOP-profiler measurement.
* ``measure_latency_memory(method, x)`` — wall-clock latency (ms) and peak
  tensor memory (MiB) on the device of ``x``; memory requires CUDA.

``report_overhead(method)`` bundles the three into one JSON-serializable dict.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..methods.riskflow import RiskFlowMethod


def deployment_params(method: RiskFlowMethod) -> int:
    return sum(p.numel() for mod in method.inference_modules() for p in mod.parameters())


def added_macs(method: RiskFlowMethod, batch: int = 1) -> float:
    """Analytic per-example MACs of the modules RiskFlow adds on the backbone.

    Only the riskflow mechanism is counted (adapters + shared cell + readouts);
    the backbone is identical across the compared methods. Recurrent modes use
    ``L`` sites, state dim ``D``, cell hidden ``H``.
    """
    if method.variant == "concat":
        inner = 256
        in_dim = sum(a.proj.in_features for a in method.adapters.values()) \
            + method.backbone.final_dim
        macs = in_dim * inner + inner * 1
        return float(macs * batch)
    if method.variant == "heads":
        per = 0.0
        for s in method.site_names:
            a = method.adapters[s]
            per += a.proj.in_features + a.proj.in_features * method.state_dim
        per += len(method.site_names) * method.state_dim   # per-depth head
        return float(per * batch)

    L = len(method.site_names)
    D = method.state_dim
    H = method.cell_hidden
    per = 0.0
    for s in method.site_names:
        a = method.adapters[s]
        d_in = a.proj.in_features
        per += d_in + d_in * D                        # LayerNorm + Linear
    per += L * (2 * D * H + H * D)                     # shared cell.update
    if method.gate_mode == "sample":
        per += L * (2 * D)                              # shared cell.gate
    per += (L + 1) * D                                  # readout_hard at each depth
    if method.readout_soft is not None:
        per += (L + 1) * D                              # readout_soft (also runs)
    return float(per * batch)


def measure_latency_memory(method: RiskFlowMethod, x: torch.Tensor,
                           warmup: int = 2, reps: int = 5) -> Dict[str, float]:
    device = x.device
    method.eval()
    with torch.no_grad():
        for _ in range(warmup):
            method.predict_batch(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(reps):
            method.predict_batch(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1000.0
        mem_mib = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == "cuda" else 0.0)
    return {"latency_ms": float(ms), "peak_mem_mib": float(mem_mib)}


def report_overhead(method: RiskFlowMethod, x: torch.Tensor) -> Dict[str, float]:
    base = measure_latency_memory(method, x) if x is not None else {
        "latency_ms": 0.0, "peak_mem_mib": 0.0}
    return {
        "deployment_params": int(deployment_params(method)),
        "added_macs_per_example": float(added_macs(method, batch=max(1, x.shape[0]))),
        **base,
    }


__all__ = ["added_macs", "deployment_params", "measure_latency_memory", "report_overhead"]
