"""Deterministic seeding and RNG-state snapshots for exact resume."""

from __future__ import annotations

import os
import random

import numpy as np

import torch

_DEFAULT = {"seed": 13, "torch_threads": 4}


def seed_all(seed: int, benchmark: bool = False, deterministic: bool = True) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark


def capture_global_state(device=None, generator=None) -> dict:
    """Snapshot every RNG whose drift would change a resumed run."""
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = {
            i: torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        }
    if generator is not None:
        state["generator"] = generator.get_state().clone(),
    return state


def restore_global_state(state: dict, generator=None) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state:
        for i, st in state["torch_cuda"].items():
            torch.cuda.set_rng_state(st.cpu(), i)
    if generator is not None and "generator" in state:
        generator.set_state(state["generator"][0])


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g