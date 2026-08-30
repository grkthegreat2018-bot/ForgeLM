"""Synthetic test domain simulator — Rastrigin landscape with shifted optimum."""
from __future__ import annotations

import numpy as np
import torch

from . import register


@register("synthetic_simulate")
def synthetic_simulate(config: dict, domain=None) -> dict:
    """Rastrigin-like function with shifted optimum.

    Expected config keys: {"x": np.ndarray or list of floats in [0,1]}
    """
    dim = 8
    # Parse x — handle JSON round-trip (list or string repr of ndarray)
    x_val = config.get("x")
    if x_val is None:
        # Fallback: extract numeric values from config
        nums = [v for v in config.values() if isinstance(v, (int, float))]
        x = np.array(nums[:dim], dtype=np.float64)
        if len(x) < dim:
            x = np.pad(x, (0, dim - len(x)))
    elif isinstance(x_val, str):
        s = x_val.strip().strip('[]')
        x = np.array([float(v) for v in s.split()], dtype=np.float64)
    else:
        x = np.array(x_val, dtype=np.float64)

    # Fixed optimum (seed=42 from original SyntheticDomain)
    rng = np.random.RandomState(42)
    optimum = rng.rand(dim)
    shifted = x - optimum
    rastrigin = 10 * dim + np.sum(
        (shifted * 5) ** 2 - 10 * np.cos(2 * np.pi * shifted * 5)
    )
    deceptive = 2 * np.sum(np.cos(3 * np.pi * x))
    # Score shifted to positive (matches original: -(rastrigin + deceptive) + 50)
    raw_score = -(rastrigin + deceptive) + 50
    return {
        "raw_score": float(raw_score),
        "rastrigin": float(rastrigin),
        "deceptive": float(deceptive),
        "param_mean": float(np.mean(x)),
        "param_std": float(np.std(x)),
        "behavioral_0": float(np.mean(x)),
        "behavioral_1": float(np.std(x)),
    }
