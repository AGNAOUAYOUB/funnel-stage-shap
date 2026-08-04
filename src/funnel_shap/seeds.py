"""Determinism controls (protocol Sec. 6.2, Appendix B).

The seed list is frozen. Every reported number is a mean +/- std over these five
runs; single-run results are never reported.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEEDS: tuple[int, ...] = (7, 17, 23, 42, 101)


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    torch is imported lazily so that the tabular arm runs without it installed.
    """
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not in the frozen seed list {SEEDS}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # cuBLAS needs this set before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
