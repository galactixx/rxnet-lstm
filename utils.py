"""
Utility functions for reproducible machine learning experiments.

Seeds random number generators across Python, NumPy, and PyTorch for
reproducible results, including CUDA operations.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility (incl. CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make CUDA algorithms deterministic where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
