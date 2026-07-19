"""
random_seed.py
---------------
Centralized reproducibility helper. Every pipeline stage that touches
randomness (train/test split, model initialization, etc.) imports and
calls `set_global_seed()` at the start of its `main()`, so all runs are
deterministic given the same seed value in params.yaml.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[random_seed] Global random seed set to {seed}")
    return seed

if __name__ == "__main__":
    set_global_seed(42)