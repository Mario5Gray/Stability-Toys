"""Generic backend utilities."""

import numpy as np
from typing import Tuple


def parse_size(size_str: str) -> Tuple[int, int]:
    w_str, h_str = size_str.lower().split("x")
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return w, h


def gen_seed_8_digits() -> int:
    return int(np.random.randint(0, 100_000_000))
