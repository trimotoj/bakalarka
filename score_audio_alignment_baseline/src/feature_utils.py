from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D array."""
    if x.ndim != 2:
        raise ValueError("Expected a 2D array.")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0

    return x / norms
