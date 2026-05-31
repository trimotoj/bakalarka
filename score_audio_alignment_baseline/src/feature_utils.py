from __future__ import annotations

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of a 2D array, row by row.

    Zero rows stay zero to avoid division by zero.
    """
    if values.ndim != 2:
        raise ValueError("Expected a 2D array.")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return values / safe_norms
