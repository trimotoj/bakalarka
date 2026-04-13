from __future__ import annotations

import numpy as np


DTW_STEPS = [(-1, -1), (-1, 0), (0, -1)]


def _validate_cost(cost: np.ndarray) -> None:
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")
    if 0 in cost.shape:
        raise ValueError("cost must not be empty")


def _accumulate_cost(cost: np.ndarray) -> np.ndarray:
    n_rows, n_cols = cost.shape
    acc = np.full((n_rows, n_cols), np.inf, dtype=float)
    acc[0, 0] = float(cost[0, 0])

    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0 and j == 0:
                continue

            best_prev = min(
                acc[i + di, j + dj]
                for di, dj in DTW_STEPS
                if i + di >= 0 and j + dj >= 0
            )
            acc[i, j] = float(cost[i, j]) + best_prev

    return acc


def backtrack_path(acc: np.ndarray) -> np.ndarray:
    """Recover the optimal DTW path from an accumulated cost matrix."""
    i = acc.shape[0] - 1
    j = acc.shape[1] - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        candidates = []
        if i > 0 and j > 0:
            candidates.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((acc[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((acc[i, j - 1], i, j - 1))

        _, i, j = min(candidates, key=lambda item: item[0])
        path.append((i, j))

    path.reverse()
    return np.asarray(path, dtype=int)


def dtw(cost: np.ndarray) -> np.ndarray:
    """Compute the forward DTW path."""
    _validate_cost(cost)
    acc = _accumulate_cost(cost)
    return backtrack_path(acc)


def dtw_reverse(cost: np.ndarray) -> np.ndarray:
    """Compute a DTW path by running DTW on the reversed cost matrix."""
    _validate_cost(cost)
    n_rows, n_cols = cost.shape

    reversed_cost = np.flip(cost, axis=(0, 1))
    reversed_path = dtw(reversed_cost)

    path = reversed_path.copy()
    path[:, 0] = n_rows - 1 - path[:, 0]
    path[:, 1] = n_cols - 1 - path[:, 1]
    return path[::-1]
