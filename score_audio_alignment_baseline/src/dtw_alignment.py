from __future__ import annotations

import numpy as np

DTW_STEPS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (0, -1))


def _validate_cost_matrix(cost: np.ndarray) -> None:
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")
    if 0 in cost.shape:
        raise ValueError("cost must not be empty")


def _accumulate_cost(cost: np.ndarray) -> np.ndarray:
    n_rows, n_cols = cost.shape
    accumulated = np.full((n_rows, n_cols), np.inf, dtype=float)
    accumulated[0, 0] = float(cost[0, 0])

    for row in range(n_rows):
        for col in range(n_cols):
            if row == 0 and col == 0:
                continue

            previous_cost = min(
                accumulated[row + row_step, col + col_step]
                for row_step, col_step in DTW_STEPS
                if row + row_step >= 0 and col + col_step >= 0
            )
            accumulated[row, col] = float(cost[row, col]) + previous_cost

    return accumulated


def backtrack_path(accumulated: np.ndarray) -> np.ndarray:
    """Backtrack the optimal DTW path from an accumulated cost matrix."""
    _validate_cost_matrix(accumulated)

    row = accumulated.shape[0] - 1
    col = accumulated.shape[1] - 1
    path = [(row, col)]

    while row > 0 or col > 0:
        candidates: list[tuple[float, int, int]] = []
        if row > 0 and col > 0:
            candidates.append((accumulated[row - 1, col - 1], row - 1, col - 1))
        if row > 0:
            candidates.append((accumulated[row - 1, col], row - 1, col))
        if col > 0:
            candidates.append((accumulated[row, col - 1], row, col - 1))

        _, row, col = min(candidates, key=lambda item: item[0])
        path.append((row, col))

    path.reverse()
    return np.asarray(path, dtype=int)


def dtw(cost: np.ndarray) -> np.ndarray:
    """Return the optimal DTW path for a pairwise cost matrix."""
    _validate_cost_matrix(cost)
    accumulated = _accumulate_cost(cost)
    return backtrack_path(accumulated)
