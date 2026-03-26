import numpy as np


DTW_STEPS = [(-1, -1), (-1, 0), (0, -1)]


def dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost.ndim != 2:
        raise ValueError("cost must be 2D")
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost must not be empty")

    n_rows, n_cols = cost.shape
    acc = np.full((n_rows, n_cols), np.inf, dtype=float)
    acc[0, 0] = float(cost[0, 0])

    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0 and j == 0:
                continue

            best_prev = np.inf
            for di, dj in DTW_STEPS:
                pi = i + di
                pj = j + dj
                if pi >= 0 and pj >= 0:
                    if acc[pi, pj] < best_prev:
                        best_prev = acc[pi, pj]

            acc[i, j] = float(cost[i, j]) + best_prev

    path = backtrack_path(acc)
    return acc, path


def backtrack_path(acc: np.ndarray) -> np.ndarray:
    i = acc.shape[0] - 1
    j = acc.shape[1] - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        options = []

        if i > 0 and j > 0:
            options.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            options.append((acc[i - 1, j], i - 1, j))
        if j > 0:
            options.append((acc[i, j - 1], i, j - 1))

        _, i, j = min(options, key=lambda x: x[0])
        path.append((i, j))

    path.reverse()
    return np.asarray(path, dtype=int)
