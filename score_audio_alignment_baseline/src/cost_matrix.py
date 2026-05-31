from __future__ import annotations

import numpy as np


def cosine_cost_matrix(
    score_chroma: np.ndarray,
    audio_chroma: np.ndarray,
) -> np.ndarray:
    """Compute cosine distance between score and audio chroma frames.

    The input chroma rows are expected to be L2-normalized. The result has shape
    ``(n_score_frames, n_audio_frames)``.
    """
    if score_chroma.ndim != 2 or audio_chroma.ndim != 2:
        raise ValueError("score_chroma and audio_chroma must be 2D arrays")
    if score_chroma.shape[1] != audio_chroma.shape[1]:
        raise ValueError(
            "score_chroma and audio_chroma must have the same feature size"
        )

    similarity = np.clip(score_chroma @ audio_chroma.T, -1.0, 1.0)
    return 1.0 - similarity
