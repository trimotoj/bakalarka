import numpy as np


def cosine_cost_matrix(
    score_chroma: np.ndarray, audio_chroma: np.ndarray
) -> np.ndarray:
    sim = np.clip(score_chroma @ audio_chroma.T, -1.0, 1.0)
    return 1.0 - sim
