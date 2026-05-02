from __future__ import annotations

import numpy as np


def make_tempomap(
    path: np.ndarray,
    score_times: np.ndarray,
    audio_times: np.ndarray,
) -> np.ndarray:
    """Convert DTW frame path to score-time/audio-time pairs."""
    return np.column_stack((score_times[path[:, 0]], audio_times[path[:, 1]]))


def remove_duplicate_points(tempomap: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate points from a tempomap."""
    if len(tempomap) == 0:
        return tempomap

    keep = np.ones(len(tempomap), dtype=bool)
    keep[1:] = ~np.all(np.isclose(tempomap[1:], tempomap[:-1]), axis=1)

    return tempomap[keep]


def smooth_tempomap(tempomap: np.ndarray, window: int = 9) -> np.ndarray:
    """Smooth only the audio-time axis of the tempomap using a moving average."""
    if window <= 1 or len(tempomap) < window:
        return tempomap.copy()

    if window % 2 == 0:
        raise ValueError("window must be odd")

    result = tempomap.copy()

    pad = window // 2
    padded_audio_times = np.pad(tempomap[:, 1], (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window

    result[:, 1] = np.convolve(padded_audio_times, kernel, mode="valid")

    return result


def tempomap_to_json_data(tempomap: np.ndarray) -> list[dict[str, float]]:
    """Convert a tempomap array to JSON-serializable data."""
    return [
        {
            "score_time": float(score_time),
            "audio_time": float(audio_time),
        }
        for score_time, audio_time in tempomap
    ]
