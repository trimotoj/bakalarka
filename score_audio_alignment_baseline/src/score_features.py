from __future__ import annotations

import numpy as np
import partitura

from src.feature_utils import normalize_rows

N_CHROMA_BINS = 12


def score_to_note_array(score) -> np.ndarray:
    """Convert a Partitura score object to a structured note array."""
    return partitura.utils.ensure_notearray(score)


def build_score_time_grid(note_array: np.ndarray, fps: int = 40) -> np.ndarray:
    """Create an evenly sampled score-time grid in beats."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if len(note_array) == 0:
        raise ValueError("note_array must not be empty")

    step = 1.0 / fps
    onsets = note_array["onset_beat"].astype(float)
    offsets = onsets + note_array["duration_beat"].astype(float)

    start = float(np.floor(onsets.min() / step) * step)
    end = float(np.ceil(offsets.max() / step) * step)

    return np.arange(start, end + step, step)


def note_array_to_chroma(note_array: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    """Render score notes into a frame-wise chroma representation."""
    if len(note_array) == 0:
        raise ValueError("note_array must not be empty")

    onsets = note_array["onset_beat"].astype(float)
    offsets = onsets + note_array["duration_beat"].astype(float)
    pitches = note_array["pitch"].astype(int)

    chroma = np.zeros((len(frame_times), N_CHROMA_BINS), dtype=float)

    for frame_idx, time_point in enumerate(frame_times):
        active = (onsets <= time_point) & (time_point < offsets)
        if not np.any(active):
            continue

        pitch_classes = pitches[active] % N_CHROMA_BINS
        chroma[frame_idx] = np.bincount(
            pitch_classes,
            minlength=N_CHROMA_BINS,
        ).astype(float)

    return normalize_rows(chroma)
