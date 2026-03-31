import numpy as np
import partitura


def score_to_note_array(score) -> np.ndarray:
    return partitura.utils.ensure_notearray(score)


def build_score_time_grid(note_array: np.ndarray, fps: int = 40) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    step = 1.0 / fps
    onsets = note_array["onset_beat"].astype(float)
    durations = note_array["duration_beat"].astype(float)
    offsets = onsets + durations

    start = float(np.floor(onsets.min() / step) * step)
    end = float(np.ceil(offsets.max() / step) * step)

    return np.arange(start, end + step, step)


def note_array_to_chroma(note_array: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    onsets = note_array["onset_beat"].astype(float)
    durations = note_array["duration_beat"].astype(float)
    offsets = onsets + durations
    pitches = note_array["pitch"].astype(int)

    chroma = np.zeros((len(frame_times), 12), dtype=float)

    for i, t in enumerate(frame_times):
        active = (onsets <= t) & (t < offsets)
        if not np.any(active):
            continue

        pitch_classes = pitches[active] % 12
        values = np.bincount(pitch_classes, minlength=12).astype(float)
        norm = np.linalg.norm(values)
        if norm > 0:
            chroma[i] = values / norm

    return chroma
