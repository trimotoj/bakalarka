from __future__ import annotations

import librosa
import numpy as np

from src.feature_utils import normalize_rows

DEFAULT_TRIM_DB = 30.0


def trim_leading_silence(
    y: np.ndarray,
    top_db: float = DEFAULT_TRIM_DB,
) -> tuple[np.ndarray, int]:
    """Trim leading and trailing silence and return the trimmed signal and start sample."""
    trimmed, (start_sample, _) = librosa.effects.trim(y, top_db=top_db)
    return trimmed, int(start_sample)


def audio_to_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = 1024,
) -> np.ndarray:
    """Convert audio to frame-wise normalized chroma."""
    if sr <= 0:
        raise ValueError("sr must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return normalize_rows(chroma.T)


def get_audio_frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    """Return the time, in seconds, of each audio frame."""
    if n_frames < 0:
        raise ValueError("n_frames must not be negative")
    if sr <= 0:
        raise ValueError("sr must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    frames = np.arange(n_frames)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
