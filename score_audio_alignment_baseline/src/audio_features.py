from __future__ import annotations

import librosa
import numpy as np


DEFAULT_TRIM_DB = 30.0


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


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
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return _normalize_rows(chroma.T)


def get_audio_frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    """Return the time, in seconds, of each audio frame."""
    frames = np.arange(n_frames)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
