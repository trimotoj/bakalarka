from __future__ import annotations

import librosa
import numpy as np

from src.feature_utils import normalize_rows

DEFAULT_TRIM_DB = 30.0


def trim_leading_silence(
    audio: np.ndarray,
    top_db: float = DEFAULT_TRIM_DB,
) -> tuple[np.ndarray, int]:
    """Trim silence and return ``(trimmed_audio, start_sample)``."""
    trimmed_audio, (start_sample, _) = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio, int(start_sample)


def audio_to_chroma(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 1024,
) -> np.ndarray:
    """Convert audio to frame-wise L2-normalized chroma features."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    return normalize_rows(chroma.T)


def get_audio_frame_times(
    n_frames: int,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    """Return audio-frame timestamps in seconds."""
    if n_frames < 0:
        raise ValueError("n_frames must not be negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0")

    frames = np.arange(n_frames)
    return librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
