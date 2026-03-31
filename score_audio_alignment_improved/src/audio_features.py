import librosa
import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def trim_leading_silence(y: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, int]:
    trimmed, (start_sample, _) = librosa.effects.trim(y, top_db=top_db)
    return trimmed, int(start_sample)


def audio_to_chroma_stft(
    y: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    n_chroma: int = 12,
) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )
    return _normalize_rows(chroma.T)


def audio_to_chroma_cqt(
    y: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    n_chroma: int = 12,
) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )
    return _normalize_rows(chroma.T)


def compute_audio_features(
    y: np.ndarray,
    sr: int,
    backend: str = "chroma_stft",
    hop_length: int = 1024,
    n_chroma: int = 12,
) -> np.ndarray:
    if backend == "chroma_stft":
        return audio_to_chroma_stft(y, sr, hop_length=hop_length, n_chroma=n_chroma)
    if backend == "chroma_cqt":
        return audio_to_chroma_cqt(y, sr, hop_length=hop_length, n_chroma=n_chroma)

    raise ValueError(f"Unknown audio backend: {backend}")


def get_audio_frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    frames = np.arange(n_frames)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
