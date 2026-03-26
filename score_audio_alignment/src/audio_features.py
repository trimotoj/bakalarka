import librosa
import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def trim_leading_silence(y: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, int]:
    trimmed, (start_sample, _) = librosa.effects.trim(y, top_db=top_db)
    return trimmed, int(start_sample)


def audio_to_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    method: str = "stft",
) -> np.ndarray:
    if method != "stft":
        raise ValueError("Only stft method is supported")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return normalize_rows(chroma.T)


def get_audio_frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    frames = np.arange(n_frames)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
