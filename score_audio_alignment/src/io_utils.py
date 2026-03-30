import json
from pathlib import Path

import librosa
import numpy as np
import partitura


def load_score(path: str | Path):
    return partitura.load_score(str(path))


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def save_tempomap_csv(path: str | Path, tempomap: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        tempomap,
        delimiter=",",
        header="score_time,audio_time",
        comments="",
    )


def save_path_csv(path: str | Path, path_points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        path_points,
        delimiter=",",
        header="score_frame,audio_frame",
        comments="",
        fmt="%d",
    )


def save_score_beats_json(path: str | Path, note_array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    unique_onsets = np.unique(note_array["onset_beat"].astype(float))
    data = [{"score_time": float(onset)} for onset in unique_onsets]

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
