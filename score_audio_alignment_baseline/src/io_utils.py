from __future__ import annotations

import csv
import json
from pathlib import Path

import librosa
import numpy as np
import partitura


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_score(path: str | Path):
    return partitura.load_score(str(path))


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def save_json(path: str | Path, data: dict | list) -> None:
    path = ensure_parent(path)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    path = ensure_parent(path)
    np.save(path, array)


def save_score_beats_json(path: str | Path, note_array: np.ndarray) -> None:
    unique_onsets = np.unique(note_array["onset_beat"].astype(float))
    data = [{"score_time": float(onset)} for onset in unique_onsets]

    save_json(path, data)


def save_note_array_csv(path: str | Path, note_array: np.ndarray) -> None:
    path = ensure_parent(path)

    if note_array.dtype.names is None:
        raise ValueError("Expected a structured note array with named fields.")

    preferred_fields = [
        "id",
        "pitch",
        "onset_beat",
        "duration_beat",
        "onset_quarter",
        "duration_quarter",
        "voice",
        "staff",
        "measure",
    ]

    fields = [field for field in preferred_fields if field in note_array.dtype.names]

    if not fields:
        fields = list(note_array.dtype.names)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)

        for row in note_array:
            values = []

            for field in fields:
                value = row[field]

                if isinstance(value, np.generic):
                    value = value.item()

                values.append(value)

            writer.writerow(values)


def save_path_csv(path: str | Path, path_indices: np.ndarray) -> None:
    path = ensure_parent(path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_idx", "score_frame_idx", "audio_frame_idx"])

        for path_idx, (score_frame_idx, audio_frame_idx) in enumerate(path_indices):
            writer.writerow(
                [
                    int(path_idx),
                    int(score_frame_idx),
                    int(audio_frame_idx),
                ]
            )
