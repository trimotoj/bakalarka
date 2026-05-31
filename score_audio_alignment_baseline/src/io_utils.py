from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import partitura

PathLike = str | Path
NOTE_ARRAY_EXPORT_FIELDS = (
    "id",
    "pitch",
    "onset_beat",
    "duration_beat",
    "onset_quarter",
    "duration_quarter",
    "voice",
    "staff",
    "measure",
)


def ensure_parent(path: PathLike) -> Path:
    """Create a file's parent directory and return the normalized path."""
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def load_score(path: PathLike):
    """Load a symbolic score using Partitura."""
    return partitura.load_score(str(path))


def load_audio(path: PathLike, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load mono audio with librosa."""
    audio, loaded_sample_rate = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio, loaded_sample_rate


def save_json(path: PathLike, data: Any) -> None:
    """Save JSON with UTF-8 encoding and readable indentation."""
    normalized_path = ensure_parent(path)

    with normalized_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_npy(path: PathLike, array: np.ndarray) -> None:
    """Save a NumPy array and create the parent directory if needed."""
    np.save(ensure_parent(path), array)


def save_score_beats_json(path: PathLike, note_array: np.ndarray) -> None:
    """Save unique score onset positions in beats."""
    unique_onsets = np.unique(note_array["onset_beat"].astype(float))
    data = [{"score_time": float(onset)} for onset in unique_onsets]
    save_json(path, data)


def _select_note_array_fields(note_array: np.ndarray) -> list[str]:
    if note_array.dtype.names is None:
        raise ValueError("Expected a structured note array with named fields.")

    fields = [
        field for field in NOTE_ARRAY_EXPORT_FIELDS if field in note_array.dtype.names
    ]
    return fields or list(note_array.dtype.names)


def _to_plain_python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_note_array_csv(path: PathLike, note_array: np.ndarray) -> None:
    """Save a structured note array as CSV."""
    normalized_path = ensure_parent(path)
    fields = _select_note_array_fields(note_array)

    with normalized_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)

        for row in note_array:
            writer.writerow([_to_plain_python_value(row[field]) for field in fields])


def save_path_csv(path: PathLike, path_indices: np.ndarray) -> None:
    """Save the DTW path as ``path_idx, score_frame_idx, audio_frame_idx``."""
    normalized_path = ensure_parent(path)

    with normalized_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_idx", "score_frame_idx", "audio_frame_idx"])

        for path_idx, (score_frame_idx, audio_frame_idx) in enumerate(path_indices):
            writer.writerow([int(path_idx), int(score_frame_idx), int(audio_frame_idx)])
