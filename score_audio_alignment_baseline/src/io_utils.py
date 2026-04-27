from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import partitura
import soundfile as sf


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_score(path: str | Path):
    return partitura.load_score(str(path))


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def save_score_beats_json(path: str | Path, note_array: np.ndarray) -> None:
    path = _ensure_parent(path)
    unique_onsets = np.unique(note_array["onset_beat"].astype(float))
    data = [{"score_time": float(onset)} for onset in unique_onsets]

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    path = _ensure_parent(path)
    np.save(path, array)


def save_json(path: str | Path, data: dict | list) -> None:
    path = _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_audio_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = _ensure_parent(path)
    sf.write(str(path), audio, sr)


def save_note_array_npy(path: str | Path, note_array: np.ndarray) -> None:
    save_npy(path, note_array)


def save_note_array_csv(path: str | Path, note_array: np.ndarray) -> None:
    path = _ensure_parent(path)
    preferred = [
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
    fields = [name for name in preferred if name in note_array.dtype.names]
    if not fields:
        fields = list(note_array.dtype.names)

    with open(path, "w", newline="", encoding="utf-8") as handle:
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


def save_path_times_csv(
    path: str | Path,
    path_indices: np.ndarray,
    score_times: np.ndarray,
    audio_times: np.ndarray,
) -> None:
    path = _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "path_idx",
                "score_frame_idx",
                "audio_frame_idx",
                "score_time",
                "audio_time",
            ]
        )
        for path_idx, (score_idx, audio_idx) in enumerate(path_indices):
            writer.writerow(
                [
                    int(path_idx),
                    int(score_idx),
                    int(audio_idx),
                    float(score_times[score_idx]),
                    float(audio_times[audio_idx]),
                ]
            )


def save_analysis_bundle(
    bundle_dir: str | Path,
    piece_name: str,
    score_path: str | Path,
    audio_path: str | Path,
    audio_full: np.ndarray,
    sr: int,
    note_array: np.ndarray,
    score_times: np.ndarray,
    score_chroma: np.ndarray,
    audio_times: np.ndarray,
    audio_chroma: np.ndarray,
    path: np.ndarray,
    warped_score_chroma: np.ndarray,
    tempomap_raw: np.ndarray,
    tempomap_smooth: np.ndarray,
    start_time: float,
    score_fps: int,
    hop_length: int,
    smooth_window: int,
    copy_inputs: bool = True,
) -> None:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    path_times = np.column_stack((score_times[path[:, 0]], audio_times[path[:, 1]]))

    save_audio_wav(bundle_dir / "audio_full.wav", audio_full, sr)
    save_npy(bundle_dir / "score_times.npy", score_times)
    save_npy(bundle_dir / "score_chroma.npy", score_chroma)
    save_npy(bundle_dir / "audio_times.npy", audio_times)
    save_npy(bundle_dir / "audio_chroma.npy", audio_chroma)
    save_note_array_npy(bundle_dir / "score_note_array.npy", note_array)
    save_note_array_csv(bundle_dir / "score_notes.csv", note_array)
    save_npy(bundle_dir / "path_frames.npy", path)
    save_npy(bundle_dir / "path_times.npy", path_times)
    save_path_times_csv(bundle_dir / "path_times.csv", path, score_times, audio_times)
    save_npy(bundle_dir / "score_chroma_on_audio_time.npy", warped_score_chroma)
    save_npy(bundle_dir / "tempomap_raw.npy", tempomap_raw)
    save_npy(bundle_dir / "tempomap_smooth.npy", tempomap_smooth)

    manifest = {
        "piece_name": piece_name,
        "score_path": str(score_path),
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "audio_hop_length": int(hop_length),
        "score_fps": int(score_fps),
        "trim_start_time": float(start_time),
        "n_score_frames": int(len(score_times)),
        "n_audio_frames": int(len(audio_times)),
        "score_chroma_shape": list(score_chroma.shape),
        "audio_chroma_shape": list(audio_chroma.shape),
        "warped_score_chroma_shape": list(warped_score_chroma.shape),
        "n_path_points": int(len(path)),
        "n_score_notes": int(len(note_array)),
        "smooth_window": int(smooth_window),
        "files": {
            "audio": "audio_full.wav",
            "score_times": "score_times.npy",
            "score_chroma": "score_chroma.npy",
            "audio_times": "audio_times.npy",
            "audio_chroma": "audio_chroma.npy",
            "score_note_array": "score_note_array.npy",
            "score_notes_csv": "score_notes.csv",
            "path_frames": "path_frames.npy",
            "path_times": "path_times.npy",
            "path_times_csv": "path_times.csv",
            "score_chroma_on_audio_time": "score_chroma_on_audio_time.npy",
            "tempomap_raw": "tempomap_raw.npy",
            "tempomap_smooth": "tempomap_smooth.npy",
        },
    }
    save_json(bundle_dir / "manifest.json", manifest)

    if copy_inputs:
        copied_dir = bundle_dir / "inputs"
        copied_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(score_path, copied_dir / Path(score_path).name)
        shutil.copy2(audio_path, copied_dir / Path(audio_path).name)
