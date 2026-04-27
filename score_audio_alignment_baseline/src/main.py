from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.audio_features import (
    audio_to_chroma,
    get_audio_frame_times,
    trim_leading_silence,
)
from src.cost_matrix import cosine_cost_matrix
from src.dtw_alignment import dtw
from src.io_utils import (
    load_audio,
    load_score,
    save_score_beats_json,
)
from src.score_features import (
    build_score_time_grid,
    note_array_to_chroma,
    score_to_note_array,
)
from src.tempomap import (
    export_path_csv,
    export_tempomap_json,
    make_tempomap,
    remove_duplicate_points,
    smooth_tempomap,
)
from src.visualization import (
    plot_aligned_chromas,
    plot_chroma,
    plot_cost_matrix_with_path,
    plot_local_cost_matrix_with_path,
    plot_tempomap,
    warp_score_chroma_to_audio_time,
)


SCORE_PATH = Path("data/score/misatango-gloria.musicxml")
AUDIO_PATH = Path("data/audio/misatango-gloria.wav")
OUTPUT_DIR = Path("data/output")

AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512
SCORE_FPS = 40
SMOOTH_WINDOW = 9
SHOW_PLOTS = False


def get_piece_name(score_path: Path, audio_path: Path) -> str:
    return audio_path.stem or score_path.stem


def build_output_paths(piece_name: str) -> dict[str, Path]:
    piece_dir = OUTPUT_DIR / piece_name
    exports_dir = piece_dir / "exports"
    plots_dir = piece_dir / "plots"
    analysis_dir = piece_dir / "analysis"

    return {
        "piece_dir": piece_dir,
        "exports_dir": exports_dir,
        "plots_dir": plots_dir,
        "analysis_dir": analysis_dir,
        "score_beats_json": exports_dir / "score_beats.json",
        "path_csv": exports_dir / "path.csv",
        "tempomap_raw_json": exports_dir / "tempomap_raw.json",
        "tempomap_smooth_json": exports_dir / "tempomap.json",
        "score_chroma_plot": plots_dir / "score_chroma.png",
        "audio_chroma_plot": plots_dir / "audio_chroma.png",
        "cost_plot": plots_dir / "cost_matrix_with_path.png",
        "tempomap_plot": plots_dir / "tempomap.png",
        "aligned_plot": plots_dir / "aligned_chromas.png",
        "local_cost_plot": plots_dir / "local_cost_beat2.png",
    }


def ensure_output_dirs(output_paths: dict[str, Path]) -> None:
    for key in ("piece_dir", "exports_dir", "plots_dir", "analysis_dir"):
        output_paths[key].mkdir(parents=True, exist_ok=True)


def build_tempomap(path: np.ndarray, score_times: np.ndarray, audio_times: np.ndarray):
    tempomap_raw = make_tempomap(path, score_times, audio_times)
    tempomap_raw = remove_duplicate_points(tempomap_raw)
    tempomap_smooth = smooth_tempomap(tempomap_raw, window=SMOOTH_WINDOW)
    return tempomap_raw, tempomap_smooth


def save_note_array_csv(note_array: np.ndarray, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if note_array.dtype.names is None:
        raise ValueError("Expected a structured note array with named fields.")

    fieldnames = list(note_array.dtype.names)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for row in note_array:
            values = []
            for name in fieldnames:
                value = row[name]
                if isinstance(value, np.generic):
                    value = value.item()
                values.append(value)
            writer.writerow(values)


def save_notebook_analysis_bundle(
    analysis_dir: Path,
    note_array: np.ndarray,
    score_times: np.ndarray,
    score_chroma: np.ndarray,
    audio_times: np.ndarray,
    audio_chroma: np.ndarray,
    path: np.ndarray,
    warped_score_chroma: np.ndarray,
    tempomap_raw: np.ndarray,
    tempomap_smooth: np.ndarray,
) -> None:
    """
    Save only the files required by the lightweight analysis notebook
    that works without uploading the original WAV file.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)

    np.save(analysis_dir / "audio_times.npy", np.asarray(audio_times))
    np.save(analysis_dir / "audio_chroma.npy", np.asarray(audio_chroma))
    np.save(analysis_dir / "score_times.npy", np.asarray(score_times))
    np.save(analysis_dir / "score_chroma.npy", np.asarray(score_chroma))
    np.save(
        analysis_dir / "score_chroma_on_audio_time.npy", np.asarray(warped_score_chroma)
    )
    np.save(analysis_dir / "path_frames.npy", np.asarray(path, dtype=int))
    np.save(analysis_dir / "tempomap_raw.npy", np.asarray(tempomap_raw))
    np.save(analysis_dir / "tempomap_smooth.npy", np.asarray(tempomap_smooth))

    save_note_array_csv(note_array, analysis_dir / "score_notes.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    piece_name = get_piece_name(SCORE_PATH, AUDIO_PATH)
    output_paths = build_output_paths(piece_name)
    ensure_output_dirs(output_paths)

    score = load_score(SCORE_PATH)
    note_array = score_to_note_array(score)

    audio_full, sr = load_audio(AUDIO_PATH, sr=AUDIO_SR)
    audio_trimmed, start_sample = trim_leading_silence(audio_full, top_db=30)
    start_time = start_sample / sr

    score_times = build_score_time_grid(note_array, fps=SCORE_FPS)
    score_chroma = note_array_to_chroma(note_array, score_times)

    audio_chroma = audio_to_chroma(audio_trimmed, sr, hop_length=AUDIO_HOP_LENGTH)
    audio_times = (
        get_audio_frame_times(len(audio_chroma), sr, AUDIO_HOP_LENGTH) + start_time
    )

    cost = cosine_cost_matrix(score_chroma, audio_chroma)
    path = dtw(cost)
    tempomap_raw, tempomap_smooth = build_tempomap(path, score_times, audio_times)
    warped_score_chroma = warp_score_chroma_to_audio_time(
        score_chroma,
        path,
        n_audio_frames=len(audio_chroma),
    )

    save_score_beats_json(output_paths["score_beats_json"], note_array)
    export_path_csv(path, output_paths["path_csv"])
    export_tempomap_json(tempomap_raw, output_paths["tempomap_raw_json"])
    export_tempomap_json(tempomap_smooth, output_paths["tempomap_smooth_json"])

    plot_chroma(
        score_chroma,
        score_times,
        title="Score chroma",
        x_label="Score time [beats]",
        save_path=output_paths["score_chroma_plot"],
        show=SHOW_PLOTS,
    )
    plot_chroma(
        audio_chroma,
        audio_times,
        title="Audio chroma",
        x_label="Audio time [s]",
        save_path=output_paths["audio_chroma_plot"],
        show=SHOW_PLOTS,
    )
    plot_cost_matrix_with_path(
        cost,
        path,
        save_path=output_paths["cost_plot"],
        show=SHOW_PLOTS,
        title="Cost matrix with DTW path",
    )
    plot_tempomap(
        tempomap_smooth,
        save_path=output_paths["tempomap_plot"],
        show=SHOW_PLOTS,
        title="Tempomap",
    )
    plot_aligned_chromas(
        warped_score_chroma,
        audio_chroma,
        audio_times,
        save_path=output_paths["aligned_plot"],
        show=SHOW_PLOTS,
        title_prefix="",
    )
    plot_local_cost_matrix_with_path(
        cost=cost,
        path=path,
        score_times=score_times,
        audio_times=audio_times,
        score_min=1.5,
        score_max=2.5,
        audio_min=2.0,
        audio_max=3.5,
        save_path=output_paths["local_cost_plot"],
        show=SHOW_PLOTS,
        title="Local cost matrix around score beat 2.0",
    )

    save_notebook_analysis_bundle(
        analysis_dir=output_paths["analysis_dir"],
        note_array=note_array,
        score_times=score_times,
        score_chroma=score_chroma,
        audio_times=audio_times,
        audio_chroma=audio_chroma,
        path=path,
        warped_score_chroma=warped_score_chroma,
        tempomap_raw=tempomap_raw,
        tempomap_smooth=tempomap_smooth,
    )

    print(f"piece: {piece_name}")
    print(f"output root: {output_paths['piece_dir']}")
    print(f"exports dir: {output_paths['exports_dir']}")
    print(f"plots dir: {output_paths['plots_dir']}")
    print(f"analysis dir: {output_paths['analysis_dir']}")
    print(f"trim start: {start_time:.6f} s")
    print(f"score_chroma shape: {score_chroma.shape}")
    print(f"audio_chroma shape: {audio_chroma.shape}")
    print(f"cost shape: {cost.shape}")
    print(f"path shape: {path.shape}")
    print(f"raw tempomap shape: {tempomap_raw.shape}")
    print(f"smooth tempomap shape: {tempomap_smooth.shape}")
    print(f"score beats json: {output_paths['score_beats_json']}")
    print(f"path csv: {output_paths['path_csv']}")
    print(f"raw tempomap json: {output_paths['tempomap_raw_json']}")
    print(f"smooth tempomap json: {output_paths['tempomap_smooth_json']}")
    print("analysis bundle files:")
    print(f"  - {output_paths['analysis_dir'] / 'audio_times.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'audio_chroma.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'score_times.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'score_chroma.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'score_chroma_on_audio_time.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'path_frames.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'tempomap_raw.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'tempomap_smooth.npy'}")
    print(f"  - {output_paths['analysis_dir'] / 'score_notes.csv'}")


if __name__ == "__main__":
    main()
