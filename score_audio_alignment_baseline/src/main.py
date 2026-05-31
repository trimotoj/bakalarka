from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.audio_features import (
    DEFAULT_TRIM_DB,
    audio_to_chroma,
    get_audio_frame_times,
    trim_leading_silence,
)
from src.config import DEFAULT_SONGS_CONFIG, get_song_from_config, load_song_config
from src.cost_matrix import cosine_cost_matrix
from src.dtw_alignment import dtw
from src.io_utils import (
    load_audio,
    load_score,
    save_json,
    save_note_array_csv,
    save_path_csv,
    save_score_beats_json,
)
from src.score_features import (
    build_score_time_grid,
    note_array_to_chroma,
    score_to_note_array,
)
from src.tempomap import (
    make_tempomap,
    remove_duplicate_points,
    smooth_tempomap,
    tempomap_to_json_data,
)
from src.visualization import (
    AX_AUDIO_TIME,
    AX_SCORE_TIME,
    TITLE_AUDIO_CHROMA,
    TITLE_COST_MATRIX,
    TITLE_SCORE_CHROMA,
    TITLE_TEMPOMAP_COMPARISON,
    TITLE_TEMPOMAP_RAW,
    TITLE_TEMPOMAP_SMOOTH,
    plot_aligned_chromas,
    plot_chroma,
    plot_cost_matrix_with_path,
    plot_tempomap,
    plot_tempomap_comparison,
    warp_score_chroma_to_audio_time,
)

DEFAULT_OUTPUT_DIR = Path("output")

AUDIO_SAMPLE_RATE = 22050
AUDIO_HOP_LENGTH = 512
SCORE_FPS = 40
SMOOTH_WINDOW = 9
SHOW_PLOTS = False


@dataclass(frozen=True)
class AlignmentInputs:
    piece_name: str
    score_path: Path
    audio_path: Path


@dataclass(frozen=True)
class OutputPaths:
    piece_dir: Path
    exports_dir: Path
    plots_dir: Path
    analysis_dir: Path
    score_beats_json: Path
    path_csv: Path
    tempomap_raw_json: Path
    tempomap_smooth_json: Path
    score_chroma_plot: Path
    audio_chroma_plot: Path
    cost_plot: Path
    tempomap_plot: Path
    tempomap_raw_plot: Path
    tempomap_comparison_plot: Path
    aligned_plot: Path

    @classmethod
    def from_piece(cls, piece_name: str, output_dir: Path) -> "OutputPaths":
        piece_dir = output_dir / piece_name
        exports_dir = piece_dir / "exports"
        plots_dir = piece_dir / "plots"
        analysis_dir = piece_dir / "analysis"

        return cls(
            piece_dir=piece_dir,
            exports_dir=exports_dir,
            plots_dir=plots_dir,
            analysis_dir=analysis_dir,
            score_beats_json=exports_dir / "score_beats.json",
            path_csv=exports_dir / "path.csv",
            tempomap_raw_json=exports_dir / "tempomap_raw.json",
            tempomap_smooth_json=exports_dir / "tempomap.json",
            score_chroma_plot=plots_dir / "score_chroma.png",
            audio_chroma_plot=plots_dir / "audio_chroma.png",
            cost_plot=plots_dir / "cost_matrix_with_path.png",
            tempomap_plot=plots_dir / "tempomap.png",
            tempomap_raw_plot=plots_dir / "tempomap_raw.png",
            tempomap_comparison_plot=plots_dir / "tempomap_raw_vs_smooth.png",
            aligned_plot=plots_dir / "aligned_chromas.png",
        )

    def ensure_dirs(self) -> None:
        for directory in (
            self.piece_dir,
            self.exports_dir,
            self.plots_dir,
            self.analysis_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_piece_name(score_path: Path, audio_path: Path) -> str:
    """Choose a stable output folder name from input paths."""
    return audio_path.stem or score_path.stem


def build_tempomap(
    path: np.ndarray,
    score_times: np.ndarray,
    audio_times: np.ndarray,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw and smoothed tempomaps from a DTW path."""
    tempomap_raw = remove_duplicate_points(
        make_tempomap(path, score_times, audio_times)
    )
    tempomap_smooth = smooth_tempomap(tempomap_raw, window=smooth_window)
    return tempomap_raw, tempomap_smooth


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
    """Save compact intermediate data used by the analysis notebook."""
    analysis_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "audio_times.npy": audio_times,
        "audio_chroma.npy": audio_chroma,
        "score_times.npy": score_times,
        "score_chroma.npy": score_chroma,
        "score_chroma_on_audio_time.npy": warped_score_chroma,
        "path_frames.npy": np.asarray(path, dtype=int),
        "tempomap_raw.npy": tempomap_raw,
        "tempomap_smooth.npy": tempomap_smooth,
    }

    for filename, array in arrays.items():
        np.save(analysis_dir / filename, np.asarray(array))

    save_note_array_csv(analysis_dir / "score_notes.csv", note_array)


def resolve_inputs_from_args(args: argparse.Namespace) -> AlignmentInputs:
    """Resolve CLI arguments into one score/audio pair."""
    if args.song is not None:
        song = get_song_from_config(args.song, args.config)
        return AlignmentInputs(
            piece_name=args.song,
            score_path=Path(song["score"]),
            audio_path=Path(song["audio"]),
        )

    if args.score is None or args.audio is None:
        raise ValueError("Use either --song or both --score and --audio.")

    piece_name = args.piece_name or get_piece_name(args.score, args.audio)
    return AlignmentInputs(
        piece_name=piece_name,
        score_path=args.score,
        audio_path=args.audio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute score-audio alignment and export tempomap."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all songs from the songs config.",
    )
    parser.add_argument(
        "--song",
        type=str,
        default=None,
        help="Song ID from config/songs.json.",
    )
    parser.add_argument(
        "--score",
        type=Path,
        default=None,
        help="Path to MusicXML score file.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to audio file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SONGS_CONFIG,
        help="Path to songs config JSON file.",
    )
    parser.add_argument(
        "--piece-name",
        type=str,
        default=None,
        help="Optional output folder name. If omitted, the audio file stem is used.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots during execution.",
    )

    return parser.parse_args()


def compute_alignment(score_path: Path, audio_path: Path) -> dict[str, Any]:
    """Run the complete feature extraction and DTW alignment pipeline."""
    score = load_score(score_path)
    note_array = score_to_note_array(score)

    audio_full, sample_rate = load_audio(audio_path, sample_rate=AUDIO_SAMPLE_RATE)
    audio_trimmed, start_sample = trim_leading_silence(
        audio_full, top_db=DEFAULT_TRIM_DB
    )
    start_time = start_sample / sample_rate

    score_times = build_score_time_grid(note_array, fps=SCORE_FPS)
    score_chroma = note_array_to_chroma(note_array, score_times)

    audio_chroma = audio_to_chroma(
        audio_trimmed,
        sample_rate=sample_rate,
        hop_length=AUDIO_HOP_LENGTH,
    )
    audio_times = (
        get_audio_frame_times(
            n_frames=len(audio_chroma),
            sample_rate=sample_rate,
            hop_length=AUDIO_HOP_LENGTH,
        )
        + start_time
    )

    cost = cosine_cost_matrix(score_chroma, audio_chroma)
    path = dtw(cost)

    tempomap_raw, tempomap_smooth = build_tempomap(
        path=path,
        score_times=score_times,
        audio_times=audio_times,
        smooth_window=SMOOTH_WINDOW,
    )
    warped_score_chroma = warp_score_chroma_to_audio_time(
        score_chroma,
        path,
        n_audio_frames=len(audio_chroma),
    )

    return {
        "note_array": note_array,
        "score_times": score_times,
        "score_chroma": score_chroma,
        "audio_times": audio_times,
        "audio_chroma": audio_chroma,
        "cost": cost,
        "path": path,
        "tempomap_raw": tempomap_raw,
        "tempomap_smooth": tempomap_smooth,
        "warped_score_chroma": warped_score_chroma,
        "trim_start_time": start_time,
    }


def save_alignment_outputs(
    paths: OutputPaths,
    data: dict[str, Any],
    show_plots: bool,
) -> None:
    """Save exports, plots, and notebook analysis data."""
    save_score_beats_json(paths.score_beats_json, data["note_array"])
    save_path_csv(paths.path_csv, data["path"])
    save_json(paths.tempomap_raw_json, tempomap_to_json_data(data["tempomap_raw"]))
    save_json(
        paths.tempomap_smooth_json, tempomap_to_json_data(data["tempomap_smooth"])
    )

    plot_chroma(
        data["score_chroma"],
        data["score_times"],
        title=TITLE_SCORE_CHROMA,
        x_label=AX_SCORE_TIME,
        save_path=paths.score_chroma_plot,
        show=show_plots,
    )
    plot_chroma(
        data["audio_chroma"],
        data["audio_times"],
        title=TITLE_AUDIO_CHROMA,
        x_label=AX_AUDIO_TIME,
        save_path=paths.audio_chroma_plot,
        show=show_plots,
    )
    plot_cost_matrix_with_path(
        data["cost"],
        data["path"],
        save_path=paths.cost_plot,
        show=show_plots,
        title=TITLE_COST_MATRIX,
    )
    plot_tempomap(
        data["tempomap_raw"],
        save_path=paths.tempomap_raw_plot,
        show=show_plots,
        title=TITLE_TEMPOMAP_RAW,
    )
    plot_tempomap(
        data["tempomap_smooth"],
        save_path=paths.tempomap_plot,
        show=show_plots,
        title=TITLE_TEMPOMAP_SMOOTH,
    )
    plot_tempomap_comparison(
        data["tempomap_raw"],
        data["tempomap_smooth"],
        save_path=paths.tempomap_comparison_plot,
        show=show_plots,
        title=TITLE_TEMPOMAP_COMPARISON,
    )
    plot_aligned_chromas(
        data["warped_score_chroma"],
        data["audio_chroma"],
        data["audio_times"],
        save_path=paths.aligned_plot,
        show=show_plots,
    )

    save_notebook_analysis_bundle(
        analysis_dir=paths.analysis_dir,
        note_array=data["note_array"],
        score_times=data["score_times"],
        score_chroma=data["score_chroma"],
        audio_times=data["audio_times"],
        audio_chroma=data["audio_chroma"],
        path=data["path"],
        warped_score_chroma=data["warped_score_chroma"],
        tempomap_raw=data["tempomap_raw"],
        tempomap_smooth=data["tempomap_smooth"],
    )


def print_run_summary(
    piece_name: str, paths: OutputPaths, data: dict[str, Any]
) -> None:
    """Print a compact terminal summary after one alignment run."""
    print(f"piece: {piece_name}")
    print(f"output root: {paths.piece_dir}")
    print(f"exports dir: {paths.exports_dir}")
    print(f"plots dir: {paths.plots_dir}")
    print(f"analysis dir: {paths.analysis_dir}")
    print(f"trim start: {data['trim_start_time']:.6f} s")
    print(f"score_chroma shape: {data['score_chroma'].shape}")
    print(f"audio_chroma shape: {data['audio_chroma'].shape}")
    print(f"cost shape: {data['cost'].shape}")
    print(f"path shape: {data['path'].shape}")
    print(f"raw tempomap shape: {data['tempomap_raw'].shape}")
    print(f"smooth tempomap shape: {data['tempomap_smooth'].shape}")


def run_alignment(
    piece_name: str,
    score_path: Path,
    audio_path: Path,
    output_dir: Path,
    show_plots: bool,
) -> None:
    """Run alignment for one piece and write all outputs."""
    paths = OutputPaths.from_piece(piece_name, output_dir)
    paths.ensure_dirs()

    data = compute_alignment(score_path=score_path, audio_path=audio_path)
    save_alignment_outputs(paths, data, show_plots=show_plots)
    print_run_summary(piece_name, paths, data)


def main() -> None:
    args = parse_args()
    show_plots = args.show_plots or SHOW_PLOTS

    if args.all:
        songs = load_song_config(args.config)
        for piece_name, song in songs.items():
            print(f"\n=== Processing {piece_name} ===")
            run_alignment(
                piece_name=piece_name,
                score_path=Path(song["score"]),
                audio_path=Path(song["audio"]),
                output_dir=args.output,
                show_plots=show_plots,
            )
        return

    inputs = resolve_inputs_from_args(args)
    run_alignment(
        piece_name=inputs.piece_name,
        score_path=inputs.score_path,
        audio_path=inputs.audio_path,
        output_dir=args.output,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
