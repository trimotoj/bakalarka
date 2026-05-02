from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import DEFAULT_SONGS_CONFIG, load_song_config

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
    plot_aligned_chromas,
    plot_chroma,
    plot_cost_matrix_with_path,
    plot_tempomap,
    plot_tempomap_comparison,
    warp_score_chroma_to_audio_time,
)

DEFAULT_OUTPUT_DIR = Path("output")

AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512
SCORE_FPS = 40
SMOOTH_WINDOW = 9
SHOW_PLOTS = False


def get_piece_name(score_path: Path, audio_path: Path) -> str:
    return audio_path.stem or score_path.stem


def build_output_paths(piece_name: str, output_dir: Path) -> dict[str, Path]:
    piece_dir = output_dir / piece_name
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
        "tempomap_raw_plot": plots_dir / "tempomap_raw.png",
        "tempomap_comparison_plot": plots_dir / "tempomap_raw_vs_smooth.png",
        "aligned_plot": plots_dir / "aligned_chromas.png",
    }


def ensure_output_dirs(output_paths: dict[str, Path]) -> None:
    for key in ("piece_dir", "exports_dir", "plots_dir", "analysis_dir"):
        output_paths[key].mkdir(parents=True, exist_ok=True)


def build_tempomap(
    path: np.ndarray,
    score_times: np.ndarray,
    audio_times: np.ndarray,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    tempomap_raw = make_tempomap(path, score_times, audio_times)
    tempomap_raw = remove_duplicate_points(tempomap_raw)
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

    save_note_array_csv(analysis_dir / "score_notes.csv", note_array)


def resolve_inputs_from_args(args: argparse.Namespace) -> tuple[str, Path, Path]:
    if args.song is not None:
        songs = load_song_config(args.config)

        if args.song not in songs:
            available = ", ".join(sorted(songs.keys()))
            raise KeyError(
                f"Song '{args.song}' is not defined in {args.config}. "
                f"Available songs: {available}"
            )

        song = songs[args.song]
        return args.song, Path(song["score"]), Path(song["audio"])

    if args.score is None or args.audio is None:
        raise ValueError("Use either --song or both --score and --audio.")

    piece_name = args.piece_name or get_piece_name(args.score, args.audio)
    return piece_name, args.score, args.audio


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
        help="Optional name of the output folder. If omitted, audio file stem is used.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots during execution.",
    )

    return parser.parse_args()


def run_alignment(
    piece_name: str,
    score_path: Path,
    audio_path: Path,
    output_dir: Path,
    show_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = build_output_paths(piece_name, output_dir)
    ensure_output_dirs(output_paths)

    score = load_score(score_path)
    note_array = score_to_note_array(score)

    audio_full, sr = load_audio(audio_path, sr=AUDIO_SR)
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

    tempomap_raw, tempomap_smooth = build_tempomap(
        path,
        score_times,
        audio_times,
        smooth_window=SMOOTH_WINDOW,
    )

    warped_score_chroma = warp_score_chroma_to_audio_time(
        score_chroma,
        path,
        n_audio_frames=len(audio_chroma),
    )

    save_score_beats_json(output_paths["score_beats_json"], note_array)
    save_path_csv(output_paths["path_csv"], path)
    save_json(output_paths["tempomap_raw_json"], tempomap_to_json_data(tempomap_raw))
    save_json(
        output_paths["tempomap_smooth_json"], tempomap_to_json_data(tempomap_smooth)
    )

    plot_chroma(
        score_chroma,
        score_times,
        title="Score chroma",
        x_label="Score time [beats]",
        save_path=output_paths["score_chroma_plot"],
        show=show_plots,
    )
    plot_chroma(
        audio_chroma,
        audio_times,
        title="Audio chroma",
        x_label="Audio time [s]",
        save_path=output_paths["audio_chroma_plot"],
        show=show_plots,
    )
    plot_cost_matrix_with_path(
        cost,
        path,
        save_path=output_paths["cost_plot"],
        show=show_plots,
        title="Cost matrix with DTW path",
    )
    plot_tempomap(
        tempomap_raw,
        save_path=output_paths["tempomap_raw_plot"],
        show=show_plots,
        title="Raw tempomap",
    )
    plot_tempomap(
        tempomap_smooth,
        save_path=output_paths["tempomap_plot"],
        show=show_plots,
        title="Smoothed tempomap",
    )
    plot_tempomap_comparison(
        tempomap_raw,
        tempomap_smooth,
        save_path=output_paths["tempomap_comparison_plot"],
        show=show_plots,
        title="Raw vs. smoothed tempomap",
    )
    plot_aligned_chromas(
        warped_score_chroma,
        audio_chroma,
        audio_times,
        save_path=output_paths["aligned_plot"],
        show=show_plots,
        title_prefix="",
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


def main() -> None:
    args = parse_args()

    output_dir = args.output
    show_plots = args.show_plots or SHOW_PLOTS

    if args.all:
        songs = load_song_config(args.config)

        for piece_name, song in songs.items():
            print()
            print(f"=== Processing {piece_name} ===")

            run_alignment(
                piece_name=piece_name,
                score_path=Path(song["score"]),
                audio_path=Path(song["audio"]),
                output_dir=output_dir,
                show_plots=show_plots,
            )

        return

    piece_name, score_path, audio_path = resolve_inputs_from_args(args)

    run_alignment(
        piece_name=piece_name,
        score_path=score_path,
        audio_path=audio_path,
        output_dir=output_dir,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
