from pathlib import Path

from src.audio_features import (
    audio_to_chroma,
    get_audio_frame_times,
    trim_leading_silence,
)
from src.cost_matrix import cosine_cost_matrix
from src.dtw_alignment import dtw, dtw_reverse
from src.io_utils import (
    load_audio,
    load_score,
    save_path_csv,
    save_score_beats_json,
    save_tempomap_csv,
)
from src.score_features import (
    build_score_time_grid,
    note_array_to_chroma,
    score_to_note_array,
)
from src.tempomap import (
    export_tempomap_json,
    make_tempomap,
    remove_duplicate_points,
    smooth_tempomap,
)
from src.visualization import (
    plot_aligned_chromas,
    plot_chroma,
    plot_cost_matrix_with_path,
    plot_paths_side_by_side,
    plot_tempomap,
    warp_score_chroma_to_audio_time,
)


SCORE_PATH = Path("data/score/chopin.musicxml")
AUDIO_PATH = Path("data/audio/chopin.wav")
OUTPUT_DIR = Path("data/output")
PLOTS_DIR = OUTPUT_DIR / "plots"

AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512
SCORE_FPS = 40
SMOOTH_WINDOW = 9
SHOW_PLOTS = True

DTW_MODE = "all"  # "forward" | "reverse" | "all"


def get_piece_name(score_path: Path, audio_path: Path) -> str:
    if score_path.stem == audio_path.stem:
        return score_path.stem
    return score_path.stem or audio_path.stem


def build_output_paths(piece_name: str) -> dict[str, Path]:
    return {
        "tempomap_csv": OUTPUT_DIR / f"{piece_name}_tempomap.csv",
        "path_csv": OUTPUT_DIR / f"{piece_name}_path.csv",
        "tempomap_json": OUTPUT_DIR / f"{piece_name}_tempomap.json",
        "cost_plot": PLOTS_DIR / f"{piece_name}_cost_matrix_with_path.png",
        "tempomap_plot": PLOTS_DIR / f"{piece_name}_tempomap.png",
        "aligned_chromas_plot": PLOTS_DIR / f"{piece_name}_aligned_chromas.png",
        "reverse_tempomap_csv": OUTPUT_DIR / f"{piece_name}_tempomap_reverse.csv",
        "reverse_path_csv": OUTPUT_DIR / f"{piece_name}_path_reverse.csv",
        "reverse_tempomap_json": OUTPUT_DIR / f"{piece_name}_tempomap_reverse.json",
        "reverse_cost_plot": PLOTS_DIR
        / f"{piece_name}_cost_matrix_with_path_reverse.png",
        "reverse_tempomap_plot": PLOTS_DIR / f"{piece_name}_tempomap_reverse.png",
        "reverse_aligned_chromas_plot": PLOTS_DIR
        / f"{piece_name}_aligned_chromas_reverse.png",
        "score_beats_json": OUTPUT_DIR / f"{piece_name}_score_beats.json",
        "score_chroma_plot": PLOTS_DIR / f"{piece_name}_score_chroma.png",
        "audio_chroma_plot": PLOTS_DIR / f"{piece_name}_audio_chroma.png",
        "paths_side_by_side_plot": PLOTS_DIR / f"{piece_name}_paths_side_by_side.png",
    }


def save_common_plots(
    score_chroma,
    audio_chroma,
    score_times,
    audio_times,
    output_paths: dict[str, Path],
) -> None:
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


def get_variant_keys(variant: str) -> dict[str, str]:
    if variant == "forward":
        return {
            "tempomap_csv": "tempomap_csv",
            "path_csv": "path_csv",
            "tempomap_json": "tempomap_json",
            "cost_plot": "cost_plot",
            "tempomap_plot": "tempomap_plot",
            "aligned_plot": "aligned_chromas_plot",
        }
    if variant == "reverse":
        return {
            "tempomap_csv": "reverse_tempomap_csv",
            "path_csv": "reverse_path_csv",
            "tempomap_json": "reverse_tempomap_json",
            "cost_plot": "reverse_cost_plot",
            "tempomap_plot": "reverse_tempomap_plot",
            "aligned_plot": "reverse_aligned_chromas_plot",
        }
    raise ValueError(f"Unknown variant: {variant}")


def save_variant_results(
    variant: str,
    path,
    tempomap,
    cost,
    score_chroma,
    audio_chroma,
    audio_times,
    output_paths: dict[str, Path],
) -> None:
    keys = get_variant_keys(variant)

    save_tempomap_csv(output_paths[keys["tempomap_csv"]], tempomap)
    save_path_csv(output_paths[keys["path_csv"]], path)
    export_tempomap_json(tempomap, output_paths[keys["tempomap_json"]])

    plot_cost_matrix_with_path(
        cost,
        path,
        save_path=output_paths[keys["cost_plot"]],
        show=SHOW_PLOTS,
        title=f"Cost matrix with {variant} DTW path",
    )
    plot_tempomap(
        tempomap,
        save_path=output_paths[keys["tempomap_plot"]],
        show=SHOW_PLOTS,
        title=f"{variant.capitalize()} tempomap",
    )

    score_chroma_on_audio_time = warp_score_chroma_to_audio_time(
        score_chroma,
        path,
        n_audio_frames=len(audio_chroma),
    )
    plot_aligned_chromas(
        score_chroma_on_audio_time,
        audio_chroma,
        audio_times,
        save_path=output_paths[keys["aligned_plot"]],
        show=SHOW_PLOTS,
        title_prefix=f"{variant.capitalize()} – ",
    )

    print(f"{variant} path: {path.shape}")
    print(f"{variant} tempomap: {tempomap.shape}")
    print(f"{variant} tempomap json: {output_paths[keys['tempomap_json']]}")


def build_tempomap_from_path(path, score_times, audio_times):
    tempomap = make_tempomap(path, score_times, audio_times)
    tempomap = remove_duplicate_points(tempomap)
    tempomap = smooth_tempomap(tempomap, window=SMOOTH_WINDOW)
    return tempomap


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    piece_name = get_piece_name(SCORE_PATH, AUDIO_PATH)
    output_paths = build_output_paths(piece_name)

    score = load_score(SCORE_PATH)
    note_array = score_to_note_array(score)

    audio, sr = load_audio(AUDIO_PATH, sr=AUDIO_SR)
    audio, start_sample = trim_leading_silence(audio, top_db=30)
    start_time = start_sample / sr

    score_times = build_score_time_grid(note_array, fps=SCORE_FPS)
    score_chroma = note_array_to_chroma(note_array, score_times)

    audio_chroma = audio_to_chroma(audio, sr, hop_length=AUDIO_HOP_LENGTH)
    audio_times = (
        get_audio_frame_times(len(audio_chroma), sr, AUDIO_HOP_LENGTH) + start_time
    )

    cost = cosine_cost_matrix(score_chroma, audio_chroma)

    save_score_beats_json(output_paths["score_beats_json"], note_array)
    save_common_plots(
        score_chroma,
        audio_chroma,
        score_times,
        audio_times,
        output_paths,
    )

    print(f"piece: {piece_name}")
    print(f"trim start: {start_time:.6f} s")
    print(f"score_chroma: {score_chroma.shape}")
    print(f"audio_chroma: {audio_chroma.shape}")
    print(f"cost: {cost.shape}")
    print(f"dtw mode: {DTW_MODE}")
    print(f"score beats json: {output_paths['score_beats_json']}")

    if DTW_MODE == "forward":
        path = dtw(cost)
        tempomap = build_tempomap_from_path(path, score_times, audio_times)
        save_variant_results(
            "forward",
            path,
            tempomap,
            cost,
            score_chroma,
            audio_chroma,
            audio_times,
            output_paths,
        )

    elif DTW_MODE == "reverse":
        path = dtw_reverse(cost)
        tempomap = build_tempomap_from_path(path, score_times, audio_times)
        save_variant_results(
            "reverse",
            path,
            tempomap,
            cost,
            score_chroma,
            audio_chroma,
            audio_times,
            output_paths,
        )

    elif DTW_MODE == "all":
        forward_path = dtw(cost)
        reverse_path = dtw_reverse(cost)

        forward_tempomap = build_tempomap_from_path(
            forward_path, score_times, audio_times
        )
        reverse_tempomap = build_tempomap_from_path(
            reverse_path, score_times, audio_times
        )

        save_variant_results(
            "forward",
            forward_path,
            forward_tempomap,
            cost,
            score_chroma,
            audio_chroma,
            audio_times,
            output_paths,
        )
        save_variant_results(
            "reverse",
            reverse_path,
            reverse_tempomap,
            cost,
            score_chroma,
            audio_chroma,
            audio_times,
            output_paths,
        )

        plot_paths_side_by_side(
            cost,
            forward_path,
            reverse_path,
            save_path=output_paths["paths_side_by_side_plot"],
            show=SHOW_PLOTS,
        )

    else:
        raise ValueError('DTW_MODE must be one of: "forward", "reverse", "all"')


if __name__ == "__main__":
    main()
