from __future__ import annotations

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


SCORE_PATH = Path("data/score/aka-si-mi-krasna.musicxml")
AUDIO_PATH = Path("data/audio/aka-si-mi-krasna.wav")
OUTPUT_DIR = Path("data/output")
PLOTS_DIR = OUTPUT_DIR / "plots"

AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512
SCORE_FPS = 40
SMOOTH_WINDOW = 9
SHOW_PLOTS = True

DTW_MODE = "all"  # "forward" | "reverse" | "all"


def get_piece_name(score_path: Path, audio_path: Path) -> str:
    return (
        score_path.stem
        if score_path.stem == audio_path.stem
        else (score_path.stem or audio_path.stem)
    )


def build_output_paths(piece_name: str) -> dict[str, object]:
    common = {
        "score_beats_json": OUTPUT_DIR / f"{piece_name}_score_beats.json",
        "score_chroma_plot": PLOTS_DIR / f"{piece_name}_score_chroma.png",
        "audio_chroma_plot": PLOTS_DIR / f"{piece_name}_audio_chroma.png",
        "paths_side_by_side_plot": PLOTS_DIR / f"{piece_name}_paths_side_by_side.png",
    }

    variants = {}
    for name, suffix in {"forward": "", "reverse": "_reverse"}.items():
        variants[name] = {
            "tempomap_csv": OUTPUT_DIR / f"{piece_name}_tempomap{suffix}.csv",
            "path_csv": OUTPUT_DIR / f"{piece_name}_path{suffix}.csv",
            "tempomap_json": OUTPUT_DIR / f"{piece_name}_tempomap{suffix}.json",
            "cost_plot": PLOTS_DIR / f"{piece_name}_cost_matrix_with_path{suffix}.png",
            "tempomap_plot": PLOTS_DIR / f"{piece_name}_tempomap{suffix}.png",
            "aligned_plot": PLOTS_DIR / f"{piece_name}_aligned_chromas{suffix}.png",
        }

    return {"common": common, "variants": variants}


def build_tempomap_from_path(path, score_times, audio_times):
    tempomap = make_tempomap(path, score_times, audio_times)
    tempomap = remove_duplicate_points(tempomap)
    return smooth_tempomap(tempomap, window=SMOOTH_WINDOW)


def save_common_outputs(
    note_array,
    score_chroma,
    audio_chroma,
    score_times,
    audio_times,
    output_paths: dict[str, object],
) -> None:
    common = output_paths["common"]
    save_score_beats_json(common["score_beats_json"], note_array)

    plot_chroma(
        score_chroma,
        score_times,
        title="Score chroma",
        x_label="Score time [beats]",
        save_path=common["score_chroma_plot"],
        show=SHOW_PLOTS,
    )
    plot_chroma(
        audio_chroma,
        audio_times,
        title="Audio chroma",
        x_label="Audio time [s]",
        save_path=common["audio_chroma_plot"],
        show=SHOW_PLOTS,
    )


def save_variant_outputs(
    variant: str,
    path,
    tempomap,
    cost,
    score_chroma,
    audio_chroma,
    audio_times,
    output_paths: dict[str, object],
) -> None:
    paths = output_paths["variants"][variant]

    save_tempomap_csv(paths["tempomap_csv"], tempomap)
    save_path_csv(paths["path_csv"], path)
    export_tempomap_json(tempomap, paths["tempomap_json"])

    plot_cost_matrix_with_path(
        cost,
        path,
        save_path=paths["cost_plot"],
        show=SHOW_PLOTS,
        title=f"Cost matrix with {variant} DTW path",
    )
    plot_tempomap(
        tempomap,
        save_path=paths["tempomap_plot"],
        show=SHOW_PLOTS,
        title=f"{variant.capitalize()} tempomap",
    )

    warped_score_chroma = warp_score_chroma_to_audio_time(
        score_chroma,
        path,
        n_audio_frames=len(audio_chroma),
    )
    plot_aligned_chromas(
        warped_score_chroma,
        audio_chroma,
        audio_times,
        save_path=paths["aligned_plot"],
        show=SHOW_PLOTS,
        title_prefix=f"{variant.capitalize()} – ",
    )

    print(f"{variant} path shape: {path.shape}")
    print(f"{variant} tempomap shape: {tempomap.shape}")
    print(f"{variant} tempomap json: {paths['tempomap_json']}")


def compute_variant_paths(cost):
    if DTW_MODE == "forward":
        return {"forward": dtw(cost)}
    if DTW_MODE == "reverse":
        return {"reverse": dtw_reverse(cost)}
    if DTW_MODE == "all":
        return {"forward": dtw(cost), "reverse": dtw_reverse(cost)}
    raise ValueError('DTW_MODE must be one of: "forward", "reverse", "all"')


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

    save_common_outputs(
        note_array,
        score_chroma,
        audio_chroma,
        score_times,
        audio_times,
        output_paths,
    )

    print(f"piece: {piece_name}")
    print(f"trim start: {start_time:.6f} s")
    print(f"score_chroma shape: {score_chroma.shape}")
    print(f"audio_chroma shape: {audio_chroma.shape}")
    print(f"cost shape: {cost.shape}")
    print(f"dtw mode: {DTW_MODE}")
    print(f"score beats json: {output_paths['common']['score_beats_json']}")

    variant_paths = compute_variant_paths(cost)
    for variant, path in variant_paths.items():
        tempomap = build_tempomap_from_path(path, score_times, audio_times)
        save_variant_outputs(
            variant,
            path,
            tempomap,
            cost,
            score_chroma,
            audio_chroma,
            audio_times,
            output_paths,
        )

    if {"forward", "reverse"}.issubset(variant_paths):
        plot_paths_side_by_side(
            cost,
            variant_paths["forward"],
            variant_paths["reverse"],
            save_path=output_paths["common"]["paths_side_by_side_plot"],
            show=SHOW_PLOTS,
        )


if __name__ == "__main__":
    main()
