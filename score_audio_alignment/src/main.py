import json
from pathlib import Path

import numpy as np

from src.audio_features import (
    audio_to_chroma,
    get_audio_frame_times,
    trim_leading_silence,
)
from src.cost_matrix import cosine_cost_matrix
from src.dtw_alignment import dtw
from src.io_utils import load_audio, load_score, save_path_csv, save_tempomap_csv
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
    plot_both_chromas,
    plot_chroma,
    plot_cost_matrix_with_path,
    plot_tempomap,
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


def export_score_beats_json(score_path: str | Path, out_path: str | Path) -> None:
    score = load_score(score_path)
    note_array = score_to_note_array(score)
    unique_onsets = np.unique(note_array["onset_beat"].astype(float))

    data = []
    for onset in unique_onsets:
        data.append({"score_time": float(onset)})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    acc, path = dtw(cost)

    tempomap = make_tempomap(path, score_times, audio_times)
    tempomap = remove_duplicate_points(tempomap)
    tempomap = smooth_tempomap(tempomap, window=SMOOTH_WINDOW)

    save_tempomap_csv(OUTPUT_DIR / "tempomap.csv", tempomap)
    save_path_csv(OUTPUT_DIR / "path.csv", path)
    export_tempomap_json(tempomap, OUTPUT_DIR / "tempomap.json")
    export_score_beats_json(SCORE_PATH, OUTPUT_DIR / "score_beats.json")

    print(f"trim start: {start_time:.6f} s")
    print("score_chroma:", score_chroma.shape)
    print("audio_chroma:", audio_chroma.shape)
    print("cost:", cost.shape)
    print("acc:", acc.shape)
    print("path:", path.shape)
    print("tempomap:", tempomap.shape)

    plot_chroma(
        score_chroma,
        score_times,
        title="Score chroma",
        x_label="Score time [beats]",
        save_path=PLOTS_DIR / "score_chroma.png",
        show=SHOW_PLOTS,
    )
    plot_chroma(
        audio_chroma,
        audio_times,
        title="Audio chroma",
        x_label="Audio time [s]",
        save_path=PLOTS_DIR / "audio_chroma.png",
        show=SHOW_PLOTS,
    )
    plot_both_chromas(
        score_chroma,
        audio_chroma,
        score_times,
        audio_times,
        save_path=PLOTS_DIR / "both_chromas.png",
        show=SHOW_PLOTS,
    )
    plot_cost_matrix_with_path(
        cost,
        path,
        save_path=PLOTS_DIR / "cost_matrix_with_path.png",
        show=SHOW_PLOTS,
    )
    plot_tempomap(
        tempomap,
        save_path=PLOTS_DIR / "tempomap.png",
        show=SHOW_PLOTS,
    )


if __name__ == "__main__":
    main()
