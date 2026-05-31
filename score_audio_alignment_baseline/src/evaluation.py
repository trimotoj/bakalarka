from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_SONGS_CONFIG, load_song_config

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_REFERENCES_PATH = Path("evaluation/reference_points.csv")
DEFAULT_TOLERANCES = [0.25, 0.5, 1.0, 2.0]
REFERENCE_COLUMNS = {"song", "label", "score_time", "audio_time_ref"}
ReferenceRow = dict[str, Any]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_tempomap_from_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing tempomap file: {path}")

    tempomap = np.load(path)

    if tempomap.ndim != 2 or tempomap.shape[1] != 2:
        raise ValueError(f"Invalid tempomap shape in {path}: {tempomap.shape}")

    return tempomap.astype(float)


def load_tempomap_from_json(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing tempomap file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tempomap = np.asarray(
        [[item["score_time"], item["audio_time"]] for item in data],
        dtype=float,
    )

    if tempomap.ndim != 2 or tempomap.shape[1] != 2:
        raise ValueError(f"Invalid tempomap shape in {path}: {tempomap.shape}")

    return tempomap


def load_tempomap(piece_dir: Path, kind: str) -> np.ndarray:
    analysis_path = piece_dir / "analysis" / f"tempomap_{kind}.npy"

    if analysis_path.exists():
        return load_tempomap_from_npy(analysis_path)

    if kind == "smooth":
        json_path = piece_dir / "exports" / "tempomap.json"
    else:
        json_path = piece_dir / "exports" / "tempomap_raw.json"

    return load_tempomap_from_json(json_path)


def group_tempomap_by_score_time(tempomap: np.ndarray) -> np.ndarray:
    """
    DTW path can contain multiple audio times for the same score time.
    For interpolation score_time -> audio_time, we collapse duplicate
    score positions by taking the median audio time.
    """
    order = np.argsort(tempomap[:, 0], kind="stable")
    sorted_tempomap = tempomap[order]

    score_times = sorted_tempomap[:, 0]
    audio_times = sorted_tempomap[:, 1]

    unique_score_times = np.unique(score_times)
    grouped_audio_times = np.zeros_like(unique_score_times, dtype=float)

    for idx, score_time in enumerate(unique_score_times):
        grouped_audio_times[idx] = np.median(audio_times[score_times == score_time])

    return np.column_stack([unique_score_times, grouped_audio_times])


def predict_audio_time(tempomap: np.ndarray, score_time: float) -> float:
    grouped = group_tempomap_by_score_time(tempomap)

    score_times = grouped[:, 0]
    audio_times = grouped[:, 1]

    if score_time < score_times[0] or score_time > score_times[-1]:
        return float("nan")

    return float(np.interp(score_time, score_times, audio_times))


def load_reference_points(path: Path, song_id: str) -> list[ReferenceRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference points file: {path}")

    points: list[ReferenceRow] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Empty reference file: {path}")

        missing = REFERENCE_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")

        for row in reader:
            if row["song"] != song_id:
                continue

            points.append(
                {
                    "song": row["song"],
                    "label": row["label"],
                    "score_time": float(row["score_time"]),
                    "audio_time_ref": float(row["audio_time_ref"]),
                    "note": row.get("note", ""),
                }
            )

    if not points:
        raise ValueError(f"No reference points found for song: {song_id}")

    return points


def evaluate_reference_points(
    tempomap: np.ndarray,
    reference_points: list[ReferenceRow],
) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []

    for point in reference_points:
        predicted = predict_audio_time(tempomap, point["score_time"])

        if np.isnan(predicted):
            error = float("nan")
            abs_error = float("nan")
            valid = False
        else:
            error = predicted - point["audio_time_ref"]
            abs_error = abs(error)
            valid = True

        rows.append(
            {
                "song": point["song"],
                "label": point["label"],
                "score_time": point["score_time"],
                "audio_time_ref": point["audio_time_ref"],
                "audio_time_pred": predicted,
                "error_seconds": error,
                "absolute_error_seconds": abs_error,
                "valid": valid,
                "note": point["note"],
            }
        )

    return rows


def summarize_errors(
    rows: list[ReferenceRow], tolerances: list[float]
) -> dict[str, Any]:
    valid_errors = np.asarray(
        [
            row["absolute_error_seconds"]
            for row in rows
            if row["valid"] and not np.isnan(row["absolute_error_seconds"])
        ],
        dtype=float,
    )

    signed_errors = np.asarray(
        [
            row["error_seconds"]
            for row in rows
            if row["valid"] and not np.isnan(row["error_seconds"])
        ],
        dtype=float,
    )

    if len(valid_errors) == 0:
        raise ValueError("No valid reference points to evaluate.")

    summary = {
        "num_points_total": len(rows),
        "num_points_valid": int(len(valid_errors)),
        "mae_seconds": float(np.mean(valid_errors)),
        "median_absolute_error_seconds": float(np.median(valid_errors)),
        "max_absolute_error_seconds": float(np.max(valid_errors)),
        "mean_signed_error_seconds": float(np.mean(signed_errors)),
        "std_signed_error_seconds": float(np.std(signed_errors)),
    }

    for tolerance in tolerances:
        key = f"alignment_rate_at_{str(tolerance).replace('.', '_')}s"
        summary[key] = float(np.mean(valid_errors <= tolerance))

    return summary


def save_reference_errors_csv(path: Path, rows: list[ReferenceRow]) -> None:
    path = ensure_parent(path)

    fields = [
        "song",
        "label",
        "score_time",
        "audio_time_ref",
        "audio_time_pred",
        "error_seconds",
        "absolute_error_seconds",
        "valid",
        "note",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def max_boolean_run(mask: np.ndarray) -> int:
    max_run = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return int(max_run)


def diagnose_path(piece_dir: Path) -> dict[str, Any]:
    path_file = piece_dir / "analysis" / "path_frames.npy"

    if not path_file.exists():
        return {
            "path_available": False,
        }

    path = np.load(path_file)

    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return {
            "path_available": False,
        }

    diff = np.diff(path, axis=0)

    score_steps = diff[:, 0]
    audio_steps = diff[:, 1]

    horizontal_steps = (score_steps > 0) & (audio_steps == 0)
    vertical_steps = (score_steps == 0) & (audio_steps > 0)
    diagonal_steps = (score_steps > 0) & (audio_steps > 0)

    total_steps = len(diff)

    return {
        "path_available": True,
        "path_length": int(len(path)),
        "horizontal_steps": int(np.sum(horizontal_steps)),
        "vertical_steps": int(np.sum(vertical_steps)),
        "diagonal_steps": int(np.sum(diagonal_steps)),
        "horizontal_step_ratio": float(np.sum(horizontal_steps) / total_steps),
        "vertical_step_ratio": float(np.sum(vertical_steps) / total_steps),
        "max_horizontal_run": max_boolean_run(horizontal_steps),
        "max_vertical_run": max_boolean_run(vertical_steps),
    }


def diagnose_tempomap(tempomap: np.ndarray) -> dict[str, Any]:
    grouped = group_tempomap_by_score_time(tempomap)

    score_times = grouped[:, 0]
    audio_times = grouped[:, 1]

    score_diffs = np.diff(score_times)
    audio_diffs = np.diff(audio_times)

    eps = 1e-9

    valid_intervals = score_diffs > eps
    seconds_per_beat = audio_diffs[valid_intervals] / score_diffs[valid_intervals]

    positive_seconds_per_beat = seconds_per_beat[seconds_per_beat > eps]

    if len(positive_seconds_per_beat) > 0:
        local_tempo_bpm = 60.0 / positive_seconds_per_beat
        tempo_summary = {
            "local_tempo_bpm_min": float(np.min(local_tempo_bpm)),
            "local_tempo_bpm_median": float(np.median(local_tempo_bpm)),
            "local_tempo_bpm_max": float(np.max(local_tempo_bpm)),
        }
    else:
        tempo_summary = {
            "local_tempo_bpm_min": None,
            "local_tempo_bpm_median": None,
            "local_tempo_bpm_max": None,
        }

    duplicate_score_times = len(tempomap[:, 0]) - len(np.unique(tempomap[:, 0]))
    duplicate_audio_times = len(tempomap[:, 1]) - len(np.unique(tempomap[:, 1]))

    flat_steps = (score_diffs > eps) & (np.abs(audio_diffs) <= 0.02)
    backwards_steps = audio_diffs < -0.02

    result = {
        "tempomap_points": int(len(tempomap)),
        "grouped_tempomap_points": int(len(grouped)),
        "score_time_min": float(np.min(score_times)),
        "score_time_max": float(np.max(score_times)),
        "audio_time_min": float(np.min(audio_times)),
        "audio_time_max": float(np.max(audio_times)),
        "duplicate_score_times": int(duplicate_score_times),
        "duplicate_audio_times": int(duplicate_audio_times),
        "flat_steps_20ms": int(np.sum(flat_steps)),
        "backwards_steps_20ms": int(np.sum(backwards_steps)),
        "max_flat_run_20ms": max_boolean_run(flat_steps),
    }

    result.update(tempo_summary)
    return result


def plot_error_curve(
    path: Path, rows: list[ReferenceRow], tolerances: list[float]
) -> None:
    valid_rows = [row for row in rows if row["valid"]]

    score_times = np.asarray([row["score_time"] for row in valid_rows], dtype=float)
    errors = np.asarray([row["error_seconds"] for row in valid_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(score_times, errors, marker="o", linewidth=1.5)
    ax.axhline(0.0, linewidth=1.0)

    if 0.5 in tolerances:
        ax.axhline(0.5, linestyle="--", linewidth=1.0)
        ax.axhline(-0.5, linestyle="--", linewidth=1.0)

    if 1.0 in tolerances:
        ax.axhline(1.0, linestyle=":", linewidth=1.0)
        ax.axhline(-1.0, linestyle=":", linewidth=1.0)

    ax.set_title("Chyba predikcie času zvukovej nahrávky na referenčných bodoch")
    ax.set_xlabel("čas notového zápisu [doby]")
    ax.set_ylabel("chyba predikcie [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = ensure_parent(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_song(
    song_id: str,
    output_dir: Path,
    references_path: Path,
    tempomap_kind: str,
    tolerances: list[float],
) -> dict[str, Any]:
    piece_dir = output_dir / song_id
    evaluation_dir = piece_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    tempomap = load_tempomap(piece_dir, kind=tempomap_kind)
    reference_points = load_reference_points(references_path, song_id)

    rows = evaluate_reference_points(tempomap, reference_points)
    summary = summarize_errors(rows, tolerances)

    tempomap_diagnostics = diagnose_tempomap(tempomap)
    path_diagnostics = diagnose_path(piece_dir)

    result = {
        "song": song_id,
        "tempomap_kind": tempomap_kind,
        "summary": summary,
        "tempomap_diagnostics": tempomap_diagnostics,
        "path_diagnostics": path_diagnostics,
    }

    save_reference_errors_csv(evaluation_dir / "reference_errors.csv", rows)
    save_json(evaluation_dir / "evaluation_summary.json", result)
    plot_error_curve(evaluation_dir / "error_plot.png", rows, tolerances)

    return result


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]

    print()
    print(f"=== {result['song']} ===")
    print(f"tempomap: {result['tempomap_kind']}")
    print(f"valid points: {summary['num_points_valid']}/{summary['num_points_total']}")
    print(f"MAE: {summary['mae_seconds']:.3f} s")
    print(f"Median AE: {summary['median_absolute_error_seconds']:.3f} s")
    print(f"Max AE: {summary['max_absolute_error_seconds']:.3f} s")

    for key, value in summary.items():
        if key.startswith("alignment_rate_at_"):
            print(f"{key}: {100.0 * value:.1f} %")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated score-audio tempomaps using reference points."
    )

    parser.add_argument(
        "--song",
        type=str,
        default=None,
        help="Evaluate one song ID.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all songs from config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory containing generated alignment outputs.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES_PATH,
        help="CSV file with manual reference points.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SONGS_CONFIG,
        help="Path to songs config JSON file.",
    )
    parser.add_argument(
        "--tempomap",
        choices=["smooth", "raw"],
        default="smooth",
        help="Which tempomap to evaluate.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=DEFAULT_TOLERANCES,
        help="Tolerance thresholds in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all and args.song is None:
        raise ValueError("Use either --song SONG_ID or --all.")

    if args.all:
        songs = load_song_config(args.config)
        selected_song_ids = list(songs.keys())
    else:
        selected_song_ids = [args.song]

    all_results = []

    for song_id in selected_song_ids:
        result = evaluate_song(
            song_id=song_id,
            output_dir=args.output,
            references_path=args.references,
            tempomap_kind=args.tempomap,
            tolerances=args.tolerances,
        )
        print_summary(result)
        all_results.append(result)

    if len(all_results) > 1:
        save_json(args.output / "evaluation_summary_all.json", {"songs": all_results})


if __name__ == "__main__":
    main()
