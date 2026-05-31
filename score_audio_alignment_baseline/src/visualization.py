from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.feature_utils import normalize_rows

CHROMA_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

AX_SCORE_TIME = "čas notového zápisu [doby]"
AX_AUDIO_TIME = "čas zvukovej nahrávky [s]"
AX_SCORE_FRAME = "rámec notového zápisu"
AX_AUDIO_FRAME = "rámec zvukovej nahrávky"
AX_PITCH_CLASS = "tónová trieda"
CB_CHROMA = "intenzita"
CB_COST = "náklad"

TITLE_SCORE_CHROMA = "Chroma reprezentácia notového zápisu"
TITLE_AUDIO_CHROMA = "Chroma reprezentácia zvukovej nahrávky"
TITLE_ALIGNED_CHROMAS = "Porovnanie chroma reprezentácií po zarovnaní"
TITLE_COST_MATRIX = "Matica nákladov so zarovnávacou cestou DTW"
TITLE_LOCAL_COST_MATRIX = "Lokálna matica nákladov so zarovnávacou cestou DTW"
TITLE_TEMPOMAP_RAW = "Surová tempomapa"
TITLE_TEMPOMAP_SMOOTH = "Vyhladená tempomapa"
TITLE_TEMPOMAP_COMPARISON = "Porovnanie surovej a vyhladenej tempomapy"


def _validate_chroma(chroma: np.ndarray) -> None:
    if chroma.ndim != 2:
        raise ValueError("chroma must be a 2D array")
    if chroma.shape[1] != 12:
        raise ValueError("chroma must have 12 pitch-class columns")


def _validate_times(times: np.ndarray, expected_length: int) -> None:
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if len(times) != expected_length:
        raise ValueError("times length must match the number of frames")


def _validate_path(path: np.ndarray) -> None:
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (n_points, 2)")


def _validate_tempomap(tempomap: np.ndarray, name: str = "tempomap") -> None:
    if tempomap.ndim != 2 or tempomap.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_points, 2)")


def _finish_figure(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def _format_chroma_axis(ax: plt.Axes, ylabel: str = AX_PITCH_CLASS) -> None:
    ax.set_yticks(range(12))
    ax.set_yticklabels(CHROMA_LABELS)
    ax.set_ylabel(ylabel)


def _plot_path_overlay(
    ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray
) -> None:
    ax.plot(x_values, y_values, color="white", linewidth=3.2, alpha=0.95, zorder=3)
    ax.plot(x_values, y_values, color="red", linewidth=1.8, alpha=1.0, zorder=4)


def plot_chroma(
    chroma: np.ndarray,
    times: np.ndarray | None = None,
    title: str = "Chroma reprezentácia",
    x_label: str = "čas",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    _validate_chroma(chroma)
    if times is not None:
        _validate_times(times, chroma.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))

    if times is None:
        image = ax.imshow(
            chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlabel("rámec")
    else:
        extent = [float(times[0]), float(times[-1]), -0.5, 11.5]
        image = ax.imshow(
            chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_xlabel(x_label)

    _format_chroma_axis(ax)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=CB_CHROMA)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def warp_score_chroma_to_audio_time(
    score_chroma: np.ndarray,
    path: np.ndarray,
    n_audio_frames: int,
) -> np.ndarray:
    """Project score chroma to the audio timeline using the DTW path."""
    _validate_chroma(score_chroma)
    _validate_path(path)

    if n_audio_frames <= 0:
        raise ValueError("n_audio_frames must be greater than 0")
    if len(path) == 0:
        raise ValueError("path must not be empty")

    warped = np.full((n_audio_frames, score_chroma.shape[1]), np.nan, dtype=float)
    score_indices_by_audio_frame: list[list[int]] = [[] for _ in range(n_audio_frames)]

    for score_idx, audio_idx in path:
        if 0 <= audio_idx < n_audio_frames:
            score_indices_by_audio_frame[audio_idx].append(score_idx)

    for audio_idx, score_indices in enumerate(score_indices_by_audio_frame):
        if score_indices:
            warped[audio_idx] = score_chroma[np.asarray(score_indices)].mean(axis=0)

    valid = np.flatnonzero(~np.isnan(warped).all(axis=1))
    if len(valid) == 0:
        raise ValueError("could not warp score chroma to audio time")

    if valid[0] > 0:
        warped[: valid[0]] = warped[valid[0]]
    if valid[-1] < n_audio_frames - 1:
        warped[valid[-1] + 1 :] = warped[valid[-1]]

    frame_indices = np.arange(n_audio_frames)
    for pitch_class in range(warped.shape[1]):
        warped[:, pitch_class] = np.interp(
            frame_indices,
            valid,
            warped[valid, pitch_class],
        )

    return normalize_rows(warped)


def plot_aligned_chromas(
    score_chroma_on_audio_time: np.ndarray,
    audio_chroma: np.ndarray,
    audio_times: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title_prefix: str = "",
) -> None:
    _validate_chroma(score_chroma_on_audio_time)
    _validate_chroma(audio_chroma)
    _validate_times(audio_times, audio_chroma.shape[0])

    if score_chroma_on_audio_time.shape[0] != audio_chroma.shape[0]:
        raise ValueError("score and audio chroma must have the same number of frames")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    extent = [float(audio_times[0]), float(audio_times[-1]), -0.5, 11.5]

    score_image = axes[0].imshow(
        score_chroma_on_audio_time.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    audio_image = axes[1].imshow(
        audio_chroma.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    axes[0].set_title(
        f"{title_prefix}Chroma notového zápisu premietnutá na časovú os zvukovej nahrávky"
    )
    axes[1].set_title(f"{title_prefix}{TITLE_AUDIO_CHROMA}")
    axes[1].set_xlabel(AX_AUDIO_TIME)

    for ax in axes:
        _format_chroma_axis(ax)

    fig.colorbar(score_image, ax=axes[0], label=CB_CHROMA)
    fig.colorbar(audio_image, ax=axes[1], label=CB_CHROMA)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_cost_matrix_with_path(
    cost: np.ndarray,
    path: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = TITLE_COST_MATRIX,
) -> None:
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")
    _validate_path(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cost.T, origin="lower", aspect="auto", interpolation="nearest")
    _plot_path_overlay(ax, path[:, 0], path[:, 1])
    ax.set_xlabel(AX_SCORE_FRAME)
    ax.set_ylabel(AX_AUDIO_FRAME)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=CB_COST)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_tempomap(
    tempomap: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Tempomapa",
) -> None:
    _validate_tempomap(tempomap)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tempomap[:, 0], tempomap[:, 1], linewidth=1.8)
    ax.set_xlabel(AX_SCORE_TIME)
    ax.set_ylabel(AX_AUDIO_TIME)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_tempomap_comparison(
    tempomap_raw: np.ndarray,
    tempomap_smooth: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = TITLE_TEMPOMAP_COMPARISON,
) -> None:
    _validate_tempomap(tempomap_raw, "tempomap_raw")
    _validate_tempomap(tempomap_smooth, "tempomap_smooth")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        tempomap_raw[:, 0],
        tempomap_raw[:, 1],
        linewidth=1.0,
        alpha=0.45,
        label=TITLE_TEMPOMAP_RAW,
    )
    ax.plot(
        tempomap_smooth[:, 0],
        tempomap_smooth[:, 1],
        linewidth=2.0,
        label=TITLE_TEMPOMAP_SMOOTH,
    )
    ax.set_xlabel(AX_SCORE_TIME)
    ax.set_ylabel(AX_AUDIO_TIME)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_local_cost_matrix_with_path(
    cost: np.ndarray,
    path: np.ndarray,
    score_times: np.ndarray,
    audio_times: np.ndarray,
    score_min: float,
    score_max: float,
    audio_min: float,
    audio_max: float,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = TITLE_LOCAL_COST_MATRIX,
) -> None:
    score_mask = (score_times >= score_min) & (score_times <= score_max)
    audio_mask = (audio_times >= audio_min) & (audio_times <= audio_max)

    score_idx = np.flatnonzero(score_mask)
    audio_idx = np.flatnonzero(audio_mask)

    if len(score_idx) == 0 or len(audio_idx) == 0:
        raise ValueError("Selected local window is empty.")

    score_start, score_end = score_idx[0], score_idx[-1]
    audio_start, audio_end = audio_idx[0], audio_idx[-1]
    local_cost = cost[score_start : score_end + 1, audio_start : audio_end + 1]

    local_path = [
        (score_frame - score_start, audio_frame - audio_start)
        for score_frame, audio_frame in path
        if score_start <= score_frame <= score_end
        and audio_start <= audio_frame <= audio_end
    ]
    local_path = (
        np.asarray(local_path, dtype=int) if local_path else np.empty((0, 2), dtype=int)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [
        float(score_times[score_start]),
        float(score_times[score_end]),
        float(audio_times[audio_start]),
        float(audio_times[audio_end]),
    ]
    image = ax.imshow(
        local_cost.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    if len(local_path) > 0:
        path_score_times = score_times[score_start : score_end + 1][local_path[:, 0]]
        path_audio_times = audio_times[audio_start : audio_end + 1][local_path[:, 1]]
        _plot_path_overlay(ax, path_score_times, path_audio_times)

    ax.set_xlabel(AX_SCORE_TIME)
    ax.set_ylabel(AX_AUDIO_TIME)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=CB_COST)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)
