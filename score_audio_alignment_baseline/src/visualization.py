from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHROMA_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _finish_figure(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def _format_chroma_axis(ax, ylabel: str = "Pitch class") -> None:
    ax.set_yticks(range(12))
    ax.set_yticklabels(CHROMA_LABELS)
    ax.set_ylabel(ylabel)


def _plot_path_overlay(ax, path: np.ndarray) -> None:
    ax.plot(path[:, 0], path[:, 1], color="white", linewidth=3.2, alpha=0.95, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color="red", linewidth=1.8, alpha=1.0, zorder=4)


def plot_chroma(
    chroma: np.ndarray,
    times: np.ndarray | None = None,
    title: str = "Chroma",
    x_label: str = "Time",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    if times is not None and len(times) == chroma.shape[0]:
        extent = [float(times[0]), float(times[-1]), -0.5, 11.5]
        image = ax.imshow(
            chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_xlabel(x_label)
    else:
        image = ax.imshow(
            chroma.T, origin="lower", aspect="auto", interpolation="nearest"
        )
        ax.set_xlabel("Frame")

    _format_chroma_axis(ax)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def warp_score_chroma_to_audio_time(
    score_chroma: np.ndarray,
    path: np.ndarray,
    n_audio_frames: int,
) -> np.ndarray:
    if len(path) == 0:
        raise ValueError("path must not be empty")

    warped = np.full((n_audio_frames, score_chroma.shape[1]), np.nan, dtype=float)
    buckets: list[list[int]] = [[] for _ in range(n_audio_frames)]

    for score_idx, audio_idx in path:
        if 0 <= audio_idx < n_audio_frames:
            buckets[audio_idx].append(score_idx)

    for audio_idx, score_indices in enumerate(buckets):
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
            frame_indices, valid, warped[valid, pitch_class]
        )

    return _normalize_rows(warped)


def plot_aligned_chromas(
    score_chroma_on_audio_time: np.ndarray,
    audio_chroma: np.ndarray,
    audio_times: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title_prefix: str = "",
) -> None:
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

    axes[0].set_title(f"{title_prefix}Score chroma mapped to audio time")
    axes[1].set_title(f"{title_prefix}Audio chroma")
    axes[1].set_xlabel("Audio time [s]")

    for ax in axes:
        _format_chroma_axis(ax)

    fig.colorbar(score_image, ax=axes[0])
    fig.colorbar(audio_image, ax=axes[1])
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_cost_matrix_with_path(
    cost: np.ndarray,
    path: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Cost matrix with DTW path",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cost.T, origin="lower", aspect="auto", interpolation="nearest")
    _plot_path_overlay(ax, path)
    ax.set_xlabel("Score frame")
    ax.set_ylabel("Audio frame")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_tempomap(
    tempomap: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Tempomap",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tempomap[:, 0], tempomap[:, 1], linewidth=1.8)
    ax.set_xlabel("Score beat")
    ax.set_ylabel("Audio time [s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_tempomap_comparison(
    tempomap_raw: np.ndarray,
    tempomap_smooth: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Raw vs. smoothed tempomap",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        tempomap_raw[:, 0],
        tempomap_raw[:, 1],
        linewidth=1.0,
        alpha=0.45,
        label="Raw tempomap",
    )
    ax.plot(
        tempomap_smooth[:, 0],
        tempomap_smooth[:, 1],
        linewidth=2.0,
        label="Smoothed tempomap",
    )
    ax.set_xlabel("Score beat")
    ax.set_ylabel("Audio time [s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_paths_side_by_side(
    cost: np.ndarray,
    forward_path: np.ndarray,
    reverse_path: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    image = None

    for ax, (title, path) in zip(
        axes, [("Forward", forward_path), ("Reverse", reverse_path)]
    ):
        image = ax.imshow(
            cost.T, origin="lower", aspect="auto", interpolation="nearest"
        )
        _plot_path_overlay(ax, path)
        ax.set_title(title)
        ax.set_xlabel("Score frame")

    axes[0].set_ylabel("Audio frame")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.85)

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
    title: str = "Local cost matrix with DTW path",
) -> None:
    score_mask = (score_times >= score_min) & (score_times <= score_max)
    audio_mask = (audio_times >= audio_min) & (audio_times <= audio_max)

    score_idx = np.flatnonzero(score_mask)
    audio_idx = np.flatnonzero(audio_mask)

    if len(score_idx) == 0 or len(audio_idx) == 0:
        raise ValueError("Selected local window is empty.")

    s0, s1 = score_idx[0], score_idx[-1]
    a0, a1 = audio_idx[0], audio_idx[-1]

    local_cost = cost[s0 : s1 + 1, a0 : a1 + 1]

    local_path = []
    for si, ai in path:
        if s0 <= si <= s1 and a0 <= ai <= a1:
            local_path.append((si - s0, ai - a0))

    local_path = (
        np.asarray(local_path, dtype=int) if local_path else np.empty((0, 2), dtype=int)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [
        float(score_times[s0]),
        float(score_times[s1]),
        float(audio_times[a0]),
        float(audio_times[a1]),
    ]

    image = ax.imshow(
        local_cost.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    if len(local_path) > 0:
        path_score_times = score_times[s0 : s1 + 1][local_path[:, 0]]
        path_audio_times = audio_times[a0 : a1 + 1][local_path[:, 1]]
        ax.plot(
            path_score_times,
            path_audio_times,
            color="white",
            linewidth=3.2,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            path_score_times,
            path_audio_times,
            color="red",
            linewidth=1.8,
            alpha=1.0,
            zorder=4,
        )

    ax.set_xlabel("Score time [beats]")
    ax.set_ylabel("Audio time [s]")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)
