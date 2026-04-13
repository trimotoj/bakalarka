from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHROMA_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _save_or_show(save_path: str | Path | None = None, show: bool = True) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _plot_path_overlay(ax, path: np.ndarray) -> None:
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="white",
        linewidth=3.2,
        alpha=0.95,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="red",
        linewidth=1.8,
        alpha=1.0,
        solid_capstyle="round",
        zorder=4,
    )


def plot_chroma(
    chroma: np.ndarray,
    times: np.ndarray | None = None,
    title: str = "Chroma",
    x_label: str = "Time",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(10, 4))

    if times is not None and len(times) == chroma.shape[0]:
        extent = [float(times[0]), float(times[-1]), -0.5, 11.5]
        plt.imshow(
            chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        plt.xlabel(x_label)
    else:
        plt.imshow(chroma.T, origin="lower", aspect="auto", interpolation="nearest")
        plt.xlabel("Frame")

    plt.yticks(range(12), CHROMA_LABELS)
    plt.ylabel("Pitch class")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    _save_or_show(save_path, show)


def warp_score_chroma_to_audio_time(
    score_chroma: np.ndarray,
    path: np.ndarray,
    n_audio_frames: int,
) -> np.ndarray:
    if len(path) == 0:
        raise ValueError("path must not be empty")

    warped = np.full((n_audio_frames, score_chroma.shape[1]), np.nan, dtype=float)

    buckets = [[] for _ in range(n_audio_frames)]
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

    x = np.arange(n_audio_frames)
    for pc in range(warped.shape[1]):
        warped[:, pc] = np.interp(x, valid, warped[valid, pc])

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

    im1 = axes[0].imshow(
        score_chroma_on_audio_time.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    im2 = axes[1].imshow(
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
        ax.set_yticks(range(12))
        ax.set_yticklabels(CHROMA_LABELS)
        ax.set_ylabel("Pitch class")

    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    _save_or_show(save_path, show)


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
    plt.tight_layout()
    _save_or_show(save_path, show)


def plot_tempomap(
    tempomap: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Tempomap",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(tempomap[:, 0], tempomap[:, 1], color="red", linewidth=1.8)
    plt.xlabel("Score beat")
    plt.ylabel("Audio time [s]")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path, show)


def plot_paths_side_by_side(
    cost: np.ndarray,
    forward_path: np.ndarray,
    reverse_path: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    variants = [
        ("Forward", forward_path),
        ("Reverse", reverse_path),
    ]

    image = None
    for ax, (title, path) in zip(axes, variants):
        image = ax.imshow(
            cost.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        _plot_path_overlay(ax, path)
        ax.set_title(title)
        ax.set_xlabel("Score frame")

    axes[0].set_ylabel("Audio frame")

    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.85)

    plt.tight_layout()
    _save_or_show(save_path, show)
