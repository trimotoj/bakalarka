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


def plot_both_chromas(
    score_chroma: np.ndarray,
    audio_chroma: np.ndarray,
    score_times: np.ndarray | None = None,
    audio_times: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=True)

    if score_times is not None and len(score_times) == score_chroma.shape[0]:
        extent = [float(score_times[0]), float(score_times[-1]), -0.5, 11.5]
        im1 = axes[0].imshow(
            score_chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        axes[0].set_xlabel("Score time [beats]")
    else:
        im1 = axes[0].imshow(
            score_chroma.T, origin="lower", aspect="auto", interpolation="nearest"
        )
        axes[0].set_xlabel("Score frame")

    if audio_times is not None and len(audio_times) == audio_chroma.shape[0]:
        extent = [float(audio_times[0]), float(audio_times[-1]), -0.5, 11.5]
        im2 = axes[1].imshow(
            audio_chroma.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        axes[1].set_xlabel("Audio time [s]")
    else:
        im2 = axes[1].imshow(
            audio_chroma.T, origin="lower", aspect="auto", interpolation="nearest"
        )
        axes[1].set_xlabel("Audio frame")

    axes[0].set_title("Score chroma")
    axes[1].set_title("Audio chroma")

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
) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cost.T, origin="lower", aspect="auto", interpolation="nearest")
    plt.plot(path[:, 0], path[:, 1], linewidth=1)
    plt.xlabel("Score frame")
    plt.ylabel("Audio frame")
    plt.title("Cost matrix with DTW path")
    plt.colorbar()
    plt.tight_layout()
    _save_or_show(save_path, show)


def plot_tempomap(
    tempomap: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(tempomap[:, 0], tempomap[:, 1])
    plt.xlabel("Score beat")
    plt.ylabel("Audio time [s]")
    plt.title("Tempomap")
    plt.tight_layout()
    _save_or_show(save_path, show)
