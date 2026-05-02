from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from src.config import DEFAULT_SONGS_CONFIG, load_song_config

DEFAULT_CONFIG_PATH = DEFAULT_SONGS_CONFIG
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_WEB_DATA_DIR = Path("web/data")


def ensure_dirs(web_data_dir: Path) -> dict[str, Path]:
    paths = {
        "alignment": web_data_dir / "alignment",
        "audio": web_data_dir / "audio",
        "score": web_data_dir / "score",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_song_web_data(
    song_id: str,
    song: dict,
    output_dir: Path,
    web_paths: dict[str, Path],
) -> None:
    score_path = Path(song["score"])
    audio_path = Path(song["audio"])

    exports_dir = output_dir / song_id / "exports"

    tempomap_path = exports_dir / "tempomap.json"
    score_beats_path = exports_dir / "score_beats.json"

    copy_file(
        tempomap_path,
        web_paths["alignment"] / f"{song_id}_tempomap.json",
    )
    copy_file(
        score_beats_path,
        web_paths["alignment"] / f"{song_id}_score_beats.json",
    )
    copy_file(
        audio_path,
        web_paths["audio"] / audio_path.name,
    )
    copy_file(
        score_path,
        web_paths["score"] / score_path.name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare files required by the web visualization."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to songs config JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing generated alignment outputs.",
    )
    parser.add_argument(
        "--web-data",
        type=Path,
        default=DEFAULT_WEB_DATA_DIR,
        help="Target web data directory.",
    )
    parser.add_argument(
        "--song",
        type=str,
        default=None,
        help="Prepare only one song ID. If omitted, all songs are prepared.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    songs = load_song_config(args.config)
    web_paths = ensure_dirs(args.web_data)

    if args.song is not None:
        if args.song not in songs:
            available = ", ".join(sorted(songs.keys()))
            raise KeyError(
                f"Song '{args.song}' is not defined in {args.config}. "
                f"Available songs: {available}"
            )

        selected_songs = {args.song: songs[args.song]}
    else:
        selected_songs = songs

    for song_id, song in selected_songs.items():
        print(f"Preparing web data for {song_id}")
        prepare_song_web_data(
            song_id=song_id,
            song=song,
            output_dir=args.output,
            web_paths=web_paths,
        )

    print(f"Web data prepared in: {args.web_data}")


if __name__ == "__main__":
    main()
