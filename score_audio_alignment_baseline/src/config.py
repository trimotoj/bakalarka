from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SONGS_CONFIG = Path("config/songs.json")
SongConfig = dict[str, dict[str, Any]]


def load_song_config(config_path: Path = DEFAULT_SONGS_CONFIG) -> SongConfig:
    """Load and validate the song configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Song config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    songs = data.get("songs")
    if not isinstance(songs, dict):
        raise ValueError("Invalid songs config. Expected top-level key 'songs'.")

    return songs


def get_song_from_config(
    song_id: str,
    config_path: Path = DEFAULT_SONGS_CONFIG,
) -> dict[str, Any]:
    """Return one song entry from the configuration."""
    songs = load_song_config(config_path)

    if song_id not in songs:
        available = ", ".join(sorted(songs.keys()))
        raise KeyError(
            f"Song '{song_id}' is not defined in {config_path}. "
            f"Available songs: {available}"
        )

    return songs[song_id]
