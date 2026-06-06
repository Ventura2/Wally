"""Manifest generation for exported WebDataset shards."""

import json
import os
from pathlib import Path
from typing import Any


def generate_manifest(
    shard_infos: list[tuple[str, int]],
    output_dir: str | Path,
    episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Generate and write manifest.json for an exported dataset.

    Args:
        shard_infos: List of (shard_path, transition_count) tuples as returned
            by ``ShardWriter.write_shards()``.
        output_dir: Directory where manifest.json will be written.  Created
            automatically if it does not exist.
        episode_ids: Optional set of unique episode IDs.  When provided the
            manifest's ``total_episodes`` field is set to ``len(episode_ids)``.
            When ``None``, ``total_episodes`` is set to 0.

    Returns:
        The manifest dictionary that was written to disk.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_transitions = sum(count for _, count in shard_infos)

    shards = [
        {"path": Path(path).name, "transitions": count}
        for path, count in shard_infos
    ]

    manifest: dict[str, Any] = {
        "total_transitions": total_transitions,
        "total_episodes": len(episode_ids) if episode_ids is not None else 0,
        "shard_count": len(shard_infos),
        "shards": shards,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
