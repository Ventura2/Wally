"""Inspector and validator for WebDataset trajectory shards."""

import json
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def inspect_shard(shard_path: str | Path) -> dict[str, Any]:
    """Inspect a .tar shard and extract summary statistics.

    Args:
        shard_path: Path to the .tar shard file.

    Returns:
        Dict with keys: transition_count, episode_count, observation_shape,
        action_keys, timestamp_range (min, max).
    """
    shard_path = Path(shard_path)
    keys_by_prefix: dict[str, set[str]] = defaultdict(set)
    episode_ids: set[str] = set()
    action_keys: set[str] = set()
    timestamps: list[float] = []
    observation_shape: tuple[int, ...] | None = None

    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.endswith(".jpg"):
                prefix = name[:-4]
                keys_by_prefix[prefix].add("jpg")
                if observation_shape is None:
                    f = tar.extractfile(member)
                    if f is not None:
                        img = Image.open(f)
                        observation_shape = img.size  # (width, height)
            elif name.endswith(".json"):
                prefix = name[:-5]
                keys_by_prefix[prefix].add("json")
                f = tar.extractfile(member)
                if f is not None:
                    data = json.loads(f.read().decode("utf-8"))
                    if "episode_id" in data:
                        episode_ids.add(data["episode_id"])
                    if "action" in data and isinstance(data["action"], dict):
                        action_keys.update(data["action"].keys())
                    if "timestamp" in data:
                        timestamps.append(float(data["timestamp"]))

    transition_count = len(keys_by_prefix)
    timestamp_range = (min(timestamps), max(timestamps)) if timestamps else (0.0, 0.0)

    return {
        "transition_count": transition_count,
        "episode_count": len(episode_ids),
        "observation_shape": observation_shape,
        "action_keys": sorted(action_keys),
        "timestamp_range": timestamp_range,
    }


def validate_shard(shard_path: str | Path) -> dict[str, Any]:
    """Validate a .tar shard for schema correctness and data integrity.

    Checks that every .jpg has a matching .json and vice versa, and that
    every .jpg can be decoded without errors.

    Args:
        shard_path: Path to the .tar shard file.

    Returns:
        Dict with keys: valid (bool), errors (list of error strings with
        sample keys).
    """
    shard_path = Path(shard_path)
    errors: list[str] = []
    jpg_keys: set[str] = set()
    json_keys: set[str] = set()
    corrupt_jpgs: list[str] = []

    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.endswith(".jpg"):
                jpg_keys.add(name[:-4])
            elif name.endswith(".json"):
                json_keys.add(name[:-5])

        missing_json = jpg_keys - json_keys
        missing_jpg = json_keys - jpg_keys

        if missing_json:
            samples = sorted(missing_json)[:3]
            errors.append(
                f"{len(missing_json)} .jpg files have no matching .json. "
                f"Sample keys: {samples}"
            )

        if missing_jpg:
            samples = sorted(missing_jpg)[:3]
            errors.append(
                f"{len(missing_jpg)} .json files have no matching .jpg. "
                f"Sample keys: {samples}"
            )

        for member in tar.getmembers():
            if member.name.endswith(".jpg"):
                try:
                    f = tar.extractfile(member)
                    if f is not None:
                        img = Image.open(f)
                        img.load()
                except Exception:
                    corrupt_jpgs.append(member.name)

        if corrupt_jpgs:
            samples = corrupt_jpgs[:3]
            errors.append(
                f"{len(corrupt_jpgs)} corrupt JPEG files detected. "
                f"Sample keys: {samples}"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def compute_action_distribution(
    shard_path_or_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Compute per-action-key statistics from shard JSON sidecars.

    For numeric values: mean, std, min, max.
    For discrete values (strings, bools, ints with few unique values): value counts.

    Args:
        shard_path_or_dir: Path to a single .tar shard or a directory
            containing .tar shards.

    Returns:
        Dict mapping action_key -> stats dict.
    """
    path = Path(shard_path_or_dir)
    shard_paths: list[Path] = []

    if path.is_dir():
        shard_paths = sorted(path.glob("*.tar"))
    else:
        shard_paths = [path]

    action_values: dict[str, list[Any]] = defaultdict(list)

    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    if f is not None:
                        data = json.loads(f.read().decode("utf-8"))
                        if "action" in data and isinstance(data["action"], dict):
                            for key, value in data["action"].items():
                                action_values[key].append(value)

    distributions: dict[str, dict[str, Any]] = {}
    for key, values in action_values.items():
        if _is_numeric(values):
            nums = [float(v) for v in values]
            distributions[key] = {
                "type": "continuous",
                "mean": statistics.mean(nums),
                "std": statistics.stdev(nums) if len(nums) > 1 else 0.0,
                "min": min(nums),
                "max": max(nums),
            }
        else:
            counts: dict[Any, int] = defaultdict(int)
            for v in values:
                counts[v] += 1
            distributions[key] = {
                "type": "discrete",
                "value_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
                "unique_values": len(counts),
            }

    return distributions


def _is_numeric(values: list[Any]) -> bool:
    """Check if all values are numeric (int or float)."""
    return all(isinstance(v, (int, float)) for v in values)


def inspect_dataset_dir(dataset_dir: str | Path) -> dict[str, Any]:
    """Inspect all shards in a dataset directory and aggregate statistics.

    Reads manifest.json if present for expected counts.

    Args:
        dataset_dir: Path to directory containing .tar shards and optionally
            a manifest.json.

    Returns:
        Dict with aggregated stats: total_transitions, total_episodes,
        shard_count, per_shard stats, and manifest info if available.
    """
    dataset_dir = Path(dataset_dir)
    shard_paths = sorted(dataset_dir.glob("*.tar"))

    manifest: dict[str, Any] | None = None
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    all_episode_ids: set[str] = set()
    total_transitions = 0
    all_action_keys: set[str] = set()
    all_timestamps: list[float] = []
    observation_shape: tuple[int, ...] | None = None
    per_shard: list[dict[str, Any]] = []

    for shard_path in shard_paths:
        info = inspect_shard(shard_path)
        per_shard.append({"path": shard_path.name, **info})
        total_transitions += info["transition_count"]
        all_action_keys.update(info["action_keys"])
        ts_min, ts_max = info["timestamp_range"]
        if ts_min > 0:
            all_timestamps.extend([ts_min, ts_max])
        if observation_shape is None and info["observation_shape"] is not None:
            observation_shape = info["observation_shape"]

    timestamp_range = (
        (min(all_timestamps), max(all_timestamps)) if all_timestamps else (0.0, 0.0)
    )

    result: dict[str, Any] = {
        "total_transitions": total_transitions,
        "shard_count": len(shard_paths),
        "observation_shape": observation_shape,
        "action_keys": sorted(all_action_keys),
        "timestamp_range": timestamp_range,
        "per_shard": per_shard,
    }

    if manifest is not None:
        result["manifest"] = manifest

    return result
