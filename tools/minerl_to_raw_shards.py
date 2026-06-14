#!/usr/bin/env python3
"""
Convert MineRL-v0 Zenodo datasets (ZIPs of MP4+NPZ trajectories)
into raw shards (.tar with .jpg + .json pairs) compatible with wally-convert.

Usage:
    python3 tools/minerl_to_raw_shards.py \
        --zips data/external/MineRLObtainIronPickaxe-v0.zip \
        --output data/raw/minerl_ironpickaxe \
        --episodes-per-shard 20

    python3 tools/minerl_to_raw_shards.py \
        --zips data/external/MineRL*.zip \
        --output data/raw/minerl_all \
        --episodes-per-shard 50
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import subprocess
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# MineRL-v0 action keys mapped to our schema
# Our schema (25 dims): forward, backward, left, right, jump, sneak, sprint,
# attack, use, drop, camera_pitch, camera_yaw, hotbar_1..9, inventory,
# pickItem, placeItem, craft
ACTION_MAP: dict[str, str] = {
    "action$forward": "forward",
    "action$back": "backward",
    "action$left": "left",
    "action$right": "right",
    "action$jump": "jump",
    "action$sneak": "sneak",
    "action$sprint": "sprint",
    "action$attack": "attack",
    "action$camera": "camera",  # special: 2D → pitch, yaw
}

OUR_SCHEMA = [
    "forward", "backward", "left", "right", "jump", "sneak", "sprint",
    "attack", "use", "drop",
    "camera_pitch", "camera_yaw",
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5",
    "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
    "inventory", "pickItem", "placeItem", "craft",
]

OUR_SCHEMA_SET = set(OUR_SCHEMA)


def build_action_dict(minerl_data: dict[str, Any], step: int) -> dict[str, float]:
    """Convert MineRL-v0 step actions to our action schema dict."""
    result: dict[str, float] = {}
    for key in ACTION_MAP:
        if key not in minerl_data:
            continue
        val = minerl_data[key]
        if key == "action$camera":
            camera = val[step]
            result["camera_pitch"] = float(camera[1])  # [yaw, pitch] → pitch
            result["camera_yaw"] = float(camera[0])    # [yaw, pitch] → yaw
        else:
            result[ACTION_MAP[key]] = float(val[step])
    # All other schema keys default to 0.0 (use, drop, hotbar, inventory, etc.)
    return result


def process_zip(
    zip_path: str,
    output_dir: Path,
    episodes_per_shard: int,
    max_episodes: int | None = None,
    shard_offset: int = 0,
    skip_trajs: set[str] | None = None,
) -> dict[str, int]:
    """Convert all trajectories in a ZIP to raw shards (streaming)."""
    zip_name = Path(zip_path).stem
    logger.info("Processing %s ...", zip_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            traj_dirs: set[str] = set()
            for name in zf.namelist():
                parts = name.strip("/").split("/")
                if len(parts) >= 2 and parts[1].startswith("v"):
                    traj_dirs.add(parts[1])
            traj_dirs = sorted(traj_dirs)
            if skip_trajs:
                skipped = len([t for t in traj_dirs if t in skip_trajs])
                traj_dirs = [t for t in traj_dirs if t not in skip_trajs]
                logger.info("  Skipping %d already-processed trajectories", skipped)
            if max_episodes:
                traj_dirs = traj_dirs[:max_episodes]
            logger.info("  Found %d trajectories", len(traj_dirs))

            episode_id = 0
            total_shards = 0
            total_steps = 0
            buffer: list[tuple[str, list[tuple[bytes, dict[str, float]]]]] = []
            # buffer stores (ep_label, [(jpg_bytes, action_dict), ...])

            for traj_name in traj_dirs:
                prefix = f"{zip_name}/{traj_name}/"
                npz_rel = meta_rel = mp4_rel = None
                for name in zf.namelist():
                    rel = name[len(prefix):]
                    if rel == "rendered.npz":
                        npz_rel = name
                    elif rel == "metadata.json":
                        meta_rel = name
                    elif rel == "recording.mp4":
                        mp4_rel = name
                if not all([npz_rel, meta_rel, mp4_rel]):
                    logger.warning("  Skipping %s: missing files", traj_name)
                    continue

                zf.extract(npz_rel, tmp_dir)
                zf.extract(meta_rel, tmp_dir)
                zf.extract(mp4_rel, tmp_dir)

                npz_path = os.path.join(tmp_dir, npz_rel)
                mp4_path = os.path.join(tmp_dir, mp4_rel)

                try:
                    data = np.load(npz_path)
                except Exception as e:
                    logger.warning("  Skipping %s: %s", traj_name, e)
                    continue

                steps = len(data["reward"])
                if steps < 2:
                    logger.warning("  Skipping %s: too few steps (%d)", traj_name, steps)
                    continue

                # Extract frames via ffmpeg pipe (OpenCV lacks MP4 support in this container)
                frame_dir = os.path.join(tmp_dir, f"frames_{traj_name}")
                os.makedirs(frame_dir, exist_ok=True)
                extract_cmd = [
                    "ffmpeg", "-i", mp4_path,
                    "-f", "image2",
                    os.path.join(frame_dir, "frame_%06d.jpg"),
                    "-loglevel", "error", "-y",
                ]
                subprocess.run(extract_cmd, check=True)
                frame_files = sorted(
                    f for f in os.listdir(frame_dir) if f.endswith(".jpg")
                )
                usable = 0
                episode_transitions: list[tuple[bytes, dict[str, float]]] = []

                for t in range(steps):
                    if t >= len(frame_files):
                        break
                    with open(os.path.join(frame_dir, frame_files[t]), "rb") as f:
                        jpg_bytes = f.read()
                    action_dict = build_action_dict(data, t)
                    episode_transitions.append((jpg_bytes, action_dict))
                    usable += 1

                if usable < 2:
                    continue

                ep_label = f"{zip_name}_{traj_name}"
                buffer.append((ep_label, episode_transitions))
                total_steps += usable
                episode_id += 1

                if len(buffer) >= episodes_per_shard:
                    total_shards += 1
                    _write_shard(output_dir, shard_offset + total_shards, buffer)
                    buffer = []

                if episode_id % 20 == 0:
                    logger.info("  Processed %d episodes, %d steps", episode_id, total_steps)

            if buffer:
                total_shards += 1
                _write_shard(output_dir, shard_offset + total_shards, buffer)

            logger.info(
                "  Done: %d episodes, %d shards, %d steps",
                episode_id, total_shards, total_steps,
            )
            return {"episodes": episode_id, "shards": total_shards, "steps": total_steps}


def _write_shard(
    output_dir: Path,
    shard_num: int,
    buffer: list[tuple[str, list[tuple[bytes, dict[str, float]]]]],
) -> None:
    """Write one raw shard .tar file (pre-encoded .jpg bytes)."""
    shard_path = output_dir / f"shard_{shard_num:06d}.tar"
    with tarfile.open(shard_path, "w") as tar:
        for ep_label, transitions in buffer:
            for t, (jpg_bytes, action_dict) in enumerate(transitions):
                meta = {
                    "episode_id": ep_label,
                    "step_index": t,
                    "action": action_dict,
                    "timestamp": 0.0,
                    "frame_skip": 1,
                    "seed": 0,
                }
                json_bytes = json.dumps(meta).encode("utf-8")

                key = f"{ep_label}_{t:06d}"

                jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
                jpg_info.size = len(jpg_bytes)
                tar.addfile(jpg_info, io.BytesIO(jpg_bytes))

                json_info = tarfile.TarInfo(name=f"{key}.json")
                json_info.size = len(json_bytes)
                tar.addfile(json_info, io.BytesIO(json_bytes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zips",
        nargs="+",
        required=True,
        help="MineRL-v0 ZIP files (supports globs like data/external/MineRL*.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for raw shards",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=20,
        help="Episodes per raw shard (default: 20)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Max episodes to process per ZIP (for testing)",
    )
    parser.add_argument(
        "--skip-trajectories-file",
        type=Path,
        default=None,
        help="File with trajectory names (one per line) to skip",
    )
    parser.add_argument(
        "--shard-start",
        type=int,
        default=1,
        help="Starting shard number (default: 1)",
    )
    args = parser.parse_args()

    skip_trajs: set[str] | None = None
    if args.skip_trajectories_file:
        with open(args.skip_trajectories_file) as f:
            skip_trajs = {line.strip() for line in f if line.strip()}
            logger.info("Skipping %d already-processed trajectories", len(skip_trajs))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand globs
    zip_paths: list[str] = []
    for pattern in args.zips:
        from glob import glob
        matches = glob(pattern)
        if matches:
            zip_paths.extend(matches)
        else:
            zip_paths.append(pattern)

    total_eps = 0
    total_shards = args.shard_start - 1  # offset so first shard uses --shard-start
    total_steps = 0

    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            logger.warning("Skipping %s: not found", zip_path)
            continue
        stats = process_zip(
            zip_path, output_dir, args.episodes_per_shard, args.max_episodes, total_shards,
            skip_trajs=skip_trajs,
        )
        total_eps += stats["episodes"]
        total_shards += stats["shards"]
        total_steps += stats["steps"]

    logger.info(
        "Total: %d episodes, %d shards, %d steps written to %s",
        total_eps, total_shards, total_steps, output_dir,
    )


if __name__ == "__main__":
    main()
