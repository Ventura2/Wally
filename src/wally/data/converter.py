from __future__ import annotations

import io
import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def convert_shards(
    input_dir: str | Path,
    output_dir: str | Path,
    action_schema: list[str],
    episodes_per_shard: int = 50,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_raw_shards(input_dir)

    if not episodes:
        return {
            "episode_count": 0,
            "shard_count": 0,
            "skipped_episodes": 0,
            "total_transitions": 0,
        }

    processed: list[tuple[str, Any, Any]] = []
    skipped = 0
    total_transitions = 0

    for episode_id in sorted(episodes):
        transitions = episodes[episode_id]
        if not transitions:
            skipped += 1
            continue

        transitions.sort(key=lambda t: t["step_index"])
        total_transitions += len(transitions)

        frames_list: list[Any] = []
        actions_list: list[Any] = []

        for t in transitions:
            frames_list.append(t["observation"])
            actions_list.append(_encode_action(t["action"], action_schema))

        frames_arr = np.stack(frames_list, axis=0)
        actions_arr = np.stack(actions_list, axis=0).astype(np.float32)
        processed.append((episode_id, frames_arr, actions_arr))

    shard_count = 0
    for i in range(0, len(processed), episodes_per_shard):
        chunk = processed[i : i + episodes_per_shard]
        shard_path = output_dir / f"shard_{shard_count:06d}.tar"
        episode_data = [(frames, actions) for _, frames, actions in chunk]
        episode_ids = [eid for eid, _, _ in chunk]
        _write_training_shard(shard_path, episode_data, episode_ids)
        shard_count += 1

    return {
        "episode_count": len(processed),
        "shard_count": shard_count,
        "skipped_episodes": skipped,
        "total_transitions": total_transitions,
    }


def _load_raw_shards(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tar_files = sorted(input_dir.rglob("*.tar"))

    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tar:
            members_by_key: dict[str, dict[str, bytes]] = defaultdict(dict)

            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith(".jpg"):
                    key = name[: -len(".jpg")]
                    f = tar.extractfile(member)
                    if f is not None:
                        members_by_key[key]["jpg"] = f.read()
                elif name.endswith(".json"):
                    key = name[: -len(".json")]
                    f = tar.extractfile(member)
                    if f is not None:
                        members_by_key[key]["json"] = f.read()

            for key, parts in members_by_key.items():
                if "jpg" not in parts or "json" not in parts:
                    logger.warning("Incomplete pair for key %s in %s", key, tar_path)
                    continue

                try:
                    observation = _decode_jpeg(parts["jpg"])
                except Exception:
                    logger.warning("Failed to decode JPEG for key %s", key)
                    continue

                try:
                    meta = json.loads(parts["json"].decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("Failed to parse JSON for key %s", key)
                    continue

                episode_id = meta.get("episode_id", "")
                step_index = meta.get("step_index", 0)
                action = meta.get("action", {})

                episodes[episode_id].append(
                    {
                        "observation": observation,
                        "action": action,
                        "episode_id": episode_id,
                        "step_index": step_index,
                    }
                )

    return dict(episodes)


def _encode_action(action_dict: dict[str, Any], schema: list[str]) -> Any:
    vector = np.zeros(len(schema), dtype=np.float32)
    for i, key in enumerate(schema):
        if key in action_dict:
            vector[i] = float(action_dict[key])
        else:
            logger.warning("Missing action key '%s', defaulting to 0.0", key)

    extra = set(action_dict.keys()) - set(schema)
    if extra:
        logger.warning("Extra action keys ignored: %s", sorted(extra))

    return vector


def _decode_jpeg(jpeg_bytes: bytes) -> Any:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _write_training_shard(
    shard_path: Path,
    episodes: list[tuple[Any, Any]],
    episode_ids: list[str] | None = None,
) -> None:
    with tarfile.open(shard_path, "w") as tar:
        for i, (frames, actions) in enumerate(episodes):
            if episode_ids is not None:
                key = episode_ids[i]
            else:
                key = f"episode_{i:06d}"

            buf = io.BytesIO()
            np.savez_compressed(buf, frames=frames, actions=actions)
            data = buf.getvalue()

            info = tarfile.TarInfo(name=f"{key}.npz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
