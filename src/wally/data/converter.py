from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
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
    shard_start: int = 1,
    chunk_frames: int = 64,
) -> dict[str, Any]:
    """Convert raw shards into training shards.

    Each episode is split into chunks of ``chunk_frames`` consecutive frames
    written as separate ``.npz`` entries. Chunking is what makes the data
    loader fast: a full MineRLTreechop episode is 144-335 MB compressed, so a
    batch of 16 samples = 2-5 GB of CPU work per step. Splitting at convert
    time brings each ``.npz`` down to ~17 MB (64 frames at 224x224 uint8), so
    the same batch is ~270 MB — roughly 10x less decompress + alloc work
    per step, and the GPU no longer starves.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(input_dir.rglob("*.tar"))
    if not tar_files:
        return {"episode_count": 0, "shard_count": 0, "skipped_episodes": 0, "total_transitions": 0}

    # Pass 1: extract each episode, split into chunks, write to a temp dir
    with tempfile.TemporaryDirectory(prefix="wally_convert_") as tmp_dir:
        tmp = Path(tmp_dir)
        episode_count = 0
        chunk_count = 0
        skipped = 0
        total_steps = 0

        for tar_path in tar_files:
            logger.info("Extracting %s ...", tar_path.name)
            for ep_id, frames, actions in _iter_episodes(tar_path, action_schema):
                if frames.shape[0] < 2:
                    skipped += 1
                    continue
                total_steps += frames.shape[0]
                episode_count += 1

                safe_name = ep_id.replace("/", "_").replace(":", "_")
                # Chunk along time dimension
                ep_len = frames.shape[0]
                n_chunks = (ep_len + chunk_frames - 1) // chunk_frames
                for ci in range(n_chunks):
                    s = ci * chunk_frames
                    e = min(s + chunk_frames, ep_len)
                    npz_path = tmp / f"{safe_name}__chunk{ci:03d}.npz"
                    buf = io.BytesIO()
                    np.savez_compressed(
                        buf, frames=frames[s:e], actions=actions[s:e]
                    )
                    with open(npz_path, "wb") as f:
                        f.write(buf.getvalue())
                    chunk_count += 1

                del frames, actions, buf

        # Pass 2: combine chunk .npz files into shards
        npz_files = sorted(tmp.glob("*.npz"))
        shard_count = shard_start - 1
        for i in range(0, len(npz_files), episodes_per_shard):
            chunk = npz_files[i : i + episodes_per_shard]
            shard_count += 1
            _write_shard_from_npz(chunk, output_dir, shard_count)

    return {
        "episode_count": episode_count,
        "chunk_count": chunk_count,
        "shard_count": shard_count,
        "skipped_episodes": skipped,
        "total_transitions": total_steps,
    }


def _write_shard_from_npz(npz_files: list[Path], output_dir: Path, shard_count: int) -> None:
    shard_path = output_dir / f"shard_{shard_count:06d}.tar"
    with tarfile.open(shard_path, "w") as tar:
        for npz_path in npz_files:
            with open(npz_path, "rb") as f:
                data = f.read()
            info = tarfile.TarInfo(name=f"{npz_path.stem}.npz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _iter_episodes(tar_path: Path, action_schema: list[str]) -> Any:
    """Stream raw shard entries, yield (episode_id, frames, actions) per episode."""
    with tarfile.open(tar_path, "r") as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.isfile()),
            key=lambda m: m.name,
        )

        current_ep_id: str | None = None
        frames: list[Any] = []
        actions: list[Any] = []

        def flush() -> Any:
            nonlocal current_ep_id, frames, actions
            if current_ep_id and len(frames) > 0 and len(actions) > 0:
                frames_arr = np.stack(frames, axis=0)
                actions_arr = np.stack(actions, axis=0).astype(np.float32)
                yield (current_ep_id, frames_arr, actions_arr)
            current_ep_id = None
            frames = []
            actions = []

        # First pass: collect .json meta (faster, no decoding)
        step_map: dict[str, dict[str, Any]] = {}
        for m in members:
            name = m.name
            if name.endswith(".json"):
                key = name[: -len(".json")]
                f = tar.extractfile(m)
                if f is None:
                    continue
                try:
                    meta = json.loads(f.read().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                step_map[key] = meta

        # Second pass: iterate by sorted keys, group by episode_id
        jpg_members = {m.name[:-4]: m for m in members if m.name.endswith(".jpg")}

        sorted_keys = sorted(k for k in step_map.keys() if k in jpg_members)

        for key in sorted_keys:
            meta = step_map[key]
            ep_id = meta.get("episode_id", "")

            if current_ep_id is not None and ep_id != current_ep_id:
                yield from flush()

            current_ep_id = ep_id
            act = meta.get("action", {})
            actions.append(_encode_action(act, action_schema))

            m = jpg_members[key]
            f = tar.extractfile(m)
            if f is None:
                continue
            try:
                obs = _decode_jpeg(f.read())
            except Exception:
                continue
            frames.append(obs)

        yield from flush()


def _encode_action(action_dict: dict[str, Any], schema: list[str]) -> Any:
    vector = np.zeros(len(schema), dtype=np.float32)
    for i, key in enumerate(schema):
        if key in action_dict:
            vector[i] = float(action_dict[key])
    return vector


def _load_raw_shards(input_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load all raw shards from a directory. Used by tests with small data."""
    from collections import defaultdict
    import tarfile as tf

    input_dir = Path(input_dir)
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tar_files = sorted(input_dir.rglob("*.tar"))

    for tar_path in tar_files:
        with tf.open(tar_path, "r") as tar:
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
                    continue
                try:
                    observation = _decode_jpeg(parts["jpg"])
                except Exception:
                    continue
                try:
                    meta = json.loads(parts["json"].decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                episode_id = meta.get("episode_id", "")
                step_index = meta.get("step_index", 0)
                action = meta.get("action", {})
                episodes[episode_id].append({
                    "observation": observation,
                    "action": action,
                    "episode_id": episode_id,
                    "step_index": step_index,
                })

    return dict(episodes)


def _decode_jpeg(jpeg_bytes: bytes, target_size: tuple[int, int] = (224, 224)) -> Any:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _write_training_shard(
    shard_path: Path,
    episodes: list[tuple[Any, Any]],
    episode_ids: list[str] | None = None,
) -> None:
    """Write a training shard from a list of (frames, actions) episodes. Used by tests."""
    with tarfile.open(shard_path, "w") as tar:
        for i, (frames, actions) in enumerate(episodes):
            key = episode_ids[i] if episode_ids else f"episode_{i:06d}"
            buf = io.BytesIO()
            np.savez_compressed(buf, frames=frames, actions=actions)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.npz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
