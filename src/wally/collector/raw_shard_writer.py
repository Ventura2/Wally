"""Streaming writer for raw trajectory shards.

Writes transitions to .tar shards incrementally, one transition at a time.
Used by TrajectoryCollector to persist data as it is produced, so the
collector never needs to hold all transitions in memory.

Each sample in a shard contains {key}.jpg (JPEG observation) and
{key}.json (action + metadata), matching the format expected by
``wally-convert`` and the converter pipeline.
"""

from __future__ import annotations

import io
import json
import numpy as np
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image


class RawShardWriter:
    """Streaming writer for raw .tar shards (one transition at a time).

    Use as a context manager. Transitions are added via ``add()``; the writer
    rotates to a new shard when the current shard's transition count would
    exceed ``shard_size``. Episodes are never split across shards.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 1000,
        jpeg_quality: int = 85,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.jpeg_quality = jpeg_quality
        self._shard_idx = 0
        self._current_count = 0
        self._current_episode_id: str | None = None
        self._tar: tarfile.TarFile | None = None
        self._tar_path: Path | None = None

    def __enter__(self) -> "RawShardWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def add(self, transition: dict[str, Any]) -> None:
        episode_id = transition["episode_id"]
        step_index = transition["step_index"]
        key = f"{episode_id}_{step_index:06d}"

        if self._needs_new_shard(episode_id):
            self._open_new_shard(episode_id)

        jpg_bytes = self._encode_observation(transition["observation"])
        json_bytes = self._encode_metadata(transition)

        self._add_member(f"{key}.jpg", jpg_bytes)
        self._add_member(f"{key}.json", json_bytes)

        self._current_count += 1

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            self._tar_path = None

    def _needs_new_shard(self, episode_id: str) -> bool:
        if self._tar is None:
            return True
        if (
            self._current_episode_id is not None
            and self._current_episode_id != episode_id
        ):
            return True
        if self._current_count >= self.shard_size:
            return True
        return False

    def _open_new_shard(self, first_episode_id: str) -> None:
        if self._tar is not None:
            self._tar.close()

        self._tar_path = self.output_dir / f"shard_{self._shard_idx:06d}.tar"
        self._tar = tarfile.open(self._tar_path, "w")
        self._shard_idx += 1
        self._current_count = 0
        self._current_episode_id = first_episode_id

    def _encode_observation(self, observation: Any) -> bytes:
        if not isinstance(observation, Image.Image):
            observation = Image.fromarray(observation)
        buf = io.BytesIO()
        observation.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def _encode_metadata(self, transition: dict[str, Any]) -> bytes:
        sidecar = {
            "action": transition["action"],
            "timestamp": transition["timestamp"],
            "episode_id": transition["episode_id"],
            "step_index": transition["step_index"],
            "frame_skip": transition.get("frame_skip", 1),
            "seed": transition.get("seed"),
        }
        return json.dumps(sidecar, default=_numpy_default).encode("utf-8")


    def _add_member(self, name: str, data: bytes) -> None:
        if self._tar is None:
            raise RuntimeError("Writer is not open")
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))


def _numpy_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
