"""WebDataset shard writer for Minecraft trajectory transitions."""

import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


class ShardWriter:
    """Writes transitions to .tar shards in WebDataset format.

    Groups transitions by episode to preserve episode boundaries within shards.
    Each sample in a shard contains {key}.jpg (JPEG observation) and
    {key}.json (action dict + metadata).
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_shards(
        self, transitions: list[dict[str, Any]]
    ) -> list[tuple[str, int]]:
        """Write transitions to sharded .tar files, respecting episode boundaries.

        Args:
            transitions: List of transition dicts, each containing at minimum
                'episode_id', 'step_index', 'observation', 'action', 'timestamp',
                'frame_skip', 'seed'.

        Returns:
            List of (shard_path, transition_count) tuples for each written shard.
        """
        episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in transitions:
            episodes[t["episode_id"]].append(t)

        shards: list[list[dict[str, Any]]] = []
        current_shard: list[dict[str, Any]] = []
        current_count = 0

        for episode_transitions in episodes.values():
            if (
                current_count + len(episode_transitions) > self.shard_size
                and current_shard
            ):
                shards.append(current_shard)
                current_shard = []
                current_count = 0
            current_shard.extend(episode_transitions)
            current_count += len(episode_transitions)

        if current_shard:
            shards.append(current_shard)

        results: list[tuple[str, int]] = []
        for shard_idx, shard_transitions in enumerate(shards):
            shard_path = self.output_dir / f"shard_{shard_idx:06d}.tar"
            self._write_single_shard(shard_path, shard_transitions)
            results.append((str(shard_path), len(shard_transitions)))

        return results

    def _write_single_shard(
        self, shard_path: Path, transitions: list[dict[str, Any]]
    ) -> None:
        """Write a single .tar shard containing all given transitions."""
        with tarfile.open(shard_path, "w") as tar:
            for idx, transition in enumerate(transitions):
                key, jpg_bytes, json_bytes = self._encode_transition(
                    transition, idx
                )
                self._add_to_tar(tar, f"{key}.jpg", jpg_bytes)
                self._add_to_tar(tar, f"{key}.json", json_bytes)

    def _encode_transition(
        self, transition: dict[str, Any], index: int
    ) -> tuple[str, bytes, bytes]:
        """Encode a transition into a shard key, JPEG bytes, and JSON bytes.

        Args:
            transition: Transition dict with observation, action, and metadata.
            index: Position index within the shard (unused for key generation).

        Returns:
            Tuple of (key, jpeg_bytes, json_bytes).
        """
        episode_id = transition["episode_id"]
        step_index = transition["step_index"]
        key = f"{episode_id}_{step_index:06d}"

        observation = transition["observation"]
        if not isinstance(observation, Image.Image):
            observation = Image.fromarray(observation)
        buf = io.BytesIO()
        observation.save(buf, format="JPEG", quality=self.jpeg_quality)
        jpg_bytes = buf.getvalue()

        sidecar = {
            "action": transition["action"],
            "timestamp": transition["timestamp"],
            "episode_id": episode_id,
            "step_index": step_index,
            "frame_skip": transition.get("frame_skip", 1),
            "seed": transition.get("seed"),
        }
        json_bytes = json.dumps(sidecar).encode("utf-8")

        return key, jpg_bytes, json_bytes

    @staticmethod
    def _add_to_tar(
        tar: tarfile.TarFile, name: str, data: bytes
    ) -> None:
        """Add raw bytes as a file entry to an open tar archive."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
