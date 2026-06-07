"""Tests for the streaming RawShardWriter used by TrajectoryCollector."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from src.collector.raw_shard_writer import RawShardWriter


def _make_transition(episode_id: str, step_index: int, seed: int = 42) -> dict:
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    return {
        "episode_id": episode_id,
        "step_index": step_index,
        "observation": img,
        "action": {"forward": 1, "attack": 0},
        "timestamp": 1000.0 + step_index,
        "frame_skip": 2,
        "seed": seed,
    }


def _read_tar_members(tar_path: str | Path) -> dict[str, bytes]:
    with tarfile.open(tar_path, "r") as tar:
        out: dict[str, bytes] = {}
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f is not None:
                    out[member.name] = f.read()
    return out


class TestRawShardWriterContextManager:
    def test_creates_output_dir_on_enter(self, tmp_path):
        target = tmp_path / "nested" / "shards"
        with RawShardWriter(output_dir=target, shard_size=100):
            pass
        assert target.is_dir()

    def test_closes_current_shard_on_exit(self, tmp_path):
        writer = RawShardWriter(output_dir=tmp_path, shard_size=100)
        with writer:
            writer.add(_make_transition("ep01", 0))
        tar_files = list(tmp_path.glob("*.tar"))
        assert len(tar_files) == 1


class TestRawShardWriterAddsTransitions:
    def test_single_transition_written_to_shard(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=100) as writer:
            writer.add(_make_transition("ep01", 0))

        tar_files = list(tmp_path.glob("*.tar"))
        assert len(tar_files) == 1
        members = _read_tar_members(tar_files[0])
        assert "ep01_000000.jpg" in members
        assert "ep01_000000.json" in members

    def test_multiple_transitions_same_episode(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=100) as writer:
            for i in range(5):
                writer.add(_make_transition("ep01", i))

        tar_files = list(tmp_path.glob("*.tar"))
        assert len(tar_files) == 1
        members = _read_tar_members(tar_files[0])

        for i in range(5):
            assert f"ep01_{i:06d}.jpg" in members
            assert f"ep01_{i:06d}.json" in members

    def test_jpeg_is_decodable(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=100) as writer:
            writer.add(_make_transition("ep01", 0))

        members = _read_tar_members(list(tmp_path.glob("*.tar"))[0])
        img = Image.open(__import__("io").BytesIO(members["ep01_000000.jpg"]))
        img.load()
        assert img.size[0] > 0 and img.size[1] > 0

    def test_json_contains_action_and_metadata(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=100) as writer:
            writer.add(_make_transition("ep01", 0, seed=99))

        members = _read_tar_members(list(tmp_path.glob("*.tar"))[0])
        data = json.loads(members["ep01_000000.json"].decode("utf-8"))

        assert data["episode_id"] == "ep01"
        assert data["step_index"] == 0
        assert data["seed"] == 99
        assert data["frame_skip"] == 2
        assert isinstance(data["action"], dict)


class TestRawShardWriterShardRotation:
    def test_closes_shard_when_size_exceeded(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=3) as writer:
            for i in range(3):
                writer.add(_make_transition("ep01", i))
            writer.add(_make_transition("ep02", 0))

        tar_files = sorted(tmp_path.glob("*.tar"))
        assert len(tar_files) == 2

    def test_does_not_split_episode_across_shards(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=3) as writer:
            for i in range(5):
                writer.add(_make_transition("ep01", i))
            writer.add(_make_transition("ep02", 0))

        for tar_path in sorted(tmp_path.glob("*.tar")):
            members = _read_tar_members(tar_path)
            episode_ids = set()
            for name, data in members.items():
                if name.endswith(".json"):
                    payload = json.loads(data.decode("utf-8"))
                    episode_ids.add(payload["episode_id"])
            assert len(episode_ids) == 1


class TestRawShardWriterEdgeCases:
    def test_empty_writer_creates_no_shards(self, tmp_path):
        with RawShardWriter(output_dir=tmp_path, shard_size=100):
            pass
        assert list(tmp_path.glob("*.tar")) == []

    def test_observation_as_ndarray_works(self, tmp_path):
        transition = _make_transition("ep01", 0)
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transition["observation"] = arr

        with RawShardWriter(output_dir=tmp_path, shard_size=100) as writer:
            writer.add(transition)

        members = _read_tar_members(list(tmp_path.glob("*.tar"))[0])
        assert "ep01_000000.jpg" in members
