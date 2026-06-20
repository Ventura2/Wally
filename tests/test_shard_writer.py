import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from wally.exporter.shard_writer import ShardWriter


def _make_transitions(
    episode_id: str, count: int, seed: int = 42
) -> list[dict]:
    transitions = []
    for i in range(count):
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        transitions.append(
            {
                "episode_id": episode_id,
                "step_index": i,
                "observation": img,
                "action": {"forward": 1, "attack": 0},
                "timestamp": 1000.0 + i,
                "frame_skip": 2,
                "seed": seed,
            }
        )
    return transitions


class TestShardWriterCreatesTar:
    def test_write_shards_creates_tar_files(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 5)
        results = writer.write_shards(transitions)
        assert len(results) == 1
        shard_path = Path(results[0][0])
        assert shard_path.exists()
        assert shard_path.suffix == ".tar"

    def test_write_shards_multiple_shards(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=3)
        transitions = _make_transitions("ep01", 3) + _make_transitions("ep02", 3)
        results = writer.write_shards(transitions)
        assert len(results) == 2
        for path_str, count in results:
            assert Path(path_str).exists()


class TestShardWriterTarContents:
    def test_tar_contains_jpg_json_pairs(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 4)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        with tarfile.open(shard_path, "r") as tar:
            names = [m.name for m in tar.getmembers()]

        for t in transitions:
            key = f"ep01_{t['step_index']:06d}"
            assert f"{key}.jpg" in names
            assert f"{key}.json" in names

    def test_tar_member_count(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 5)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()

        assert len(members) == 10  # 5 jpg + 5 json


class TestShardWriterKeyNaming:
    def test_key_follows_pattern(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("my_episode", 3)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        with tarfile.open(shard_path, "r") as tar:
            names = sorted(m.name for m in tar.getmembers())

        assert "my_episode_000000.jpg" in names
        assert "my_episode_000000.json" in names
        assert "my_episode_000001.jpg" in names
        assert "my_episode_000001.json" in names
        assert "my_episode_000002.jpg" in names
        assert "my_episode_000002.json" in names


class TestShardWriterJpegDecodability:
    def test_jpg_files_are_decodable(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 3)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".jpg"):
                    f = tar.extractfile(member)
                    assert f is not None
                    img = Image.open(f)
                    img.load()
                    assert img.size[0] > 0 and img.size[1] > 0


class TestShardWriterJsonParseability:
    def test_json_files_parseable_with_expected_keys(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 3)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        expected_keys = {"action", "timestamp", "episode_id", "step_index", "frame_skip", "seed"}

        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    assert f is not None
                    data = json.loads(f.read().decode("utf-8"))
                    assert expected_keys.issubset(data.keys())
                    assert data["episode_id"] == "ep01"
                    assert isinstance(data["action"], dict)

    def test_json_values_match_input(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("ep01", 1, seed=99)
        results = writer.write_shards(transitions)
        shard_path = results[0][0]

        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    data = json.loads(f.read().decode("utf-8"))
                    assert data["step_index"] == 0
                    assert data["timestamp"] == 1000.0
                    assert data["frame_skip"] == 2
                    assert data["seed"] == 99


class TestShardWriterEpisodeBoundary:
    def test_episodes_not_split_across_shards(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=5)
        ep1 = _make_transitions("ep01", 4)
        ep2 = _make_transitions("ep02", 4)
        transitions = ep1 + ep2

        results = writer.write_shards(transitions)

        episode_ids_per_shard = []
        for path_str, count in results:
            ids = set()
            with tarfile.open(path_str, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".json"):
                        f = tar.extractfile(member)
                        data = json.loads(f.read().decode("utf-8"))
                        ids.add(data["episode_id"])
            episode_ids_per_shard.append(ids)

        for ids in episode_ids_per_shard:
            assert len(ids) == 1  # each shard has exactly one episode


class TestShardWriterShardSize:
    def test_shard_size_targeting(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=10)
        transitions = []
        for i in range(3):
            transitions.extend(_make_transitions(f"ep{i:02d}", 4))

        results = writer.write_shards(transitions)

        for path_str, count in results:
            assert count <= 10

    def test_single_large_episode_single_shard(self, tmp_path):
        writer = ShardWriter(output_dir=tmp_path, shard_size=100)
        transitions = _make_transitions("big_ep", 50)
        results = writer.write_shards(transitions)
        assert len(results) == 1
        assert results[0][1] == 50
