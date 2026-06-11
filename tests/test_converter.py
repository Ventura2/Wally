from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from wally.data.converter import (
    _decode_jpeg,
    _encode_action,
    _load_raw_shards,
    _write_training_shard,
    convert_shards,
)


def _make_jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_raw_shard(
    shard_path: Path,
    episodes: dict[str, list[dict]],
) -> None:
    with tarfile.open(shard_path, "w") as tar:
        for episode_id, transitions in episodes.items():
            for t in transitions:
                step = t["step_index"]
                key = f"{episode_id}_{step:06d}"

                jpg_bytes = t.get("jpg_bytes", _make_jpeg_bytes())
                info = tarfile.TarInfo(name=f"{key}.jpg")
                info.size = len(jpg_bytes)
                tar.addfile(info, io.BytesIO(jpg_bytes))

                meta = {
                    "action": t["action"],
                    "timestamp": t.get("timestamp", 0.0),
                    "episode_id": episode_id,
                    "step_index": step,
                    "frame_skip": t.get("frame_skip", 1),
                    "seed": t.get("seed"),
                }
                json_bytes = json.dumps(meta).encode("utf-8")
                info = tarfile.TarInfo(name=f"{key}.json")
                info.size = len(json_bytes)
                tar.addfile(info, io.BytesIO(json_bytes))


def _create_mock_raw_shards(
    raw_dir: Path,
    num_episodes: int = 10,
    steps_per_episode: int = 20,
    action_keys: list[str] | None = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if action_keys is None:
        action_keys = ["forward", "jump", "attack"]

    jpg_bytes = _make_jpeg_bytes(64, 64)
    all_episodes: dict[str, list[dict]] = {}
    for ep in range(num_episodes):
        episode_id = f"ep_{ep:03d}"
        transitions = []
        for step in range(steps_per_episode):
            action = {k: float(np.random.randint(0, 2)) for k in action_keys}
            transitions.append({
                "step_index": step,
                "action": action,
                "jpg_bytes": jpg_bytes,
            })
        all_episodes[episode_id] = transitions

    mid = num_episodes // 2
    eps_items = list(all_episodes.items())
    _make_raw_shard(raw_dir / "shard_000000.tar", dict(eps_items[:mid]))
    _make_raw_shard(raw_dir / "shard_000001.tar", dict(eps_items[mid:]))


class TestEncodeAction:
    def test_encode_action_basic(self):
        action = {"forward": 1.0, "jump": 0.0, "attack": 1.0}
        schema = ["forward", "jump", "attack"]
        vec = _encode_action(action, schema)
        assert vec.shape == (3,)
        assert vec.dtype == np.float32
        np.testing.assert_array_equal(vec, [1.0, 0.0, 1.0])

    def test_encode_action_missing_keys(self):
        action = {"forward": 1.0}
        schema = ["forward", "jump", "attack"]
        vec = _encode_action(action, schema)
        assert vec.shape == (3,)
        np.testing.assert_array_equal(vec, [1.0, 0.0, 0.0])

    def test_encode_action_extra_keys(self):
        action = {"forward": 1.0, "jump": 0.5, "sneak": 1.0}
        schema = ["forward", "jump"]
        vec = _encode_action(action, schema)
        assert vec.shape == (2,)
        np.testing.assert_array_equal(vec, [1.0, 0.5])

    def test_encode_action_empty(self):
        action: dict = {}
        schema = ["forward", "jump"]
        vec = _encode_action(action, schema)
        np.testing.assert_array_equal(vec, [0.0, 0.0])


class TestDecodeJpeg:
    def test_decode_jpeg(self):
        jpg_bytes = _make_jpeg_bytes(32, 48)
        arr = _decode_jpeg(jpg_bytes)
        assert arr.shape == (224, 224, 3), "should resize to 224x224"
        assert arr.dtype == np.uint8

    def test_decode_jpeg_rgb(self):
        jpg_bytes = _make_jpeg_bytes(64, 64)
        arr = _decode_jpeg(jpg_bytes)
        assert arr.shape[2] == 3


class TestLoadRawShards:
    def test_load_raw_shards(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _create_mock_raw_shards(raw_dir, num_episodes=4, steps_per_episode=5)
        episodes = _load_raw_shards(raw_dir)
        assert len(episodes) == 4
        for ep_id, transitions in episodes.items():
            assert len(transitions) == 5
            for t in transitions:
                assert "observation" in t
                assert "action" in t
                assert isinstance(t["observation"], np.ndarray)
                assert t["observation"].ndim == 3
                assert t["observation"].shape == (224, 224, 3)

    def test_load_raw_shards_empty_dir(self, tmp_path):
        raw_dir = tmp_path / "empty"
        raw_dir.mkdir()
        episodes = _load_raw_shards(raw_dir)
        assert episodes == {}


class TestWriteTrainingShard:
    def test_write_training_shard(self, tmp_path):
        frames = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randn(10, 3).astype(np.float32)
        shard_path = tmp_path / "test.tar"
        _write_training_shard(shard_path, [(frames, actions)], ["ep_000"])

        with tarfile.open(shard_path, "r") as tar:
            names = tar.getnames()
            assert "ep_000.npz" in names
            f = tar.extractfile("ep_000.npz")
            assert f is not None
            data = np.load(io.BytesIO(f.read()))
            np.testing.assert_array_equal(data["frames"], frames)
            np.testing.assert_allclose(data["actions"], actions)


class TestConvertShards:
    def test_convert_shards_basic(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _create_mock_raw_shards(raw_dir, num_episodes=6, steps_per_episode=10)

        output_dir = tmp_path / "training"
        schema = ["forward", "jump", "attack"]
        stats = convert_shards(raw_dir, output_dir, schema, episodes_per_shard=3)

        assert stats["episode_count"] == 6
        assert stats["shard_count"] == 2
        assert stats["skipped_episodes"] == 0
        assert stats["total_transitions"] == 60

        tar_files = sorted(output_dir.glob("*.tar"))
        assert len(tar_files) == 2

        with tarfile.open(tar_files[0], "r") as tar:
            npz_names = [n for n in tar.getnames() if n.endswith(".npz")]
            assert len(npz_names) == 3
            for name in npz_names:
                f = tar.extractfile(name)
                assert f is not None
                data = np.load(io.BytesIO(f.read()))
                assert "frames" in data
                assert "actions" in data
                assert data["frames"].shape == (10, 224, 224, 3)
                assert data["actions"].shape == (10, 3)
                assert data["frames"].dtype == np.uint8
                assert data["actions"].dtype == np.float32

    def test_convert_shards_empty(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "training"
        stats = convert_shards(raw_dir, output_dir, ["forward"])
        assert stats["episode_count"] == 0
        assert stats["shard_count"] == 0

    def test_convert_shards_single_shard(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _create_mock_raw_shards(raw_dir, num_episodes=3, steps_per_episode=5)

        output_dir = tmp_path / "training"
        schema = ["forward", "jump", "attack"]
        stats = convert_shards(raw_dir, output_dir, schema, episodes_per_shard=50)

        assert stats["episode_count"] == 3
        assert stats["shard_count"] == 1

    def test_convert_shards_preserves_order(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _create_mock_raw_shards(raw_dir, num_episodes=4, steps_per_episode=8)

        output_dir = tmp_path / "training"
        schema = ["forward", "jump", "attack"]
        convert_shards(raw_dir, output_dir, schema, episodes_per_shard=50)

        tar_files = sorted(output_dir.glob("*.tar"))
        with tarfile.open(tar_files[0], "r") as tar:
            npz_names = sorted(n for n in tar.getnames() if n.endswith(".npz"))
            assert len(npz_names) == 4
            for name in npz_names:
                f = tar.extractfile(name)
                assert f is not None
                data = np.load(io.BytesIO(f.read()))
                assert data["frames"].shape[0] == 8

    def test_convert_shards_training_format_compatible(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _create_mock_raw_shards(
            raw_dir, num_episodes=4, steps_per_episode=20,
            action_keys=["forward", "jump", "attack"],
        )

        output_dir = tmp_path / "training"
        schema = ["forward", "jump", "attack"]
        stats = convert_shards(raw_dir, output_dir, schema, episodes_per_shard=4)
        assert stats["episode_count"] == 4

        from wally.data.dataset import (
            decode_sample,
            preprocess_frames,
            sample_subsequence,
        )

        tar_files = sorted(output_dir.glob("*.tar"))
        assert len(tar_files) == 1

        samples_loaded = 0
        with tarfile.open(tar_files[0], "r") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".npz"):
                    continue
                f = tar.extractfile(member)
                assert f is not None
                npz_bytes = f.read()

                sample = {member.name: npz_bytes}
                decoded = decode_sample(sample)
                assert "frames" in decoded
                assert "actions" in decoded

                frames = preprocess_frames(decoded["frames"])
                actions = decoded["actions"]
                result = sample_subsequence(
                    frames, actions, seq_length=8, skip_short=True,
                )
                assert result is not None
                assert result["frames"].shape == (8, 3, 224, 224), f"got {result['frames'].shape}"
                assert result["actions"].shape == (8, 3)
                samples_loaded += 1

        assert samples_loaded == 4
