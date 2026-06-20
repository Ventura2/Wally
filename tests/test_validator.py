import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from wally.validator.inspector import inspect_shard, validate_shard


def _make_tar_shard(
    tmp_path: Path,
    name: str,
    transitions: list[dict],
    corrupt_jpg_keys: set[str] | None = None,
    omit_json_keys: set[str] | None = None,
) -> Path:
    shard_path = tmp_path / name
    corrupt_jpg_keys = corrupt_jpg_keys or set()
    omit_json_keys = omit_json_keys or set()

    with tarfile.open(shard_path, "w") as tar:
        for t in transitions:
            key = f"{t['episode_id']}_{t['step_index']:06d}"

            if key in corrupt_jpg_keys:
                jpg_bytes = b"not a real jpeg"
            else:
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                jpg_bytes = buf.getvalue()

            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(jpg_bytes)
            tar.addfile(info, io.BytesIO(jpg_bytes))

            if key not in omit_json_keys:
                sidecar = {
                    "action": t.get("action", {"forward": 1}),
                    "timestamp": t.get("timestamp", 1000.0),
                    "episode_id": t["episode_id"],
                    "step_index": t["step_index"],
                    "frame_skip": t.get("frame_skip", 1),
                    "seed": t.get("seed", 42),
                }
                json_bytes = json.dumps(sidecar).encode("utf-8")
                info = tarfile.TarInfo(name=f"{key}.json")
                info.size = len(json_bytes)
                tar.addfile(info, io.BytesIO(json_bytes))

    return shard_path


def _make_transitions(count: int, episode_id: str = "ep01") -> list[dict]:
    return [
        {
            "episode_id": episode_id,
            "step_index": i,
            "action": {"forward": 1, "attack": 0},
            "timestamp": 1000.0 + i,
            "frame_skip": 2,
            "seed": 42,
        }
        for i in range(count)
    ]


class TestValidateShardPasses:
    def test_valid_shard_passes(self, tmp_path):
        transitions = _make_transitions(5)
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        result = validate_shard(shard_path)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_valid_multi_episode_shard_passes(self, tmp_path):
        transitions = _make_transitions(3, "ep01") + _make_transitions(3, "ep02")
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        result = validate_shard(shard_path)
        assert result["valid"] is True


class TestValidateShardMissingSidecar:
    def test_missing_json_detected(self, tmp_path):
        transitions = _make_transitions(3)
        shard_path = _make_tar_shard(
            tmp_path, "shard.tar", transitions, omit_json_keys={"ep01_000001"}
        )
        result = validate_shard(shard_path)
        assert result["valid"] is False
        assert any(".json" in e for e in result["errors"])

    def test_multiple_missing_json_reported(self, tmp_path):
        transitions = _make_transitions(4)
        shard_path = _make_tar_shard(
            tmp_path,
            "shard.tar",
            transitions,
            omit_json_keys={"ep01_000000", "ep01_000002"},
        )
        result = validate_shard(shard_path)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1


class TestValidateShardCorruptJpg:
    def test_corrupt_jpg_detected(self, tmp_path):
        transitions = _make_transitions(3)
        shard_path = _make_tar_shard(
            tmp_path, "shard.tar", transitions, corrupt_jpg_keys={"ep01_000001"}
        )
        result = validate_shard(shard_path)
        assert result["valid"] is False
        assert any("corrupt" in e.lower() or "JPEG" in e for e in result["errors"])

    def test_all_corrupt_jpgs_detected(self, tmp_path):
        transitions = _make_transitions(2)
        shard_path = _make_tar_shard(
            tmp_path,
            "shard.tar",
            transitions,
            corrupt_jpg_keys={"ep01_000000", "ep01_000001"},
        )
        result = validate_shard(shard_path)
        assert result["valid"] is False


class TestInspectShard:
    def test_transition_count(self, tmp_path):
        transitions = _make_transitions(5)
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        info = inspect_shard(shard_path)
        assert info["transition_count"] == 5

    def test_episode_count(self, tmp_path):
        transitions = _make_transitions(3, "ep01") + _make_transitions(2, "ep02")
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        info = inspect_shard(shard_path)
        assert info["episode_count"] == 2

    def test_action_keys(self, tmp_path):
        transitions = [
            {
                "episode_id": "ep01",
                "step_index": i,
                "action": {"forward": 1, "jump": 0, "attack": 0},
                "timestamp": 1000.0 + i,
                "frame_skip": 1,
                "seed": 42,
            }
            for i in range(3)
        ]
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        info = inspect_shard(shard_path)
        assert set(info["action_keys"]) == {"attack", "forward", "jump"}

    def test_timestamp_range(self, tmp_path):
        transitions = _make_transitions(3, "ep01")
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        info = inspect_shard(shard_path)
        ts_min, ts_max = info["timestamp_range"]
        assert ts_min == 1000.0
        assert ts_max == 1002.0

    def test_observation_shape_not_none(self, tmp_path):
        transitions = _make_transitions(1)
        shard_path = _make_tar_shard(tmp_path, "shard.tar", transitions)
        info = inspect_shard(shard_path)
        assert info["observation_shape"] is not None
        assert len(info["observation_shape"]) == 2  # (width, height)
