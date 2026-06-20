"""Integration test: mock MineStudio env -> collector -> shards -> validator."""

import json
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from wally.collector.buffer import TrajectoryBuffer
from wally.collector.config import CollectorConfig
from wally.collector.recorder import TransitionRecorder
from wally.exporter.metadata import generate_manifest
from wally.exporter.shard_writer import ShardWriter
from wally.validator.inspector import inspect_shard, validate_shard


class MockMineStudioEnv:
    """Mimics MineStudio's env interface without the real dependency."""

    def __init__(self, obs_shape=(480, 640, 3)):
        self._obs_shape = obs_shape
        self._step_count = 0
        self._episode_steps = 0
        self._max_steps = 0
        self._new_episode()

    def _new_episode(self):
        self._step_count = 0
        self._max_steps = random.randint(5, 20)
        self._episode_steps = 0

    def reset(self) -> np.ndarray:
        self._new_episode()
        return np.random.randint(0, 255, self._obs_shape, dtype=np.uint8)

    def step(self, action: dict) -> tuple:
        self._step_count += 1
        self._episode_steps += 1
        obs = np.random.randint(0, 255, self._obs_shape, dtype=np.uint8)
        reward = random.random()
        done = self._episode_steps >= self._max_steps
        return obs, reward, done, {}

    @property
    def action_space(self):
        class _ActionSpace:
            def sample(self):
                return {"forward": 1, "jump": 0, "attack": 0}
        return _ActionSpace()


def _collect_transitions(
    config: CollectorConfig, env: MockMineStudioEnv, target: int
) -> list[dict]:
    """Run episodes until at least *target* transitions are collected."""
    recorder = TransitionRecorder(config)
    collected: list[dict] = []

    buffer = TrajectoryBuffer(
        max_size=config.buffer_size,
        flush_callback=lambda batch: collected.extend(batch),
    )

    while len(collected) < target:
        env.reset()
        recorder.start_episode()
        done = False
        while not done:
            action = {"forward": 1, "jump": 0, "attack": 0}
            transition = recorder.record_step(env, action)
            buffer.add(transition)
            done = transition["done"]

    buffer.shutdown()
    return collected


class TestFullPipeline:
    """End-to-end: collect -> export -> validate -> inspect -> CLI."""

    def test_pipeline(self, tmp_path):
        random.seed(0)
        np.random.seed(0)

        config = CollectorConfig(
            frame_skip=1,
            resize=(224, 224),
            jpeg_quality=85,
            buffer_size=50,
            output_dir=str(tmp_path / "raw"),
        )
        env = MockMineStudioEnv()

        # --- collect ~200 transitions ---
        transitions = _collect_transitions(config, env, target=200)
        assert len(transitions) >= 200
        assert "observation" in transitions[0]
        assert "episode_id" in transitions[0]

        # --- export to shards ---
        shard_dir = tmp_path / "shards"
        writer = ShardWriter(
            output_dir=shard_dir, shard_size=100, jpeg_quality=85
        )
        shard_infos = writer.write_shards(transitions)
        assert len(shard_infos) >= 2

        for path_str, count in shard_infos:
            assert Path(path_str).exists()
            assert count > 0

        # --- generate manifest ---
        episode_ids = {t["episode_id"] for t in transitions}
        manifest = generate_manifest(
            shard_infos, output_dir=shard_dir, episode_ids=episode_ids
        )
        assert manifest["total_transitions"] == len(transitions)
        assert manifest["total_episodes"] == len(episode_ids)
        assert manifest["shard_count"] == len(shard_infos)

        # --- validate each shard ---
        total_validated = 0
        for path_str, count in shard_infos:
            result = validate_shard(path_str)
            assert result["valid"], f"Shard invalid: {result['errors']}"
            total_validated += count
        assert total_validated == len(transitions)

        # --- inspect each shard ---
        total_inspected = 0
        all_episodes: set[str] = set()
        for path_str, count in shard_infos:
            info = inspect_shard(path_str)
            assert info["transition_count"] == count
            total_inspected += info["transition_count"]
            with tarfile.open(path_str, "r") as tar:
                for m in tar.getmembers():
                    if m.name.endswith(".json"):
                        f = tar.extractfile(m)
                        if f:
                            data = json.loads(f.read())
                            all_episodes.add(data["episode_id"])
        assert total_inspected == len(transitions)
        assert len(all_episodes) == len(episode_ids)

        # --- CLI validate (subprocess, src/ on PYTHONPATH) ---
        result = subprocess.run(
            [sys.executable, "-m", "validator.cli", "validate", str(shard_dir)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent / "src"),
        )
        assert result.returncode == 0, (
            f"CLI validate failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # --- manifest on disk matches ---
        manifest_path = shard_dir / "manifest.json"
        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text())
        assert on_disk["total_transitions"] == len(transitions)
        assert on_disk["total_episodes"] == len(episode_ids)
        assert on_disk["shard_count"] == len(shard_infos)
