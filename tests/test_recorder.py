from unittest.mock import MagicMock

import numpy as np
import pytest
from src.collector.config import CollectorConfig
from src.collector.recorder import TransitionRecorder


def _make_recorder(frame_skip=4):
    config = CollectorConfig(frame_skip=frame_skip)
    return TransitionRecorder(config)


def _mock_env(reward_per_step=1.0, done_after=None):
    env = MagicMock()
    step_count = {"n": 0}

    def step_fn(action):
        step_count["n"] += 1
        done = step_count["n"] == done_after if done_after else False
        obs = np.zeros((224, 224, 3), dtype=np.uint8)
        return obs, reward_per_step, done, {}

    env.step = MagicMock(side_effect=step_fn)
    return env, step_count


class TestFrameSkip:
    def test_executes_frame_skip_steps(self):
        recorder = _make_recorder(frame_skip=4)
        recorder.start_episode()
        env, step_count = _mock_env()
        action = {"forward": 1}
        recorder.record_step(env, action)
        assert step_count["n"] == 4

    def test_frame_skip_1(self):
        recorder = _make_recorder(frame_skip=1)
        recorder.start_episode()
        env, step_count = _mock_env()
        recorder.record_step(env, {})
        assert step_count["n"] == 1

    def test_accumulated_reward(self):
        recorder = _make_recorder(frame_skip=4)
        recorder.start_episode()
        env, _ = _mock_env(reward_per_step=2.5)
        transition = recorder.record_step(env, {})
        assert transition["reward"] == pytest.approx(10.0)

    def test_stores_final_observation(self):
        call_obs = []

        recorder = _make_recorder(frame_skip=3)
        recorder.start_episode()
        env = MagicMock()

        def step_fn(action):
            obs = np.full((224, 224, 3), len(call_obs), dtype=np.uint8)
            call_obs.append(obs)
            return obs, 0.0, False, {}

        env.step = MagicMock(side_effect=step_fn)
        transition = recorder.record_step(env, {})
        np.testing.assert_array_equal(transition["observation"], call_obs[-1])

    def test_episode_metadata(self):
        recorder = _make_recorder()
        episode_id = recorder.start_episode(seed=42)
        env, _ = _mock_env()
        transition = recorder.record_step(env, {})
        assert transition["episode_id"] == episode_id
        assert transition["step_index"] == 0
        assert transition["seed"] == 42

    def test_step_index_increments(self):
        recorder = _make_recorder()
        recorder.start_episode()
        env, _ = _mock_env()
        t1 = recorder.record_step(env, {})
        t2 = recorder.record_step(env, {})
        assert t1["step_index"] == 0
        assert t2["step_index"] == 1

    def test_done_stops_early(self):
        recorder = _make_recorder(frame_skip=4)
        recorder.start_episode()
        env, step_count = _mock_env(done_after=2)
        transition = recorder.record_step(env, {})
        assert step_count["n"] == 2
        assert transition["done"] is True

    def test_done_clears_episode(self):
        recorder = _make_recorder(frame_skip=2)
        recorder.start_episode()
        env, _ = _mock_env(done_after=1)
        recorder.record_step(env, {})
        assert recorder.episode_id is None

    def test_record_step_without_episode_raises(self):
        recorder = _make_recorder()
        env = MagicMock()
        with pytest.raises(RuntimeError, match="No active episode"):
            recorder.record_step(env, {})

    def test_reward_accumulated_with_early_done(self):
        recorder = _make_recorder(frame_skip=4)
        recorder.start_episode()
        env, _ = _mock_env(reward_per_step=3.0, done_after=2)
        transition = recorder.record_step(env, {})
        assert transition["reward"] == pytest.approx(6.0)
