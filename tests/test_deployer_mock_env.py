"""Tests for ``MockServerEnv`` (no Minecraft server required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wally.deployer.config import DeployConfig
from wally.deployer.env import MockServerEnv


def _make_env() -> MockServerEnv:
    config = DeployConfig.default()
    return MockServerEnv(config)


class TestMockServerEnvReset:
    def test_reset_returns_tensor(self) -> None:
        env = _make_env()
        obs = env.reset()
        assert isinstance(obs, torch.Tensor)

    def test_reset_shape(self) -> None:
        env = _make_env()
        obs = env.reset()
        assert obs.shape == (3, 224, 224)

    def test_reset_dtype(self) -> None:
        env = _make_env()
        obs = env.reset()
        assert obs.dtype == torch.float32

    def test_reset_deterministic(self) -> None:
        env1 = _make_env()
        env2 = _make_env()
        a = env1.reset().numpy()
        b = env2.reset().numpy()
        np.testing.assert_array_equal(a, b)

    def test_reset_raises_when_closed(self) -> None:
        env = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.reset()


class TestMockServerEnvStep:
    def test_step_returns_tuple_of_four(self) -> None:
        env = _make_env()
        env.reset()
        result = env.step(torch.zeros(25))
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_step_obs_is_tensor_correct_shape(self) -> None:
        env = _make_env()
        env.reset()
        obs, _r, _d, _i = env.step(torch.zeros(25))
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (3, 224, 224)

    def test_step_reward_is_zero(self) -> None:
        env = _make_env()
        env.reset()
        _obs, reward, _d, _i = env.step(torch.zeros(25))
        assert reward == 0.0

    def test_step_done_is_false(self) -> None:
        env = _make_env()
        env.reset()
        _obs, _r, done, _i = env.step(torch.zeros(25))
        assert done is False

    def test_step_info_has_packets(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(25, dtype=np.float32)
        action[0] = 1.0  # forward
        _obs, _r, _d, info = env.step(action)
        assert "packets" in info
        assert isinstance(info["packets"], list)

    def test_step_info_has_step_counter(self) -> None:
        env = _make_env()
        env.reset()
        _obs, _r, _d, info1 = env.step(torch.zeros(25))
        _obs, _r, _d, info2 = env.step(torch.zeros(25))
        assert info1["step"] == 1
        assert info2["step"] == 2

    def test_step_clamps_action(self) -> None:
        env = _make_env()
        env.reset()
        action = torch.ones(25) * 5.0
        _obs, _r, _d, info = env.step(action)
        assert isinstance(info, dict)

    def test_step_with_zero_action(self) -> None:
        env = _make_env()
        env.reset()
        _obs, _r, _d, info = env.step(torch.zeros(25))
        assert info["packets"] == []

    @pytest.mark.smoke
    def test_step_info_includes_pov(self) -> None:
        env = _make_env()
        env.reset()
        _obs, _r, _d, info = env.step(torch.zeros(25))
        assert "pov" in info
        pov = info["pov"]
        assert isinstance(pov, np.ndarray)
        assert pov.dtype == np.uint8
        assert pov.shape == (224, 224, 3)


class TestMockServerEnvPositionTracking:
    def test_movement_packet_updates_position(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(25, dtype=np.float32)
        action[0] = 1.0
        env.step(action)
        assert env._position[0] == 1.0
        assert env._position[2] == 0.0

    def test_strafe_updates_z(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(25, dtype=np.float32)
        action[3] = 1.0
        env.step(action)
        assert env._position[2] == 1.0

    def test_rotation_updates_yaw(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(25, dtype=np.float32)
        action[8] = 1.0
        env.step(action)
        assert env._yaw != 0.0

    def test_zero_action_does_not_move(self) -> None:
        env = _make_env()
        env.reset()
        env.step(torch.zeros(25))
        assert env._position == (0.0, 64.0, 0.0)
        assert env._yaw == 0.0


class TestMockServerEnvClose:
    def test_close_marks_closed(self) -> None:
        env = _make_env()
        env.close()
        assert env._closed is True

    def test_step_after_close_raises(self) -> None:
        env = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.step(torch.zeros(25))

    def test_reset_after_close_raises(self) -> None:
        env = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.reset()


class TestMockServerEnvNoConnection:
    def test_step_does_not_write_to_connection(self) -> None:
        env = _make_env()
        env.reset()
        # The executor's connection is None; no write_packet call should be attempted.
        assert env._executor._connection is None
        action = np.zeros(25, dtype=np.float32)
        action[0] = 1.0
        env.step(action)  # no exception
