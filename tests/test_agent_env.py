from unittest.mock import patch

import numpy as np
import pytest
import torch

from wally.agent.config import AgentConfig


@pytest.fixture
def mock_env():
    with patch("wally.collector.env._MinecraftSim") as mock:
        sim = mock.return_value
        sim.reset.return_value = (
            {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
            {},
        )
        sim.step.return_value = (
            {"image": np.ones((224, 224, 3), dtype=np.uint8)},
            1.0,
            False,
            False,
            {},
        )
        from wally.agent.env import MineStudioAgentEnv

        config = AgentConfig(resize=(64, 64))
        env = MineStudioAgentEnv(config)
        yield env, sim


class TestMineStudioAgentEnv:
    def test_reset_returns_tensor_correct_shape(self, mock_env):
        env, sim = mock_env
        obs = env.reset()
        assert isinstance(obs, Tensor)
        assert obs.shape == (3, 64, 64)
        sim.reset.assert_called_once()

    def test_reset_values_in_unit_range(self, mock_env):
        env, _ = mock_env
        obs = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_step_accepts_25_tensor(self, mock_env):
        env, sim = mock_env
        env.reset()
        action = torch.zeros(25)
        action[2] = 1.0
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, Tensor)
        assert obs.shape == (3, 64, 64)
        assert reward == 1.0
        assert done is False
        assert isinstance(info, dict)
        sim.step.assert_called_once()

    def test_step_returns_unit_range_values(self, mock_env):
        env, _ = mock_env
        env.reset()
        action = torch.zeros(25)
        obs, _, _, _ = env.step(action)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_close_prevents_further_steps(self, mock_env):
        env, sim = mock_env
        env.reset()
        env.close()
        sim.close.assert_called_once()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.step(torch.zeros(25))

    def test_action_clipping(self, mock_env):
        env, _ = mock_env
        env.reset()
        action = torch.full((25,), 5.0)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, Tensor)
        assert obs.shape == (3, 64, 64)
