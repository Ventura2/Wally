from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from deployer.config import DeployConfig


def _make_env() -> tuple:
    mock_connector = MagicMock()
    mock_connector.connection = MagicMock()

    mock_session = MagicMock()
    mock_session.position = (0.0, 64.0, 0.0)

    mock_executor = MagicMock()
    mock_executor.execute.return_value = [{"type": "movement", "dx": 1.0}]

    mock_renderer = MagicMock()
    mock_renderer.render.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)

    mock_safety = MagicMock()
    mock_safety.check.return_value = True

    mock_throttler = MagicMock()

    with (
        patch("deployer.env.ServerConnector", return_value=mock_connector),
        patch("deployer.env.SessionManager", return_value=mock_session),
        patch("deployer.env.ActionExecutor", return_value=mock_executor),
        patch("deployer.env.FrameRenderer", return_value=mock_renderer),
        patch("deployer.env.SafetyFilter", return_value=mock_safety),
        patch("deployer.env.ActionThrottler", return_value=mock_throttler),
    ):
        from deployer.env import ServerEnv

        config = DeployConfig.default()
        env = ServerEnv(config)

    return env, mock_connector, mock_session, mock_executor, mock_renderer, mock_safety


class TestServerEnvReset:
    def test_reset_returns_tensor(self):
        env, _, _, _, mock_renderer, _ = _make_env()
        obs = env.reset()
        assert isinstance(obs, torch.Tensor)

    def test_reset_returns_correct_shape(self):
        env, _, _, _, mock_renderer, _ = _make_env()
        mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
        obs = env.reset()
        assert obs.shape == (3, 224, 224)

    def test_reset_raises_when_closed(self):
        env, _, _, _, _, _ = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.reset()

    def test_reset_connects_via_session(self):
        env, _, mock_session, _, _, _ = _make_env()
        env.reset()
        mock_session.join.assert_called_once()

    def test_reset_sets_executor_connection(self):
        env, mock_connector, _, mock_executor, _, _ = _make_env()
        env.reset()
        assert mock_executor.connection is mock_connector.connection

    def test_reset_renders_initial_frame(self):
        env, _, mock_session, _, mock_renderer, _ = _make_env()
        env.reset()
        mock_renderer.render.assert_called_once_with(
            mock_session.position, 0.0, 0.0
        )
        mock_renderer.preprocess.assert_called_once()


class TestServerEnvStep:
    def test_step_returns_tuple(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_step_returns_tensor_obs(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, torch.Tensor)

    def test_step_returns_float_reward(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        _, reward, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_step_returns_bool_done(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        _, _, done, _ = env.step(action)
        assert isinstance(done, bool)

    def test_step_returns_dict_info(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        _, _, _, info = env.step(action)
        assert isinstance(info, dict)

    def test_step_with_valid_action_produces_observation(self):
        env, _, _, mock_executor, mock_renderer, _ = _make_env()
        env.reset()
        mock_renderer.render.reset_mock()
        mock_renderer.preprocess.reset_mock()
        action = torch.zeros(25)
        obs, _, _, info = env.step(action)
        mock_executor.execute.assert_called_once()
        assert obs.shape == (3, 224, 224)
        assert "packets" in info
        assert "step" in info

    def test_step_raises_when_closed(self):
        env, _, _, _, _, _ = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.step(torch.zeros(25))

    def test_step_safety_violation_returns_flag(self):
        env, _, mock_session, _, mock_renderer, mock_safety = _make_env()
        env.reset()
        mock_safety.check.return_value = False
        mock_renderer.render.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
        action = torch.zeros(25)
        obs, reward, done, info = env.step(action)
        assert info.get("safety_violation") is True
        assert reward == 0.0
        assert done is False

    def test_step_increments_step_count(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        action = torch.zeros(25)
        _, _, _, info1 = env.step(action)
        _, _, _, info2 = env.step(action)
        assert info1["step"] == 1
        assert info2["step"] == 2

    def test_step_accepts_numpy_action(self):
        env, _, _, mock_executor, _, _ = _make_env()
        env.reset()
        action_np = np.zeros(25, dtype=np.float32)
        env.step(action_np)
        mock_executor.execute.assert_called_once()


class TestServerEnvClose:
    def test_close_shuts_down_session(self):
        env, _, mock_session, _, _, _ = _make_env()
        env.close()
        mock_session.shutdown.assert_called_once()

    def test_double_close_is_noop(self):
        env, _, mock_session, _, _, _ = _make_env()
        env.close()
        env.close()
        mock_session.shutdown.assert_called_once()

    def test_step_after_close_raises(self):
        env, _, _, _, _, _ = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.step(torch.zeros(25))

    def test_reset_after_close_raises(self):
        env, _, _, _, _, _ = _make_env()
        env.close()
        with pytest.raises(RuntimeError, match="Environment is closed"):
            env.reset()


class TestServerEnvInterface:
    def test_has_reset_method(self):
        env, _, _, _, _, _ = _make_env()
        assert hasattr(env, "reset")
        assert callable(env.reset)

    def test_has_step_method(self):
        env, _, _, _, _, _ = _make_env()
        assert hasattr(env, "step")
        assert callable(env.step)

    def test_has_close_method(self):
        env, _, _, _, _, _ = _make_env()
        assert hasattr(env, "close")
        assert callable(env.close)

    def test_reset_returns_tensor(self):
        env, _, _, _, _, _ = _make_env()
        obs = env.reset()
        assert isinstance(obs, torch.Tensor)

    def test_step_accepts_tensor(self):
        env, _, _, _, _, _ = _make_env()
        env.reset()
        obs, reward, done, info = env.step(torch.zeros(25))
        assert isinstance(obs, torch.Tensor)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
