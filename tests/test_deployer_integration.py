from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from wally.deployer.config import DeployConfig, ReconnectConfig
from wally.deployer.connector import ServerConnector
from wally.deployer.env import ServerEnv
from wally.deployer.session import SessionManager


class TestDeploymentPipeline:
    def test_full_pipeline_connect_reset_step_close(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
            patch("wally.deployer.env.SafetyFilter") as mock_safety_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_connector.state = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            mock_safety = MagicMock()
            mock_safety.check.return_value = True
            mock_safety_cls.return_value = mock_safety

            env = ServerEnv(config)

            obs = env.reset()
            assert isinstance(obs, torch.Tensor)
            assert obs.shape == (3, 224, 224)

            for _ in range(5):
                action = torch.zeros(25)
                obs, reward, done, info = env.step(action)
                assert isinstance(obs, torch.Tensor)
                assert obs.shape == (3, 224, 224)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)

            env.close()

    def test_executor_receives_valid_actions(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
            patch("wally.deployer.env.SafetyFilter") as mock_safety_cls,
            patch("wally.deployer.env.ActionExecutor") as mock_executor_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            mock_safety = MagicMock()
            mock_safety.check.return_value = True
            mock_safety_cls.return_value = mock_safety

            mock_executor = MagicMock()
            mock_executor.execute.return_value = [{"type": "movement", "dx": 0.5}]
            mock_executor_cls.return_value = mock_executor

            env = ServerEnv(config)
            env.reset()

            action = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0]
            )
            env.step(action)

            mock_executor.execute.assert_called_once()
            passed_action = mock_executor.execute.call_args[0][0]
            assert isinstance(passed_action, np.ndarray)
            assert passed_action.shape == (25,)

            env.close()

    def test_step_returns_packets_in_info(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
            patch("wally.deployer.env.SafetyFilter") as mock_safety_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            mock_safety = MagicMock()
            mock_safety.check.return_value = True
            mock_safety_cls.return_value = mock_safety

            env = ServerEnv(config)
            env.reset()

            _, _, _, info = env.step(torch.zeros(25))
            assert "packets" in info
            assert "step" in info
            assert info["step"] == 1

            env.close()

    def test_close_is_idempotent(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            env = ServerEnv(config)
            env.close()
            env.close()

    def test_step_after_close_raises(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            env = ServerEnv(config)
            env.close()

            with pytest.raises(RuntimeError, match="Environment is closed"):
                env.step(torch.zeros(25))

    def test_reset_after_close_raises(self, tmp_path):
        config = DeployConfig.default().model_copy(
            update={"checkpoint_path": str(tmp_path / "checkpoint.pt")}
        )

        with (
            patch("wally.deployer.env.ServerConnector") as mock_conn_cls,
            patch("wally.deployer.env.FrameRenderer") as mock_renderer_cls,
        ):
            mock_connector = MagicMock()
            mock_connector.connection = MagicMock()
            mock_conn_cls.return_value = mock_connector

            mock_renderer = MagicMock()
            mock_renderer.render.return_value = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )
            mock_renderer.preprocess.return_value = torch.zeros(3, 224, 224)
            mock_renderer_cls.return_value = mock_renderer

            env = ServerEnv(config)
            env.close()

            with pytest.raises(RuntimeError, match="Environment is closed"):
                env.reset()


class TestReconnectionFlow:
    def test_state_persistence_across_reconnect(self, tmp_path):
        checkpoint = tmp_path / "state.json"

        mock_connector = MagicMock(spec=ServerConnector)
        mock_connector.on_disconnect = MagicMock()

        session = SessionManager(
            mock_connector,
            reconnect_config=ReconnectConfig(max_attempts=3),
            checkpoint_path=checkpoint,
        )

        session.position = (10.0, 64.0, -30.0)
        session.inventory = {"diamond_sword": 1}
        session.goal_progress = {"target": "mine_diamonds", "progress": 0.5}

        session.shutdown()

        assert checkpoint.exists()

        mock_connector2 = MagicMock(spec=ServerConnector)
        mock_connector2.on_disconnect = MagicMock()
        session2 = SessionManager(
            mock_connector2,
            checkpoint_path=checkpoint,
        )
        session2.join()

        assert session2.position == (10.0, 64.0, -30.0)
        assert session2.inventory == {"diamond_sword": 1}
        assert session2.goal_progress["target"] == "mine_diamonds"
        assert session2.goal_progress["progress"] == 0.5

    def test_shutdown_saves_state_and_disconnects(self, tmp_path):
        checkpoint = tmp_path / "shutdown_state.json"

        mock_connector = MagicMock(spec=ServerConnector)
        mock_connector.on_disconnect = MagicMock()

        session = SessionManager(
            mock_connector,
            checkpoint_path=checkpoint,
        )
        session.position = (100.0, 70.0, 200.0)
        session.inventory = {"cobblestone": 64, "torch": 32}

        session.shutdown()

        mock_connector.disconnect.assert_called_once()
        assert checkpoint.exists()

        import json

        data = json.loads(checkpoint.read_text())
        assert data["position"] == [100.0, 70.0, 200.0]
        assert data["inventory"] == {"cobblestone": 64, "torch": 32}

    def test_join_without_checkpoint_starts_fresh(self, tmp_path):
        checkpoint = tmp_path / "no_existing_state.json"

        mock_connector = MagicMock(spec=ServerConnector)
        mock_connector.on_disconnect = MagicMock()

        session = SessionManager(
            mock_connector,
            checkpoint_path=checkpoint,
        )
        session.join()

        assert session.position == (0.0, 0.0, 0.0)
        assert session.inventory == {}
        assert session.goal_progress == {}
        mock_connector.connect.assert_called_once()

    def test_multiple_save_restore_cycles(self, tmp_path):
        checkpoint = tmp_path / "cycle_state.json"

        positions = [
            (0.0, 64.0, 0.0),
            (10.0, 65.0, -5.0),
            (50.0, 70.0, 100.0),
        ]

        for pos in positions:
            mock_conn = MagicMock(spec=ServerConnector)
            mock_conn.on_disconnect = MagicMock()
            session = SessionManager(mock_conn, checkpoint_path=checkpoint)
            session.position = pos
            session.shutdown()

        mock_conn_final = MagicMock(spec=ServerConnector)
        mock_conn_final.on_disconnect = MagicMock()
        session_final = SessionManager(
            mock_conn_final, checkpoint_path=checkpoint
        )
        session_final.join()

        assert session_final.position == positions[-1]
