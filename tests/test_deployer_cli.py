from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from deployer.cli import main, parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.server is None
        assert args.checkpoint is None
        assert args.goal_frame is None
        assert args.config is None
        assert args.record is False
        assert args.output_dir is None

    def test_server(self):
        args = parse_args(["--server", "myhost:12345"])
        assert args.server == "myhost:12345"

    def test_checkpoint(self):
        args = parse_args(["--checkpoint", "/path/to/ckpt.pt"])
        assert args.checkpoint == Path("/path/to/ckpt.pt")

    def test_goal_frame(self):
        args = parse_args(["--goal-frame", "/path/to/goal.png"])
        assert args.goal_frame == Path("/path/to/goal.png")

    def test_config(self):
        args = parse_args(["--config", "/path/to/config.yaml"])
        assert args.config == Path("/path/to/config.yaml")

    def test_record_flag(self):
        args = parse_args(["--record"])
        assert args.record is True

    def test_output_dir(self):
        args = parse_args(["--output-dir", "/path/to/output"])
        assert args.output_dir == Path("/path/to/output")

    def test_all_args(self):
        args = parse_args([
            "--server", "host:9999",
            "--checkpoint", "ckpt.pt",
            "--goal-frame", "goal.png",
            "--config", "cfg.yaml",
            "--record",
            "--output-dir", "out/",
        ])
        assert args.server == "host:9999"
        assert args.checkpoint == Path("ckpt.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config == Path("cfg.yaml")
        assert args.record is True
        assert args.output_dir == Path("out/")


class TestMainConfig:
    @patch("deployer.env.ServerEnv")
    def test_main_defaults(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main([])

        mock_env_cls.assert_called_once()
        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "localhost"
        assert config_passed.server_port == 25565
        mock_env.reset.assert_called_once()
        mock_env.close.assert_called_once()

    @patch("deployer.env.ServerEnv")
    def test_main_server_override(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--server", "myhost:12345"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "myhost"
        assert config_passed.server_port == 12345

    @patch("deployer.env.ServerEnv")
    def test_main_server_host_only(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--server", "myhost"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "myhost"
        assert config_passed.server_port == 25565

    @patch("deployer.env.ServerEnv")
    def test_main_checkpoint_override(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--checkpoint", "/path/to/model.pt"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.checkpoint_path == str(Path("/path/to/model.pt"))

    @patch("deployer.env.ServerEnv")
    def test_main_goal_frame_override(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--goal-frame", "/path/to/goal.png"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.goal_frame_path == str(Path("/path/to/goal.png"))

    @patch("deployer.env.ServerEnv")
    def test_main_record_flag(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--record"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.record_trajectory is True

    @patch("deployer.env.ServerEnv")
    def test_main_output_dir_override(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--output-dir", "/tmp/recordings"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.output_dir == str(Path("/tmp/recordings"))

    @patch("deployer.env.ServerEnv")
    def test_main_config_yaml(self, mock_env_cls, tmp_path):
        yaml_file = tmp_path / "deploy.yaml"
        yaml_file.write_text("server_host: yamlhost\nserver_port: 11111\n")

        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--config", str(yaml_file)])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "yamlhost"
        assert config_passed.server_port == 11111

    @patch("deployer.env.ServerEnv")
    def test_main_config_yaml_with_cli_override(self, mock_env_cls, tmp_path):
        yaml_file = tmp_path / "deploy.yaml"
        yaml_file.write_text("server_host: yamlhost\nserver_port: 11111\n")

        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.return_value = (torch.zeros(3, 224, 224), 0.0, True, {})
        mock_env_cls.return_value = mock_env

        main(["--config", str(yaml_file), "--server", "clihost:22222"])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "clihost"
        assert config_passed.server_port == 22222

    @patch("deployer.env.ServerEnv")
    def test_main_step_loop(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.side_effect = [
            (torch.zeros(3, 224, 224), 1.0, False, {}),
            (torch.zeros(3, 224, 224), 2.0, False, {}),
            (torch.zeros(3, 224, 224), 3.0, True, {}),
        ]
        mock_env_cls.return_value = mock_env

        main([])

        assert mock_env.step.call_count == 3
        mock_env.close.assert_called_once()

    @patch("deployer.env.ServerEnv")
    def test_main_keyboard_interrupt(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = torch.zeros(3, 224, 224)
        mock_env.step.side_effect = KeyboardInterrupt()
        mock_env_cls.return_value = mock_env

        main([])

        mock_env.close.assert_called_once()
