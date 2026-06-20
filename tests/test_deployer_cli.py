from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deployer.cli import main, parse_args


def _make_paths(tmp_path: Path) -> tuple[Path, Path]:
    ckpt = tmp_path / "model.pt"
    ckpt.touch()
    goal = tmp_path / "goal.png"
    from PIL import Image
    Image.new("RGB", (8, 8), (128, 128, 128)).save(goal)
    return ckpt, goal


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.server is None
        assert args.checkpoint is None
        assert args.goal_frame is None
        assert args.config is None
        assert args.record is False
        assert args.output_dir is None
        assert args.planner == "cem"
        assert args.mock is False

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

    def test_mock_flag(self):
        args = parse_args(["--mock"])
        assert args.mock is True

    @pytest.mark.smoke
    def test_viewer_default(self):
        args = parse_args([])
        assert args.viewer == "cv2"

    @pytest.mark.smoke
    def test_viewer_explicit_none(self):
        args = parse_args(["--viewer", "none"])
        assert args.viewer == "none"

    @pytest.mark.smoke
    def test_viewer_explicit_cv2(self):
        args = parse_args(["--viewer", "cv2"])
        assert args.viewer == "cv2"

    @pytest.mark.smoke
    def test_no_viewer_alias(self):
        args = parse_args(["--no-viewer"])
        assert args.viewer == "none"

    @pytest.mark.smoke
    def test_invalid_viewer_choice(self):
        with pytest.raises(SystemExit):
            parse_args(["--viewer", "bogus"])

    @pytest.mark.parametrize("planner", ["cem", "gradient", "hierarchical"])
    def test_planner_choices(self, planner: str) -> None:
        args = parse_args(["--planner", planner])
        assert args.planner == planner

    def test_invalid_planner(self):
        with pytest.raises(SystemExit):
            parse_args(["--planner", "invalid"])

    def test_all_args(self):
        args = parse_args([
            "--server", "host:9999",
            "--checkpoint", "ckpt.pt",
            "--goal-frame", "goal.png",
            "--config", "cfg.yaml",
            "--record",
            "--output-dir", "out/",
            "--planner", "gradient",
            "--mock",
        ])
        assert args.server == "host:9999"
        assert args.checkpoint == Path("ckpt.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config == Path("cfg.yaml")
        assert args.record is True
        assert args.output_dir == Path("out/")
        assert args.planner == "gradient"
        assert args.mock is True


class TestMainConfig:
    def test_missing_checkpoint_exits(self, tmp_path):
        from PIL import Image
        goal = tmp_path / "goal.png"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(goal)
        with pytest.raises(SystemExit) as exc_info:
            main(["--goal-frame", str(goal), "--mock"])
        assert exc_info.value.code == 1

    def test_missing_goal_frame_exits(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        with pytest.raises(SystemExit) as exc_info:
            main(["--checkpoint", str(ckpt), "--mock"])
        assert exc_info.value.code == 1

    def test_checkpoint_path_not_found_exits(self, tmp_path):
        goal = tmp_path / "goal.png"
        goal.write_bytes(b"")
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(tmp_path / "nope.pt"),
                "--goal-frame", str(goal),
                "--mock",
            ])
        assert exc_info.value.code == 1

    def test_goal_frame_path_not_found_exits(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(ckpt),
                "--goal-frame", str(tmp_path / "nope.png"),
                "--mock",
            ])
        assert exc_info.value.code == 1

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    def test_main_mock_runs_episode(  # noqa: E501
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_loop = MagicMock()
        from agent.protocol import EpisodeResult
        mock_loop.run_episode.return_value = EpisodeResult(
            steps=2, final_cost=0.5, duration_seconds=0.1
        )
        mock_loop_cls.return_value = mock_loop

        main([
            "--mock",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        mock_env_cls.assert_called_once()
        mock_env.close.assert_called_once()

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    @pytest.mark.smoke
    def test_main_default_viewer_is_null_when_no_viewer(
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env_cls.return_value = MagicMock()
        mock_loop_cls.return_value = MagicMock(
            run_episode=MagicMock(return_value=MagicMock(steps=0, final_cost=0.0, duration_seconds=0.0))
        )
        from agent.viewer import NullViewer

        main([
            "--mock",
            "--no-viewer",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        viewer_kwarg = mock_loop_cls.call_args.kwargs.get("viewer")
        assert isinstance(viewer_kwarg, NullViewer)

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    @pytest.mark.smoke
    def test_main_explicit_cv2_constructs_frame_viewer(
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env_cls.return_value = MagicMock()
        mock_loop_cls.return_value = MagicMock(
            run_episode=MagicMock(return_value=MagicMock(steps=0, final_cost=0.0, duration_seconds=0.0))
        )
        from agent.viewer import FrameViewer

        main([
            "--mock",
            "--viewer", "cv2",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        viewer_kwarg = mock_loop_cls.call_args.kwargs.get("viewer")
        assert isinstance(viewer_kwarg, FrameViewer)
        assert viewer_kwarg._window_name == "wally-deploy"

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    def test_main_server_override(  # noqa: E501
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_loop = MagicMock()
        from agent.protocol import EpisodeResult
        mock_loop.run_episode.return_value = EpisodeResult(
            steps=1, final_cost=0.0, duration_seconds=0.0
        )
        mock_loop_cls.return_value = mock_loop

        main([
            "--mock",
            "--server", "myhost:12345",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "myhost"
        assert config_passed.server_port == 12345

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    def test_main_record_flag(  # noqa: E501
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_loop = MagicMock()
        from agent.protocol import EpisodeResult
        mock_loop.run_episode.return_value = EpisodeResult(
            steps=1, final_cost=0.0, duration_seconds=0.0
        )
        mock_loop_cls.return_value = mock_loop

        main([
            "--mock",
            "--record",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.record_trajectory is True

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    def test_main_config_yaml(  # noqa: E501
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        yaml_file = tmp_path / "deploy.yaml"
        yaml_file.write_text("server_host: yamlhost\nserver_port: 11111\n")
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_loop = MagicMock()
        from agent.protocol import EpisodeResult
        mock_loop.run_episode.return_value = EpisodeResult(
            steps=1, final_cost=0.0, duration_seconds=0.0
        )
        mock_loop_cls.return_value = mock_loop

        main([
            "--config", str(yaml_file),
            "--mock",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        config_passed = mock_env_cls.call_args[0][0]
        assert config_passed.server_host == "yamlhost"
        assert config_passed.server_port == 11111

    @patch("deployer.cli.AgentLoop")
    @patch("deployer.cli.LatentRollout")
    @patch("deployer.cli.build_planner")
    @patch("deployer.env.MockServerEnv")
    def test_main_keyboard_interrupt(  # noqa: E501
        self, mock_env_cls, _planner, _rollout, mock_loop_cls, tmp_path
    ):
        ckpt, goal = _make_paths(tmp_path)
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_loop = MagicMock()
        mock_loop.run_episode.side_effect = KeyboardInterrupt()
        mock_loop_cls.return_value = mock_loop

        main([
            "--mock",
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])

        mock_env.close.assert_called_once()
