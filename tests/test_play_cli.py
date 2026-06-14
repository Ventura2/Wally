from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.play import main, parse_args


class TestParseArgs:
    def test_parse_args_minimal(self) -> None:
        args = parse_args(["--checkpoint", "model.pt", "--goal-frame", "goal.png"])
        assert args.checkpoint == Path("model.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config is None
        assert args.record is False
        assert args.output_dir == Path(".")
        assert args.planner == "cem"
        assert args.viewer == "cv2"

    def test_parse_args_all_options(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("replan_interval: 8\n")
        output_dir = tmp_path / "out"

        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--config", str(config_path),
            "--record",
            "--output-dir", str(output_dir),
            "--planner", "gradient",
        ])
        assert args.checkpoint == Path("model.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config == Path(str(config_path))
        assert args.record is True
        assert args.output_dir == output_dir
        assert args.planner == "gradient"
        assert args.viewer == "cv2"

    @pytest.mark.parametrize("choice", ["cem", "gradient", "hierarchical"])
    def test_parse_args_planner_choices(self, choice: str) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--planner", choice,
        ])
        assert args.planner == choice

    def test_parse_args_invalid_planner(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([
                "--checkpoint", "model.pt",
                "--goal-frame", "goal.png",
                "--planner", "invalid",
            ])

    @pytest.mark.smoke
    def test_parse_args_viewer_none(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--viewer", "none",
        ])
        assert args.viewer == "none"

    def test_parse_args_no_viewer_alias(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--no-viewer",
        ])
        assert args.viewer == "none"

    def test_parse_args_invalid_viewer(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([
                "--checkpoint", "model.pt",
                "--goal-frame", "goal.png",
                "--viewer", "invalid",
            ])


class TestMain:
    def test_missing_checkpoint(self) -> None:
        with pytest.raises(SystemExit):
            main(["--goal-frame", "goal.png"])

    def test_checkpoint_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(tmp_path / "nonexistent.pt"),
                "--goal-frame", "goal.png",
            ])
        assert exc_info.value.code == 1

    def test_goal_frame_not_found(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(checkpoint),
                "--goal-frame", str(tmp_path / "nonexistent.png"),
            ])
        assert exc_info.value.code == 1

    def test_config_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_data = {"replan_interval": 8, "episode_timeout": 500}
        config_path.write_text(yaml.dump(config_data))

        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        goal = tmp_path / "goal.png"
        goal.touch()

        args = parse_args([
            "--checkpoint", str(checkpoint),
            "--goal-frame", str(goal),
            "--config", str(config_path),
        ])

        from agent.config import AgentConfig

        loaded = AgentConfig.from_yaml(args.config)
        assert loaded.replan_interval == 8
        assert loaded.episode_timeout == 500
