from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from wally.planner.cli import main


def _create_test_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(path)


class TestFramesMode:
    def test_happy_path(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _create_test_image(frames_dir / "current.png")
        _create_test_image(frames_dir / "goal.png")

        output_path = tmp_path / "actions.pt"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()

        mock_actions = torch.randn(8, 25)

        with (
            patch("wally.planner.cli.LatentRollout") as MockRollout,
            patch("wally.planner.cli.GoalConditionedPlanner") as MockPlanner,
        ):
            mock_rollout_instance = MagicMock()
            mock_rollout_instance._model.encode = MagicMock(
                side_effect=lambda x: torch.randn(x.shape[0], 192)
            )
            MockRollout.from_checkpoint.return_value = mock_rollout_instance

            mock_planner_instance = MagicMock()
            mock_planner_instance.plan.return_value = mock_actions
            MockPlanner.return_value = mock_planner_instance

            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--frames",
                    str(frames_dir),
                    "--output",
                    str(output_path),
                ]
            )

        assert output_path.exists()
        saved_actions = torch.load(output_path, weights_only=True)
        assert torch.equal(saved_actions, mock_actions)
        mock_planner_instance.plan.assert_called_once()


class TestEnvMode:
    def test_missing_goal_file(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()
        output_path = tmp_path / "actions.pt"
        goal_path = tmp_path / "nonexistent.png"

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--env",
                    "test_env",
                    "--goal",
                    str(goal_path),
                    "--output",
                    str(output_path),
                ]
            )

        assert exc_info.value.code == 1

    def test_goal_not_provided(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()
        output_path = tmp_path / "actions.pt"

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--env",
                    "test_env",
                    "--output",
                    str(output_path),
                ]
            )

        assert exc_info.value.code == 1


class TestMissingFiles:
    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _create_test_image(frames_dir / "current.png")
        _create_test_image(frames_dir / "goal.png")

        output_path = tmp_path / "actions.pt"
        checkpoint_path = tmp_path / "nonexistent.pt"

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--frames",
                    str(frames_dir),
                    "--output",
                    str(output_path),
                ]
            )

        assert exc_info.value.code == 1

    def test_missing_current_frame(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _create_test_image(frames_dir / "goal.png")

        output_path = tmp_path / "actions.pt"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--frames",
                    str(frames_dir),
                    "--output",
                    str(output_path),
                ]
            )

        assert exc_info.value.code == 1

    def test_missing_goal_frame(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _create_test_image(frames_dir / "current.png")

        output_path = tmp_path / "actions.pt"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--frames",
                    str(frames_dir),
                    "--output",
                    str(output_path),
                ]
            )

        assert exc_info.value.code == 1
