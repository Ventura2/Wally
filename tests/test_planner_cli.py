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


class TestGradientPolicyPassthrough:
    """Regression tests for the ``gradient_policy`` plumbing.

    The CLI passes ``cem_config.gradient_policy`` through to
    ``LatentRollout.from_checkpoint``; ``from_checkpoint`` must accept and
    forward that kwarg, otherwise the CLI crashes with
    ``TypeError: from_checkpoint() got an unexpected keyword argument
    'gradient_policy'``. The default value is ``"detach"``.
    """

    def _stub_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from wally.planner import rollout as rollout_mod

        def _fake_load(self, checkpoint_path, *, device=None, model_config=None):
            return MagicMock()

        monkeypatch.setattr(
            rollout_mod.LatentRollout, "_load_from_checkpoint", _fake_load
        )

    def test_from_checkpoint_accepts_gradient_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wally.planner.rollout import LatentRollout

        self._stub_loader(monkeypatch)
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        rollout = LatentRollout.from_checkpoint(
            ckpt, gradient_policy="straight_through"
        )
        assert rollout._gradient_policy == "straight_through"

    def test_from_checkpoint_default_is_detach(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wally.planner.rollout import LatentRollout

        self._stub_loader(monkeypatch)
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        rollout = LatentRollout.from_checkpoint(ckpt)
        assert rollout._gradient_policy == "detach"

    def test_cli_passes_gradient_policy_without_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wally.planner import cli as cli_mod

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _create_test_image(frames_dir / "current.png")
        _create_test_image(frames_dir / "goal.png")

        output_path = tmp_path / "actions.pt"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()

        mock_actions = torch.zeros(8, 25)
        captured: dict[str, object] = {}

        class _FakeRollout:
            def __init__(self, ckpt, **kwargs: object) -> None:
                captured["gradient_policy"] = kwargs.get("gradient_policy")
                self._model = MagicMock()
                self._model.encode = MagicMock(
                    side_effect=lambda x: torch.zeros(x.shape[0], 192)
                )

            @classmethod
            def from_checkpoint(cls, ckpt, **kwargs: object) -> "_FakeRollout":
                captured["gradient_policy"] = kwargs.get("gradient_policy")
                return cls(ckpt, **kwargs)

        class _FakePlanner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def plan(self, current_frame, goal_frame, return_cost: bool = False):
                return mock_actions

        monkeypatch.setattr(cli_mod, "LatentRollout", _FakeRollout)
        monkeypatch.setattr(cli_mod, "GoalConditionedPlanner", _FakePlanner)

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
        assert captured["gradient_policy"] == "detach"



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
