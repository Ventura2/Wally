"""End-to-end integration test for the wally-deploy CLI with a planner."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from deployer.cli import main
from wally.models.lewm import LeWorldModel


def _build_dummy_checkpoint(path: Path) -> None:
    """Build a tiny LeWorldModel checkpoint using the CNN encoder (no downloads)."""
    model = LeWorldModel(
        encoder_type="cnn",
        embed_dim=32,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        action_dim=25,
        num_frames=4,
        pretrained=False,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "encoder_type": "cnn",
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "action_dim": 25,
            "num_frames": 4,
        },
        "global_step": 0,
    }
    torch.save(checkpoint, path)


def _make_goal_frame(path: Path) -> None:
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path)


def _force_cpu():
    """Patch torch.cuda.is_available + the planner's device auto-detect to CPU."""
    return [
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "wally.planner.plan.torch.cuda.is_available", return_value=False
        ),
    ]


def test_cli_cem_with_mock_runs_episode(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _build_dummy_checkpoint(ckpt)
    goal = tmp_path / "goal.png"
    _make_goal_frame(goal)

    buf = io.StringIO()
    patches = _force_cpu()
    for p in patches:
        p.start()
    try:
        with redirect_stdout(buf):
            main([
                "--mock",
                "--planner", "cem",
                "--checkpoint", str(ckpt),
                "--goal-frame", str(goal),
            ])
    finally:
        for p in patches:
            p.stop()

    output = buf.getvalue()
    assert "Episode complete" in output
    assert "steps" in output
    assert "cost=" in output
    assert "duration=" in output


def test_cli_gradient_planner_constructs(tmp_path: Path) -> None:
    """The gradient planner requires grad-enabled parameters; with the eval-mode
    LatentRollout it can build but the per-step backward pass cannot run on
    a frozen model. We assert that ``build_planner('gradient', ...)`` returns
    a FlatPlannerAdapter (the CLI builds the planner successfully)."""
    from agent.planner_factory import build_planner
    from agent.protocol import FlatPlannerAdapter
    from wally.planner.rollout import LatentRollout

    ckpt = tmp_path / "model.pt"
    _build_dummy_checkpoint(ckpt)

    rollout = LatentRollout.from_checkpoint(ckpt)
    encoder = rollout._model.encode
    planner = build_planner("gradient", rollout, encoder)
    assert isinstance(planner, FlatPlannerAdapter)


def test_cli_hierarchical_planner_constructs(tmp_path: Path) -> None:
    from agent.planner_factory import build_planner
    from agent.protocol import HierarchicalPlannerAdapter
    from wally.planner.rollout import LatentRollout

    ckpt = tmp_path / "model.pt"
    _build_dummy_checkpoint(ckpt)

    rollout = LatentRollout.from_checkpoint(ckpt)
    encoder = rollout._model.encode
    planner = build_planner("hierarchical", rollout, encoder)
    assert isinstance(planner, HierarchicalPlannerAdapter)


def test_cli_writes_trajectory_when_recorded(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _build_dummy_checkpoint(ckpt)
    goal = tmp_path / "goal.png"
    _make_goal_frame(goal)
    out_dir = tmp_path / "out"

    patches = _force_cpu()
    for p in patches:
        p.start()
    try:
        main([
            "--mock",
            "--planner", "cem",
            "--record",
            "--output-dir", str(out_dir),
            "--checkpoint", str(ckpt),
            "--goal-frame", str(goal),
        ])
    finally:
        for p in patches:
            p.stop()

    npz_path = out_dir / "episode_0.npz"
    assert npz_path.exists()
    assert npz_path.stat().st_size > 0
