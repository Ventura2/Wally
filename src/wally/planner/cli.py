from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan action sequences using a trained LeWorldModel.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to planner YAML config (uses defaults if not provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save output action sequence (.pt).",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--frames",
        type=Path,
        default=None,
        help="Directory containing current.png and goal.png.",
    )
    mode_group.add_argument(
        "--env",
        type=str,
        default=None,
        help="MineStudio environment name.",
    )

    parser.add_argument(
        "--goal",
        type=Path,
        default=None,
        help="Path to goal frame (required with --env).",
    )

    return parser.parse_args(argv)


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    tensor = torch.from_numpy(
        __import__("numpy").array(img, dtype="float32") / 255.0
    )
    return tensor.permute(2, 0, 1)


def _run_frames_mode(args: argparse.Namespace) -> None:
    frames_dir: Path = args.frames
    current_path = frames_dir / "current.png"
    goal_path = frames_dir / "goal.png"

    if not current_path.is_file():
        logger.error("Current frame not found: %s", current_path)
        sys.exit(1)
    if not goal_path.is_file():
        logger.error("Goal frame not found: %s", goal_path)
        sys.exit(1)

    current_frame = _load_image_as_tensor(current_path)
    goal_frame = _load_image_as_tensor(goal_path)

    cem_config = (
        CEMConfig.from_yaml(args.config) if args.config else CEMConfig.default()
    )

    rollout = LatentRollout.from_checkpoint(
        args.checkpoint, gradient_policy=cem_config.gradient_policy
    )
    planner = GoalConditionedPlanner(
        rollout,
        rollout._model.encode,
        cem_config,
    )

    actions = planner.plan(current_frame, goal_frame)
    torch.save(actions, args.output)
    logger.info("Saved action sequence to %s", args.output)


def _run_env_mode(args: argparse.Namespace) -> None:
    if args.goal is None:
        logger.error("--goal is required when using --env mode.")
        sys.exit(1)
    if not args.goal.is_file():
        logger.error("Goal frame not found: %s", args.goal)
        sys.exit(1)

    logger.info(
        "MineStudio environment mode not yet implemented (env=%s).", args.env
    )
    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if args.frames is not None:
        _run_frames_mode(args)
    else:
        _run_env_mode(args)


if __name__ == "__main__":
    main()
