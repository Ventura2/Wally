from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from pydantic import BaseModel

from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPCConfig
from wally.planner.high_level_planner import HighLevelPlannerConfig
from wally.planner.rollout import LatentRollout

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)


class HierarchicalPlannerConfig(BaseModel):
    cem_config: CEMConfig = CEMConfig.default()
    high_level_config: HighLevelPlannerConfig = HighLevelPlannerConfig.default()
    gradient_mpc_config: GradientMPCConfig = GradientMPCConfig.default()
    subgoal_timeout: int = 50
    max_replans: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> HierarchicalPlannerConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> HierarchicalPlannerConfig:
        return cls()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical planning with high-level subgoal generation"
            " and low-level CEM."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to low-level model checkpoint.",
    )
    parser.add_argument(
        "--high-level-checkpoint",
        type=Path,
        default=None,
        help="Path to high-level model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to hierarchical planner YAML config.",
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


def _run_frames_mode(
    args: argparse.Namespace, config: HierarchicalPlannerConfig,
) -> None:
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

    rollout = LatentRollout.from_checkpoint(
        args.checkpoint,
        gradient_policy=config.cem_config.gradient_policy,
    )

    result: dict[str, object] = {
        "current_frame": current_path.as_posix(),
        "goal_frame": goal_path.as_posix(),
        "config": config.model_dump(),
    }

    if args.high_level_checkpoint is not None:
        from wally.planner.high_level_planner import HighLevelPlanner

        hl_rollout = LatentRollout.from_checkpoint(
            args.high_level_checkpoint,
            gradient_policy=config.cem_config.gradient_policy,
        )
        high_level_planner = HighLevelPlanner(
            hl_rollout._model,
            hl_rollout._model.encode,
            config.high_level_config,
        )
        subgoal_latents, hl_cost = high_level_planner.plan_subgoals(
            current_frame, goal_frame
        )
        result["subgoal_latents"] = subgoal_latents
        result["high_level_cost"] = hl_cost
        logger.info(
            "High-level planner produced %d subgoals (cost=%.4f)",
            subgoal_latents.shape[0],
            hl_cost,
        )

    from wally.planner.plan import GoalConditionedPlanner

    low_level_planner = GoalConditionedPlanner(
        rollout,
        rollout._model.encode,
        config.cem_config,
    )
    actions = low_level_planner.plan(current_frame, goal_frame)

    torch.save({"actions": actions, "metadata": result}, args.output)
    logger.info("Saved hierarchical plan to %s", args.output)


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

    if args.config is not None:
        if not args.config.is_file():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        config = HierarchicalPlannerConfig.from_yaml(args.config)
    else:
        config = HierarchicalPlannerConfig.default()

    logger.info("Hierarchical planner config: %s", config.model_dump())

    if args.frames is not None:
        _run_frames_mode(args, config)
    else:
        _run_env_mode(args)


if __name__ == "__main__":
    main()
