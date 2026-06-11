from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agent.config import AgentConfig
from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig
from wally.planner.hierarchical_planner import (
    HierarchicalPlanner,
    HierarchicalPlannerConfig,
)
from wally.planner.high_level_planner import (
    HighLevelPlanner,
    HighLevelPlannerConfig,
)
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a goal-conditioned agent episode using a trained LeWorldModel."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to LeWorldModel checkpoint.",
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        required=True,
        help="Path to goal image.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to AgentConfig YAML.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Enable trajectory recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for trajectory export.",
    )
    parser.add_argument(
        "--planner",
        choices=["cem", "gradient", "hierarchical"],
        default="cem",
        help="Planner type.",
    )
    return parser.parse_args(argv)


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.permute(2, 0, 1)


def _build_planner(
    args: argparse.Namespace,
    rollout: LatentRollout,
    encoder: torch.nn.Module,
):
    cem_config = CEMConfig.default()
    gradient_mpc_config = GradientMPCConfig.default()
    high_level_config = HighLevelPlannerConfig.default()
    hier_config = HierarchicalPlannerConfig.default()

    if args.planner == "cem":
        from agent.protocol import FlatPlannerAdapter

        planner = GoalConditionedPlanner(rollout, encoder, cem_config)
        return FlatPlannerAdapter(planner)

    if args.planner == "gradient":
        from agent.protocol import FlatPlannerAdapter

        planner = GradientMPC(rollout, encoder, gradient_mpc_config)
        return FlatPlannerAdapter(planner)

    from agent.protocol import HierarchicalPlannerAdapter

    high_level = HighLevelPlanner(rollout._model, encoder, high_level_config)
    low_level = GoalConditionedPlanner(rollout, encoder, cem_config)
    planner = HierarchicalPlanner(high_level, low_level, hier_config)
    return HierarchicalPlannerAdapter(planner)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if not args.goal_frame.is_file():
        logger.error("Goal frame not found: %s", args.goal_frame)
        sys.exit(1)

    config = AgentConfig.from_yaml(args.config) if args.config else AgentConfig()
    config = config.model_copy(update={"record_trajectory": args.record})

    goal_frame = _load_image_as_tensor(args.goal_frame)

    rollout = LatentRollout.from_checkpoint(args.checkpoint)
    encoder = rollout._model.encode
    planner = _build_planner(args, rollout, encoder)

    try:
        from agent.env import MineStudioAgentEnv
    except ImportError as exc:
        logger.error("MineStudio is not installed: %s", exc)
        sys.exit(1)

    env = MineStudioAgentEnv(config)

    from agent.loop import AgentLoop

    loop = AgentLoop(env, planner, config)
    result = loop.run_episode(goal_frame)

    print(
        f"Episode complete: {result.steps} steps, "
        f"cost={result.final_cost:.4f}, "
        f"duration={result.duration_seconds:.2f}s"
    )

    if args.record and result.trajectory is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "episode_0.npz"
        np.savez(out_path, **result.trajectory)
        logger.info("Saved trajectory to %s", out_path)


if __name__ == "__main__":
    main()
