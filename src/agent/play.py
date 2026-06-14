from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure `src/` is on sys.path so `agent`, `wally`, `collector` are importable
# when the package is invoked without `pip install -e .` (e.g. inside the
# wally-dev Podman container where Python 3.10 + minestudio is used).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.config import AgentConfig
from agent.planner_factory import build_planner
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
    planner = build_planner(args.planner, rollout, encoder)

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
