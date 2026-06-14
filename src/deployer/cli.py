from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agent.config import AgentConfig
from agent.loop import AgentLoop
from agent.planner_factory import build_planner
from deployer.config import DeployConfig
from wally.planner.rollout import LatentRollout

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)

_PLANNER_CHOICES = ("cem", "gradient", "hierarchical")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a trained Wally agent to a Minecraft server.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Server address as host:port (default: localhost:25565)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to LeWorldModel checkpoint",
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        default=None,
        help="Path to goal frame image",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to DeployConfig YAML file",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record trajectory during deployment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for recordings",
    )
    parser.add_argument(
        "--planner",
        choices=_PLANNER_CHOICES,
        default="cem",
        help="Planner type (cem, gradient, hierarchical)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run against a MockServerEnv (no live Minecraft server)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=None,
        help="Path to AgentConfig YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for the world model (e.g. cuda, cuda:0, cpu). "
        "Default: cuda if available, else cpu.",
    )
    return parser.parse_args(argv)


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.permute(2, 0, 1)


def _build_config(args: argparse.Namespace) -> DeployConfig:
    if args.config is not None and args.config.is_file():
        config = DeployConfig.from_yaml(args.config)
    else:
        config = DeployConfig.default()

    if args.server is not None:
        parts = args.server.rsplit(":", 1)
        config = config.model_copy(update={"server_host": parts[0]})
        if len(parts) == 2:
            config = config.model_copy(update={"server_port": int(parts[1])})

    if args.checkpoint is not None:
        config = config.model_copy(
            update={"checkpoint_path": str(args.checkpoint)}
        )

    if args.goal_frame is not None:
        config = config.model_copy(
            update={"goal_frame_path": str(args.goal_frame)}
        )

    if args.record:
        config = config.model_copy(update={"record_trajectory": True})

    if args.output_dir is not None:
        config = config.model_copy(update={"output_dir": str(args.output_dir)})

    return config


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    if args.planner not in _PLANNER_CHOICES:
        logger.error(
            "Invalid planner kind: %s. Expected one of %s",
            args.planner,
            _PLANNER_CHOICES,
        )
        sys.exit(1)

    if args.checkpoint is None or not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if args.goal_frame is None or not args.goal_frame.is_file():
        logger.error("Goal frame not found: %s", args.goal_frame)
        sys.exit(1)

    config = _build_config(args)

    goal_frame = _load_image_as_tensor(args.goal_frame)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(args.checkpoint, device=device_str)
    encoder = rollout._model.encode
    planner = build_planner(args.planner, rollout, encoder)  # type: ignore[arg-type]

    if args.mock:
        from deployer.env import MockServerEnv

        env = MockServerEnv(config)
    else:
        from deployer.env import ServerEnv

        env = ServerEnv(config)

    agent_config = (
        AgentConfig.from_yaml(args.agent_config) if args.agent_config else AgentConfig()
    )
    agent_config = agent_config.model_copy(
        update={"record_trajectory": args.record}
    )

    loop = AgentLoop(env, planner, agent_config)
    try:
        result = loop.run_episode(goal_frame)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        env.close()
        return
    except Exception:
        logger.exception("Agent loop failed")
        env.close()
        raise

    print(
        f"Episode complete: {result.steps} steps, "
        f"cost={result.final_cost:.4f}, "
        f"duration={result.duration_seconds:.2f}s"
    )

    if args.record and result.trajectory is not None:
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(config.output_dir)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "episode_0.npz"
        np.savez(out_path, **result.trajectory)
        logger.info("Saved trajectory to %s", out_path)

    env.close()


if __name__ == "__main__":
    main()
