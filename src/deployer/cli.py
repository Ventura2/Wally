from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deployer.config import DeployConfig

logger = logging.getLogger(__name__)


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
        help="Path to YAML config file",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

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
        config = config.model_copy(update={"checkpoint_path": str(args.checkpoint)})

    if args.goal_frame is not None:
        config = config.model_copy(update={"goal_frame_path": str(args.goal_frame)})

    if args.record:
        config = config.model_copy(update={"record_trajectory": True})

    if args.output_dir is not None:
        config = config.model_copy(update={"output_dir": str(args.output_dir)})

    from deployer.env import ServerEnv

    env = ServerEnv(config)

    step_count = 0
    try:
        obs = env.reset()
        logger.info("Agent deployed. Observation shape: %s", obs.shape)
        logger.info("Starting autonomous gameplay loop...")
        done = False
        while not done:
            import torch

            action = torch.zeros(25)
            obs, reward, done, info = env.step(action)
            step_count += 1
            if step_count % 100 == 0:
                logger.info("Step %d, reward=%.2f", step_count, reward)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        env.close()
        logger.info("Deployment ended after %d steps", step_count)


if __name__ == "__main__":
    main()
