from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# collector/validator/exporter use `src.` prefix imports; ensure src is on path
# __file__ = src/wally/cli/collect.py → need project root (4 levels up) so that
# `from src.collector...` resolves to src/collector/
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if _project_root.is_dir() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.collector.collector import TrajectoryCollector  # noqa: E402
from src.collector.config import CollectorConfig, load_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect trajectories from Minecraft via MineStudio.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (optional).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to collect (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for collected data.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Frame skip value (default: 4).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Resize observations (default: 224 224).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Buffer size before flush (default: 1000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args(argv)

    if args.config:
        if not args.config.is_file():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        config = load_config(str(args.config))
    else:
        config = CollectorConfig()

    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.frame_skip is not None:
        config.frame_skip = args.frame_skip
    if args.resize is not None:
        config.resize = tuple(args.resize)
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size

    logger.info(
        "Collecting %d episodes (output=%s, frame_skip=%d, resize=%s)",
        args.episodes,
        config.output_dir,
        config.frame_skip,
        config.resize,
    )

    collector = TrajectoryCollector(config)
    try:
        transitions = collector.run(num_episodes=args.episodes)
        logger.info("Collected %d transitions", len(transitions))
    finally:
        collector.close()


if __name__ == "__main__":
    main()
