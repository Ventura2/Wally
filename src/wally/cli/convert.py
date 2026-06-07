from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from wally.data.converter import convert_shards

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw trajectory shards to training format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with raw shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for training shards.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (optional).",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=50,
        help="Episodes per output shard (default: 50).",
    )
    parser.add_argument(
        "--action-schema",
        nargs="+",
        default=None,
        help="Action keys in order (overrides config).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args(argv)

    if args.config:
        if not args.config.is_file():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        action_schema: list[str] | None = config.get("action_schema")
        episodes_per_shard = config.get("episodes_per_shard", args.episodes_per_shard)
    else:
        action_schema = args.action_schema
        episodes_per_shard = args.episodes_per_shard

    if args.action_schema:
        action_schema = args.action_schema

    if not action_schema:
        logger.error("Action schema required (--action-schema or --config)")
        sys.exit(1)

    if not args.input.is_dir():
        logger.error("Input directory not found: %s", args.input)
        sys.exit(1)

    stats = convert_shards(
        input_dir=args.input,
        output_dir=args.output,
        action_schema=action_schema,
        episodes_per_shard=episodes_per_shard,
    )

    logger.info(
        "Converted %d episodes to %d shards",
        stats["episode_count"],
        stats["shard_count"],
    )
    if stats.get("skipped_episodes", 0) > 0:
        logger.warning("Skipped %d episodes", stats["skipped_episodes"])


if __name__ == "__main__":
    main()
