from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from wally.training.curriculum import CurriculumConfig, CurriculumTrainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeWorldModel with curriculum learning.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to training data shards.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to resume from checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to curriculum YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to save checkpoints.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated horizon stages (e.g., '8,16,32,64').",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=None,
        help="Loss threshold for stage advancement.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Epochs below threshold before advancing.",
    )
    return parser.parse_args(argv)


def _build_curriculum_config(args: argparse.Namespace) -> CurriculumConfig:
    if args.config is not None:
        if not args.config.is_file():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        config = CurriculumConfig.from_yaml(args.config)
    else:
        config = CurriculumConfig.default()

    overrides: dict[str, object] = {}
    if args.stages is not None:
        overrides["stages"] = [int(s.strip()) for s in args.stages.split(",")]
    if args.loss_threshold is not None:
        overrides["loss_threshold"] = args.loss_threshold
    if args.patience is not None:
        overrides["patience"] = args.patience

    if overrides:
        config = config.model_copy(update=overrides)

    return config


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args(argv)

    if not args.data_dir.is_dir():
        logger.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    curriculum_config = _build_curriculum_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    CurriculumTrainer(curriculum_config, device=device)

    logger.info("Curriculum config: %s", curriculum_config.model_dump())
    logger.info("Stages: %s", curriculum_config.stages)
    logger.info("Loss threshold: %s", curriculum_config.loss_threshold)
    logger.info("Patience: %s", curriculum_config.patience)

    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            logger.error("Checkpoint not found: %s", args.checkpoint)
            sys.exit(1)
        logger.info("Resuming from checkpoint: %s", args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Curriculum training would start with %d stages: %s",
        len(curriculum_config.stages),
        curriculum_config.stages,
    )
    logger.info("Output directory: %s", args.output_dir)
    logger.info(
        "Initial horizon: %d, final horizon: %d",
        curriculum_config.stages[0],
        curriculum_config.stages[-1],
    )


if __name__ == "__main__":
    main()
