from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from wally.config.loader import load_config
from wally.data.dataloader import create_dataloader
from wally.models.lewm import LeWorldModel
from wally.training.sigreg import SIGReg
from wally.training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeWorldModel on collected gameplay trajectories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to train on (default: auto).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    args = parse_args(argv)

    if not args.config.is_file():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    train_config, model_config = load_config(args.config)

    if not Path(train_config.data_dir).is_dir():
        logger.error("Data directory not found: %s", train_config.data_dir)
        sys.exit(1)

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model = LeWorldModel(
        vit_variant=model_config.vit_variant,
        embed_dim=model_config.embed_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        dropout=model_config.dropout,
        action_dim=model_config.action_dim,
        pretrained=model_config.pretrained,
        encoder_type=model_config.encoder_type,
    )

    sigreg = SIGReg(
        num_proj=train_config.sigreg_num_proj,
        knots=train_config.sigreg_knots,
    )

    dataloader = create_dataloader(
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        seq_length=train_config.seq_length,
        skip_short=train_config.skip_short,
        persistent_workers=train_config.persistent_workers,
        prefetch_factor=train_config.prefetch_factor,
    )

    config_dict = train_config.to_dict()
    config_dict["device"] = device

    trainer = Trainer(model, sigreg, dataloader, config_dict)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
