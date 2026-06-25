from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
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
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Device to train on. Training always uses GPU; the default is "
            "'cuda'. Pass 'cpu' only for fast smoke tests on tiny configs "
            "(a warning is logged). If 'cuda' is selected but "
            "torch.cuda.is_available() is False, the CLI exits with a clear "
            "error pointing at docs/gpu-setup.md."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a log file. When set, every trainer INFO "
            "record is also appended to this file in addition to stdout."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(file_handler)
        logger.info("Logging to file: %s", args.log_file)

    if not args.config.is_file():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    train_config, model_config = load_config(args.config)

    if not Path(train_config.data_dir).is_dir():
        logger.error("Data directory not found: %s", train_config.data_dir)
        sys.exit(1)

    if args.device == "cpu":
        logger.warning(
            "Training on CPU is not supported for production runs. This path "
            "exists only for fast smoke tests on tiny configs. Expect OOMs or "
            "extreme slowness on real datasets."
        )
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            logger.error(
                "Training requires a GPU but torch.cuda.is_available() is "
                "False. This usually means PyTorch was installed without CUDA "
                "support or the active venv is CPU-only. Reinstall torch from "
                "the TheRock multi-arch index as shown in docs/gpu-setup.md "
                "(\"Windows — recommended for training\")."
            )
            sys.exit(2)
        device = torch.device("cuda")
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

    trainer = Trainer(
        model, sigreg, dataloader, config_dict, model_config=asdict(model_config)
    )

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
