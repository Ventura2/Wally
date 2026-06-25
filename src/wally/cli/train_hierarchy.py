"""``wally-train-hierarchy`` — train a hierarchy layer (L1, L2, or L3).

Mirrors the structure of :mod:`wally.cli.train` but operates on top of a
frozen L0 LeWorldModel checkpoint. Only the JEPA world model and the
layer's linear projection are trained.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Literal

import torch

from wally.data.concat_dataset import create_concat_dataloader
from wally.data.dataloader import create_dataloader
from wally.hierarchy.config import HierarchyConfig
from wally.hierarchy.encoders import L1Encoder, L2Encoder, L3Encoder
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.trainer import HierarchyTrainer
from wally.models.lewm import LeWorldModel
from wally.training.checkpoint import load_checkpoint
from wally.training.sigreg import SIGReg

logger = logging.getLogger(__name__)

LayerArg = Literal["l1", "l2", "l3"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a hierarchy layer (L1, L2, or L3) on top of a frozen L0 "
            "LeWorldModel checkpoint."
        ),
    )
    parser.add_argument(
        "--layer",
        choices=["l1", "l2", "l3"],
        required=True,
        help="Which layer to train.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the hierarchy YAML config.",
    )
    parser.add_argument(
        "--l0-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the L0 LeWorldModel checkpoint. Overrides "
            "HierarchyConfig.l0_checkpoint when given. Required for L1."
        ),
    )
    parser.add_argument(
        "--lower-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the lower layer's checkpoint (L1 for L2, L2 for L3). "
            "Required for L2 and L3 training."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Device to train on. Training always uses GPU; the default is "
            "'cuda'. Pass 'cpu' only for fast smoke tests on tiny configs "
            "(a warning is logged)."
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


def _load_l0_model(checkpoint: Path) -> LeWorldModel:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mc = ck.get("model_config", {}) or {}
    model = LeWorldModel(
        embed_dim=int(mc.get("embed_dim", 192)),
        depth=int(mc.get("depth", 4)),
        num_heads=int(mc.get("num_heads", 4)),
        mlp_ratio=float(mc.get("mlp_ratio", 4.0)),
        dropout=float(mc.get("dropout", 0.1)),
        encoder_type=mc.get("encoder_type", "cnn"),
        pretrained=False,
    )
    load_checkpoint(str(checkpoint), model)
    return model


def _build_encoder_and_model(
    layer: LayerArg,
    config: HierarchyConfig,
    l0_model: LeWorldModel,
    lower_checkpoint: Path | None,
) -> tuple[torch.nn.Module, JEPAWorldModel]:
    spec = config.layers[0]
    if layer == "l1":
        encoder: torch.nn.Module = L1Encoder(l0_model, D1=spec.D)
    elif layer == "l2":
        if lower_checkpoint is None:
            raise ValueError("--lower-checkpoint is required for L2 training")
        encoder = L2Encoder.from_l1_checkpoint(
            str(lower_checkpoint), l0_model, D1=64, D2=spec.D
        )
    else:
        if lower_checkpoint is None:
            raise ValueError("--lower-checkpoint is required for L3 training")
        encoder = L3Encoder.from_l2_checkpoint(
            str(lower_checkpoint), l0_model, D1=64, D2=32, D3=spec.D
        )
    jepa = JEPAWorldModel(
        state_dim=spec.D,
        target_dim=spec.D,
        hidden_dim=spec.D * 2,
        depth=spec.depth,
        num_heads=spec.heads,
    )
    return encoder, jepa


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
    config = HierarchyConfig.from_yaml(args.config)

    if args.l0_checkpoint is not None:
        config.l0_checkpoint = str(args.l0_checkpoint)
    if not config.l0_checkpoint:
        logger.error(
            "HierarchyConfig.l0_checkpoint is required (or pass --l0-checkpoint)"
        )
        sys.exit(1)
    l0_ckpt = Path(config.l0_checkpoint)
    if not l0_ckpt.is_file():
        logger.error("L0 checkpoint not found: %s", l0_ckpt)
        sys.exit(1)

    if args.device == "cpu":
        logger.warning(
            "Training on CPU is not supported for production runs. This path "
            "exists only for fast smoke tests on tiny configs."
        )
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            logger.error(
                "Training requires a GPU but torch.cuda.is_available() is "
                "False. See docs/gpu-setup.md for the TheRock multi-arch "
                "install steps."
            )
            sys.exit(2)
        device = torch.device("cuda")
    logger.info("Using device: %s", device)
    logger.info("Training layer %s (K=%d, D=%d, depth=%d, heads=%d)",
                args.layer, config.layers[0].K, config.layers[0].D,
                config.layers[0].depth, config.layers[0].heads)

    # Kick off the dataloader in a background thread so its
    # (potentially slow) ``ConcatenatedShardDataset`` index build and
    # first-batch worker spinup run in parallel with the L0 checkpoint
    # load below. With the on-disk index cache the build itself is
    # <1 s, but on a cold cache the first run still pays 30-50 s for
    # the scan — overlapping it with the model load hides that cost.
    dataloader_holder: list = [None]
    dataloader_error: list = [None]

    def _build_dataloader() -> None:
        try:
            if config.use_concat_dataloader:
                logger.info(
                    "Using ConcatenatedShardDataset "
                    "(multi-chunk windows, seq_length=%d)",
                    config.seq_length,
                )
                dataloader_holder[0] = create_concat_dataloader(
                    data_dir=config.data_dir,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    persistent_workers=config.persistent_workers,
                    prefetch_factor=config.prefetch_factor,
                    seq_length=config.seq_length,
                    skip_short=True,
                )
            else:
                logger.info(
                    "Using per-chunk dataloader (single-chunk windows, seq_length=%d)",
                    config.seq_length,
                )
                dataloader_holder[0] = create_dataloader(
                    data_dir=config.data_dir,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    persistent_workers=config.persistent_workers,
                    prefetch_factor=config.prefetch_factor,
                    seq_length=config.seq_length,
                    skip_short=True,
                )
        except Exception as exc:  # noqa: BLE001
            dataloader_error[0] = exc

    dl_thread = threading.Thread(target=_build_dataloader, name="dataloader-build")
    dl_thread.start()

    l0_model = _load_l0_model(l0_ckpt)
    encoder, jepa = _build_encoder_and_model(
        args.layer, config, l0_model, args.lower_checkpoint
    )

    sigreg = SIGReg(num_proj=128, knots=9)

    dl_thread.join()
    if dataloader_error[0] is not None:
        raise dataloader_error[0]
    dataloader = dataloader_holder[0]

    trainer = HierarchyTrainer(
        config=config,
        encoder=encoder,
        world_model=jepa,
        sigreg=sigreg,
        dataloader=dataloader,
        device=device,
    )
    trainer.train(logger=logger)
    logger.info("Done. Checkpoints written to %s", config.output_dir)


if __name__ == "__main__":
    main()
