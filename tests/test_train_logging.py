"""Tests for wally.cli.train logging configuration.

The trainer's logger output must reach ``--log-file`` (in addition to
stdout) so future runs leave an auditable loss curve on disk. This was
the regression observed on 2026-06-14: stdout was fully buffered, the
log file stayed at 0 bytes for hours, and the loss collapse was only
diagnosable by checkpoint mtimes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from wally.cli import train as cli_train
from wally.config.model import ModelConfig
from wally.config.training import TrainConfig
from wally.models.lewm import LeWorldModel
from wally.training.sigreg import SIGReg
from wally.training.trainer import Trainer


class _OneStepIterableDataset(IterableDataset):
    """Iterable dataset that yields exactly one batch."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self._batch = batch

    def __iter__(self):
        yield self._batch


def _make_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "frames": torch.rand(2, 4, 3, 64, 64),
        "actions": torch.clamp(torch.randn(2, 4, 25), -1.0, 1.0),
    }


def _make_train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        lr=1e-4,
        weight_decay=1e-5,
        warmup_steps=1,
        max_steps=1,
        batch_size=2,
        seq_length=4,
        alpha=0.1,
        sigreg_num_proj=32,
        sigreg_knots=9,
        use_amp=False,
        checkpoint_interval=1000,
        log_interval=1,
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "ckpts"),
        num_workers=0,
    )


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        vit_variant="vit_tiny_patch16_224",
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        action_dim=25,
        pretrained=False,
        encoder_type="cnn",
    )


def _reset_root_logger() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


class TestTrainLogging:
    @pytest.mark.smoke
    def test_basicconfig_uses_stdout(self, tmp_path) -> None:
        """main() must configure the root logger with a stdout StreamHandler."""
        _reset_root_logger()
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text("model: {}\ntraining: {}\n")
        with (
            patch.object(cli_train, "load_config") as mock_load,
            patch.object(cli_train, "create_dataloader"),
            patch.object(cli_train, "Trainer") as mock_trainer_cls,
            patch.object(cli_train, "LeWorldModel"),
            patch.object(cli_train, "SIGReg"),
        ):
            tmp_cfg = TrainConfig(data_dir="data", log_interval=1, max_steps=1)
            mock_load.return_value = (tmp_cfg, _make_model_config())
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.train.return_value = None

            cli_train.main(
                [
                    "--config",
                    str(config_file),
                    "--device",
                    "cpu",
                ]
            )

        root = logging.getLogger()
        assert root.level == logging.INFO, "root logger must be at INFO"
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert stream_handlers, "root logger must have a StreamHandler"
        assert stream_handlers[0].stream is not None

    @pytest.mark.smoke
    def test_log_file_writes_metric_lines(self, tmp_path) -> None:
        """When --log-file is set, the trainer's metric line must be appended."""
        _reset_root_logger()
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text("model: {}\ntraining: {}\n")
        log_file = tmp_path / "train.log"
        batch = _make_batch()
        train_config = _make_train_config(tmp_path)
        model_config = _make_model_config()

        with (
            patch.object(cli_train, "load_config") as mock_load,
            patch.object(cli_train, "create_dataloader") as mock_loader,
            patch.object(cli_train, "LeWorldModel") as mock_model_cls,
            patch("wally.training.trainer.init_wandb"),
            patch("wally.training.trainer.log_metrics"),
        ):
            mock_load.return_value = (train_config, model_config)
            model = LeWorldModel(
                encoder_type="cnn", embed_dim=64, depth=2, num_heads=4,
                action_dim=25, num_frames=4,
            )
            mock_model_cls.return_value = model
            loader = DataLoader(
                _OneStepIterableDataset(batch),
                batch_size=None,
            )
            mock_loader.return_value = loader

            cli_train.main(
                [
                    "--config",
                    str(config_file),
                    "--device",
                    "cpu",
                    "--log-file",
                    str(log_file),
                ]
            )

        # Force the file handler to flush.
        for h in logging.getLogger().handlers:
            h.flush()

        contents = log_file.read_text(encoding="utf-8")
        assert contents.strip(), f"log file {log_file} is empty"
        metric_pattern = re.compile(
            r"Step \d+ \| prediction_loss=[\d.eE+-]+ \| sigreg_loss=[\d.eE+-]+ "
            r"\| total_loss=[\d.eE+-]+ \| lr=[\d.eE+-]+"
        )
        assert metric_pattern.search(contents), (
            f"no metric line found in log file. Contents:\n{contents}"
        )

    @pytest.mark.smoke
    def test_init_wandb_forwards_name(self) -> None:
        """``init_wandb`` forwards the ``name`` kwarg to ``wandb.init``."""
        from wally.training.logging import init_wandb

        with patch("wally.training.logging.wandb") as mock_wandb:
            init_wandb({}, project_name="wally", name="wally-step-50000")

        mock_wandb.init.assert_called_once()
        assert mock_wandb.init.call_args.kwargs.get("name") == "wally-step-50000"

    @pytest.mark.smoke
    @pytest.mark.parametrize("global_step", [0, 50000])
    def test_trainer_run_name_uses_global_step(
        self, tmp_path, global_step: int
    ) -> None:
        """Trainer passes ``name='<project>-step-<N>'`` to ``init_wandb``.

        Covers both a fresh run (global_step=0) and a resumed run
        (global_step=50000) to verify the name reflects the resume state.
        """
        _reset_root_logger()
        batch = _make_batch()
        train_config = _make_train_config(tmp_path)

        model = LeWorldModel(
            encoder_type="cnn", embed_dim=64, depth=2, num_heads=4,
            action_dim=25, num_frames=4,
        )
        sigreg = SIGReg(
            num_proj=train_config.sigreg_num_proj,
            knots=train_config.sigreg_knots,
        )
        loader = DataLoader(
            _OneStepIterableDataset(batch),
            batch_size=None,
        )

        config_dict = train_config.to_dict()
        config_dict["device"] = torch.device("cpu")
        config_dict["wandb_project"] = "wally"

        trainer = Trainer(model, sigreg, loader, config_dict)
        trainer.global_step = global_step

        captured: dict[str, object] = {}

        def fake_init_wandb(config, *, name=None):
            captured["name"] = name

        with (
            patch("wally.training.trainer.init_wandb", fake_init_wandb),
            patch("wally.training.trainer.log_metrics"),
        ):
            trainer.train()

        assert "name" in captured, "init_wandb was not called"
        name = captured["name"]
        assert name == f"wally-step-{global_step}", (
            f"expected 'wally-step-{global_step}', got {name!r}"
        )
