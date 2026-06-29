"""Tests for the hierarchy trainer's wandb + early-stop + checkpoint_best.

Mirrors the L0 trainer tests in `tests/test_train_logging.py` and the
EMA-based early-stop pattern documented in `AGENTS.md##-early-stopping`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from wally.data.dataloader import create_dataloader
from wally.hierarchy.config import HierarchyConfig, LayerSpec
from wally.hierarchy.encoders import L1Encoder
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.trainer import HierarchyTrainer
from wally.models.lewm import LeWorldModel
from wally.training.checkpoint import load_checkpoint
from wally.training.sigreg import SIGReg

L0_CHECKPOINT = Path("checkpoints/wood_1000/checkpoint_1000.pt")
SHARD_DIR = Path("data/shards/treechop_full")

pytestmark = pytest.mark.skipif(
    not (L0_CHECKPOINT.is_file() and SHARD_DIR.is_dir()),
    reason="L0 checkpoint and/or shard directory not available",
)


def _load_l0() -> LeWorldModel:
    ck = torch.load(L0_CHECKPOINT, map_location="cpu", weights_only=False)
    mc = ck.get("model_config", {})
    model = LeWorldModel(
        embed_dim=int(mc.get("embed_dim", 192)),
        depth=int(mc.get("depth", 4)),
        num_heads=int(mc.get("num_heads", 4)),
        mlp_ratio=float(mc.get("mlp_ratio", 4.0)),
        dropout=float(mc.get("dropout", 0.1)),
        encoder_type=mc.get("encoder_type", "cnn"),
        pretrained=False,
    )
    load_checkpoint(str(L0_CHECKPOINT), model)
    return model


def _build_trainer(tmp: Path, *, early_stop: bool, patience: int = 100,
                   min_step: int = 0, ema_alpha: float = 0.1,
                   wandb_enabled: bool = True) -> HierarchyTrainer:
    l0 = _load_l0()
    enc = L1Encoder(l0, D1=64)
    jepa = JEPAWorldModel(state_dim=64, target_dim=64, hidden_dim=64,
                          depth=1, num_heads=2)
    sigreg = SIGReg(num_proj=64, knots=9)

    cfg = HierarchyConfig(
        layers=[LayerSpec("l1", K=2, D=64, depth=1, heads=2, drift_epsilon=0.10)],
        l0_checkpoint=str(L0_CHECKPOINT),
        max_steps=20,
        warmup_steps=0,
        seq_length=4,
        log_interval=1,
        checkpoint_interval=1,
        output_dir=str(tmp),
        batch_size=2,
        data_dir=str(SHARD_DIR),
        early_stop=early_stop,
        early_stop_patience=patience,
        early_stop_min_step=min_step,
        early_stop_ema_alpha=ema_alpha,
        early_stop_min_delta=0.0,
        wandb_enabled=wandb_enabled,
    )
    dl = create_dataloader(
        data_dir=str(SHARD_DIR),
        batch_size=2,
        num_workers=0,
        seq_length=4,
        skip_short=True,
    )
    return HierarchyTrainer(cfg, enc, jepa, sigreg, dl, device="cpu")


class TestHierarchyConfigRoundTrip:
    def test_to_dict_contains_new_fields(self):
        cfg = HierarchyConfig(
            layers=[LayerSpec("l1", K=2, D=64, depth=1, heads=2, drift_epsilon=0.1)],
            early_stop=True,
            early_stop_patience=250,
            early_stop_min_step=500,
            early_stop_ema_alpha=0.05,
            early_stop_min_delta=1e-4,
            wandb_project="wally-test",
            wandb_enabled=False,
        )
        d = cfg.to_dict()
        for key in (
            "early_stop", "early_stop_patience", "early_stop_min_step",
            "early_stop_ema_alpha", "early_stop_min_delta",
            "wandb_project", "wandb_enabled",
        ):
            assert key in d, f"to_dict() missing {key!r}"

    def test_from_dict_round_trip(self):
        original = HierarchyConfig(
            layers=[LayerSpec("l1", K=2, D=64, depth=1, heads=2, drift_epsilon=0.1)],
            early_stop=True,
            early_stop_patience=250,
            early_stop_min_step=500,
            early_stop_ema_alpha=0.05,
            early_stop_min_delta=1e-4,
            wandb_project="wally-test",
            wandb_enabled=False,
        )
        d = original.to_dict()
        restored = HierarchyConfig.from_dict(d)
        assert restored.early_stop is True
        assert restored.early_stop_patience == 250
        assert restored.early_stop_min_step == 500
        assert restored.early_stop_ema_alpha == pytest.approx(0.05)
        assert restored.early_stop_min_delta == pytest.approx(1e-4)
        assert restored.wandb_project == "wally-test"
        assert restored.wandb_enabled is False

    def test_validation_rejects_bad_ema_alpha(self):
        l1 = [LayerSpec("l1", K=2, D=64, depth=1, heads=2, drift_epsilon=0.1)]
        with pytest.raises(ValueError, match="early_stop_ema_alpha"):
            HierarchyConfig(layers=l1, early_stop_ema_alpha=0.0)
        with pytest.raises(ValueError, match="early_stop_ema_alpha"):
            HierarchyConfig(layers=l1, early_stop_ema_alpha=1.5)

    def test_validation_rejects_bad_patience(self):
        l1 = [LayerSpec("l1", K=2, D=64, depth=1, heads=2, drift_epsilon=0.1)]
        with pytest.raises(ValueError, match="early_stop_patience"):
            HierarchyConfig(layers=l1, early_stop_patience=0)


class TestEarlyStopUpdate:
    """Unit tests for HierarchyTrainer._update_early_stop, isolated from
    the full training loop. These pin the early-stop state machine
    (EMA update, best-checkpoint save, patience trigger) so a future
    refactor can't silently break it."""

    def test_ema_initialised_on_first_call(self):
        with tempfile.TemporaryDirectory(prefix="hier_es_") as tmp:
            trainer = _build_trainer(Path(tmp), early_stop=True, min_step=0)
            trainer._state.global_step = 1
            trainer._update_early_stop(1.0)
            assert trainer._ema_total_loss == 1.0
            assert trainer._best_ema_total_loss == 1.0
            assert trainer._best_step == 1
            assert (Path(tmp) / "checkpoint_best.pt").is_file()

    def test_min_step_skips_early_stop(self):
        with tempfile.TemporaryDirectory(prefix="hier_es_") as tmp:
            trainer = _build_trainer(Path(tmp), early_stop=True, min_step=100)
            trainer._state.global_step = 5
            trainer._update_early_stop(1.0)
            assert trainer._ema_total_loss is None
            assert trainer._best_ema_total_loss == float("inf")
            assert not (Path(tmp) / "checkpoint_best.pt").exists()

    def test_decreasing_loss_writes_best_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="hier_es_") as tmp:
            trainer = _build_trainer(
                Path(tmp), early_stop=True, min_step=0, ema_alpha=0.5
            )
            for step, loss in enumerate([1.0, 0.5, 0.25, 0.1], start=1):
                trainer._state.global_step = step
                trainer._update_early_stop(loss)
            assert trainer._best_step == 4
            assert (Path(tmp) / "checkpoint_best.pt").is_file()
            assert trainer._steps_since_best == 0

    def test_plateau_triggers_stop_after_patience(self):
        with tempfile.TemporaryDirectory(prefix="hier_es_") as tmp:
            trainer = _build_trainer(
                Path(tmp), early_stop=True, min_step=0, patience=3, ema_alpha=1.0,
            )
            trainer._state.global_step = 1
            trainer._update_early_stop(0.5)
            assert not trainer._stop_training
            for step in range(2, 2 + 3):
                trainer._state.global_step = step
                trainer._update_early_stop(0.5)
            assert trainer._stop_training, (
                "Training should have stopped after 3 steps without improvement"
            )
            assert trainer._best_step == 1


class TestWandbIntegration:
    def test_wandb_init_called_when_enabled(self):
        with tempfile.TemporaryDirectory(prefix="hier_wb_") as tmp:
            with patch("wally.hierarchy.trainer.init_wandb") as mock_init, \
                 patch("wally.hierarchy.trainer.log_metrics") as mock_log:
                trainer = _build_trainer(
                    Path(tmp), early_stop=False, wandb_enabled=True
                )
                trainer.train()
            assert mock_init.called, "init_wandb was not called when wandb_enabled=True"
            call_kwargs = mock_init.call_args.kwargs
            assert "name" in call_kwargs
            assert call_kwargs["name"].startswith("wally-2-64-step-")
            assert mock_log.called, "log_metrics was not called when wandb_enabled=True"

    def test_wandb_init_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory(prefix="hier_wb_") as tmp:
            with patch("wally.hierarchy.trainer.init_wandb") as mock_init, \
                 patch("wally.hierarchy.trainer.log_metrics") as mock_log:
                trainer = _build_trainer(
                    Path(tmp), early_stop=False, wandb_enabled=False
                )
                trainer.train()
            assert not mock_init.called, (
                "init_wandb was called even though wandb_enabled=False"
            )
            assert not mock_log.called, (
                "log_metrics was called even though wandb_enabled=False"
            )


class TestCheckpointBestPayload:
    def test_checkpoint_best_contains_required_keys(self):
        with tempfile.TemporaryDirectory(prefix="hier_bp_") as tmp:
            trainer = _build_trainer(
                Path(tmp), early_stop=True, min_step=0
            )
            trainer._state.global_step = 5
            trainer._update_early_stop(0.42)
            best_path = Path(tmp) / "checkpoint_best.pt"
            assert best_path.is_file()
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            for key in (
                "model_state_dict", "encoder_state_dict", "global_step",
                "config", "best_ema_total_loss",
            ):
                assert key in payload, f"checkpoint_best.pt missing {key!r}"
            assert payload["global_step"] == 5
            assert payload["best_ema_total_loss"] == pytest.approx(0.42)
