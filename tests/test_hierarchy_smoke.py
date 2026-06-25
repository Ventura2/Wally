"""End-to-end smoke tests for the hierarchy.

These tests exercise the full L1-training path on a real L0 checkpoint
and a small number of steps. They are intended for ``pytest -m smoke``,
not for nightly benchmarks.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from wally.data.dataloader import create_dataloader
from wally.hierarchy.config import HierarchyConfig, LayerSpec
from wally.hierarchy.encoders import L1Encoder
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.loss import temporal_coherence_loss
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


class TestHierarchySmoke:
    def test_l1_encoder_loads_real_checkpoint(self):
        l0 = _load_l0()
        enc = L1Encoder(l0, D1=64)
        assert enc.D == 64
        assert enc.proj.weight.requires_grad
        assert not next(enc.l0_model.parameters()).requires_grad

    def test_l1_jepa_loss_decreases_after_a_few_steps(self):
        torch.manual_seed(0)
        l0 = _load_l0()
        enc = L1Encoder(l0, D1=64)
        jepa = JEPAWorldModel(state_dim=64, target_dim=64, hidden_dim=64, depth=2, num_heads=4)
        sigreg = SIGReg(num_proj=64, knots=9)

        cfg = HierarchyConfig(
            layers=[LayerSpec("l1", K=2, D=64, depth=2, heads=4, drift_epsilon=0.10)],
            l0_checkpoint=str(L0_CHECKPOINT),
            max_steps=10,
            warmup_steps=0,
            seq_length=4,
            log_interval=1,
            checkpoint_interval=1,
            output_dir=str(tempfile.mkdtemp(prefix="hierarchy_smoke_")),
            batch_size=2,
            data_dir=str(SHARD_DIR),
        )
        dl = create_dataloader(
            data_dir=str(SHARD_DIR),
            batch_size=2,
            num_workers=0,
            seq_length=4,
            skip_short=True,
        )
        trainer = HierarchyTrainer(cfg, enc, jepa, sigreg, dl, device="cpu")
        losses: list[float] = []
        for i, batch in enumerate(dl):
            if i >= 10:
                break
            metrics, _, _ = trainer._training_step(batch)
            losses.append(metrics["total_loss"])
        assert len(losses) == 10
        assert sum(losses[:5]) > sum(losses[-5:])

    def test_l1_temporal_coherence_loss_is_smaller_within_trajectory(self):
        torch.manual_seed(0)
        l0 = _load_l0()
        enc = L1Encoder(l0, D1=64).eval()
        for p in enc.parameters():
            p.requires_grad = False

        frames_t0 = torch.randn(2, 3, 224, 224)
        frames_t1 = frames_t0 + 0.01 * torch.randn_like(frames_t0)
        frames_random = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            e_t0 = enc.encode_sequence(frames_t0.unsqueeze(1)).squeeze(1)
            e_t1 = enc.encode_sequence(frames_t1.unsqueeze(1)).squeeze(1)
            e_rand = enc.encode_sequence(frames_random.unsqueeze(1)).squeeze(1)

        same = temporal_coherence_loss(e_t0, e_t1)
        diff = temporal_coherence_loss(e_t0, e_rand)
        assert same.item() < diff.item()
