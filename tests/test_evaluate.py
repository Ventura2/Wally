from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytest

from wally.training.evaluate import evaluate, _plot_latent_comparison


class SimpleModel(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, frames, actions):
        B, T = frames.shape[:2]
        pred = torch.randn(B, T - 1, self.embed_dim)
        target = torch.randn(B, T - 1, self.embed_dim)
        return pred, target


class TestPlotLatentComparison:
    def test_creates_png(self, tmp_path):
        pred = torch.randn(2, 15, 64)
        target = torch.randn(2, 15, 64)
        out_path = tmp_path / "test_viz.png"
        _plot_latent_comparison(pred, target, out_path, step=100)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        pred = torch.randn(2, 15, 64)
        target = torch.randn(2, 15, 64)
        out_path = tmp_path / "subdir" / "test_viz.png"
        _plot_latent_comparison(pred, target, out_path, step=100)
        assert out_path.exists()


class TestEvaluate:
    def test_returns_float(self, tmp_path):
        model = SimpleModel(embed_dim=64)

        frames = torch.randn(2, 4, 3, 224, 224)
        actions = torch.randn(2, 4, 25)
        dataset = [{"frames": frames, "actions": actions}]
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(dataset))

        val_loss = evaluate(model, loader, torch.device("cpu"), tmp_path, step=0)
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0

    def test_creates_visualization(self, tmp_path):
        model = SimpleModel(embed_dim=64)

        frames = torch.randn(2, 4, 3, 224, 224)
        actions = torch.randn(2, 4, 25)
        dataset = [{"frames": frames, "actions": actions}]
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(dataset))

        evaluate(model, loader, torch.device("cpu"), tmp_path, step=42)
        viz_path = tmp_path / "eval_step_42.png"
        assert viz_path.exists()

    def test_model_restored_to_train_mode(self, tmp_path):
        model = SimpleModel(embed_dim=64)
        model.train()

        frames = torch.randn(2, 4, 3, 224, 224)
        actions = torch.randn(2, 4, 25)
        dataset = [{"frames": frames, "actions": actions}]
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(dataset))

        evaluate(model, loader, torch.device("cpu"), tmp_path, step=0)
        assert model.training
