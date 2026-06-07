from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from wally.training.optimizer import create_optimizer
from wally.training.scheduler import create_scheduler
from wally.training.checkpoint import save_checkpoint, load_checkpoint


class TestCreateOptimizer:
    def test_returns_adamw(self):
        model = nn.Linear(10, 5)
        opt = create_optimizer(model, lr=1e-4, weight_decay=1e-5)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        model = nn.Linear(10, 5)
        opt = create_optimizer(model, lr=1e-4, weight_decay=1e-5)
        assert len(opt.param_groups) == 2

    def test_no_weight_decay_for_bias(self):
        model = nn.Linear(10, 5)
        opt = create_optimizer(model, lr=1e-4, weight_decay=0.1)
        # bias group should have weight_decay=0
        for group in opt.param_groups:
            if any(p.ndim <= 1 for p in group["params"]):
                assert group["weight_decay"] == 0.0

    def test_learning_rate_set(self):
        model = nn.Linear(10, 5)
        opt = create_optimizer(model, lr=3e-4, weight_decay=1e-5)
        assert opt.param_groups[0]["lr"] == 3e-4


class TestCreateScheduler:
    def test_warmup_phase(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(opt, warmup_steps=100, max_steps=1000)

        # step 0: lr should be 0
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-8)

        # step 50: lr should be ~0.5 * base_lr
        for _ in range(50):
            scheduler.step()
        lr_mid_warmup = scheduler.get_last_lr()[0]
        assert lr_mid_warmup == pytest.approx(0.5 * 1e-3, rel=0.01)

    def test_peak_at_warmup_end(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(opt, warmup_steps=100, max_steps=1000)

        for _ in range(100):
            scheduler.step()
        lr_at_warmup_end = scheduler.get_last_lr()[0]
        assert lr_at_warmup_end == pytest.approx(1e-3, rel=0.01)

    def test_cosine_decay(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(opt, warmup_steps=10, max_steps=100)

        # skip warmup
        for _ in range(10):
            scheduler.step()

        # after warmup, lr should decrease
        lr_start_decay = scheduler.get_last_lr()[0]
        for _ in range(40):
            scheduler.step()
        lr_mid_decay = scheduler.get_last_lr()[0]
        assert lr_mid_decay < lr_start_decay

    def test_lr_never_negative(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(opt, warmup_steps=10, max_steps=100)

        for _ in range(200):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr >= 0.0


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-3)

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, critic_optimizer, 42, {"lr": 1e-3})

        assert ckpt_path.exists()

        model2 = nn.Linear(10, 5)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        critic_opt2 = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-3)

        step = load_checkpoint(ckpt_path, model2, opt2, critic_opt2)
        assert step == 42

    def test_checkpoint_contents(self, tmp_path):
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-3)

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, critic_optimizer, 100, {"batch_size": 8})

        data = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in data
        assert "optimizer_state_dict" in data
        assert "critic_optimizer_state_dict" in data
        assert "global_step" in data
        assert "config" in data
        assert data["global_step"] == 100
        assert data["config"]["batch_size"] == 8

    def test_model_weights_restored(self, tmp_path):
        model = nn.Linear(10, 5)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-3)

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, critic_optimizer, 0, {})

        model2 = nn.Linear(10, 5)
        torch.nn.init.zeros_(model2.weight)
        load_checkpoint(ckpt_path, model2)

        assert torch.equal(model.weight, model2.weight)

    def test_load_without_optimizers(self, tmp_path):
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-3)

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, critic_optimizer, 50, {})

        model2 = nn.Linear(10, 5)
        step = load_checkpoint(ckpt_path, model2)
        assert step == 50
