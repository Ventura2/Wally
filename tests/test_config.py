from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from wally.config.model import ModelConfig
from wally.config.training import TrainConfig
from wally.config.loader import load_config


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.vit_variant == "vit_tiny_patch16_224"
        assert cfg.embed_dim == 192
        assert cfg.depth == 6
        assert cfg.num_heads == 4
        assert cfg.mlp_ratio == 4.0
        assert cfg.dropout == 0.1
        assert cfg.action_dim == 25
        assert cfg.pretrained is True

    def test_custom_values(self):
        cfg = ModelConfig(depth=4, embed_dim=256, pretrained=False)
        assert cfg.depth == 4
        assert cfg.embed_dim == 256
        assert cfg.pretrained is False


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1e-5
        assert cfg.warmup_steps == 1000
        assert cfg.max_steps == 100_000
        assert cfg.batch_size == 8
        assert cfg.seq_length == 16
        assert cfg.alpha == 0.1
        assert cfg.use_amp is False
        assert cfg.checkpoint_interval == 1000
        assert cfg.log_interval == 10
        assert cfg.data_dir == "data/shards"
        assert cfg.output_dir == "checkpoints"
        assert cfg.num_workers == 4
        assert cfg.skip_short is True
        assert cfg.wandb_project == "wally"
        assert cfg.resume_from is None

    def test_to_dict(self):
        cfg = TrainConfig(lr=3e-4, batch_size=16)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["lr"] == 3e-4
        assert d["batch_size"] == 16
        assert "max_steps" in d

    def test_custom_values(self):
        cfg = TrainConfig(lr=1e-3, max_steps=50000, use_amp=True)
        assert cfg.lr == 1e-3
        assert cfg.max_steps == 50000
        assert cfg.use_amp is True


class TestLoadConfig:
    def test_load_full_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
model:
  vit_variant: vit_tiny_patch16_224
  embed_dim: 192
  depth: 6
  num_heads: 4

training:
  lr: 0.0003
  batch_size: 16
  max_steps: 50000
"""
        )
        train_cfg, model_cfg = load_config(config_file)
        assert model_cfg.vit_variant == "vit_tiny_patch16_224"
        assert model_cfg.embed_dim == 192
        assert model_cfg.depth == 6
        assert train_cfg.lr == 0.0003
        assert train_cfg.batch_size == 16
        assert train_cfg.max_steps == 50000

    def test_missing_sections_use_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
model:
  depth: 4
"""
        )
        train_cfg, model_cfg = load_config(config_file)
        assert model_cfg.depth == 4
        assert model_cfg.embed_dim == 192  # default
        assert train_cfg.lr == 1e-4  # default

    def test_empty_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        train_cfg, model_cfg = load_config(config_file)
        assert model_cfg == ModelConfig()
        assert train_cfg == TrainConfig()

    def test_unknown_keys_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
model:
  depth: 4
  unknown_field: hello

training:
  lr: 0.001
  also_unknown: 42
"""
        )
        train_cfg, model_cfg = load_config(config_file)
        assert model_cfg.depth == 4
        assert train_cfg.lr == 0.001

    def test_load_default_config(self):
        config_path = Path("configs/lewm_default.yaml")
        if config_path.exists():
            train_cfg, model_cfg = load_config(config_path)
            assert model_cfg.vit_variant == "vit_tiny_patch16_224"
            assert train_cfg.lr == 1e-4
