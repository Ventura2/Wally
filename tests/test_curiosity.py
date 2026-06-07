from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from wally.training.curiosity import CuriosityConfig, CuriosityModule


class TestCuriosityConfigDefaults:
    def test_default_values(self):
        cfg = CuriosityConfig.default()
        assert cfg.latent_dim == 192
        assert cfg.action_dim == 25
        assert cfg.hidden_dim == 128
        assert cfg.reward_scale == 1.0
        assert cfg.update_frequency == 1
        assert cfg.learning_rate == 1e-3

    def test_constructor_defaults(self):
        cfg = CuriosityConfig()
        assert cfg.latent_dim == 192
        assert cfg.action_dim == 25
        assert cfg.hidden_dim == 128
        assert cfg.reward_scale == 1.0
        assert cfg.update_frequency == 1
        assert cfg.learning_rate == 1e-3


class TestCuriosityConfigValidation:
    def test_negative_reward_scale_fails(self):
        with pytest.raises(ValidationError, match="reward_scale"):
            CuriosityConfig(reward_scale=-1.0)

    def test_zero_reward_scale_fails(self):
        with pytest.raises(ValidationError, match="reward_scale"):
            CuriosityConfig(reward_scale=0.0)

    def test_zero_update_frequency_fails(self):
        with pytest.raises(ValidationError, match="update_frequency"):
            CuriosityConfig(update_frequency=0)

    def test_negative_update_frequency_fails(self):
        with pytest.raises(ValidationError, match="update_frequency"):
            CuriosityConfig(update_frequency=-1)

    def test_zero_learning_rate_fails(self):
        with pytest.raises(ValidationError, match="learning_rate"):
            CuriosityConfig(learning_rate=0.0)

    def test_negative_learning_rate_fails(self):
        with pytest.raises(ValidationError, match="learning_rate"):
            CuriosityConfig(learning_rate=-1e-3)

    def test_zero_hidden_dim_fails(self):
        with pytest.raises(ValidationError, match="hidden_dim"):
            CuriosityConfig(hidden_dim=0)

    def test_valid_custom_values(self):
        cfg = CuriosityConfig(
            latent_dim=64,
            action_dim=10,
            hidden_dim=32,
            reward_scale=2.0,
            update_frequency=5,
            learning_rate=1e-4,
        )
        assert cfg.latent_dim == 64
        assert cfg.action_dim == 10
        assert cfg.hidden_dim == 32
        assert cfg.reward_scale == 2.0
        assert cfg.update_frequency == 5
        assert cfg.learning_rate == 1e-4


class TestCuriosityConfigFromYaml:
    def test_load_valid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "curiosity.yaml"
        yaml_file.write_text(
            "latent_dim: 64\n"
            "action_dim: 10\n"
            "hidden_dim: 32\n"
            "reward_scale: 2.0\n"
            "update_frequency: 3\n"
            "learning_rate: 1e-4\n"
        )
        cfg = CuriosityConfig.from_yaml(yaml_file)
        assert cfg.latent_dim == 64
        assert cfg.action_dim == 10
        assert cfg.hidden_dim == 32
        assert cfg.reward_scale == 2.0
        assert cfg.update_frequency == 3
        assert cfg.learning_rate == 1e-4

    def test_empty_yaml_uses_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "curiosity.yaml"
        yaml_file.write_text("")
        cfg = CuriosityConfig.from_yaml(yaml_file)
        assert cfg == CuriosityConfig.default()

    def test_invalid_yaml_value_fails(self, tmp_path: Path):
        yaml_file = tmp_path / "curiosity.yaml"
        yaml_file.write_text("reward_scale: -1.0\n")
        with pytest.raises(ValidationError, match="reward_scale"):
            CuriosityConfig.from_yaml(yaml_file)


class TestCuriosityModuleForwardModel:
    def test_output_shape(self):
        cfg = CuriosityConfig(latent_dim=192, action_dim=25, hidden_dim=128)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 192)
        action = torch.randn(4, 25)
        x = torch.cat([current, action], dim=-1)
        out = module.forward_model(x)
        assert out.shape == (4, 192)

    def test_output_shape_custom_dims(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(8, 64)
        action = torch.randn(8, 10)
        x = torch.cat([current, action], dim=-1)
        out = module.forward_model(x)
        assert out.shape == (8, 64)


class TestCuriosityModuleIntrinsicReward:
    def test_reward_shape(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        reward = module.compute_intrinsic_reward(current, action, next_lat)
        assert reward.shape == (4,)

    def test_reward_non_negative(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        reward = module.compute_intrinsic_reward(current, action, next_lat)
        assert (reward >= 0).all()

    def test_zero_reward_when_prediction_perfect(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        x = torch.cat([current, action], dim=-1)
        with torch.no_grad():
            predicted = module.forward_model(x)
        reward = module.compute_intrinsic_reward(current, action, predicted)
        assert torch.allclose(reward, torch.zeros(4), atol=1e-6)

    def test_reward_scaling(self):
        common = dict(latent_dim=64, action_dim=10, hidden_dim=32)
        cfg1 = CuriosityConfig(**common, reward_scale=1.0)
        cfg2 = CuriosityConfig(**common, reward_scale=2.0)
        module1 = CuriosityModule(cfg1)
        module2 = CuriosityModule(cfg2)
        module2.load_state_dict(module1.state_dict())

        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)

        r1 = module1.compute_intrinsic_reward(current, action, next_lat)
        r2 = module2.compute_intrinsic_reward(current, action, next_lat)
        assert torch.allclose(r2, 2.0 * r1, atol=1e-6)


class TestCuriosityModuleTrainStep:
    def test_returns_loss_float(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        loss = module.train_step(current, action, next_lat)
        assert isinstance(loss, float)

    def test_loss_decreases_over_steps(self):
        cfg = CuriosityConfig(
            latent_dim=64, action_dim=10, hidden_dim=32, learning_rate=1e-2,
        )
        module = CuriosityModule(cfg)

        torch.manual_seed(42)
        current = torch.randn(16, 64)
        action = torch.randn(16, 10)
        next_lat = torch.randn(16, 64)

        losses = [module.train_step(current, action, next_lat) for _ in range(50)]
        assert losses[-1] < losses[0]


class TestCuriosityModulePriority:
    def test_priority_shape(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        priority = module.compute_priority(current, action, next_lat)
        assert priority.shape == (4,)

    def test_priority_non_negative(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        priority = module.compute_priority(current, action, next_lat)
        assert (priority >= 0).all()

    def test_priority_equals_unscaled_error(self):
        cfg = CuriosityConfig(
            latent_dim=64, action_dim=10, hidden_dim=32, reward_scale=5.0,
        )
        module = CuriosityModule(cfg)
        current = torch.randn(4, 64)
        action = torch.randn(4, 10)
        next_lat = torch.randn(4, 64)
        priority = module.compute_priority(current, action, next_lat)
        reward = module.compute_intrinsic_reward(current, action, next_lat)
        assert torch.allclose(reward, 5.0 * priority, atol=1e-6)


class TestCuriosityModuleBatched:
    def test_batch_computation(self):
        cfg = CuriosityConfig(latent_dim=64, action_dim=10, hidden_dim=32)
        module = CuriosityModule(cfg)
        batch_sizes = [1, 4, 16, 64]
        for bs in batch_sizes:
            current = torch.randn(bs, 64)
            action = torch.randn(bs, 10)
            next_lat = torch.randn(bs, 64)
            reward = module.compute_intrinsic_reward(current, action, next_lat)
            assert reward.shape == (bs,)
            priority = module.compute_priority(current, action, next_lat)
            assert priority.shape == (bs,)
