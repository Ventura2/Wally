from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from wally.planner.high_level_planner import (
    HighLevelPlanner,
    HighLevelPlannerConfig,
    HighLevelWorldModel,
    SubgoalExecutionResult,
    train_high_level_model,
)


def _make_mock_model(latent_dim: int = 8) -> MagicMock:
    mock = MagicMock()

    def encode(frame: torch.Tensor) -> torch.Tensor:
        B = frame.shape[0]
        return torch.randn(B, latent_dim)

    def predict(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(z)

    mock.encode = MagicMock(side_effect=encode)
    mock.predict = MagicMock(side_effect=predict)
    return mock


def _make_encoder(latent_dim: int = 8) -> MagicMock:
    def encoder(frame: torch.Tensor) -> torch.Tensor:
        B = frame.shape[0]
        return torch.randn(B, latent_dim)

    return MagicMock(side_effect=encoder)


def _make_low_level_planner() -> MagicMock:
    planner = MagicMock()
    planner.return_value = torch.randn(25)
    return planner


class TestHighLevelPlannerConfigDefaults:
    def test_default_values(self):
        cfg = HighLevelPlannerConfig.default()
        assert cfg.macro_horizon == 5
        assert cfg.macro_action_dim == 25
        assert cfg.population_size == 32
        assert cfg.elite_frac == 0.1
        assert cfg.n_iterations == 5
        assert cfg.subgoal_timeout == 50
        assert cfg.max_replans == 3
        assert cfg.action_low == -1.0
        assert cfg.action_high == 1.0

    def test_constructor_defaults(self):
        cfg = HighLevelPlannerConfig()
        assert cfg.macro_horizon == 5
        assert cfg.population_size == 32


class TestHighLevelPlannerConfigValidation:
    def test_macro_horizon_below_range(self):
        with pytest.raises(ValidationError, match="macro_horizon"):
            HighLevelPlannerConfig(macro_horizon=4)

    def test_macro_horizon_above_range(self):
        with pytest.raises(ValidationError, match="macro_horizon"):
            HighLevelPlannerConfig(macro_horizon=11)

    def test_macro_horizon_at_lower_bound(self):
        cfg = HighLevelPlannerConfig(macro_horizon=5)
        assert cfg.macro_horizon == 5

    def test_macro_horizon_at_upper_bound(self):
        cfg = HighLevelPlannerConfig(macro_horizon=10)
        assert cfg.macro_horizon == 10

    def test_population_size_one_fails(self):
        with pytest.raises(ValidationError, match="population_size"):
            HighLevelPlannerConfig(population_size=1)

    def test_population_size_zero_fails(self):
        with pytest.raises(ValidationError, match="population_size"):
            HighLevelPlannerConfig(population_size=0)

    def test_elite_frac_zero_fails(self):
        with pytest.raises(ValidationError, match="elite_frac"):
            HighLevelPlannerConfig(elite_frac=0.0)

    def test_elite_frac_one_fails(self):
        with pytest.raises(ValidationError, match="elite_frac"):
            HighLevelPlannerConfig(elite_frac=1.0)

    def test_n_iterations_zero_fails(self):
        with pytest.raises(ValidationError, match="n_iterations"):
            HighLevelPlannerConfig(n_iterations=0)

    def test_subgoal_timeout_zero_fails(self):
        with pytest.raises(ValidationError, match="subgoal_timeout"):
            HighLevelPlannerConfig(subgoal_timeout=0)

    def test_max_replans_negative_fails(self):
        with pytest.raises(ValidationError, match="max_replans"):
            HighLevelPlannerConfig(max_replans=-1)


class TestHighLevelPlannerConfigFromYaml:
    def test_load_valid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "hl.yaml"
        yaml_file.write_text(
            "macro_horizon: 7\n"
            "population_size: 64\n"
            "n_iterations: 8\n"
        )
        cfg = HighLevelPlannerConfig.from_yaml(yaml_file)
        assert cfg.macro_horizon == 7
        assert cfg.population_size == 64
        assert cfg.n_iterations == 8

    def test_empty_yaml_uses_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "hl.yaml"
        yaml_file.write_text("")
        cfg = HighLevelPlannerConfig.from_yaml(yaml_file)
        assert cfg == HighLevelPlannerConfig.default()

    def test_invalid_yaml_value_fails(self, tmp_path: Path):
        yaml_file = tmp_path / "hl.yaml"
        yaml_file.write_text("macro_horizon: 3\n")
        with pytest.raises(ValidationError, match="macro_horizon"):
            HighLevelPlannerConfig.from_yaml(yaml_file)


class TestHighLevelPlannerPlanSubgoals:
    def test_returns_subgoal_latents_and_cost(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(
            macro_horizon=5, population_size=8, n_iterations=2
        )
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_frame = torch.randn(3, 64, 64)

        subgoals, cost = planner.plan_subgoals(current_frame, goal_frame)
        assert subgoals.dim() == 2
        assert subgoals.shape[0] == 5
        assert isinstance(cost, float)

    def test_encoder_called_for_both_frames(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(
            macro_horizon=5, population_size=8, n_iterations=2
        )
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_frame = torch.randn(3, 64, 64)
        planner.plan_subgoals(current_frame, goal_frame)

        assert encoder.call_count == 2

    def test_3d_input_handled(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(
            macro_horizon=5, population_size=8, n_iterations=2
        )
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        subgoals, cost = planner.plan_subgoals(current_frame, goal_frame)
        assert subgoals.shape[0] == 5


class TestSubgoalsToTargets:
    def test_converts_to_list_of_latents(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(macro_horizon=5)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        subgoal_latents = torch.randn(5, 8)
        targets = planner.subgoals_to_targets(subgoal_latents)

        assert isinstance(targets, list)
        assert len(targets) == 5
        for t in targets:
            assert isinstance(t, torch.Tensor)
            assert t.shape == (8,)

    def test_preserves_values(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(macro_horizon=5)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        subgoal_latents = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = planner.subgoals_to_targets(subgoal_latents)

        assert torch.allclose(targets[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(targets[1], torch.tensor([3.0, 4.0]))
        assert torch.allclose(targets[2], torch.tensor([5.0, 6.0]))


class TestExecuteSubgoals:
    def test_all_subgoals_reached(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(subgoal_timeout=10)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        targets = [torch.zeros(8) for _ in range(3)]
        low_level = _make_low_level_planner()

        encode_fn = MagicMock(return_value=torch.zeros(1, 8))

        result = planner.execute_subgoals(
            targets, low_level, torch.randn(64, 64),
            reach_threshold=1.0, encode_fn=encode_fn,
        )

        assert result.success is True
        assert result.completed_subgoals == 3
        assert result.total_subgoals == 3
        assert len(result.steps_per_subgoal) == 3

    def test_timeout_detection(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(subgoal_timeout=5)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        targets = [torch.randn(8) for _ in range(3)]
        low_level = _make_low_level_planner()

        encode_fn = MagicMock(return_value=torch.ones(1, 8) * 100)

        result = planner.execute_subgoals(
            targets, low_level, torch.randn(64, 64),
            reach_threshold=0.001, encode_fn=encode_fn,
        )

        assert result.success is False
        assert result.failed is True
        assert result.steps_per_subgoal[0] == 5

    def test_without_encode_fn(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(subgoal_timeout=5)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        targets = [torch.randn(8) for _ in range(2)]
        low_level = _make_low_level_planner()

        result = planner.execute_subgoals(
            targets, low_level, torch.randn(64, 64),
            reach_threshold=1.0,
        )

        assert result.success is True
        assert result.completed_subgoals == 2


class TestReplanning:
    def test_replan_returns_new_subgoals(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(
            macro_horizon=5, population_size=8, n_iterations=2, max_replans=3
        )
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        result = planner.replan(current_frame, goal_frame)
        assert result is not None
        subgoals, cost = result
        assert subgoals.shape[0] == 5

    def test_replan_increments_count(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(max_replans=3)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        planner.replan(current_frame, goal_frame)
        planner.replan(current_frame, goal_frame)
        assert planner._replan_count == 2

    def test_max_replans_exceeded_returns_none(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(max_replans=2)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        planner.replan(current_frame, goal_frame)
        planner.replan(current_frame, goal_frame)
        result = planner.replan(current_frame, goal_frame)

        assert result is None

    def test_reset_replan_count(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(max_replans=3)
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        planner.replan(current_frame, goal_frame)
        planner.replan(current_frame, goal_frame)
        planner.reset_replan_count()
        assert planner._replan_count == 0

    def test_replan_after_reset_works(self):
        model = _make_mock_model()
        encoder = _make_encoder()
        cfg = HighLevelPlannerConfig(
            macro_horizon=5, population_size=8, n_iterations=2, max_replans=1
        )
        planner = HighLevelPlanner(model, encoder, cfg, device="cpu")

        current_frame = torch.randn(64, 64)
        goal_frame = torch.randn(64, 64)

        planner.replan(current_frame, goal_frame)
        assert planner.replan(current_frame, goal_frame) is None

        planner.reset_replan_count()
        result = planner.replan(current_frame, goal_frame)
        assert result is not None


class TestHighLevelWorldModel:
    def test_encode_delegates_to_encoder(self):
        encoder = MagicMock(return_value=torch.randn(2, 8))
        model = HighLevelWorldModel(encoder=encoder, latent_dim=8)

        frame = torch.randn(2, 3, 64, 64)
        result = model.encode(frame)

        encoder.assert_called_once_with(frame)
        assert result.shape == (2, 8)

    def test_predict_returns_correct_shape(self):
        encoder = MagicMock(return_value=torch.randn(2, 8))
        model = HighLevelWorldModel(encoder=encoder, latent_dim=8)

        z = torch.randn(2, 8)
        action = torch.randn(2, 25)
        result = model.predict(z, action)

        assert result.shape == (2, 8)


class TestTrainHighLevelModel:
    def test_returns_model_with_correct_interface(self):
        encoder = MagicMock(return_value=torch.randn(4, 8))
        start_latents = torch.randn(10, 8)
        macro_actions = torch.randn(10, 25)
        end_latents = torch.randn(10, 8)

        model = train_high_level_model(
            encoder=encoder,
            start_latents=start_latents,
            macro_actions=macro_actions,
            end_latents=end_latents,
            epochs=5,
        )

        assert hasattr(model, "encode")
        assert hasattr(model, "predict")

    def test_predict_output_shape(self):
        encoder = MagicMock(return_value=torch.randn(4, 8))
        start_latents = torch.randn(10, 8)
        macro_actions = torch.randn(10, 25)
        end_latents = torch.randn(10, 8)

        model = train_high_level_model(
            encoder=encoder,
            start_latents=start_latents,
            macro_actions=macro_actions,
            end_latents=end_latents,
            epochs=5,
        )

        z = torch.randn(2, 8)
        action = torch.randn(2, 25)
        result = model.predict(z, action)
        assert result.shape == (2, 8)

    def test_loss_decreases(self):
        encoder = MagicMock(return_value=torch.randn(4, 8))
        torch.manual_seed(42)
        start_latents = torch.randn(20, 8)
        macro_actions = torch.randn(20, 25)
        end_latents = start_latents + macro_actions[:, :8] * 0.1

        model = HighLevelWorldModel(encoder=encoder, latent_dim=8)
        optimizer = torch.optim.Adam(model._predictor.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        initial_loss = None
        final_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            predicted = model._predictor(
                torch.cat([start_latents, macro_actions], dim=-1)
            )
            loss = loss_fn(predicted, end_latents)
            loss.backward()
            optimizer.step()
            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss


class TestSubgoalExecutionResult:
    def test_success_result(self):
        result = SubgoalExecutionResult(
            success=True,
            completed_subgoals=3,
            total_subgoals=3,
            steps_per_subgoal=[5, 8, 3],
        )
        assert result.success is True
        assert result.failed is False
        assert result.completed_subgoals == 3

    def test_failure_result(self):
        result = SubgoalExecutionResult(
            success=False,
            completed_subgoals=1,
            total_subgoals=3,
            steps_per_subgoal=[5, 50],
            failed=True,
        )
        assert result.success is False
        assert result.failed is True
        assert result.completed_subgoals == 1
