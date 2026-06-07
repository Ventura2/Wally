from __future__ import annotations

import pytest
import torch

from wally.training.curriculum import CurriculumConfig, CurriculumTrainer


class TestCurriculumConfigDefaults:
    def test_default_values(self):
        cfg = CurriculumConfig.default()
        assert cfg.stages == [8, 16, 32, 64]
        assert cfg.loss_threshold == 0.01
        assert cfg.patience == 5
        assert cfg.mix_shorter_sequences is True
        assert cfg.mix_ratio == 0.2
        assert cfg.shaping_weight == 0.1

    def test_constructor_defaults(self):
        cfg = CurriculumConfig()
        assert cfg.stages == [8, 16, 32, 64]


class TestCurriculumConfigValidation:
    def test_empty_stages_fails(self):
        with pytest.raises(Exception, match="stages"):
            CurriculumConfig(stages=[])

    def test_unsorted_stages_fails(self):
        with pytest.raises(Exception, match="stages"):
            CurriculumConfig(stages=[32, 8, 16])

    def test_negative_threshold_fails(self):
        with pytest.raises(Exception, match="loss_threshold"):
            CurriculumConfig(loss_threshold=-0.1)

    def test_zero_threshold_fails(self):
        with pytest.raises(Exception, match="loss_threshold"):
            CurriculumConfig(loss_threshold=0.0)

    def test_zero_patience_fails(self):
        with pytest.raises(Exception, match="patience"):
            CurriculumConfig(patience=0)

    def test_negative_patience_fails(self):
        with pytest.raises(Exception, match="patience"):
            CurriculumConfig(patience=-1)

    def test_mix_ratio_negative_fails(self):
        with pytest.raises(Exception, match="mix_ratio"):
            CurriculumConfig(mix_ratio=-0.1)

    def test_mix_ratio_above_one_fails(self):
        with pytest.raises(Exception, match="mix_ratio"):
            CurriculumConfig(mix_ratio=1.5)

    def test_negative_shaping_weight_fails(self):
        with pytest.raises(Exception, match="shaping_weight"):
            CurriculumConfig(shaping_weight=-0.1)

    def test_valid_custom_values(self):
        cfg = CurriculumConfig(
            stages=[4, 8, 16],
            loss_threshold=0.05,
            patience=3,
            mix_shorter_sequences=False,
            mix_ratio=0.5,
            shaping_weight=0.3,
        )
        assert cfg.stages == [4, 8, 16]
        assert cfg.loss_threshold == 0.05
        assert cfg.patience == 3
        assert cfg.mix_shorter_sequences is False
        assert cfg.mix_ratio == 0.5
        assert cfg.shaping_weight == 0.3

    def test_single_stage_valid(self):
        cfg = CurriculumConfig(stages=[32])
        assert cfg.stages == [32]


class TestStageProgression:
    def test_advance_after_patience_epochs_below_threshold(self):
        cfg = CurriculumConfig(stages=[8, 16, 32], loss_threshold=0.1, patience=3)
        trainer = CurriculumTrainer(cfg, device="cpu")
        assert trainer.current_stage == 0
        assert trainer.current_horizon == 8

        assert trainer.step(0.05) is False
        assert trainer.step(0.05) is False
        assert trainer.step(0.05) is True
        assert trainer.current_stage == 1
        assert trainer.current_horizon == 16

    def test_no_advance_when_above_threshold(self):
        cfg = CurriculumConfig(stages=[8, 16], loss_threshold=0.1, patience=3)
        trainer = CurriculumTrainer(cfg, device="cpu")

        for _ in range(10):
            assert trainer.step(0.5) is False
        assert trainer.current_stage == 0

    def test_counter_resets_on_above_threshold(self):
        cfg = CurriculumConfig(stages=[8, 16], loss_threshold=0.1, patience=3)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trainer.step(0.05)
        trainer.step(0.05)
        trainer.step(0.5)
        trainer.step(0.05)
        trainer.step(0.05)
        trainer.step(0.05)
        assert trainer.current_stage == 1

    def test_best_val_loss_tracks_minimum(self):
        cfg = CurriculumConfig(stages=[8, 16], loss_threshold=0.1, patience=5)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trainer.step(0.05)
        assert trainer.best_val_loss == 0.05
        trainer.step(0.03)
        assert trainer.best_val_loss == 0.03
        trainer.step(0.08)
        assert trainer.best_val_loss == 0.03

    def test_epoch_count_increments(self):
        cfg = CurriculumConfig(stages=[8, 16], loss_threshold=0.1, patience=5)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trainer.step(0.05)
        trainer.step(0.05)
        trainer.step(0.05)
        assert trainer.epoch_count == 3


class TestIsComplete:
    def test_not_complete_at_start(self):
        cfg = CurriculumConfig(stages=[8, 16, 32], loss_threshold=0.1, patience=1)
        trainer = CurriculumTrainer(cfg, device="cpu")
        assert trainer.is_complete is False

    def test_complete_after_all_stages(self):
        cfg = CurriculumConfig(stages=[8, 16], loss_threshold=0.1, patience=1)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trainer.step(0.05)
        assert trainer.current_stage == 1
        assert trainer.is_complete is False

        trainer.step(0.05)
        assert trainer.current_stage == 2
        assert trainer.is_complete is True


class TestDataSlicing:
    def test_slice_to_current_horizon(self):
        cfg = CurriculumConfig(stages=[8, 16], mix_shorter_sequences=False)
        trainer = CurriculumTrainer(cfg, device="cpu")

        frames = torch.randn(4, 32, 3, 64, 64)
        actions = torch.randn(4, 32, 25)
        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (4, 8, 3, 64, 64)
        assert sliced_a.shape == (4, 8, 25)

    def test_slice_at_second_stage(self):
        cfg = CurriculumConfig(
            stages=[8, 16], loss_threshold=0.1, patience=1,
            mix_shorter_sequences=False,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")
        trainer.step(0.05)
        assert trainer.current_horizon == 16

        frames = torch.randn(4, 32, 3, 64, 64)
        actions = torch.randn(4, 32, 25)
        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (4, 16, 3, 64, 64)
        assert sliced_a.shape == (4, 16, 25)

    def test_slice_with_mix_shorter_sequences(self):
        cfg = CurriculumConfig(
            stages=[8],
            mix_shorter_sequences=True,
            mix_ratio=0.5,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        torch.manual_seed(42)
        frames = torch.randn(8, 16, 3, 64, 64)
        actions = torch.randn(8, 16, 25)
        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (8, 8, 3, 64, 64)
        assert sliced_a.shape == (8, 8, 25)

    def test_slice_no_mix_when_disabled(self):
        cfg = CurriculumConfig(stages=[4], mix_shorter_sequences=False)
        trainer = CurriculumTrainer(cfg, device="cpu")

        frames = torch.randn(4, 16, 3, 32, 32)
        actions = torch.randn(4, 16, 25)
        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert torch.equal(sliced_f, frames[:, :4])
        assert torch.equal(sliced_a, actions[:, :4])


class TestShapedCost:
    def test_shaped_cost_without_subgoals(self):
        cfg = CurriculumConfig(shaping_weight=0.5)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trajectory = torch.randn(4, 9, 16)
        goal = torch.randn(4, 16)
        cost = trainer.shaped_cost(trajectory, goal)
        assert cost.shape == (4,)

    def test_shaped_cost_without_subgoals_matches_base(self):
        cfg = CurriculumConfig(shaping_weight=0.0)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trajectory = torch.randn(2, 5, 8)
        goal = torch.randn(2, 8)
        cost = trainer.shaped_cost(trajectory, goal)
        expected = torch.norm(trajectory[:, -1] - goal, p=2, dim=-1)
        torch.testing.assert_close(cost, expected)

    def test_shaped_cost_with_subgoals(self):
        cfg = CurriculumConfig(shaping_weight=0.3)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trajectory = torch.randn(4, 9, 16)
        goal = torch.randn(4, 16)
        subgoals = torch.randn(4, 3, 16)
        cost = trainer.shaped_cost(trajectory, goal, subgoals)
        assert cost.shape == (4,)

    def test_shaped_cost_zero_weight(self):
        cfg = CurriculumConfig(shaping_weight=0.0)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trajectory = torch.randn(2, 5, 8)
        goal = torch.randn(2, 8)
        subgoals = torch.randn(2, 3, 8)
        cost = trainer.shaped_cost(trajectory, goal, subgoals)
        expected = torch.norm(trajectory[:, -1] - goal, p=2, dim=-1)
        torch.testing.assert_close(cost, expected)

    def test_shaped_cost_full_weight(self):
        cfg = CurriculumConfig(shaping_weight=1.0)
        trainer = CurriculumTrainer(cfg, device="cpu")

        trajectory = torch.randn(2, 5, 8)
        goal = torch.randn(2, 8)
        subgoals = torch.randn(2, 3, 8)
        cost = trainer.shaped_cost(trajectory, goal, subgoals)
        assert cost.shape == (2,)


class TestCheckpointSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = CurriculumConfig(stages=[8, 16, 32], loss_threshold=0.05, patience=3)
        trainer = CurriculumTrainer(cfg, device="cpu")
        trainer.current_stage = 1
        trainer.epoch_count = 42
        trainer.best_val_loss = 0.03
        trainer.epochs_below_threshold = 2

        path = tmp_path / "curriculum_state.pt"
        trainer.save_state(path)

        new_trainer = CurriculumTrainer(cfg, device="cpu")
        assert new_trainer.current_stage == 0
        assert new_trainer.epoch_count == 0

        new_trainer.load_state(path)
        assert new_trainer.current_stage == 1
        assert new_trainer.epoch_count == 42
        assert new_trainer.best_val_loss == 0.03
        assert new_trainer.epochs_below_threshold == 2

    def test_save_creates_file(self, tmp_path):
        cfg = CurriculumConfig.default()
        trainer = CurriculumTrainer(cfg, device="cpu")
        path = tmp_path / "state.pt"
        trainer.save_state(path)
        assert path.exists()
