from __future__ import annotations

from pathlib import Path

import torch

from wally.training.curriculum import CurriculumConfig, CurriculumTrainer


class TestCurriculumTrainingE2E:
    def test_full_stage_progression(self) -> None:
        cfg = CurriculumConfig(
            stages=[4, 8, 16],
            loss_threshold=0.01,
            patience=3,
            mix_shorter_sequences=False,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        assert trainer.current_stage == 0
        assert trainer.current_horizon == 4

        for _ in range(3):
            result = trainer.step(0.005)
        assert result is True
        assert trainer.current_stage == 1
        assert trainer.current_horizon == 8

        for _ in range(3):
            result = trainer.step(0.005)
        assert result is True
        assert trainer.current_stage == 2
        assert trainer.current_horizon == 16

        for _ in range(3):
            result = trainer.step(0.005)
        assert result is True
        assert trainer.current_stage == 3
        assert trainer.is_complete is True

    def test_data_slicing_at_each_stage(self) -> None:
        cfg = CurriculumConfig(
            stages=[4, 8, 16],
            loss_threshold=0.01,
            patience=2,
            mix_shorter_sequences=False,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        frames = torch.randn(8, 32, 3, 64, 64)
        actions = torch.randn(8, 32, 25)

        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (8, 4, 3, 64, 64)
        assert sliced_a.shape == (8, 4, 25)

        for _ in range(2):
            trainer.step(0.005)
        assert trainer.current_horizon == 8

        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (8, 8, 3, 64, 64)
        assert sliced_a.shape == (8, 8, 25)

        for _ in range(2):
            trainer.step(0.005)
        assert trainer.current_horizon == 16

        sliced_f, sliced_a = trainer.slice_data(frames, actions)
        assert sliced_f.shape == (8, 16, 3, 64, 64)
        assert sliced_a.shape == (8, 16, 25)

    def test_shaped_cost_computation(self) -> None:
        cfg = CurriculumConfig(
            stages=[4, 8],
            loss_threshold=0.01,
            patience=2,
            shaping_weight=0.3,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        B, H, Z = 4, 10, 16
        trajectory = torch.randn(B, H, Z)
        goal = torch.randn(B, Z)
        subgoals = torch.randn(B, 3, Z)

        cost_no_subgoals = trainer.shaped_cost(trajectory, goal)
        assert cost_no_subgoals.shape == (B,)
        assert (cost_no_subgoals >= 0).all()

        cost_with_subgoals = trainer.shaped_cost(trajectory, goal, subgoals)
        assert cost_with_subgoals.shape == (B,)
        assert (cost_with_subgoals >= 0).all()

    def test_checkpoint_save_load_roundtrip(self, tmp_path: Path) -> None:
        cfg = CurriculumConfig(
            stages=[4, 8, 16],
            loss_threshold=0.01,
            patience=3,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        for _ in range(5):
            trainer.step(0.005)
        for _ in range(3):
            trainer.step(0.005)

        assert trainer.current_stage == 2
        assert trainer.epoch_count == 8

        path = tmp_path / "curriculum_e2e.pt"
        trainer.save_state(path)
        assert path.exists()

        new_trainer = CurriculumTrainer(cfg, device="cpu")
        assert new_trainer.current_stage == 0
        assert new_trainer.epoch_count == 0

        new_trainer.load_state(path)
        assert new_trainer.current_stage == trainer.current_stage
        assert new_trainer.epoch_count == trainer.epoch_count
        assert new_trainer.best_val_loss == trainer.best_val_loss
        assert (
            new_trainer.epochs_below_threshold
            == trainer.epochs_below_threshold
        )

    def test_progression_with_patience_reset(self) -> None:
        cfg = CurriculumConfig(
            stages=[4, 8],
            loss_threshold=0.01,
            patience=3,
        )
        trainer = CurriculumTrainer(cfg, device="cpu")

        trainer.step(0.005)
        trainer.step(0.005)
        trainer.step(0.5)
        trainer.step(0.005)
        trainer.step(0.005)
        assert trainer.current_stage == 0

        trainer.step(0.005)
        assert trainer.current_stage == 1
