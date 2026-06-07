from __future__ import annotations

from typing import Callable

import torch

from wally.planner.high_level_planner import (
    HighLevelPlanner,
    HighLevelPlannerConfig,
    HighLevelWorldModel,
    train_high_level_model,
)
from wally.planner.subgoal_detector import (
    SubgoalDetector,
    SubgoalDetectorConfig,
)

EMBED_DIM = 16
ACTION_DIM = 4
N_PHASES = 3
STEPS_PER_PHASE = 20
TOTAL_STEPS = N_PHASES * STEPS_PER_PHASE


def _make_phase_latents(phase_id: int, n_steps: int, dim: int) -> torch.Tensor:
    torch.manual_seed(phase_id * 1000)
    base = torch.randn(dim) * (phase_id + 1)
    return (
        base.unsqueeze(0).expand(n_steps, -1).clone()
        + torch.randn(n_steps, dim) * 0.1
    )


def _make_mock_encoder(dim: int = EMBED_DIM) -> Callable[[torch.Tensor], torch.Tensor]:
    def encode(frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 2:
            return frames
        if frames.dim() == 1:
            return frames.unsqueeze(0)
        return torch.randn(frames.shape[0], dim)

    return encode


class TestSubgoalDetectionToHierarchicalPlanning:
    def test_full_pipeline(self) -> None:
        torch.manual_seed(42)

        phase_latents = []
        for p in range(N_PHASES):
            phase_latents.append(_make_phase_latents(p, STEPS_PER_PHASE, EMBED_DIM))
        latents = torch.cat(phase_latents, dim=0)
        assert latents.shape == (TOTAL_STEPS, EMBED_DIM)

        actions = torch.randn(TOTAL_STEPS - 1, ACTION_DIM)

        class MultiPhaseModel:
            def __init__(self, latents: torch.Tensor) -> None:
                self._latents = latents

            def encode(self, frame: torch.Tensor) -> torch.Tensor:
                n = frame.shape[0]
                return self._latents[:n]

            def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(z)

        model = MultiPhaseModel(latents)

        detector_cfg = SubgoalDetectorConfig(
            threshold=1.0,
            smoothing_window=3,
            min_segment_length=5,
        )
        detector = SubgoalDetector(detector_cfg)

        frames_dummy = torch.randn(TOTAL_STEPS, 3, 64, 64)
        errors = detector.compute_prediction_errors(model, frames_dummy, actions)
        assert errors.shape == (TOTAL_STEPS - 1,)

        smoothed = detector.smooth_errors(errors)
        assert smoothed.shape == errors.shape

        change_points = detector.detect_change_points(smoothed)
        assert isinstance(change_points, list)

        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        assert len(transitions) >= 2

        start_latents = torch.stack([t[0] for t in transitions])
        end_latents = torch.stack([t[1] for t in transitions])
        macro_actions = torch.stack([t[2] for t in transitions])

        n_transitions = len(transitions)
        assert start_latents.shape == (n_transitions, EMBED_DIM)
        assert macro_actions.shape == (n_transitions, ACTION_DIM)

        encoder = _make_mock_encoder()
        hl_model = train_high_level_model(
            encoder=encoder,
            start_latents=start_latents,
            macro_actions=macro_actions,
            end_latents=end_latents,
            latent_dim=EMBED_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=32,
            lr=1e-3,
            epochs=50,
        )
        assert isinstance(hl_model, HighLevelWorldModel)

        test_z = start_latents[:1]
        test_a = macro_actions[:1]
        predicted = hl_model.predict(test_z, test_a)
        assert predicted.shape == (1, EMBED_DIM)

        hl_config = HighLevelPlannerConfig(
            macro_horizon=5,
            macro_action_dim=ACTION_DIM,
            population_size=16,
            elite_frac=0.25,
            n_iterations=3,
        )
        planner = HighLevelPlanner(
            high_level_model=hl_model,
            encoder=encoder,
            config=hl_config,
            device="cpu",
        )

        current_frame = torch.randn(EMBED_DIM)
        goal_frame = torch.randn(EMBED_DIM)
        subgoal_latents, cost = planner.plan_subgoals(current_frame, goal_frame)

        assert subgoal_latents.dim() == 2
        assert subgoal_latents.shape[0] == hl_config.macro_horizon
        assert subgoal_latents.shape[1] == EMBED_DIM
        assert isinstance(cost, float)

        targets = planner.subgoals_to_targets(subgoal_latents)
        assert len(targets) == hl_config.macro_horizon
        for target in targets:
            assert target.shape == (EMBED_DIM,)

    def test_pipeline_with_known_boundaries(self) -> None:
        torch.manual_seed(123)

        latents = torch.zeros(TOTAL_STEPS, EMBED_DIM)
        for p in range(N_PHASES):
            start_idx = p * STEPS_PER_PHASE
            end_idx = (p + 1) * STEPS_PER_PHASE
            latents[start_idx:end_idx] = torch.randn(STEPS_PER_PHASE, EMBED_DIM) * 0.1
            latents[start_idx:end_idx] += p * 5.0

        boundary_indices = [STEPS_PER_PHASE - 1, 2 * STEPS_PER_PHASE - 1]

        class BoundaryModel:
            def __init__(self, latents: torch.Tensor, boundaries: list[int]) -> None:
                self._latents = latents
                self._boundaries = set(boundaries)

            def encode(self, frame: torch.Tensor) -> torch.Tensor:
                n = frame.shape[0]
                return self._latents[:n]

            def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                for i in range(TOTAL_STEPS - 1):
                    if torch.allclose(self._latents[i : i + 1], z, atol=1e-5):
                        if i in self._boundaries:
                            return self._latents[i : i + 1] + 10.0
                        if i + 1 < TOTAL_STEPS:
                            return self._latents[i + 1 : i + 2]
                return z

        model = BoundaryModel(latents, boundary_indices)
        detector_cfg = SubgoalDetectorConfig(
            threshold=1.0,
            smoothing_window=1,
            min_segment_length=5,
        )
        detector = SubgoalDetector(detector_cfg)

        frames_dummy = torch.randn(TOTAL_STEPS, 3, 64, 64)
        actions = torch.randn(TOTAL_STEPS - 1, ACTION_DIM)

        errors = detector.compute_prediction_errors(model, frames_dummy, actions)
        smoothed = detector.smooth_errors(errors)
        change_points = detector.detect_change_points(smoothed)

        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        assert len(transitions) >= 2

        for start, end, macro in transitions:
            assert start.shape == (EMBED_DIM,)
            assert end.shape == (EMBED_DIM,)
            assert macro.shape == (ACTION_DIM,)
