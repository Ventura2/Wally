from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from wally.planner.protocols import WorldModelProtocol
from wally.planner.subgoal_detector import SubgoalDetector, SubgoalDetectorConfig


class TestSubgoalDetectorConfigDefaults:
    def test_default_values(self):
        cfg = SubgoalDetectorConfig.default()
        assert cfg.threshold == 1.0
        assert cfg.smoothing_window == 5
        assert cfg.min_segment_length == 8

    def test_constructor_defaults(self):
        cfg = SubgoalDetectorConfig()
        assert cfg.threshold == 1.0
        assert cfg.smoothing_window == 5
        assert cfg.min_segment_length == 8


class TestSubgoalDetectorConfigValidation:
    def test_threshold_zero_fails(self):
        with pytest.raises(Exception, match="threshold"):
            SubgoalDetectorConfig(threshold=0.0)

    def test_threshold_negative_fails(self):
        with pytest.raises(Exception, match="threshold"):
            SubgoalDetectorConfig(threshold=-0.5)

    def test_smoothing_window_zero_fails(self):
        with pytest.raises(Exception, match="smoothing_window"):
            SubgoalDetectorConfig(smoothing_window=0)

    def test_min_segment_length_one_fails(self):
        with pytest.raises(Exception, match="min_segment_length"):
            SubgoalDetectorConfig(min_segment_length=1)

    def test_valid_custom_values(self):
        cfg = SubgoalDetectorConfig(
            threshold=0.5, smoothing_window=3, min_segment_length=4,
        )
        assert cfg.threshold == 0.5
        assert cfg.smoothing_window == 3
        assert cfg.min_segment_length == 4


class TestComputePredictionErrors:
    def _make_mock_model(self, z_dim: int = 8) -> MagicMock:
        model = MagicMock(spec=WorldModelProtocol)

        def encode_fn(frames: torch.Tensor) -> torch.Tensor:
            if frames.dim() == 4:
                B = frames.shape[0]
                return torch.randn(B, z_dim)
            return torch.randn(frames.shape[0], z_dim)

        def predict_fn(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(z)

        model.encode.side_effect = encode_fn
        model.predict.side_effect = predict_fn
        return model

    def test_1d_output_shape(self):
        detector = SubgoalDetector()
        model = self._make_mock_model()
        T, C, H, W = 10, 3, 224, 224
        A = 25
        frames = torch.randn(T, C, H, W)
        actions = torch.randn(T - 1, A)
        errors = detector.compute_prediction_errors(model, frames, actions)
        assert errors.shape == (T - 1,)

    def test_2d_output_shape(self):
        detector = SubgoalDetector()
        model = self._make_mock_model()
        B, T, C, H, W = 2, 10, 3, 224, 224
        A = 25
        frames = torch.randn(B, T, C, H, W)
        actions = torch.randn(B, T - 1, A)
        errors = detector.compute_prediction_errors(model, frames, actions)
        assert errors.shape == (B, T - 1)

    def test_errors_are_nonnegative(self):
        detector = SubgoalDetector()
        model = self._make_mock_model()
        T, C, H, W = 8, 3, 224, 224
        A = 25
        frames = torch.randn(T, C, H, W)
        actions = torch.randn(T - 1, A)
        errors = detector.compute_prediction_errors(model, frames, actions)
        assert (errors >= 0).all()

    def test_known_l2_distance(self):
        class FixedModel:
            def __init__(self) -> None:
                self._latents = torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                )

            def encode(self, frame: torch.Tensor) -> torch.Tensor:
                T = frame.shape[0]
                return self._latents[:T]

            def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(z)

        detector = SubgoalDetector()
        model = FixedModel()
        frames = torch.randn(3, 3, 224, 224)
        actions = torch.randn(2, 25)
        errors = detector.compute_prediction_errors(model, frames, actions)
        expected_0 = torch.norm(torch.tensor([1.0, 0.0, 0.0, 0.0]), p=2)
        expected_1 = torch.norm(torch.tensor([1.0, 1.0, 0.0, 0.0]), p=2)
        assert torch.allclose(errors[0], expected_0, atol=1e-6)
        assert torch.allclose(errors[1], expected_1, atol=1e-6)


class TestSmoothErrors:
    def test_1d_smoothing(self):
        cfg = SubgoalDetectorConfig(smoothing_window=3)
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0])
        smoothed = detector.smooth_errors(errors)
        assert smoothed.shape == errors.shape
        assert smoothed[2] > smoothed[0]

    def test_window_1_preserves(self):
        cfg = SubgoalDetectorConfig(smoothing_window=1)
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
        smoothed = detector.smooth_errors(errors)
        assert torch.allclose(smoothed, errors)

    def test_2d_smoothing(self):
        cfg = SubgoalDetectorConfig(smoothing_window=3)
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0, 0.0]])
        smoothed = detector.smooth_errors(errors)
        assert smoothed.shape == errors.shape

    def test_uniform_region_unchanged(self):
        cfg = SubgoalDetectorConfig(smoothing_window=3)
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        smoothed = detector.smooth_errors(errors)
        assert torch.allclose(smoothed, errors, atol=1e-5)


class TestDetectChangePoints:
    def test_known_peaks(self):
        cfg = SubgoalDetectorConfig(
            threshold=1.0, smoothing_window=1, min_segment_length=2,
        )
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.1, 0.2, 5.0, 0.1, 0.2, 0.1, 4.0, 0.2, 0.1])
        cps = detector.detect_change_points(errors)
        assert 2 in cps
        assert 6 in cps

    def test_below_threshold_ignored(self):
        cfg = SubgoalDetectorConfig(
            threshold=10.0, smoothing_window=1, min_segment_length=2,
        )
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.1, 0.2, 5.0, 0.1, 0.2])
        cps = detector.detect_change_points(errors)
        assert len(cps) == 0

    def test_min_segment_length_enforcement(self):
        cfg = SubgoalDetectorConfig(
            threshold=1.0, smoothing_window=1, min_segment_length=5,
        )
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.1, 3.0, 0.1, 0.1, 0.1, 4.0, 0.1, 0.1, 0.1, 0.1])
        cps = detector.detect_change_points(errors)
        if len(cps) >= 2:
            for i in range(len(cps) - 1):
                assert cps[i + 1] - cps[i] >= 5

    def test_empty_for_flat_signal(self):
        cfg = SubgoalDetectorConfig(
            threshold=1.0, smoothing_window=1, min_segment_length=2,
        )
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        cps = detector.detect_change_points(errors)
        assert len(cps) == 0

    def test_higher_peak_survives(self):
        cfg = SubgoalDetectorConfig(
            threshold=1.0, smoothing_window=1, min_segment_length=3,
        )
        detector = SubgoalDetector(cfg)
        errors = torch.tensor([0.1, 2.0, 0.1, 0.1, 5.0, 0.1, 0.1])
        cps = detector.detect_change_points(errors)
        assert 4 in cps


class TestExtractAbstractTransitions:
    def test_basic_segmentation(self):
        detector = SubgoalDetector()
        T, Z, A = 10, 4, 6
        latents = torch.randn(T, Z)
        actions = torch.randn(T - 1, A)
        change_points = [3, 7]
        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        assert len(transitions) == 3

    def test_start_end_latents(self):
        detector = SubgoalDetector()
        T, Z, A = 10, 4, 6
        latents = torch.randn(T, Z)
        actions = torch.randn(T - 1, A)
        change_points = [5]
        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        assert len(transitions) == 2
        start0, end0, _ = transitions[0]
        assert torch.equal(start0, latents[0])
        assert torch.equal(end0, latents[5])
        start1, end1, _ = transitions[1]
        assert torch.equal(start1, latents[6])
        assert torch.equal(end1, latents[9])

    def test_macro_action_mean_pooling(self):
        detector = SubgoalDetector()
        T, Z = 6, 4
        latents = torch.randn(T, Z)
        actions = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        )
        change_points = [3]
        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        _, _, macro0 = transitions[0]
        expected0 = actions[:3].float().mean(dim=0)
        assert torch.allclose(macro0, expected0)
        _, _, macro1 = transitions[1]
        expected1 = actions[3:5].float().mean(dim=0)
        assert torch.allclose(macro1, expected1)

    def test_no_change_points_single_segment(self):
        detector = SubgoalDetector()
        T, Z, A = 5, 4, 2
        latents = torch.randn(T, Z)
        actions = torch.randn(T - 1, A)
        transitions = detector.extract_abstract_transitions(latents, actions, [])
        assert len(transitions) == 1
        start, end, macro = transitions[0]
        assert torch.equal(start, latents[0])
        assert torch.equal(end, latents[T - 1])
        assert torch.allclose(macro, actions.float().mean(dim=0))


class TestFullPipeline:
    def test_end_to_end_synthetic(self):
        z_dim = 8
        T = 30

        class SyntheticModel:
            def __init__(self) -> None:
                torch.manual_seed(42)
                self._latents = torch.randn(T, z_dim)
                for t in range(10, 12):
                    self._latents[t] += 10.0
                for t in range(20, 22):
                    self._latents[t] += 10.0

            def encode(self, frame: torch.Tensor) -> torch.Tensor:
                n = frame.shape[0]
                return self._latents[:n]

            def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                idx = 0
                for i in range(T):
                    if torch.equal(self._latents[i : i + 1], z):
                        idx = i
                        break
                if idx + 1 < T:
                    return self._latents[idx + 1 : idx + 2]
                return z

        cfg = SubgoalDetectorConfig(
            threshold=1.0, smoothing_window=3, min_segment_length=3,
        )
        detector = SubgoalDetector(cfg)
        model = SyntheticModel()

        frames = torch.randn(T, 3, 224, 224)
        actions = torch.randn(T - 1, 25)

        errors = detector.compute_prediction_errors(model, frames, actions)
        assert errors.shape == (T - 1,)

        smoothed = detector.smooth_errors(errors)
        assert smoothed.shape == errors.shape

        change_points = detector.detect_change_points(smoothed)
        assert isinstance(change_points, list)

        latents = model._latents
        transitions = detector.extract_abstract_transitions(
            latents, actions, change_points,
        )
        for start, end, macro in transitions:
            assert start.shape == (z_dim,)
            assert end.shape == (z_dim,)
            assert macro.shape == (25,)
