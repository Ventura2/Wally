from __future__ import annotations

import pytest
import torch
import yaml

from wally.planner.actions import (
    ActionDimension,
    MineStudioActionVocab,
    continuous_to_discrete,
    discrete_to_continuous,
)


def _small_vocab() -> MineStudioActionVocab:
    return MineStudioActionVocab(
        dimensions=[
            ActionDimension(name="pitch", low=-1.0, high=1.0, bins=11),
            ActionDimension(name="forward", low=0.0, high=1.0, bins=2),
            ActionDimension(name="jump", low=0.0, high=1.0, bins=2),
        ]
    )


class TestMineStudioActionVocab:
    def test_default_has_25_dims(self):
        vocab = MineStudioActionVocab.default()
        assert vocab.action_dim == 25
        names = [d.name for d in vocab.dimensions]
        assert "camera_pitch" in names
        assert "forward" in names
        assert "hotbar_9" in names

    def test_from_yaml(self, tmp_path):
        cfg = {
            "dimensions": [
                {"name": "x", "low": -2.0, "high": 2.0, "bins": 5},
                {"name": "y", "low": 0.0, "high": 1.0, "bins": 3},
            ]
        }
        p = tmp_path / "vocab.yaml"
        p.write_text(yaml.dump(cfg))
        vocab = MineStudioActionVocab.from_yaml(p)
        assert vocab.action_dim == 2
        assert vocab.dimensions[0].name == "x"
        assert vocab.dimensions[0].bins == 5
        assert vocab.dimensions[1].high == 1.0


class TestContinuousToDiscrete:
    def test_basic_quantization(self):
        vocab = _small_vocab()
        acts = torch.tensor([[0.0, 0.5, 0.5]])
        result = continuous_to_discrete(acts, vocab)
        assert len(result) == 1
        assert result[0]["pitch"] == 5
        assert result[0]["forward"] == 1
        assert result[0]["jump"] == 1

    def test_boundary_values(self):
        vocab = _small_vocab()
        acts = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = continuous_to_discrete(acts, vocab)
        assert result[0]["pitch"] == 0
        assert result[1]["pitch"] == 10
        assert result[0]["forward"] == 0
        assert result[1]["forward"] == 1

    def test_out_of_bounds_raises(self):
        vocab = _small_vocab()
        acts = torch.tensor([[0.0, 1.5, 0.5]])
        with pytest.raises(ValueError, match="out of bounds.*timestep 0.*index 1"):
            continuous_to_discrete(acts, vocab)

    def test_negative_out_of_bounds(self):
        vocab = _small_vocab()
        acts = torch.tensor([[-1.1, 0.5, 0.5]])
        with pytest.raises(ValueError, match="out of bounds"):
            continuous_to_discrete(acts, vocab)

    def test_wrong_shape_raises(self):
        vocab = _small_vocab()
        with pytest.raises(ValueError, match="2-D"):
            continuous_to_discrete(torch.zeros(2, 3, 4), vocab)

    def test_wrong_action_dim_raises(self):
        vocab = _small_vocab()
        with pytest.raises(ValueError, match="Action dim mismatch"):
            continuous_to_discrete(torch.zeros(2, 5), vocab)

    def test_multiple_timesteps(self):
        vocab = _small_vocab()
        acts = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ])
        result = continuous_to_discrete(acts, vocab)
        assert len(result) == 3


class TestDiscreteToContinuous:
    def test_basic_dequantization(self):
        vocab = _small_vocab()
        acts = [{"pitch": 5, "forward": 1, "jump": 0}]
        result = discrete_to_continuous(acts, vocab)
        assert result.shape == (1, 3)
        expected_pitch = -1.0 + (5 + 0.5) * 2.0 / 11
        assert abs(result[0, 0].item() - expected_pitch) < 1e-6
        expected_forward = 0.0 + (1 + 0.5) * 1.0 / 2
        assert abs(result[0, 1].item() - expected_forward) < 1e-6

    def test_empty_input(self):
        vocab = _small_vocab()
        result = discrete_to_continuous([], vocab)
        assert result.shape == (0, 3)

    def test_missing_dim_raises(self):
        vocab = _small_vocab()
        with pytest.raises(ValueError, match="Missing dimension"):
            discrete_to_continuous([{"pitch": 5, "forward": 1}], vocab)

    def test_bin_out_of_range_raises(self):
        vocab = _small_vocab()
        with pytest.raises(ValueError, match="Bin index out of range"):
            discrete_to_continuous([{"pitch": 11, "forward": 0, "jump": 0}], vocab)


class TestRoundTrip:
    def test_round_trip_within_bin_width(self):
        vocab = _small_vocab()
        torch.manual_seed(42)
        raw = torch.rand(20, 3)
        raw[:, 0] = raw[:, 0] * 2.0 - 1.0

        discrete = continuous_to_discrete(raw, vocab)
        recovered = discrete_to_continuous(discrete, vocab)

        for j, dim in enumerate(vocab.dimensions):
            bin_width = (dim.high - dim.low) / dim.bins
            diffs = (raw[:, j] - recovered[:, j]).abs()
            assert (diffs <= bin_width / 2 + 1e-6).all(), (
                f"Dim '{dim.name}' exceeded bin width"
            )

    def test_round_trip_exact_for_bin_centers(self):
        vocab = _small_vocab()
        centers = torch.tensor([
            [
                -1.0 + (5 + 0.5) * 2.0 / 11,
                0.0 + (0 + 0.5) * 1.0 / 2,
                0.0 + (1 + 0.5) * 1.0 / 2,
            ],
        ])
        discrete = continuous_to_discrete(centers, vocab)
        assert discrete[0]["pitch"] == 5
        assert discrete[0]["forward"] == 0
        assert discrete[0]["jump"] == 1
        recovered = discrete_to_continuous(discrete, vocab)
        assert torch.allclose(centers, recovered, atol=1e-6)
