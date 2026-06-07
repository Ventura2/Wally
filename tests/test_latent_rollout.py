from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from wally.planner.protocols import WorldModelProtocol
from wally.planner.rollout import (
    LatentRollout,
    LeWorldModelAdapter,
    ModelNotLoadedError,
)


class _DummyModel:
    def __init__(self, z_dim: int = 8, a_dim: int = 4) -> None:
        self.z_dim = z_dim
        self.a_dim = a_dim

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.zeros(frame.size(0), self.z_dim)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + action.sum(dim=-1, keepdim=True).expand_as(z)


class TestLatentRolloutBasic:
    def test_initial_latent_preserved(self) -> None:
        model = _DummyModel()
        rollout = LatentRollout(model=model)
        z_0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        result = rollout.rollout(z_0, actions)
        assert torch.equal(result[:, 0, :], z_0)

    def test_output_shape(self) -> None:
        model = _DummyModel()
        rollout = LatentRollout(model=model)
        B, H, Z = 3, 5, 8
        z_0 = torch.randn(B, Z)
        actions = torch.randn(B, H, 4)
        result = rollout.rollout(z_0, actions)
        assert result.shape == (B, H + 1, Z)

    def test_detach_blocks_gradients(self) -> None:
        model = _DummyModel()
        rollout = LatentRollout(model=model, gradient_policy="detach")
        z_0 = torch.randn(2, 8, requires_grad=True)
        actions = torch.randn(2, 3, 4)
        result = rollout.rollout(z_0, actions)
        result[:, 1, :].sum().backward()
        assert z_0.grad is not None
        assert torch.all(z_0.grad == 0)

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ModelNotLoadedError):
            LatentRollout()


class TestLeWorldModelAdapter:
    def test_adapter_satisfies_protocol(self) -> None:
        lewm = MagicMock()
        adapter = LeWorldModelAdapter(lewm)
        assert isinstance(adapter, WorldModelProtocol)

    def test_encode_mean_pools(self) -> None:
        lewm = MagicMock()
        tokens = torch.randn(2, 196, 192)
        lewm.encoder.return_value = tokens
        adapter = LeWorldModelAdapter(lewm)
        out = adapter.encode(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 192)
        expected = tokens.mean(dim=1)
        assert torch.allclose(out, expected)

    def test_predict_single_step(self) -> None:
        lewm = MagicMock()
        embed_dim = 192
        z = torch.randn(2, embed_dim)
        action = torch.randn(2, 25)
        predicted = torch.randn(2, 1, embed_dim)
        lewm.action_embedder.return_value = torch.randn(2, 1, embed_dim)
        lewm.predictor.return_value = predicted
        adapter = LeWorldModelAdapter(lewm)
        out = adapter.predict(z, action)
        assert out.shape == (2, embed_dim)


class TestFromCheckpoint:
    def test_parameters_frozen(self, tmp_path) -> None:
        lewm = MagicMock()
        state_dict = {}
        lewm.state_dict.return_value = state_dict
        lewm.parameters = [torch.nn.Parameter(torch.zeros(1))]

        checkpoint = {
            "model_state_dict": state_dict,
            "config": {"model": {"embed_dim": 8, "depth": 1, "num_heads": 2}},
            "global_step": 0,
        }
        ckpt_path = tmp_path / "model.pt"
        torch.save(checkpoint, ckpt_path)

        with patch(
            "wally.planner.rollout.LeWorldModel",
        ) as MockLewm:
            mock_instance = MagicMock()
            mock_instance.parameters.return_value = iter(
                [torch.nn.Parameter(torch.zeros(1))]
            )
            MockLewm.return_value = mock_instance

            LatentRollout.from_checkpoint(ckpt_path)

        for p in mock_instance.parameters():
            assert not p.requires_grad
