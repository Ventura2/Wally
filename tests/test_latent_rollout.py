from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from wally.models.lewm import LeWorldModel
from wally.planner.protocols import WorldModelProtocol
from wally.planner.rollout import (
    LatentRollout,
    LeWorldModelAdapter,
    ModelNotLoadedError,
)


class _DummyModel:
    """Dummy world model matching the residual-loss contract.

    ``predict(z, action)`` returns the per-step change Δ (frame-to-frame
    delta in latent space), NOT the absolute next latent. The rollout
    reconstructs the next latent as ``z + Δ``.
    """

    def __init__(self, z_dim: int = 8, a_dim: int = 4) -> None:
        self.z_dim = z_dim
        self.a_dim = a_dim

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.zeros(frame.size(0), self.z_dim)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Returns the change Δ, independent of z, so the rollout's
        # detach policy is the only thing that breaks the autograd chain.
        return action.sum(dim=-1, keepdim=True).expand_as(z)


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
        # With detach policy, the rollout chain is broken after each step,
        # so the gradient of result[:, 1, :] with respect to z_0 must be
        # zero. The test asserts the new contract: predict returns the
        # delta Δ, the rollout does z_next = z + Δ, and detach breaks the
        # chain at z_next.
        result[:, 1, :].sum().backward()
        assert z_0.grad is not None
        assert torch.all(z_0.grad == 0)

    def test_next_latent_is_current_plus_delta(self) -> None:
        """z_{t+1} = z_t + Δ under the residual-loss contract."""
        model = _DummyModel()
        rollout = LatentRollout(model=model)
        z_0 = torch.zeros(1, 4)
        # actions[:, 0, :] = [1, 0, 0, 0] → sum = 1 → Δ = 1
        actions = torch.zeros(1, 1, 4)
        actions[0, 0, 0] = 1.0
        result = rollout.rollout(z_0, actions)
        assert torch.allclose(result[:, 0, :], z_0)
        assert torch.allclose(result[:, 1, :], z_0 + 1.0)

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
        lewm._is_cnn = False
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
        predicted_change = torch.randn(2, 1, embed_dim)
        lewm.action_embedder.return_value = torch.randn(2, 1, embed_dim)
        lewm.predictor.return_value = predicted_change
        lewm.pred_proj.return_value = predicted_change
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


class TestCNNCheckpointLoading:
    def test_cnn_checkpoint_loads_and_rolls_out(self, tmp_path: Path) -> None:
        """A checkpoint saved with encoder_type=cnn must load and run
        LatentRollout.rollout without a state_dict mismatch.

        The previous _load_from_checkpoint defaulted to encoder_type="vit"
        and dropped the encoder_type field, so any cnn checkpoint failed.

        Uses ``action_dim=25`` to match the production wally action
        schema (the rollout's ``_translate_agent_action_to_l0`` permutes
        a 25-dim agent action into the L0's training layout).
        """
        model = LeWorldModel(
            encoder_type="cnn",
            embed_dim=32,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            action_dim=25,
            num_frames=4,
            pretrained=False,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "encoder_type": "cnn",
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "action_dim": 25,
                "num_frames": 4,
            },
            "global_step": 0,
        }
        ckpt_path = tmp_path / "cnn_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        rollout = LatentRollout.from_checkpoint(ckpt_path)

        assert rollout._model._is_cnn is True

        z_0 = torch.randn(2, 32)
        actions = torch.randn(2, 3, 25)
        result = rollout.rollout(z_0, actions)
        assert result.shape == (2, 4, 32)

    def test_pre_adaln_vit_checkpoint_skipped(self) -> None:
        """The pre-AdaLN ViT checkpoints under
        ``checkpoints/_incompatible_pre_adaln/`` use a different model
        architecture (interleaved-input TransformerEncoder) and cannot be
        loaded by the current code — see AGENTS.md "Checkpoint
        compatibility". Skip when no compatible pre-AdaLN ViT checkpoint
        is available.
        """
        pytest.skip("no compatible pre-AdaLN ViT checkpoint available")


class TestLeWorldModelAdapterEncode:
    def test_cnn_encode_returns_2d_latent(self) -> None:
        """LeWorldModelAdapter.encode on a CNN model must return (B, embed_dim)
        without mean-pooling across the embedding axis."""
        model = LeWorldModel(
            encoder_type="cnn",
            embed_dim=64,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            action_dim=25,
            num_frames=4,
            pretrained=False,
        )
        adapter = LeWorldModelAdapter(model)
        frame = torch.randn(2, 3, 224, 224)
        out = adapter.encode(frame)
        assert out.shape == (2, 64)

    def test_vit_encode_mean_pools_over_token_axis(self) -> None:
        """LeWorldModelAdapter.encode on a ViT model must mean-pool over
        dim=1 (tokens) and match the model.encoder(frame).mean(dim=1)
        contract within floating-point tolerance.

        Uses ``img_size=64`` because ``LeWorldModelAdapter.encode``
        resizes the agent's 224x224 frame to 64x64 (the L0's training
        distribution, per ``src/wally/data/dataset.py``) before
        encoding. The expected side applies the same resize so the
        ViT (built for 64x64) can encode it.
        """
        import torch.nn.functional as F

        model = LeWorldModel(
            encoder_type="vit",
            embed_dim=192,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            action_dim=25,
            num_frames=4,
            pretrained=False,
            img_size=64,
        )
        adapter = LeWorldModelAdapter(model)
        frame = torch.randn(2, 3, 224, 224)
        out = adapter.encode(frame)
        assert out.shape == (2, 192)
        resized = F.interpolate(
            frame, size=(64, 64), mode="bilinear", align_corners=False
        )
        expected = model.encoder(resized).mean(dim=1)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_cnn_encode_feeds_predict_without_shape_mismatch(self) -> None:
        """encode → predict round-trip on a CNN model must not raise and
        must return (B, embed_dim)."""
        model = LeWorldModel(
            encoder_type="cnn",
            embed_dim=64,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            action_dim=25,
            num_frames=4,
            pretrained=False,
        )
        adapter = LeWorldModelAdapter(model)
        frame = torch.randn(2, 3, 224, 224)
        z = adapter.encode(frame)
        action = torch.randn(2, 25)
        delta = adapter.predict(z, action)
        assert delta.shape == (2, 64)
