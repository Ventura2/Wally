from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from wally.models.recurrent_encoder import RecurrentEncoder


def _make_mock_vit(embed_dim: int = 64, num_patches: int = 16) -> MagicMock:
    mock_vit = MagicMock(spec=nn.Module)
    mock_vit.embed_dim = embed_dim
    mock_vit.num_patches = num_patches

    def _forward(frames: torch.Tensor) -> torch.Tensor:
        B = frames.shape[0]
        return torch.randn(B, num_patches, embed_dim)

    mock_vit.side_effect = _forward
    mock_vit.parameters = lambda: iter([])
    return mock_vit


def _build_encoder(
    embed_dim: int = 64,
    num_patches: int = 16,
    hidden_size: int | None = None,
    recurrence: bool = True,
    memory_length: int = 16,
) -> RecurrentEncoder:
    if hidden_size is None:
        hidden_size = embed_dim
    encoder = RecurrentEncoder.__new__(RecurrentEncoder)
    nn.Module.__init__(encoder)
    encoder.vit_encoder = _make_mock_vit(embed_dim, num_patches)
    encoder.lstm = nn.LSTM(
        input_size=embed_dim,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    )
    encoder.output_proj = nn.Linear(hidden_size, embed_dim)
    encoder.hidden_size = hidden_size
    encoder.memory_length = memory_length
    encoder.recurrence = recurrence
    encoder._hidden = None
    return encoder


class TestRecurrentEncoderForward:
    def test_single_frame_output_shape(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (2, 64)

    def test_single_batch_single_frame(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(1, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (1, 64)

    def test_output_dtype_is_float(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 3, 224, 224)
        out = encoder(frames)
        assert out.dtype == torch.float32


class TestRecurrentEncoderSequence:
    def test_sequence_output_shape(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 5, 3, 224, 224)
        latents, hidden = encoder.forward_sequence(frames)
        assert latents.shape == (2, 5, 64)
        assert hidden is not None
        assert hidden[0].shape == (1, 2, 64)
        assert hidden[1].shape == (1, 2, 64)

    def test_sequence_single_timestep(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(1, 1, 3, 224, 224)
        latents, hidden = encoder.forward_sequence(frames)
        assert latents.shape == (1, 1, 64)
        assert hidden is not None

    def test_sequence_returns_final_hidden(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(3, 8, 3, 224, 224)
        _, hidden = encoder.forward_sequence(frames)
        assert hidden is not None
        h, c = hidden
        assert h.shape[1] == 3
        assert c.shape[1] == 3


class TestHiddenStateManagement:
    def test_hidden_state_persists_across_calls(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 3, 224, 224)
        encoder(frames)
        h1 = encoder.get_hidden()
        assert h1 is not None

        encoder(frames)
        h2 = encoder.get_hidden()
        assert h2 is not None
        assert not torch.equal(h1[0], h2[0])

    def test_reset_hidden_clears_state(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 3, 224, 224)
        encoder(frames)
        assert encoder.get_hidden() is not None

        encoder.reset_hidden()
        assert encoder.get_hidden() is None

    def test_get_hidden_returns_none_initially(self):
        encoder = _build_encoder(embed_dim=64)
        assert encoder.get_hidden() is None

    def test_set_hidden_restores_state(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(2, 3, 224, 224)
        encoder(frames)
        saved = encoder.get_hidden()
        assert saved is not None

        encoder.reset_hidden()
        assert encoder.get_hidden() is None

        encoder.set_hidden(saved)
        restored = encoder.get_hidden()
        assert restored is not None
        assert torch.equal(saved[0], restored[0])
        assert torch.equal(saved[1], restored[1])

    def test_auto_reset_on_none_hidden(self):
        encoder = _build_encoder(embed_dim=64)
        assert encoder._hidden is None
        frames = torch.randn(1, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (1, 64)
        assert encoder._hidden is not None


class TestDropInCompatibility:
    def test_embed_dim_matches_vit(self):
        encoder = _build_encoder(embed_dim=64)
        assert encoder.embed_dim == 64

    def test_output_shape_matches_vit_mean_pool(self):
        encoder = _build_encoder(embed_dim=64)
        frames = torch.randn(4, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (4, 64)


class TestRecurrenceBypass:
    def test_bypass_returns_pooled_without_lstm(self):
        encoder = _build_encoder(embed_dim=64, recurrence=False)
        frames = torch.randn(2, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (2, 64)
        assert encoder.get_hidden() is None

    def test_bypass_sequence_returns_none_hidden(self):
        encoder = _build_encoder(embed_dim=64, recurrence=False)
        frames = torch.randn(2, 5, 3, 224, 224)
        latents, hidden = encoder.forward_sequence(frames)
        assert latents.shape == (2, 5, 64)
        assert hidden is None

    def test_bypass_multiple_calls_no_state(self):
        encoder = _build_encoder(embed_dim=64, recurrence=False)
        frames = torch.randn(1, 3, 224, 224)
        encoder(frames)
        encoder(frames)
        assert encoder.get_hidden() is None


class TestDifferentHiddenSize:
    def test_different_hidden_size_forward(self):
        encoder = _build_encoder(embed_dim=64, hidden_size=128)
        frames = torch.randn(2, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (2, 64)

    def test_different_hidden_size_sequence(self):
        encoder = _build_encoder(embed_dim=64, hidden_size=128)
        frames = torch.randn(2, 4, 3, 224, 224)
        latents, hidden = encoder.forward_sequence(frames)
        assert latents.shape == (2, 4, 64)
        assert hidden[0].shape == (1, 2, 128)

    def test_smaller_hidden_size(self):
        encoder = _build_encoder(embed_dim=64, hidden_size=32)
        frames = torch.randn(1, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (1, 64)
