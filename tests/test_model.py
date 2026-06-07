from __future__ import annotations

import torch
import pytest

from wally.models.encoder import ViTEncoder
from wally.models.action_embedder import ActionEmbedder
from wally.models.predictor import CausalTransformerPredictor
from wally.models.lewm import LeWorldModel
from wally.config.model import ModelConfig


class TestViTEncoder:
    def test_output_shape(self):
        encoder = ViTEncoder(pretrained=False)
        frames = torch.randn(4, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (4, 196, 192)

    def test_single_frame(self):
        encoder = ViTEncoder(pretrained=False)
        frames = torch.randn(1, 3, 224, 224)
        out = encoder(frames)
        assert out.shape == (1, 196, 192)

    def test_embed_dim_attribute(self):
        encoder = ViTEncoder(pretrained=False)
        assert encoder.embed_dim == 192

    def test_output_is_float(self):
        encoder = ViTEncoder(pretrained=False)
        frames = torch.randn(2, 3, 224, 224)
        out = encoder(frames)
        assert out.dtype == torch.float32


class TestActionEmbedder:
    def test_output_shape(self):
        embedder = ActionEmbedder(action_dim=25, embed_dim=192)
        actions = torch.randn(4, 16, 25)
        out = embedder(actions)
        assert out.shape == (4, 16, 192)

    def test_different_dims(self):
        embedder = ActionEmbedder(action_dim=10, embed_dim=64)
        actions = torch.randn(2, 8, 10)
        out = embedder(actions)
        assert out.shape == (2, 8, 64)

    def test_gradient_flows(self):
        embedder = ActionEmbedder(action_dim=25, embed_dim=192)
        actions = torch.randn(2, 4, 25, requires_grad=True)
        out = embedder(actions)
        out.sum().backward()
        assert actions.grad is not None


class TestCausalTransformerPredictor:
    def test_output_shape(self):
        predictor = CausalTransformerPredictor(embed_dim=192, depth=2, num_heads=4)
        x = torch.randn(4, 32, 192)  # 2*T interleaved
        out = predictor(x)
        assert out.shape == (4, 16, 192)  # even positions = T predictions

    def test_causal_masking(self):
        predictor = CausalTransformerPredictor(embed_dim=64, depth=1, num_heads=4)
        x = torch.randn(1, 10, 64)
        out = predictor(x)
        assert out.shape == (1, 5, 64)

    def test_gradient_flows(self):
        predictor = CausalTransformerPredictor(embed_dim=64, depth=1, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = predictor(x)
        out.sum().backward()
        assert x.grad is not None


class TestLeWorldModel:
    def test_forward_output_shapes(self):
        model = LeWorldModel(
            vit_variant="vit_tiny_patch16_224",
            embed_dim=192,
            depth=2,
            num_heads=4,
            pretrained=False,
        )
        frames = torch.randn(2, 4, 3, 224, 224)
        actions = torch.randn(2, 4, 25)
        predicted, target = model(frames, actions)
        assert predicted.shape == (2, 3, 192)
        assert target.shape == (2, 3, 192)

    def test_single_batch(self):
        model = LeWorldModel(
            vit_variant="vit_tiny_patch16_224",
            embed_dim=192,
            depth=2,
            num_heads=4,
            pretrained=False,
        )
        frames = torch.randn(1, 8, 3, 224, 224)
        actions = torch.randn(1, 8, 25)
        predicted, target = model(frames, actions)
        assert predicted.shape == (1, 7, 192)
        assert target.shape == (1, 7, 192)

    def test_gradient_flows(self):
        model = LeWorldModel(
            vit_variant="vit_tiny_patch16_224",
            embed_dim=192,
            depth=2,
            num_heads=4,
            pretrained=False,
        )
        frames = torch.randn(1, 3, 3, 224, 224)
        actions = torch.randn(1, 3, 25)
        predicted, target = model(frames, actions)
        loss = (predicted - target).pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_model_config_integration(self):
        config = ModelConfig(depth=2, num_heads=4, pretrained=False)
        model = LeWorldModel(
            vit_variant=config.vit_variant,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            action_dim=config.action_dim,
            pretrained=config.pretrained,
        )
        frames = torch.randn(1, 3, 3, 224, 224)
        actions = torch.randn(1, 3, config.action_dim)
        predicted, target = model(frames, actions)
        assert predicted.shape == (1, 2, config.embed_dim)


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.vit_variant == "vit_tiny_patch16_224"
        assert config.embed_dim == 192
        assert config.depth == 6
        assert config.num_heads == 4
        assert config.mlp_ratio == 4.0
        assert config.dropout == 0.1
        assert config.action_dim == 25
        assert config.pretrained is True

    def test_custom_values(self):
        config = ModelConfig(depth=4, num_heads=8, embed_dim=256)
        assert config.depth == 4
        assert config.num_heads == 8
        assert config.embed_dim == 256
