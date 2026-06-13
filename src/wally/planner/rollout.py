from __future__ import annotations

from pathlib import Path

import torch

from wally.models.lewm import LeWorldModel
from wally.planner.protocols import WorldModelProtocol


class ModelNotLoadedError(RuntimeError):
    """Raised when rollout is attempted without a loaded model."""


class LeWorldModelAdapter:
    def __init__(self, model: LeWorldModel) -> None:
        self._model = model

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        tokens = self._model.encoder(frame)
        return tokens.mean(dim=1)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z_seq = z.unsqueeze(1)
        a_emb = self._model.action_embedder(action.unsqueeze(1))
        predicted = self._model.predictor(z_seq, a_emb)
        return predicted.squeeze(1)

    @property
    def parameters(self):
        return self._model.parameters


class LatentRollout:
    def __init__(
        self,
        model: WorldModelProtocol | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        gradient_policy: str = "detach",
    ) -> None:
        if model is not None:
            self._model = model
        elif checkpoint_path is not None:
            self._model = self._load_from_checkpoint(checkpoint_path, device=device)
        else:
            raise ModelNotLoadedError(
                "Either a model or checkpoint_path must be provided."
            )
        self._device = torch.device(device) if device is not None else None
        self._gradient_policy = gradient_policy

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> LatentRollout:
        return cls(checkpoint_path=checkpoint_path, device=device)

    def _load_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> LeWorldModelAdapter:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = checkpoint.get("config", {}).get("model", {})
        model = LeWorldModel(
            vit_variant=config.get("vit_variant", "vit_tiny_patch16_224"),
            embed_dim=config.get("embed_dim", 192),
            depth=config.get("depth", 6),
            num_heads=config.get("num_heads", 4),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.1),
            action_dim=config.get("action_dim", 25),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if device is not None:
            model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return LeWorldModelAdapter(model)

    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, H, _ = actions.shape
        latents = [z_0]
        z = z_0
        for h in range(H):
            a_h = actions[:, h, :]
            z_next = self._model.predict(z, a_h)
            if self._gradient_policy == "detach":
                z_next = z_next.detach()
            latents.append(z_next)
            z = z_next
        return torch.stack(latents, dim=1)
