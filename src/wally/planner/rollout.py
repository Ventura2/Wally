from __future__ import annotations

from pathlib import Path

import torch

from wally.models.lewm import LeWorldModel
from wally.planner.protocols import WorldModelProtocol


class ModelNotLoadedError(RuntimeError):
    """Raised when rollout is attempted without a loaded model."""


def _infer_encoder_type(state_dict: dict[str, object]) -> str:
    """Best-effort encoder type detection from a raw state dict.

    Used as a fallback when the checkpoint predates the embedded
    ``model_config`` and so does not declare an encoder type explicitly.
    CNN checkpoints expose ``encoder.conv1.weight``; ViT checkpoints
    expose ``encoder.backbone.cls_token``. Anything else is treated as
    ViT to preserve the historical default.
    """
    if any(k.startswith("encoder.conv1") for k in state_dict):
        return "cnn"
    if any(k.startswith("encoder.backbone") for k in state_dict):
        return "vit"
    return "vit"


class LeWorldModelAdapter:
    """Adapts a LeWorldModel to the WorldModelProtocol used by the planner.

    ``predict`` returns the **predicted change** Δ (frame-to-frame delta in
    latent space), matching the new model contract. The next latent is
    reconstructed as ``z + Δ`` in ``LatentRollout.rollout``.
    """

    def __init__(self, model: LeWorldModel) -> None:
        self._model = model
        self._is_cnn = bool(getattr(model, "_is_cnn", False))

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        out = self._model.encoder(frame)
        if self._is_cnn:
            return out
        return out.mean(dim=1)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z_seq = z.unsqueeze(1)
        a_emb = self._model.action_embedder(action.unsqueeze(1))
        pred_emb = self._model.predictor(z_seq, a_emb)
        predicted_change = self._model.pred_proj(pred_emb)
        return predicted_change.squeeze(1)

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
        model_config: dict[str, object] | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        elif checkpoint_path is not None:
            self._model = self._load_from_checkpoint(
                checkpoint_path, device=device, model_config=model_config,
            )
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
        model_config: dict[str, object] | None = None,
    ) -> LatentRollout:
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            model_config=model_config,
        )

    def _load_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        model_config: dict[str, object] | None = None,
    ) -> LeWorldModelAdapter:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if model_config is not None:
            model_cfg = dict(model_config)
        else:
            model_cfg = dict(
                checkpoint.get("model_config")
                or checkpoint.get("config", {}).get("model", {})
                or {}
            )
        if "encoder_type" not in model_cfg:
            model_cfg["encoder_type"] = _infer_encoder_type(
                checkpoint.get("model_state_dict", {}),
            )
        model = LeWorldModel(
            vit_variant=model_cfg.get("vit_variant", "vit_tiny_patch16_224"),
            embed_dim=model_cfg.get("embed_dim", 192),
            depth=model_cfg.get("depth", 6),
            num_heads=model_cfg.get("num_heads", 4),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            dropout=model_cfg.get("dropout", 0.1),
            action_dim=model_cfg.get("action_dim", 25),
            encoder_type=model_cfg.get("encoder_type", "vit"),
            num_frames=model_cfg.get("num_frames", 16),
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
        """Roll out a trajectory in latent space.

        The model is expected to return a **predicted change** Δ (per the
        residual-loss contract); the next latent is reconstructed as
        ``z_{t+1} = z_t + Δ``. Gradient policy (``detach`` vs
        ``straight_through``) is applied to the next latent after the
        add, so subsequent steps do not backprop through earlier ones.
        """
        B, H, _ = actions.shape
        latents = [z_0]
        z = z_0
        for h in range(H):
            a_h = actions[:, h, :]
            delta = self._model.predict(z, a_h)
            z_next = z + delta
            if self._gradient_policy == "detach":
                z_next = z_next.detach()
            latents.append(z_next)
            z = z_next
        return torch.stack(latents, dim=1)
