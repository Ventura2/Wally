from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

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


def _infer_depth(state_dict: dict[str, object]) -> int:
    """Infer predictor transformer depth from a raw state dict.

    Used as a fallback when the checkpoint predates the embedded
    ``model_config``. Returns the max layer index + 1 (0-indexed), or
    6 (the historical default) if no transformer layers are present.
    """
    layer_indices: set[int] = set()
    for k in state_dict:
        if "predictor.transformer.layers." in k:
            try:
                layer_indices.add(int(k.split("predictor.transformer.layers.")[1].split(".")[0]))
            except (ValueError, IndexError):
                continue
    if not layer_indices:
        return 6
    return max(layer_indices) + 1


def _infer_embed_dim(state_dict: dict[str, object]) -> int | None:
    """Infer embed_dim from the encoder's last conv output channel count.

    Walks ``encoder.conv*`` keys in sorted order and returns the largest
    output channel count. This is the encoder's final feature dim and
    matches the predictor's ``input_dim`` for CNN-encoder checkpoints.
    """
    max_out: int | None = None
    for k in state_dict:
        if k.startswith("encoder.conv") and k.endswith(".weight"):
            v = state_dict[k]
            if hasattr(v, "shape") and len(v.shape) == 4:
                out = int(v.shape[0])
                if max_out is None or out > max_out:
                    max_out = out
    return max_out


# The L0 was trained on shards produced by ``configs/converter_default.yaml``,
# which uses the MineStudio "env" action schema:
#   0=forward, 1=backward, 2=left, 3=right, 4=jump, 5=sneak, 6=sprint,
#   7=attack, 8=use, 9=drop, 10=camera_pitch, 11=camera_yaw,
#   12..20=hotbar_1..9, 21=inventory, 22=pickItem, 23=placeItem, 24=craft
# The agent's planner vocab in ``src/wally/planner/actions.py`` is in a
# different order:
#   0=camera_pitch, 1=camera_yaw, 2=forward, 3=back, 4=left, 5=right,
#   6=jump, 7=sneak, 8=sprint, 9=use, 10=attack, 11=drop, 12=inventory,
#   13=swap_hand, 14=pick_block, 15..23=hotbar_1..9, 24=noop
# Plus the agent's camera lives in [-1, 1] but the training data stored
# camera values in raw degrees (which the converter did NOT clamp). Without
# this translation the L0 sees the planner's "camera_pitch at idx 0" as
# "forward button at idx 0" and the predicted next-latent is meaningless.
_AGENT_TO_TRAINING_PERMUTATION: tuple[int, ...] = (
    2,   # agent 2  forward       -> training 0
    3,   # agent 3  back          -> training 1
    4,   # agent 4  left          -> training 2
    5,   # agent 5  right         -> training 3
    6,   # agent 6  jump          -> training 4
    7,   # agent 7  sneak         -> training 5
    8,   # agent 8  sprint        -> training 6
    9,   # agent 9  use           -> training 7
    10,  # agent 10 attack        -> training 8
    11,  # agent 11 drop          -> training 9
    0,   # agent 0  camera_pitch  -> training 10
    1,   # agent 1  camera_yaw    -> training 11
    15,  # agent 15 hotbar_1      -> training 12
    16,  # agent 16 hotbar_2      -> training 13
    17,  # agent 17 hotbar_3      -> training 14
    18,  # agent 18 hotbar_4      -> training 15
    19,  # agent 19 hotbar_5      -> training 16
    20,  # agent 20 hotbar_6      -> training 17
    21,  # agent 21 hotbar_7      -> training 18
    22,  # agent 22 hotbar_8      -> training 19
    23,  # agent 23 hotbar_9      -> training 20
    12,  # agent 12 inventory     -> training 21
    13,  # agent 13 swap_hand     -> training 22 (pickItem; no real eq.)
    14,  # agent 14 pick_block    -> training 23 (placeItem)
    24,  # agent 24 noop          -> training 24 (craft; no real eq.)
)
# Indices in the AGENT vector that are camera and need the [-1, 1] -> degrees
# scaling so the L0 sees values in the same scale it was trained on.
_AGENT_CAMERA_INDICES: tuple[int, ...] = (0, 1)
# Scale factor: the L0 was trained on raw camera deltas in degrees, clamped
# to [-1, 1] by `src/wally/data/dataset.py:66` (observed range -42 to +37
# degrees). The agent plans in [-1, 1] (which is already in the L0's
# training distribution) and the env at `src/wally/agent/env.py:92-99`
# passes the value through unchanged as degrees to MineStudio. We do NOT
# rescale here — the L0 must see the same [-1, 1] range it was trained on.
# Setting this to 180.0 (an earlier mistake) caused the L0 to extrapolate
# outside its training distribution on every camera decision, producing the
# saturated ±1.0 camera shake observed in the 1k-step L0 runs.
_CAMERA_DEGREE_SCALE: float = 1.0


def _translate_agent_action_to_l0(agent_action: torch.Tensor) -> torch.Tensor:
    """Reorder + rescale an agent-vocab action vector into the format
    the L0 was trained on. The L0 expects the agent's normalized [-1, 1]
    action values (its training data was clamped to that range by the
    dataloader; the env separately rescales to degrees for MineStudio).
    See ``_AGENT_TO_TRAINING_PERMUTATION``.
    """
    perm = torch.tensor(_AGENT_TO_TRAINING_PERMUTATION, dtype=torch.long, device=agent_action.device)
    l0_action = agent_action.index_select(-1, perm)
    # Camera columns (the two columns the permutation put at training
    # indices 10/11) keep the agent's [-1, 1] scale — do NOT multiply by
    # 180 here. The env handles the degrees conversion independently.
    scale = torch.ones(l0_action.shape[-1], dtype=l0_action.dtype, device=agent_action.device)
    scale[10] = _CAMERA_DEGREE_SCALE
    scale[11] = _CAMERA_DEGREE_SCALE
    return l0_action * scale

def _resize_frame_to_l0(frame: torch.Tensor) -> torch.Tensor:
    """Resize the agent's 224x224 frame to the L0's 64x64."""
    return F.interpolate(frame, size=(64, 64), mode="bilinear", align_corners=False)


class LeWorldModelAdapter:
    """Adapts a LeWorldModel to the WorldModelProtocol used by the planner.

    ``predict`` returns the **predicted change** Δ (frame-to-frame delta in
    latent space), matching the new model contract. The next latent is
    reconstructed as ``z + Δ`` in ``LatentRollout.rollout``.

    Actions from the planner are in the agent's vocab order
    (see :data:`_AGENT_TO_TRAINING_PERMUTATION`); ``predict`` translates
    them into the schema the L0 was trained on before embedding.
    """

    def __init__(self, model: LeWorldModel) -> None:
        self._model = model
        self._is_cnn = bool(getattr(model, "_is_cnn", False))

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        frame = _resize_frame_to_l0(frame)
        out = self._model.encoder(frame)
        if self._is_cnn:
            return out
        return out.mean(dim=1)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        l0_action = _translate_agent_action_to_l0(action)
        z_seq = z.unsqueeze(1)
        a_emb = self._model.action_embedder(l0_action.unsqueeze(1))
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
        gradient_policy: str = "detach",
        model_config: dict[str, object] | None = None,
    ) -> LatentRollout:
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            gradient_policy=gradient_policy,
            model_config=model_config,
        )

    def _load_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        model_config: dict[str, object] | None = None,
    ) -> LeWorldModelAdapter:
        map_location = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=map_location,
        )
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
        if "depth" not in model_cfg:
            model_cfg["depth"] = _infer_depth(
                checkpoint.get("model_state_dict", {}),
            )
        if "embed_dim" not in model_cfg:
            inferred_embed_dim = _infer_embed_dim(
                checkpoint.get("model_state_dict", {}),
            )
            if inferred_embed_dim is not None:
                model_cfg["embed_dim"] = inferred_embed_dim
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
