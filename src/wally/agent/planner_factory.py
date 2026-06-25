"""Shared planner factory used by ``wally-play`` and ``wally-deploy``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig
from wally.planner.hierarchical_planner import (
    HierarchicalPlanner,
    HierarchicalPlannerConfig,
)
from wally.planner.high_level_planner import (
    HighLevelPlanner,
    HighLevelPlannerConfig,
)
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout

if TYPE_CHECKING:
    from wally.agent.protocol import PlannerProtocol


def build_planner(
    planner_kind: str,
    rollout: LatentRollout,
    encoder: torch.nn.Module,
    *,
    hierarchy_checkpoint: str | Path | None = None,
    layer_depth: int = 0,
) -> "PlannerProtocol":
    """Build a planner matching ``planner_kind``.

    Args:
        planner_kind: One of ``"cem"``, ``"gradient"``, ``"hierarchical"``,
            or ``"hierarchical-embedding"``.
        rollout: A :class:`LatentRollout` for the trained LeWorldModel.
        encoder: The encoder module (typically ``rollout._model.encode``).
        hierarchy_checkpoint: Path to a saved hierarchy checkpoint (used
            by ``"hierarchical-embedding"`` only).
        layer_depth: Number of upper layers to activate
            (``"hierarchical-embedding"`` only).

    Returns:
        An object implementing :class:`agent.protocol.PlannerProtocol`.

    Raises:
        ValueError: If ``planner_kind`` is not one of the four supported kinds.
    """
    cem_config = CEMConfig.default()
    gradient_mpc_config = GradientMPCConfig.default()
    high_level_config = HighLevelPlannerConfig.default()
    hier_config = HierarchicalPlannerConfig.default()

    if planner_kind == "cem":
        from wally.agent.protocol import FlatPlannerAdapter

        cem_config = cem_config.model_copy(
            update={
                "inventory_stall_penalty": 0.25,
                "diversity_penalty": 1.0e-3,
                "camera_still_penalty": 1.0e-3,
            }
        )
        planner = GoalConditionedPlanner(rollout, encoder, cem_config)
        return FlatPlannerAdapter(planner)

    if planner_kind == "gradient":
        from wally.agent.protocol import FlatPlannerAdapter

        planner = GradientMPC(rollout, encoder, gradient_mpc_config)
        return FlatPlannerAdapter(planner)

    if planner_kind == "hierarchical":
        from wally.agent.protocol import HierarchicalPlannerAdapter

        high_level = HighLevelPlanner(rollout._model, encoder, high_level_config)
        cem_config = cem_config.model_copy(
            update={"inventory_stall_penalty": 0.25}
        )
        low_level = GoalConditionedPlanner(rollout, encoder, cem_config)
        planner = HierarchicalPlanner(high_level, low_level, hier_config)
        return HierarchicalPlannerAdapter(planner)

    if planner_kind == "hierarchical-embedding":
        from wally.agent.protocol import HierarchicalEmbeddingPlannerAdapter
        from wally.hierarchy.config import HierarchyConfig, LayerSpec
        from wally.hierarchy.encoders import L1Encoder
        from wally.hierarchy.jepa import JEPAWorldModel
        from wally.hierarchy.planner import HierarchicalEmbeddingPlanner

        if hierarchy_checkpoint is None:
            raise ValueError(
                "planner_kind='hierarchical-embedding' requires hierarchy_checkpoint"
            )
        ck = torch.load(hierarchy_checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = ck.get("config", {}) or {}
        if "layers" in cfg_dict and isinstance(cfg_dict["layers"], list):
            cfg_dict = {
                **cfg_dict,
                "layers": [LayerSpec(**layer) for layer in cfg_dict["layers"]],
            }
        config = HierarchyConfig(
            **{
                k: v
                for k, v in cfg_dict.items()
                if k in HierarchyConfig.__dataclass_fields__
            }
        )
        if not config.layers:
            raise ValueError("HierarchyConfig must contain at least one layer spec")

        l0_inner = getattr(rollout._model, "_model", rollout._model)
        l0_dim = l0_inner.projector.net[3].out_features
        layer_specs = config.layers[: max(1, layer_depth)]
        if not layer_specs:
            layer_specs = [config.layers[0]]

        l0_ckpt = config.l0_checkpoint
        if not l0_ckpt:
            raise ValueError("HierarchyConfig.l0_checkpoint is empty in the checkpoint")

        l0_model = l0_inner
        encoders: list = []
        wms: list = []
        for i, spec in enumerate(layer_specs):
            if i == 0:
                l1_enc = L1Encoder(l0_model, D1=spec.D)
                l1_enc.load_state_dict(ck["encoder_state_dict"], strict=False)
                encoders.append(l1_enc)
            else:
                prev = encoders[-1]
                from wally.hierarchy.encoders import L2Encoder, L3Encoder
                if isinstance(prev, L1Encoder):
                    enc = L2Encoder(prev, D2=spec.D)
                elif isinstance(prev, L2Encoder):
                    enc = L3Encoder(prev, D3=spec.D)
                else:
                    raise NotImplementedError(
                        "Hierarchy deeper than L3 is not supported"
                    )
                encoders.append(enc)
            wm = JEPAWorldModel(
                state_dim=spec.D,
                target_dim=spec.D,
                hidden_dim=spec.D * 2,
                depth=spec.depth,
                num_heads=spec.heads,
            )
            wm.load_state_dict(ck["model_state_dict"])
            wms.append(wm)

        layers_for_planner: list[tuple[str, LayerSpec, JEPAWorldModel, str | None]] = []
        for i, (enc, wm, spec) in enumerate(zip(encoders, wms, layer_specs)):
            name = f"l{i + 1}"
            above = f"l{i}" if i + 1 < len(layer_specs) else None
            layers_for_planner.append((name, spec, wm, above))

        def l0_state_fn(frame: torch.Tensor) -> torch.Tensor:
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            lat = rollout._model.encode(frame)
            lat = lat.unsqueeze(1) if lat.dim() == 2 else lat
            projected = l0_inner._projector_fp32(lat)
            return projected.squeeze(1) if projected.dim() == 3 else projected

        low_level = GoalConditionedPlanner(
            rollout,
            encoder,
            cem_config.model_copy(update={"inventory_stall_penalty": 0.25}),
        )
        hier_planner = HierarchicalEmbeddingPlanner(
            l0_planner=low_level,
            l0_state_fn=l0_state_fn,
            layers=layers_for_planner,
            cem_config=cem_config,
            device="cpu",
            l0_dim=l0_dim,
            lowest_encoder=encoders[-1],
        )
        return HierarchicalEmbeddingPlannerAdapter(hier_planner)

    raise ValueError(
        f"Unknown planner kind: {planner_kind!r}. "
        f"Expected one of: 'cem', 'gradient', 'hierarchical', 'hierarchical-embedding'."
    )
