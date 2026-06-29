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
    lower_hierarchy_checkpoint: str | Path | None = None,
    trm_head_checkpoint: str | Path | None = None,
    trm_lambda: float = 0.5,
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
        lower_hierarchy_checkpoint: Path to the next-lower layer's
            checkpoint (used by ``"hierarchical-embedding"`` when
            ``layer_depth >= 2``). Required because the topmost
            checkpoint's ``model_state_dict`` only carries the
            topmost layer's JEPAWorldModel — lower layers' JEPAs must
            be loaded from their own checkpoints. The lower checkpoint
            also provides the L1 ``LayerSpec`` (K/depth/heads) when the
            topmost checkpoint's ``config.layers`` only contains the
            topmost spec.
        trm_head_checkpoint: Path to a trained TRM reachability head
            (``tools/train_trm_head.py`` output). When provided, the
            CEM low-level planner uses a hybrid latent-cost + TRM
            reachability cost (TRM paper Eq. 6). Ignored for non-CEM
            planner kinds.
        trm_lambda: TRM weight in the hybrid cost (Eq. 6); 0.5 by
            default per the paper's PushT boundary guidance.

    Returns:
        An object implementing :class:`agent.protocol.PlannerProtocol`.

    Raises:
        ValueError: If ``planner_kind`` is not one of the four supported kinds.
    """
    cem_config = CEMConfig.default()
    gradient_mpc_config = GradientMPCConfig.default()
    high_level_config = HighLevelPlannerConfig.default()
    hier_config = HierarchicalPlannerConfig.default()

    trm_head = None
    if trm_head_checkpoint is not None:
        from wally.planner.trm_head import TRMHead

        ckpt = torch.load(str(trm_head_checkpoint), map_location="cpu", weights_only=False)
        latent_dim = ckpt["latent_dim"]
        hidden_dim = ckpt.get("hidden_dim", 256)
        trm_head = TRMHead(latent_dim=latent_dim, hidden_dim=hidden_dim)
        trm_head.load_state_dict(ckpt["state_dict"])

    if planner_kind == "cem":
        from wally.agent.protocol import FlatPlannerAdapter

        cem_config = cem_config.model_copy(
            update={
                "inventory_stall_penalty": 0.25,
                "diversity_penalty": 0.0,
                "camera_still_penalty": 0.0,
            }
        )
        planner = GoalConditionedPlanner(
            rollout, encoder, cem_config,
            trm_head=trm_head, trm_lambda=trm_lambda,
        )
        return FlatPlannerAdapter(planner)

    if planner_kind == "gradient":
        from wally.agent.protocol import FlatPlannerAdapter

        planner = GradientMPC(rollout, encoder, gradient_mpc_config)
        return FlatPlannerAdapter(planner)

    if planner_kind == "hierarchical":
        from wally.agent.protocol import HierarchicalPlannerAdapter

        high_level = HighLevelPlanner(rollout._model, encoder, high_level_config)
        cem_config = cem_config.model_copy(
            update={
                "inventory_stall_penalty": 0.25,
                "diversity_penalty": 0.0,
                "camera_still_penalty": 0.0,
            }
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

        # When the topmost checkpoint bundles lower-layer encoders (e.g. an
        # L2 checkpoint's encoder_state_dict contains the L1's proj under the
        # `l1_encoder.` prefix), prepend a synthetic L1 LayerSpec so the chain
        # in layer_specs matches the actual saved encoder chain. The L1's
        # JEPA-world-model weights are NOT bundled in the topmost checkpoint —
        # they must be loaded from the separate lower_hierarchy_checkpoint.
        if "l1_encoder.proj.weight" in ck["encoder_state_dict"] and not any(
            s.name == "l1" for s in layer_specs
        ):
            l1_D = int(ck["encoder_state_dict"]["l1_encoder.proj.weight"].shape[0])
            l1_K, l1_depth, l1_heads, l1_eps = 32, 2, 4, 0.1
            if lower_hierarchy_checkpoint is not None:
                l1_ck = torch.load(
                    lower_hierarchy_checkpoint, map_location="cpu", weights_only=False
                )
                l1_cfg = l1_ck.get("config", {}) or {}
                l1_layers = l1_cfg.get("layers", []) if isinstance(l1_cfg, dict) else []
                if l1_layers:
                    s = l1_layers[0]
                    l1_K = int(s.get("K", l1_K))
                    l1_depth = int(s.get("depth", l1_depth))
                    l1_heads = int(s.get("heads", l1_heads))
                    l1_eps = float(s.get("drift_epsilon", l1_eps))
            layer_specs = [
                LayerSpec(
                    name="l1",
                    K=l1_K,
                    D=l1_D,
                    depth=l1_depth,
                    heads=l1_heads,
                    drift_epsilon=l1_eps,
                )
            ] + layer_specs

        l0_ckpt = config.l0_checkpoint
        if not l0_ckpt:
            raise ValueError("HierarchyConfig.l0_checkpoint is empty in the checkpoint")

        l0_model = l0_inner
        encoders: list = []
        wms: list = []
        for i, spec in enumerate(layer_specs):
            if i == 0:
                l1_enc = L1Encoder(l0_model, D1=spec.D)
                # L1's encoder weights are either top-level (an L1 checkpoint)
                # or under the `l1_encoder.` prefix (an L2/L3 checkpoint that
                # bundles the L1).
                if "l1_encoder.proj.weight" in ck["encoder_state_dict"]:
                    enc_sd = {
                        k.removeprefix("l1_encoder."): v
                        for k, v in ck["encoder_state_dict"].items()
                        if k.startswith("l1_encoder.")
                    }
                else:
                    enc_sd = ck["encoder_state_dict"]
                l1_enc.load_state_dict(enc_sd, strict=False)
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
                # Load only this layer's own encoder weights (top-level proj,
                # since the lower layer's encoder is already loaded into prev).
                top_sd = {
                    k: v
                    for k, v in ck["encoder_state_dict"].items()
                    if not any(
                        k.startswith(p)
                        for p in ("l1_encoder.", "l2_encoder.", "l3_encoder.")
                    )
                }
                enc.load_state_dict(top_sd, strict=False)
                encoders.append(enc)
            wm = JEPAWorldModel(
                state_dim=spec.D,
                target_dim=spec.D,
                hidden_dim=spec.D * 2,
                depth=spec.depth,
                num_heads=spec.heads,
            )
            # The model_state_dict only carries the topmost layer's JEPA. For
            # lower layers in the chain, load the JEPA from the
            # lower_hierarchy_checkpoint.
            is_topmost = i == len(layer_specs) - 1
            if is_topmost:
                wm.load_state_dict(ck["model_state_dict"])
            elif lower_hierarchy_checkpoint is not None:
                lower_ck = torch.load(
                    lower_hierarchy_checkpoint, map_location="cpu", weights_only=False
                )
                wm.load_state_dict(lower_ck["model_state_dict"])
            else:
                raise ValueError(
                    f"Cannot load JEPAWorldModel for layer {spec.name!r}: the "
                    f"topmost checkpoint's model_state_dict only contains the "
                    f"topmost layer's weights and no --lower-hierarchy-checkpoint "
                    f"was provided."
                )
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
            cem_config.model_copy(
                update={
                    "inventory_stall_penalty": 0.25,
                    "diversity_penalty": 0.0,
                    "camera_still_penalty": 0.0,
                }
            ),
            trm_head=trm_head,
            trm_lambda=trm_lambda,
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
