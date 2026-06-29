from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


class TrajectoryBuffer:
    """Accumulates frames and actions during an episode."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._events: list[dict[str, Any] | None] = []
        self._costs: list[float] = []
        self._replan_z_H: list[np.ndarray] = []
        self._replan_costs: list[np.ndarray] = []
        self._replan_l2_costs: list[np.ndarray] = []
        self._replan_z_g: list[np.ndarray] = []

    def add(
        self,
        frame: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        info: Mapping[str, Any] | None = None,
        cost: float | None = None,
    ) -> None:
        if isinstance(frame, torch.Tensor):
            frame = (
                frame.permute(1, 2, 0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self._frames.append(frame)
        self._actions.append(action)
        self._events.append(self._extract_event(info))
        if cost is not None:
            self._costs.append(float(cost))

    def add_replan(
        self,
        z_H: torch.Tensor,
        planner_costs: torch.Tensor,
        z_g: torch.Tensor,
    ) -> None:
        """Record per-replan SCSA data: end-latents of the final CEM
        population, the planner's cost for each, and the diagnostic L2
        cost `||z_H - z_g||^2`. Used by the same-candidate selection
        audit (TRM paper, App. B.1) to check whether the planner's cost
        ranks candidates the way a trusted diagnostic does.
        """
        z_H_np = z_H.detach().cpu().numpy().astype(np.float32)
        c_np = planner_costs.detach().cpu().numpy().astype(np.float32)
        z_g_np = z_g.detach().cpu().numpy().astype(np.float32)
        l2 = ((z_H_np - z_g_np[None, :]) ** 2).sum(axis=-1).astype(np.float32)
        self._replan_z_H.append(z_H_np)
        self._replan_costs.append(c_np)
        self._replan_l2_costs.append(l2)
        self._replan_z_g.append(z_g_np)

    def to_dict(self) -> dict[str, np.ndarray]:
        if not self._frames:
            raise ValueError("Buffer is empty")
        result: dict[str, np.ndarray] = {
            "frames": np.stack(self._frames, axis=0),
            "actions": np.stack(self._actions, axis=0),
        }
        if any(event is not None for event in self._events):
            result["events"] = np.asarray(self._events, dtype=object)
        if self._costs:
            result["costs"] = np.asarray(self._costs, dtype=np.float32)
        if self._replan_z_H:
            result["scsa_z_H"] = np.stack(self._replan_z_H, axis=0)
            result["scsa_costs"] = np.stack(self._replan_costs, axis=0)
            result["scsa_l2_costs"] = np.stack(self._replan_l2_costs, axis=0)
            result["scsa_z_g"] = np.stack(self._replan_z_g, axis=0)
        return result

    def _extract_event(
        self, info: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if not info:
            return None

        relevant: dict[str, Any] = {}
        for key, value in info.items():
            if self._is_relevant_key(key):
                relevant[key] = self._to_serializable(value)
        return relevant or None

    @staticmethod
    def _is_relevant_key(key: str) -> bool:
        key_lower = key.lower()
        return any(
            token in key_lower
            for token in (
                "inventory",
                "block",
                "break",
                "pickup",
                "pick_up",
                "item",
                "craft",
                "damage",
                "health",
                "food",
                "saturation",
            )
        )

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, Mapping):
            return {
                str(key): cls._to_serializable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._to_serializable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __len__(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._actions.clear()
        self._events.clear()
        self._costs.clear()
        self._replan_z_H.clear()
        self._replan_costs.clear()
        self._replan_l2_costs.clear()
        self._replan_z_g.clear()
