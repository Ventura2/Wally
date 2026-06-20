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

    def add(
        self,
        frame: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        info: Mapping[str, Any] | None = None,
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

    def to_dict(self) -> dict[str, np.ndarray]:
        if not self._frames:
            raise ValueError("Buffer is empty")
        result: dict[str, np.ndarray] = {
            "frames": np.stack(self._frames, axis=0),
            "actions": np.stack(self._actions, axis=0),
        }
        if any(event is not None for event in self._events):
            result["events"] = np.asarray(self._events, dtype=object)
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
