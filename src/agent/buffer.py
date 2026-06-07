from __future__ import annotations

import numpy as np
import torch


class TrajectoryBuffer:
    """Accumulates frames and actions during an episode."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def add(self, frame: np.ndarray | torch.Tensor, action: np.ndarray | torch.Tensor) -> None:
        if isinstance(frame, torch.Tensor):
            frame = (frame.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self._frames.append(frame)
        self._actions.append(action)

    def to_dict(self) -> dict[str, np.ndarray]:
        if not self._frames:
            raise ValueError("Buffer is empty")
        return {
            "frames": np.stack(self._frames, axis=0),
            "actions": np.stack(self._actions, axis=0),
        }

    def __len__(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._actions.clear()
