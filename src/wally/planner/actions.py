from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor


@dataclass
class ActionDimension:
    name: str
    low: float
    high: float
    bins: int


@dataclass
class MineStudioActionVocab:
    dimensions: list[ActionDimension]

    @classmethod
    def from_yaml(cls, path: str | Path) -> MineStudioActionVocab:
        with open(path) as f:
            data = yaml.safe_load(f)
        dims = [
            ActionDimension(
                name=d["name"],
                low=float(d["low"]),
                high=float(d["high"]),
                bins=int(d["bins"]),
            )
            for d in data["dimensions"]
        ]
        return cls(dimensions=dims)

    @classmethod
    def default(cls) -> MineStudioActionVocab:
        return cls(
            dimensions=[
                ActionDimension(name="camera_pitch", low=-1.0, high=1.0, bins=11),
                ActionDimension(name="camera_yaw", low=-1.0, high=1.0, bins=11),
                ActionDimension(name="forward", low=0.0, high=1.0, bins=2),
                ActionDimension(name="back", low=0.0, high=1.0, bins=2),
                ActionDimension(name="left", low=0.0, high=1.0, bins=2),
                ActionDimension(name="right", low=0.0, high=1.0, bins=2),
                ActionDimension(name="jump", low=0.0, high=1.0, bins=2),
                ActionDimension(name="sneak", low=0.0, high=1.0, bins=2),
                ActionDimension(name="sprint", low=0.0, high=1.0, bins=2),
                ActionDimension(name="use", low=0.0, high=1.0, bins=2),
                ActionDimension(name="attack", low=0.0, high=1.0, bins=2),
                ActionDimension(name="drop", low=0.0, high=1.0, bins=2),
                ActionDimension(name="inventory", low=0.0, high=1.0, bins=2),
                ActionDimension(name="swap_hand", low=0.0, high=1.0, bins=2),
                ActionDimension(name="pick_block", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_1", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_2", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_3", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_4", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_5", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_6", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_7", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_8", low=0.0, high=1.0, bins=2),
                ActionDimension(name="hotbar_9", low=0.0, high=1.0, bins=2),
                ActionDimension(name="noop", low=0.0, high=1.0, bins=2),
            ]
        )

    @property
    def action_dim(self) -> int:
        return len(self.dimensions)


def continuous_to_discrete(
    actions: Tensor, vocab: MineStudioActionVocab
) -> list[dict[str, int]]:
    if actions.ndim != 2:
        raise ValueError(
            f"Expected 2-D tensor (H, A), got shape {tuple(actions.shape)}"
        )
    h, a = actions.shape
    if a != vocab.action_dim:
        raise ValueError(
            f"Action dim mismatch: tensor has {a} columns, vocab has {vocab.action_dim}"
        )

    results: list[dict[str, int]] = []
    for t in range(h):
        row: dict[str, int] = {}
        for j, dim in enumerate(vocab.dimensions):
            val = actions[t, j].item()
            if val < dim.low or val > dim.high:
                raise ValueError(
                    f"Action out of bounds at timestep {t}, "
                    f"index {j} ('{dim.name}'): "
                    f"value {val:.6f} outside [{dim.low}, {dim.high}]"
                )
            span = dim.high - dim.low
            idx = math.floor((val - dim.low) / span * dim.bins)
            idx = max(0, min(idx, dim.bins - 1))
            row[dim.name] = idx
        results.append(row)
    return results


def discrete_to_continuous(
    actions: list[dict[str, int]], vocab: MineStudioActionVocab
) -> Tensor:
    if not actions:
        return torch.empty(0, vocab.action_dim)

    h = len(actions)
    out = torch.empty(h, vocab.action_dim)
    for t, row in enumerate(actions):
        for j, dim in enumerate(vocab.dimensions):
            if dim.name not in row:
                raise ValueError(
                    f"Missing dimension '{dim.name}' at timestep {t}"
                )
            idx = row[dim.name]
            if idx < 0 or idx >= dim.bins:
                raise ValueError(
                    f"Bin index out of range at timestep {t}, "
                    f"index {j} ('{dim.name}'): "
                    f"index {idx} outside [0, {dim.bins - 1}]"
                )
            span = dim.high - dim.low
            out[t, j] = dim.low + (idx + 0.5) * span / dim.bins
    return out
