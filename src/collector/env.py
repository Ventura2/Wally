from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from src.collector.config import CollectorConfig

try:
    from minestudio.simulator import MinecraftSim as _MinecraftSim
except (ImportError, OSError):
    _MinecraftSim = None


class MineStudioEnv:
    def __init__(self, config: CollectorConfig) -> None:
        if _MinecraftSim is None:
            raise ImportError(
                "MineStudio is not installed. "
                "Install it with: pip install minestudio"
            )
        self.config = config
        self._sim = _MinecraftSim(
            obs_size=config.resize,
            action_type=config.action_type,
        )

    @property
    def action_space(self) -> Any:
        return self._sim.action_space

    def reset(self) -> np.ndarray:
        obs_dict, _info = self._sim.reset()
        return obs_dict["image"]

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs_dict, reward, terminated, truncated, info = self._sim.step(action)
        done = terminated or truncated
        info = dict(info)
        if "pov" in obs_dict:
            info["pov"] = obs_dict["pov"]
        return obs_dict["image"], reward, done, info

    def close(self) -> None:
        self._sim.close()
