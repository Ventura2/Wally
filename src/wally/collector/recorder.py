from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
from wally.collector.config import CollectorConfig


class TransitionRecorder:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.episode_id: Optional[str] = None
        self.step_index: int = 0
        self.seed: Optional[int] = None

    def start_episode(self, seed: Optional[int] = None) -> str:
        self.episode_id = str(uuid.uuid4())
        self.step_index = 0
        self.seed = seed
        return self.episode_id

    def record_step(
        self, env: Any, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.episode_id is None:
            raise RuntimeError(
                "No active episode. Call start_episode() before record_step()."
            )

        accumulated_reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        obs: Optional[np.ndarray] = None

        for _ in range(self.config.frame_skip):
            obs, reward, done, info = env.step(action)
            accumulated_reward += reward
            if done:
                break

        timestamp = int(time.time() * 1000)

        transition: Dict[str, Any] = {
            "observation": obs,
            "action": action,
            "reward": accumulated_reward,
            "done": done,
            "timestamp": timestamp,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "seed": self.seed,
        }

        self.step_index += 1

        if done:
            self.episode_id = None

        return transition
