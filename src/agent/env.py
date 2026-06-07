from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.agent.config import AgentConfig
from src.collector.config import CollectorConfig
from src.collector.env import MineStudioEnv
from src.wally.planner.actions import (
    MineStudioActionVocab,
    continuous_to_discrete,
)


class MineStudioAgentEnv:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        collector_cfg = CollectorConfig(resize=config.resize)
        self._env = MineStudioEnv(collector_cfg)
        if config.action_vocab_path is not None:
            self._vocab = MineStudioActionVocab.from_yaml(config.action_vocab_path)
        else:
            self._vocab = MineStudioActionVocab.default()
        self._closed = False

    def _preprocess_frame(self, frame: np.ndarray) -> Tensor:
        img = Image.fromarray(frame).resize(
            (self.config.resize[1], self.config.resize[0]),
            Image.BILINEAR,
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t

    def reset(self) -> Tensor:
        frame = self._env.reset()
        return self._preprocess_frame(frame)

    def step(
        self, action: Tensor
    ) -> Tuple[Tensor, float, bool, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment is closed")

        lows = torch.tensor([d.low for d in self._vocab.dimensions])
        highs = torch.tensor([d.high for d in self._vocab.dimensions])
        action = torch.clamp(action, lows, highs)

        batched = action.unsqueeze(0)
        discrete_actions = continuous_to_discrete(batched, self._vocab)
        action_dict = discrete_actions[0]

        frame, reward, done, info = self._env.step(action_dict)
        return self._preprocess_frame(frame), reward, done, info

    def close(self) -> None:
        self._closed = True
        self._env.close()
