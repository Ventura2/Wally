from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from wally.agent.config import AgentConfig
from wally.collector.config import CollectorConfig
from wally.collector.env import MineStudioEnv
from wally.planner.actions import (
    MineStudioActionVocab,
    continuous_to_discrete,
)

# Map from the agent's vocab key to the MineStudio env (HumanSurvival) key.
# Keys not in this map (e.g. agent's `noop`, `swap_hand`, `drop`) are
# silently dropped — the env fills missing sub-actions with the per-action
# no-op (zero camera, unpressed button). The keys below are the ones the
# env actually understands.
_MINESTUDIO_KEY_MAP: Dict[str, str] = {
    "forward": "forward",
    "back": "back",
    "left": "left",
    "right": "right",
    "jump": "jump",
    "sneak": "sneak",
    "sprint": "sprint",
    "use": "use",
    "attack": "attack",
    "inventory": "inventory",
    "pick_block": "pickItem",
    "hotbar_1": "hotbar.1",
    "hotbar_2": "hotbar.2",
    "hotbar_3": "hotbar.3",
    "hotbar_4": "hotbar.4",
    "hotbar_5": "hotbar.5",
    "hotbar_6": "hotbar.6",
    "hotbar_7": "hotbar.7",
    "hotbar_8": "hotbar.8",
    "hotbar_9": "hotbar.9",
}

# Camera is the only non-button action and uses a 2D vector in degrees,
# not a binned discrete key.
_CAMERA_DEG_PER_BIN = 360.0 / 11.0  # 11 bins span 360 degrees; bin 5 = 0 deg


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

    def _translate_action(
        self, binned: Dict[str, int], continuous: Tensor
    ) -> Dict[str, Any]:
        """Convert the agent's binned-per-dim action dict to the MineStudio
        env's expected action dict.

        Camera is the only non-button action and the only one we want to
        keep at full continuous resolution: binning it to 11 discrete
        values per step would let the planner jump the view by up to
        ~327 degrees per step, which looks like the camera is
        teleporting. The env expects a single ``camera`` key with a 2D
        vector in degrees. The agent's continuous value is **already
        in degrees** because the L0 was trained on raw camera deltas
        (in degrees) clamped to ``[-1, 1]`` by
        ``src/wally/data/dataset.py:66`` (observed range -42 to +37
        degrees in the training shards). We pass the value through
        unchanged — multiplying by 180 here would be a 180x
        overshoot vs the L0's training distribution.
        """
        out: Dict[str, Any] = {}
        for k, v in binned.items():
            if k == "camera_pitch":
                out.setdefault("camera", [0.0, 0.0])[0] = (
                    continuous[0].item()
                )
            elif k == "camera_yaw":
                out.setdefault("camera", [0.0, 0.0])[1] = (
                    continuous[1].item()
                )
            elif k in _MINESTUDIO_KEY_MAP:
                out[_MINESTUDIO_KEY_MAP[k]] = v
            # else: unknown key (e.g. `noop`, `swap_hand`, `drop`) — drop
        return out

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
        env_action = self._translate_action(discrete_actions[0], action)

        frame, reward, done, info = self._env.step(env_action)
        return self._preprocess_frame(frame), reward, done, info

    def close(self) -> None:
        self._closed = True
        self._env.close()
