"""ServerEnv adapter matching MineStudioAgentEnv interface."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from torch import Tensor

from deployer.config import DeployConfig
from deployer.connector import ServerConnector
from deployer.executor import ActionExecutor
from deployer.frame_renderer import FrameRenderer
from deployer.safety import ActionContext, SafetyFilter
from deployer.session import SessionManager
from deployer.throttler import ActionThrottler

logger = logging.getLogger(__name__)


class ServerEnv:
    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._connector = ServerConnector(config.server_host, config.server_port)
        self._session = SessionManager(
            self._connector,
            reconnect_config=config.reconnect,
            checkpoint_path=config.checkpoint_path + ".session.json",
        )
        self._executor = ActionExecutor()
        self._renderer = FrameRenderer(
            resolution=(224, 224),
            render_distance=config.render_distance,
        )
        self._safety = SafetyFilter(config.safety)
        self._yaw = 0.0
        self._pitch = 0.0
        self._closed = False
        self._step_count = 0

        async def _noop_handler(action: object) -> None:
            pass

        self._throttler = ActionThrottler(handler=_noop_handler)

    def reset(self) -> Tensor:
        if self._closed:
            raise RuntimeError("Environment is closed")
        self._session.join()
        self._executor.connection = self._connector.connection
        self._step_count = 0
        frame = self._renderer.render(
            self._session.position, self._yaw, self._pitch
        )
        return cast(Tensor, self._renderer.preprocess(frame))

    def step(self, action: Tensor) -> tuple[Tensor, float, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment is closed")

        action_np = (
            action.detach().cpu().numpy()
            if isinstance(action, Tensor)
            else np.asarray(action)
        )

        safety_ctx = ActionContext(
            action_type="step",
            player_position=self._session.position,
        )
        if not self._safety.check(safety_ctx):
            frame = self._renderer.render(
                self._session.position, self._yaw, self._pitch
            )
            return (
                cast(Tensor, self._renderer.preprocess(frame)),
                0.0,
                False,
                {"safety_violation": True},
            )

        packets = self._executor.execute(action_np)

        self._step_count += 1
        frame = self._renderer.render(
            self._session.position, self._yaw, self._pitch
        )
        obs = cast(Tensor, self._renderer.preprocess(frame))
        reward = 0.0
        done = False
        info: dict[str, Any] = {"packets": packets, "step": self._step_count}
        return obs, reward, done, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.shutdown()
