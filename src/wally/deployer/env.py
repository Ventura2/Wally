"""ServerEnv adapter matching MineStudioAgentEnv interface."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from wally.deployer.config import DeployConfig
from wally.deployer.connector import ServerConnector
from wally.deployer.executor import ActionExecutor
from wally.deployer.frame_renderer import FrameRenderer
from wally.deployer.safety import ActionContext, SafetyFilter
from wally.deployer.session import SessionManager
from wally.deployer.throttler import ActionThrottler

logger = logging.getLogger(__name__)


class ServerEnv:
    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._connector = ServerConnector(
            config.server_host, config.server_port, username=config.username
        )
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
        self._throttler_interval = config.throttler_interval
        self._throttler = ActionThrottler(
            handler=self._executor.send_packets,
            interval=self._throttler_interval,
        )

    def reset(self) -> Tensor:
        if self._closed:
            raise RuntimeError("Environment is closed")
        self._session.join()
        self._executor.session = self._session
        self._executor.connection = self._connector.connection
        self._executor.position_provider = self._position_provider
        self._executor.yaw = 0.0
        self._executor.pitch = 0.0
        self._connector.on_position_update(self._sync_session_to_connector)
        self._step_count = 0
        if self._throttler.is_enabled and not self._throttler.is_running:
            self._throttler.start()
        frame = self._renderer.render(
            self._session.position, self._yaw, self._pitch
        )
        return cast(Tensor, self._renderer.preprocess(frame))

    def _position_provider(self) -> tuple[float, float, float]:
        """Return the current authoritative position from the connector.

        The connector's position is updated by the spawn/teleport packet
        listener, so it tracks the server's view. The executor's deltas
        are applied to this base to avoid spoofed "teleport" positions.
        """
        return self._connector.position

    def _sync_session_to_connector(
        self,
        position: tuple[float, float, float],
        yaw: float,
        pitch: float,
    ) -> None:
        self._session.position = position
        self._yaw = yaw
        self._pitch = pitch
        if hasattr(self._executor, "yaw"):
            self._executor.yaw = yaw
        if hasattr(self._executor, "pitch"):
            self._executor.pitch = pitch

    def step(self, action: Tensor) -> tuple[Tensor, float, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment is closed")

        if self._connector.state.value == "disconnected":
            raise ConnectionError(
                "Server connection is down; aborting episode."
            )

        if self._is_connection_dead():
            self._connector._mark_disconnected()
            raise ConnectionError(
                "Server connection dropped (socket closed); aborting episode."
            )

        action_np = (
            action.detach().cpu().numpy()
            if isinstance(action, Tensor)
            else np.asarray(action)
        )
        action_np = np.clip(action_np, -1.0, 1.0).astype(action_np.dtype)

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
                {"safety_violation": True, "pov": frame, "step": self._step_count},
            )

        packets = self._executor.execute(action_np)
        self._throttler.submit_sync(packets)

        self._step_count += 1
        frame = self._renderer.render(
            self._session.position, self._yaw, self._pitch
        )
        obs = cast(Tensor, self._renderer.preprocess(frame))
        reward = 0.0
        done = False
        info: dict[str, Any] = {
            "packets": packets,
            "step": self._step_count,
            "pov": frame,
        }
        return obs, reward, done, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._throttler.stop()
        self._session.shutdown()

    def _is_connection_dead(self) -> bool:
        conn = self._connector.connection
        if conn is None:
            return self._connector.state.value == "disconnected"
        net_thread = getattr(conn, "networking_thread", None)
        if net_thread is not None and not net_thread.is_alive():
            return True
        socket = getattr(conn, "socket", None)
        if socket is not None:
            try:
                if socket.fileno() == -1:
                    return True
            except (OSError, AttributeError):
                return True
        return False


class MockServerEnv:
    """No-server environment for offline planner integration smoke tests."""

    _RENDER_RESOLUTION = (224, 224)
    _CONSTANT_COLOR = (128, 128, 128)

    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._closed = False
        self._step_count = 0
        self._position: tuple[float, float, float] = (0.0, 64.0, 0.0)
        self._yaw = 0.0
        self._pitch = 0.0
        self._executor = ActionExecutor(connection=None)
        self._renderer = FrameRenderer(
            resolution=self._RENDER_RESOLUTION,
            render_distance=config.render_distance,
        )
        self._rng = np.random.default_rng(seed=0)
        self._last_pov: np.ndarray = np.zeros(  # type: ignore[type-arg]
            (self._RENDER_RESOLUTION[0], self._RENDER_RESOLUTION[1], 3),
            dtype=np.uint8,
        )

    def reset(self) -> Tensor:
        if self._closed:
            raise RuntimeError("Environment is closed")
        self._step_count = 0
        self._position = (0.0, 64.0, 0.0)
        self._yaw = 0.0
        self._pitch = 0.0
        self._rng = np.random.default_rng(seed=0)
        return self._synthetic_frame()

    def step(self, action: Tensor) -> tuple[Tensor, float, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment is closed")

        action_np = (
            action.detach().cpu().numpy()
            if isinstance(action, Tensor)
            else np.asarray(action)
        )
        action_np = np.clip(action_np, -1.0, 1.0).astype(action_np.dtype)

        packets = self._executor.execute(action_np)
        self._apply_movement(packets)

        self._step_count += 1
        obs = self._synthetic_frame()
        reward = 0.0
        done = False
        info: dict[str, Any] = {
            "packets": packets,
            "step": self._step_count,
            "pov": self._last_pov,
        }
        return obs, reward, done, info

    def close(self) -> None:
        self._closed = True

    def _synthetic_frame(self) -> Tensor:
        arr = self._rng.random(
            (self._RENDER_RESOLUTION[0], self._RENDER_RESOLUTION[1], 3),
            dtype=np.float32,
        )
        self._last_pov = (arr * 255.0).astype(np.uint8)
        chw = arr.transpose(2, 0, 1)
        return torch.from_numpy(chw)

    def _apply_movement(self, packets: list[dict[str, Any]]) -> None:
        x, y, z = self._position
        for pkt in packets:
            kind = pkt.get("type")
            if kind == "movement":
                x = x + float(pkt.get("dx", 0.0))
                z = z + float(pkt.get("dz", 0.0))
            elif kind == "rotation":
                self._yaw = (self._yaw + float(pkt.get("dyaw", 0.0))) % 360.0
                self._pitch = self._pitch + float(pkt.get("dpitch", 0.0))
        self._position = (x, y, z)
