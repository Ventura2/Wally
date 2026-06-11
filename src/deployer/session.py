"""Session lifecycle, reconnection, and state persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from deployer.config import ReconnectConfig
from deployer.connector import ServerConnector

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        connector: ServerConnector,
        reconnect_config: ReconnectConfig | None = None,
        checkpoint_path: str | Path = "checkpoint.json",
    ) -> None:
        self._connector = connector
        self._reconnect_config = reconnect_config or ReconnectConfig()
        self._checkpoint_path = Path(checkpoint_path)
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._inventory: dict[str, Any] = {}
        self._goal_progress: dict[str, Any] = {}
        self._shutting_down = False
        self._connector.on_disconnect(self._on_unexpected_disconnect)

    def join(self) -> None:
        self._connector.connect()
        if self._checkpoint_path.exists():
            self._restore_state()

    def shutdown(self) -> None:
        self._shutting_down = True
        self._save_state()
        self._connector.disconnect()

    @property
    def position(self) -> tuple[float, float, float]:
        return self._position

    @position.setter
    def position(self, value: tuple[float, float, float]) -> None:
        self._position = value

    @property
    def inventory(self) -> dict[str, Any]:
        return self._inventory

    @inventory.setter
    def inventory(self, value: dict[str, Any]) -> None:
        self._inventory = value

    @property
    def goal_progress(self) -> dict[str, Any]:
        return self._goal_progress

    @goal_progress.setter
    def goal_progress(self, value: dict[str, Any]) -> None:
        self._goal_progress = value

    def _on_unexpected_disconnect(self, reason: str | None) -> None:
        if self._shutting_down:
            return
        logger.warning("Unexpected disconnect: %s. Attempting reconnection...", reason)
        asyncio.ensure_future(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        cfg = self._reconnect_config
        backoff = cfg.initial_backoff_s
        for attempt in range(1, cfg.max_attempts + 1):
            logger.info(
                "Reconnection attempt %d/%d (backoff=%.1fs)",
                attempt,
                cfg.max_attempts,
                backoff,
            )
            await asyncio.sleep(backoff)
            try:
                self._connector.connect()
                logger.info("Reconnected successfully on attempt %d", attempt)
                self._restore_state()
                return
            except ConnectionError:
                backoff = min(backoff * cfg.backoff_multiplier, cfg.max_backoff_s)
        logger.error("Max reconnection attempts (%d) exhausted", cfg.max_attempts)
        self._save_state()

    def _save_state(self) -> None:
        data = {
            "position": list(self._position),
            "inventory": self._inventory,
            "goal_progress": self._goal_progress,
        }
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(json.dumps(data, indent=2))
        logger.info("State saved to %s", self._checkpoint_path)

    def _restore_state(self) -> None:
        if not self._checkpoint_path.exists():
            return
        data = json.loads(self._checkpoint_path.read_text())
        self._position = tuple(data["position"])
        self._inventory = data.get("inventory", {})
        self._goal_progress = data.get("goal_progress", {})
        logger.info("State restored from %s", self._checkpoint_path)
