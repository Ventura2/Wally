"""Structured logging setup for deployment."""

from __future__ import annotations

import asyncio
import json
import logging as _logging
import logging.handlers as _handlers
from pathlib import Path
from typing import Any, Callable


class JSONFormatter(_logging.Formatter):
    def format(self, record: _logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry)


def setup_logging(
    log_dir: str = "logs/deploy",
    log_to_stdout: bool = False,
    level: int = _logging.INFO,
) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = _logging.getLogger("deployer")
    root_logger.setLevel(level)

    file_handler = _handlers.RotatingFileHandler(
        log_path / "deploy.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    if log_to_stdout:
        stdout_handler = _logging.StreamHandler()
        stdout_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(stdout_handler)


class ActionLogger:
    def __init__(self) -> None:
        self._logger = _logging.getLogger("deployer.actions")

    def log_action(
        self, action_vector: Any, position: tuple[float, float, float],
    ) -> None:
        if hasattr(action_vector, "tolist"):
            action_summary = sum(abs(x) for x in action_vector.tolist())
        else:
            action_summary = sum(abs(x) for x in action_vector)
        self._logger.info(
            "action executed",
            extra={"extra_data": {
                "action_norm": float(action_summary),
                "position": {"x": position[0], "y": position[1], "z": position[2]},
            }},
        )


class PositionTracker:
    def __init__(self, interval: float = 5.0) -> None:
        self._interval = interval
        self._logger = _logging.getLogger("deployer.position")
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._get_position: Callable[[], tuple[float, float, float]] | None = None

    def start(self, get_position: Callable[[], tuple[float, float, float]]) -> None:
        self._get_position = get_position
        self._running = True
        self._task = asyncio.create_task(self._track_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _track_loop(self) -> None:
        while self._running and self._get_position:
            pos = self._get_position()
            self._logger.info(
                "position",
                extra={"extra_data": {"x": pos[0], "y": pos[1], "z": pos[2]}},
            )
            await asyncio.sleep(self._interval)


class ServerEventLogger:
    def __init__(self) -> None:
        self._logger = _logging.getLogger("deployer.events")

    def on_chat_message(self, message: str, sender: str) -> None:
        self._logger.info(
            "chat",
            extra={"extra_data": {"sender": sender, "message": message}},
        )

    def on_player_join(self, username: str) -> None:
        self._logger.info(
            "player_join",
            extra={"extra_data": {"username": username}},
        )

    def on_death(self, message: str) -> None:
        self._logger.info(
            "death",
            extra={"extra_data": {"message": message}},
        )
