"""Server connection management via pyCraft."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

try:
    from minecraft import Connection as _PyCraftConnection
    from minecraft.networking.packets import Packet as _PyCraftPacket

    _HAS_PYCRAFT = True
except ImportError:
    _PyCraftConnection = None
    _PyCraftPacket = None
    _HAS_PYCRAFT = False

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


class ServerConnector:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._state = ConnectionState.DISCONNECTED
        self._connection: _PyCraftConnection | None = None
        self._connect_callbacks: list[Callable[[], None]] = []
        self._disconnect_callbacks: list[Callable[[str | None], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def connection(self) -> _PyCraftConnection | None:
        return self._connection

    def connect(self) -> None:
        if not _HAS_PYCRAFT:
            raise ImportError(
                "pyCraft is not installed. "
                "Install it with: pip install pyCraft"
            )

        self._state = ConnectionState.CONNECTING
        logger.info("Connecting to %s:%d", self._host, self._port)

        try:
            conn = _PyCraftConnection(self._host, self._port)
            conn.connect()
            self._connection = conn
            self._state = ConnectionState.CONNECTED
            logger.info("Connected to %s:%d", self._host, self._port)
            self._fire_connect()
        except Exception as exc:
            self._state = ConnectionState.DISCONNECTED
            self._connection = None
            logger.error("Connection failed: %s", exc)
            self._fire_error(exc)
            raise ConnectionError(
                f"Failed to connect to {self._host}:{self._port}"
            ) from exc

    def disconnect(self) -> None:
        if self._state == ConnectionState.DISCONNECTED:
            return

        reason: str | None = None
        try:
            if self._connection is not None:
                self._connection.disconnect()
                reason = "Client disconnected"
        except Exception as exc:
            reason = str(exc)
            logger.warning("Error during disconnect: %s", exc)
        finally:
            self._connection = None
            self._state = ConnectionState.DISCONNECTED
            logger.info("Disconnected from %s:%d", self._host, self._port)
            self._fire_disconnect(reason)

    def on_connect(self, callback: Callable[[], None]) -> None:
        self._connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[str | None], None]) -> None:
        self._disconnect_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        self._error_callbacks.append(callback)

    def _fire_connect(self) -> None:
        for cb in self._connect_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Connect callback raised an exception")

    def _fire_disconnect(self, reason: str | None) -> None:
        for cb in self._disconnect_callbacks:
            try:
                cb(reason)
            except Exception:
                logger.exception("Disconnect callback raised an exception")

    def _fire_error(self, exc: Exception) -> None:
        for cb in self._error_callbacks:
            try:
                cb(exc)
            except Exception:
                logger.exception("Error callback raised an exception")
