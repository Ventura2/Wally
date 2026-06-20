"""Server connection management via pyCraft."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

try:
    from minecraft.networking.connection import Connection as _PyCraftConnection
    from minecraft.networking.packets import Packet as _PyCraftPacket
    from minecraft.networking.packets.clientbound.play import (
        KeepAlivePacket as _ClientboundKeepAlivePacket,
        PlayerPositionAndLookPacket as _ClientboundPositionAndLookPacket,
    )
    from minecraft.networking.packets.serverbound.play import (
        KeepAlivePacket as _ServerboundKeepAlivePacket,
        TeleportConfirmPacket as _TeleportConfirmPacket,
    )

    _HAS_PYCRAFT = True
except ImportError:
    _PyCraftConnection = None
    _PyCraftPacket = None
    _ClientboundKeepAlivePacket = None
    _ClientboundPositionAndLookPacket = None
    _ServerboundKeepAlivePacket = None
    _TeleportConfirmPacket = None
    _HAS_PYCRAFT = False

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


class ServerConnector:
    def __init__(self, host: str, port: int, username: str = "WallyAgent") -> None:
        self._host = host
        self._port = port
        self._username = username
        self._state = ConnectionState.DISCONNECTED
        self._connection: _PyCraftConnection | None = None
        self._connect_callbacks: list[Callable[[], None]] = []
        self._disconnect_callbacks: list[Callable[[str | None], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._yaw: float = 0.0
        self._pitch: float = 0.0
        self._position_callbacks: list[
            Callable[[tuple[float, float, float], float, float], None]
        ] = []

    @property
    def state(self) -> ConnectionState:
        return self._state

    def _mark_disconnected(self) -> None:
        """Mark the connector as disconnected (e.g. network thread died)."""
        if self._state == ConnectionState.DISCONNECTED:
            return
        self._state = ConnectionState.DISCONNECTED
        self._connection = None
        logger.info("Connector marked disconnected")
        self._fire_disconnect("Connection lost")

    @property
    def connection(self) -> _PyCraftConnection | None:
        return self._connection

    @property
    def position(self) -> tuple[float, float, float]:
        return self._position

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def pitch(self) -> float:
        return self._pitch

    def on_position_update(
        self,
        callback: Callable[[tuple[float, float, float], float, float], None],
    ) -> None:
        self._position_callbacks.append(callback)

    def connect(self) -> None:
        if not _HAS_PYCRAFT:
            raise ImportError(
                "pyCraft is not installed. "
                "Install it with: pip install pyCraft"
            )

        self._state = ConnectionState.CONNECTING
        logger.info("Connecting to %s:%d", self._host, self._port)

        try:
            conn = _PyCraftConnection(self._host, self._port, username=self._username)
            self._register_packet_handlers(conn)
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

    def _register_packet_handlers(self, conn: _PyCraftConnection) -> None:
        """Register listeners for the spawn/teleport position packet.

        pyCraft delivers the server's spawn position via
        ``PlayerPositionAndLookPacket`` (clientbound). We need to:
          1. Confirm the teleport with ``TeleportConfirmPacket`` so the
             server stops resending the spawn.
          2. Update the connector's tracked position/yaw/pitch so the
             executor can build deltas from a valid base.
        """
        if not _HAS_PYCRAFT:
            return

        def _on_spawn(pkt):  # type: ignore[no-untyped-def]
            try:
                # PlayerPositionAndLookPacket has .x .y .z .yaw .pitch, plus
                # optional .teleport_id on protocol >= 107.
                self._position = (float(pkt.x), float(pkt.y), float(pkt.z))
                self._yaw = float(pkt.yaw)
                self._pitch = float(pkt.pitch)
                logger.info(
                    "Spawn position: (%.2f, %.2f, %.2f) yaw=%.1f pitch=%.1f",
                    self._position[0], self._position[1], self._position[2],
                    self._yaw, self._pitch,
                )
                for cb in self._position_callbacks:
                    try:
                        cb(self._position, self._yaw, self._pitch)
                    except Exception:
                        logger.exception("Position callback raised")
                if hasattr(pkt, "teleport_id"):
                    confirm = _TeleportConfirmPacket()
                    confirm.teleport_id = int(pkt.teleport_id)
                    try:
                        conn.write_packet(confirm)
                    except Exception:
                        logger.exception("Failed to send TeleportConfirmPacket")
            except Exception:
                logger.exception("Spawn handler raised")

        conn.register_packet_listener(
            _on_spawn,
            _ClientboundPositionAndLookPacket,
        )

        def _on_keep_alive(pkt):  # type: ignore[no-untyped-def]
            try:
                reply = _ServerboundKeepAlivePacket()
                if hasattr(pkt, "keep_alive_id"):
                    reply.keep_alive_id = pkt.keep_alive_id
                conn.write_packet(reply)
            except Exception:
                logger.exception("Failed to reply to KeepAlive")

        conn.register_packet_listener(
            _on_keep_alive,
            _ClientboundKeepAlivePacket,
        )

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
