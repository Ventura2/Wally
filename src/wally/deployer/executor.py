"""Translate action vectors to pyCraft packets and write them on the connection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from minecraft.networking.packets import Packet as _Packet
    from minecraft.networking.packets.serverbound.play import (
        PlayerBlockPlacementPacket as _PlayerBlockPlacementPacket,
        PositionAndLookPacket as _PositionAndLookPacket,
    )
    from minecraft.networking.types import BlockFace as _BlockFace
    from minecraft.networking.types import Position as _Position
    from minecraft.networking.types.basic import Byte as _PyByte
    from minecraft.networking.types.basic import Integer as _PyInt
    from minecraft.networking.types.basic import Short as _PyShort
    from minecraft.networking.types.basic import UnsignedByte as _PyUnsignedByte
    from minecraft.networking.types.basic import VarInt as _PyVarInt

    _HAS_PYCRAFT = True
except ImportError:
    _Packet = None
    _PositionAndLookPacket = None
    _PlayerBlockPlacementPacket = None
    _BlockFace = None
    _Position = None
    _PyByte = None
    _PyInt = None
    _PyShort = None
    _PyUnsignedByte = None
    _PyVarInt = None
    _HAS_PYCRAFT = False

if TYPE_CHECKING:
    from wally.deployer.session import SessionManager

logger = logging.getLogger(__name__)

ACTION_THRESHOLD = 0.5
ACTION_DIM = 25
ROTATION_SCALE = 10.0

_DIG_STATUS_STARTED = 0
_DIG_STATUS_FINISHED = 2
_DIG_STATUS_START_DIGGING = 0
_DIG_STATUS_FACE_SELF_VALUE = 255

_ENTITY_ACTION_START_SNEAK = 0
_ENTITY_ACTION_STOP_SNEAK = 1
_ENTITY_ACTION_START_SPRINT = 2
_ENTITY_ACTION_STOP_SPRINT = 3
_ENTITY_ACTION_START_JUMP = 8

if _HAS_PYCRAFT:

    class _PlayerDiggingPacket(_Packet):  # type: ignore[misc]
        """Minimal player digging packet (serverbound play)."""

        @staticmethod
        def get_id(context: Any) -> int:
            return 0x14 if context.protocol_later_eq(755) else 0x13

        packet_name = "player digging"
        get_definition = staticmethod(
            lambda context: [
                {"status": _PyInt},
                {"location": _Position},
                {"face": _PyUnsignedByte},
            ]
        )

    class _EntityActionPacket(_Packet):  # type: ignore[misc]
        """Minimal entity action packet (serverbound play)."""

        @staticmethod
        def get_id(context: Any) -> int:
            return 0x1C if context.protocol_later_eq(755) else 0x1B

        packet_name = "entity action"
        get_definition = staticmethod(
            lambda context: [
                {"entity_id": _PyVarInt},
                {"action_id": _PyVarInt},
                {"action_parameter": _PyVarInt},
            ]
        )

    class _HeldItemChangePacket(_Packet):  # type: ignore[misc]
        """Minimal held item change packet (serverbound play)."""

        @staticmethod
        def get_id(context: Any) -> int:
            return 0x25 if context.protocol_later_eq(755) else 0x23

        packet_name = "held item change"
        definition = [{"slot": _PyShort}]

else:
    _PlayerDiggingPacket = None  # type: ignore[misc]
    _EntityActionPacket = None  # type: ignore[misc]
    _HeldItemChangePacket = None  # type: ignore[misc]


class ActionExecutor:
    def __init__(
        self,
        connection: Any = None,
        session: "SessionManager | None" = None,
    ) -> None:
        self._connection = connection
        self._session = session
        self._position_provider: Callable[[], tuple[float, float, float]] | None = None
        self._yaw: float = 0.0
        self._pitch: float = 0.0

    @property
    def connection(self) -> Any:
        return self._connection

    @connection.setter
    def connection(self, value: Any) -> None:
        self._connection = value

    @property
    def session(self) -> "SessionManager | None":
        return self._session

    @session.setter
    def session(self, value: "SessionManager | None") -> None:
        self._session = value

    @property
    def position_provider(self) -> Callable[[], tuple[float, float, float]] | None:
        return self._position_provider

    @position_provider.setter
    def position_provider(
        self, value: Callable[[], tuple[float, float, float]] | None
    ) -> None:
        self._position_provider = value

    @property
    def yaw(self) -> float:
        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._yaw = float(value)

    @property
    def pitch(self) -> float:
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._pitch = float(value)

    def execute(self, action: np.ndarray) -> list[dict[str, Any]]:
        if not self.validate(action):
            return []

        packets: list[dict[str, Any]] = []
        packets.extend(self._translate_movement(action))
        packets.extend(self._translate_block_interaction(action))
        packets.extend(self._translate_inventory(action))
        return packets

    def send_packets(self, packets: list[dict[str, Any]]) -> None:
        for pkt in packets:
            self._send_packet(pkt)

    def validate(self, action: np.ndarray) -> bool:
        if action.shape != (ACTION_DIM,):
            logger.warning(
                "Invalid action shape: %s, expected (%d,)", action.shape, ACTION_DIM
            )
            return False
        if np.any(np.abs(action) > 1.0 + 1e-6):
            logger.warning("Action values out of bounds [-1, 1]")
            return False
        return True

    def _translate_movement(self, action: np.ndarray) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        dx = float(action[0]) - float(action[1])
        dz = float(action[3]) - float(action[2])
        dyaw = float(action[8]) - float(action[7])
        dpitch = float(action[10]) - float(action[9])

        if abs(dx) > ACTION_THRESHOLD or abs(dz) > ACTION_THRESHOLD:
            pkt: dict[str, Any] = {
                "type": "movement",
                "dx": dx,
                "dz": dz,
            }
            if abs(dyaw) > ACTION_THRESHOLD:
                pkt["dyaw"] = dyaw * ROTATION_SCALE
            if abs(dpitch) > ACTION_THRESHOLD:
                pkt["dpitch"] = dpitch * ROTATION_SCALE
            self._send_packet(pkt)
            packets.append(pkt)
        elif abs(dyaw) > ACTION_THRESHOLD or abs(dpitch) > ACTION_THRESHOLD:
            rot_pkt: dict[str, Any] = {"type": "rotation"}
            if abs(dyaw) > ACTION_THRESHOLD:
                rot_pkt["dyaw"] = dyaw * ROTATION_SCALE
            if abs(dpitch) > ACTION_THRESHOLD:
                rot_pkt["dpitch"] = dpitch * ROTATION_SCALE
            self._send_packet(rot_pkt)
            packets.append(rot_pkt)

        if float(action[4]) > ACTION_THRESHOLD:
            pkt_jump: dict[str, Any] = {"type": "jump"}
            self._send_packet(pkt_jump)
            packets.append(pkt_jump)
        if float(action[5]) > ACTION_THRESHOLD:
            pkt_sneak: dict[str, Any] = {"type": "sneak"}
            self._send_packet(pkt_sneak)
            packets.append(pkt_sneak)
        if float(action[6]) > ACTION_THRESHOLD:
            pkt_sprint: dict[str, Any] = {"type": "sprint"}
            self._send_packet(pkt_sprint)
            packets.append(pkt_sprint)
        return packets

    def _translate_block_interaction(self, action: np.ndarray) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        if float(action[11]) > ACTION_THRESHOLD:
            pkt_start: dict[str, Any] = {"type": "dig_start"}
            self._send_packet(pkt_start)
            packets.append(pkt_start)
            pkt_stop: dict[str, Any] = {"type": "dig_stop"}
            self._send_packet(pkt_stop)
            packets.append(pkt_stop)
        if float(action[12]) > ACTION_THRESHOLD:
            pkt_place: dict[str, Any] = {"type": "place_block"}
            self._send_packet(pkt_place)
            packets.append(pkt_place)
        if float(action[13]) > ACTION_THRESHOLD:
            pkt_pick: dict[str, Any] = {"type": "pick_block"}
            self._send_packet(pkt_pick)
            packets.append(pkt_pick)
        return packets

    def _translate_inventory(self, action: np.ndarray) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        if float(action[14]) > ACTION_THRESHOLD:
            pkt_craft: dict[str, Any] = {"type": "craft"}
            self._send_packet(pkt_craft)
            packets.append(pkt_craft)
        for i in range(10):
            if float(action[15 + i]) > ACTION_THRESHOLD:
                pkt_slot: dict[str, Any] = {"type": "select_slot", "slot": i}
                self._send_packet(pkt_slot)
                packets.append(pkt_slot)
                break
        return packets

    def _send_packet(self, packet: dict[str, Any]) -> None:
        if self._connection is None:
            return

        if not _HAS_PYCRAFT:
            logger.debug(
                "pyCraft not installed; dropping packet %s on the floor", packet
            )
            return

        try:
            kind = packet.get("type")
            if kind == "movement":
                self._write_movement(packet)
            elif kind == "rotation":
                self._write_rotation(packet)
            elif kind == "jump":
                self._write_entity_action(_ENTITY_ACTION_START_JUMP)
            elif kind == "sneak":
                self._write_entity_action(_ENTITY_ACTION_START_SNEAK)
            elif kind == "sprint":
                self._write_entity_action(_ENTITY_ACTION_START_SPRINT)
            elif kind in ("dig_start", "dig_stop"):
                status = (
                    _DIG_STATUS_STARTED
                    if kind == "dig_start"
                    else _DIG_STATUS_FINISHED
                )
                self._write_digging(status)
            elif kind == "place_block":
                self._write_block_placement()
            elif kind == "pick_block":
                self._write_digging(
                    _DIG_STATUS_START_DIGGING,
                    face=_DIG_STATUS_FACE_SELF_VALUE,
                )
            elif kind == "craft":
                logger.info(
                    "craft action received; server-side crafting via CloseWindowPacket"
                    " is not yet implemented, skipping packet"
                )
            elif kind == "select_slot":
                self._write_held_item_change(packet["slot"])
        except Exception as exc:
            logger.warning("Failed to write packet %s: %s", packet, exc)

    def _session_position(self) -> tuple[float, float, float]:
        if self._position_provider is not None:
            return self._position_provider()
        if self._session is None:
            return (0.0, 0.0, 0.0)
        return tuple(self._session.position)

    def _session_look(self) -> tuple[float, float]:
        return self._yaw, self._pitch

    def _write_movement(self, packet: dict[str, Any]) -> None:
        x, y, z = self._session_position()
        yaw, pitch = self._session_look()
        new_x = x + float(packet.get("dx", 0.0))
        new_z = z + float(packet.get("dz", 0.0))
        new_yaw = (yaw + float(packet.get("dyaw", 0.0))) % 360.0
        new_pitch = pitch + float(packet.get("dpitch", 0.0))
        pkt = _PositionAndLookPacket()
        pkt.x = new_x
        pkt.feet_y = y
        pkt.z = new_z
        pkt.yaw = new_yaw
        pkt.pitch = new_pitch
        pkt.on_ground = True
        self._connection.write_packet(pkt)
        if self._session is not None:
            self._session.position = (new_x, y, new_z)
            if hasattr(self._session, "yaw"):
                self._session.yaw = new_yaw
            if hasattr(self._session, "pitch"):
                self._session.pitch = new_pitch

    def _write_rotation(self, packet: dict[str, Any]) -> None:
        x, y, z = self._session_position()
        yaw, pitch = self._session_look()
        new_yaw = (yaw + float(packet.get("dyaw", 0.0))) % 360.0
        new_pitch = pitch + float(packet.get("dpitch", 0.0))
        pkt = _PositionAndLookPacket()
        pkt.x = x
        pkt.feet_y = y
        pkt.z = z
        pkt.yaw = new_yaw
        pkt.pitch = new_pitch
        pkt.on_ground = True
        self._connection.write_packet(pkt)
        if self._session is not None:
            if hasattr(self._session, "yaw"):
                self._session.yaw = new_yaw
            if hasattr(self._session, "pitch"):
                self._session.pitch = new_pitch

    def _write_entity_action(self, action_id: int) -> None:
        pkt = _EntityActionPacket()
        pkt.entity_id = 0
        pkt.action_id = action_id
        pkt.action_parameter = 0
        self._connection.write_packet(pkt)

    def _write_digging(self, status: int, face: int = 0) -> None:
        x, y, z = self._session_position()
        pkt = _PlayerDiggingPacket()
        pkt.status = status
        pkt.location = _Position(
            int(x), int(y), int(z),
        )
        pkt.face = face
        self._connection.write_packet(pkt)

    def _write_block_placement(self) -> None:
        x, y, z = self._session_position()
        pkt = _PlayerBlockPlacementPacket()
        pkt.location = _Position(int(x), int(y), int(z))
        pkt.face = _BlockFace.TOP
        pkt.hand = 0
        pkt.x = 0.5
        pkt.y = 0.0
        pkt.z = 0.5
        pkt.inside_block = False
        self._connection.write_packet(pkt)

    def _write_held_item_change(self, slot: int) -> None:
        pkt = _HeldItemChangePacket()
        pkt.slot = int(slot)
        self._connection.write_packet(pkt)
