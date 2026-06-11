"""Translate action vectors to pyCraft packets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from minecraft.networking.packets import BlockFace as _BlockFace  # noqa: F401
    from minecraft.networking.packets import (  # noqa: F401
        PlayerDiggingPacket as _PlayerDiggingPacket,
    )
    from minecraft.networking.packets import (  # noqa: F401
        PlayerPositionAndLookPacket as _PlayerPositionAndLookPacket,
    )

    _HAS_PYCRAFT = True
except ImportError:
    _HAS_PYCRAFT = False

logger = logging.getLogger(__name__)

ACTION_THRESHOLD = 0.5
ACTION_DIM = 25


class ActionExecutor:
    def __init__(self, connection: Any = None) -> None:
        self._connection = connection

    @property
    def connection(self) -> Any:
        return self._connection

    @connection.setter
    def connection(self, value: Any) -> None:
        self._connection = value

    def execute(self, action: np.ndarray) -> list[dict[str, Any]]:
        if not self.validate(action):
            return []

        packets: list[dict[str, Any]] = []
        packets.extend(self._translate_movement(action))
        packets.extend(self._translate_block_interaction(action))
        packets.extend(self._translate_inventory(action))
        return packets

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
            packets.append({"type": "movement", "dx": dx, "dz": dz})
        if abs(dyaw) > ACTION_THRESHOLD:
            packets.append({"type": "rotation", "dyaw": dyaw * 10.0})
        if abs(dpitch) > ACTION_THRESHOLD:
            packets.append({"type": "rotation", "dpitch": dpitch * 10.0})
        if float(action[4]) > ACTION_THRESHOLD:
            packets.append({"type": "jump"})
        if float(action[5]) > ACTION_THRESHOLD:
            packets.append({"type": "sneak"})
        if float(action[6]) > ACTION_THRESHOLD:
            packets.append({"type": "sprint"})
        return packets

    def _translate_block_interaction(self, action: np.ndarray) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        if float(action[11]) > ACTION_THRESHOLD:
            packets.append({"type": "dig_start"})
            packets.append({"type": "dig_stop"})
        if float(action[12]) > ACTION_THRESHOLD:
            packets.append({"type": "place_block"})
        if float(action[13]) > ACTION_THRESHOLD:
            packets.append({"type": "pick_block"})
        return packets

    def _translate_inventory(self, action: np.ndarray) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        if float(action[14]) > ACTION_THRESHOLD:
            packets.append({"type": "craft"})
        for i in range(10):
            if float(action[15 + i]) > ACTION_THRESHOLD:
                packets.append({"type": "select_slot", "slot": i})
                break
        return packets
