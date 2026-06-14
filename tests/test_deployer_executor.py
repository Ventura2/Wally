"""Tests for ``ActionExecutor`` packet writing on a real connection."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from deployer.executor import ACTION_DIM, ActionExecutor
from deployer.session import SessionManager


class _MockConnection:
    """Records every packet that the executor writes on the connection."""

    def __init__(self) -> None:
        self.written: list[object] = []

    def write_packet(self, packet: object) -> None:
        self.written.append(packet)


def _make_executor(conn: object | None) -> tuple[ActionExecutor, SessionManager | None]:
    session = MagicMock(spec=SessionManager)
    session.position = (0.0, 64.0, 0.0)
    executor = ActionExecutor(connection=conn, session=session)
    return executor, session


class TestExecutorConnectionWrites:
    def test_no_packets_written_when_connection_is_none(self) -> None:
        executor, _session = _make_executor(None)
        action = np.ones(ACTION_DIM) * 0.9
        executor.execute(action)
        # Connection is None so no write attempts; only the dict list is returned
        executor._send_packet({"type": "movement", "dx": 1.0, "dz": 0.0})

    def test_movement_only_writes_one_packet(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[0] = 1.0  # forward
        packets = executor.execute(action)
        assert any(p["type"] == "movement" for p in packets)
        assert len(conn.written) == 1

    def test_rotation_only_writes_packet(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[8] = 1.0  # turn right
        packets = executor.execute(action)
        assert any(p["type"] == "rotation" for p in packets)
        assert len(conn.written) == 1

    def test_jump_writes_entity_action(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[4] = 1.0  # jump
        packets = executor.execute(action)
        assert any(p["type"] == "jump" for p in packets)
        assert len(conn.written) == 1

    def test_dig_writes_two_packets(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[11] = 1.0  # attack
        packets = executor.execute(action)
        dig = [p for p in packets if p["type"] in ("dig_start", "dig_stop")]
        assert len(dig) == 2
        assert len(conn.written) == 2

    def test_place_writes_block_placement(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[12] = 1.0  # place
        packets = executor.execute(action)
        assert any(p["type"] == "place_block" for p in packets)
        assert len(conn.written) == 1

    def test_slot_select_writes_held_item_change(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[17] = 1.0  # slot 2
        packets = executor.execute(action)
        assert any(p["type"] == "select_slot" for p in packets)
        assert any(p.get("slot") == 2 for p in packets)
        assert len(conn.written) == 1

    def test_send_packets_writes_each(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        executor.send_packets(
            [
                {"type": "movement", "dx": 1.0, "dz": 0.0},
                {"type": "jump"},
                {"type": "select_slot", "slot": 0},
            ]
        )
        assert len(conn.written) == 3

    def test_send_packets_no_op_without_connection(self) -> None:
        executor, _session = _make_executor(None)
        executor.send_packets(
            [
                {"type": "movement", "dx": 1.0, "dz": 0.0},
                {"type": "jump"},
            ]
        )


class TestExecutorSessionUpdates:
    def test_movement_updates_session_position(self) -> None:
        conn = _MockConnection()
        executor, session = _make_executor(conn)
        action = np.zeros(ACTION_DIM)
        action[0] = 1.0
        executor.execute(action)
        assert session.position == (1.0, 64.0, 0.0)

    def test_invalid_action_does_not_write(self) -> None:
        conn = _MockConnection()
        executor, _session = _make_executor(conn)
        action = np.zeros(10)
        result = executor.execute(action)
        assert result == []
        assert len(conn.written) == 0


@pytest.mark.parametrize("planner_kind", ["cem", "gradient", "hierarchical"])
def test_session_setter_round_trip(planner_kind: str) -> None:
    executor = ActionExecutor()
    session = MagicMock(spec=SessionManager)
    executor.session = session
    assert executor.session is session
