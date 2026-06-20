from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from wally.deployer.config import ReconnectConfig
from wally.deployer.session import SessionManager


def _make_connector() -> MagicMock:
    connector = MagicMock()
    connector.on_disconnect = MagicMock()
    connector.connect = MagicMock()
    connector.disconnect = MagicMock()
    return connector


class TestSessionManagerJoin:
    def test_join_connects_to_server(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "checkpoint.json"
        )
        session.join()
        connector.connect.assert_called_once()

    def test_join_restores_state_if_checkpoint_exists(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps({
            "position": [10.0, 64.0, -30.0],
            "inventory": {"diamond": 5},
            "goal_progress": {"stage": 2},
        }))

        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session.join()

        assert session.position == (10.0, 64.0, -30.0)
        assert session.inventory == {"diamond": 5}
        assert session.goal_progress == {"stage": 2}

    def test_join_no_restore_when_no_checkpoint(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "nope.json"
        )
        session.join()
        assert session.position == (0.0, 0.0, 0.0)
        assert session.inventory == {}
        assert session.goal_progress == {}

    def test_join_registers_disconnect_callback(self, tmp_path: Path):
        connector = _make_connector()
        SessionManager(connector, checkpoint_path=tmp_path / "cp.json")
        connector.on_disconnect.assert_called_once()


class TestSessionManagerShutdown:
    def test_shutdown_saves_state_and_disconnects(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session.position = (1.0, 2.0, 3.0)
        session.inventory = {"cobble": 10}
        session.goal_progress = {"task": "mine"}

        session.shutdown()

        connector.disconnect.assert_called_once()
        data = json.loads(checkpoint.read_text())
        assert data["position"] == [1.0, 2.0, 3.0]
        assert data["inventory"] == {"cobble": 10}
        assert data["goal_progress"] == {"task": "mine"}

    def test_shutdown_sets_shutting_down_flag(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "cp.json"
        )
        assert session._shutting_down is False
        session.shutdown()
        assert session._shutting_down is True

    def test_shutdown_prevents_reconnection(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "cp.json"
        )
        session.shutdown()
        disconnect_cb = connector.on_disconnect.call_args[0][0]
        disconnect_cb("Client disconnected")
        assert session._shutting_down is True


class TestReconnectionBackoff:
    def test_exponential_backoff_timing(self, tmp_path: Path):
        connector = _make_connector()
        connector.connect.side_effect = ConnectionError("fail")
        cfg = ReconnectConfig(
            max_attempts=4,
            initial_backoff_s=1.0,
            max_backoff_s=60.0,
            backoff_multiplier=2.0,
        )
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=tmp_path / "cp.json"
        )

        sleep_times: list[float] = []

        async def fake_sleep(t: float) -> None:
            sleep_times.append(t)

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep", side_effect=fake_sleep):
                await session._reconnect_loop()

        asyncio.run(_run())
        assert sleep_times == [1.0, 2.0, 4.0, 8.0]

    def test_max_backoff_cap(self, tmp_path: Path):
        connector = _make_connector()
        connector.connect.side_effect = ConnectionError("fail")
        cfg = ReconnectConfig(
            max_attempts=8,
            initial_backoff_s=1.0,
            max_backoff_s=10.0,
            backoff_multiplier=2.0,
        )
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=tmp_path / "cp.json"
        )

        sleep_times: list[float] = []

        async def fake_sleep(t: float) -> None:
            sleep_times.append(t)

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep", side_effect=fake_sleep):
                await session._reconnect_loop()

        asyncio.run(_run())
        assert sleep_times == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]

    def test_max_attempts_limit(self, tmp_path: Path):
        connector = _make_connector()
        connector.connect.side_effect = ConnectionError("fail")
        cfg = ReconnectConfig(max_attempts=3, initial_backoff_s=0.01)
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=tmp_path / "cp.json"
        )

        async def fake_sleep(t: float) -> None:
            pass

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep", side_effect=fake_sleep):
                await session._reconnect_loop()

        asyncio.run(_run())
        assert connector.connect.call_count == 3

    def test_successful_reconnection_restores_state(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps({
            "position": [5.0, 70.0, 15.0],
            "inventory": {"iron": 3},
            "goal_progress": {"step": 1},
        }))

        connector = _make_connector()
        connector.connect.side_effect = [
            ConnectionError("fail"),
            None,
        ]
        cfg = ReconnectConfig(max_attempts=3, initial_backoff_s=0.01)
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=checkpoint
        )

        session.position = (0.0, 0.0, 0.0)
        session.inventory = {}

        async def fake_sleep(t: float) -> None:
            pass

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep", side_effect=fake_sleep):
                await session._reconnect_loop()

        asyncio.run(_run())
        assert session.position == (5.0, 70.0, 15.0)
        assert session.inventory == {"iron": 3}
        assert session.goal_progress == {"step": 1}

    def test_exhausted_reconnection_saves_state(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        connector = _make_connector()
        connector.connect.side_effect = ConnectionError("fail")
        cfg = ReconnectConfig(max_attempts=2, initial_backoff_s=0.01)
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=checkpoint
        )
        session.position = (99.0, 1.0, 2.0)

        async def fake_sleep(t: float) -> None:
            pass

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep", side_effect=fake_sleep):
                await session._reconnect_loop()

        asyncio.run(_run())
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert data["position"] == [99.0, 1.0, 2.0]

    def test_unexpected_disconnect_triggers_reconnect(self, tmp_path: Path):
        connector = _make_connector()
        cfg = ReconnectConfig(max_attempts=1, initial_backoff_s=0.01)
        session = SessionManager(
            connector, reconnect_config=cfg, checkpoint_path=tmp_path / "cp.json"
        )

        disconnect_cb = connector.on_disconnect.call_args[0][0]

        async def _run():
            with patch("wally.deployer.session.asyncio.sleep"):
                with patch("wally.deployer.session.asyncio.ensure_future") as mock_ensure:
                    mock_ensure.return_value = MagicMock()
                    disconnect_cb("server closed")
                    mock_ensure.assert_called_once()
                    coro = mock_ensure.call_args[0][0]
                    coro.close()

        asyncio.run(_run())
        assert session._shutting_down is False

    def test_no_reconnect_during_shutdown(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "cp.json"
        )
        session.shutdown()
        disconnect_cb = connector.on_disconnect.call_args[0][0]

        with patch("wally.deployer.session.asyncio.ensure_future") as mock_ensure:
            mock_ensure.return_value = MagicMock()
            disconnect_cb("Client disconnected")
            mock_ensure.assert_not_called()


class TestStatePersistence:
    def test_save_state_writes_json(self, tmp_path: Path):
        checkpoint = tmp_path / "sub" / "checkpoint.json"
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session.position = (1.0, 2.0, 3.0)
        session._save_state()

        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert data["position"] == [1.0, 2.0, 3.0]

    def test_restore_state_reads_json(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps({
            "position": [7.0, 8.0, 9.0],
            "inventory": {"stick": 64},
            "goal_progress": {"done": True},
        }))
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session._restore_state()

        assert session.position == (7.0, 8.0, 9.0)
        assert session.inventory == {"stick": 64}
        assert session.goal_progress == {"done": True}

    def test_restore_with_missing_file_is_noop(self, tmp_path: Path):
        connector = _make_connector()
        session = SessionManager(
            connector, checkpoint_path=tmp_path / "missing.json"
        )
        session.position = (1.0, 1.0, 1.0)
        session._restore_state()
        assert session.position == (1.0, 1.0, 1.0)
        assert session.inventory == {}

    def test_all_fields_saved_and_restored(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session.position = (100.5, -20.0, 300.25)
        session.inventory = {"diamond_sword": 1, "bread": 32}
        session.goal_progress = {"phase": "gather", "count": 42}
        session._save_state()

        connector2 = _make_connector()
        session2 = SessionManager(connector2, checkpoint_path=checkpoint)
        session2._restore_state()

        assert session2.position == (100.5, -20.0, 300.25)
        assert session2.inventory == {"diamond_sword": 1, "bread": 32}
        assert session2.goal_progress == {"phase": "gather", "count": 42}

    def test_restore_with_missing_keys_uses_defaults(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps({"position": [0.0, 0.0, 0.0]}))
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session._restore_state()
        assert session.inventory == {}
        assert session.goal_progress == {}

    def test_save_creates_parent_directories(self, tmp_path: Path):
        checkpoint = tmp_path / "a" / "b" / "c" / "checkpoint.json"
        connector = _make_connector()
        session = SessionManager(connector, checkpoint_path=checkpoint)
        session._save_state()
        assert checkpoint.exists()
