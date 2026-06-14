from __future__ import annotations

import signal
from unittest.mock import AsyncMock, MagicMock, patch

from deployer.shutdown import GracefulShutdown


def _make_mocks() -> tuple[MagicMock, AsyncMock, MagicMock]:
    session_manager = MagicMock()
    throttler = AsyncMock()
    connector = MagicMock()
    return session_manager, throttler, connector


class TestGracefulShutdown:
    def test_register_installs_signal_handlers(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()

        with patch("deployer.shutdown.signal.signal") as mock_signal:
            shutdown.register(session_manager, throttler, connector)
            assert mock_signal.call_count == 2
            mock_signal.assert_any_call(signal.SIGINT, shutdown._handle_signal)
            mock_signal.assert_any_call(signal.SIGTERM, shutdown._handle_signal)

    def test_register_saves_original_handlers(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()

        with (
            patch(
                "deployer.shutdown.signal.getsignal",
                return_value="original_handler",
            ),
            patch("deployer.shutdown.signal.signal"),
        ):
            shutdown.register(session_manager, throttler, connector)
            assert shutdown._original_sigint == "original_handler"
            assert shutdown._original_sigterm == "original_handler"

    def test_unregister_restores_original_handlers(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()

        with (
            patch(
                "deployer.shutdown.signal.getsignal",
                return_value="original_handler",
            ),
            patch("deployer.shutdown.signal.signal") as mock_signal,
        ):
            shutdown.register(session_manager, throttler, connector)

        with patch("deployer.shutdown.signal.signal") as mock_signal:
            shutdown.unregister()
            mock_signal.assert_any_call(signal.SIGINT, "original_handler")
            mock_signal.assert_any_call(signal.SIGTERM, "original_handler")

    def test_unregister_noop_when_not_registered(self):
        shutdown = GracefulShutdown()
        with patch("deployer.shutdown.signal.signal") as mock_signal:
            shutdown.unregister()
            mock_signal.assert_not_called()


class TestShutdownSequence:
    def test_calls_session_manager_shutdown_first(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        call_order: list[str] = []
        session_manager.shutdown.side_effect = lambda: call_order.append("session")

        def _mock_stop() -> None:
            call_order.append("throttler")

        throttler.stop = _mock_stop
        connector.disconnect.side_effect = lambda: call_order.append("connector")

        shutdown._shutdown_sequence()

        assert call_order[0] == "session"

    def test_calls_throttler_stop_second(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        call_order: list[str] = []
        session_manager.shutdown.side_effect = lambda: call_order.append("session")

        def _mock_stop() -> None:
            call_order.append("throttler")

        throttler.stop = _mock_stop
        connector.disconnect.side_effect = lambda: call_order.append("connector")

        shutdown._shutdown_sequence()

        assert call_order[1] == "throttler"

    def test_calls_connector_disconnect_third(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        call_order: list[str] = []
        session_manager.shutdown.side_effect = lambda: call_order.append("session")

        def _mock_stop() -> None:
            call_order.append("throttler")

        throttler.stop = _mock_stop
        connector.disconnect.side_effect = lambda: call_order.append("connector")

        shutdown._shutdown_sequence()

        assert call_order[2] == "connector"

    def test_all_three_steps_called_in_order(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        call_order: list[str] = []
        session_manager.shutdown.side_effect = lambda: call_order.append("session")

        def _mock_stop() -> None:
            call_order.append("throttler")

        throttler.stop = _mock_stop
        connector.disconnect.side_effect = lambda: call_order.append("connector")

        shutdown._shutdown_sequence()

        assert call_order == ["session", "throttler", "connector"]

    def test_continues_when_session_manager_raises(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        session_manager.shutdown.side_effect = RuntimeError("save failed")

        def _mock_stop() -> None:
            pass

        throttler.stop = _mock_stop

        shutdown._shutdown_sequence()

        connector.disconnect.assert_called_once()

    def test_continues_when_throttler_raises(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        def _mock_stop() -> None:
            raise RuntimeError("flush failed")

        throttler.stop = _mock_stop

        shutdown._shutdown_sequence()

        session_manager.shutdown.assert_called_once()
        connector.disconnect.assert_called_once()

    def test_continues_when_connector_raises(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        connector.disconnect.side_effect = RuntimeError("disconnect failed")

        def _mock_stop() -> None:
            pass

        throttler.stop = _mock_stop

        shutdown._shutdown_sequence()

        session_manager.shutdown.assert_called_once()


class TestSignalHandling:
    def test_sigint_triggers_shutdown_sequence(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        def _mock_stop() -> None:
            pass

        throttler.stop = _mock_stop

        with patch("deployer.shutdown.sys.exit") as mock_exit:
            shutdown._handle_signal(signal.SIGINT, None)

        session_manager.shutdown.assert_called_once()
        connector.disconnect.assert_called_once()
        mock_exit.assert_called_once_with(0)

    def test_sigterm_triggers_shutdown_sequence(self):
        shutdown = GracefulShutdown()
        session_manager, throttler, connector = _make_mocks()
        shutdown._session_manager = session_manager
        shutdown._throttler = throttler
        shutdown._connector = connector

        def _mock_stop() -> None:
            pass

        throttler.stop = _mock_stop

        with patch("deployer.shutdown.sys.exit") as mock_exit:
            shutdown._handle_signal(signal.SIGTERM, None)

        session_manager.shutdown.assert_called_once()
        connector.disconnect.assert_called_once()
        mock_exit.assert_called_once_with(0)
