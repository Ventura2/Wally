from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestConnectionState:
    def test_initial_state_is_disconnected(self):
        from wally.deployer.connector import ConnectionState, ServerConnector

        connector = ServerConnector("localhost", 25565)
        assert connector.state is ConnectionState.DISCONNECTED

    def test_connection_property_is_none_initially(self):
        from wally.deployer.connector import ServerConnector

        connector = ServerConnector("localhost", 25565)
        assert connector.connection is None


class TestServerConnectorConnect:
    def test_successful_connect(self):
        from wally.deployer.connector import ConnectionState, ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.connect()

        assert connector.state is ConnectionState.CONNECTED
        assert connector.connection is mock_conn_instance
        mock_conn_cls.assert_called_once_with("localhost", 25565)
        mock_conn_instance.connect.assert_called_once()

    def test_connect_failure_raises_connection_error(self):
        from wally.deployer.connector import ConnectionState, ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_instance.connect.side_effect = OSError("refused")
        mock_conn_cls.return_value = mock_conn_instance

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    connector.connect()

        assert connector.state is ConnectionState.DISCONNECTED
        assert connector.connection is None

    def test_connect_fires_on_connect_callback(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        callback = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_connect(callback)
                connector.connect()

        callback.assert_called_once()

    def test_connect_failure_fires_on_error_callback(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_instance.connect.side_effect = OSError("refused")
        mock_conn_cls.return_value = mock_conn_instance

        callback = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_error(callback)
                with pytest.raises(ConnectionError):
                    connector.connect()

        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], OSError)

    def test_connect_raises_import_error_when_pycraft_missing(self):
        from wally.deployer.connector import ServerConnector

        with patch("wally.deployer.connector._HAS_PYCRAFT", False):
            connector = ServerConnector("localhost", 25565)
            with pytest.raises(ImportError, match="pyCraft is not installed"):
                connector.connect()


class TestServerConnectorDisconnect:
    def _make_connected_connector(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.connect()

        return connector, mock_conn_instance

    def test_disconnect_from_connected(self):
        from wally.deployer.connector import ConnectionState

        connector, mock_conn = self._make_connected_connector()
        connector.disconnect()

        assert connector.state is ConnectionState.DISCONNECTED
        assert connector.connection is None
        mock_conn.disconnect.assert_called_once()

    def test_disconnect_fires_callback(self):
        connector, _ = self._make_connected_connector()
        callback = MagicMock()
        connector.on_disconnect(callback)
        connector.disconnect()

        callback.assert_called_once()
        assert callback.call_args[0][0] == "Client disconnected"

    def test_disconnect_when_already_disconnected_is_noop(self):
        from wally.deployer.connector import ConnectionState, ServerConnector

        connector = ServerConnector("localhost", 25565)
        callback = MagicMock()
        connector.on_disconnect(callback)
        connector.disconnect()

        assert connector.state is ConnectionState.DISCONNECTED
        callback.assert_not_called()


class TestServerConnectorCallbacks:
    def test_register_multiple_callbacks(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_connect(cb1)
                connector.on_connect(cb2)
                connector.on_connect(cb3)
                connector.connect()

        cb1.assert_called_once()
        cb2.assert_called_once()
        cb3.assert_called_once()

    def test_all_disconnect_callbacks_fire(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        cb1 = MagicMock()
        cb2 = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_disconnect(cb1)
                connector.on_disconnect(cb2)
                connector.connect()
                connector.disconnect()

        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_callback_exception_does_not_prevent_others(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_cls.return_value = mock_conn_instance

        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_connect(bad_cb)
                connector.on_connect(good_cb)
                connector.connect()

        bad_cb.assert_called_once()
        good_cb.assert_called_once()

    def test_error_callback_exception_does_not_prevent_others(self):
        from wally.deployer.connector import ServerConnector

        mock_conn_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_conn_instance.connect.side_effect = OSError("refused")
        mock_conn_cls.return_value = mock_conn_instance

        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()

        with patch("wally.deployer.connector._PyCraftConnection", mock_conn_cls):
            with patch("wally.deployer.connector._HAS_PYCRAFT", True):
                connector = ServerConnector("localhost", 25565)
                connector.on_error(bad_cb)
                connector.on_error(good_cb)
                with pytest.raises(ConnectionError):
                    connector.connect()

        bad_cb.assert_called_once()
        good_cb.assert_called_once()
