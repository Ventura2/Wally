from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from deployer.config import DeployConfig


class TestOfflineAuth:
    @patch("deployer.auth._import_pycraft")
    def test_offline_auth_creates_connection(self, mock_import):
        mock_auth_module = MagicMock()
        mock_auth_token = MagicMock()
        mock_auth_module.AuthenticationToken.return_value = mock_auth_token
        mock_auth_module.Profile.from_username.return_value = MagicMock()

        mock_connection_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_connection_cls.return_value = mock_conn_instance

        mock_context_cls = MagicMock()

        mock_import.return_value = (
            mock_auth_module,
            mock_connection_cls,
            mock_context_cls,
        )

        from deployer.auth import authenticate_offline

        result = authenticate_offline("TestPlayer", "localhost", 25565)

        mock_auth_module.Profile.from_username.assert_called_once_with("TestPlayer")
        mock_connection_cls.assert_called_once()
        call_kwargs = mock_connection_cls.call_args
        assert call_kwargs.kwargs["address"] == "localhost"
        assert call_kwargs.kwargs["port"] == 25565
        mock_conn_instance.connect.assert_called_once()
        assert result is mock_conn_instance

    @patch("deployer.auth._import_pycraft")
    def test_offline_auth_connection_failure(self, mock_import):
        mock_auth_module = MagicMock()
        mock_auth_module.Profile.from_username.return_value = MagicMock()

        mock_conn_instance = MagicMock()
        mock_conn_instance.connect.side_effect = OSError("Connection refused")
        mock_connection_cls = MagicMock(return_value=mock_conn_instance)
        mock_context_cls = MagicMock()

        mock_import.return_value = (
            mock_auth_module,
            mock_connection_cls,
            mock_context_cls,
        )

        from deployer.auth import authenticate_offline

        with pytest.raises(ConnectionError, match="Failed to connect"):
            authenticate_offline("TestPlayer", "localhost", 25565)

    def test_offline_auth_raises_import_error(self):
        from deployer.auth import authenticate_offline

        msg = (
            "pyCraft is required for Minecraft authentication. "
            "Install it with: pip install pyCraft"
        )
        with patch(
            "deployer.auth._import_pycraft",
            side_effect=ImportError(msg),
        ):
            with pytest.raises(ImportError, match="pyCraft is required"):
                authenticate_offline("TestPlayer", "localhost", 25565)


class TestOnlineAuth:
    @patch("deployer.auth._import_pycraft")
    @patch("deployer.auth._do_microsoft_oauth")
    @patch("deployer.auth._load_cached_token", return_value=None)
    def test_missing_token_triggers_full_oauth(
        self, mock_load, mock_oauth, mock_import
    ):
        token_data = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_at": str(int(time.time()) + 3600),
        }
        mock_oauth.return_value = token_data

        mock_auth_module = MagicMock()
        mock_connection_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_connection_cls.return_value = mock_conn_instance
        mock_context_cls = MagicMock()
        mock_import.return_value = (
            mock_auth_module,
            mock_connection_cls,
            mock_context_cls,
        )

        from deployer.auth import authenticate_online

        result = authenticate_online("localhost", 25565)

        mock_oauth.assert_called_once()
        mock_connection_cls.assert_called_once()
        assert result is mock_conn_instance

    @patch("deployer.auth._import_pycraft")
    @patch("deployer.auth._refresh_token")
    @patch("deployer.auth._load_cached_token")
    def test_expired_token_triggers_refresh(
        self, mock_load, mock_refresh, mock_import
    ):
        expired_token = {
            "access_token": "old_access",
            "refresh_token": "old_refresh",
            "expires_at": str(int(time.time()) - 100),
        }
        refreshed_token = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_at": str(int(time.time()) + 3600),
        }
        mock_load.return_value = expired_token
        mock_refresh.return_value = refreshed_token

        mock_auth_module = MagicMock()
        mock_connection_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_connection_cls.return_value = mock_conn_instance
        mock_context_cls = MagicMock()
        mock_import.return_value = (
            mock_auth_module,
            mock_connection_cls,
            mock_context_cls,
        )

        from deployer.auth import authenticate_online

        result = authenticate_online("localhost", 25565)

        mock_refresh.assert_called_once_with("old_refresh")
        assert result is mock_conn_instance

    @patch("deployer.auth._import_pycraft")
    @patch("deployer.auth._do_microsoft_oauth")
    @patch("deployer.auth._load_cached_token")
    def test_cached_token_reused_when_valid(
        self, mock_load, mock_oauth, mock_import
    ):
        valid_token = {
            "access_token": "valid_access",
            "refresh_token": "valid_refresh",
            "expires_at": str(int(time.time()) + 3600),
        }
        mock_load.return_value = valid_token

        mock_auth_module = MagicMock()
        mock_connection_cls = MagicMock()
        mock_conn_instance = MagicMock()
        mock_connection_cls.return_value = mock_conn_instance
        mock_context_cls = MagicMock()
        mock_import.return_value = (
            mock_auth_module,
            mock_connection_cls,
            mock_context_cls,
        )

        from deployer.auth import authenticate_online

        result = authenticate_online("localhost", 25565)

        mock_oauth.assert_not_called()
        assert result is mock_conn_instance

    def test_save_and_load_token(self, tmp_path):
        from deployer.auth import _load_cached_token, _save_token

        token_path = tmp_path / "auth_token.json"

        token_data = {
            "access_token": "test_access",
            "refresh_token": "test_refresh",
            "expires_at": "1234567890",
        }

        with patch("deployer.auth._TOKEN_CACHE_PATH", token_path), \
             patch("deployer.auth._TOKEN_CACHE_DIR", tmp_path):
            _save_token(token_data)
            loaded = _load_cached_token()

        assert loaded == token_data

    @patch("deployer.auth._load_cached_token", return_value=None)
    def test_load_token_returns_none_when_missing(self, mock_exists):
        from deployer.auth import _load_cached_token

        with patch("deployer.auth._TOKEN_CACHE_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = _load_cached_token()

        assert result is None


class TestAuthSelection:
    @patch("deployer.auth.authenticate_offline")
    def test_offline_mode_selects_offline_auth(self, mock_offline):
        mock_offline.return_value = MagicMock()
        config = DeployConfig(auth_mode="offline", username="TestPlayer")

        from deployer.auth import authenticate

        result = authenticate(config)

        mock_offline.assert_called_once_with("TestPlayer", "localhost", 25565)
        assert result is mock_offline.return_value

    @patch("deployer.auth.authenticate_online")
    def test_online_mode_selects_online_auth(self, mock_online):
        mock_online.return_value = MagicMock()
        config = DeployConfig(
            auth_mode="online",
            server_host="mc.example.com",
            server_port=25566,
        )

        from deployer.auth import authenticate

        result = authenticate(config)

        mock_online.assert_called_once_with("mc.example.com", 25566)
        assert result is mock_online.return_value
