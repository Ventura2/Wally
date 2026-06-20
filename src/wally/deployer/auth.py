"""Authentication for online and offline Minecraft connections."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from wally.deployer.config import DeployConfig

logger = logging.getLogger(__name__)

_TOKEN_CACHE_DIR = Path.home() / ".wally"
_TOKEN_CACHE_PATH = _TOKEN_CACHE_DIR / "auth_token.json"

_MSFT_AUTH_URL = "https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize"
_MSFT_TOKEN_URL = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
_MSFT_CLIENT_ID = "00000000402b5328"
_MSFT_SCOPES = "XboxLive.signin XboxLive.offline_access"


def _import_pycraft() -> Any:
    try:
        from minecraft import authentication  # type: ignore[import-not-found]
        from minecraft.networking.connection import (  # type: ignore[import-not-found]
            Connection,
        )
        from minecraft.networking.connection.connection import (  # type: ignore[import-not-found]
            ConnectionContext,
        )
    except ImportError as exc:
        raise ImportError(
            "pyCraft is required for Minecraft authentication. "
            "Install it with: pip install pyCraft"
        ) from exc
    return authentication, Connection, ConnectionContext


def authenticate_offline(username: str, host: str, port: int) -> Any:
    authentication, Connection, ConnectionContext = _import_pycraft()

    context = ConnectionContext.get_context()
    auth_token = authentication.AuthenticationToken()
    auth_token.profile = authentication.Profile.from_username(username)

    conn = Connection(
        address=host,
        port=port,
        auth_token=auth_token,
        context=context,
    )

    try:
        conn.connect()
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to {host}:{port} in offline mode: {exc}"
        ) from exc

    logger.info("Connected to %s:%d in offline mode as %s", host, port, username)
    return conn


def _load_cached_token() -> dict[str, str] | None:
    if not _TOKEN_CACHE_PATH.exists():
        return None

    try:
        data = json.loads(_TOKEN_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read cached token: %s", exc)
        return None

    required_keys = {"access_token", "refresh_token", "expires_at"}
    if not required_keys.issubset(data.keys()):
        logger.warning("Cached token is missing required keys")
        return None

    return data  # type: ignore[no-any-return]


def _save_token(token_data: dict[str, str]) -> None:
    _TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _TOKEN_CACHE_PATH.write_text(json.dumps(token_data, indent=2))
    logger.debug("Token saved to %s", _TOKEN_CACHE_PATH)


def _refresh_token(refresh_token: str) -> dict[str, str]:
    import requests

    resp = requests.post(
        _MSFT_TOKEN_URL,
        data={
            "client_id": _MSFT_CLIENT_ID,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "scope": _MSFT_SCOPES,
        },
        timeout=30,
    )
    resp.raise_for_status()
    token_json = resp.json()

    token_data = {
        "access_token": token_json["access_token"],
        "refresh_token": token_json.get("refresh_token", refresh_token),
        "expires_at": str(int(time.time()) + token_json.get("expires_in", 3600)),
    }
    _save_token(token_data)
    logger.info("Token refreshed successfully")
    return token_data


def _do_microsoft_oauth() -> dict[str, str]:
    import webbrowser

    import requests

    device_code_url = "https://login.microsoftonline.com/consumers/oauth2/v2.0/devicecode"

    device_resp = requests.post(
        device_code_url,
        data={
            "client_id": _MSFT_CLIENT_ID,
            "scope": _MSFT_SCOPES,
        },
        timeout=30,
    )
    device_resp.raise_for_status()
    device_data = device_resp.json()

    logger.info("Open this URL to authenticate: %s", device_data["verification_uri"])
    logger.info("Enter code: %s", device_data["user_code"])
    webbrowser.open(device_data["verification_uri"])

    interval = device_data.get("interval", 5)
    expires_in = device_data.get("expires_in", 900)
    deadline = time.time() + expires_in

    while time.time() < deadline:
        time.sleep(interval)
        token_resp = requests.post(
            _MSFT_TOKEN_URL,
            data={
                "client_id": _MSFT_CLIENT_ID,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_data["device_code"],
            },
            timeout=30,
        )
        token_json = token_resp.json()

        if "error" in token_json:
            if token_json["error"] == "authorization_pending":
                continue
            error_msg = token_json.get(
                "error_description", token_json["error"]
            )
            raise RuntimeError(f"OAuth error: {error_msg}")

        token_data = {
            "access_token": token_json["access_token"],
            "refresh_token": token_json.get("refresh_token", ""),
            "expires_at": str(
                int(time.time()) + token_json.get("expires_in", 3600)
            ),
        }
        _save_token(token_data)
        logger.info("Microsoft OAuth completed successfully")
        return token_data

    raise TimeoutError("Microsoft OAuth timed out waiting for user authorization")


def authenticate_online(host: str, port: int) -> Any:
    authentication, Connection, ConnectionContext = _import_pycraft()

    token_data = _load_cached_token()

    if token_data is None:
        logger.info("No cached token found, initiating Microsoft OAuth")
        token_data = _do_microsoft_oauth()
    elif int(token_data["expires_at"]) <= int(time.time()):
        logger.info("Cached token expired, attempting refresh")
        try:
            token_data = _refresh_token(token_data["refresh_token"])
        except Exception:
            logger.warning("Token refresh failed, initiating full OAuth")
            token_data = _do_microsoft_oauth()

    auth_token = authentication.AuthenticationToken()
    auth_token.access_token = token_data["access_token"]

    context = ConnectionContext.get_context()
    conn = Connection(
        address=host,
        port=port,
        auth_token=auth_token,
        context=context,
    )

    try:
        conn.connect()
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to {host}:{port} in online mode: {exc}"
        ) from exc

    logger.info("Connected to %s:%d in online mode", host, port)
    return conn


def authenticate(config: DeployConfig) -> Any:
    if config.auth_mode == "offline":
        return authenticate_offline(
            config.username, config.server_host, config.server_port
        )
    else:
        return authenticate_online(config.server_host, config.server_port)
