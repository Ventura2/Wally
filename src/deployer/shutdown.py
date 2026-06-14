"""Signal handling and graceful shutdown sequence."""

from __future__ import annotations

import logging
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)


class GracefulShutdown:
    def __init__(self) -> None:
        self._handlers: list[Any] = []
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

    def register(
        self,
        session_manager: Any,
        throttler: Any,
        connector: Any,
    ) -> None:
        self._session_manager = session_manager
        self._throttler = throttler
        self._connector = connector
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        self._shutdown_sequence()
        sys.exit(0)

    def _shutdown_sequence(self) -> None:
        logger.info("Step 1: Saving agent state...")
        try:
            self._session_manager.shutdown()
        except Exception:
            logger.exception("Error saving agent state during shutdown")

        logger.info("Step 2: Flushing action queue...")
        try:
            self._throttler.stop()
        except Exception:
            logger.exception("Error flushing action queue during shutdown")

        logger.info("Step 3: Disconnecting from server...")
        try:
            self._connector.disconnect()
        except Exception:
            logger.exception("Error disconnecting during shutdown")

        logger.info("Graceful shutdown complete.")

    def unregister(self) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
