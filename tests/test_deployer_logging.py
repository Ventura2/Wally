from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from wally.deployer.logging import (
    ActionLogger,
    JSONFormatter,
    PositionTracker,
    ServerEventLogger,
    setup_logging,
)


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestJSONFormatter:
    def test_output_is_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_required_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger", level=logging.WARNING, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert parsed["level"] == "WARNING"
        assert "logger" in parsed
        assert parsed["logger"] == "test.logger"
        assert "message" in parsed
        assert parsed["message"] == "test message"

    def test_exception_info_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error occurred", args=(), exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_extra_data_included(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="with data", args=(), exc_info=None,
        )
        record.extra_data = {"key": "value"}
        parsed = json.loads(formatter.format(record))
        assert "data" in parsed
        assert parsed["data"]["key"] == "value"

    def test_no_extra_data_field_when_absent(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="plain", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "data" not in parsed


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        log_dir = str(tmp_path / "logs" / "deploy")
        setup_logging(log_dir=log_dir)
        assert (tmp_path / "logs" / "deploy").is_dir()
        _cleanup_deployer_logger()

    def test_file_handler_added(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir)
        logger = logging.getLogger("deployer")
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1
        _cleanup_deployer_logger()

    def test_stdout_handler_added_when_enabled(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir, log_to_stdout=True)
        logger = logging.getLogger("deployer")
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        _cleanup_deployer_logger()

    def test_stdout_handler_not_added_when_disabled(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir, log_to_stdout=False)
        logger = logging.getLogger("deployer")
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0
        _cleanup_deployer_logger()


class TestActionLogger:
    def test_log_action_produces_correct_entry(self):
        action_logger = ActionLogger()
        capture = _CaptureHandler()
        action_logger._logger.addHandler(capture)
        action_logger._logger.setLevel(logging.DEBUG)
        try:
            action_logger.log_action([1.0, -2.0, 3.0], (10.0, 20.0, 30.0))
            assert len(capture.records) == 1
            record = capture.records[0]
            assert record.getMessage() == "action executed"
            assert record.extra_data["position"] == {"x": 10.0, "y": 20.0, "z": 30.0}
            assert record.extra_data["action_norm"] == 6.0
        finally:
            action_logger._logger.removeHandler(capture)

    def test_handles_numpy_arrays(self):
        action_logger = ActionLogger()
        capture = _CaptureHandler()
        action_logger._logger.addHandler(capture)
        action_logger._logger.setLevel(logging.DEBUG)
        try:
            mock_array = SimpleNamespace(tolist=lambda: [1.0, 2.0, 3.0])
            action_logger.log_action(mock_array, (0.0, 0.0, 0.0))
            assert len(capture.records) == 1
            assert capture.records[0].extra_data["action_norm"] == 6.0
        finally:
            action_logger._logger.removeHandler(capture)

    def test_handles_plain_lists(self):
        action_logger = ActionLogger()
        capture = _CaptureHandler()
        action_logger._logger.addHandler(capture)
        action_logger._logger.setLevel(logging.DEBUG)
        try:
            action_logger.log_action([0.5, -0.5, 1.0], (1.0, 2.0, 3.0))
            assert capture.records[0].extra_data["action_norm"] == 2.0
        finally:
            action_logger._logger.removeHandler(capture)


class TestPositionTracker:
    def test_logs_position_at_interval(self):
        async def _run():
            tracker = PositionTracker(interval=0.05)
            capture = _CaptureHandler()
            tracker._logger.addHandler(capture)
            tracker._logger.setLevel(logging.DEBUG)
            try:
                tracker.start(lambda: (1.0, 2.0, 3.0))
                await asyncio.sleep(0.2)
                tracker.stop()
                assert len(capture.records) >= 1
                data = capture.records[0].extra_data
                assert data["x"] == 1.0
                assert data["y"] == 2.0
                assert data["z"] == 3.0
            finally:
                tracker._logger.removeHandler(capture)

        asyncio.run(_run())

    def test_stop_cancels_tracking(self):
        async def _run():
            tracker = PositionTracker(interval=0.05)
            capture = _CaptureHandler()
            tracker._logger.addHandler(capture)
            tracker._logger.setLevel(logging.DEBUG)
            try:
                tracker.start(lambda: (0.0, 0.0, 0.0))
                await asyncio.sleep(0.1)
                tracker.stop()
                count_at_stop = len(capture.records)
                await asyncio.sleep(0.2)
                assert len(capture.records) == count_at_stop
                assert tracker._task is None
            finally:
                tracker._logger.removeHandler(capture)

        asyncio.run(_run())


class TestServerEventLogger:
    def test_chat_message_logging(self):
        event_logger = ServerEventLogger()
        capture = _CaptureHandler()
        event_logger._logger.addHandler(capture)
        event_logger._logger.setLevel(logging.DEBUG)
        try:
            event_logger.on_chat_message("hello world", "Steve")
            assert len(capture.records) == 1
            assert capture.records[0].getMessage() == "chat"
            assert capture.records[0].extra_data["sender"] == "Steve"
            assert capture.records[0].extra_data["message"] == "hello world"
        finally:
            event_logger._logger.removeHandler(capture)

    def test_player_join_logging(self):
        event_logger = ServerEventLogger()
        capture = _CaptureHandler()
        event_logger._logger.addHandler(capture)
        event_logger._logger.setLevel(logging.DEBUG)
        try:
            event_logger.on_player_join("Alex")
            assert len(capture.records) == 1
            assert capture.records[0].getMessage() == "player_join"
            assert capture.records[0].extra_data["username"] == "Alex"
        finally:
            event_logger._logger.removeHandler(capture)

    def test_death_logging(self):
        event_logger = ServerEventLogger()
        capture = _CaptureHandler()
        event_logger._logger.addHandler(capture)
        event_logger._logger.setLevel(logging.DEBUG)
        try:
            event_logger.on_death("Steve fell from a high place")
            assert len(capture.records) == 1
            assert capture.records[0].getMessage() == "death"
            assert capture.records[0].extra_data["message"] == (
                "Steve fell from a high place"
            )
        finally:
            event_logger._logger.removeHandler(capture)


def _cleanup_deployer_logger() -> None:
    logger = logging.getLogger("deployer")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
