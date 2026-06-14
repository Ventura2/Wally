from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock

from deployer.throttler import ActionThrottler


class TestActionThrottlerTiming:
    def test_default_interval_is_50ms(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler)
        assert throttler.interval == 0.05

    def test_actions_are_processed(self):
        processed: list[object] = []

        async def handler(action: object) -> None:
            processed.append(action)

        throttler = ActionThrottler(handler=handler, interval=0.01)
        throttler.start()
        throttler.submit_sync("action_1")
        time.sleep(0.1)
        throttler.stop()
        assert "action_1" in processed

    def test_custom_interval(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.1)
        assert throttler.interval == 0.1

    def test_disabled_interval_skips_thread(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=None)
        throttler.start()
        throttler.submit_sync("a")
        throttler.stop()
        assert handler.await_count == 1


class TestActionThrottlerBackpressure:
    def test_warning_logged_when_queue_exceeds_max(self):
        async def _run():
            handler = AsyncMock()
            throttler = ActionThrottler(
                handler=handler, interval=10.0, max_queue_depth=2
            )
            throttler._queue = asyncio.Queue()
            logger = logging.getLogger("deployer.throttler")
            records: list[logging.LogRecord] = []
            handler_log = logging.Handler()
            handler_log.emit = lambda record: records.append(record)
            logger.addHandler(handler_log)
            logger.setLevel(logging.WARNING)
            try:
                await throttler.submit("a1")
                await throttler.submit("a2")
                await throttler.submit("a3")
            finally:
                logger.removeHandler(handler_log)
            throttler._queue = None
            warnings = [r for r in records if r.levelno == logging.WARNING]
            assert len(warnings) >= 1

        asyncio.run(_run())

    def test_actions_still_processed_under_backpressure(self):
        processed: list[object] = []

        async def handler(action: object) -> None:
            processed.append(action)

        throttler = ActionThrottler(
            handler=handler, interval=0.01, max_queue_depth=2
        )
        throttler.start()
        for i in range(5):
            throttler.submit_sync(f"action_{i}")
        time.sleep(0.5)
        throttler.stop()
        assert len(processed) == 5


class TestActionThrottlerFlush:
    def test_queue_emptied_on_stop(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=10.0)
        throttler.start()
        throttler.submit_sync("a1")
        throttler.submit_sync("a2")
        throttler.submit_sync("a3")
        throttler.stop()
        assert throttler._queue is None

    def test_no_actions_processed_after_stop(self):
        processed: list[object] = []

        async def handler(action: object) -> None:
            processed.append(action)

        throttler = ActionThrottler(handler=handler, interval=0.01)
        throttler.start()
        throttler.submit_sync("before_stop")
        time.sleep(0.05)
        throttler.stop()
        count_after_stop = len(processed)
        time.sleep(0.05)
        assert len(processed) == count_after_stop


class TestActionThrottlerAdaptiveTiming:
    def test_update_tps_changes_processing_rate(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.01)
        throttler.update_tps(5.0)
        assert throttler._current_tps == 5.0

    def test_tps_clamped_to_range(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler)

        throttler.update_tps(0.5)
        assert throttler._current_tps == 1.0

        throttler.update_tps(100.0)
        assert throttler._current_tps == 20.0

        throttler.update_tps(10.0)
        assert throttler._current_tps == 10.0


class TestActionThrottlerLifecycle:
    def test_start_is_idempotent(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.start()
        thread1 = throttler._thread
        throttler.start()
        thread2 = throttler._thread
        throttler.stop()
        assert thread1 is thread2

    def test_stop_when_not_started_is_noop(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.stop()
        assert throttler._thread is None

    def test_double_stop_is_safe(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.start()
        throttler.stop()
        throttler.stop()
        assert throttler._thread is None
