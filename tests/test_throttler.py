from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from deployer.throttler import ActionThrottler


class TestActionThrottlerTiming:
    def test_default_interval_is_50ms(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler)
        assert throttler._interval == 0.05

    def test_actions_are_processed(self):
        async def _run():
            processed: list[object] = []

            async def handler(action: object) -> None:
                processed.append(action)

            throttler = ActionThrottler(handler=handler, interval=0.01)
            await throttler.start()
            await throttler.submit("action_1")
            await asyncio.sleep(0.1)
            await throttler.stop()
            assert "action_1" in processed

        asyncio.run(_run())

    def test_custom_interval(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.1)
        assert throttler._interval == 0.1


class TestActionThrottlerBackpressure:
    def test_warning_logged_when_queue_exceeds_max(self):
        async def _run():
            handler = AsyncMock()
            throttler = ActionThrottler(
                handler=handler, interval=10.0, max_queue_depth=2
            )
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
            await throttler.stop()
            warnings = [r for r in records if r.levelno == logging.WARNING]
            assert len(warnings) >= 1

        asyncio.run(_run())

    def test_actions_still_processed_under_backpressure(self):
        async def _run():
            processed: list[object] = []

            async def handler(action: object) -> None:
                processed.append(action)

            throttler = ActionThrottler(
                handler=handler, interval=0.01, max_queue_depth=2
            )
            await throttler.start()
            for i in range(5):
                await throttler.submit(f"action_{i}")
            await asyncio.sleep(0.5)
            await throttler.stop()
            assert len(processed) == 5

        asyncio.run(_run())


class TestActionThrottlerFlush:
    def test_queue_emptied_on_stop(self):
        async def _run():
            handler = AsyncMock()
            throttler = ActionThrottler(handler=handler, interval=10.0)
            await throttler.start()
            await throttler.submit("a1")
            await throttler.submit("a2")
            await throttler.submit("a3")
            await throttler.stop()
            assert throttler._queue.empty()

        asyncio.run(_run())

    def test_no_actions_processed_after_stop(self):
        async def _run():
            processed: list[object] = []

            async def handler(action: object) -> None:
                processed.append(action)

            throttler = ActionThrottler(handler=handler, interval=0.01)
            await throttler.start()
            await throttler.submit("before_stop")
            await asyncio.sleep(0.05)
            await throttler.stop()
            count_after_stop = len(processed)
            await asyncio.sleep(0.05)
            assert len(processed) == count_after_stop

        asyncio.run(_run())


class TestActionThrottlerAdaptiveTiming:
    def test_update_tps_changes_processing_rate(self):
        async def _run():
            handler = AsyncMock()
            throttler = ActionThrottler(handler=handler, interval=0.01)
            throttler.update_tps(5.0)
            assert throttler._current_tps == 5.0

        asyncio.run(_run())

    def test_tps_clamped_to_range(self):
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler)

        throttler.update_tps(0.5)
        assert throttler._current_tps == 1.0

        throttler.update_tps(100.0)
        assert throttler._current_tps == 20.0

        throttler.update_tps(10.0)
        assert throttler._current_tps == 10.0
