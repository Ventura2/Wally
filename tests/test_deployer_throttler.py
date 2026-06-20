"""Tests for ``ActionThrottler`` sync submit, bypass, and lifecycle."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from wally.deployer.throttler import ActionThrottler


class TestSubmitSync:
    def test_submit_sync_calls_handler_immediately_when_disabled(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=None)
        throttler.submit_sync("a")
        assert handler.await_count == 1
        assert handler.await_args[0][0] == "a"
        throttler.stop()

    def test_submit_sync_calls_handler_when_interval_zero(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0)
        throttler.submit_sync("a")
        assert handler.await_count == 1

    def test_submit_sync_at_50ms_spacing(self) -> None:
        timestamps: list[float] = []

        async def handler(item: object) -> None:
            timestamps.append(time.monotonic())

        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.start()
        for i in range(3):
            throttler.submit_sync(f"item_{i}")
        time.sleep(0.4)
        throttler.stop()

        assert len(timestamps) >= 3
        deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for delta in deltas:
            assert delta >= 0.04

    def test_submit_sync_invokes_handler(self) -> None:
        seen: list[object] = []

        async def handler(item: object) -> None:
            seen.append(item)

        throttler = ActionThrottler(handler=handler, interval=0.01)
        throttler.start()
        throttler.submit_sync("a")
        throttler.submit_sync("b")
        time.sleep(0.1)
        throttler.stop()
        assert "a" in seen
        assert "b" in seen


class TestStopFlushes:
    def test_stop_flushes_queue(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=10.0)
        throttler.start()
        throttler.submit_sync("a")
        throttler.submit_sync("b")
        throttler.stop()
        assert throttler._queue is None

    def test_stop_does_not_run_handler(self) -> None:
        seen: list[object] = []

        async def handler(item: object) -> None:
            seen.append(item)

        throttler = ActionThrottler(handler=handler, interval=10.0)
        throttler.start()
        throttler.submit_sync("a")
        throttler.stop()
        count = len(seen)
        time.sleep(0.05)
        assert len(seen) == count


class TestBackgroundTaskLifecycle:
    def test_start_is_idempotent(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.start()
        thread1 = throttler._thread
        throttler.start()
        thread2 = throttler._thread
        throttler.stop()
        assert thread1 is thread2

    def test_stop_when_not_started_is_safe(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.stop()
        assert throttler._thread is None

    def test_double_stop_is_safe(self) -> None:
        handler = AsyncMock()
        throttler = ActionThrottler(handler=handler, interval=0.05)
        throttler.start()
        throttler.stop()
        throttler.stop()
        assert throttler._thread is None


class TestAsyncSubmit:
    def test_async_submit_works_on_existing_queue(self) -> None:
        async def _run() -> None:
            handler = AsyncMock()
            throttler = ActionThrottler(handler=handler, interval=0.05)
            throttler._queue = asyncio.Queue()
            await throttler.submit("a")
            await throttler.submit("b")
            assert throttler._queue.qsize() == 2

        asyncio.run(_run())
