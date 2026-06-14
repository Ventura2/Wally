"""Rate-limited action dispatch with backpressure and a background thread loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

ActionHandler = Callable[[object], Awaitable[None]]


class ActionThrottler:
    def __init__(
        self,
        handler: ActionHandler,
        interval: float | None = 0.05,
        max_queue_depth: int = 10,
    ) -> None:
        self._handler = handler
        self._interval = interval
        self._max_queue_depth = max_queue_depth
        self._queue: asyncio.Queue[object] | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._current_tps = 20.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def interval(self) -> float | None:
        return self._interval

    @property
    def max_queue_depth(self) -> int:
        return self._max_queue_depth

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_enabled(self) -> bool:
        return self._interval is not None and self._interval > 0

    def start(self) -> None:
        if not self.is_enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._start_loop(), self._loop)
        fut.result(timeout=5.0)

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._stop_loop(), self._loop
                )
                fut.result(timeout=5.0)
            except Exception:
                logger.exception("Error stopping throttler loop")
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                logger.debug("Loop already stopped")
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._loop = None
        self._task = None
        self._running = False
        self._queue = None

    def submit_sync(self, item: object) -> None:
        if not self.is_enabled:
            asyncio.run(self._invoke_handler(item))
            return
        if self._loop is None or not self._loop.is_running():
            self.start()
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(self.submit(item), self._loop)
        fut.result(timeout=5.0)

    async def submit(self, action: object) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._queue.qsize() >= self._max_queue_depth:
            logger.warning(
                "Action queue depth %d exceeds max %d",
                self._queue.qsize(),
                self._max_queue_depth,
            )
        await self._queue.put(action)

    def update_tps(self, tps: float) -> None:
        self._current_tps = max(1.0, min(tps, 20.0))

    async def _start_loop(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def _stop_loop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _invoke_handler(self, item: object) -> None:
        try:
            result = self._handler(item)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Throttler handler raised an exception")

    async def _process_loop(self) -> None:
        assert self._queue is not None
        while self._running:
            try:
                action = await asyncio.wait_for(
                    self._queue.get(), timeout=self._interval
                )
            except asyncio.TimeoutError:
                continue
            await self._invoke_handler(action)
            actual_interval = self._interval
            if self._current_tps < 20.0 and self._current_tps > 0:
                actual_interval = 1.0 / self._current_tps
            if actual_interval is not None:
                await asyncio.sleep(actual_interval)
