"""Rate-limited action dispatch with backpressure."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

ActionHandler = Callable[[object], Awaitable[None]]


class ActionThrottler:
    def __init__(
        self,
        handler: ActionHandler,
        interval: float = 0.05,
        max_queue_depth: int = 10,
    ) -> None:
        self._handler = handler
        self._interval = interval
        self._max_queue_depth = max_queue_depth
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._current_tps = 20.0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def submit(self, action: object) -> None:
        if self._queue.qsize() >= self._max_queue_depth:
            logger.warning(
                "Action queue depth %d exceeds max %d",
                self._queue.qsize(),
                self._max_queue_depth,
            )
        await self._queue.put(action)

    def update_tps(self, tps: float) -> None:
        self._current_tps = max(1.0, min(tps, 20.0))

    async def _process_loop(self) -> None:
        while self._running:
            try:
                action = await asyncio.wait_for(
                    self._queue.get(), timeout=self._interval
                )
            except asyncio.TimeoutError:
                continue
            await self._handler(action)
            actual_interval = self._interval
            if self._current_tps < 20.0 and self._current_tps > 0:
                actual_interval = 1.0 / self._current_tps
            await asyncio.sleep(actual_interval)
