from typing import Any, Callable, Optional


class TrajectoryBuffer:
    def __init__(
        self,
        max_size: int = 1000,
        flush_callback: Optional[Callable[[list[dict[str, Any]]], None]] = None,
    ) -> None:
        self.max_size = max_size
        self.flush_callback = flush_callback
        self._buffer: list[dict[str, Any]] = []

    def add(self, transition: dict[str, Any]) -> bool:
        self._buffer.append(transition)
        if len(self._buffer) >= self.max_size:
            self.flush()
            return True
        return False

    def flush(self) -> int:
        count = len(self._buffer)
        if count > 0 and self.flush_callback is not None:
            self.flush_callback(self._buffer)
        self._buffer.clear()
        return count

    def shutdown(self) -> None:
        self.flush()

    def __len__(self) -> int:
        return len(self._buffer)
