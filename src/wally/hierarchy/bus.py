"""Thread-safe message bus for inter-layer communication.

Each layer has its own input and output queues. The bus is intentionally
minimal: a producer (the layer below, or the planner above) drops a
:class:`LayerMessage` into the queue; the consumer (the layer runtime)
drains it non-blockingly. Bounded queues ensure that a slow layer cannot
be flooded by a fast producer.
"""

from __future__ import annotations

from collections import deque
from threading import Lock

from wally.hierarchy.types import LayerMessage


class MessageBus:
    """Per-layer bounded FIFO of :class:`LayerMessage` objects.

    Two queues per layer:
    - ``in_queue`` (top-down): the layer's planner writes a new
      ``target_embedding`` here. The runtime reads it.
    - ``out_queue`` (bottom-up): the runtime writes the latest
      ``state_embedding`` here. The layer above reads it.

    ``maxlen=1`` is the right default: a layer only cares about the
    latest target embedding, and the latest state embedding. Older
    messages are dropped on push.
    """

    def __init__(self, maxlen: int = 1) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._queues: dict[str, deque[LayerMessage]] = {}
        self._lock = Lock()

    def register(self, layer_name: str) -> None:
        with self._lock:
            if layer_name in self._queues:
                raise ValueError(f"layer {layer_name!r} already registered")
            self._queues[layer_name] = {
                "in": deque(maxlen=self._maxlen),
                "out": deque(maxlen=self._maxlen),
            }

    def push_down(self, layer_name: str, msg: LayerMessage) -> None:
        """Push a top-down message (target_embedding) into ``layer_name``."""
        with self._lock:
            if layer_name not in self._queues:
                raise ValueError(f"layer {layer_name!r} not registered")
            q = self._queues[layer_name]["in"]
            q.append(msg)

    def push_up(self, layer_name: str, msg: LayerMessage) -> None:
        """Push a bottom-up message (state_embedding) into ``layer_name``."""
        with self._lock:
            if layer_name not in self._queues:
                raise ValueError(f"layer {layer_name!r} not registered")
            q = self._queues[layer_name]["out"]
            q.append(msg)

    def pop_down(self, layer_name: str) -> LayerMessage | None:
        """Pop the oldest top-down message for ``layer_name``, or ``None``."""
        with self._lock:
            q = self._queues[layer_name]["in"]
            if not q:
                return None
            return q.popleft()

    def pop_up(self, layer_name: str) -> LayerMessage | None:
        with self._lock:
            q = self._queues[layer_name]["out"]
            if not q:
                return None
            return q.popleft()

    def drain_down(self, layer_name: str) -> list[LayerMessage]:
        with self._lock:
            q = self._queues[layer_name]["in"]
            out = list(q)
            q.clear()
            return out

    def drain_up(self, layer_name: str) -> list[LayerMessage]:
        with self._lock:
            q = self._queues[layer_name]["out"]
            out = list(q)
            q.clear()
            return out

    def latest_down(self, layer_name: str) -> LayerMessage | None:
        with self._lock:
            q = self._queues[layer_name]["in"]
            return q[-1] if q else None

    def latest_up(self, layer_name: str) -> LayerMessage | None:
        with self._lock:
            q = self._queues[layer_name]["out"]
            return q[-1] if q else None
