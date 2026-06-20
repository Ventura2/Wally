from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FrameViewerLike(Protocol):
    def show(
        self, pov: NDArray[np.uint8], info: Optional[Dict[str, Any]] = None
    ) -> None:
        ...

    def should_quit(self) -> bool:
        ...

    def close(self) -> None:
        ...


class NullViewer:
    def show(
        self, pov: NDArray[np.uint8], info: Optional[Dict[str, Any]] = None
    ) -> None:
        return None

    def should_quit(self) -> bool:
        return False

    def close(self) -> None:
        return None


class FrameViewer:
    def __init__(
        self,
        window_name: str = "wally-play",
        show_fps: bool = True,
    ) -> None:
        self._window_name = window_name
        self._show_fps = show_fps
        self._cv2: Any = None
        self._quit = False
        self._last_step: Optional[int] = None
        self._fps_last_time: Optional[float] = None
        self._fps_frame_count: int = 0
        self._current_fps: float = 0.0
        self._window_created: bool = False

    def _ensure_cv2(self) -> Any:
        if self._cv2 is None:
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
        return self._cv2

    def show(
        self, pov: NDArray[np.uint8], info: Optional[Dict[str, Any]] = None
    ) -> None:
        if pov is None:
            return
        cv2 = self._ensure_cv2()
        frame = np.asarray(pov)
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        if not self._window_created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        now = time.monotonic()
        self._fps_frame_count += 1
        if self._fps_last_time is None:
            self._fps_last_time = now
        elif now - self._fps_last_time >= 1.0:
            self._current_fps = self._fps_frame_count / (now - self._fps_last_time)
            self._fps_frame_count = 0
            self._fps_last_time = now

        info = info or {}
        step = info.get("step", self._last_step)
        if step is not None:
            self._last_step = step
        plan_cost = info.get("plan_cost")
        done = info.get("done", False)

        text_lines: list[str] = []
        if step is not None:
            text_lines.append(f"step: {step}")
        if plan_cost is not None:
            text_lines.append(f"plan_cost: {float(plan_cost):.3f}")
        if self._show_fps:
            text_lines.append(f"fps: {self._current_fps:.1f}")
        if done:
            text_lines.append("DONE")

        y = 24
        for line in text_lines:
            cv2.putText(
                frame_bgr,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 20

        cv2.imshow(self._window_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._quit = True

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if self._cv2 is not None and self._window_created:
            try:
                self._cv2.destroyWindow(self._window_name)
            except Exception:  # noqa: BLE001
                logger.debug("destroyWindow failed", exc_info=True)
            try:
                self._cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                logger.debug("destroyAllWindows failed", exc_info=True)
        self._window_created = False
        self._cv2 = None


__all__ = ["FrameViewer", "NullViewer", "FrameViewerLike"]
