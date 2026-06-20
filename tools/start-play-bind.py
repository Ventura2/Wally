"""Windows-side viewer for the MJPEG relay served by ``wally-play --relay``.

Connects to the ``multipart/x-mixed-replace`` MJPEG stream at
``http://localhost:8081/stream`` (default) and renders it in an OpenCV
window. Reconnects automatically when the stream drops and polls the
relay's ``/healthz`` endpoint so the window can show whether the agent
side is alive.

Run from the repo root with the Windows venv activated (the relay port
is forwarded to the Windows host by ``podman run -p 8081:8081``)::

    python tools/start-play-bind.py
    python tools/start-play-bind.py --url http://localhost:8081/stream
    python tools/start-play-bind.py --fullscreen

Press ``q`` or ``Esc`` in the window to quit. ``Ctrl+C`` also works.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger("start-play-bind")

DEFAULT_URL = "http://localhost:8081/stream"
DEFAULT_HEALTH_URL = "http://localhost:8081/healthz"
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 8.0)
WINDOW_NAME = "wally-live-pov"
PLACEHOLDER_TEXT = "Waiting for wally-play relay at {url}"
DISCONNECTED_TEXT = "Stream dropped - retry #{n} in {delay:.1f}s"


def _check_health(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _build_placeholder(url: str, width: int = 640, height: int = 360) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        "wally live POV",
        "",
        PLACEHOLDER_TEXT.format(url=url),
        "(agent not yet serving frames)",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 60 + i * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return frame


def _format_status(
    frame: np.ndarray,
    *,
    fps: float,
    healthy: bool,
    reconnected: int,
    stream_url: str,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    band_h = 28
    cv2.rectangle(out, (0, 0), (w, band_h), (0, 0, 0), thickness=-1)
    dot_color = (0, 255, 0) if healthy else (0, 0, 255)
    cv2.circle(out, (14, band_h // 2), 6, dot_color, thickness=-1)
    status = "LIVE" if healthy else "WAITING"
    cv2.putText(
        out,
        f"{status}  fps={fps:4.1f}  reconnects={reconnected}",
        (28, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        stream_url,
        (28, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return out


def _format_disconnected(
    width: int, height: int, attempt: int, delay: float
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        DISCONNECTED_TEXT.format(n=attempt, delay=delay),
        (16, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (90, 90, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _open_capture(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cv2.VideoCapture could not open {url!r}")
    return cap


def _run(
    url: str,
    health_url: str,
    health_timeout: float,
    health_interval: float,
    fullscreen: bool,
    window_scale: float,
) -> int:
    placeholder = _build_placeholder(url)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    elif window_scale != 1.0:
        h, w = placeholder.shape[:2]
        cv2.resizeWindow(WINDOW_NAME, int(w * window_scale), int(h * window_scale))

    cap: cv2.VideoCapture | None = None
    reconnected = 0
    last_health_check = 0.0
    healthy = False
    fps_history: deque[float] = deque(maxlen=30)
    last_frame_t = time.monotonic()

    def _close_capture() -> None:
        nonlocal cap
        if cap is not None:
            cap.release()
            cap = None

    def _show(frame: np.ndarray) -> None:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt

    try:
        while True:
            now = time.monotonic()
            if now - last_health_check >= health_interval:
                healthy = _check_health(health_url, health_timeout)
                last_health_check = now

            if cap is None:
                try:
                    cap = _open_capture(url)
                    reconnected += 1
                    logger.info(
                        "Opened MJPEG stream at %s (attempt #%d)", url, reconnected
                    )
                except Exception as exc:
                    logger.warning("Cannot open %s: %s", url, exc)
                    backoff = RECONNECT_BACKOFF_S[
                        min(reconnected, len(RECONNECT_BACKOFF_S) - 1)
                    ]
                    _show(
                        _format_status(
                            placeholder,
                            fps=0.0,
                            healthy=False,
                            reconnected=reconnected,
                            stream_url=url,
                        )
                    )
                    time.sleep(backoff)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Stream read failed; will reconnect.")
                _close_capture()
                w, h = placeholder.shape[1], placeholder.shape[0]
                delay = RECONNECT_BACKOFF_S[0]
                _show(_format_disconnected(w, h, reconnected + 1, delay))
                time.sleep(delay)
                continue

            t = time.monotonic()
            dt = t - last_frame_t
            if dt > 0:
                fps_history.append(1.0 / dt)
            last_frame_t = t
            fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            overlay = _format_status(
                frame,
                fps=fps,
                healthy=healthy,
                reconnected=reconnected,
                stream_url=url,
            )
            _show(overlay)
    except KeyboardInterrupt:
        logger.info("Interrupted - shutting down.")
    finally:
        _close_capture()
        cv2.destroyWindow(WINDOW_NAME)
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Bind to the wally-play MJPEG relay from Windows and render the "
            "agent POV in an OpenCV window."
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"MJPEG stream URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
        help=f"Health check URL (default: {DEFAULT_HEALTH_URL})",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=2.0,
        help="Seconds between /healthz polls (default: 2.0).",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for each /healthz request (default: 1.0).",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Open the OpenCV window in fullscreen.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Resize factor for the OpenCV window (default: 1.0).",
    )
    args = parser.parse_args(argv)
    return _run(
        args.url,
        args.health_url,
        args.health_timeout,
        args.health_interval,
        args.fullscreen,
        args.window_scale,
    )


if __name__ == "__main__":
    sys.exit(main())
