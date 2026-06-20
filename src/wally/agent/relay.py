from __future__ import annotations

import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Tuple, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BOUNDARY = b"frame"

UInt8Array = np.ndarray[Any, np.dtype[np.uint8]]


class RelayBuffer:
    def __init__(
        self,
        max_width: int = 640,
        max_height: int = 360,
        jpeg_quality: int = 80,
    ) -> None:
        if max_width < 1 or max_height < 1:
            raise ValueError("max_width and max_height must be >= 1")
        if not 0 < jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in (0, 100]")
        self._max_width = max_width
        self._max_height = max_height
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._frame_bgr: UInt8Array | None = None
        self._timestamp: float = 0.0

    def update(self, pov_rgb: UInt8Array | None) -> None:
        if pov_rgb is None:
            with self._lock:
                self._frame_jpeg = None
                self._frame_bgr = None
                self._timestamp = 0.0
            return

        frame = np.asarray(pov_rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"pov_rgb must have shape (H, W, 3); got {frame.shape}"
            )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resized = self._letterbox_resize(bgr, self._max_width, self._max_height)
        ok, buf = cv2.imencode(
            ".jpg",
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the frame as JPEG")
        jpeg_bytes = buf.tobytes()
        ts = time.monotonic()
        with self._lock:
            self._frame_jpeg = jpeg_bytes
            self._frame_bgr = resized
            self._timestamp = ts

    def snapshot(self) -> Tuple[bytes | None, UInt8Array | None, float]:
        with self._lock:
            jpeg = self._frame_jpeg
            bgr = self._frame_bgr
            ts = self._timestamp
        if bgr is None:
            return None, None, 0.0
        return jpeg, bgr.copy(), ts

    @staticmethod
    def _letterbox_resize(
        bgr: UInt8Array, max_width: int, max_height: int
    ) -> UInt8Array:
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return bgr
        scale = min(max_width / w, max_height / h)
        if scale >= 1.0:
            return bgr
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _stream_parts(
    buffer: RelayBuffer, interval_s: float
) -> Iterator[bytes]:
    while True:
        jpeg, _, _ = buffer.snapshot()
        if jpeg is not None:
            header = (
                b"--" + BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
            )
            yield header + jpeg + b"\r\n"
        if interval_s > 0:
            time.sleep(interval_s)


class _RelayHandler(BaseHTTPRequestHandler):
    server_version = "WallyRelay/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format, *args)

    def do_GET(self) -> None:  # noqa: N802
        relay: RelayHTTPServer | None = getattr(self.server, "_wally_relay", None)
        if self.path == "/healthz":
            body = b"ok\n"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream":
            if relay is None:
                body = b"server not initialized\n"
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._serve_stream(relay)
            return
        body = b"not found\n"
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, relay: RelayHTTPServer) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        wfile = cast(Any, self.wfile)
        wfile._flush = True  # noqa: SLF001 — force flush on each write
        interval_s = max(0.0, relay.min_frame_interval_ms / 1000.0)
        for part in _stream_parts(relay.buffer, interval_s):
            try:
                wfile.write(part)
            except (BrokenPipeError, ConnectionResetError):
                return


class RelayHTTPServer:
    def __init__(
        self,
        host: str,
        port: int,
        buffer: RelayBuffer,
        min_frame_interval_ms: int = 33,
    ) -> None:
        self._host = host
        self._port = port
        self.buffer = buffer
        self.min_frame_interval_ms = min_frame_interval_ms
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        return int(self._server.server_address[1])

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self._host, self._port), _RelayHandler)
        server._wally_relay = self  # type: ignore[attr-defined]
        self._server = server
        thread = threading.Thread(
            target=server.serve_forever, name="wally-relay", daemon=True
        )
        self._thread = thread
        thread.start()
        logger.info(
            "Wally relay listening on http://%s:%d/stream",
            self._host,
            self.port,
        )

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
        finally:
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                logger.debug("server_close failed", exc_info=True)
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


__all__ = ["RelayBuffer", "RelayHTTPServer"]
