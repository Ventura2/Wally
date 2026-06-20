from __future__ import annotations

import socket
import threading
import time
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from agent.relay import BOUNDARY, RelayBuffer, RelayHTTPServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get(url: str, timeout: float = 2.0) -> Tuple[int, dict, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, dict(resp.headers), body


def _http_get_status(url: str, timeout: float = 2.0) -> int:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(1)
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


class TestRelayBufferRoundTrip:
    def test_update_then_snapshot_bgr_matches_cvtColor(self) -> None:
        buf = RelayBuffer(max_width=320, max_height=240, jpeg_quality=90)
        rgb = np.zeros((128, 200, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30

        buf.update(rgb)
        jpeg, bgr, ts = buf.snapshot()

        assert jpeg is not None and len(jpeg) > 0
        assert bgr is not None
        assert ts > 0.0
        expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        np.testing.assert_array_equal(bgr, expected)

    def test_snapshot_returns_cached_bytes_on_repeat(self) -> None:
        buf = RelayBuffer()
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        buf.update(rgb)

        jpeg1, bgr1, _ = buf.snapshot()
        jpeg2, bgr2, _ = buf.snapshot()

        assert jpeg1 == jpeg2
        np.testing.assert_array_equal(bgr1, bgr2)

    def test_update_invalid_shape_raises(self) -> None:
        buf = RelayBuffer()
        with pytest.raises(ValueError):
            buf.update(np.zeros((64, 64), dtype=np.uint8))
        with pytest.raises(ValueError):
            buf.update(np.zeros((64, 64, 4), dtype=np.uint8))

    def test_update_invalid_quality_raises(self) -> None:
        with pytest.raises(ValueError):
            RelayBuffer(jpeg_quality=0)
        with pytest.raises(ValueError):
            RelayBuffer(jpeg_quality=101)


class TestRelayBufferClear:
    def test_update_none_clears_slot(self) -> None:
        buf = RelayBuffer()
        buf.update(np.full((16, 16, 3), 50, dtype=np.uint8))
        jpeg, bgr, ts = buf.snapshot()
        assert jpeg is not None and bgr is not None and ts > 0.0

        buf.update(None)
        jpeg2, bgr2, ts2 = buf.snapshot()
        assert jpeg2 is None
        assert bgr2 is None
        assert ts2 == 0.0


class TestRelayBufferDownsampling:
    def test_letterbox_preserves_aspect_ratio(self) -> None:
        buf = RelayBuffer(max_width=320, max_height=180)
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        buf.update(rgb)
        jpeg, bgr, _ = buf.snapshot()
        assert jpeg is not None and bgr is not None

        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = decoded.shape[:2]
        assert h == 180
        assert w * 9 == h * 16

    def test_smaller_frame_passes_through_unchanged(self) -> None:
        buf = RelayBuffer(max_width=640, max_height=360)
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        buf.update(rgb)
        _, bgr, _ = buf.snapshot()
        assert bgr is not None
        assert bgr.shape == (100, 200, 3)


class TestRelayBufferThreadSafety:
    def test_concurrent_update_snapshot_does_not_deadlock(self) -> None:
        buf = RelayBuffer()
        errors: List[BaseException] = []

        def writer() -> None:
            try:
                for i in range(200):
                    rgb = np.full((32, 32, 3), i % 256, dtype=np.uint8)
                    buf.update(rgb)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(500):
                    buf.snapshot()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=reader, daemon=True),
            threading.Thread(target=reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors, errors
        for t in threads:
            assert not t.is_alive()


class TestRelayHTTPServerHealthz:
    def test_healthz_returns_ok(self) -> None:
        buf = RelayBuffer()
        port = _free_port()
        server = RelayHTTPServer("127.0.0.1", port, buf)
        server.start()
        try:
            status, headers, body = _http_get(f"http://127.0.0.1:{port}/healthz")
            assert status == 200
            assert headers.get("Content-Type", "").startswith("text/plain")
            assert body == b"ok\n"
        finally:
            server.stop()

    def test_unknown_path_returns_404(self) -> None:
        buf = RelayBuffer()
        port = _free_port()
        server = RelayHTTPServer("127.0.0.1", port, buf)
        server.start()
        try:
            assert _http_get_status(f"http://127.0.0.1:{port}/nope") == 404
        finally:
            server.stop()


class TestRelayHTTPServerStream:
    def test_stream_yields_mjpeg_frames(self) -> None:
        import socket as _socket

        buf = RelayBuffer(max_width=160, max_height=120, jpeg_quality=80)
        port = _free_port()
        server = RelayHTTPServer(
            "127.0.0.1", port, buf, min_frame_interval_ms=20
        )
        server.start()
        try:
            stop = threading.Event()

            def pusher() -> None:
                i = 0
                while not stop.is_set():
                    rgb = np.full((64, 64, 3), (i * 7) % 256, dtype=np.uint8)
                    buf.update(rgb)
                    i += 1
                    time.sleep(0.01)

            t = threading.Thread(target=pusher, daemon=True)
            t.start()

            chunks: list[bytes] = []
            try:
                with _socket.create_connection(
                    ("127.0.0.1", port), timeout=2.0
                ) as sock:
                    sock.sendall(
                        b"GET /stream HTTP/1.0\r\n"
                        b"Host: 127.0.0.1\r\n\r\n"
                    )
                    sock.settimeout(1.0)
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        try:
                            chunk = sock.recv(4096)
                        except _socket.timeout:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                        if (
                            b"--frame" in b"".join(chunks)
                            and b"image/jpeg" in b"".join(chunks)
                        ):
                            break
            finally:
                stop.set()
                t.join(timeout=2.0)

            body = b"".join(chunks)
            assert b"200 OK" in body
            assert b"multipart/x-mixed-replace" in body
            assert f"boundary={BOUNDARY.decode()}".encode() in body
            assert b"--frame" in body
            assert b"image/jpeg" in body
        finally:
            server.stop()

    def test_daemon_thread_does_not_block_process_exit(self) -> None:
        buf = RelayBuffer()
        port = _free_port()
        server = RelayHTTPServer("127.0.0.1", port, buf)
        server.start()
        assert server._thread is not None  # type: ignore[attr-defined]
        assert server._thread.daemon is True  # type: ignore[attr-defined]
        server.stop()


class TestRelayHTTPServerLifecycle:
    def test_stop_releases_port(self) -> None:
        buf = RelayBuffer()
        port = _free_port()
        server = RelayHTTPServer("127.0.0.1", port, buf)
        server.start()
        server.stop()
        server2 = RelayHTTPServer("127.0.0.1", port, buf)
        server2.start()
        server2.stop()

    def test_double_start_is_noop(self) -> None:
        buf = RelayBuffer()
        port = _free_port()
        server = RelayHTTPServer("127.0.0.1", port, buf)
        server.start()
        first = server._server  # type: ignore[attr-defined]
        server.start()
        assert server._server is first  # type: ignore[attr-defined]
        server.stop()
