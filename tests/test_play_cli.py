from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from wally.agent.play import main, parse_args


class TestParseArgs:
    def test_parse_args_minimal(self) -> None:
        args = parse_args(["--checkpoint", "model.pt", "--goal-frame", "goal.png"])
        assert args.checkpoint == Path("model.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config is None
        assert args.record is False
        assert args.output_dir == Path(".")
        assert args.planner == "cem"
        assert args.viewer == "cv2"

    def test_parse_args_all_options(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("replan_interval: 8\n")
        output_dir = tmp_path / "out"

        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--config", str(config_path),
            "--record",
            "--output-dir", str(output_dir),
            "--planner", "gradient",
        ])
        assert args.checkpoint == Path("model.pt")
        assert args.goal_frame == Path("goal.png")
        assert args.config == Path(str(config_path))
        assert args.record is True
        assert args.output_dir == output_dir
        assert args.planner == "gradient"
        assert args.viewer == "cv2"

    @pytest.mark.parametrize("choice", ["cem", "gradient", "hierarchical"])
    def test_parse_args_planner_choices(self, choice: str) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--planner", choice,
        ])
        assert args.planner == choice

    def test_parse_args_invalid_planner(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([
                "--checkpoint", "model.pt",
                "--goal-frame", "goal.png",
                "--planner", "invalid",
            ])

    @pytest.mark.smoke
    def test_parse_args_viewer_none(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--viewer", "none",
        ])
        assert args.viewer == "none"

    def test_parse_args_no_viewer_alias(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--no-viewer",
        ])
        assert args.viewer == "none"

    def test_parse_args_invalid_viewer(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([
                "--checkpoint", "model.pt",
                "--goal-frame", "goal.png",
                "--viewer", "invalid",
            ])

    def test_parse_args_relay_defaults(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--relay",
        ])
        assert args.relay is True
        assert args.relay_port == 8081
        assert args.relay_host == "0.0.0.0"
        assert args.relay_max_size == "640x360"
        assert args.relay_jpeg_quality == 80
        assert args.relay_min_frame_interval_ms == 33

    def test_parse_args_relay_overrides(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--relay",
            "--relay-port", "9999",
            "--relay-host", "127.0.0.1",
            "--relay-max-size", "320x180",
            "--relay-jpeg-quality", "60",
            "--relay-min-frame-interval-ms", "16",
        ])
        assert args.relay_port == 9999
        assert args.relay_host == "127.0.0.1"
        assert args.relay_max_size == "320x180"
        assert args.relay_jpeg_quality == 60
        assert args.relay_min_frame_interval_ms == 16

    def test_apply_relay_args_populates_config(self) -> None:
        from wally.agent.config import AgentConfig
        from wally.agent.play import _apply_relay_args, _parse_max_size

        args = parse_args([
            "--checkpoint", "model.pt",
            "--goal-frame", "goal.png",
            "--relay",
            "--relay-port", "9001",
            "--relay-host", "127.0.0.1",
            "--relay-max-size", "800x450",
            "--relay-jpeg-quality", "70",
            "--relay-min-frame-interval-ms", "50",
        ])
        config = _apply_relay_args(AgentConfig(), args)
        assert config.relay_enabled is True
        assert config.relay_port == 9001
        assert config.relay_host == "127.0.0.1"
        assert config.relay_max_width == 800
        assert config.relay_max_height == 450
        assert config.relay_jpeg_quality == 70
        assert config.relay_min_frame_interval_ms == 50

        w, h = _parse_max_size("640x360")
        assert (w, h) == (640, 360)
        w, h = _parse_max_size("320X180")
        assert (w, h) == (320, 180)


class TestMain:
    def test_missing_checkpoint(self) -> None:
        with pytest.raises(SystemExit):
            main(["--goal-frame", "goal.png"])

    def test_checkpoint_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(tmp_path / "nonexistent.pt"),
                "--goal-frame", "goal.png",
            ])
        assert exc_info.value.code == 1

    def test_goal_frame_not_found(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--checkpoint", str(checkpoint),
                "--goal-frame", str(tmp_path / "nonexistent.png"),
            ])
        assert exc_info.value.code == 1

    def test_config_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_data = {"replan_interval": 8, "episode_timeout": 500}
        config_path.write_text(yaml.dump(config_data))

        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        goal = tmp_path / "goal.png"
        goal.touch()

        args = parse_args([
            "--checkpoint", str(checkpoint),
            "--goal-frame", str(goal),
            "--config", str(config_path),
        ])

        from wally.agent.config import AgentConfig

        loaded = AgentConfig.from_yaml(args.config)
        assert loaded.replan_interval == 8
        assert loaded.episode_timeout == 500


class TestRelayEndToEnd:
    @pytest.mark.smoke
    def test_wally_play_relay_smoke(self, tmp_path: Path, monkeypatch) -> None:
        import socket
        import threading
        import urllib.request
        from unittest.mock import MagicMock

        import numpy as np
        import torch
        from PIL import Image

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])

        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        goal = tmp_path / "goal.png"
        Image.new("RGB", (16, 16), color=(64, 128, 192)).save(goal)

        class _FakeEnv:
            def __init__(self) -> None:
                self._calls = 0
                self._limit = 40

            def reset(self) -> torch.Tensor:
                return torch.zeros(3, 64, 64)

            def step(self, action):
                self._calls += 1
                time.sleep(0.05)
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                rgb[..., 0] = (self._calls * 30) % 255
                rgb[..., 1] = 80
                rgb[..., 2] = 200
                done = self._calls >= self._limit
                return torch.zeros(3, 64, 64), 0.0, done, {"pov": rgb}

            def close(self) -> None:
                return None

        fake_env = _FakeEnv()
        monkeypatch.setattr("wally.agent.env.MineStudioAgentEnv", lambda c: fake_env)

        fake_model = MagicMock()
        fake_model.encode = MagicMock()

        class _FakeRollout:
            def __init__(self, ckpt):
                self._model = fake_model

        class _FakeRolloutFactory:
            from_checkpoint = staticmethod(lambda ckpt: _FakeRollout(ckpt))

        monkeypatch.setattr("wally.agent.play.LatentRollout", _FakeRolloutFactory)

        class _FakePlanner:
            def plan(self, current_frame, goal_frame):
                from wally.agent.protocol import PlanResult

                return PlanResult(actions=torch.zeros(2, 25), cost=0.0)

        monkeypatch.setattr(
            "wally.agent.play.build_planner",
            lambda *a, **k: _FakePlanner(),
        )

        healthz_status: dict[str, int] = {}
        healthz_body: dict[str, bytes] = {}
        stream_body: dict[str, bytes] = {}

        def check_healthz() -> None:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/healthz")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                healthz_status["status"] = resp.status
                healthz_body["body"] = resp.read()

        def grab_stream() -> None:
            import socket as _socket

            with _socket.create_connection(
                ("127.0.0.1", port), timeout=2.0
            ) as sock:
                sock.sendall(
                    b"GET /stream HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
                )
                sock.settimeout(1.0)
                chunks: list[bytes] = []
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    try:
                        chunk = sock.recv(4096)
                    except _socket.timeout:
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
                    joined = b"".join(chunks)
                    if b"--frame" in joined and b"image/jpeg" in joined:
                        break
            stream_body["body"] = b"".join(chunks)

        main_thread = threading.Thread(
            target=main,
            kwargs={"argv": [
                "--checkpoint", str(checkpoint),
                "--goal-frame", str(goal),
                "--relay",
                "--relay-port", str(port),
                "--relay-host", "127.0.0.1",
                "--relay-min-frame-interval-ms", "20",
            ]},
            daemon=True,
        )
        main_thread.start()

        time.sleep(0.3)
        for _ in range(20):
            try:
                check_healthz()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        grab_stream()

        main_thread.join(timeout=10.0)
        assert not main_thread.is_alive(), "main() did not finish in time"

        assert healthz_status.get("status") == 200
        assert healthz_body.get("body") == b"ok\n"
        body = stream_body.get("body", b"")
        assert b"--frame" in body
        assert b"image/jpeg" in body
