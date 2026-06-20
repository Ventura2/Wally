from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure `src/` is on sys.path so `agent`, `wally`, `collector` are importable
# when the package is invoked without `pip install -e .` (e.g. inside the
# wally-dev Podman container where Python 3.10 + minestudio is used).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.config import AgentConfig
from agent.planner_factory import build_planner
from wally.planner.rollout import LatentRollout

logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a goal-conditioned agent episode using a trained LeWorldModel."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to LeWorldModel checkpoint.",
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        required=True,
        help="Path to goal image.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to AgentConfig YAML.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Enable trajectory recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for trajectory export.",
    )
    parser.add_argument(
        "--planner",
        choices=["cem", "gradient", "hierarchical"],
        default="cem",
        help="Planner type.",
    )
    parser.add_argument(
        "--viewer",
        choices=["cv2", "none"],
        default="cv2",
        help="Live POV viewer: 'cv2' shows an OpenCV window, 'none' is headless.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_const",
        const="none",
        dest="viewer",
        help="Disable the live POV viewer (alias for --viewer none).",
    )
    parser.add_argument(
        "--relay",
        action="store_true",
        default=False,
        help=(
            "Expose the latest POV frame over an MJPEG HTTP relay at "
            "http://<relay-host>:<relay-port>/stream. Useful for WSL2 -> "
            "Windows viewing of the MineStudio render."
        ),
    )
    parser.add_argument(
        "--relay-port",
        type=int,
        default=8081,
        help="TCP port for the MJPEG relay (default: 8081).",
    )
    parser.add_argument(
        "--relay-host",
        type=str,
        default="0.0.0.0",
        help="Bind host for the MJPEG relay (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--relay-max-size",
        type=str,
        default="640x360",
        help='Letterboxed max frame size as "WxH" (default: "640x360").',
    )
    parser.add_argument(
        "--relay-jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality 1-100 for the relayed frame (default: 80).",
    )
    parser.add_argument(
        "--relay-min-frame-interval-ms",
        type=int,
        default=33,
        help="Minimum ms between relayed frames; controls viewer fps (default: 33).",
    )
    return parser.parse_args(argv)


def _parse_max_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'--relay-max-size must be "WxH" (got {value!r})'
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'--relay-max-size must be "WxH" with integers (got {value!r})'
        ) from exc
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(
            f"--relay-max-size values must be >= 1 (got {value!r})"
        )
    return w, h


def _apply_relay_args(
    config: AgentConfig, args: argparse.Namespace
) -> AgentConfig:
    max_w, max_h = _parse_max_size(args.relay_max_size)
    return config.model_copy(
        update={
            "relay_enabled": args.relay,
            "relay_port": args.relay_port,
            "relay_host": args.relay_host,
            "relay_max_width": max_w,
            "relay_max_height": max_h,
            "relay_jpeg_quality": args.relay_jpeg_quality,
            "relay_min_frame_interval_ms": args.relay_min_frame_interval_ms,
        }
    )


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.permute(2, 0, 1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if not args.goal_frame.is_file():
        logger.error("Goal frame not found: %s", args.goal_frame)
        sys.exit(1)

    config = AgentConfig.from_yaml(args.config) if args.config else AgentConfig()
    config = config.model_copy(update={"record_trajectory": args.record})
    config = _apply_relay_args(config, args)

    goal_frame = _load_image_as_tensor(args.goal_frame)

    rollout = LatentRollout.from_checkpoint(args.checkpoint)
    encoder = rollout._model.encode
    planner = build_planner(args.planner, rollout, encoder)

    try:
        from agent.env import MineStudioAgentEnv
    except ImportError as exc:
        logger.error("MineStudio is not installed: %s", exc)
        sys.exit(1)

    env = MineStudioAgentEnv(config)

    from agent.loop import AgentLoop
    from agent.viewer import FrameViewer, NullViewer

    if args.relay:
        viewer = NullViewer()
        logger.info(
            "Live POV viewer disabled (relay enabled) "
            "— see http://%s:%d/stream",
            args.relay_host,
            args.relay_port,
        )
    elif args.viewer == "cv2":
        viewer = FrameViewer()
        logger.info("Live POV viewer enabled (cv2). Press 'q' or 'Esc' to quit.")
    else:
        viewer = NullViewer()
        logger.info("Live POV viewer disabled (--viewer none).")

    relay_buffer = None
    relay_server = None
    if args.relay:
        from agent.relay import RelayBuffer, RelayHTTPServer

        relay_buffer = RelayBuffer(
            max_width=config.relay_max_width,
            max_height=config.relay_max_height,
            jpeg_quality=config.relay_jpeg_quality,
        )
        relay_server = RelayHTTPServer(
            host=args.relay_host,
            port=args.relay_port,
            buffer=relay_buffer,
            min_frame_interval_ms=config.relay_min_frame_interval_ms,
        )
        relay_server.start()

    loop = AgentLoop(env, planner, config, viewer=viewer, relay=relay_buffer)
    try:
        result = loop.run_episode(goal_frame)
    finally:
        viewer.close()
        if relay_buffer is not None:
            try:
                relay_buffer.update(None)
            except Exception:  # noqa: BLE001
                logger.debug("relay.update(None) failed", exc_info=True)
        if relay_server is not None:
            try:
                relay_server.stop()
            except Exception:  # noqa: BLE001
                logger.debug("relay_server.stop() failed", exc_info=True)

    print(
        f"Episode complete: {result.steps} steps, "
        f"cost={result.final_cost:.4f}, "
        f"duration={result.duration_seconds:.2f}s"
    )

    if args.record and result.trajectory is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "episode_0.npz"
        np.savez(out_path, **result.trajectory)
        logger.info("Saved trajectory to %s", out_path)


if __name__ == "__main__":
    main()
