from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wally.agent.config import AgentConfig
from wally.agent.planner_factory import build_planner
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
        default=None,
        help=(
            "Path to goal image. Optional when --target-embedding is "
            "provided (e.g. for the hierarchical-embedding planner)."
        ),
    )
    parser.add_argument(
        "--target-embedding",
        type=Path,
        default=None,
        help=(
            "Path to a .pt file containing a goal embedding tensor. "
            "Optional when --goal-frame is provided."
        ),
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
        choices=["cem", "gradient", "hierarchical", "hierarchical-embedding"],
        default="cem",
        help="Planner type.",
    )
    parser.add_argument(
        "--hierarchy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a saved hierarchy checkpoint (produced by "
            "wally-train-hierarchy). Required for --planner "
            "hierarchical-embedding."
        ),
    )
    parser.add_argument(
        "--layer-depth",
        type=int,
        default=0,
        help=(
            "Number of additional hierarchy layers above L0 (0 disables "
            "the hierarchy, 1 = L1, 2 = L0+L1+L2, 3 = L0+L1+L2+L3)."
        ),
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

    if args.goal_frame is None and args.target_embedding is None:
        logger.error(
            "Either --goal-frame or --target-embedding must be provided"
        )
        sys.exit(1)
    if args.goal_frame is not None and not args.goal_frame.is_file():
        logger.error("Goal frame not found: %s", args.goal_frame)
        sys.exit(1)
    if args.target_embedding is not None and not args.target_embedding.is_file():
        logger.error("Target embedding not found: %s", args.target_embedding)
        sys.exit(1)

    if args.planner == "hierarchical-embedding" and args.hierarchy_checkpoint is None:
        logger.error(
            "--planner hierarchical-embedding requires --hierarchy-checkpoint"
        )
        sys.exit(1)

    config = AgentConfig.from_yaml(args.config) if args.config else AgentConfig()
    config = config.model_copy(update={"record_trajectory": args.record})
    config = _apply_relay_args(config, args)

    goal_frame: torch.Tensor | None = None
    target_embedding: torch.Tensor | None = None
    if args.goal_frame is not None:
        goal_frame = _load_image_as_tensor(args.goal_frame)
    if args.target_embedding is not None:
        target_embedding = torch.load(
            args.target_embedding, map_location="cpu", weights_only=False
        )
        if isinstance(target_embedding, dict):
            target_embedding = target_embedding.get("g", target_embedding.get("target"))
        if not isinstance(target_embedding, torch.Tensor):
            raise ValueError(
                f"--target-embedding must contain a Tensor; got "
                f"{type(target_embedding).__name__}"
            )

    rollout = LatentRollout.from_checkpoint(args.checkpoint)
    encoder = rollout._model.encode
    planner = build_planner(
        args.planner,
        rollout,
        encoder,
        hierarchy_checkpoint=args.hierarchy_checkpoint,
        layer_depth=args.layer_depth,
    )

    try:
        from wally.agent.env import MineStudioAgentEnv
    except ImportError as exc:
        logger.error("MineStudio is not installed: %s", exc)
        sys.exit(1)

    env = MineStudioAgentEnv(config)

    from wally.agent.loop import AgentLoop
    from wally.agent.viewer import FrameViewer, NullViewer

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
        from wally.agent.relay import RelayBuffer, RelayHTTPServer

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

    loop = AgentLoop(
        env, planner, config, viewer=viewer, relay=relay_buffer, l0_encoder=encoder
    )
    try:
        if target_embedding is not None:
            fallback = goal_frame if goal_frame is not None else torch.zeros(3, 64, 64)
            result = loop.run_episode(fallback, target_embedding=target_embedding)
        else:
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
