"""Manual smoke test for the live POV viewer added by the
``live-agent-viewer`` change.

Two modes:

  1. ``synthetic`` (default) — animate a procedural frame and feed it to
     a real ``FrameViewer``. No MineStudio, no checkpoint, no goal image
     needed. Use this to confirm the OpenCV window opens, the HUD
     updates, and ``q``/``Esc`` cleanly shut it down.

  2. ``play`` — shells out to ``python -m agent.play`` with the given
     checkpoint and goal image. This is the real end-to-end path; it
     needs a working MineStudio install on Windows and a trained
     LeWorldModel checkpoint.

Usage (run from the repo root with the Windows venv activated)::

    # Quickest check: synthetic frames, cv2 window, 200 steps at 20 fps
    python tools/test_live_viewer.py

    # Same but headless (NullViewer, no window, no cv2 import)
    python tools/test_live_viewer.py --viewer none

    # Synthetic with more frames / different FPS
    python tools/test_live_viewer.py --steps 500 --fps 30

    # Real end-to-end (requires MineStudio + trained checkpoint)
    python tools/test_live_viewer.py --mode play ^
        --checkpoint checkpoints\\lewm_step_1000.pt ^
        --goal-frame goal.png

    # Real end-to-end but headless (CI-style)
    python tools/test_live_viewer.py --mode play ^
        --checkpoint checkpoints\\lewm_step_1000.pt ^
        --goal-frame goal.png ^
        --viewer none
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Make ``src/`` importable when this script is invoked directly without
# ``pip install -e .`` (matches the pattern used by ``tools/play.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def _run_synthetic(viewer_kind: str, steps: int, fps: float) -> None:
    from wally.agent.viewer import FrameViewer, NullViewer

    if viewer_kind == "cv2":
        viewer = FrameViewer(window_name="wally-viewer-smoketest")
        logger.info(
            "cv2 viewer enabled — a window titled 'wally-viewer-smoketest' "
            "should appear. Press 'q' or 'Esc' to quit early."
        )
    else:
        viewer = NullViewer()
        logger.info("NullViewer (--viewer none): no window will be shown.")

    delay = 1.0 / max(fps, 0.1)
    quit_step: int | None = None
    try:
        for step in range(steps):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            x = (step * 4) % 600
            frame[100:260, x : x + 80] = (0, 255, 0)
            frame[:] = (
                (step * 3) % 255,
                (step * 5) % 255,
                255 - (step * 2) % 255,
            )
            viewer.show(
                frame,
                info={
                    "step": step,
                    "plan_cost": max(0.0, 0.5 - step * 0.002),
                    "done": step == steps - 1,
                },
            )
            if viewer.should_quit():
                quit_step = step
                break
            time.sleep(delay)
    finally:
        viewer.close()

    if quit_step is not None:
        logger.info("Quit requested by viewer at step %d.", quit_step)
    else:
        logger.info("Ran %d synthetic frames cleanly.", steps)


def _run_play(
    checkpoint: Path, goal_frame: Path, viewer_kind: str, extra: list[str]
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "agent.play",
        "--checkpoint",
        str(checkpoint),
        "--goal-frame",
        str(goal_frame),
        "--viewer",
        viewer_kind,
        *extra,
    ]
    logger.info("Launching: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Manual smoke test for the live POV viewer "
            "(`live-agent-viewer` change)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "play"],
        default="synthetic",
        help=(
            "Synthetic animates a fake frame (no MineStudio, no checkpoint). "
            "Play shells out to `python -m agent.play`."
        ),
    )
    parser.add_argument(
        "--viewer",
        choices=["cv2", "none"],
        default="cv2",
        help="Viewer backend. 'cv2' opens a window; 'none' is headless.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of synthetic frames to render (synthetic mode).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target frames per second for the synthetic test.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to LeWorldModel checkpoint (play mode).",
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        default=None,
        help="Path to goal image (play mode).",
    )
    parser.add_argument(
        "--play-arg",
        action="append",
        default=[],
        dest="play_args",
        help=(
            "Extra arg forwarded to `agent.play` (repeatable), e.g. "
            "`--play-arg --record` or `--play-arg --planner=cem`."
        ),
    )
    args = parser.parse_args(argv)

    if args.mode == "play":
        if args.checkpoint is None or args.goal_frame is None:
            parser.error("--mode play requires --checkpoint and --goal-frame")
        if not args.checkpoint.is_file():
            logger.error("Checkpoint not found: %s", args.checkpoint)
            return 1
        if not args.goal_frame.is_file():
            logger.error("Goal frame not found: %s", args.goal_frame)
            return 1
        return _run_play(
            args.checkpoint, args.goal_frame, args.viewer, args.play_args
        )

    _run_synthetic(args.viewer, args.steps, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
