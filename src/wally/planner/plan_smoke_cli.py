"""``wally-plan-smoke`` — one-shot world-model sanity check.

Loads a checkpoint, plans a short action sequence from a "current" frame to
a "goal" frame, and reports whether the planner produced structured
output. Useful for:

- Diagnosing whether a newly trained checkpoint is alive (non-zero output)
  or is a "dead" model that should be retrained
- Sanity-checking the world model + planner end-to-end on CPU before
  investing in a full agent episode

Usage:
    uv run wally-plan-smoke
    uv run wally-plan-smoke --checkpoint checkpoints/checkpoint_100000.pt \\
        --frames-dir plan_smoke --output plan_smoke/actions.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("wally-plan-smoke")

IMAGE_SIZE = (224, 224)
REPO = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO / "checkpoints" / "checkpoint_100000.pt"
DEFAULT_FRAMES_DIR = REPO / "plan_smoke"
DEFAULT_OUTPUT = REPO / "plan_smoke" / "actions.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Smoke-test a LeWorldModel checkpoint: plan from current.png to "
            "goal.png in a frames dir and report action statistics."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT.name})",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Dir with current.png and goal.png (default: {DEFAULT_FRAMES_DIR})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save the action tensor (default: {DEFAULT_OUTPUT})",
    )
    return p.parse_args(argv)


def load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1
    for name in ("current.png", "goal.png"):
        p = args.frames_dir / name
        if not p.is_file():
            logger.error("Frame not found: %s", p)
            return 1

    from wally.planner.config import CEMConfig
    from wally.planner.plan import GoalConditionedPlanner
    from wally.planner.rollout import LatentRollout

    current = load_image_as_tensor(args.frames_dir / "current.png")
    goal = load_image_as_tensor(args.frames_dir / "goal.png")
    logger.info(
        "current %s range=[%.3f, %.3f]  goal %s range=[%.3f, %.3f]",
        tuple(current.shape), current.min(), current.max(),
        tuple(goal.shape), goal.min(), goal.max(),
    )

    cem_config = CEMConfig.default()
    logger.info(
        "planner: horizon=%d pop=%d iters=%d",
        cem_config.horizon, cem_config.population_size, cem_config.n_iterations,
    )

    rollout = LatentRollout.from_checkpoint(args.checkpoint)
    logger.info("loaded rollout from %s", args.checkpoint)

    planner = GoalConditionedPlanner(rollout, rollout._model.encode, cem_config)
    actions = planner.plan(current, goal, return_cost=False)
    logger.info("actions: shape=%s dtype=%s", tuple(actions.shape), actions.dtype)

    abs_max = actions.abs().max().item()
    abs_mean = actions.abs().mean().item()
    std = actions.std().item()
    abs_sum = actions.abs().sum().item()
    logger.info(
        "|max|=%.4f  |mean|=%.4f  std=%.4f  |sum|=%.4f",
        abs_max, abs_mean, std, abs_sum,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actions, args.output)
    logger.info("saved -> %s", args.output)

    if abs_max < 0.05:
        logger.info("VERDICT: model is DEAD - actions are essentially zero")
        return 10
    if std < 0.05:
        logger.info("VERDICT: model output is flat (low variance) - likely noise")
        return 11
    logger.info("VERDICT: model produced structured actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
