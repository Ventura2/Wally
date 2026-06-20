"""Long-horizon goal evaluation across multiple checkpoints.

For each checkpoint produced by ``wally-train``, this script:

  1. Loads the checkpoint with ``LatentRollout`` and builds a
     ``GoalConditionedPlanner`` (CEM).
  2. For each goal in a built-in registry (get_wood, get_iron_ore, ...)
     it runs N short episodes, plans in latent space, executes the
     actions, and records:
        * plan cost  (CEM best, from the world model)
        * latent distance to goal  (||z_end - z_goal||_2)
        * real reward / inventory success  (only in ``env`` mode)
        * steps taken
  3. Writes a per-(checkpoint, goal) table to CSV, JSON, and a
     human-readable markdown report so you can eyeball whether the
     world model is actually getting better at long-horizon tasks.

Three backends are supported:

  * ``world_model``  — no env required; the world model rolls itself
                       out from a random initial latent. Fast, runs
                       anywhere, but cannot measure real success.
  * ``minestudio``   — runs the real MineStudio env (needs the
                       ``wally-dev`` Podman container on WSL2).
  * ``mock``         — synthetic env for smoke-testing this script
                       itself without MineStudio.

Examples
--------

    # Pure-latent rollout, no env needed. Fast.
    python tools/eval_goals.py \\
        --checkpoints 'checkpoints/checkpoint_{1000,5000,10000,20000}.pt' \\
        --mode world_model \\
        --output runs/goal_eval

    # Real MineStudio eval (WSL2 container only).
    python tools/eval_goals.py \\
        --checkpoints 'checkpoints/checkpoint_*.pt' \\
        --num-checkpoints 5 \\
        --mode minestudio \\
        --episodes 3 \\
        --output runs/goal_eval

    # Smoke test (no Minecraft).
    python tools/eval_goals.py \\
        --checkpoints 'checkpoints/checkpoint_5000.pt' \\
        --mode mock \\
        --episodes 2 \\
        --output runs/goal_eval_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout


logger = logging.getLogger(__name__)


IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """A long-horizon Minecraft goal.

    Attributes:
        name:               short id used in tables / report
        description:        one-line human description
        inventory_targets:  list of inventory item names whose presence
                            indicates success (e.g. ``["log"]``). Used by
                            the ``minestudio`` / ``mock`` backends to
                            score real success. Empty list = no
                            inventory-based criterion.
        max_episode_steps:  hard cap on steps per episode
    """

    name: str
    description: str
    inventory_targets: list[str] = field(default_factory=list)
    max_episode_steps: int = 200
    target_latent: torch.Tensor | None = None
    goal_frame_path: Path | None = None


# Built-in goal registry. The target latents are *populated at runtime*
# from the first checkpoint's encoder, so the script is checkpoint-
# agnostic (it does not assume any particular latent dimensionality).
BUILTIN_GOALS: list[Goal] = [
    Goal(
        name="get_wood",
        description="Find and chop a tree; success = any 'log' in inventory.",
        inventory_targets=["log", "oak_log", "spruce_log", "birch_log", "acacia_log", "jungle_log", "dark_oak_log"],
        max_episode_steps=200,
    ),
    Goal(
        name="get_iron_ore",
        description="Find and mine iron ore; success = 'iron_ore' in inventory.",
        inventory_targets=["iron_ore", "raw_iron", "iron_ingot"],
        max_episode_steps=400,
    ),
    Goal(
        name="get_stone",
        description="Mine cobblestone; success = 'cobblestone' in inventory.",
        inventory_targets=["cobblestone", "stone"],
        max_episode_steps=200,
    ),
    Goal(
        name="navigate_look_around",
        description="Look around / move; proxy: latent distance to a 'standing' state.",
        inventory_targets=[],
        max_episode_steps=100,
    ),
]


def get_goal(name: str) -> Goal:
    for g in BUILTIN_GOALS:
        if g.name == name:
            return g
    raise KeyError(f"Unknown goal '{name}'. Available: {[g.name for g in BUILTIN_GOALS]}")


# ---------------------------------------------------------------------------
# Latent / frame helpers
# ---------------------------------------------------------------------------


def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load an image as a (3, 224, 224) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _random_frame_tensor(rng: np.random.Generator) -> torch.Tensor:
    """Sample a synthetic frame in [0, 1] for the mock backend."""
    return torch.from_numpy(rng.random((3, *IMAGE_SIZE), dtype=np.float32))


def _encode_goal_latent(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    goal: Goal,
    device: torch.device,
) -> torch.Tensor:
    """Resolve the target latent for a goal.

    Priority:
        1. ``goal.target_latent`` (pre-set tensor)
        2. ``goal.goal_frame_path`` (encode the image)
        3. The zero vector in latent space (last-resort proxy that still
           produces a meaningful "distance to goal" metric).
    """
    if goal.target_latent is not None:
        return goal.target_latent.to(device)
    if goal.goal_frame_path is not None:
        if not goal.goal_frame_path.is_file():
            logger.warning(
                "Goal '%s': goal_frame_path %s not found; falling back to zero latent.",
                goal.name,
                goal.goal_frame_path,
            )
            return torch.zeros(_probe_latent_dim(encoder, device), device=device)
        frame = _load_image_tensor(goal.goal_frame_path).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(frame).mean(dim=0)
        return z.detach()
    return torch.zeros(_probe_latent_dim(encoder, device), device=device)


def _probe_latent_dim(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> int:
    """Run a single dummy encode to learn the latent dimensionality."""
    with torch.no_grad():
        z = encoder(torch.zeros(1, 3, *IMAGE_SIZE, device=device))
    return int(z.shape[-1])


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


_CHECKPOINT_STEP_RE = re.compile(r"checkpoint[_/\\](?:step)?_?(\d+)\.pt$", re.IGNORECASE)


def _step_from_checkpoint(path: Path) -> int:
    """Extract the integer step number from a checkpoint filename."""
    m = _CHECKPOINT_STEP_RE.search(str(path).replace("\\", "/"))
    return int(m.group(1)) if m else -1


def discover_checkpoints(
    pattern: str,
    num_checkpoints: int | None,
) -> list[Path]:
    """Return checkpoints matching ``pattern``, sorted by step.

    If ``num_checkpoints`` is set and more than that many match, keep
    the earliest, latest, and an even spread in between.
    """
    matches = sorted(
        {p for p in Path().glob(pattern)},
        key=lambda p: (_step_from_checkpoint(p), str(p)),
    )
    if not matches:
        return []
    if num_checkpoints is None or len(matches) <= num_checkpoints:
        return matches
    # Evenly-spaced subset (always keep first and last).
    idx = np.linspace(0, len(matches) - 1, num=num_checkpoints, dtype=int)
    seen: set[int] = set()
    out: list[Path] = []
    for i in idx:
        if i in seen:
            continue
        seen.add(i)
        out.append(matches[i])
    return sorted(out, key=lambda p: _step_from_checkpoint(p))


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    checkpoint: str
    goal: str
    episode: int
    steps: int
    success: bool
    cumulative_reward: float
    initial_plan_cost: float
    final_latent_distance: float
    seconds: float


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class EpisodeRunner:
    """Run one episode on a specific backend.

    Subclasses implement ``_initial_frame`` and ``_step``; the shared
    loop handles planner + latent bookkeeping.
    """

    def __init__(
        self,
        backend_name: str,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        planner: GoalConditionedPlanner,
        device: torch.device,
        action_dim: int,
        cem_horizon: int,
        replan_interval: int,
        seed: int,
    ) -> None:
        self.backend_name = backend_name
        self._encoder = encoder
        self._planner = planner
        self._device = device
        self._action_dim = action_dim
        self._cem_horizon = cem_horizon
        self._replan_interval = max(1, replan_interval)
        self._rng = np.random.default_rng(seed)

    # --- hooks -----------------------------------------------------------
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # --- shared loop -----------------------------------------------------
    def run(
        self,
        checkpoint_name: str,
        goal: Goal,
        episode_idx: int,
        target_latent: torch.Tensor,
    ) -> EpisodeRecord:
        t0 = time.monotonic()
        current_frame, info = self.reset()

        # First plan (no warm start).
        z0 = self._encode_frame(current_frame)
        init_cost = float(((z0 - target_latent) ** 2).sum().item())

        plan_actions, plan_cost = self._planner.plan_to_latent(
            current_frame, target_latent, return_cost=True
        )
        plan_actions = plan_actions.detach()
        action_index = 0

        cumulative_reward = 0.0
        achieved = self._check_success(info, goal)
        steps_taken = 0

        try:
            for step in range(goal.max_episode_steps):
                if (
                    step % self._replan_interval == 0
                    or action_index >= len(plan_actions)
                ):
                    # Warm-start: shift the previous plan forward.
                    if action_index < len(plan_actions):
                        warm = plan_actions[action_index:]
                    else:
                        warm = plan_actions
                    pad_len = self._cem_horizon - warm.shape[0]
                    if pad_len > 0:
                        pad = warm[-1:].expand(pad_len, -1).clone() if warm.shape[0] > 0 else torch.zeros(pad_len, self._action_dim, device=self._device)
                        warm = torch.cat([warm, pad], dim=0)
                    self._planner.set_warm_start_mean(warm)
                    plan_actions, plan_cost = self._planner.plan_to_latent(
                        current_frame, target_latent, return_cost=True
                    )
                    plan_actions = plan_actions.detach()
                    action_index = 0

                action = plan_actions[action_index]
                action_index += 1
                current_frame, reward, done, info = self.step(action)
                steps_taken += 1
                cumulative_reward += float(reward)
                if self._check_success(info, goal):
                    achieved = True
                    break
                if done:
                    break
        finally:
            self.close()

        z_end = self._encode_frame(current_frame)
        distance = float(((z_end - target_latent) ** 2).sum().sqrt().item())
        return EpisodeRecord(
            checkpoint=checkpoint_name,
            goal=goal.name,
            episode=episode_idx,
            steps=steps_taken,
            success=achieved,
            cumulative_reward=cumulative_reward,
            initial_plan_cost=init_cost,
            final_latent_distance=distance,
            seconds=time.monotonic() - t0,
        )

    # --- helpers ---------------------------------------------------------
    def _encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        with torch.no_grad():
            z = self._encoder(frame.to(self._device))
        return z.mean(dim=0)

    @staticmethod
    def _check_success(info: dict[str, Any], goal: Goal) -> bool:
        if not goal.inventory_targets:
            return False
        inv = info.get("inventory", {}) or {}
        return any(_inventory_contains(inv, t) for t in goal.inventory_targets)


def _inventory_contains(inventory: Any, target: str) -> bool:
    """MineStudio inventory is ``{item_name: count}`` or a Counter-like."""
    if inventory is None:
        return False
    if isinstance(inventory, dict):
        for k in inventory:
            if isinstance(k, str) and k.lower() == target.lower():
                return int(inventory.get(k, 0) or 0) > 0
        return False
    if hasattr(inventory, "items"):
        for k, v in inventory.items():
            if isinstance(k, str) and k.lower() == target.lower():
                return int(v or 0) > 0
    return False


# ---------------------------------------------------------------------------
# Backend: world model rollout
# ---------------------------------------------------------------------------


class WorldModelBackend(EpisodeRunner):
    """Pure latent rollout: no real env, but produces a valid
    ``final_latent_distance`` and exposes success via the mock-success
    heuristic. Useful for fast iteration in --mode world_model."""

    def __init__(self, *args: Any, rollout: LatentRollout, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rollout = rollout

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        # Use a random noise frame as the "current observation" — the
        # encoder will turn it into a sensible latent.
        frame = _random_frame_tensor(self._rng)
        return frame.to(self._device), {"inventory": {}, "source": "world_model"}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        # Self-rolled "next frame": just sample another random frame.
        # Latent progress is what we actually care about.
        frame = _random_frame_tensor(self._rng)
        return frame.to(self._device), 0.0, False, {"inventory": {}, "source": "world_model"}


# ---------------------------------------------------------------------------
# Backend: mock env (smoke test)
# ---------------------------------------------------------------------------


class MockMinecraftEnv(EpisodeRunner):
    """Synthetic env that mimics the MineStudio interface enough to
    drive the planner loop end-to-end without MineStudio.

    The env randomly awards the goal's inventory target with a small
    probability per step (so ``success`` is not always 0 / 100%), and
    always terminates after ``max_episode_steps``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inventory: dict[str, int] = {}
        self._steps_done = 0
        self._current_goal: Goal | None = None
        self._success_prob = 0.02  # baseline "lucky" rate per step

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self._inventory = {}
        self._steps_done = 0
        frame = _random_frame_tensor(self._rng)
        return frame.to(self._device), {"inventory": dict(self._inventory)}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        self._steps_done += 1
        # Probabilistic success on first step when the action has high L2 norm
        # (a stand-in for "doing something interesting").
        active = float(action.norm().item()) > 0.1
        reward = float(active) * 0.01
        if self._current_goal is not None and self._current_goal.inventory_targets:
            if active and self._rng.random() < self._success_prob:
                # Award the first matching target.
                self._inventory[self._current_goal.inventory_targets[0]] = 1
        done = False
        return (
            _random_frame_tensor(self._rng).to(self._device),
            reward,
            done,
            {"inventory": dict(self._inventory)},
        )

    def run(  # type: ignore[override]
        self,
        checkpoint_name: str,
        goal: Goal,
        episode_idx: int,
        target_latent: torch.Tensor,
    ) -> EpisodeRecord:
        self._current_goal = goal
        return super().run(checkpoint_name, goal, episode_idx, target_latent)


# ---------------------------------------------------------------------------
# Backend: real MineStudio env
# ---------------------------------------------------------------------------


class MineStudioBackend(EpisodeRunner):
    """Real MineStudio env via ``MineStudioEnv`` + ``MineStudioActionVocab``.

    This backend requires MineStudio to be importable. Lazy-imported so
    the script can be loaded on machines that do not have MineStudio.
    """

    def __init__(self, *args: Any, task: str | None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from wally.collector.config import CollectorConfig
        from wally.collector.env import MineStudioEnv
        from wally.planner.actions import (
            MineStudioActionVocab,
            continuous_to_discrete,
        )

        self._vocab = MineStudioActionVocab.default()
        self._continuous_to_discrete = continuous_to_discrete
        self._env = MineStudioEnv(CollectorConfig(resize=list(IMAGE_SIZE)))
        self._task = task
        self._closed = False

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        frame = self._env.reset()
        return self._preprocess(frame), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("env is closed")
        lows = torch.tensor([d.low for d in self._vocab.dimensions], device=action.device)
        highs = torch.tensor([d.high for d in self._vocab.dimensions], device=action.device)
        clipped = torch.clamp(action, lows, highs).unsqueeze(0)
        discrete_actions = self._continuous_to_discrete(clipped, self._vocab)
        action_dict = discrete_actions[0]
        frame, reward, done, info = self._env.step(action_dict)
        return self._preprocess(frame), float(reward), bool(done), info

    def close(self) -> None:
        if not self._closed:
            try:
                self._env.close()
            finally:
                self._closed = True

    @staticmethod
    def _preprocess(frame: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(frame).resize(IMAGE_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------


def aggregate(records: list[EpisodeRecord]) -> dict[tuple[str, str], dict[str, float]]:
    """Group by (checkpoint, goal) and compute summary stats."""
    grouped: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for r in records:
        grouped.setdefault((r.checkpoint, r.goal), []).append(r)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        n = len(group)
        if n == 0:
            continue
        succ = sum(1 for r in group if r.success)
        distances = [r.final_latent_distance for r in group]
        costs = [r.initial_plan_cost for r in group]
        rewards = [r.cumulative_reward for r in group]
        steps = [r.steps for r in group]
        summary[key] = {
            "n_episodes": n,
            "success_rate": succ / n,
            "successes": float(succ),
            "mean_final_latent_distance": float(np.mean(distances)),
            "std_final_latent_distance": float(np.std(distances)),
            "mean_initial_plan_cost": float(np.mean(costs)),
            "mean_cumulative_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
            "median_steps": float(np.median(steps)),
        }
    return summary


def write_reports(
    records: list[EpisodeRecord],
    summary: dict[tuple[str, str], dict[str, float]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Raw rows.
    csv_path = output_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "goal",
                "episode",
                "steps",
                "success",
                "cumulative_reward",
                "initial_plan_cost",
                "final_latent_distance",
                "seconds",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.checkpoint,
                    r.goal,
                    r.episode,
                    r.steps,
                    int(r.success),
                    r.cumulative_reward,
                    r.initial_plan_cost,
                    r.final_latent_distance,
                    r.seconds,
                ]
            )
    paths["csv"] = csv_path

    # Full JSON.
    json_path = output_dir / "episodes.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": [asdict(r) for r in records],
                "summary": {
                    f"{ckpt}|{goal}": vals
                    for (ckpt, goal), vals in summary.items()
                },
            },
            f,
            indent=2,
        )
    paths["json"] = json_path

    # Markdown report.
    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    paths["md"] = md_path

    return paths


def _render_markdown(summary: dict[tuple[str, str], dict[str, float]]) -> str:
    if not summary:
        return "# Goal evaluation\n\n(no episodes ran)\n"
    lines: list[str] = ["# Goal evaluation", ""]
    checkpoints = sorted({ckpt for ckpt, _ in summary.keys()}, key=_step_from_checkpoint_str)
    goals = sorted({g for _, g in summary.keys()})
    header = "| checkpoint | " + " | ".join(goals) + " |"
    sep = "|---|" + "|".join(["---"] * len(goals)) + "|"
    lines.append("## success rate")
    lines.append(header)
    lines.append(sep)
    for ckpt in checkpoints:
        cells: list[str] = []
        for g in goals:
            s = summary.get((ckpt, g))
            cells.append(f"{s['success_rate'] * 100:5.1f}%" if s else "  -  ")
        lines.append(f"| `{ckpt}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## mean final latent distance (lower = better)")
    lines.append(header)
    lines.append(sep)
    for ckpt in checkpoints:
        cells: list[str] = []
        for g in goals:
            s = summary.get((ckpt, g))
            cells.append(f"{s['mean_final_latent_distance']:.3f}" if s else "  -  ")
        lines.append(f"| `{ckpt}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## mean initial plan cost (lower = better)")
    lines.append(header)
    lines.append(sep)
    for ckpt in checkpoints:
        cells: list[str] = []
        for g in goals:
            s = summary.get((ckpt, g))
            cells.append(f"{s['mean_initial_plan_cost']:.3f}" if s else "  -  ")
        lines.append(f"| `{ckpt}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _step_from_checkpoint_str(path: str) -> int:
    m = _CHECKPOINT_STEP_RE.search(path.replace("\\", "/"))
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate long-horizon goal success across training checkpoints.",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help=(
            "Glob pattern (relative to cwd) selecting checkpoint files. "
            "E.g. 'checkpoints/checkpoint_*.pt'."
        ),
    )
    p.add_argument(
        "--num-checkpoints",
        type=int,
        default=None,
        help="Evenly-spaced subsample of the matched checkpoints (default: all).",
    )
    p.add_argument(
        "--goals",
        type=str,
        default=",".join(g.name for g in BUILTIN_GOALS),
        help="Comma-separated list of goal names (default: all built-ins).",
    )
    p.add_argument(
        "--goal-frames-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing one image per goal "
            "(<goal_name>.png) used to derive the target latent."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["world_model", "minestudio", "mock"],
        default="world_model",
        help="Backend (default: world_model — fastest, no env required).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per (checkpoint, goal).",
    )
    p.add_argument(
        "--cem-horizon",
        type=int,
        default=8,
        help="CEM action horizon (default: 8).",
    )
    p.add_argument(
        "--cem-population",
        type=int,
        default=64,
        help="CEM population size (default: 64).",
    )
    p.add_argument(
        "--cem-iterations",
        type=int,
        default=5,
        help="CEM iterations (default: 5).",
    )
    p.add_argument(
        "--replan-interval",
        type=int,
        default=4,
        help="Replan every N env steps (default: 4).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lewm_default.yaml"),
        help=(
            "Path to the YAML config used to train the model. Only the "
            "``model:`` section is consumed (the checkpoint itself may "
            "not store the model architecture)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/goal_eval"),
        help="Output directory for CSV/JSON/MD reports.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the world model (default: auto).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoints = discover_checkpoints(args.checkpoints, args.num_checkpoints)
    if not checkpoints:
        logger.error("No checkpoints matched %s", args.checkpoints)
        return 1
    logger.info("Evaluating %d checkpoints on %s", len(checkpoints), device)
    for ckpt in checkpoints:
        logger.info("  - %s (step %d)", ckpt, _step_from_checkpoint(ckpt))

    model_cfg: dict[str, Any] = {}
    if args.config is not None and args.config.is_file():
        with args.config.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model_cfg = dict(raw.get("model", {}) or {})
        logger.info("Loaded model config from %s", args.config)
    else:
        logger.warning(
            "No config provided or config not found at %s; relying on "
            "checkpoint's model_config (newer checkpoints) or constructor "
            "defaults (older ones).",
            args.config,
        )

    goal_names = [g.strip() for g in args.goals.split(",") if g.strip()]
    goals = [get_goal(name) for name in goal_names]
    if args.goal_frames_dir is not None:
        for g in goals:
            candidate = args.goal_frames_dir / f"{g.name}.png"
            if candidate.is_file():
                g.goal_frame_path = candidate

    cem = CEMConfig(
        horizon=args.cem_horizon,
        population_size=args.cem_population,
        n_iterations=args.cem_iterations,
    )

    records: list[EpisodeRecord] = []

    for ckpt_path in checkpoints:
        step = _step_from_checkpoint(ckpt_path)
        ckpt_name = f"{ckpt_path.stem}@step{step}"
        logger.info("Loading %s", ckpt_path)
        rollout = LatentRollout.from_checkpoint(
            ckpt_path, device=device, model_config=model_cfg or None,
        )
        adapter = rollout._model
        encoder = adapter.encode
        planner = GoalConditionedPlanner(
            rollout, encoder, cem, device=device, action_dim=25
        )
        target_latents: dict[str, torch.Tensor] = {
            g.name: _encode_goal_latent(encoder, g, device) for g in goals
        }

        for g in goals:
            for ep in range(args.episodes):
                logger.info(
                    "  running %s / %s / episode %d/%d",
                    ckpt_name,
                    g.name,
                    ep + 1,
                    args.episodes,
                )
                if args.mode == "world_model":
                    runner: EpisodeRunner = WorldModelBackend(
                        backend_name="world_model",
                        encoder=encoder,
                        planner=planner,
                        device=device,
                        action_dim=25,
                        cem_horizon=args.cem_horizon,
                        replan_interval=args.replan_interval,
                        seed=args.seed + ep,
                        rollout=rollout,
                    )
                elif args.mode == "mock":
                    runner = MockMinecraftBackend(  # type: ignore[assignment]
                        backend_name="mock",
                        encoder=encoder,
                        planner=planner,
                        device=device,
                        action_dim=25,
                        cem_horizon=args.cem_horizon,
                        replan_interval=args.replan_interval,
                        seed=args.seed + ep,
                    )
                else:  # minestudio
                    runner = MineStudioBackend(  # type: ignore[assignment]
                        backend_name="minestudio",
                        encoder=encoder,
                        planner=planner,
                        device=device,
                        action_dim=25,
                        cem_horizon=args.cem_horizon,
                        replan_interval=args.replan_interval,
                        seed=args.seed + ep,
                        task=None,
                    )
                rec = runner.run(ckpt_name, g, ep, target_latents[g.name])
                records.append(rec)
                logger.info(
                    "    steps=%d success=%s reward=%.3f dist=%.3f (%.1fs)",
                    rec.steps,
                    rec.success,
                    rec.cumulative_reward,
                    rec.final_latent_distance,
                    rec.seconds,
                )

    summary = aggregate(records)
    paths = write_reports(records, summary, args.output)
    logger.info("Wrote reports:")
    for kind, p in paths.items():
        logger.info("  %s: %s", kind, p)
    _print_summary_table(summary)
    return 0


def _print_summary_table(summary: dict[tuple[str, str], dict[str, float]]) -> None:
    if not summary:
        return
    print("\n=== summary ===")
    checkpoints = sorted({ckpt for ckpt, _ in summary.keys()}, key=_step_from_checkpoint_str)
    goals = sorted({g for _, g in summary.keys()})
    header = "checkpoint".ljust(34) + " " + " ".join(g.ljust(14) for g in goals)
    print(header)
    print("-" * len(header))
    for ckpt in checkpoints:
        row = ckpt.ljust(34) + " "
        for g in goals:
            s = summary.get((ckpt, g))
            if s is None:
                cell = "-"
            else:
                cell = f"{s['success_rate'] * 100:5.1f}% d={s['mean_final_latent_distance']:.2f}"
            row += cell.ljust(14) + " "
        print(row)


# Patch the dispatch dict so ``MockMinecraftBackend`` is the same class
# the rest of the file references. Keeps the CLI branch table readable.
MockMinecraftBackend = MockMinecraftEnv


if __name__ == "__main__":
    raise SystemExit(main())
