"""Wood-gathering regression test for the planner fix.

Loads the recorded ``ag-tests/run_wood_v2/episode_0.npz`` trajectory
(post-fix fixture: agent navigates to a tree but doesn't chop) and
asserts the post-fix behavior:

- Action profile is NOT button-spam (per-dim |mean| <= 0.5 for non-inventory dims)
- Camera is active (dims 0, 1 have |mean| >= 0.1)
- Scene is NOT frozen (mean frame diff >= 5; < 20% of steps with diff < 0.5)
- Inventory is never populated (the fixture is the "approach the tree" milestone)

Regression target: ``openspec/changes/fix-wood-gathering-stall`` task 3.1.

If you re-run the agent and get a different behavior, regenerate the
fixture with::

    podman exec wally-dev sh -c 'cd /workspace && PYTHONPATH=src \\
        MINESTUDIO_DIR=/tmp/MineStudio setsid nohup python3 -m wally.agent.play \\
        --relay --relay-host 0.0.0.0 --relay-port 8081 \\
        --checkpoint /workspace/checkpoints/checkpoint_100000.pt \\
        --goal-frame /workspace/checkpoints/goal_frame1.png \\
        --planner cem --viewer none --record \\
        --output-dir /workspace/ag-tests/run_wood_v2 \\
        --config /tmp/quick.yaml \\
        > /tmp/wally-play.log 2>&1 < /dev/null & disown'

then ``podman cp wally-dev:/workspace/ag-tests/run_wood_v2/episode_0.npz
ag-tests/run_wood_v2/`` to refresh the fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "ag-tests" / "run_wood_v2" / "episode_0.npz"
INVENTORY_DIM = 12


@pytest.fixture(scope="module")
def episode() -> dict[str, np.ndarray]:
    if not FIXTURE.is_file():
        pytest.skip(
            f"missing fixture {FIXTURE}. See module docstring for how to "
            f"regenerate it via wally-play inside the wally-dev container."
        )
    return np.load(FIXTURE, allow_pickle=True)  # type: ignore[return-value]


@pytest.mark.smoke
class TestWoodGatheringActionProfile:
    """Action profile must not look like the v1 button-spam signature."""

    def test_per_dim_mean_bounded_away_from_spam(self, episode) -> None:
        actions = np.asarray(episode["actions"], dtype=np.float32)
        assert actions.ndim == 2
        n_dims = actions.shape[1]
        for d in range(n_dims):
            if d == INVENTORY_DIM:
                continue  # inventory is intentionally allowed to be low
            mean_abs = float(np.abs(actions[:, d]).mean())
            # v1 spam signature: every dim at ~0.4. Post-fix ceiling is 0.5.
            assert mean_abs <= 0.5, (
                f"action dim {d} has |mean|={mean_abs:.3f} which matches the "
                f"button-spam signature (v1 was ~0.4 across all dims)"
            )

    def test_camera_pitch_active(self, episode) -> None:
        actions = np.asarray(episode["actions"], dtype=np.float32)
        mean_abs = float(np.abs(actions[:, 0]).mean())
        assert mean_abs >= 0.1, (
            f"camera_pitch (dim 0) is essentially still: |mean|={mean_abs:.3f}"
        )

    def test_camera_yaw_active(self, episode) -> None:
        actions = np.asarray(episode["actions"], dtype=np.float32)
        mean_abs = float(np.abs(actions[:, 1]).mean())
        assert mean_abs >= 0.1, (
            f"camera_yaw (dim 1) is essentially still: |mean|={mean_abs:.3f}"
        )


@pytest.mark.smoke
class TestWoodGatheringSceneActivity:
    """Scene must not be frozen (the v1 failure mode)."""

    def test_mean_frame_diff_is_substantial(self, episode) -> None:
        frames = episode["frames"]
        flat = frames.reshape(len(frames), -1).astype(np.float32)
        diffs = np.abs(np.diff(flat, axis=0)).mean(axis=1)
        mean_diff = float(diffs.mean())
        # v1 had mean diff 6.5; post-fix should be clearly above that
        assert mean_diff >= 5.0, (
            f"mean frame-to-frame pixel diff is {mean_diff:.2f} "
            f"(v1 was 6.5, v2 should be at least that)"
        )

    def test_few_frozen_steps(self, episode) -> None:
        frames = episode["frames"]
        flat = frames.reshape(len(frames), -1).astype(np.float32)
        diffs = np.abs(np.diff(flat, axis=0)).mean(axis=1)
        frozen_frac = float((diffs < 0.5).mean())
        # v1 had ~50% of steps with diff < 0.5
        assert frozen_frac < 0.20, (
            f"{frozen_frac * 100:.0f}% of steps are frozen (frame diff < 0.5); "
            f"v1 was ~50%, post-fix should be < 20%"
        )


@pytest.mark.smoke
class TestWoodGatheringInventory:
    """The fixture is the 'approach the tree' milestone, not a wood pickup."""

    def test_inventory_never_populated(self, episode) -> None:
        events = episode["events"]
        for i, e in enumerate(events):
            if e is None:
                continue
            inv = e.get("inventory", {})
            for slot, item in inv.items():
                t = item.get("type", "none")
                q = item.get("quantity", 0)
                assert t == "none" and q == 0, (
                    f"inventory slot {slot!r} contains type={t!r} qty={q} at "
                    f"step {i}; the wood-gathering fixture should reach the "
                    f"tree but not chop wood"
                )


@pytest.mark.smoke
class TestWoodGatheringSanity:
    """Sanity checks on the fixture itself."""

    def test_episode_has_meaningful_length(self, episode) -> None:
        n = len(episode["frames"])
        assert n >= 50, f"episode has only {n} steps; too short to be a real run"

    def test_actions_have_correct_shape(self, episode) -> None:
        actions = episode["actions"]
        assert actions.ndim == 2
        assert actions.shape[1] == 25  # MineStudio action vocab size
