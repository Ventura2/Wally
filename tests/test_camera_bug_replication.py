"""Unit tests that replicate the camera-index / camera-rescale bugs.

These tests are intentionally written to fail against the current code so
they document (and pin down) the two bugs we are tracking:

  1. ``AgentLoop.run_episode`` clamps ``action[10:11]``, which in the
     agent's vocab is **attack** (idx 10) and **drop** (idx 11), NOT
     ``camera_pitch`` (idx 0) and ``camera_yaw`` (idx 1). The "camera-
     shake workaround" is therefore clamping the wrong dims.

  2. ``MineStudioAgentEnv._translate_action`` rescales the camera by
     ``* 180.0`` to convert the agent's ``[-1, 1]`` action into
     MineStudio's expected degrees. The L0 was trained on data where
     the camera was stored in raw degrees clamped to ``[-1, 1]`` by
     ``src/wally/data/dataset.py:66`` (observed range -42 to +37
     degrees in the raw shards). Multiplying the agent's value by 180
     is therefore a 180x scale mismatch on the env side: a planner
     proposal of ``action[0] = 0.5`` reaches MineStudio as 90 degrees
     of camera motion per step, while the L0's predicted ``Delta z``
     reflects what it learned for 0.5 *degrees*.

The third test class pins down the agent's action layout so the
index assignments above can't silently change.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from wally.agent.config import AgentConfig
from wally.agent.loop import AgentLoop
from wally.agent.protocol import PlanResult
from wally.planner.actions import MineStudioActionVocab

# ---------------------------------------------------------------------------
# Action-layout sanity check
# ---------------------------------------------------------------------------


class TestAgentVocabLayout:
    """The agent's vocab is the source of truth for the index layout.

    These assertions pin down the *current* contract. If they break
    (e.g. someone reorders the vocab to match the training data's
    env-action schema), the camera-clamp test below will need to be
    updated alongside.
    """

    def test_camera_pitch_is_at_index_0(self):
        names = [d.name for d in MineStudioActionVocab.default().dimensions]
        assert names[0] == "camera_pitch", (
            f"Agent vocab index 0 should be camera_pitch, got {names[0]!r}. "
            f"This is what the env reads at continuous[0] (env.py:92-95)."
        )

    def test_camera_yaw_is_at_index_1(self):
        names = [d.name for d in MineStudioActionVocab.default().dimensions]
        assert names[1] == "camera_yaw", (
            f"Agent vocab index 1 should be camera_yaw, got {names[1]!r}. "
            f"This is what the env reads at continuous[1] (env.py:96-99)."
        )

    def test_attack_is_at_index_10(self):
        names = [d.name for d in MineStudioActionVocab.default().dimensions]
        assert names[10] == "attack", (
            f"Agent vocab index 10 should be attack, got {names[10]!r}."
        )

    def test_drop_is_at_index_11(self):
        names = [d.name for d in MineStudioActionVocab.default().dimensions]
        assert names[11] == "drop", (
            f"Agent vocab index 11 should be drop, got {names[11]!r}."
        )

    def test_inventory_is_at_index_12(self):
        names = [d.name for d in MineStudioActionVocab.default().dimensions]
        assert names[12] == "inventory", (
            f"Agent vocab index 12 should be inventory, got {names[12]!r}. "
            f"This is the dim the loop already masks to 0 (loop.py:122-123)."
        )


# ---------------------------------------------------------------------------
# Bug #1: AgentLoop clamps attack/drop instead of camera
# ---------------------------------------------------------------------------


def _spy_env(observed: list[torch.Tensor], num_steps: int = 1) -> MagicMock:
    """Build a mock env that records the action passed to ``step``.

    Returns ``done=True`` after ``num_steps`` calls so the loop exits
    cleanly; pass ``num_steps`` larger than the episode_timeout to
    make the loop run until the timeout.
    """
    env = MagicMock()
    env.reset.return_value = torch.rand(3, 64, 64)

    def step_fn(action):
        observed.append(action.detach().clone())
        done = len(observed) >= num_steps
        return torch.rand(3, 64, 64), 0.0, done, {}

    env.step.side_effect = step_fn
    return env


class TestAgentLoopCameraClampTargetsCorrectIndices:
    """The camera-shake workaround in ``AgentLoop.run_episode`` must
    clamp ``action[0:2]`` (camera_pitch, camera_yaw), NOT
    ``action[10:12]`` (attack, drop).

    Currently fails: the loop clamps ``action[10:11]`` because the
    code was written against the *training* action schema (where
    camera is at indices 10/11) instead of the *agent* vocab (where
    camera is at indices 0/1).
    """

    def test_camera_pitch_is_clamped_to_small_value(self):
        observed: list[torch.Tensor] = []
        env = _spy_env(observed)

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 0] = 1.0
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=1, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert observed, "env.step was never called"
        env_action = observed[0]
        assert env_action[0].abs().item() <= 0.11, (
            f"Expected camera_pitch (idx 0) clamped to <= 0.1, got "
            f"{env_action[0].item():.4f}. The camera-shake clamp is "
            f"hitting the wrong index."
        )

    def test_camera_yaw_is_clamped_to_small_value(self):
        observed: list[torch.Tensor] = []
        env = _spy_env(observed)

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 1] = 1.0
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=1, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert observed, "env.step was never called"
        env_action = observed[0]
        assert env_action[1].abs().item() <= 0.11, (
            f"Expected camera_yaw (idx 1) clamped to <= 0.1, got "
            f"{env_action[1].item():.4f}."
        )

    def test_attack_is_not_clamped(self):
        """A non-zero attack proposal must reach the env unchanged.

        The current bug clamps attack to <= 0.1, which (after binning
        to 0/1 in continuous_to_discrete) means the agent can never
        press attack. The wood-gathering agent relies on attack to
        chop trees.
        """
        observed: list[torch.Tensor] = []
        env = _spy_env(observed)

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 10] = 0.7
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=1, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert observed, "env.step was never called"
        env_action = observed[0]
        assert env_action[10].item() == pytest.approx(0.7, abs=1e-5), (
            f"Expected attack (idx 10) to pass through unchanged at 0.7, "
            f"got {env_action[10].item():.4f}. The camera-shake clamp is "
            f"hitting attack instead of camera."
        )

    def test_drop_is_not_clamped(self):
        observed: list[torch.Tensor] = []
        env = _spy_env(observed)

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 11] = 0.7
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=1, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert observed, "env.step was never called"
        env_action = observed[0]
        assert env_action[11].item() == pytest.approx(0.7, abs=1e-5), (
            f"Expected drop (idx 11) to pass through unchanged at 0.7, "
            f"got {env_action[11].item():.4f}."
        )


# ---------------------------------------------------------------------------
# Bug #2: MineStudioAgentEnv rescales camera by 180x
# ---------------------------------------------------------------------------


class TestEnvCameraRescaleMatchesTrainingData:
    """The env's ``* 180.0`` is a 180x scale mismatch vs the L0's
    training distribution.

    The L0 was trained on raw camera deltas in degrees, clamped to
    ``[-1, 1]`` by ``src/wally/data/dataset.py:66`` (observed range
    -42 to +37 in the training shards). A planner proposal of
    ``action[0] = 0.5`` therefore represents "0.5 degrees" to the
    L0 — but the env currently sends ``0.5 * 180 = 90 degrees`` to
    MineStudio, a 180x overshoot.

    Once fixed, ``env._translate_action`` should pass the agent's
    value through unchanged (treating the agent's ``[-1, 1]`` as
    already-in-degrees).
    """

    @pytest.fixture
    def env_with_capture(self):
        with patch("wally.collector.env._MinecraftSim") as mock:
            sim = mock.return_value
            sim.reset.return_value = (
                {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
                {},
            )
            captured_actions: list[dict] = []

            def capture(action_dict):
                captured_actions.append({k: v for k, v in action_dict.items()})
                return (
                    {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
                    0.0,
                    False,
                    False,
                    {},
                )

            sim.step.side_effect = capture

            from wally.agent.env import MineStudioAgentEnv

            config = AgentConfig(resize=(64, 64))
            env = MineStudioAgentEnv(config)
            yield env, sim, captured_actions

    def test_camera_pitch_05_passes_through_as_05_degrees(self, env_with_capture):
        env, _, captured = env_with_capture
        env.reset()

        action = torch.zeros(25)
        action[0] = 0.5
        env.step(action)

        assert captured, "MineStudio.step was never called"
        camera = captured[0].get("camera")
        assert camera is not None, (
            f"MineStudio action dict has no 'camera' key; got {captured[0]!r}"
        )
        assert camera[0] == pytest.approx(0.5, abs=1e-4), (
            f"Expected camera[0] (pitch) = 0.5 degrees (matching L0's "
            f"training distribution), got {camera[0]!r}. The env is "
            f"rescaling by 180x, but the L0 was trained on raw degrees."
        )

    def test_camera_yaw_05_passes_through_as_05_degrees(self, env_with_capture):
        env, _, captured = env_with_capture
        env.reset()

        action = torch.zeros(25)
        action[1] = 0.5
        env.step(action)

        assert captured, "MineStudio.step was never called"
        camera = captured[0].get("camera")
        assert camera is not None
        assert camera[1] == pytest.approx(0.5, abs=1e-4), (
            f"Expected camera[1] (yaw) = 0.5 degrees, got {camera[1]!r}."
        )

    def test_camera_pitch_negative_one_passes_through_as_minus_one_degree(
        self, env_with_capture
    ):
        env, _, captured = env_with_capture
        env.reset()

        action = torch.zeros(25)
        action[0] = -1.0
        env.step(action)

        assert captured
        camera = captured[0].get("camera")
        assert camera is not None
        assert camera[0] == pytest.approx(-1.0, abs=1e-4), (
            f"Expected camera[0] = -1.0 degrees, got {camera[0]!r}."
        )


# ---------------------------------------------------------------------------
# Data distribution: verify the L0 was trained on raw degrees
# ---------------------------------------------------------------------------


class TestTrainingDataCameraDistribution:
    """Ground truth for the L0's training distribution.

    If the camera values in the converted shards are in raw degrees
    (range similar to [-42, +37] as documented in
    ``src/wally/data/dataset.py:63-65``), then the env's ``* 180.0``
    rescale is a 180x overshoot and must be removed.

    If the camera values are already normalized to [-1, 1] (range
    similar to [-0.23, +0.21] after dividing by 180), then the env's
    ``* 180.0`` is the correct translation back to degrees and the
    training-time range is what it is.

    This test is data-dependent: skipped when no shards are present.
    """

    SHARD_DIR = "data/shards/treechop_full"

    def _load_first_npz(self) -> dict | None:
        import io
        import tarfile
        from pathlib import Path

        shard_dir = Path(self.SHARD_DIR)
        if not shard_dir.is_dir():
            return None
        shards = sorted(shard_dir.glob("*.tar"))
        if not shards:
            return None
        with tarfile.open(shards[0], "r") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".npz"):
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    npz = np.load(io.BytesIO(f.read()))
                    return {k: npz[k] for k in npz.files}
        return None

    def test_camera_pitch_distribution_is_in_degrees(self):
        sample = self._load_first_npz()
        if sample is None:
            pytest.skip(f"No shards in {self.SHARD_DIR}")
        if "actions" not in sample or sample["actions"].shape[-1] < 12:
            pytest.skip("Sample actions missing camera columns")
        actions = sample["actions"]
        pitch = actions[:, 10]
        yaw = actions[:, 11]
        assert pitch.min() < -1.5 or pitch.max() > 1.5, (
            f"camera_pitch training data is NOT in raw degrees "
            f"(range [{pitch.min():.3f}, {pitch.max():.3f}]); it must "
            f"have been normalized somewhere. If so, the env's * 180.0 "
            f"rescale is correct and the fix is elsewhere."
        )
        assert yaw.min() < -1.5 or yaw.max() > 1.5, (
            f"camera_yaw training data is NOT in raw degrees "
            f"(range [{yaw.min():.3f}, {yaw.max():.3f}])."
        )


# ---------------------------------------------------------------------------
# Rollout: the L0 must see the camera at the *training* indices 10/11
# ---------------------------------------------------------------------------


class TestRolloutCameraPermutation:
    """The L0 was trained on the MineStudio env-action schema where
    camera_pitch lives at training index 10 and camera_yaw at
    training index 11. The ``LeWorldModelAdapter.predict`` permutes
    the agent's action (camera at 0/1) into the training schema
    (camera at 10/11) before embedding.

    This test pins down that permutation by calling ``predict`` on a
    tiny action and asserting the L0's action_embedder sees the
    camera value at training index 10, not at training index 0.
    """

    def test_camera_value_lands_at_training_index_10(self):
        from wally.planner.rollout import _translate_agent_action_to_l0

        agent_action = torch.zeros(25)
        agent_action[0] = 0.5

        l0_action = _translate_agent_action_to_l0(agent_action)

        assert l0_action.shape == (25,)
        assert l0_action[10].item() == pytest.approx(0.5, abs=1e-6), (
            f"Expected camera_pitch (agent idx 0) to land at training "
            f"idx 10, got {l0_action[10].item()!r}."
        )

    def test_camera_value_lands_at_training_index_11(self):
        from wally.planner.rollout import _translate_agent_action_to_l0

        agent_action = torch.zeros(25)
        agent_action[1] = 0.5

        l0_action = _translate_agent_action_to_l0(agent_action)

        assert l0_action[11].item() == pytest.approx(0.5, abs=1e-6), (
            f"Expected camera_yaw (agent idx 1) to land at training "
            f"idx 11, got {l0_action[11].item()!r}."
        )

    def test_no_180x_rescale_on_camera(self):
        """The L0 was trained on degrees clamped to [-1, 1]. The
        rollout must NOT multiply the agent's [-1, 1] by 180 again.
        """
        from wally.planner.rollout import _translate_agent_action_to_l0

        agent_action = torch.zeros(25)
        agent_action[0] = 0.5
        agent_action[1] = 0.5

        l0_action = _translate_agent_action_to_l0(agent_action)

        assert l0_action[10].abs().item() <= 1.0 + 1e-6, (
            f"camera_pitch training value out of training range "
            f"[{l0_action[10].item()!r}]; the rollout is rescaling the "
            f"camera and pushing it out of distribution."
        )
        assert l0_action[11].abs().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# End-to-end smoke: 50 steps with saturated camera, verify stable trajectory
# ---------------------------------------------------------------------------


class TestCameraTrajectorySmoke:
    """End-to-end smoke: the agent loop's camera clamp + EMA smoothing
    must keep the per-step camera motion bounded over many steps, even
    when the planner proposes saturated values.

    Before the fix (indices 10/11), the camera was unclamped, the
    attack/drop actions were clamped instead, and the env rescaled by
    180 — so a planner proposal of 1.0 reached MineStudio as 180
    degrees per step. Over 50 steps that compounds to 9000 degrees
    of camera motion (≫ looking-at-sky).

    After the fix, the camera is clamped to [-0.1, 0.1] and passed
    through unchanged as degrees, so the env receives |camera| ≤ 0.11
    per step (≤ 5.5 degrees over 50 steps — the view drifts slowly
    rather than snapping to the sky).
    """

    @staticmethod
    def _constant_planner(value: float) -> MagicMock:
        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 0] = value
        planned[0, 1] = value
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)
        return planner

    def test_50_steps_with_saturated_camera_stays_bounded(self) -> None:
        observed: list[torch.Tensor] = []
        env = _spy_env(observed, num_steps=50)

        planner = self._constant_planner(1.0)
        config = AgentConfig(episode_timeout=50, replan_interval=1)
        loop = AgentLoop(env, planner, config)

        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert len(observed) == 50, (
            f"Expected 50 env.step calls, got {len(observed)}"
        )
        max_abs_camera = max(
            max(a[0].abs().item(), a[1].abs().item()) for a in observed
        )
        assert max_abs_camera <= 0.11, (
            f"Camera action reached env with max |camera|={max_abs_camera:.4f} "
            f"over 50 steps. After the fix, the clamp+EMA should keep it "
            f"≤ 0.1 per step. The camera-shake workaround is hitting the "
            f"wrong indices or the env is over-rescaling."
        )

    def test_50_steps_with_alternating_camera_does_not_explode(self) -> None:
        """Planner alternates between +1.0 and -1.0. The EMA should
        keep the camera near 0; nothing should explode."""
        observed: list[torch.Tensor] = []
        env = _spy_env(observed, num_steps=50)

        step_counter = {"i": 0}

        def plan_fn(*args, **kwargs):
            v = 1.0 if (step_counter["i"] % 2 == 0) else -1.0
            step_counter["i"] += 1
            planned = torch.zeros(1, 25)
            planned[0, 0] = v
            planned[0, 1] = v
            return PlanResult(actions=planned, cost=0.0)

        planner = MagicMock()
        planner.plan.side_effect = plan_fn

        config = AgentConfig(episode_timeout=50, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert len(observed) == 50
        max_abs_camera = max(
            max(a[0].abs().item(), a[1].abs().item()) for a in observed
        )
        assert max_abs_camera <= 0.11, (
            f"Alternating ±1.0 camera produced max |camera|={max_abs_camera:.4f} "
            f"at the env. The EMA should keep it bounded."
        )

    def test_attack_action_actually_reaches_env(self) -> None:
        """Regression check: the camera-shake workaround must not
        silently zero the attack action. A planner that proposes
        attack=0.7 must have 0.7 reach the env (after the env's
        continuous_to_discrete binning, that maps to attack=1)."""
        observed: list[torch.Tensor] = []
        env = _spy_env(observed)

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 10] = 0.7
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=10, replan_interval=1)
        loop = AgentLoop(env, planner, config)
        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        for i, a in enumerate(observed):
            assert a[10].item() == pytest.approx(0.7, abs=1e-5), (
                f"Step {i}: attack was mutated from 0.7 to {a[10].item():.4f}. "
                f"The camera-shake clamp is hitting attack instead of camera, "
                f"which means the agent can never chop trees."
            )
