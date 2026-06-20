from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from agent.config import AgentConfig
from agent.loop import AgentLoop
from agent.protocol import PlanResult


def _make_env(
    step_side_effect: list | None = None,
    num_steps: int = 20,
) -> MagicMock:
    env = MagicMock()
    env.reset.return_value = torch.rand(3, 64, 64)
    if step_side_effect is not None:
        env.step.side_effect = step_side_effect
    else:
        env.step.side_effect = [
            (torch.rand(3, 64, 64), 0.0, False, {})
            for _ in range(num_steps)
        ]
    return env


def _make_planner(
    horizon: int = 8,
    cost: float = 1.0,
    has_warm_start: bool = True,
) -> MagicMock:
    planner = MagicMock()
    planner.plan.return_value = PlanResult(
        actions=torch.randn(horizon, 25),
        cost=cost,
    )
    if not has_warm_start:
        del planner.set_warm_start_mean
    return planner


class TestAgentLoopRunsToTimeout:
    def test_episode_runs_to_timeout(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=10, replan_interval=4)
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.steps == 10
        assert result.interrupted is False


class TestAgentLoopEndsOnDone:
    def test_episode_ends_on_done(self) -> None:
        effects = [
            (torch.rand(3, 64, 64), 0.0, False, {})
            for _ in range(4)
        ]
        effects.append((torch.rand(3, 64, 64), 0.0, True, {}))
        env = _make_env(step_side_effect=effects)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=100, replan_interval=4)
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.steps == 5
        assert result.interrupted is False


class TestAgentLoopReplanInterval:
    def test_replan_interval(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=12, replan_interval=4)
        loop = AgentLoop(env, planner, config)

        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert planner.plan.call_count == 3


class TestAgentLoopWarmStart:
    def test_warm_start_on_replan(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner(horizon=8)
        config = AgentConfig(episode_timeout=12, replan_interval=4)
        loop = AgentLoop(env, planner, config)

        first_actions = torch.arange(200, dtype=torch.float32).reshape(8, 25)
        planner.plan.return_value = PlanResult(actions=first_actions, cost=1.0)

        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert planner.set_warm_start_mean.call_count == 2
        warm_start_arg = planner.set_warm_start_mean.call_args_list[1][0][0]
        expected_shifted = first_actions[4:]
        expected_pad = expected_shifted[-1:].expand(4, -1)
        expected = torch.cat([expected_shifted, expected_pad], dim=0)
        assert warm_start_arg.shape == (8, 25)
        torch.testing.assert_close(warm_start_arg, expected)


class TestAgentLoopEarlyReplan:
    def test_early_replan_when_actions_exhausted(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner(horizon=3)
        config = AgentConfig(episode_timeout=10, replan_interval=8)
        loop = AgentLoop(env, planner, config)

        loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert planner.plan.call_count == 4


class TestAgentLoopKeyboardInterrupt:
    def test_keyboard_interrupt(self) -> None:
        effects = [
            (torch.rand(3, 64, 64), 0.0, False, {})
            for _ in range(5)
        ]
        effects.append(KeyboardInterrupt())
        env = _make_env(step_side_effect=effects)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=100, replan_interval=4)
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.interrupted is True
        assert result.steps == 5
        env.close.assert_called_once()


class TestAgentLoopRecordingEnabled:
    def test_recording_enabled(self) -> None:
        env = _make_env(
            step_side_effect=[
                (torch.rand(3, 64, 64), 0.0, False, {"inventory": {"oak_log": 1}})
                for _ in range(20)
            ]
        )
        planner = _make_planner()
        config = AgentConfig(
            episode_timeout=10, replan_interval=4, record_trajectory=True
        )
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.trajectory is not None
        assert result.trajectory["frames"].shape == (10, 64, 64, 3)
        assert result.trajectory["actions"].shape == (10, 25)
        assert result.trajectory["events"].shape == (10,)
        assert result.trajectory["events"][0]["inventory"] == {"oak_log": 1}


class TestAgentLoopActionPassthrough:
    def test_loop_does_not_mutate_inventory_action(self) -> None:
        observed_actions: list[torch.Tensor] = []

        env = MagicMock()
        env.reset.return_value = torch.rand(3, 64, 64)

        def step_fn(action):
            observed_actions.append(action.clone())
            return torch.rand(3, 64, 64), 0.0, True, {}

        env.step.side_effect = step_fn

        planner = MagicMock()
        planned = torch.zeros(1, 25)
        planned[0, 12] = 1.0
        planner.plan.return_value = PlanResult(actions=planned, cost=0.0)

        config = AgentConfig(episode_timeout=1, replan_interval=1)
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.steps == 1
        assert observed_actions
        assert observed_actions[0][12].item() == 1.0


class TestAgentLoopRecordingDisabled:
    def test_recording_disabled(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(
            episode_timeout=10, replan_interval=4, record_trajectory=False
        )
        loop = AgentLoop(env, planner, config)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.trajectory is None


class _RecordingViewer:
    def __init__(self, quit_at: int | None = None) -> None:
        self.shows: list = []
        self.close_count = 0
        self._quit_at = quit_at
        self._calls = 0

    def show(self, pov, info=None) -> None:
        self.shows.append((pov, info))
        self._calls += 1

    def should_quit(self) -> bool:
        if self._quit_at is not None and self._calls >= self._quit_at:
            return True
        return False

    def close(self) -> None:
        self.close_count += 1


class TestAgentLoopViewer:
    @pytest.mark.smoke
    def test_show_called_per_step(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=5, replan_interval=4)
        viewer = _RecordingViewer()
        loop = AgentLoop(env, planner, config, viewer=viewer)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.steps == 5
        assert result.interrupted is False
        assert len(viewer.shows) == 5
        assert viewer.close_count == 1

    def test_breaks_on_should_quit(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=100, replan_interval=4)
        viewer = _RecordingViewer(quit_at=3)
        loop = AgentLoop(env, planner, config, viewer=viewer)

        result = loop.run_episode(goal_frame=torch.rand(3, 64, 64))

        assert result.interrupted is True
        assert result.steps == 3
        assert len(viewer.shows) == 3
        assert viewer.close_count == 1

    def test_default_viewer_is_null(self) -> None:
        env = _make_env(num_steps=20)
        planner = _make_planner()
        config = AgentConfig(episode_timeout=5, replan_interval=4)
        loop = AgentLoop(env, planner, config)
        assert loop._viewer is not None
        from agent.viewer import NullViewer

        assert isinstance(loop._viewer, NullViewer)
