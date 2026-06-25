from __future__ import annotations

import time
from typing import Any, Callable

import torch
from torch import Tensor

from wally.agent.buffer import TrajectoryBuffer
from wally.agent.config import AgentConfig
from wally.agent.protocol import EpisodeResult
from wally.agent.relay import RelayBuffer
from wally.agent.viewer import FrameViewerLike, NullViewer


class AgentLoop:
    def __init__(
        self,
        env: Any,
        planner: Any,
        config: AgentConfig,
        buffer: TrajectoryBuffer | None = None,
        viewer: FrameViewerLike | None = None,
        relay: RelayBuffer | None = None,
        l0_encoder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self._env = env
        self._planner = planner
        self._config = config
        self._buffer = buffer
        self._viewer: FrameViewerLike = viewer if viewer is not None else NullViewer()
        self._relay: RelayBuffer | None = relay
        self._l0_encoder = l0_encoder

    def _l0_state_embedding(self, frame: Tensor) -> Tensor:
        if self._l0_encoder is None:
            return torch.zeros(1)
        x = frame
        if x.dim() == 3:
            x = x.unsqueeze(0)
        emb = self._l0_encoder(x)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return emb.flatten()

    def run_episode(
        self,
        goal_frame: Tensor,
        target_embedding: Tensor | None = None,
    ) -> EpisodeResult:
        start_time = time.monotonic()
        current_frame = self._env.reset()

        plan_actions: Tensor | None = None
        action_index = 0
        accumulated_cost = 0.0
        step_count = 0
        interrupted = False

        if self._config.record_trajectory and self._buffer is None:
            self._buffer = TrajectoryBuffer()

        is_hier = hasattr(self._planner, "set_target_embedding") and hasattr(
            self._planner, "tick_with_frame"
        )
        if is_hier and target_embedding is not None:
            self._planner.set_target_embedding(target_embedding)

        for step in range(self._config.episode_timeout):
            needs_replan = (
                step % self._config.replan_interval == 0
                or plan_actions is None
                or action_index >= len(plan_actions)
            )

            if needs_replan:
                if (
                    hasattr(self._planner, "set_warm_start_mean")
                    and plan_actions is not None
                ):
                    horizon = plan_actions.shape[0]
                    shifted = plan_actions[self._config.replan_interval :]
                    pad_len = horizon - shifted.shape[0]
                    if pad_len > 0:
                        if shifted.shape[0] > 0:
                            padding = shifted[-1:].expand(pad_len, -1).clone()
                        else:
                            padding = torch.zeros(
                                pad_len, plan_actions.shape[1]
                            )
                        warm_start = torch.cat([shifted, padding], dim=0)
                    else:
                        warm_start = shifted
                    self._planner.set_warm_start_mean(warm_start)

                if is_hier:
                    plan_result = self._planner.plan(current_frame, goal_frame)
                else:
                    plan_result = self._planner.plan(current_frame, goal_frame)
                plan_actions = plan_result.actions
                action_index = 0
                accumulated_cost += plan_result.cost

            action = plan_actions[action_index]
            if action.dim() == 1 and action.shape[-1] > 12:
                action = action.clone()
                action[12] = 0.0

            try:
                next_frame, reward, done, info = self._env.step(action)
            except KeyboardInterrupt:
                self._viewer.close()
                elapsed = time.monotonic() - start_time
                trajectory = None
                if self._buffer is not None and len(self._buffer) > 0:
                    trajectory = self._buffer.to_dict()
                self._env.close()
                return EpisodeResult(
                    steps=step_count,
                    final_cost=accumulated_cost,
                    duration_seconds=elapsed,
                    trajectory=trajectory,
                    interrupted=True,
                )

            step_count += 1
            if self._buffer is not None:
                self._buffer.add(current_frame, action, info=info)
            current_frame = next_frame
            action_index += 1

            if is_hier:
                self._planner.tick_with_frame(current_frame)

            viewer_info = dict(info) if info else {}
            viewer_info["step"] = step_count
            viewer_info["done"] = bool(done)
            pov = info.get("pov") if info else None
            if self._relay is not None:
                self._relay.update(pov)
            self._viewer.show(pov, info=viewer_info)
            if self._viewer.should_quit():
                interrupted = True
                break

            if done:
                break

        elapsed = time.monotonic() - start_time
        trajectory = None
        if self._buffer is not None and len(self._buffer) > 0:
            trajectory = self._buffer.to_dict()

        self._viewer.close()
        return EpisodeResult(
            steps=step_count,
            final_cost=accumulated_cost,
            duration_seconds=elapsed,
            trajectory=trajectory,
            interrupted=interrupted,
        )
