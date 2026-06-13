## 1. Planner Abstraction

- [x] 1.1 Create `src/agent/protocol.py` with `PlanResult` frozen dataclass (`actions`, `subgoals`, `success`, `cost`, `replan_count`, `low_confidence`) and `EpisodeResult` dataclass (`steps`, `final_cost`, `duration_seconds`, `trajectory`, `interrupted`)
- [x] 1.2 Add `PlannerProtocol` as a `runtime_checkable` Protocol in `src/agent/protocol.py` with `plan(current_frame, goal_frame) -> PlanResult`
- [x] 1.3 Implement `FlatPlannerAdapter` in `src/agent/protocol.py` wrapping `GoalConditionedPlanner` and `GradientMPC`, normalizing their `Tensor | (Tensor, float)` return to `PlanResult` with `return_cost=True`
- [x] 1.4 Implement `HierarchicalPlannerAdapter` in `src/agent/protocol.py` wrapping `HierarchicalPlanner`, mapping `HierarchicalPlanResult` fields to `PlanResult`
- [x] 1.5 Write tests for `PlanResult` defaults and immutability, adapter normalization for flat and hierarchical planners in `tests/test_protocol.py`

## 2. Agent Loop

- [x] 2.1 Create `src/agent/loop.py` with `AgentLoop.__init__(env, planner, config, buffer=None)` accepting any `PlannerProtocol`-satisfying planner
- [x] 2.2 Implement `AgentLoop.run_episode(goal_frame) -> EpisodeResult` with: goal frame encoding once, planner invocation before first action, step-by-step action execution, fixed-interval replanning
- [x] 2.3 Add warm-start logic: on replan for flat planners, shift previous plan tail and pass as warm-start mean via `set_warm_start_mean`
- [x] 2.4 Handle early replan when action sequence is shorter than `replan_interval`
- [x] 2.5 Add `KeyboardInterrupt` handling: catch during `env.step()`, close env, return partial `EpisodeResult` with `interrupted=True`
- [x] 2.6 Integrate `TrajectoryBuffer`: buffer frames/actions when `config.record_trajectory=True`, include in `EpisodeResult.trajectory`
- [x] 2.7 Write tests for `AgentLoop` in `tests/test_agent_loop.py` using mock env and mock planner: episode timeout, early done, replan interval, warm-start, interrupt, recording on/off

## 3. CLI Entry Point

- [x] 3.1 Create `src/agent/play.py` with `argparse`-based `main()` function: `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`, `--planner` arguments
- [x] 3.2 Implement checkpoint loading: load LeWorldModel, extract encoder and latent rollout for planner construction
- [x] 3.3 Implement planner construction by `--planner` choice: `cem` → `FlatPlannerAdapter(GoalConditionedPlanner)`, `gradient` → `FlatPlannerAdapter(GradientMPC)`, `hierarchical` → `HierarchicalPlannerAdapter(HierarchicalPlanner)`
- [x] 3.4 Implement episode execution: create `MineStudioAgentEnv`, `AgentLoop`, run one episode, print statistics (steps, cost, duration)
- [x] 3.5 Implement trajectory export: when `--record` is set, save `EpisodeResult.trajectory` as `.npz` to `--output-dir`
- [x] 3.6 Register `wally-play = "wally.agent.play:main"` in `pyproject.toml` `[project.scripts]`
- [x] 3.7 Write tests for CLI argument parsing and error handling in `tests/test_play_cli.py`

## 4. Integration and Cleanup

- [x] 4.1 Update `src/agent/__init__.py` to export `PlannerProtocol`, `PlanResult`, `EpisodeResult`, `FlatPlannerAdapter`, `HierarchicalPlannerAdapter`, `AgentLoop`
- [x] 4.2 Run `uv run ruff check .` and fix any lint issues
- [x] 4.3 Run `uv run mypy` and fix any type errors
- [x] 4.4 Run `uv run pytest` and ensure all tests pass
