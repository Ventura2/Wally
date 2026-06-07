## 1. Project Setup

- [ ] 1.1 Create `src/agent/` package with `__init__.py` and `py.typed`
- [ ] 1.2 Add `AgentConfig` Pydantic model in `src/agent/config.py` with fields: `replan_interval`, `episode_timeout`, `resize`, `action_vocab_path`, `record_trajectory`, and `from_yaml()` classmethod
- [ ] 1.3 Register `wally-play` entry point in `pyproject.toml` pointing to `agent.cli:main`

## 2. Environment Wrapper

- [ ] 2.1 Implement `MineStudioAgentEnv` in `src/agent/env.py` wrapping `MineStudioEnv` from collector
- [ ] 2.2 Add frame preprocessing: resize to configured dimensions, convert to `torch.Tensor` of shape `(3, H, W)`, normalize to `[0, 1]`
- [ ] 2.3 Add action postprocessing: accept continuous `(25,)` tensor, clip to vocab bounds, convert to discrete dict via `continuous_to_discrete()`
- [ ] 2.4 Implement `reset()`, `step()`, `close()` with proper lifecycle tracking (raise `RuntimeError` on step after close)
- [ ] 2.5 Write tests for `MineStudioAgentEnv` with mocked MineStudio simulator

## 3. Agent Loop

- [ ] 3.1 Implement `AgentLoop` class in `src/agent/loop.py` with constructor accepting `planner`, `env`, `config`, and optional `action_vocab`
- [ ] 3.2 Implement `run_episode(goal_frame)` method: plan, execute `replan_interval` steps, observe, replan, repeat
- [ ] 3.3 Add warm-start support: pass remaining actions from previous plan as initial mean to CEM optimizer on replan
- [ ] 3.4 Add episode timeout: terminate when step count reaches `episode_timeout`
- [ ] 3.5 Add action clipping: clamp planner output to vocab `[low, high]` before discretization
- [ ] 3.6 Add graceful shutdown: catch `KeyboardInterrupt`, call `env.close()`, return partial trajectory
- [ ] 3.7 Write tests for `AgentLoop` with mocked planner and mocked env

## 4. Trajectory Recording

- [ ] 4.1 Implement `TrajectoryBuffer` in `src/agent/buffer.py` that accumulates frames and actions during an episode
- [ ] 4.2 Integrate buffer into `AgentLoop.run_episode()`: record when `record_trajectory` is True, skip when False
- [ ] 4.3 Add `to_dict()` method returning `{"frames": np.ndarray(T,H,W,3), "actions": np.ndarray(T,25)}`
- [ ] 4.4 Write tests for `TrajectoryBuffer`

## 5. CLI Entry Point

- [ ] 5.1 Implement `wally-play` CLI in `src/agent/cli.py` with argparse: `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`
- [ ] 5.2 Add checkpoint loading: load LeWorldModel, extract encoder and latent rollout for planner construction
- [ ] 5.3 Add goal frame loading: read image from path, preprocess to tensor
- [ ] 5.4 Wire up `GoalConditionedPlanner`, `MineStudioAgentEnv`, and `AgentLoop` and run one episode
- [ ] 5.5 Print episode statistics: steps taken, final latent cost, wall-clock duration
- [ ] 5.6 Add optional trajectory export: save recorded trajectory as `.tar` shard when `--record` is set
- [ ] 5.7 Write tests for CLI argument parsing and main function with mocked components

## 6. Integration and Validation

- [ ] 6.1 Add integration test: end-to-end `AgentLoop` with mocked MineStudio env, verifying plan-execute-replan cycle
- [ ] 6.2 Run full test suite (`uv run pytest`) and fix any failures
- [ ] 6.3 Run linter (`uv run ruff check .`) and typechecker (`uv run mypy`) and fix any issues
