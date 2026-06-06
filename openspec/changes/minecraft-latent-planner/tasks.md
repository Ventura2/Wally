## 1. Package scaffolding

- [ ] 1.1 Create `wally/planner/` subpackage with `__init__.py` and a `py.typed` marker
- [ ] 1.2 Add `wally.planner` to public imports in `wally/__init__.py` if such an `__init__` exists in `lewm-training`
- [ ] 1.3 Add `pydantic` (or fall back to `dataclasses`) and confirm `torch`, `numpy`, `pyyaml` are already declared in `pyproject.toml` by the `lewm-training` change

## 2. CEM configuration schema

- [ ] 2.1 Implement `CEMConfig` (`wally/planner/config.py`) as a Pydantic `BaseModel` with fields `population_size`, `elite_frac`, `n_iterations`, `horizon`, `action_low`, `action_high`, and `gradient_policy`
- [ ] 2.2 Add field validators enforcing `0 < elite_frac < 1`, `population_size > 1`, `n_iterations >= 1`, `horizon >= 1`
- [ ] 2.3 Add `CEMConfig.from_yaml(path: str | Path) -> CEMConfig` classmethod that parses YAML and returns a validated instance
- [ ] 2.4 Add `CEMConfig.default()` factory returning the documented defaults (population 64, elite_frac 0.1, iterations 5, horizon 8, bounds [-1, 1])
- [ ] 2.5 Add unit tests in `tests/test_planner_config.py` covering: valid load, invalid `elite_frac`, missing field, defaults

## 3. CEM optimizer

- [ ] 3.1 Implement `CEMOptimizer` class (`wally/planner/cem.py`) with `optimize(cost_fn, *, config, rng=None) -> (best_action_sequence, cost_history)`
- [ ] 3.2 Implement bounded sampling (truncated normal, resampling out-of-bounds candidates) using a `torch.Generator` for determinism
- [ ] 3.3 Implement elite selection (`elite_size = int(population_size * elite_frac)`) and Gaussian refit on elites
- [ ] 3.4 Return the elite-best action sequence and the per-iteration best cost as a list of floats
- [ ] 3.5 Add a `RandomShooting` baseline in the same file (used by the smoke test) that does one-shot random sampling without refinement
- [ ] 3.6 Add unit tests in `tests/test_cem.py` covering: cost decrease on quadratic, bound enforcement, determinism with seeded RNG, edge case `population_size=2`, `elite_frac=0.5`

## 4. Action-space adapter

- [ ] 4.1 Define `MineStudioActionVocab` dataclass (`wally/planner/actions.py`) describing the discrete action space (camera pitch/yaw, forward/back, jump, sneak, etc.) with a `low`/`high`/`bins` per dimension
- [ ] 4.2 Implement `continuous_to_discrete(actions: Tensor, vocab: MineStudioActionVocab) -> list[dict]` that quantizes a `(H, A)` tensor into MineStudio action dicts
- [ ] 4.3 Implement `discrete_to_continuous(actions: list[dict], vocab: MineStudioActionVocab) -> Tensor` for the inverse mapping
- [ ] 4.4 Add a clear `ValueError` for out-of-grid inputs listing the offending timestep and action index
- [ ] 4.5 Add unit tests in `tests/test_action_adapter.py` covering: round-trip quantization, out-of-grid rejection, vocabulary loading from a sample YAML

## 5. Latent rollout

- [ ] 5.1 Define a `WorldModelProtocol` (`wally/planner/protocols.py`) — a `Protocol` describing the subset of `LeWorldModel` the planner needs (`encode(frame) -> z`, `predict(z, action) -> z_next`)
- [ ] 5.2 Implement `LatentRollout` (`wally/planner/rollout.py`) that takes a world model and a `gradient_policy` (`"detach"` or `"straight_through"`)
- [ ] 5.3 Implement `LatentRollout.rollout(z_0: Tensor, actions: Tensor) -> Tensor` returning shape `(B, H+1, Z)` with the initial latent prepended
- [ ] 5.4 Add `LatentRollout.from_checkpoint(checkpoint_path, *, device) -> LatentRollout` that loads a `LeWorldModel` from the `lewm-training` checkpoint format and freezes all parameters
- [ ] 5.5 Raise `ModelNotLoadedError` when constructed without a checkpoint or pre-loaded model
- [ ] 5.6 Add unit tests in `tests/test_latent_rollout.py` covering: initial latent preserved, batch dim preserved, gradients blocked under `detach`, parameters frozen after `from_checkpoint`, missing-checkpoint error

## 6. Goal-conditioned planner

- [ ] 6.1 Implement `GoalConditionedPlanner` (`wally/planner/plan.py`) with constructor `(world_model: LatentRollout, encoder, config: CEMConfig, *, device=None)`
- [ ] 6.2 Implement `GoalConditionedPlanner.plan(current_frame: Tensor, goal_frame: Tensor, *, return_cost=False) -> Tensor | tuple[Tensor, float]` that wires encoder → CEM cost → rollout → action sequence
- [ ] 6.3 Use the same encoder module for both `current_frame` and `goal_frame` (expose `id(encoder) == id(goal_encoder)` via a debug accessor for the spec scenario)
- [ ] 6.4 Default cost function is `lambda z_H, z_g: ((z_H - z_g) ** 2).sum(dim=-1)`; allow override via constructor
- [ ] 6.5 Auto-select device `"cuda"` if `torch.cuda.is_available()` else `"cpu"`, with explicit `device` argument override
- [ ] 6.6 Add unit tests in `tests/test_goal_conditioned_planner.py` covering: bounded action sequence, encoder reuse, default vs custom cost, device selection, `return_cost=True` shape

## 7. Smoke test on synthetic dynamics

- [ ] 7.1 Create `tests/test_planner_smoke.py` that builds a stand-in linear-Gaussian "world model" (a single `nn.Linear` mapping latents to next latents) and a target latent
- [ ] 7.2 Verify the planner returns a `(H, A)` action sequence within bounds
- [ ] 7.3 Verify the best cost strictly decreases from the first to the last CEM iteration
- [ ] 7.4 Verify a `RandomShooting` baseline (without refinement) does not decrease cost — confirms the smoke test is sensitive to actual CEM refinement
- [ ] 7.5 Add a CI-friendly marker (e.g. `@pytest.mark.smoke`) and ensure the test runs in under 10 seconds

## 8. CLI

- [ ] 8.1 Implement `wally-plan` entry point (`wally/planner/cli.py`) using `argparse` with flags `--checkpoint`, `--config`, `--output`, and mutually exclusive `--env` + `--goal` or `--frames`
- [ ] 8.2 Implement `--frames` mode: load `current.png` and `goal.png` from the directory, run `plan`, write the resulting action sequence to `--output` as a `.pt` tensor
- [ ] 8.3 Implement `--env` mode: instantiate a MineStudio env, reset it, capture the initial frame as `current`, load the goal frame from `--goal`, run `plan`, execute the first `execute_horizon` actions, and log the resulting frame to `--output`
- [ ] 8.4 Add `wally-plan = "wally.planner.cli:main"` to `[project.scripts]` in `pyproject.toml`
- [ ] 8.5 Add a CLI test in `tests/test_planner_cli.py` covering: `--frames` mode happy path, missing `--goal` file fails with non-zero exit code and clear error

## 9. Documentation and validation

- [ ] 9.1 Add a `wally/planner/README.md` describing the public API (`CEMOptimizer`, `LatentRollout`, `GoalConditionedPlanner`, `CEMConfig`) with a minimal end-to-end usage example
- [ ] 9.2 Add a sample `configs/planner/default.yaml` with the documented defaults
- [ ] 9.3 Run `openspec validate minecraft-latent-planner --strict` and resolve any reported issues
- [ ] 9.4 Run `ruff check`, `mypy` (or `pyright`), and `pytest` per the conventions established by the first Python package in `lewm-training`; document the commands in `AGENTS.md` if not already present
