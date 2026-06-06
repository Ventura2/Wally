## ADDED Requirements

### Requirement: End-to-end plan interface

The system SHALL provide a `GoalConditionedPlanner` class that exposes a single public method `plan(current_frame: torch.Tensor, goal_frame: torch.Tensor) -> torch.Tensor` returning a continuous action sequence of shape `(H, A)` that respects the configured action bounds.

`plan` SHALL internally: (1) encode `current_frame` into latent `z_0` using the frozen ViT-Tiny encoder, (2) encode `goal_frame` into latent `z_g` using the same encoder, (3) instantiate a `LatentRollout` over the world model, (4) run the CEM optimizer with a latent-distance cost `|| z_H - z_g ||^2` (or configured alternative), and (5) return the elite-best action sequence.

The planner SHALL batch the CEM population on GPU and SHALL support a `device` argument (default `"cuda"` if available, else `"cpu"`).

#### Scenario: Plan returns bounded action sequence

- **WHEN** `plan(current_frame, goal_frame)` is called with frames of shape `(3, 224, 224)` and `CEMConfig` with `horizon=8`, `action_dim=12`
- **THEN** the returned tensor SHALL have shape `(8, 12)` and every element SHALL satisfy the configured `action_low`/`action_high` bounds

#### Scenario: Goal latent is encoded with the same encoder

- **WHEN** `current_frame` and `goal_frame` are passed to `plan`
- **THEN** the planner SHALL use the same frozen ViT-Tiny encoder module for both (verifiable by `id(encoder) == id(goal_encoder)` in a debug accessor)

### Requirement: Cost function

The planner SHALL use a configurable latent cost function whose default is the squared L2 distance `|| z_H - z_g ||^2` between the rollout's final predicted latent and the encoded goal latent.

The cost function SHALL accept a goal-progress shaping term (a callable) that, if provided, is added to the per-candidate cost. The final cost used by CEM SHALL be returned alongside the action sequence via an optional `return_cost=True` parameter.

#### Scenario: Default cost is squared L2

- **WHEN** `plan` is called with default config
- **THEN** the cost function passed to the CEM optimizer SHALL be `lambda z_H, z_g: ((z_H - z_g) ** 2).sum(dim=-1)`

#### Scenario: Optional return cost

- **WHEN** `plan(..., return_cost=True)` is called
- **THEN** the planner SHALL return a tuple `(action_sequence, final_cost)`

### Requirement: `wally-plan` CLI

The system SHALL provide a CLI entry point `wally-plan` (registered via `pyproject.toml` `[project.scripts]`) that accepts `--checkpoint PATH`, `--config PATH`, and one of `--env` (a MineStudio environment name) or `--frames PATH` (a directory containing `current.png` and `goal.png`).

In `--env` mode the CLI SHALL reset the environment, capture the initial frame as `current`, load the goal frame from a `--goal PATH` argument, run `plan`, and execute the first `execute_horizon` actions in the environment, logging the resulting frame to `--output PATH`.

In `--frames` mode the CLI SHALL run `plan` on the provided pair and write the returned action sequence to `--output PATH` as a `.pt` tensor.

#### Scenario: Offline plan from frames

- **WHEN** `wally-plan --checkpoint ckpt.pt --config planner.yaml --frames ./pair/ --output plan.pt` is invoked with a valid checkpoint and `current.png`/`goal.png` in `./pair/`
- **THEN** a `plan.pt` file SHALL be written containing a tensor of shape `(H, A)` and the CLI SHALL exit with code 0

#### Scenario: Missing goal frame fails fast

- **WHEN** `wally-plan --env minecraft --goal ./missing.png ...` is invoked and the goal file does not exist
- **THEN** the CLI SHALL exit with a non-zero code and print a clear error message identifying the missing path

### Requirement: Planning smoke test

The system SHALL include a unit test `tests/test_planner_smoke.py` that verifies on a synthetic linear-Gaussian dynamics toy problem that the planner (1) returns valid action shapes, (2) respects action bounds, and (3) reduces cost across CEM iterations.

The test SHALL NOT require a trained Minecraft checkpoint — it SHALL construct a tiny stand-in world model (a single linear layer) and verify planner behavior on that stand-in.

#### Scenario: Cost decreases on toy dynamics

- **WHEN** the smoke test is run with a linear-Gaussian dynamics model and a fixed target latent
- **THEN** the test SHALL assert that the final iteration's best cost is strictly lower than the first iteration's best cost
