## MODIFIED Requirements

### Requirement: End-to-end plan interface
The system SHALL provide a `GoalConditionedPlanner` class that exposes a single public method `plan(current_frame: torch.Tensor, goal_frame: torch.Tensor | None = None, target_embedding: torch.Tensor | None = None) -> torch.Tensor` returning a continuous action sequence of shape `(H, A)` that respects the configured action bounds. Exactly one of `goal_frame` or `target_embedding` SHALL be provided; if both or neither are provided, the planner SHALL raise `ValueError`.

`plan` SHALL internally: (1) encode `current_frame` into latent `z_0` using the frozen ViT-Tiny encoder, (2a) if `goal_frame` is provided, encode it into latent `z_g` using the same encoder, (2b) if `target_embedding` is provided, use it directly as `z_g`, (3) instantiate a `LatentRollout` over the world model, (4) run the CEM optimizer with a latent-distance cost `|| z_H - z_g ||^2` (or configured alternative), and (5) return the elite-best action sequence.

The planner SHALL batch the CEM population on GPU and SHALL support a `device` argument (default `"cuda"` if available, else `"cpu"`).

#### Scenario: Plan returns bounded action sequence
- **WHEN** `plan(current_frame, goal_frame)` is called with frames of shape `(3, 224, 224)` and `CEMConfig` with `horizon=8`, `action_dim=12`
- **THEN** the returned tensor SHALL have shape `(8, 12)` and every element SHALL satisfy the configured `action_low`/`action_high` bounds

#### Scenario: Plan with target embedding instead of goal frame
- **WHEN** `plan(current_frame, target_embedding=t)` is called with a `target_embedding` of shape `(D,)` and no `goal_frame`
- **THEN** the returned tensor SHALL have shape `(H, A)`, the planner SHALL use `t` directly as `z_g` (skipping the encoder step for the goal), and the cost SHALL be `|| z_H - t ||^2`

#### Scenario: Both goal_frame and target_embedding provided
- **WHEN** `plan(current_frame, goal_frame=g, target_embedding=t)` is called with both arguments
- **THEN** the planner SHALL raise `ValueError("Exactly one of goal_frame or target_embedding must be provided")`

#### Scenario: Neither goal_frame nor target_embedding provided
- **WHEN** `plan(current_frame)` is called with neither argument
- **THEN** the planner SHALL raise `ValueError("Exactly one of goal_frame or target_embedding must be provided")`

#### Scenario: Goal latent is encoded with the same encoder
- **WHEN** `current_frame` and `goal_frame` are passed to `plan`
- **THEN** the planner SHALL use the same frozen ViT-Tiny encoder module for both
