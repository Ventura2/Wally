## MODIFIED Requirements

### Requirement: Latent MPC planner

The system SHALL implement a Model Predictive Control (MPC) planner that, given a current frame and a goal frame, produces a bounded continuous action sequence that (in expectation, under the trained world model) moves the agent from the current latent to the goal latent.

The planner SHALL use the Cross-Entropy Method (CEM) as its optimizer and SHALL operate in the latent space of the trained `LeWorldModel` introduced in the `minecraft-lewm-training` change.

#### Scenario: Planner inputs and outputs

- **WHEN** the planner is called with a current RGB frame and a goal RGB frame (both resized to the encoder's expected input shape)
- **THEN** the planner SHALL return a continuous action sequence whose shape and bounds match the CEM configuration

#### Scenario: Planner depends on trained world model

- **WHEN** the planner is initialized without a trained `LeWorldModel` checkpoint
- **THEN** initialization SHALL fail with a clear error identifying the missing dependency

### Requirement: Action-space handling

The planner SHALL operate over bounded continuous action sequences and SHALL provide an adapter to/from MineStudio's discrete action vocabulary. The adapter SHALL be deterministic and SHALL NOT silently drop actions outside the quantization grid.

#### Scenario: Discrete action compatibility

- **WHEN** a planned continuous action sequence is converted to discrete MineStudio actions
- **THEN** every produced action SHALL be a valid entry in the configured MineStudio action vocabulary

### Requirement: CEM configuration

The planner SHALL expose a YAML-loadable configuration covering: population size, elite fraction, iteration count, plan horizon, and continuous action bounds (low/high vectors). Reasonable defaults SHALL be provided (population size 64, elite fraction 0.1, iterations 5, horizon 8) and SHALL be overridable per environment.

#### Scenario: Default config is valid

- **WHEN** the default `CEMConfig` is loaded
- **THEN** all validation constraints (`0 < elite_frac < 1`, `population_size > 1`, etc.) SHALL pass and the planner SHALL be runnable without further configuration
