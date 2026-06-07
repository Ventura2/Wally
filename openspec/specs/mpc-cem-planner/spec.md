# MPC CEM Planner

## Purpose

Provide the core Cross-Entropy Method (CEM) optimizer, action-space adapter, and configuration schema that underpin the latent MPC planner.

## Requirements

### Requirement: Cross-Entropy Method optimizer

The system SHALL provide a Cross-Entropy Method (CEM) optimizer that operates on bounded continuous action sequences of fixed horizon `H` and action dimension `D`.

The optimizer SHALL expose a single entry point `optimize(cost_fn, *, population_size, elite_frac, n_iterations, horizon, action_dim, action_low, action_high, init_mean=None, init_std=None, rng=None) -> (best_action_sequence, cost_history)` that returns the elite-best action sequence and a list of the best cost observed at each iteration.

#### Scenario: Iterative refinement lowers cost

- **WHEN** the optimizer is called with a smooth cost function (e.g. quadratic distance from a target) and a budget of at least 5 iterations with a population of at least 64
- **THEN** the best cost in `cost_history` SHALL be monotonically non-increasing across iterations and SHALL be strictly lower at the final iteration than at the first

#### Scenario: Action bounds are respected

- **WHEN** the optimizer is called with `action_low` and `action_high` bounds
- **THEN** every action in the returned `best_action_sequence` SHALL satisfy `action_low <= action <= action_high` elementwise, and every sampled action in the search distribution SHALL also satisfy the bounds

#### Scenario: Determinism with a seeded RNG

- **WHEN** the optimizer is called twice with identical inputs and the same `rng` seed
- **THEN** the returned `best_action_sequence` and `cost_history` SHALL be identical

### Requirement: Action-space adapter for MineStudio

The system SHALL provide an action-space adapter that converts a continuous action sequence produced by the CEM optimizer into a sequence of MineStudio-compatible discrete action dicts (camera, buttons, etc.) via a deterministic quantization scheme, and a reverse adapter for the (optional) discrete-to-continuous relaxation used during search.

The adapter SHALL be configurable per MineStudio action vocabulary and SHALL NOT silently drop actions that fall outside the documented discretization grid — out-of-grid actions SHALL raise a `ValueError` listing the offending indices.

#### Scenario: Round-trip quantization is well-defined

- **WHEN** a continuous action sequence is passed to `continuous_to_discrete` and the result is passed to `discrete_to_continuous` using the same vocabulary config
- **THEN** the reconstructed continuous sequence SHALL equal the input up to the configured quantization bin width

#### Scenario: Out-of-grid action raises

- **WHEN** a continuous action outside `action_low`/`action_high` is passed to `continuous_to_discrete`
- **THEN** the adapter SHALL raise `ValueError` with a message identifying the offending timestep and action index

### Requirement: CEM configuration schema

The system SHALL provide a typed configuration schema (Pydantic dataclass or `TypedDict`) named `CEMConfig` that exposes `population_size`, `elite_frac`, `n_iterations`, `horizon`, `action_low`, `action_high`, and `cost_fn` as fields, and SHALL be loadable from a YAML file matching the same field names.

`CEMConfig` SHALL validate that `0 < elite_frac < 1`, `population_size > 1`, `n_iterations >= 1`, and `horizon >= 1`, and SHALL raise `ValueError` on violations.

#### Scenario: YAML loads into CEMConfig

- **WHEN** a valid YAML file with the documented fields is loaded via `CEMConfig.from_yaml(path)`
- **THEN** the resulting object SHALL expose the same field values as the YAML file

#### Scenario: Invalid elite_frac is rejected

- **WHEN** a YAML file sets `elite_frac: 0` or `elite_frac: 1.0` is loaded
- **THEN** `CEMConfig.from_yaml` SHALL raise `ValueError`
