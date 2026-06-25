## MODIFIED Requirements

### Requirement: Cross-Entropy Method optimizer
The system SHALL provide a Cross-Entropy Method (CEM) optimizer that operates on bounded continuous sequences of fixed horizon `H` and dimension `D`. The optimizer SHALL support two modes: (1) action-sequence mode, the current behavior; (2) embedding mode, where the search variable is a continuous `Tensor[D]` (the target embedding) rather than an action sequence.

The optimizer SHALL expose a single entry point `optimize(cost_fn, *, population_size, elite_frac, n_iterations, horizon, action_dim, action_low, action_high, init_mean=None, init_std=None, rng=None, search_space='action') -> (best_sequence, cost_history)`. When `search_space='embedding'`, `horizon` SHALL be `1` and the returned `best_sequence` SHALL be a single `Tensor[D]` rather than `(H, D)`.

#### Scenario: Iterative refinement lowers cost
- **WHEN** the optimizer is called with a smooth cost function (e.g. quadratic distance from a target) and a budget of at least 5 iterations with a population of at least 64
- **THEN** the best cost in `cost_history` SHALL be monotonically non-increasing across iterations and SHALL be strictly lower at the final iteration than at the first

#### Scenario: Action bounds are respected
- **WHEN** the optimizer is called with `action_low` and `action_high` bounds
- **THEN** every action in the returned `best_action_sequence` SHALL satisfy `action_low <= action <= action_high` elementwise

#### Scenario: Determinism with a seeded RNG
- **WHEN** the optimizer is called twice with identical inputs and the same `rng` seed
- **THEN** the returned `best_action_sequence` and `cost_history` SHALL be identical

#### Scenario: Embedding mode returns a single vector
- **WHEN** the optimizer is called with `search_space='embedding'` and a cost function that takes a `Tensor[D]` and returns a scalar
- **THEN** the returned `best_sequence` SHALL be a `Tensor[D]` (shape `(D,)`) and `cost_history` SHALL be a list of floats

#### Scenario: Embedding mode enforces horizon=1
- **WHEN** the optimizer is called with `search_space='embedding'` and `horizon=5`
- **THEN** the optimizer SHALL raise `ValueError("search_space='embedding' requires horizon=1")`
