## ADDED Requirements

### Requirement: CEM optimizer samples candidates on the configured device

`CEMOptimizer.optimize` SHALL accept an optional `device: torch.device | str | None = None` keyword argument. When `device` is non-`None`, every candidate tensor it creates (the initial `mean` and `std` if not provided, the `randn`-sampled population, the returned `best_action_sequence`, and the `cost_history` numerics derived from candidates) SHALL live on that device. When `device` is `None`, the optimizer SHALL preserve its current CPU-only behaviour so existing call sites are unaffected.

#### Scenario: device=None preserves CPU behaviour

- **WHEN** `CEMOptimizer.optimize(...)` is called without a `device` argument
- **THEN** all candidate tensors SHALL be created on CPU (the same device as `torch.randn(shape)` without an explicit device), the returned `best_action_sequence` SHALL be a CPU tensor, and the historical test suite SHALL pass unchanged

#### Scenario: device=cuda produces CUDA candidates

- **WHEN** `CEMOptimizer.optimize(..., device="cuda")` is called and CUDA is available
- **THEN** all candidate tensors SHALL be created on `cuda:0`, the returned `best_action_sequence` SHALL be a CUDA tensor, and the call SHALL NOT raise a CPU/CUDA device-mismatch error when the cost function moves a CUDA model tensor through the candidate actions

#### Scenario: device="cpu" explicitly

- **WHEN** `CEMOptimizer.optimize(..., device="cpu")` is called
- **THEN** all candidate tensors SHALL live on CPU, regardless of the cost function's model device

### Requirement: GoalConditionedPlanner aligns cost function and CEM device

`GoalConditionedPlanner` (in `src/wally/planner/plan.py`) SHALL pass its configured `self._device` to every `CEMOptimizer.optimize(...)` invocation it makes (in both `plan` and `plan_to_latent`). The planner SHALL NOT require the caller to manually move the cost-function inputs to the same device as the world model — the planner is responsible for end-to-end device alignment.

#### Scenario: planner works on cuda end-to-end

- **WHEN** a `GoalConditionedPlanner` is constructed with `device=torch.device("cuda")` and a CUDA world model and `planner.plan_to_latent(current_frame, goal_latent, return_cost=True)` is called with `current_frame` and `goal_latent` already on `cuda`
- **THEN** the call SHALL NOT raise a `RuntimeError` about CPU/CUDA tensor type mismatch and SHALL return a CUDA action tensor plus a finite cost value

#### Scenario: planner still works on cpu

- **WHEN** a `GoalConditionedPlanner` is constructed without an explicit `device` (so `self._device` resolves to CPU) and is called on a CPU world model
- **THEN** the existing CPU-only flow SHALL be unchanged: `plan` and `plan_to_latent` SHALL continue to return CPU action tensors and the historical test suite SHALL pass

### Requirement: Hierarchical planner and Gradient MPC align CEM device

`HierarchicalPlanner` (`src/wally/planner/hierarchical_planner.py`), `GradientMPC` (`src/wally/planner/gradient_mpc.py`), and `HighLevelPlanner` (`src/wally/planner/high_level_planner.py`) SHALL each forward their configured `self._device` into every `CEMOptimizer.optimize(...)` invocation they make, so that planners that internally use CEM behave identically to `GoalConditionedPlanner` with respect to device alignment.

#### Scenario: GradientMPC on cuda does not crash

- **WHEN** `GradientMPC` is constructed with `device=torch.device("cuda")` (or the equivalent) and is asked to plan from a CUDA latent tensor
- **THEN** the call SHALL NOT raise a CPU/CUDA tensor-type mismatch error and SHALL return a CUDA action tensor

#### Scenario: HierarchicalPlanner on cpu unchanged

- **WHEN** `HierarchicalPlanner` is run on a CPU world model with no explicit device override
- **THEN** the existing CPU-only behaviour SHALL be preserved (regression check for the existing `tests/test_hierarchical_planner.py` and `tests/test_high_level_planner.py`).
