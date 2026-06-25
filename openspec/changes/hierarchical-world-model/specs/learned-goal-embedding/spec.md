## ADDED Requirements

### Requirement: Goal as a learned embedding
The system SHALL represent each task's goal as a single learned vector `g_n` per layer, not as a string or a goal frame.

#### Scenario: Task "get 16 oak logs" produces a g3 vector
- **WHEN** the user provides the task "get 16 oak logs"
- **THEN** the system SHALL look up or learn the corresponding `g3: Tensor[D3]` and pass it to the L3 layer

#### Scenario: No goal frame in the runtime path for L1+
- **WHEN** the L1 planner runs
- **THEN** L1 SHALL optimize for embedding distance to `g1`, not pixel distance to a goal frame

#### Scenario: L0 keeps its goal frame for backward compatibility
- **WHEN** the L0 CEM planner runs without a target embedding
- **THEN** L0 SHALL optimize for pixel distance to a goal frame as before

### Requirement: Goal-embedding optimization
The system SHALL learn the `g_n` vectors by optimizing a task-specific reward (e.g. "how many logs collected in N steps").

#### Scenario: g3 is learned end-to-end on the task reward
- **WHEN** training the hierarchy
- **THEN** `g3` SHALL be updated by gradient descent on the same loss as L1/L2/L3

#### Scenario: Multiple tasks produce different g3 vectors
- **WHEN** the user switches from "get wood" to "find food"
- **THEN** the system SHALL load a different learned `g3` for the new task
