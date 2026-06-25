## ADDED Requirements

### Requirement: Hierarchical world model stack
The system SHALL provide a stack of world models, one per abstraction level, that predict the next state at that level's time horizon.

#### Scenario: 4-level stack on top of L0
- **WHEN** the system is initialized with `layer_depth=3`
- **THEN** the agent loop SHALL activate L0, L1, L2, L3 and leave L4 as a goal-only layer

#### Scenario: Each layer is an instance of the same JEPA architecture
- **WHEN** a new layer L_n is constructed
- **THEN** the system SHALL instantiate a `JEPAWorldModel` with hyperparameters `(K_n, D_n, depth_n, heads_n)` from the config

#### Scenario: Variable depth at runtime
- **WHEN** a task specifies `layer_depth=0`
- **THEN** the system SHALL skip L1, L2, L3 entirely and behave identically to the current flat LeWM agent

#### Scenario: Higher layers depend on lower layers
- **WHEN** L_n is queried for a planning step
- **THEN** L_n SHALL read the current state embedding from L_(n-1) and SHALL NOT access L0's raw frames directly

### Requirement: Each layer predicts in its own embedding space
The system SHALL train each layer L_n with a temporal-coherence objective that predicts the L_n-embedding of a state K_n frames in the future, conditioned on a target embedding `g_n`.

#### Scenario: Training objective for L1
- **WHEN** a training batch is sampled from the existing shards
- **THEN** the L1 loss SHALL be the L2 distance between the predicted L1-embedding of `s_{t+K1}` and the actual L1-embedding of `s_{t+K1}`

#### Scenario: Encoder sharing
- **WHEN** L1 is initialized
- **THEN** the L1 encoder SHALL start from the frozen L0 encoder weights with a learned linear projection to D1=64

### Requirement: 8x time horizon multiplier
The system SHALL default to a time horizon multiplier of 8 between consecutive layers (K_n = 8 * K_(n-1)).

#### Scenario: Default horizons
- **WHEN** the system is initialized with default hyperparameters
- **THEN** the time horizons SHALL be K0=8, K1=64, K2=512, K3=4096 frames
